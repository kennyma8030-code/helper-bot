# Wave-Style Retrieval Orchestration — Design Spec

Status: **implemented** 2026-08-23 (`loop.py`, `retrieval.py`, `prompts.py`).
v2 reconciled the draft against specs.md and the live schema on 2026-08-18.
Scope: multi-branch RAG retrieval loop over group-chat corpus, restructured from
recursive branch spawning into synchronized waves.

## Where each piece lives

| Spec | Code |
|---|---|
| Orchestrator (§2.1, §3) | `loop.investigate` |
| Planner (§2.2) | `loop._plan`, `prompts.PLANNER_PROMPT` |
| Sub-question dedup (§2.1) | `loop._dedup` |
| Worker (§2.3) | `loop._run_worker`, `prompts.WORKER_PROMPT` |
| Grading (§2.3 step 3) | inside `_run_worker`, `prompts.GRADER_PROMPT` |
| Barrier + merge (§3) | `loop._dispatch`, `loop._merge` |
| Sufficiency (§2.4) | `loop._sufficient`, `prompts.SUFFICIENCY_PROMPT` |
| Synthesis (§2.5) | `loop.synthesize`, `prompts.SYNTH_PROMPT` |
| Ledger (§4) | `loop.Ledger` |
| Budgets and knobs (§6, §12) | `loop.Budget`, `loop.WaveConfig` |
| Instruments | `retrieval.TOOLS`, `retrieval.execute_call` |

Deliberate deviations from the draft, each argued at its own section below:
`WORKER_TIMEOUT` is 90s rather than 30s (§6), there is no mechanical
contradiction detector (§4.1), config is a dataclass rather than a YAML file
(§12), and wave snapshots stay off unless a directory is named (§9).

---

## 1. Overview

**Purpose.** This system exists for multi-step reasoning over the chat
history — reference chains, disposition and pattern questions,
business-logic-style inference that spans many messages. Plain lookup is
explicitly out of scope: Discord's own search already does "find the message
where someone said X." There is no triage/lookup bypass (this supersedes
specs.md decision 6); every question enters the orchestrator. Light questions
are served by the wave-1 fallback (§10) and the cheap 1-wave configuration
(§6), not by a separate path.

**Units.** The corpus has two units, deliberately distinct — this spec never
says "chunk", because nothing here slices documents:

- **message** — the evidence unit. The only thing a fact may cite.
- **cluster** — the retrieval unit: an LLM-written summary of a contiguous
  message span, carrying its boundary ids (`first_message_id ..
  last_message_id`). A similarity hit on a cluster is a pointer to a span
  worth reading — never citable evidence, because its text is model-written
  (db.py `similarity_search` note; specs-summaries.md decision 2).

A single **orchestrator** runs the loop. Each iteration is a **wave**:

```
plan → dispatch workers (parallel) → barrier → merge → sufficiency check → replan or stop
```

Workers execute retrieval for one sub-question each. Workers do not spawn
workers. All spawn/prune/stop decisions happen at wave boundaries with full
global state visible.

Design goals, in priority order:

1. Correctness of retrieved context (measured on synthetic eval corpus)
2. Wall-clock latency
3. Token cost
4. Reproducibility (deterministic mode for eval)

---

## 2. Components

### 2.1 Orchestrator

Owns the loop, the budget, and the ledger. The only component that creates or
cancels workers. Stateless between runs; all run state lives in the ledger.

Responsibilities:

- Call the planner at wave start
- Deduplicate planned sub-questions against the ledger (embedding similarity,
  threshold `DEDUP_SIM`, default 0.90 — a placeholder; see the §12 caveat)
- Enforce budgets before dispatch
- Run the wave (`asyncio.gather` over workers, bounded by semaphore)
- Merge worker results into the ledger
- Run the sufficiency check
- Decide: another wave, or terminate and hand off to synthesis

### 2.2 Planner

One LLM call per wave. Input: original question, state document, ledger digest
(resolved sub-questions + open gaps + discovered facts). Output: structured
list of sub-questions for the next wave.

- Wave 1: decomposition of the original question
- Wave N>1: gap-driven — only sub-questions targeting what is still
  unresolved, informed by everything retrieved so far
- Model: large tier (planning quality gates everything downstream)
- Output schema: `[{sub_question, rationale, priority, expected_answer_type}]`

### 2.3 Workers

One worker per sub-question per wave. A worker is a bounded internal loop —
**no spawning rights**.

Workers own the full instrument surface, invoked by **native function calling
with manual dispatch** (as loop.py does today — no SDK auto-calling, no
parsing of model output; specs.md decision 1): `structured_search`,
`keyword_search`, `replies_to`, `messages_near`, `activity_stats`,
`similarity_search`. Instrument discipline is unchanged from v1: structured
filters and keyword first, similarity last (the most expensive instrument);
counting sub-questions go to `activity_stats` and never read message bodies.

Per iteration (max `WORKER_MAX_ITERS`, default 3):

1. Choose instruments and formulate 1–3 calls (speculative; issued
   concurrently)
2. Drill down: a similarity hit is a cluster — turn the pointer into
   messages by reading its span, `structured_search` with
   `min_id`/`max_id` set to the cluster's `first_message_id`/
   `last_message_id` (built 2026-08-23; both ends inclusive)
3. Batch-grade all candidate **messages** for relevance in **one** LLM call
   (structured output, small model). Grader scores are the run's only
   relevance scores (§2.4, §5)
4. Self-assess: resolved / needs refinement / unresolvable
5. If refinement: rewrite the calls using graded results, next iteration

Worker output: `{sub_question, status, graded_messages[], extracted_facts[],
iterations_used, model_escalated}` — facts cite message ids only.

Workers write nothing shared mid-flight; everything merges at the barrier.
(Draft v1's query reservations were cut — §11.)

### 2.4 Sufficiency checker

Runs at each barrier. Two-stage:

- **Cheap gate first**: all sub-questions status=resolved AND aggregate
  grader relevance (from §2.3 step 3) above threshold → done, skip the LLM
- **LLM check only when ambiguous**: small model, structured yes/no + gap list
- Gap list feeds the next planner call directly

### 2.5 Synthesizer

Single large-model call after termination. Input: original question, state
document, and the merged **ledger** — facts with message-id citations,
inferences, open questions, dead branches — plus, when the ledger alone is
thin, cited messages re-fetched by id and trimmed to the context budget.
Never the raw retrieved set: unjudged rows acting as de facto evidence is
exactly what the one-pass eviction rule exists to prevent (specs.md
decision 4). This is the only unavoidable large-model call besides the
planner.

---

## 3. Wave lifecycle

```
WAVE N
├── planner call ─────────────── large model, 1 call
├── dedup sub-questions ──────── embedding sim vs. ledger, no LLM
├── budget check ─────────────── counters only
├── dispatch workers ─────────── parallel, semaphore-capped
│     worker A: iter 1..k       small model, escalate on evidence
│     worker B: iter 1..k
│     worker C: iter 1..k
├── BARRIER ──────────────────── all workers joined or timed out
├── merge ────────────────────── facts, graded messages, statuses → ledger
├── sufficiency check ────────── cheap gate, LLM only if ambiguous
└── decision ─────────────────── replan (wave N+1) | terminate → synthesize
```

Default limits: `MAX_WAVES = 3`. Empirically, if two informed waves haven't
resolved it, a third rarely does — wave 3 exists for the tail.

Worker timeout: workers that exceed `WORKER_TIMEOUT` (default 90s) are
cancelled at the barrier; their sub-question is marked unresolved and returns
to the planner as a gap. A slow worker must never stall the wave indefinitely.

The 90s is a correction to the draft's 30s. A worker runs up to
`WORKER_MAX_ITERS` rounds, and each round is two sequential model calls (choose
instruments, then grade) plus the searches between them. At 30s a worker that
actually used its rounds would be cancelled every time, which makes the timeout
the normal exit rather than the safety net.

---

## 4. Ledger

Split by consistency need — not one object, one lock.

### 4.1 Fact store

**Append-only, as in v1** (specs.md decision 3): entries are never rewritten
and citations are never dropped — rewritten state rots (citations drop,
hedges erode, inferences launder into facts). Typed entries, validated in
code each merge:

- `fact` — claim + message-id citations (required; a fact without citations
  is rejected)
- `inference` — claim + supporting fact ids + competing explanations
- `open_question`, `dead_branch`

A new fact that contradicts an existing one is **appended alongside it**,
never merged or replaced — the disagreement is left standing, because a
conflict is signal. Temporal precedence (later statements supersede, per
speaker) is applied at synthesis time, not by mutating the store. Draft v1's
`(entity, predicate)` replace-on-write fact store is dropped.

There is deliberately **no mechanical contradiction detector**. Deciding
whether two English claims conflict is a reading task, not a comparison of
keys, and the only version of that check worth having is a model doing it with
both claims and their citations in front of it — which is what synthesis
already is. Code that flagged conflicts by matching entities would fire on
paraphrases and miss real ones.

Locked writes; lock-free reads.

### 4.2 Retrieved-evidence set

`set[message_id]` of everything retrieved, plus the graded messages with
their scores. Merged at barriers. Used for cross-wave dedup and final
re-ranking. Cluster ids already drilled into are tracked alongside, so a
later wave doesn't re-read a span — but clusters themselves are pointers,
not evidence (§1).

This set is **bookkeeping, not context**: message bodies are seen by the
worker that retrieved them and by its grader, then never re-injected into a
prompt. Later waves and the synthesizer see the ledger's facts, and re-fetch
a cited message by id when they need the primary text again (specs.md
decision 4 — context cost is rows × passes, and unjudged rows sitting in
context act as de facto evidence).

### 4.3 Control state

Budget counters, wave number, cancellation flags. Written only by the
orchestrator; read by workers at iteration boundaries.

### 4.4 What enters worker prompts

Never the full ledger. Each worker gets a **filtered view**: facts whose
embedding similarity to its sub-question exceeds `CTX_SIM` (default 0.75,
placeholder — §12 caveat), capped at `WORKER_CTX_TOKENS` (default 1,500).
The state document (alias registry, durable facts) is always included — it
is small by contract (§7).

---

## 5. Model tiering & escalation

| Call | Default tier |
|---|---|
| Planner | large |
| Query formulation / rewrite | small |
| Message grading (batched) | small |
| Sufficiency (when LLM needed) | small |
| Synthesis | large |

Tier bindings: **large = `gemini-3.5-flash`**, **small =
`gemini-3.1-flash-lite`** — both already priced in `llm.MODEL_PRICES`.

Prompt caching: each role's static prompt is cached separately. The v1
single planner cache (prompt + tools together, loop.py `_planner_cache`)
does not survive the split — the wave planner has no tools; workers carry
them now.

Workers **start small and escalate on evidence**, never on prediction:

- aggregate grader relevance below `ESCALATE_SCORE` after iteration 1
  (grader scores only — cosine distance ranks cluster hits and gates
  nothing; most instruments return no score at all), or
- self-assessed ambiguity in grading output, or
- contradiction detected against the fact store

Escalation is per-worker, one-way, logged in worker output.

---

## 6. Budgets

Enforced by the orchestrator at dispatch time. All counters in control state.

| Budget | Default | On breach |
|---|---|---|
| `MAX_WAVES` | 3 | terminate, synthesize from what exists |
| `MAX_CONCURRENT_WORKERS` | 6 (semaphore) | queue within wave |
| `MAX_TOTAL_LLM_CALLS` | 40 | terminate |
| `MAX_TOTAL_TOKENS` | run-configurable | terminate |
| `WORKER_MAX_ITERS` | 3 | worker returns unresolved |
| `WORKER_TIMEOUT` | 90s | cancel at barrier |

Two budget systems, deliberately: the counters above govern **a single
run**; the **dollar budget** in `api_usage` (fed by `llm.track_usage`, the
cap that stops the 4-hour schedule) is the global kill switch across runs.
Counters terminate a run; the dollar cap terminates the service. Every call
this system makes — planner, workers, graders, sufficiency, synthesis, and
embeddings for dedup/filtering — must be recorded via `track_usage` so it
counts toward that cap.

One knob philosophy: the same architecture runs cheap (1 wave, 3 workers) or
thorough (3 waves, 6 workers) by configuration, not code change.

Rate limiting: the semaphore bounds in-flight requests below the account RPM
ceiling. Batched grading keeps call count low even at full width.

---

## 7. Interaction with the state document

**Prerequisite — does not exist yet.** The always-injected state document
(alias registry, durable facts, decisions, norms) is assumed by this spec,
but nothing in the repo builds or maintains it. The closest existing
material is `day_summaries` prose + facets, which are retrieval targets,
not injected state. It must be specced and built before or alongside this
system. When it exists, it touches this system twice:

1. **Query rewriting**: the planner and workers resolve aliases and implicit
   references against it before embedding queries
2. **Fact extraction feedback**: durable facts discovered during retrieval are
   proposed to the state-document maintainer (separate process) — this run
   does not write to it

Hard cap on state document size is owned by its maintainer; this system
assumes ≤ 2,000 tokens.

---

## 8. Concurrency model

- Single-process `asyncio`; no true parallelism, so locks guard interleaving
  at await points only
- `asyncio.Lock` per ledger section (§4), critical sections contain set/dict
  operations only — no LLM calls, no embedding, no I/O inside a lock
- Workers dispatched via `asyncio.gather(*workers, return_exceptions=True)`;
  a worker exception marks its sub-question unresolved rather than failing
  the wave
- Cancellation: orchestrator cancels worker tasks directly (`task.cancel()`);
  workers additionally check control-state flags between iterations

---

## 9. Logging, determinism & eval mode

**Logging is always on, not an eval-mode feature** (specs.md decision 7).
Every wave logs: the planner's sub-questions, each worker's instrument calls
and their row counts, grader verdicts, ledger diffs, the sufficiency verdict,
escalations, and budget spent. This is the debugging tool, the eval
substrate, and the evidence base for any future architecture change —
including the deferred items in §11, which are gated on what these logs
show. The JSONL barrier snapshots below are the eval-mode addition on top.

**Prerequisite — does not exist yet.** The eval harness and synthetic corpus
are net-new work (specs.md tags synthetic eval `[later]`; test.py is the
budget/schedule test). Deterministic mode, threshold tuning
(`ESCALATE_SCORE`, `DEDUP_SIM`, `CTX_SIM`), and snapshot replay all depend
on it.

`deterministic=True` flag:

- Workers execute sequentially in planner-priority order
- Speculative calls issued sequentially, first-success-wins by fixed order
- Temperature 0 on all calls; seeds pinned where the API supports it
- Wave state snapshotted to JSONL at every barrier
  (`runs/<run_id>/wave_<n>.jsonl`)

Eval harness runs against the synthetic corpus (fact-spec-first generation,
per prior design): questions with known ground truth, scored on answer
accuracy, message recall vs. provenance labels, calls used, tokens used,
wall-clock. Every run tagged with corpus version + config hash.

Barrier snapshots make any wave replayable: feed `wave_<n>.jsonl` back in and
re-run from that point with modified config.

---

## 10. Failure handling

| Failure | Behavior |
|---|---|
| Worker exception | sub-question → unresolved gap; wave continues |
| Worker timeout | cancelled at barrier; gap |
| Planner returns no sub-questions on wave 1 | fall back to direct retrieval on the original question |
| All sub-questions unresolvable | synthesize with explicit "insufficient context" framing; never fabricate |
| Budget exhausted mid-wave | current wave completes its barrier; no next wave |

---

## 11. Extension points (explicitly deferred)

- **Query reservations** (cut from draft v1): a shared board where workers
  post in-flight query embeddings so near-duplicate searches await each
  other's results instead of re-running. Cut because it is exactly the
  mid-flight shared memory specs.md's branch design forbids ("do not fix
  this with shared memory"), and plan-level dedup at wave start catches the
  common case. Revisit only if wave logs show heavy duplicate retrieval
  *within* waves.
- ~~`read_span`~~ — **built 2026-08-23**, as a filter rather than a new
  instrument: `min_id`/`max_id` bounds on `discord_message_id` in
  `db._build_where`, which gives the id range to `structured_search`,
  `keyword_search`, and `activity_stats` at once. This is the §2.3 drill-down.
- **Planned instruments** — the schema supports these today; the worker
  tool surface should grow into them:
  - Thread walker — follow `reply_to_message_id` *upward* to rebuild a
    reply chain (only the downward `replies_to` exists). Structural, immune
    to the interleaved-topic noise time windows pick up.
  - Day-summary reader — `day_summaries` prose + facets as cheap
    orientation before drilling. Never citable, like clusters.
  - Facet search — the GIN-indexed `facets` JSONB: "days whose facets
    mention X", no embeddings.
  - Real text search — tsvector full-text and/or pg_trgm fuzzy match,
    replacing the ILIKE placeholder (db.py's own TODO).
  - `ask_user` — per specs.md's deferred design: the asker as the cheapest
    instrument for facts only they hold; hard rate limit, asked early.

  Excluded on purpose: social-trivia instruments (interaction stats,
  first/last mention, author sampling) — outside the reasoning purpose (§1).
- **Worker spawning rights**: only if eval shows workers consistently hitting
  `WORKER_MAX_ITERS` with unresolved status AND a third wave not helping.
  The wave structure makes this a scoped change (workers request spawns,
  orchestrator grants at barrier) rather than a rewrite.
- **Cross-run sub-question cache**: normalize + embed sub-questions, cache
  resolved message sets keyed by corpus version.
- **Streaming planner**: dispatch workers as sub-questions stream from the
  planner rather than after the full plan. Latency win; costs plan-level
  dedup. Measure first.

---

## 12. Config reference

Implemented as `loop.WaveConfig`, a dataclass with these defaults, overridable
per call — and for the two knobs an operator needs at 3am without a deploy
(`WAVE_MAX_WAVES`, `WAVE_MAX_LLM_CALLS`), by environment variable. Not a YAML
file: that would be a dependency and a second place for the truth to live. The
"one knob philosophy" it exists for is intact — cheap and thorough are the same
code with different numbers.

```yaml
waves:
  max_waves: 3
  worker_timeout_s: 90
workers:
  max_concurrent: 6
  max_iters: 3
  query_variants: 2
budgets:
  max_total_llm_calls: 40
  max_total_tokens: null   # run-level override; the $ cap in api_usage is global
thresholds:
  dedup_sim: 0.90          # placeholder — see caveat below
  ctx_sim: 0.75            # placeholder — see caveat below
  escalate_score: 0.55     # grader relevance; tune on eval corpus
models:
  planner: gemini-3.5-flash
  worker: gemini-3.1-flash-lite
  sufficiency: gemini-3.1-flash-lite
  synthesis: gemini-3.5-flash
eval:
  deterministic: false
  snapshot_waves: true
```

Threshold caveat: `gemini-embedding-2` prefixes queries and documents
differently on purpose (`llm.as_query` / `llm.as_document`), and the pair is
calibrated together. Sub-question-vs-sub-question and fact-vs-sub-question
similarity are query-vs-query comparisons, a space those defaults were never
tuned for — treat `dedup_sim` and `ctx_sim` as placeholders until the eval
corpus (§9) exists to tune them. Embedding sub-questions and facts is billed
spend and is tracked like every other call (§6).
