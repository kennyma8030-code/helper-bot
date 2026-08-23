# Discord RAG Bot — Specs

> An agentic RAG system that gives a group chat the equivalent of perfect
> shared memory and careful reasoning over its own history — handles simple
> recall, multi-step reference chains, and open-ended pattern questions, with
> explicit uncertainty when the evidence doesn't support a clean answer.

Two parts: the **v1 core loop** (what gets built first, decisions locked),
and the **capability inventory** (the full design space, each item tagged
with where it lands). v1 tags: `[v1]` in scope now, `[v1-lite]` in scope in
a reduced form, `[later]` deferred until trajectory logs justify it.

---

# Part 1 — Core Loop, v1 Design

Single-loop, LLM-directed retrieval. One model, called repeatedly with
role-scoped prompts; no multi-agent machinery until logs prove it is needed.

## Shape

```
question
   │
   ▼
triage ──── simple lookup ──→ single retrieval → answer
   │
   ▼ (investigation)
┌─────────────────────────── loop (budgeted) ───────────────────────────┐
│  plan: read ledger → pose sufficiency question → if not sufficient,   │
│        emit retrieval calls (native function calling, 1..k per pass)  │
│  execute: run calls concurrently (asyncio.gather over db.py)          │
│  judge: same pass sees raw rows ONCE → extract facts/inferences into  │
│         ledger with message-ID citations → discard raw rows           │
└────────────────────────────────────────────────────────────────────────┘
   │
   ▼
synthesize from ledger only → answer (or explicit insufficiency)
```

## Decisions

1. **Native function calling, not JSON-in-text.** Retrieval requests use the
   API's function-calling protocol (manual loop — we own dispatch, never SDK
   auto-calling). Schema validation for free; no parsing of model output.
   The db.py retrieval functions are the tool surface, with flat explicit
   parameters and descriptions that teach instrument selection (keyword for
   names, SQL filters first, similarity last).

2. **Stop policy is mechanical, not vibes.** Three mechanisms, all required:
   - Hard budget (max passes / max retrieval calls) enforced in code.
   - Sufficiency is a posed question each pass — a required structured field
     ("answerable from ledger: yes / no / unanswerable"), never an implicit
     feeling. LLMs are miscalibrated stoppers in both directions.
   - "Unanswerable / insufficient evidence" is a first-class loop exit, so
     the model is never forced to manufacture an answer to escape.

3. **State is a structured, append-only ledger — not freeform carry-text.**
   Typed entries: `fact` (claim + message-ID citations, required),
   `inference` (claim + supporting fact IDs + competing explanations),
   `open_question`, `dead_branch`. Passes may add entries and close open
   questions; they may not rewrite a fact's text or drop its citations.
   Code validates shape each pass (a fact without citations is rejected).
   Rationale: freeform carried context rots — citations drop, hedges erode,
   inferences launder into facts across rewrites.

4. **Raw rows live for exactly one pass.** The pass that retrieved them sees
   full rows (content, author, timestamp, reply-link — the columns judgment
   needs), judges them into the ledger, and they are evicted. Later passes
   see ledger + their own new results only. Citations keep evidence
   re-addressable: any pass may re-fetch a cited message by ID if it needs
   the primary text again. Rationale: context cost is rows × passes (input
   tokens dominate; naive accumulation goes ~quadratic in loop depth), and
   unjudged rows persisting in context act as de facto evidence.

5. **Fork-join within a pass, no autonomous parallelism.** A pass may emit
   several retrieval calls; they execute concurrently and return together.
   Decisions stay centralized (one planner, one ledger writer); only
   execution is parallel.

6. ~~**Triage before the loop.** First call classifies: one-lookup question vs
   investigation. Simple lookups skip loop machinery entirely (routing
   discipline — don't escalate simple questions into agentic search).~~
   **Superseded 2026-08-18 by loop_spec.md §1.** Simple lookup is not this
   application's job — Discord's own search already answers "find the message
   where someone said X", so a lookup tier optimizes a path the bot should
   not be on. The bot exists for multi-step reasoning over the history, and
   every question enters the loop. What triage was protecting against —
   burning agentic budget on a light question — is handled by the wave
   design's cheap paths instead: the wave-1 "no sub-questions → direct
   retrieval" fallback, and the 1-wave configuration.

7. **Every pass is logged.** Retrieval specs issued, ledger diffs,
   sufficiency verdicts, budget spent. This is the debugging tool, the eval
   substrate, and the evidence base for any future architecture change.

## Deferred extensions — designed, NOT in v1

These are worked-out designs waiting on their trigger conditions. Do not
build them into v1; the v1 loop is deliberately their building block.

### Concurrent branch exploration (map-reduce over hypotheses)

For investigations that decompose into independent hypothesis branches
(disposition questions especially): explore branches concurrently, merge at
the end in a final call. This is NOT free-running parallel agents — the
coordination rules are what make it safe:

1. **Branches are compiled, then sealed.** The planner decomposes the
   question and writes each branch a fixed mandate + small budget (2-3
   passes). A branch is the v1 `investigate()` loop called with a narrower
   question — no new machinery. Branches never invent sub-branches or
   redefine their mission (no nesting beyond one level).
2. **Launch with a shared read-only snapshot of the parent ledger; run
   isolated.** Branches write only to their own branch ledger. No
   cross-branch communication mid-flight.
3. **Accepted cost: no mid-flight mooting.** A fact found by branch A cannot
   kill branch B before the merge. Waste is bounded by keeping per-branch
   budgets small. Do not fix this with shared memory — that rebuilds the
   coordination problem this design exists to avoid.
4. **The merge is a judgment call, not concatenation.** One strong-model
   call receives all branch ledgers and reconciles: dedupe facts, surface
   conflicts between branches (conflicts are signal), demote entries mooted
   by sibling facts, then synthesize. Works because branch ledgers carry
   citations — the merge adjudicates on evidence, not branch confidence.

Latency effect: N branches of depth d go from N*d sequential passes to ~d
wall-clock. Implementation is small once v1 exists: a "decompose into branch
specs" planner output mode, asyncio.gather over the existing loop, one merge
prompt.

### Execution-model routing (FUTURE — not in v1)

IN THE FUTURE, incoming questions get routed to different execution models
based on their category and complexity. Triage stops being a binary
lookup/investigation switch and becomes a dispatcher over a registry of
execution models, picking the cheapest one that can handle the question.

Execution models so far (only two conceived; the registry is open):

1. **Simple linear loop** — the v1 budgeted while loop. Default for
   lookups, single-anchor questions, and shallow investigations.
2. **Concurrent branching loops** — the map-reduce-over-hypotheses design
   below. For questions that decompose into independent hypothesis branches
   (disposition/pattern questions especially).

Design intent: every execution model shares the same contracts (question in;
ledger + answer out; same tool surface, same logging), so routing is a
dispatch decision, not an architectural fork. Adding a third execution model
later (e.g. a pure-SQL aggregation pipeline for meta-corpus questions) means
registering it with the router, not touching the others. Routing criteria
(category, expected depth, budget class) get tuned from trajectory logs.

### Ask-the-user clarification (human as a retrieval instrument)

Let the loop ask the question-asker for missing context instead of guessing
or burning budget: ambiguous references ("the trip" — which trip?),
half-remembered cues the corpus can't resolve, or a premise only the user
can confirm. Design constraints when built:

- It is a *tool* the planner can call (`ask_user(question)`), ranked
  cheapest-of-all instruments for facts only the user holds, but rate-limited
  hard: at most one clarification per investigation, asked early (during
  triage/first pass), never mid-loop minutes in — late questions feel broken
  in chat.
- The user's reply is evidence like any other: it enters the ledger as a
  fact attributed to the asker ("user says X"), not as ground truth — users
  misremember (see: showcase questions where the premise is wrong).
- Non-blocking: the investigation either pauses awaiting reply (Discord
  followup + wait, with timeout) or proceeds on branches that don't depend
  on the answer.
- Trigger to build it: logs show investigations wasting budget resolving
  references the asker could have disambiguated in one line.

---

# Part 2 — Capability Inventory

## Core architecture

- ~~`[v1-lite]` Tiered retrieval (simple lookup / multi-hop / aggregation /
  no-answer)~~ **Dropped 2026-08-18** with the triage decision above: there
  is no lookup tier, because lookup is not this bot's job
- ~~`[v1]` Query router — classify question type before retrieving
  (= triage)~~ **Dropped 2026-08-18**, same reason — see decision 6
- `[v1]` LLM-directed retrieval: planner authors queries; the question
  itself may never be embedded
- `[v1-lite]` Backward-chaining / dependency-discovery retrieval — falls out
  of the plan step naturally; no dedicated machinery in v1
- `[later]` Abductive hypothesis generation on dead ends ("what traces would
  the answer leave?")
- `[later]` Hypothesis branches compiled into concrete queries with
  unrelated vocabulary
- `[v1-lite]` Planner / integrator / synthesizer as separate prompts — v1:
  plan+judge share a pass, synthesis is its own call; full separation later
- `[v1]` Explicit working state (scratchpad = the ledger) instead of context
  accumulation
- `[v1]` Fact vs. inference separation, citation-required facts (validated
  in code)
- `[v1]` Opportunistic extraction (judge harvests facts the retrieval wasn't
  aiming for)
- `[v1]` Dead-end as first-class output; mid-chain no-answer
- `[v1-lite]` Retrieval failure diagnosis: evidence absent vs. query phrased
  badly (re-compile before closing branch) — v1 relies on the planner
  retrying phrasings; explicit diagnosis step later
- `[later]` Branch pruning: cheap-probe triage, scored frontier
  (plausibility × discriminative power) — v1 keeps only the global budget
  and sufficiency stopping
- `[v1]` Answer discloses unexplored branches (open_questions surface in
  synthesis)

## Retrieval mechanics

- `[v1]` Metadata-filtered vector search (structured columns + similarity in
  one query)
- `[v1]` Classification at ingestion time, not query time
- `[later]` Hybrid dense + sparse (BM25 / tsvector) with reciprocal rank
  fusion — v1 has ILIKE keyword search as the sparse stand-in
- `[v1]` Anchor-based retrieval (responses to a message, timestamp proximity)
- `[v1]` Aggregation computed in SQL, LLM narrates — never counts
- `[v1-lite]` Multi-phrasing per hypothesis, union results — planner may
  emit several phrasings in one pass; automatic union/dedup later
- `[later]` Multiple embedding spaces (general + domain-tuned),
  route/fuse/rerank patterns
- `[later]` Cross-encoder reranking as second pass
- `[v1-lite]` Wide-recall retrieval for pattern questions vs. top-k for
  lookups — expressed as per-call `limit`; no dedicated recall mode yet

## Data & epistemic discipline

- `[v1]` Temporal precedence (later statements supersede, per speaker)
- `[v1-lite]` Normalization (per-person rates against overall message volume).
  `db.message_counts_by_author` still supplies the denominator; the rate
  function that went with it was `category_rate_by_author`, removed on
  2026-08-16 with the `category` column. A numerator needs a new basis — see
  the Infrastructure note below.
- `[v1]` Small-sample hedging in synthesis (raw counts always alongside
  rates)
- `[v1-lite]` Silence-as-weak-evidence handling — prompt-level rule in v1;
  structural handling with the hypothesis layer later
- `[later]` Verification pass over the finished chain

## Infrastructure

- `[v1]` pgvector: Postgres holds the structured columns and the vectors, no
  sync problem. **Revised 2026-08-16:** the vectors are on `clusters`, not on
  `messages` — see specs-summaries.md's addendum.
- `[v1]` HNSW indexing (built over `clusters.embedding`; tuning deferred until
  data volume exists)
- `[later]` Filtered-ANN degradation and quantization (scale awareness —
  document the tradeoff, irrelevant at 3-person volume)
- ~~`[v1]` Async ingestion (fire-and-forget classification, non-blocking bot)~~
  **Dropped 2026-08-16.** Per-message classification (`category`, `sentiment`,
  `target_person_id`) was specced and indexed but never written by anything,
  and this doc already flagged its noise on joke-heavy sarcastic chat as an
  unmeasured risk. The columns are gone. Nullable columns nothing populates
  are worse than no columns: they make filters and rates that match nothing
  look available. What they were meant to answer is now carried by the
  summarizer's facets and cluster summaries, which cite message ids.
- `[later]` Synthetic evaluation via persona-seeded bot conversations with
  known ground truth — the eval substrate in v1 is trajectory logs

## Deferred-item triggers

Deferred items are not a wishlist — each has an observable trigger:

- Frontier scoring / hypothesis branching → logs show disposition questions
  starving useful branches or wasting budget on hopeless ones
- Role separation into distinct calls/tiers → judge step degrades when
  sharing a pass with planning, or plan quality is the observed bottleneck
- Hybrid sparse + RRF, reranking → keyword ILIKE demonstrably missing
  recall that tsvector/BM25 would catch
- Verification pass → synthesized answers found citing facts the chain
  doesn't support
- Synthetic eval → prompt/architecture changes need regression protection
  beyond spot-checking logs
- Concurrent branch exploration → disposition questions measurably slow
  (deep sequential pass chains) AND their branches are genuinely independent
  in the logs
- Ask-the-user clarification → logs show budget wasted resolving references
  the asker could have disambiguated in one line
- Execution-model routing → a second execution model exists (i.e. concurrent
  branching ships) — before that there is nothing to route between
