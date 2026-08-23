"""The wave loop — orchestrated multi-branch retrieval (loop_spec.md).

One orchestrator runs the show. Each iteration is a wave:

    plan -> dispatch workers (parallel) -> barrier -> merge -> sufficiency
         -> replan or stop -> synthesize

A worker owns exactly one sub-question and has no spawning rights, so every
decision about what to explore next is made at a wave boundary with the whole
ledger visible. Workers write nothing while they run; everything they found
merges at the barrier.

Entry point for bot.py:  answer(question) -> str   (Discord-ready, <2000 chars)
"""

import asyncio
import json
import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Sequence

from google.genai import types

import db
import retrieval
from llm import (
    LARGE_MODEL,
    SMALL_MODEL,
    ask_gemini,
    client,
    embed_texts,
    extract_json as _extract_json,
    track_usage,
)
from prompts import (
    GRADER_PROMPT,
    PLANNER_PROMPT,
    SUFFICIENCY_PROMPT,
    SYNTH_PROMPT,
    WORKER_PROMPT,
)

log = logging.getLogger(__name__)

# Model tiers come from llm.py (LARGE_MODEL, SMALL_MODEL): planning gates
# everything downstream and synthesis is the answer a human reads, so both take
# the large tier; the per-iteration work that runs many times over takes the
# small one and escalates only on evidence that it is stuck.


@dataclass
class WaveConfig:
    """Every knob, in one place (loop_spec.md §12).

    The same architecture runs cheap (one wave, three workers) or thorough
    (three waves, six workers) by configuration rather than by code change, so
    the shape of a run is a decision the caller makes.

    Not a YAML file: the knobs are these defaults, overridable per call and —
    for the two an operator needs at 3am without a deploy — by environment
    variable. A config file would be a dependency and a second place for the
    truth to live.
    """

    # Waves
    max_waves: int = int(os.environ.get("WAVE_MAX_WAVES", "3"))

    # 90s, not the 30 the spec first guessed: a worker makes up to
    # worker_max_iters rounds and each round is two sequential model calls plus
    # its searches, so 30 seconds would cancel nearly every worker that used its
    # rounds and turn the timeout from a safety net into the normal exit.
    worker_timeout_s: float = 90.0

    # Workers
    max_concurrent: int = 6
    worker_max_iters: int = 3

    # Budgets. These bound one run; the dollar ceiling in api_usage bounds the
    # service, and llm.track_usage is what feeds it.
    max_llm_calls: int = int(os.environ.get("WAVE_MAX_LLM_CALLS", "40"))
    max_tokens: Optional[int] = None

    # Thresholds. Both compare a query-shaped embedding to another query-shaped
    # embedding, which is not the pairing gemini-embedding-2's prefixes were
    # tuned for — treat them as placeholders until there is an eval corpus to
    # tune them on (loop_spec.md §9).
    #
    # There is no relevance threshold here any more. The grader used to score
    # every row 0.0-1.0 and two gates compared that score against a number, but
    # the score came from the same call, the same model, and the same rows as
    # the grader's own status — so it was one witness answering twice, not
    # corroboration, and both numbers were guesses nothing had calibrated. The
    # grader's status and its ambiguity flag say the same thing without the
    # arithmetic.
    dedup_sim: float = 0.90
    ctx_sim: float = 0.75

    # How much ledger a worker may be shown, and when it is worth paying an
    # embedding call to choose which part. Under this many facts, everything
    # fits and filtering would cost more than it saves.
    worker_ctx_tokens: int = 1500
    ctx_filter_min_facts: int = 12

    # Eval / debugging
    deterministic: bool = False
    snapshot_dir: Optional[str] = os.environ.get("WAVE_SNAPSHOT_DIR") or None


# ---------------------------------------------------------------------------
# Budget — the counters that end a run
# ---------------------------------------------------------------------------

@dataclass
class Budget:
    """Run-scoped counters. Not money: money is api_usage, and every call here
    is recorded there too."""

    max_llm_calls: int
    max_tokens: Optional[int] = None
    llm_calls: int = 0
    tokens: int = 0

    @property
    def exhausted(self) -> bool:
        if self.llm_calls >= self.max_llm_calls:
            return True
        return self.max_tokens is not None and self.tokens >= self.max_tokens

    @property
    def remaining_calls(self) -> int:
        return max(self.max_llm_calls - self.llm_calls, 0)

    def record(self, response: Any) -> None:
        self.llm_calls += 1
        usage = getattr(response, "usage_metadata", None)
        if usage is not None:
            self.tokens += getattr(usage, "total_token_count", 0) or 0


# ---------------------------------------------------------------------------
# Ledger — append-only working state (specs.md decision 3)
# ---------------------------------------------------------------------------

@dataclass
class Ledger:
    """What the run has established. Entries are added, never rewritten.

    Append-only is the point: rewritten state rots — citations drop, hedges
    erode, and an inference launders itself into a fact over a few passes. So a
    fact that contradicts an earlier one is appended alongside it rather than
    replacing it, and the disagreement is left standing for synthesis to
    adjudicate on the evidence. There is deliberately no mechanical
    contradiction detector: deciding whether two English claims conflict is a
    reading task, and a model doing it with both claims and their citations in
    front of it is the only version of that check worth having.
    """

    question: str
    facts: list[dict[str, Any]] = field(default_factory=list)
    inferences: list[dict[str, Any]] = field(default_factory=list)
    open_questions: list[str] = field(default_factory=list)
    dead_branches: list[str] = field(default_factory=list)

    # Sub-question bookkeeping, so the planner is never handed work already done.
    resolved: list[str] = field(default_factory=list)
    unresolved: list[str] = field(default_factory=list)
    attempted: list[str] = field(default_factory=list)
    attempted_vectors: list[Sequence[float]] = field(default_factory=list)

    # Evidence seen. Bookkeeping, not context: message bodies are read by the
    # worker that retrieved them and by its grader, then never re-injected —
    # unjudged rows sitting in a prompt act as de facto evidence, and the cost
    # of carrying them is rows x waves (specs.md decision 4). Anything needed
    # again is re-fetched by id.
    message_ids: set[str] = field(default_factory=set)

    # Embeddings of fact claims, for choosing which facts a worker is shown.
    # Keyed by fact id so each claim is embedded once per run, not once per wave.
    fact_vectors: dict[str, Sequence[float]] = field(default_factory=dict)

    def apply(self, updates: dict[str, Any], *, source: str = "") -> list[str]:
        """Add one grader's output. Returns notes about what was rejected.

        Facts are validated here rather than trusted: no citations dict, no
        fact. This is the one rule that keeps the answer traceable.
        """
        rejected: list[str] = []

        for fact in updates.get("facts") or []:
            claim = fact.get("claim")
            citations = fact.get("citations")
            if not claim or not isinstance(citations, dict) or not citations:
                rejected.append(
                    f"fact rejected (needs claim + non-empty citations dict): {str(fact)[:150]}"
                )
                continue
            self.facts.append({
                "id": f"F{len(self.facts) + 1}",
                "claim": str(claim),
                "citations": {str(k): str(v) for k, v in citations.items()},
                "source": source,
            })

        for inf in updates.get("inferences") or []:
            claim = inf.get("claim")
            if not claim:
                rejected.append(f"inference rejected (no claim): {str(inf)[:150]}")
                continue
            self.inferences.append({
                "id": f"I{len(self.inferences) + 1}",
                "claim": str(claim),
                "based_on": [str(x) for x in inf.get("based_on") or []],
                "competing": [str(x) for x in inf.get("competing") or []],
                "source": source,
            })

        for q in updates.get("open_questions") or []:
            if q not in self.open_questions:
                self.open_questions.append(str(q))
        for q in updates.get("resolved_questions") or []:
            if q in self.open_questions:
                self.open_questions.remove(q)
        for b in updates.get("dead_branches") or []:
            if b not in self.dead_branches:
                self.dead_branches.append(str(b))

        return rejected

    def render(self) -> str:
        """The whole ledger, for synthesis and for the log."""
        if not (self.facts or self.inferences or self.open_questions or self.dead_branches):
            return "(empty — nothing established yet)"
        lines: list[str] = []
        if self.facts:
            lines.append("FACTS (citation-backed):")
            for f in self.facts:
                lines.append(f"  {f['id']}. {f['claim']}")
                lines.append(f"      citations: {json.dumps(f['citations'], ensure_ascii=False)}")
        if self.inferences:
            lines.append("INFERENCES (interpretations, not facts):")
            for i in self.inferences:
                based = ", ".join(i["based_on"]) or "-"
                lines.append(f"  {i['id']}. {i['claim']} [based on: {based}]")
                for c in i["competing"]:
                    lines.append(f"      competing: {c}")
        if self.open_questions:
            lines.append("OPEN QUESTIONS:")
            lines.extend(f"  - {q}" for q in self.open_questions)
        if self.dead_branches:
            lines.append("DEAD BRANCHES:")
            lines.extend(f"  - {b}" for b in self.dead_branches)
        return "\n".join(lines)

    def digest(self) -> str:
        """What the planner sees: established facts, and what is still open.

        Not the full ledger — the planner writes sub-questions, so it needs to
        know what is answered and what is missing, not every excerpt behind
        every claim.
        """
        parts: list[str] = []
        if self.facts:
            parts.append("ESTABLISHED:\n" + "\n".join(
                f"  - {f['claim']}" for f in self.facts))
        if self.inferences:
            parts.append("INTERPRETED (not established):\n" + "\n".join(
                f"  - {i['claim']}" for i in self.inferences))
        if self.resolved:
            parts.append("SUB-QUESTIONS ALREADY ANSWERED:\n" + "\n".join(
                f"  - {q}" for q in self.resolved))
        if self.unresolved:
            parts.append("SUB-QUESTIONS TRIED AND NOT ANSWERED:\n" + "\n".join(
                f"  - {q}" for q in self.unresolved))
        if self.open_questions:
            parts.append("OPEN:\n" + "\n".join(f"  - {q}" for q in self.open_questions))
        if self.dead_branches:
            parts.append("CLOSED (do not re-open):\n" + "\n".join(
                f"  - {b}" for b in self.dead_branches))
        return "\n\n".join(parts) or "(empty — this is the first wave)"


# ---------------------------------------------------------------------------
# Records
# ---------------------------------------------------------------------------

@dataclass
class SubQuestion:
    text: str
    rationale: str = ""
    priority: int = 1
    expected_answer_type: str = ""
    vector: Optional[Sequence[float]] = None


@dataclass
class WorkerResult:
    sub_question: str
    status: str                      # resolved | refine | unresolvable | error | timeout
    facts: list[dict[str, Any]] = field(default_factory=list)
    inferences: list[dict[str, Any]] = field(default_factory=list)
    message_ids: list[str] = field(default_factory=list)
    calls: list[dict[str, Any]] = field(default_factory=list)
    iterations: int = 0
    escalated: bool = False
    gap: str = ""
    note: str = ""


@dataclass
class WaveRun:
    question: str
    ledger: Ledger
    verdict: str                     # yes | unanswerable | budget_exhausted | stalled
    waves: int
    llm_calls: int
    tokens: int
    seconds: float
    trajectory: list[dict[str, Any]] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Model calls — one door, so nothing goes uncounted
# ---------------------------------------------------------------------------

# This loop dispatches tool calls itself (see retrieval.execute_call), so the
# SDK must not try to call anything on its own. TOOLS is declarative —
# FunctionDeclaration objects, not Python callables — so automatic function
# calling has nothing it could invoke and never fires either way; the SDK still
# logs a recommendation to use AsyncChat.send_message on every call with tools
# attached. Saying no outright silences that and makes the manual dispatch
# explicit.
_NO_AUTO_CALLING = types.AutomaticFunctionCallingConfig(disable=True)

CACHE_TTL_SECONDS = 3600

# The worker prompt and the six tool schemas are identical on every iteration of
# every worker of every wave — the one payload in this system big enough and
# repeated enough to be worth caching server-side. The other roles send their
# prompt once per wave and are mostly below the minimum cacheable size anyway.
_caches: dict[str, str] = {}
_cache_lock = asyncio.Lock()


async def _worker_cache(model: str) -> Optional[str]:
    """The cached worker prompt + tools, created on first use. None if
    unavailable, which is a normal outcome rather than an error: a prompt under
    the model's minimum cacheable size is refused, and the API may decline for
    its own reasons. The caller then sends the prompt inline, which costs more
    per call and behaves identically.
    """
    key = f"worker:{model}"
    if key in _caches:
        return _caches[key]

    async with _cache_lock:
        # Someone else may have created it while this call waited for the lock.
        if key in _caches:
            return _caches[key]
        try:
            cache = await client.aio.caches.create(
                model=model,
                config=types.CreateCachedContentConfig(
                    system_instruction=WORKER_PROMPT,
                    tools=[retrieval.TOOLS],
                    ttl=f"{CACHE_TTL_SECONDS}s",
                    display_name=f"helper-bot worker ({model})",
                ),
            )
        except Exception as e:
            log.warning("worker cache unavailable for %s (%s: %s); sending the "
                        "prompt inline instead", model, type(e).__name__, e)
            return None
        if not cache.name:
            # A cache with no handle cannot be referenced, so it is the same as
            # not having one.
            return None
        _caches[key] = cache.name
        log.info("worker cache created for %s: %s (ttl %ds)",
                 model, cache.name, CACHE_TTL_SECONDS)
        return cache.name


def _forget_cache(model: str, name: str) -> None:
    """Drop a cache that stopped working, unless it has already been replaced."""
    key = f"worker:{model}"
    if _caches.get(key) == name:
        _caches.pop(key, None)


async def _call_model(
    *,
    model: str,
    contents: str,
    budget: Budget,
    label: str = "call",
    system: Optional[str] = None,
    tools: bool = False,
    cache_name: Optional[str] = None,
    json_out: bool = False,
    deterministic: bool = False,
) -> Any:
    """One model call, counted twice: against this run's budget and against the
    dollar ceiling in api_usage.

    These calls go straight to the client rather than through ask_gemini, so
    track_usage has to happen here or the loop — the most expensive thing this
    service does — would not show up in the spend at all.
    """
    def _config(with_cache: Optional[str]) -> types.GenerateContentConfig:
        kwargs: dict[str, Any] = {"automatic_function_calling": _NO_AUTO_CALLING}
        if with_cache:
            # With a cache the prompt and tools live inside it and must not be
            # repeated here — sending both is rejected.
            kwargs["cached_content"] = with_cache
        else:
            if system:
                kwargs["system_instruction"] = system
            if tools:
                kwargs["tools"] = [retrieval.TOOLS]
        if json_out:
            # Forces syntactically valid JSON, which removes a whole class of
            # "the model wrapped it in prose" failure. Cannot be combined with
            # tools, and none of the JSON roles have any.
            kwargs["response_mime_type"] = "application/json"
        if deterministic:
            kwargs["temperature"] = 0.0
        return types.GenerateContentConfig(**kwargs)

    try:
        response = await client.aio.models.generate_content(
            model=model, contents=contents, config=_config(cache_name),
        )
    except Exception as e:
        if cache_name is None:
            raise
        # The cache expires on its own schedule and can be deleted from under us
        # mid-run. Retry inline rather than losing the call; if that fails too,
        # the error is real and propagates.
        log.warning("call failed with the cache (%s: %s); retrying without it",
                    type(e).__name__, e)
        _forget_cache(model, cache_name)
        response = await client.aio.models.generate_content(
            model=model, contents=contents, config=_config(None),
        )

    budget.record(response)
    await track_usage(model, response)

    # One line per model call, so a wave's cost is readable as it happens rather
    # than reconstructed from the running total afterwards. `cached` is the part
    # of the input the server billed at the cache rate — it is what tells you
    # whether the worker cache actually engaged, which nothing else reveals.
    usage = getattr(response, "usage_metadata", None)
    prompt_tokens = getattr(usage, "prompt_token_count", 0) or 0 if usage else 0
    total_tokens = getattr(usage, "total_token_count", 0) or 0 if usage else 0
    cached_tokens = getattr(usage, "cached_content_token_count", 0) or 0 if usage else 0
    log.info("    call %-11s %-22s in=%-6d out=%-5d%s  [%d/%d calls]",
             label, model, prompt_tokens, max(total_tokens - prompt_tokens, 0),
             f" cached={cached_tokens}" if cached_tokens else "",
             budget.llm_calls, budget.max_llm_calls)
    return response


def _response_parts(response: Any) -> tuple[str, list[Any]]:
    """Split a Gemini response into its text and its function calls."""
    text_chunks: list[str] = []
    calls: list[Any] = []
    for cand in (response.candidates or [])[:1]:
        content = cand.content
        for part in (content.parts or []) if content else []:
            if getattr(part, "function_call", None):
                calls.append(part.function_call)
            elif getattr(part, "text", None):
                text_chunks.append(part.text)
    return "\n".join(text_chunks), calls


def _log_block(title: str, body: str) -> None:
    """Log a multi-line body under one heading, indented so it stays readable
    as one unit in a stream of interleaved log lines."""
    log.info("%s\n%s", title, "\n".join(f"    {line}" for line in body.splitlines()))


def _cosine(a: Sequence[float], b: Sequence[float]) -> float:
    """Dot product. Both sides come from llm.embed_texts, which returns unit
    vectors, so the norms are 1 and the division cosine would do is a no-op."""
    return sum(x * y for x, y in zip(a, b))


# ---------------------------------------------------------------------------
# Planner — one call per wave, no tools
# ---------------------------------------------------------------------------

async def _plan(
    question: str,
    ledger: Ledger,
    wave: int,
    gaps: list[str],
    cfg: WaveConfig,
    budget: Budget,
) -> tuple[list[SubQuestion], str]:
    """Sub-questions for this wave, and the planner's one-line note."""
    parts = [
        f"ORIGINAL QUESTION: {question}",
        f"WAVE: {wave} of at most {cfg.max_waves}",
        "WHAT IS KNOWN SO FAR:\n" + ledger.digest(),
    ]
    if gaps:
        parts.append("GAPS THE LAST WAVE LEFT (aim this wave at these):\n"
                     + "\n".join(f"  - {g}" for g in gaps))
    if wave == 1:
        parts.append("This is the first wave: decompose the original question.")
    else:
        parts.append("Later wave: write sub-questions ONLY for what is still "
                     "missing. Do not re-ask anything already answered.")

    response = await _call_model(
        model=LARGE_MODEL,
        contents="\n\n".join(parts),
        budget=budget,
        label="planner",
        system=PLANNER_PROMPT,
        json_out=True,
        deterministic=cfg.deterministic,
    )
    text, _ = _response_parts(response)
    parsed = _extract_json(text) or {}

    subs: list[SubQuestion] = []
    for raw in parsed.get("sub_questions") or []:
        text_q = (raw.get("sub_question") or "").strip() if isinstance(raw, dict) else ""
        if not text_q:
            continue
        try:
            priority = int(raw.get("priority") or 1)
        except (TypeError, ValueError):
            priority = 1
        subs.append(SubQuestion(
            text=text_q,
            rationale=str(raw.get("rationale") or ""),
            priority=priority,
            expected_answer_type=str(raw.get("expected_answer_type") or ""),
        ))
    subs.sort(key=lambda s: s.priority)
    return subs, str(parsed.get("note") or "")


async def _dedup(
    subs: list[SubQuestion], ledger: Ledger, cfg: WaveConfig,
) -> tuple[list[SubQuestion], list[str]]:
    """Drop sub-questions already worked, and near-duplicates within the wave.

    One embedding call for the whole wave. Similarity here is query-shaped text
    against query-shaped text, which is not the pairing the prefixes in
    llm.as_query/as_document were tuned for, so dedup_sim is deliberately high:
    a missed duplicate costs one worker, a wrong merge costs a whole line of
    inquiry.
    """
    if not subs:
        return [], []

    try:
        vectors = await embed_texts([s.text for s in subs], is_query=True)
    except Exception as e:
        # Losing dedup costs duplicate work; failing the wave costs the answer.
        log.warning("sub-question dedup unavailable (%s: %s); running them all",
                    type(e).__name__, e)
        return subs, []

    kept: list[SubQuestion] = []
    dropped: list[str] = []
    for sub, vec in zip(subs, vectors):
        sub.vector = vec
        prior = list(ledger.attempted_vectors) + [k.vector for k in kept if k.vector]
        if any(_cosine(vec, p) >= cfg.dedup_sim for p in prior if p):
            dropped.append(sub.text)
            continue
        kept.append(sub)
    return kept, dropped


# ---------------------------------------------------------------------------
# Worker — one sub-question, bounded, no spawning rights
# ---------------------------------------------------------------------------

def _estimate_tokens(text: str) -> int:
    """Rough chars/4. Precise enough to keep a prompt inside a cap, and it
    costs nothing — a real count would be an API round trip per worker."""
    return len(text) // 4


async def _worker_view(ledger: Ledger, sub: SubQuestion, cfg: WaveConfig) -> str:
    """The slice of the ledger one worker is shown.

    Never the whole thing: a worker chasing one sub-question does not need
    every fact the other five found, and a prompt that grows with the ledger
    makes every wave cost more than the last. Below ctx_filter_min_facts
    everything fits and the embedding call to choose would cost more than it
    saves; above it, facts are ranked against this sub-question and the tail is
    dropped.
    """
    if not ledger.facts:
        return "(nothing established yet)"

    facts = ledger.facts
    if len(facts) > cfg.ctx_filter_min_facts and sub.vector is not None:
        missing = [f for f in facts if f["id"] not in ledger.fact_vectors]
        if missing:
            try:
                vecs = await embed_texts([f["claim"] for f in missing], is_query=True)
                for fact, vec in zip(missing, vecs):
                    ledger.fact_vectors[fact["id"]] = vec
            except Exception as e:
                log.warning("fact embedding unavailable (%s: %s); showing the "
                            "most recent facts instead", type(e).__name__, e)
        scored = [
            (_cosine(sub.vector, ledger.fact_vectors[f["id"]]), f)
            for f in facts if f["id"] in ledger.fact_vectors
        ]
        if scored:
            scored.sort(key=lambda pair: pair[0], reverse=True)
            facts = [f for score, f in scored if score >= cfg.ctx_sim] or \
                    [f for _, f in scored[:3]]

    lines: list[str] = []
    used = 0
    for fact in facts:
        line = f"  {fact['id']}. {fact['claim']}"
        cost = _estimate_tokens(line)
        if used + cost > cfg.worker_ctx_tokens:
            lines.append("  … (older facts omitted)")
            break
        lines.append(line)
        used += cost
    return "ESTABLISHED SO FAR (do not re-establish these):\n" + "\n".join(lines)


def _grading_input(executed: list[tuple[str, dict[str, Any], Any]]) -> tuple[str, list[str]]:
    """What the grader sees, and the message ids inside it.

    Counts and cluster hits go in alongside the messages: the grader has to be
    able to say a count answered the sub-question, and to see that a cluster hit
    points at a span nobody has read yet. Only messages carry ids it may cite.
    """
    ids: list[str] = []
    for name, _args, result in executed:
        if isinstance(result, BaseException) or name not in retrieval.MESSAGE_TOOLS:
            continue
        for row in result:
            mid = row.get("discord_message_id")
            if mid is not None:
                ids.append(str(mid))
    return retrieval.render_results(executed), ids


async def _run_worker(
    sub: SubQuestion,
    ledger: Ledger,
    cfg: WaveConfig,
    budget: Budget,
) -> WorkerResult:
    """Retrieve for one sub-question. Bounded iterations, no spawning."""
    result = WorkerResult(sub_question=sub.text, status="unresolvable")
    model = SMALL_MODEL
    view = await _worker_view(ledger, sub, cfg)
    history: list[str] = []

    for iteration in range(1, cfg.worker_max_iters + 1):
        if budget.exhausted:
            result.note = "stopped: run budget exhausted"
            break
        result.iterations = iteration

        parts = [
            f"YOUR SUB-QUESTION: {sub.text}",
            f"WHAT A GOOD ANSWER LOOKS LIKE: {sub.expected_answer_type or 'whatever the evidence supports'}",
            f"(The original question this serves: {ledger.question})",
            view,
            f"ROUND {iteration} of {cfg.worker_max_iters}.",
        ]
        if history:
            parts.append("WHAT YOUR EARLIER ROUNDS FOUND:\n" + "\n\n".join(history))

        cache_name = await _worker_cache(model)
        response = await _call_model(
            model=model,
            contents="\n\n".join(parts),
            budget=budget,
            label="worker",
            system=WORKER_PROMPT,
            tools=True,
            cache_name=cache_name,
            deterministic=cfg.deterministic,
        )
        text, calls = _response_parts(response)

        if not calls:
            # Nothing left worth searching. Whatever earlier rounds established
            # stands; the status from the last grading (if any) is kept.
            result.note = (text or "worker requested no searches").strip()[:300]
            if iteration == 1:
                result.status = "unresolvable"
            break

        specs = [(c.name, dict(c.args or {})) for c in calls]
        for name, args in specs:
            log.info("    [w] %s: %s(%s)", sub.text[:40], name,
                     json.dumps(args, default=str))
            result.calls.append({"tool": name, "args": args, "round": iteration})

        raw = await asyncio.gather(
            *(retrieval.execute_call(name, args) for name, args in specs),
            return_exceptions=True,
        )
        executed = [(name, args, res) for (name, args), res in zip(specs, raw)]
        for name, args, res in executed:
            if isinstance(res, BaseException):
                log.warning("    [w] %s failed: %s: %s", name, type(res).__name__, res)

        rendered, ids = _grading_input(executed)
        result.message_ids.extend(ids)

        if budget.exhausted:
            result.note = "stopped before grading: run budget exhausted"
            break

        grade_response = await _call_model(
            model=model,
            contents=(
                f"SUB-QUESTION: {sub.text}\n\n"
                f"{view}\n\n"
                f"SEARCH RESULTS TO JUDGE:\n{rendered}"
            ),
            budget=budget,
            label="grader",
            system=GRADER_PROMPT,
            json_out=True,
            deterministic=cfg.deterministic,
        )
        grade_text, _ = _response_parts(grade_response)
        graded = _extract_json(grade_text) or {}

        if not graded:
            log.warning("    [w] %s: grader returned no JSON", sub.text[:40])
            history.append(f"Round {iteration}: results could not be judged.")
            continue

        result.facts.extend(graded.get("facts") or [])
        result.inferences.extend(graded.get("inferences") or [])
        status = str(graded.get("status") or "refine")
        result.status = status if status in ("resolved", "refine", "unresolvable") else "refine"
        result.gap = str(graded.get("gap") or "")
        result.note = str(graded.get("note") or "")
        ambiguous = bool(graded.get("ambiguous"))

        log.info("    [w] %s: round %d -> %s%s (%d facts)",
                 sub.text[:40], iteration, result.status,
                 " [ambiguous]" if ambiguous else "",
                 len(graded.get("facts") or []))

        if result.status in ("resolved", "unresolvable"):
            break

        # Escalate on evidence, never on prediction. Two triggers, both of them
        # something that already happened rather than a number about it:
        #
        #   - the grader says it cannot tell (ambiguous), or
        #   - the worker is about to start its last round and still has not
        #     resolved this, so the cheap model has had every chance it is
        #     going to get.
        #
        # The last-round rule is what bounds the cost: rounds before it stay on
        # the small model, and only the final attempt is expensive. One way,
        # and only once — a second escalation would be the same model again.
        last_round = iteration >= cfg.worker_max_iters - 1
        if not result.escalated and (ambiguous or last_round):
            model = LARGE_MODEL
            result.escalated = True
            log.info("    [w] %s: escalating to %s (%s)", sub.text[:40], LARGE_MODEL,
                     "grader was unsure" if ambiguous else "final round")

        found = "; ".join(f"{f.get('claim')}" for f in (graded.get("facts") or [])[:5])
        history.append(
            f"Round {iteration}: {result.note or '(no note)'}\n"
            f"  established: {found or '(nothing)'}\n"
            f"  still missing: {result.gap or '(unspecified)'}"
        )
        view = await _worker_view(ledger, sub, cfg)

    return result


async def _dispatch(
    subs: list[SubQuestion],
    ledger: Ledger,
    cfg: WaveConfig,
    budget: Budget,
) -> list[WorkerResult]:
    """Run one wave's workers and join them at the barrier.

    A worker that raises or times out costs its sub-question, never the wave:
    it comes back as a gap and the planner can aim the next wave at it.
    """
    semaphore = asyncio.Semaphore(cfg.max_concurrent)

    async def guarded(sub: SubQuestion) -> WorkerResult:
        async with semaphore:
            try:
                return await asyncio.wait_for(
                    _run_worker(sub, ledger, cfg, budget),
                    timeout=cfg.worker_timeout_s,
                )
            except asyncio.TimeoutError:
                log.warning("  worker timed out after %.0fs: %s",
                            cfg.worker_timeout_s, sub.text)
                return WorkerResult(
                    sub_question=sub.text, status="timeout",
                    note=f"cancelled at the barrier after {cfg.worker_timeout_s:.0f}s",
                )
            except Exception as e:
                # Caught here rather than only at the gather below, so the
                # sequential path fails the same way the parallel one does. A
                # model error on one sub-question must cost that sub-question,
                # never the wave.
                log.warning("  worker failed on %r: %s: %s",
                            sub.text, type(e).__name__, e)
                return WorkerResult(
                    sub_question=sub.text, status="error",
                    note=f"{type(e).__name__}: {e}",
                )

    if cfg.deterministic:
        # Sequential, in planner-priority order, so a run can be replayed.
        return [await guarded(sub) for sub in subs]

    # return_exceptions stays on as a backstop: guarded() handles Exception, but
    # a BaseException (a cancellation propagating in) would otherwise take the
    # whole gather down mid-wave.
    settled = await asyncio.gather(*(guarded(sub) for sub in subs),
                                   return_exceptions=True)
    results: list[WorkerResult] = []
    for sub, outcome in zip(subs, settled):
        if isinstance(outcome, BaseException):
            log.warning("  worker failed on %r: %s: %s",
                        sub.text, type(outcome).__name__, outcome)
            results.append(WorkerResult(
                sub_question=sub.text, status="error",
                note=f"{type(outcome).__name__}: {outcome}",
            ))
        else:
            results.append(outcome)
    return results


def _merge(ledger: Ledger, results: list[WorkerResult]) -> list[str]:
    """Fold one wave's workers into the ledger. The only writer."""
    notes: list[str] = []
    for result in results:
        notes.extend(ledger.apply(
            {"facts": result.facts, "inferences": result.inferences},
            source=result.sub_question[:80],
        ))
        ledger.message_ids.update(result.message_ids)
        if result.sub_question not in ledger.attempted:
            ledger.attempted.append(result.sub_question)
        if result.status == "resolved":
            ledger.resolved.append(result.sub_question)
        else:
            ledger.unresolved.append(result.sub_question)
            if result.gap and result.gap not in ledger.open_questions:
                ledger.open_questions.append(result.gap)
        if result.status == "unresolvable" and result.note:
            branch = f"{result.sub_question} — {result.note}"
            if branch not in ledger.dead_branches:
                ledger.dead_branches.append(branch)
    return notes


# ---------------------------------------------------------------------------
# Sufficiency — the original question, read against the whole ledger
# ---------------------------------------------------------------------------

async def _sufficient(
    ledger: Ledger,
    results: list[WorkerResult],
    cfg: WaveConfig,
    budget: Budget,
) -> tuple[str, list[str], str]:
    """(verdict, gaps, note). Verdict is yes | no | unanswerable.

    This is the only place the ORIGINAL question is read against the whole
    ledger. Every worker sees one sub-question and nothing else, so all of them
    can succeed while the question stays unanswered — the answer often lives in
    how the parts compose, and no worker is in a position to look at that. It
    is also where the gaps that aim the next wave come from, and the only thing
    that can call the whole run unanswerable.

    There was a "cheap gate" here that returned yes without asking the model
    whenever every sub-question came back resolved. It fired in exactly the
    case the check exists for: workers all reporting success is when a bad
    decomposition is invisible, because nothing has compared what they found
    against what was asked. It saved one small-model call per wave and skipped
    the loop's only end-to-end verification to do it.
    """
    if budget.exhausted:
        return "no", [], "budget exhausted before the sufficiency check"

    statuses = "\n".join(
        f"  - [{r.status}] {r.sub_question}"
        + (f"\n      gap: {r.gap}" if r.gap else "")
        for r in results
    ) or "  (no sub-questions ran)"

    response = await _call_model(
        model=SMALL_MODEL,
        contents=(
            f"QUESTION: {ledger.question}\n\n"
            f"LEDGER:\n{ledger.render()}\n\n"
            f"SUB-QUESTION STATUS THIS WAVE:\n{statuses}"
        ),
        budget=budget,
        label="sufficiency",
        system=SUFFICIENCY_PROMPT,
        json_out=True,
        deterministic=cfg.deterministic,
    )
    text, _ = _response_parts(response)
    parsed = _extract_json(text) or {}
    verdict = str(parsed.get("sufficient") or "no")
    if verdict not in ("yes", "no", "unanswerable"):
        verdict = "no"
    gaps = [str(g) for g in (parsed.get("gaps") or []) if str(g).strip()]
    return verdict, gaps, str(parsed.get("note") or "")


# ---------------------------------------------------------------------------
# Snapshots — a wave, replayable
# ---------------------------------------------------------------------------

def _snapshot(cfg: WaveConfig, run_id: str, wave: int, payload: dict[str, Any]) -> None:
    """Write one wave's state to JSONL, when a snapshot directory is set.

    Off unless WAVE_SNAPSHOT_DIR names somewhere, because the container this
    runs in has an ephemeral filesystem — a snapshot written there survives
    until the next deploy, so the logs are the durable record and this is for
    local eval runs.
    """
    if not cfg.snapshot_dir:
        return
    try:
        path = Path(cfg.snapshot_dir) / run_id
        path.mkdir(parents=True, exist_ok=True)
        with (path / f"wave_{wave}.jsonl").open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(payload, ensure_ascii=False, default=str) + "\n")
    except Exception as e:
        log.warning("could not write wave snapshot (%s: %s)", type(e).__name__, e)


# ---------------------------------------------------------------------------
# The orchestrator
# ---------------------------------------------------------------------------

async def investigate(question: str, *, cfg: Optional[WaveConfig] = None) -> WaveRun:
    """Run the wave loop. Returns the full record of what happened."""
    cfg = cfg or WaveConfig()
    ledger = Ledger(question)
    budget = Budget(max_llm_calls=cfg.max_llm_calls, max_tokens=cfg.max_tokens)
    trajectory: list[dict[str, Any]] = []
    run_id = f"{int(time.time())}"
    started = time.monotonic()
    gaps: list[str] = []
    verdict = "no"
    wave = 0

    await db.open_pool()

    log.info("investigating %r (run %s; up to %d waves, %d workers, %d llm calls)",
             question, run_id, cfg.max_waves, cfg.max_concurrent, cfg.max_llm_calls)

    while wave < cfg.max_waves:
        wave += 1
        if budget.exhausted:
            verdict = "budget_exhausted"
            log.info("wave %d not started: budget exhausted", wave)
            break

        subs, plan_note = await _plan(question, ledger, wave, gaps, cfg, budget)
        log.info("wave %d: planner proposed %d sub-question(s)%s",
                 wave, len(subs), f" — {plan_note}" if plan_note else "")

        subs, dropped = await _dedup(subs, ledger, cfg)
        for text in dropped:
            log.info("  deduped (already worked): %s", text)

        if not subs:
            if wave == 1:
                # The planner could not decompose it. Ask the original question
                # as a single sub-question rather than ending a run that has not
                # searched for anything yet (loop_spec.md §10).
                log.info("  wave 1 produced no sub-questions; falling back to "
                         "direct retrieval on the original question")
                subs = [SubQuestion(text=question,
                                    rationale="direct retrieval fallback")]
            else:
                verdict = "stalled"
                log.info("  nothing left to ask; stopping")
                break

        for sub in subs:
            log.info("  sub-question (p%d): %s", sub.priority, sub.text)

        results = await _dispatch(subs, ledger, cfg, budget)
        # The sub-questions are now spent: record their vectors so a later
        # wave's planner cannot hand the same work out again under new wording.
        ledger.attempted_vectors.extend(s.vector for s in subs if s.vector)
        rejections = _merge(ledger, results)
        for note in rejections:
            log.warning("  merge: %s", note)

        for result in results:
            log.info("  [%s] %s (%d round(s)%s) — %s",
                     result.status, result.sub_question, result.iterations,
                     ", escalated" if result.escalated else "",
                     result.note or "no note")

        verdict, gaps, suff_note = await _sufficient(ledger, results, cfg, budget)
        log.info("wave %d: sufficient=%s (%s); calls %d/%d",
                 wave, verdict, suff_note or "no note",
                 budget.llm_calls, budget.max_llm_calls)
        for gap in gaps:
            log.info("  gap: %s", gap)
        _log_block(f"  ledger after wave {wave}:", ledger.render())

        entry = {
            "wave": wave,
            "plan_note": plan_note,
            "sub_questions": [s.text for s in subs],
            "deduped": dropped,
            "results": [
                {"sub_question": r.sub_question, "status": r.status,
                 "iterations": r.iterations, "escalated": r.escalated,
                 "facts": len(r.facts), "messages_seen": len(r.message_ids),
                 "calls": r.calls, "gap": r.gap, "note": r.note}
                for r in results
            ],
            "verdict": verdict,
            "gaps": gaps,
            "facts_total": len(ledger.facts),
            "llm_calls": budget.llm_calls,
            "tokens": budget.tokens,
        }
        trajectory.append(entry)
        _snapshot(cfg, run_id, wave, {**entry, "ledger": ledger.render()})

        if verdict in ("yes", "unanswerable"):
            break
        if budget.exhausted:
            verdict = "budget_exhausted"
            break
    else:
        # Wave budget ran out with the checker still wanting more.
        if verdict == "no":
            verdict = "budget_exhausted"

    seconds = time.monotonic() - started
    log.info("investigation finished: verdict=%s after %d wave(s), %d llm calls, "
             "%d tokens, %.1fs", verdict, wave, budget.llm_calls, budget.tokens, seconds)
    _log_block("final ledger:", ledger.render())

    return WaveRun(
        question=question,
        ledger=ledger,
        verdict=verdict,
        waves=wave,
        llm_calls=budget.llm_calls,
        tokens=budget.tokens,
        seconds=seconds,
        trajectory=trajectory,
    )


# ---------------------------------------------------------------------------
# Synthesis
# ---------------------------------------------------------------------------

async def synthesize(run: WaveRun) -> str:
    """Write the final answer from the ledger alone (loop_spec.md §2.5).

    Not from the retrieved rows: they were judged once, by the worker that
    fetched them, and what survived that judgment is in the ledger with its
    citations. Anything else needed is re-fetchable by id.
    """
    message = (
        f"QUESTION: {run.question}\n\n"
        f"INVESTIGATION OUTCOME: {run.verdict} "
        f"(after {run.waves} wave(s), {run.llm_calls} model calls)\n\n"
        f"LEDGER:\n{run.ledger.render()}"
    )
    text = (await ask_gemini(SYNTH_PROMPT, message, model=LARGE_MODEL,
                             web_search=False)).strip()
    # Discord hard limit is 2000; the prompt asks for <1900, this is the backstop.
    return text[:1990] + "…" if len(text) > 1990 else text


async def answer(question: str, *, cfg: Optional[WaveConfig] = None) -> str:
    """Full pipeline: investigate -> synthesize. The bot.py seam."""
    run = await investigate(question, cfg=cfg)
    answer_text = await synthesize(run)
    log.info("answer (%d chars): %s", len(answer_text), answer_text)
    return answer_text
