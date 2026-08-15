"""Core agent loop — LLM-directed retrieval over the message store (specs.md Part 1).

Shape: budgeted while loop (plan+judge per pass) -> synthesize.
One model, role-scoped prompts, native function calling with manual dispatch.
State lives in an append-only ledger; raw rows live for exactly one pass.

Entry point for bot.py:  answer(question) -> str   (Discord-ready, <2000 chars)
"""

import asyncio
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from google.genai import types

import db
from llm import ask_gemini, client
from prompts import PLANNER_PROMPT, SYNTH_PROMPT

log = logging.getLogger(__name__)

MODEL = "gemini-3.5-flash"
EMBED_MODEL = "gemini-embedding-001"

# Hard budget — enforced in code, not by the prompt (specs.md decision 2).
MAX_PASSES = 16
MAX_RETRIEVALS = 40

# Rows are context we pay for on every subsequent token — keep result sets small.
MAX_ROWS_PER_CALL = 30
MAX_CONTENT_CHARS = 300


# ---------------------------------------------------------------------------
# Tool surface — flat-parameter declarations over db.py's instruments
# ---------------------------------------------------------------------------

def _s(type_: str, desc: str) -> types.Schema:
    return types.Schema(type=type_, description=desc)


def _obj(props: dict[str, types.Schema], required: list[str] | None = None) -> types.Schema:
    return types.Schema(type="OBJECT", properties=props, required=required or [])



# Shared filter params; the schema (not prose) is what teaches the model
# which columns are filterable.
_FILTER_PROPS: dict[str, types.Schema] = {
    "author_id": _s("INTEGER", "Only messages sent by this Discord user id"),
    "channel_id": _s("INTEGER", "Only messages in this channel id"),
    "day_of_week": _s("INTEGER", "0=Monday .. 6=Sunday"),
    "hour_of_day": _s("INTEGER", "Hour of day, 0..23 (UTC)"),
    "after": _s("STRING", "Only messages at/after this ISO datetime, e.g. 2026-03-01T00:00:00"),
    "before": _s("STRING", "Only messages before this ISO datetime (exclusive)"),
}
_FILTER_KEYS = set(_FILTER_PROPS)

_LIMIT = _s("INTEGER", f"Max rows to return (default 20, capped at {MAX_ROWS_PER_CALL})")

TOOLS = types.Tool(function_declarations=[
    types.FunctionDeclaration(
        name="structured_search",
        description=(
            "Fetch messages by metadata alone — no text matching. The cheapest, "
            "most reliable instrument; prefer it and use its filters to narrow "
            "every other search."
        ),
        parameters=_obj({
            **_FILTER_PROPS,
            "order_by": _s("STRING", "One of: 'created_at DESC' (default), "
                                     "'created_at ASC', 'author_id', "
                                     "'id ASC', 'id DESC'"),
            "limit": _LIMIT,
        }),
    ),
    types.FunctionDeclaration(
        name="keyword_search",
        description=(
            "Case-insensitive substring match on message text. Use for names, "
            "places, and exact terms — embeddings are weak on these."
        ),
        parameters=_obj({"term": _s("STRING", "The substring to find"),
                         **_FILTER_PROPS, "limit": _LIMIT},
                        required=["term"]),
    ),
    types.FunctionDeclaration(
        name="similarity_search",
        description=(
            "Semantic search: messages whose meaning is close to the query text. "
            "Last resort, for when vocabulary won't match exactly. Only searches "
            "messages that have been embedded; may return nothing on a fresh corpus."
        ),
        parameters=_obj({"query": _s("STRING", "Text expressing the meaning to find"),
                         **_FILTER_PROPS, "limit": _LIMIT},
                        required=["query"]),
    ),
    types.FunctionDeclaration(
        name="replies_to",
        description="All direct replies to a given message (anchor-based, structural).",
        parameters=_obj({
            "discord_message_id": _s("INTEGER", "The message id replies point at"),
            "limit": _LIMIT,
        }, required=["discord_message_id"]),
    ),
    types.FunctionDeclaration(
        name="messages_near",
        description=(
            "Messages within a time window around a timestamp — reconstructs the "
            "conversation surrounding a message you already found."
        ),
        parameters=_obj({
            "anchor": _s("STRING", "ISO datetime at the center of the window"),
            "window_minutes": _s("INTEGER", "Half-width of the window (default 30)"),
            "channel_id": _s("INTEGER", "Optionally restrict to one channel"),
            "limit": _LIMIT,
        }, required=["anchor"]),
    ),
])


# ---------------------------------------------------------------------------
# Dispatch — execute one model-requested call against db.py
# ---------------------------------------------------------------------------

async def _embed_query(text: str) -> list[float]:
    resp = await client.aio.models.embed_content(
        model=EMBED_MODEL,
        contents=text,
        config=types.EmbedContentConfig(
            task_type="RETRIEVAL_QUERY",
            output_dimensionality=db.EMBED_DIM,
        ),
    )
    return list(resp.embeddings[0].values)


def _split_args(args: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    """Split flat tool args into db-style filters vs everything else."""
    filters: dict[str, Any] = {}
    rest: dict[str, Any] = {}
    for key, value in args.items():
        if value is None:
            continue
        if key in _FILTER_KEYS:
            filters[key] = value
        else:
            rest[key] = value
    for key in ("after", "before"):
        if key in filters:
            filters[key] = datetime.fromisoformat(str(filters[key]))
    return filters, rest


def _compact_row(row: dict[str, Any]) -> dict[str, Any]:
    """Strip a db row to what the model needs; rows are paid-for context."""
    out: dict[str, Any] = {}
    for key, value in row.items():
        if key in ("embedding", "id") or value is None:
            continue
        if isinstance(value, datetime):
            value = value.isoformat(timespec="minutes")
        elif key == "content" and isinstance(value, str) and len(value) > MAX_CONTENT_CHARS:
            value = value[:MAX_CONTENT_CHARS] + "…"
        elif isinstance(value, float):
            value = round(value, 4)
        out[key] = value
    return out


async def _execute_call(name: str, args: dict[str, Any]) -> list[dict[str, Any]]:
    filters, rest = _split_args(args)
    limit = max(1, min(int(rest.get("limit", 20)), MAX_ROWS_PER_CALL))

    if name == "structured_search":
        rows = await db.structured_search(
            filters=filters or None,
            order_by=str(rest.get("order_by", "created_at DESC")),
            limit=limit,
        )
    elif name == "keyword_search":
        rows = await db.keyword_search(str(rest["term"]), filters=filters or None, limit=limit)
    elif name == "similarity_search":
        vec = await _embed_query(str(rest["query"]))
        rows = await db.similarity_search(vec, filters=filters or None, limit=limit)
    elif name == "replies_to":
        rows = await db.replies_to(int(rest["discord_message_id"]), limit=limit)
    elif name == "messages_near":
        rows = await db.messages_near(
            datetime.fromisoformat(str(rest["anchor"])),
            window_minutes=int(rest.get("window_minutes", 30)),
            channel_id=int(filters["channel_id"]) if "channel_id" in filters else None,
            limit=limit,
        )
    else:
        raise ValueError(f"unknown tool: {name}")

    return [_compact_row(r) for r in rows]


def _render_results(executed: list[tuple[str, dict[str, Any], Any]]) -> str:
    """Serialize this pass's results for the next pass — then they are evicted."""
    blocks = []
    for name, args, result in executed:
        header = f"### {name}({json.dumps(args, default=str)})"
        if isinstance(result, Exception):
            # Errors go back to the model so it can re-phrase or switch tools.
            blocks.append(f"{header}\nERROR: {type(result).__name__}: {result}")
        else:
            blocks.append(
                f"{header} -> {len(result)} rows\n"
                + json.dumps(result, ensure_ascii=False, default=str)
            )
    return "\n\n".join(blocks)


# ---------------------------------------------------------------------------
# Ledger — append-only working state (specs.md decision 3)
# ---------------------------------------------------------------------------

@dataclass
class Ledger:
    question: str
    facts: list[dict[str, Any]] = field(default_factory=list)
    inferences: list[dict[str, Any]] = field(default_factory=list)
    open_questions: list[str] = field(default_factory=list)
    dead_branches: list[str] = field(default_factory=list)

    def apply(self, updates: dict[str, Any]) -> list[str]:
        """Apply one pass's ledger_updates. Returns rejection notes.

        Facts are append-only and validated: no citations dict, no fact.
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


@dataclass
class Investigation:
    question: str
    route: str
    ledger: Ledger
    verdict: str          # yes | unanswerable | stalled | budget_exhausted
    passes: int
    retrievals: int
    trajectory: list[dict[str, Any]] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Planner cache — the part of every request that never changes
# ---------------------------------------------------------------------------

# PLANNER_PROMPT and the tool schemas are identical on every pass of every
# investigation, and MAX_PASSES means one question can resend them 16 times.
# Gemini will hold them server-side and charge less for the repeats, so the
# cache is built once per process and reused by every investigation after.
#
# The ledger is deliberately not in here. It changes on every pass and dies
# with the investigation, so caching it would cost more than it saves; it
# stays inline in the per-pass context.
CACHE_TTL_SECONDS = 3600

_cache_name: str | None = None

# Two investigations can start at once (/ask is not serialized), and both would
# otherwise create their own cache. The second one waits and reuses the first.
_cache_lock = asyncio.Lock()


async def _planner_cache() -> str | None:
    """The cached prompt+tools, created on first use. None if unavailable.

    None is a normal outcome, not an error: a prompt under the model's minimum
    cacheable size is refused, and the API can decline for its own reasons.
    Callers fall back to sending the prompt inline, which costs more per pass
    and behaves identically otherwise.
    """
    global _cache_name

    if _cache_name is not None:
        return _cache_name

    async with _cache_lock:
        # Someone else may have created it while this call waited for the lock.
        if _cache_name is not None:
            return _cache_name

        try:
            cache = await client.aio.caches.create(
                model=MODEL,
                config=types.CreateCachedContentConfig(
                    system_instruction=PLANNER_PROMPT,
                    tools=[TOOLS],
                    ttl=f"{CACHE_TTL_SECONDS}s",
                    display_name="helper-bot planner",
                ),
            )
        except Exception as e:
            log.warning("planner cache unavailable (%s: %s); sending the prompt "
                        "inline instead", type(e).__name__, e)
            return None

        _cache_name = cache.name
        log.info("planner cache created: %s (ttl %ds)", _cache_name, CACHE_TTL_SECONDS)
        return _cache_name


def _forget_cache(name: str) -> None:
    """Drop a cache that stopped working, unless it has already been replaced."""
    global _cache_name
    if _cache_name == name:
        _cache_name = None


def _planner_config(cache_name: str | None) -> types.GenerateContentConfig:
    """Config for one planner pass.

    With a cache the prompt and tools live inside it and must not be repeated
    here — sending both is rejected.
    """
    if cache_name:
        return types.GenerateContentConfig(cached_content=cache_name)
    return types.GenerateContentConfig(
        system_instruction=PLANNER_PROMPT,
        tools=[TOOLS],
    )


async def _planner_pass(context: str, passes: int) -> Any:
    """One planner call, with the cache and without it as a fallback."""
    cache_name = await _planner_cache()
    try:
        return await client.aio.models.generate_content(
            model=MODEL,
            contents=context,
            config=_planner_config(cache_name),
        )
    except Exception as e:
        if cache_name is None:
            raise
        # The cache expires on its own schedule and can be deleted from under
        # us mid-investigation. Retry this pass inline rather than losing the
        # whole run; if that fails too, the error is real and propagates.
        log.warning("pass %d failed with the planner cache (%s: %s); retrying "
                    "without it", passes, type(e).__name__, e)
        _forget_cache(cache_name)
        return await client.aio.models.generate_content(
            model=MODEL,
            contents=context,
            config=_planner_config(None),
        )


# ---------------------------------------------------------------------------
# The loop
# ---------------------------------------------------------------------------

def _extract_json(text: str) -> dict[str, Any] | None:
    """First parseable JSON object anywhere in the text (fence-tolerant)."""
    decoder = json.JSONDecoder()
    for i, ch in enumerate(text):
        if ch == "{":
            try:
                obj, _ = decoder.raw_decode(text[i:])
            except ValueError:
                continue
            if isinstance(obj, dict):
                return obj
    return None


def _pass_context(
    question: str,
    ledger: Ledger,
    passes: int, max_passes: int,
    retrievals: int, max_retrievals: int,
    results_text: str,
    notes: list[str],
) -> str:
    budget = f"BUDGET: pass {passes}/{max_passes}; searches used {retrievals}/{max_retrievals}."
    if retrievals >= max_retrievals:
        budget += " Search budget exhausted — conclude with yes or unanswerable."
    parts = [
        f"QUESTION: {question}",
        budget,
        "LEDGER:\n" + ledger.render(),
        "RESULTS OF LAST PASS'S SEARCHES:\n" + results_text,
    ]
    if notes:
        parts.append("HARNESS NOTES:\n- " + "\n- ".join(notes))
    return "\n\n".join(parts)


def _log_block(title: str, body: str) -> None:
    """Log a multi-line body under one heading, indented so it stays readable
    as one unit in a stream of interleaved log lines."""
    log.info("%s\n%s", title, "\n".join(f"    {line}" for line in body.splitlines()))


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


async def investigate(
    question: str,
    *,
    route: str = "investigation",
    max_passes: int = MAX_PASSES,
    max_retrievals: int = MAX_RETRIEVALS,
) -> Investigation:
    """Run the budgeted plan/judge loop. Returns the full investigation record."""
    ledger = Ledger(question)
    trajectory: list[dict[str, Any]] = []
    results_text = "(none — this is the first pass; request your first searches)"
    notes: list[str] = []
    verdict = "no"
    passes = 0
    retrievals = 0

    log.info("investigating %r (route=%s, budget: %d passes / %d searches)",
             question, route, max_passes, max_retrievals)

    while passes < max_passes:
        passes += 1
        context = _pass_context(
            question, ledger, passes, max_passes, retrievals, max_retrievals,
            results_text, notes,
        )
        notes = []

        response = await _planner_pass(context, passes)
        text, calls = _response_parts(response)

        # Every planner reply, verbatim and before any parsing — a pass is only
        # reconstructable if what the model actually said is on the record,
        # parsed or not. Function calls arrive as structured parts, not text,
        # so a reply that is pure tool calls logs as empty here; they are
        # logged below under "requested".
        _log_block(f"  pass {passes} raw reply:",
                   text or "(no text — function calls only)")

        parsed = _extract_json(text) or {}
        if parsed:
            _log_block(f"  pass {passes} parsed JSON:",
                       json.dumps(parsed, ensure_ascii=False, indent=2, default=str))
        else:
            log.warning("  pass %d: no JSON object found in the reply", passes)

        verdict = parsed.get("sufficient")
        if verdict not in ("yes", "no", "unanswerable"):
            log.warning("  pass %d: no usable verdict (sufficient=%r)", passes, verdict)
            notes.append(
                "your previous reply had no parseable verdict JSON — respond with "
                "the required JSON object"
            )
            verdict = "no"

        notes.extend(ledger.apply(parsed.get("ledger_updates") or {}))

        # Clamp requested searches to the remaining budget.
        remaining = max_retrievals - retrievals
        if len(calls) > remaining:
            calls = calls[:remaining]
            notes.append("search budget nearly exhausted; extra requests were dropped")

        specs = [(c.name, dict(c.args or {})) for c in calls]
        trajectory.append({
            "pass": passes,
            "verdict": verdict,
            "notes": parsed.get("notes"),
            "calls": specs,
            "facts": len(ledger.facts),
            "inferences": len(ledger.inferences),
            "open_questions": len(ledger.open_questions),
            "harness_notes": list(notes),
        })
        log.info(
            "pass %d/%d verdict=%s searches=%d/%d facts=%d inferences=%d "
            "open_questions=%d dead_branches=%d",
            passes, max_passes, verdict, retrievals, max_retrievals,
            len(ledger.facts), len(ledger.inferences),
            len(ledger.open_questions), len(ledger.dead_branches),
        )
        if parsed.get("notes"):
            log.info("  planner notes: %s", parsed["notes"])
        for name, args in specs:
            log.info("  requested: %s(%s)", name, json.dumps(args, default=str))
        if notes:
            _log_block("  harness notes:", "\n".join(f"- {n}" for n in notes))
        # The ledger is the whole working state — log it in full, not just counts.
        _log_block(f"  ledger after pass {passes}:", ledger.render())

        if verdict != "no":
            break
        if not specs:
            # "no" with nothing left to run: out of budget, or the model stalled.
            verdict = "budget_exhausted" if retrievals >= max_retrievals else "stalled"
            break

        # Fork-join: execute this pass's searches concurrently (decision 5).
        results = await asyncio.gather(
            *(_execute_call(name, args) for name, args in specs),
            return_exceptions=True,
        )
        retrievals += len(specs)
        for (name, args), res in zip(specs, results):
            if isinstance(res, BaseException):
                log.warning("  %s(%s) failed: %s: %s",
                            name, json.dumps(args, default=str),
                            type(res).__name__, res)
            else:
                log.info("  %s -> %d rows", name, len(res))
        # Raw rows enter the next pass's context, then are evicted (decision 4).
        results_text = _render_results(
            [(name, args, res) for (name, args), res in zip(specs, results)]
        )
    else:
        # Pass budget ran out with the model still wanting more.
        if verdict == "no":
            verdict = "budget_exhausted"

    log.info("investigation finished: verdict=%s after %d passes, %d searches",
             verdict, passes, retrievals)
    _log_block("final ledger:", ledger.render())

    return Investigation(
        question=question,
        route=route,
        ledger=ledger,
        verdict=verdict,
        passes=passes,
        retrievals=retrievals,
        trajectory=trajectory,
    )


# ---------------------------------------------------------------------------
# Synthesis
# ---------------------------------------------------------------------------

async def synthesize(inv: Investigation) -> str:
    """Write the final answer from the ledger alone (specs.md: synth is confined)."""
    message = (
        f"QUESTION: {inv.question}\n\n"
        f"INVESTIGATION OUTCOME: {inv.verdict} "
        f"(after {inv.passes} passes, {inv.retrievals} searches)\n\n"
        f"LEDGER:\n{inv.ledger.render()}"
    )
    text = (await ask_gemini(SYNTH_PROMPT, message, model=MODEL, web_search=False)).strip()
    # Discord hard limit is 2000; the prompt asks for <1900, this is the backstop.
    return text[:1990] + "…" if len(text) > 1990 else text


async def answer(question: str) -> str:
    """Full pipeline: investigate -> synthesize. The bot.py seam."""
    inv = await investigate(question)
    answer_text = await synthesize(inv)
    log.info("answer (%d chars): %s", len(answer_text), answer_text)
    return answer_text
