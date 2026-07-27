"""Core agent loop — LLM-directed retrieval over the message store (specs.md Part 1).

Shape: triage -> budgeted while loop (plan+judge per pass) -> synthesize.
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
from prompts import PLANNER_PROMPT, SYNTH_PROMPT, TRIAGE_PROMPT

log = logging.getLogger(__name__)

MODEL = "gemini-3.5-flash"
EMBED_MODEL = "gemini-embedding-001"

# Hard budget — enforced in code, not by the prompt (specs.md decision 2).
MAX_PASSES = 16
MAX_RETRIEVALS = 40

# Lookup fast path: same loop, tiny budget (specs.md decision 6).
LOOKUP_MAX_PASSES = 3
LOOKUP_MAX_RETRIEVALS = 6

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

    while passes < max_passes:
        passes += 1
        context = _pass_context(
            question, ledger, passes, max_passes, retrievals, max_retrievals,
            results_text, notes,
        )
        notes = []

        response = await client.aio.models.generate_content(
            model=MODEL,
            contents=context,
            config=types.GenerateContentConfig(
                system_instruction=PLANNER_PROMPT,
                tools=[TOOLS],
            ),
        )
        text, calls = _response_parts(response)

        parsed = _extract_json(text) or {}
        verdict = parsed.get("sufficient")
        if verdict not in ("yes", "no", "unanswerable"):
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
        log.info("pass %d/%d verdict=%s calls=%s facts=%d",
                 passes, max_passes, verdict,
                 [s[0] for s in specs], len(ledger.facts))

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
        # Raw rows enter the next pass's context, then are evicted (decision 4).
        results_text = _render_results(
            [(name, args, res) for (name, args), res in zip(specs, results)]
        )
    else:
        # Pass budget ran out with the model still wanting more.
        if verdict == "no":
            verdict = "budget_exhausted"

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
# Triage and synthesis
# ---------------------------------------------------------------------------

async def triage(question: str) -> str:
    """Route a question: 'lookup' or 'investigation'. Defaults to investigation."""
    try:
        raw = await ask_gemini(TRIAGE_PROMPT, question, model=MODEL, web_search=False)
    except Exception:
        log.exception("triage call failed; defaulting to investigation")
        return "investigation"
    parsed = _extract_json(raw) or {}
    route = parsed.get("route")
    return route if route in ("lookup", "investigation") else "investigation"


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
    """Full pipeline: triage -> investigate -> synthesize. The bot.py seam."""
    route = await triage(question)
    if route == "lookup":
        inv = await investigate(
            question, route=route,
            max_passes=LOOKUP_MAX_PASSES, max_retrievals=LOOKUP_MAX_RETRIEVALS,
        )
    else:
        inv = await investigate(question, route=route)
    log.info("investigation done: route=%s verdict=%s passes=%d retrievals=%d",
             route, inv.verdict, inv.passes, inv.retrievals)
    return await synthesize(inv)
