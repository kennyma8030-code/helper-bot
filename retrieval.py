"""The retrieval instruments, as a tool surface the model can call.

Split out of loop.py so that file is the loop and this one is the kit it
reaches for. Nothing here decides anything: it declares the six instruments
db.py exposes, dispatches one model-requested call against them, and trims the
rows on the way back.

Instrument discipline lives in the tool descriptions on purpose — the schema is
what teaches the model which columns are filterable and which instrument is
expensive, and prose in a prompt cannot be validated the way a schema can.
"""

import json
import logging
from datetime import datetime
from typing import Any

from google.genai import types

import db
from llm import embed_texts

log = logging.getLogger(__name__)

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
    "min_id": _s("INTEGER", "Only messages with this discord message id or higher (inclusive)"),
    "max_id": _s("INTEGER", "Only messages with this discord message id or lower (inclusive)"),
}
_FILTER_KEYS = set(_FILTER_PROPS)

_LIMIT = _s("INTEGER", f"Max rows to return (default 20, capped at {MAX_ROWS_PER_CALL})")

TOOLS = types.Tool(function_declarations=[
    types.FunctionDeclaration(
        name="structured_search",
        description=(
            "Fetch messages by metadata alone — no text matching. The cheapest, "
            "most reliable instrument; prefer it and use its filters to narrow "
            "every other search. With min_id/max_id it is also how you read a "
            "span of conversation back out message by message."
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
            "places, and exact terms the chat used verbatim."
        ),
        parameters=_obj({"term": _s("STRING", "The substring to find"),
                         **_FILTER_PROPS, "limit": _LIMIT},
                        required=["term"]),
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
    types.FunctionDeclaration(
        name="activity_stats",
        description=(
            "Counts only — messages per author, channel, weekday, hour, day, or "
            "month. Answer 'who most', 'when', and 'how often' with this rather "
            "than paging through messages; it returns totals, never text."
        ),
        parameters=_obj({
            "group_by": _s("STRING", "One of: 'author_id' (default), 'channel_id', "
                                     "'day_of_week', 'hour_of_day', 'day', 'month'"),
            **_FILTER_PROPS,
            "limit": _LIMIT,
        }),
    ),
    types.FunctionDeclaration(
        name="similarity_search",
        description=(
            "Semantic search over conversation summaries — finds discussions by "
            "meaning when you do not know the words they used. The most "
            "expensive instrument: try structured_search and keyword_search "
            "first. Returns spans to read, not quotable text: a hit is a "
            "summary someone wrote about the conversation, never a quote from "
            "it. Read the real messages behind a hit with structured_search, "
            "passing min_id=first_message_id and max_id=last_message_id, and "
            "cite those."
        ),
        parameters=_obj({
            "query": _s("STRING", "What to look for, in natural language"),
            "channel_id": _s("INTEGER", "Optionally restrict to one channel"),
            "limit": _LIMIT,
        }, required=["query"]),
    ),
])

# Which tools hand back message rows, as opposed to something else. A cluster
# hit is a pointer and a count is a number; neither is evidence, and neither
# belongs in front of the grader.
MESSAGE_TOOLS = {"structured_search", "keyword_search", "replies_to", "messages_near"}


# ---------------------------------------------------------------------------
# Dispatch — execute one model-requested call against db.py
# ---------------------------------------------------------------------------

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
    # Discord ids are past 2^53, where JSON numbers stop being exact. If one
    # arrives as a float it has already lost its low bits and would name a
    # message that does not exist, so int() here is a normalizer, not a fix —
    # what it reliably rescues is the id that arrives as a string.
    for key in ("min_id", "max_id"):
        if key in filters:
            filters[key] = int(float(filters[key]) if isinstance(filters[key], float)
                               else filters[key])
    return filters, rest


def compact_row(row: dict[str, Any]) -> dict[str, Any]:
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


async def execute_call(name: str, args: dict[str, Any]) -> list[dict[str, Any]]:
    """Run one instrument. Raises on an unknown tool or a bad argument."""
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
    elif name == "replies_to":
        rows = await db.replies_to(int(rest["discord_message_id"]), limit=limit)
    elif name == "messages_near":
        rows = await db.messages_near(
            datetime.fromisoformat(str(rest["anchor"])),
            window_minutes=int(rest.get("window_minutes", 30)),
            channel_id=int(filters["channel_id"]) if "channel_id" in filters else None,
            limit=limit,
        )
    elif name == "activity_stats":
        rows = await db.activity_stats(
            group_by=str(rest.get("group_by", "author_id")),
            filters=filters or None,
            limit=limit,
        )
    elif name == "similarity_search":
        # is_query: the stored side embedded summaries as documents, and the two
        # sides are not interchangeable for this model.
        vectors = await embed_texts([str(rest["query"])], is_query=True)
        rows = await db.similarity_search(
            vectors[0],
            channel_id=int(filters["channel_id"]) if "channel_id" in filters else None,
            limit=limit,
        )
    else:
        raise ValueError(f"unknown tool: {name}")

    return [compact_row(r) for r in rows]


def render_results(executed: list[tuple[str, dict[str, Any], Any]]) -> str:
    """Serialize one round of results for the model that asked for them."""
    blocks = []
    for name, args, result in executed:
        header = f"### {name}({json.dumps(args, default=str)})"
        if isinstance(result, BaseException):
            # Errors go back to the model so it can re-phrase or switch tools.
            blocks.append(f"{header}\nERROR: {type(result).__name__}: {result}")
        else:
            blocks.append(
                f"{header} -> {len(result)} rows\n"
                + json.dumps(result, ensure_ascii=False, default=str)
            )
    return "\n\n".join(blocks)
