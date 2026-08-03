"""Daily summarization — the retrieval index over the message store.

One summary per (calendar day, channel), written by the LLM to be searched
rather than read (specs-summaries.md). Messages remain the ground truth; these
rows only point at days worth opening.

Watermark-driven, not scheduled: `summary_state.summarized_through` advances as
days are written, so the job is safe to call repeatedly and repairs its own gaps
after downtime instead of leaving a hole. Entry points for bot.py:

    run_once()                  -> summarize every channel up to yesterday
    summarize_day(cid, day)     -> one day, for backfills and re-runs
"""

import json
import logging
from collections.abc import Iterable
from datetime import date, datetime, time, timedelta
from typing import Any, Optional

import db
# CORPUS_TZ decides which midnight ends a day. Imported rather than defined
# here on purpose: it is the same constant hour_of_day and day_of_week are
# derived from, and two timezone constants would eventually disagree — a day
# summary cut on a different clock than the hour filters is a silent wrong
# answer nobody would think to look for.
from db import CORPUS_TZ
from llm import ask_gemini
from loop import MODEL, _extract_json
from prompts import SUMMARY_PROMPT, SUMMARY_PROMPT_VERSION

log = logging.getLogger(__name__)

# How much prior context the summarizer sees. Enough to resolve "the trip" and
# "he" and to carry threads forward; small enough to stay cheap. Eventually
# this becomes a maintained wiki — for now it is only the previous summaries.
CONTEXT_DAYS = 7

# A first run on a long history would otherwise fire one model call per day of
# the entire corpus at once. It stops here and picks the rest up next run.
MAX_DAYS_PER_RUN = 45

# Blast radius on one day's input. A day this size is not a normal day.
MAX_MESSAGES_PER_DAY = 3000


# ---------------------------------------------------------------------------
# Day boundaries
# ---------------------------------------------------------------------------

def _day_bounds(day: date) -> tuple[datetime, datetime]:
    """The UTC half-open span [start, end) covering one local calendar day."""
    start = datetime.combine(day, time.min, tzinfo=CORPUS_TZ)
    end = datetime.combine(day + timedelta(days=1), time.min, tzinfo=CORPUS_TZ)
    return start, end


def _today() -> date:
    """Today in the corpus timezone — the day that is not over yet."""
    return datetime.now(CORPUS_TZ).date()


# ---------------------------------------------------------------------------
# Rendering the model's input
# ---------------------------------------------------------------------------

def _render_messages(rows: list[dict[str, Any]]) -> str:
    """One line per message: the ids are what the summary's anchors point at."""
    lines = []
    for row in rows:
        stamp = row["created_at"].astimezone(CORPUS_TZ).strftime("%H:%M")
        line = f"[{row['discord_message_id']}] {stamp} {row['author_id']}: {row['content']}"
        if row.get("reply_to_message_id"):
            line += f"  (reply to {row['reply_to_message_id']})"
        lines.append(line)
    return "\n".join(lines)


def _render_context(summaries: list[dict[str, Any]]) -> str:
    """Previous days, oldest first. Facets go in alongside the prose: open
    threads are the part the next day is expected to pick up."""
    if not summaries:
        return "(none — this is the first summarized day for this channel)"

    blocks = []
    for s in summaries:
        facets = s.get("facets") or {}
        block = [f"### {s['summary_date']} ({s['message_count']} messages)", s["prose"]]
        for key in ("open_threads", "entities", "topics"):
            values = facets.get(key)
            if values:
                block.append(f"{key}: {json.dumps(values, ensure_ascii=False)}")
        blocks.append("\n".join(block))
    return "\n\n".join(blocks)


# ---------------------------------------------------------------------------
# Summarizing one day
# ---------------------------------------------------------------------------

async def summarize_day(channel_id: int, day: date) -> Optional[int]:
    """Summarize one channel-day and store it. Returns the row id, or None if
    the day had no messages to summarize."""
    start, end = _day_bounds(day)
    rows = await db.messages_for_span(channel_id, start, end, limit=MAX_MESSAGES_PER_DAY)

    if not rows:
        log.info("summarize %s %s: no messages", channel_id, day)
        return None
    if len(rows) == MAX_MESSAGES_PER_DAY:
        log.warning(
            "summarize %s %s: hit the %d message cap — this summary covers only "
            "part of the day", channel_id, day, MAX_MESSAGES_PER_DAY,
        )

    context = await db.recent_day_summaries(channel_id, before=day, days=CONTEXT_DAYS)

    message = (
        f"DAY: {day} (channel {channel_id}, {len(rows)} messages)\n\n"
        f"RECENT CONTEXT — your summaries of the previous days, oldest first:\n"
        f"{_render_context(context)}\n\n"
        f"THE DAY — every message, oldest first:\n{_render_messages(rows)}"
    )

    raw = await ask_gemini(SUMMARY_PROMPT, message, model=MODEL, web_search=False)
    parsed = _extract_json(raw)
    if not parsed or not parsed.get("prose"):
        # No prose means no summary. Better to leave the day unwritten and
        # retry it than to store an empty index entry that hides the day.
        raise ValueError(f"summarizer returned no usable JSON for {day}")

    facets = parsed.get("facets")
    if not isinstance(facets, dict):
        facets = {}

    summary_id = await db.upsert_day_summary(
        summary_date=day,
        channel_id=channel_id,
        started_at=rows[0]["created_at"],
        ended_at=rows[-1]["created_at"],
        prose=str(parsed["prose"]),
        facets=facets,
        first_message_id=rows[0]["discord_message_id"],
        last_message_id=rows[-1]["discord_message_id"],
        message_count=len(rows),
        model=MODEL,
        prompt_version=SUMMARY_PROMPT_VERSION,
    )
    log.info("summarized %s %s: %d messages -> summary %d",
             channel_id, day, len(rows), summary_id)
    return summary_id


# ---------------------------------------------------------------------------
# Catching up — the watermark loop
# ---------------------------------------------------------------------------

async def catch_up_channel(channel_id: int) -> int:
    """Summarize every complete day this channel is behind on. Returns the
    number of days written.

    Today is never summarized: it is not over, and a summary written at noon
    would be upserted away by the next run anyway. Days are done oldest first
    so each one has the previous day's summary as context.
    """
    watermark = await db.summary_watermark(channel_id)

    if watermark is None:
        first = await db.first_message_at(channel_id)
        if first is None:
            return 0
        start_day = first.astimezone(CORPUS_TZ).date()
    else:
        start_day = watermark + timedelta(days=1)

    last_day = _today() - timedelta(days=1)
    if start_day > last_day:
        return 0

    written = 0
    day = start_day
    while day <= last_day and written < MAX_DAYS_PER_RUN:
        try:
            await summarize_day(channel_id, day)
        except Exception:
            # Stop at the failure rather than skipping past it: the watermark
            # must never advance over a day that was not summarized, or the
            # gap becomes permanent.
            log.exception("summarize %s %s failed; leaving the watermark at %s",
                          channel_id, day, day - timedelta(days=1))
            break

        # Advance per day, not per run. An interruption then costs one day.
        await db.set_summary_watermark(channel_id, day)
        written += 1
        day += timedelta(days=1)

    if written == MAX_DAYS_PER_RUN and day <= last_day:
        log.info("channel %s: stopped at %d days, %s..%s still pending",
                 channel_id, MAX_DAYS_PER_RUN, day, last_day)
    return written


async def summarize_days(channel_id: int, days: Iterable[date]) -> int:
    """Summarize an explicit set of days. Returns how many were written.

    Oldest first, so each day still has the days before it as context. Today is
    skipped — it is not over, and tomorrow's run writes it.

    Unlike the watermark loop, a failure here does not stop the rest: these are
    named days, not a sequence, so skipping one leaves no silent gap behind a
    watermark that has moved past it.
    """
    today = _today()
    ordered = sorted(d for d in set(days) if d < today)

    written = 0
    for day in ordered[:MAX_DAYS_PER_RUN]:
        try:
            if await summarize_day(channel_id, day) is not None:
                written += 1
        except Exception:
            log.exception("summarize %s %s failed", channel_id, day)

    if len(ordered) > MAX_DAYS_PER_RUN:
        log.warning(
            "channel %s: %d backfilled days, only the oldest %d summarized this "
            "run — re-run to continue", channel_id, len(ordered), MAX_DAYS_PER_RUN,
        )
    return written


async def summarize_backfilled(channel_id: int, days: Iterable[date]) -> int:
    """Summarize the days a backfill just added messages to.

    Split on the watermark, because the two halves need opposite handling:

    - Days at or below it have already been summarized and marked done, so the
      watermark loop will never look at them again. Backfilled messages there
      would be invisible forever. They are re-summarized directly; the upsert
      replaces the stale row, and set_summary_watermark's GREATEST means
      nothing can rewind.
    - Days above it belong to the watermark loop, which will summarize them in
      order and advance through them. Doing them here as well would just pay
      for every one of those days twice.
    """
    watermark = await db.summary_watermark(channel_id)
    if watermark is None:
        # Nothing has ever been summarized here, so there is no "already done"
        # half — the watermark loop covers the whole channel from its start.
        return await catch_up_channel(channel_id)

    stale = {d for d in days if d <= watermark}
    written = await summarize_days(channel_id, stale)
    return written + await catch_up_channel(channel_id)


async def run_once() -> int:
    """Catch every known channel up. Safe to call repeatedly; this is what the
    daily timer and the startup run both call."""
    channel_ids = await db.known_channel_ids()
    total = 0
    for channel_id in channel_ids:
        try:
            total += await catch_up_channel(channel_id)
        except Exception:
            # One channel's failure must not strand the others.
            log.exception("catch_up failed for channel %s", channel_id)
    log.info("summarization run done: %d day(s) written across %d channel(s)",
             total, len(channel_ids))
    return total
