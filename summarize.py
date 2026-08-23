"""Daily summarization — the retrieval index over the message store.

One summary per (calendar day, channel), written by the LLM to be searched
rather than read (specs-summaries.md). Messages remain the ground truth; these
rows only point at days worth opening.

The same model call also cuts the day into topical CLUSTERS — contiguous
stretches split where the topic changed, each with its own summary. Cluster
summaries are the corpus's semantic index: a separate, resumable pass embeds
every cluster whose vector is still NULL. Because a cluster cut at midnight
was cut blind to how the conversation continued, each day's run re-cuts the
previous day's final cluster together with the new day's messages.

Watermark-driven, not scheduled: `summary_state.summarized_through` advances as
days are attempted. A failed day is flagged in `summary_failures` before the
watermark moves past it, and every run retries flagged days before starting new
ones — a failure costs retries, never a silent permanent gap. Entry points for
bot.py:

    run_once()                  -> summarize + embed every channel up to yesterday
    summarize_day(cid, day)     -> one day, for backfills and re-runs
"""

import json
import logging
from collections.abc import Iterable
from datetime import date, datetime, time, timedelta
from typing import Any, Optional

import db
import llm
# CORPUS_TZ decides which midnight ends a day. Imported rather than defined
# here on purpose: it is the same constant hour_of_day and day_of_week are
# derived from, and two timezone constants would eventually disagree — a day
# summary cut on a different clock than the hour filters is a silent wrong
# answer nobody would think to look for.
from db import CORPUS_TZ
from llm import ask_gemini
from llm import LARGE_MODEL as MODEL, extract_json as _extract_json
from prompts import SUMMARY_PROMPT, SUMMARY_PROMPT_VERSION

log = logging.getLogger(__name__)

# How much prior context the summarizer sees. Enough to resolve "the trip" and
# "he" and to carry threads forward; small enough to stay cheap. Eventually
# this becomes a maintained wiki — for now it is only the previous summaries.
CONTEXT_DAYS = 7

# A first run on a long history would otherwise fire one model call per day of
# the entire corpus at once. It stops here and picks the rest up next run.
# Counts attempts, not successes — failed days cost a model call too.
MAX_DAYS_PER_RUN = 45

# Blast radius on one day's input. A day this size is not a normal day.
MAX_MESSAGES_PER_DAY = 3000

# Clusters embedded per API call in the embed pass.
EMBED_BATCH = 16


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
# Clusters — validation and the input window
# ---------------------------------------------------------------------------

class _ClusterError(ValueError):
    """The model's clusters were unusable. The prose summary may already be
    stored by the time this is raised — the day is flagged stage='clusters'
    and the whole call redone on retry (the upsert makes that harmless)."""


def _validate_clusters(raw: Any, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Check the model's clusters partition `rows` exactly, and resolve them.

    The contract the prompt states, enforced: contiguous, in order, no gaps,
    no overlaps, ids copied from the input. Timestamps and counts are derived
    from the rows here — the model is never trusted with them. Any violation
    raises _ClusterError; a half-checked cluster set must never be stored.
    """
    if not isinstance(raw, list) or not raw:
        raise _ClusterError("no clusters array in the response")

    position = {row["discord_message_id"]: i for i, row in enumerate(rows)}
    clusters: list[dict[str, Any]] = []
    expected = 0

    for k, c in enumerate(raw):
        try:
            first = int(c["first_id"])
            last = int(c["last_id"])
        except (KeyError, TypeError, ValueError):
            raise _ClusterError(f"cluster {k} is missing usable first_id/last_id")
        summary = str(c.get("summary") or "").strip()
        topic = str(c.get("topic") or "").strip() or None

        i, j = position.get(first), position.get(last)
        if i is None or j is None:
            raise _ClusterError(f"cluster {k} cites ids not in the input: {first}..{last}")
        if i != expected:
            raise _ClusterError(
                f"cluster {k} starts at message position {i}, expected {expected} "
                f"(gap or overlap)"
            )
        if j < i:
            raise _ClusterError(f"cluster {k} ends before it starts")
        if not summary:
            raise _ClusterError(f"cluster {k} has no summary")

        clusters.append({
            "first_message_id": first,
            "last_message_id": last,
            "started_at": rows[i]["created_at"],
            "ended_at": rows[j]["created_at"],
            "message_count": j - i + 1,
            "topic": topic,
            "summary": summary,
        })
        expected = j + 1

    if expected != len(rows):
        raise _ClusterError(f"clusters cover {expected} of {len(rows)} messages")
    return clusters


async def _cluster_window_start(
    channel_id: int, day: date, start: datetime, end: datetime
) -> datetime:
    """Where this day's clustering input begins (its end is always `end`).

    Usually the start of the day. Two cases reach further back:

    - The normal daily run re-cuts the previous day's final cluster: that
      cluster was cut blind at midnight, so it is re-drawn with today's
      messages in view. Skipped when the previous day failed or had nothing —
      then there is no midnight cut to heal.
    - A re-run of an already-clustered day must replace overlapping clusters
      whole, never trim them, so the window extends to the earliest one.
    """
    overlapping = await db.clusters_overlapping(channel_id, start, end)
    if overlapping:
        for c in overlapping:
            if c["ended_at"] >= end:
                # A later day's run drew this cluster across our end boundary.
                # Replacing it leaves its tail unclustered until that later
                # day is itself re-run (backfills re-run days oldest first,
                # so normally it is).
                log.warning(
                    "re-cluster %s %s: replacing cluster %s..%s that extends past "
                    "the day", channel_id, day,
                    c["first_message_id"], c["last_message_id"],
                )
        return min([start] + [c["started_at"] for c in overlapping])

    prev_day = day - timedelta(days=1)
    if await db.summary_failed(channel_id, prev_day):
        return start
    last = await db.latest_cluster(channel_id)
    if last is None:
        return start
    prev_start, _ = _day_bounds(prev_day)
    # Carry over only when the corpus's last cluster actually ended on the
    # previous day; an older one means the previous day was quiet ("none").
    if prev_start <= last["ended_at"] < start:
        return last["started_at"]
    return start


# ---------------------------------------------------------------------------
# Summarizing one day
# ---------------------------------------------------------------------------

async def summarize_day(channel_id: int, day: date) -> Optional[int]:
    """Summarize + cluster one channel-day. Returns the summary row id, or
    None if the day had no messages.

    On failure the day is flagged in summary_failures (stage 'summary' or
    'clusters') before the exception re-raises, so callers may move on and the
    day is retried at the start of every later run. Success clears the flag.
    """
    try:
        summary_id = await _run_day(channel_id, day)
    except _ClusterError as e:
        await db.flag_summary_failure(channel_id, day, stage="clusters", error=str(e))
        raise
    except Exception as e:
        await db.flag_summary_failure(
            channel_id, day, stage="summary", error=f"{type(e).__name__}: {e}"
        )
        raise
    await db.clear_summary_failure(channel_id, day)
    return summary_id


async def _run_day(channel_id: int, day: date) -> Optional[int]:
    start, end = _day_bounds(day)
    window_start = await _cluster_window_start(channel_id, day, start, end)
    rows = await db.messages_for_span(
        channel_id, window_start, end, limit=MAX_MESSAGES_PER_DAY
    )

    # rows is ordered by time, so the carry-over is exactly the prefix that
    # precedes the day. The prose covers day_rows; clusters cover all of rows.
    day_rows = [r for r in rows if r["created_at"] >= start]
    carry_rows = rows[: len(rows) - len(day_rows)]

    if not day_rows:
        # Nothing to summarize, and nothing touched: the previous day's last
        # cluster stays as it is until a day with messages re-cuts it.
        log.info("summarize %s %s: no messages", channel_id, day)
        return None
    if len(rows) == MAX_MESSAGES_PER_DAY:
        log.warning(
            "summarize %s %s: hit the %d message cap — this summary covers only "
            "part of the day", channel_id, day, MAX_MESSAGES_PER_DAY,
        )

    context = await db.recent_day_summaries(channel_id, before=day, days=CONTEXT_DAYS)

    parts = [
        f"DAY: {day} (channel {channel_id}, {len(day_rows)} messages)",
        f"RECENT CONTEXT — your summaries of the previous days, oldest first:\n"
        f"{_render_context(context)}",
    ]
    if carry_rows:
        parts.append(
            f"CARRY-OVER — the final {len(carry_rows)} message(s) of the previous "
            f"day, for clustering only (not part of THE DAY):\n"
            f"{_render_messages(carry_rows)}"
        )
    parts.append(f"THE DAY — every message, oldest first:\n{_render_messages(day_rows)}")
    message = "\n\n".join(parts)

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
        started_at=day_rows[0]["created_at"],
        ended_at=day_rows[-1]["created_at"],
        prose=str(parsed["prose"]),
        facets=facets,
        first_message_id=day_rows[0]["discord_message_id"],
        last_message_id=day_rows[-1]["discord_message_id"],
        message_count=len(day_rows),
        model=MODEL,
        prompt_version=SUMMARY_PROMPT_VERSION,
    )

    # Clusters go in after the summary on purpose: if they fail validation the
    # prose is already stored, and the stage='clusters' flag redoes the day.
    clusters = _validate_clusters(parsed.get("clusters"), rows)
    for c in clusters:
        c["model"] = MODEL
        c["prompt_version"] = SUMMARY_PROMPT_VERSION
    deleted, inserted = await db.replace_clusters(channel_id, window_start, end, clusters)

    log.info(
        "summarized %s %s: %d messages (+%d carried over) -> summary %d, "
        "%d cluster(s) (replaced %d)",
        channel_id, day, len(day_rows), len(carry_rows), summary_id,
        inserted, deleted,
    )
    return summary_id


# ---------------------------------------------------------------------------
# Catching up — the watermark loop
# ---------------------------------------------------------------------------

async def catch_up_channel(channel_id: int) -> int:
    """Summarize every complete day this channel is behind on — flagged
    failures from earlier runs first, then new days from the watermark.
    Returns the number of days written.

    Today is never summarized: it is not over, and a summary written at noon
    would be upserted away by the next run anyway. Days are done oldest first
    so each one has the previous day's summary as context.

    The watermark advances past a day once it is either written or flagged as
    failed — a flagged day is retried at the top of every later run, so no
    day is silently lost. Only when the failure cannot even be flagged (the
    database itself is the problem) does the run stop with the watermark held
    back, since advancing would make that day's gap permanent.
    """
    written = 0
    attempts = 0
    today = _today()

    # Failed days from earlier runs — "the next pass attempts the failed pass
    # in addition to its own." Oldest first, so a repaired day is in context
    # for the days after it as soon as possible.
    for day in await db.failed_summary_days(channel_id):
        if attempts >= MAX_DAYS_PER_RUN:
            return written
        if day >= today:
            continue
        attempts += 1
        try:
            if await summarize_day(channel_id, day) is not None:
                written += 1
        except Exception:
            # summarize_day re-flagged it (attempts is counted in the row);
            # nothing else to do until the next run.
            log.exception("retry of flagged day %s %s failed", channel_id, day)

    watermark = await db.summary_watermark(channel_id)
    if watermark is None:
        first = await db.first_message_at(channel_id)
        if first is None:
            return written
        start_day = first.astimezone(CORPUS_TZ).date()
    else:
        start_day = watermark + timedelta(days=1)

    last_day = today - timedelta(days=1)
    day = start_day
    while day <= last_day and attempts < MAX_DAYS_PER_RUN:
        attempts += 1
        try:
            if await summarize_day(channel_id, day) is not None:
                written += 1
        except Exception:
            log.exception("summarize %s %s failed", channel_id, day)
            # Advance past the day only if its failure flag actually landed;
            # otherwise there would be no record left to retry from.
            flagged = False
            try:
                flagged = await db.summary_failed(channel_id, day)
            except Exception:
                pass
            if not flagged:
                log.error(
                    "could not flag %s %s as failed; leaving the watermark at %s",
                    channel_id, day, day - timedelta(days=1),
                )
                break

        # Advance per day, not per run. An interruption then costs one day.
        await db.set_summary_watermark(channel_id, day)
        day += timedelta(days=1)

    if attempts >= MAX_DAYS_PER_RUN and day <= last_day:
        log.info("channel %s: stopped at %d attempts, %s..%s still pending",
                 channel_id, MAX_DAYS_PER_RUN, day, last_day)
    return written


async def summarize_days(channel_id: int, days: Iterable[date]) -> int:
    """Summarize an explicit set of days. Returns how many were written.

    Oldest first, so each day still has the days before it as context. Today is
    skipped — it is not over, and tomorrow's run writes it.

    A failure here does not stop the rest: these are named days, not a
    sequence, and summarize_day flags each failure before re-raising, so the
    day is retried at the start of every later run.
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
    written += await catch_up_channel(channel_id)

    # New clusters were just written; give them vectors now rather than making
    # them wait for the next daily run.
    try:
        await embed_pending()
    except Exception:
        log.exception("embed pass after backfill failed; clusters stay pending")
    return written


# ---------------------------------------------------------------------------
# The embed pass — vectors for every cluster that lacks one
# ---------------------------------------------------------------------------

async def embed_pending() -> int:
    """Embed every cluster still missing its vector. Returns how many were
    embedded.

    Resumable by construction: the queue is `embedding IS NULL`, each batch is
    committed as it lands, and an API failure just stops the pass — the rows
    it left behind are already flagged by their NULL and picked up next run.
    """
    done = 0
    while True:
        batch = await db.unembedded_clusters(limit=EMBED_BATCH)
        if not batch:
            break
        try:
            vectors = await llm.embed_texts([c["summary"] for c in batch])
        except Exception:
            log.exception(
                "embedding a batch of %d failed; %d cluster(s) stay pending",
                len(batch), len(batch),
            )
            break
        for cluster, vector in zip(batch, vectors):
            await db.set_cluster_embedding(cluster["id"], vector, llm.EMBED_MODEL)
        done += len(batch)

    if done:
        log.info("embedded %d cluster(s)", done)
    return done


async def run_once() -> int:
    """Catch every known channel up, then embed whatever clusters are pending.
    Safe to call repeatedly; this is what the daily timer and the startup run
    both call."""
    channel_ids = await db.known_channel_ids()
    total = 0
    for channel_id in channel_ids:
        try:
            total += await catch_up_channel(channel_id)
        except Exception:
            # One channel's failure must not strand the others.
            log.exception("catch_up failed for channel %s", channel_id)

    try:
        embedded = await embed_pending()
    except Exception:
        embedded = 0
        log.exception("embed pass failed; pending clusters remain for the next run")

    log.info("summarization run done: %d day(s) written, %d cluster(s) embedded "
             "across %d channel(s)", total, embedded, len(channel_ids))
    return total
