"""PostgreSQL + pgvector storage layer for the Discord RAG bot.

This is scaffolding for the storage layer described in the project design doc.
The guiding idea from that doc shapes everything here:

  * ONE table holds the structured metadata (speaker, channel, timestamps and
    the buckets derived from them), so a single query can filter on any of it.
  * Retrieval is layered. Structured filters, keyword/exact match, anchor-based
    lookup, and SQL aggregation are first-class and often the *primary*
    mechanism. Each gets its own function below.

Embeddings live on CLUSTERS, not messages. The daily summarizer cuts each
day's messages into topically coherent stretches and writes a summary of each;
that summary text is what gets embedded, in a separate resumable pass driven
off `embedding IS NULL`. Messages themselves carry no vector — the per-message
embedding column was removed before it was ever populated.

There is no per-message classification either. The `category` / `sentiment` /
`target_person_id` columns were specced as ingestion-time labels but never
written by anything, and the design doc already flagged their noise on
joke-heavy sarcastic chat as an unmeasured risk. Empty nullable columns are
worse than absent ones: they make filters and rates that silently match
nothing look available. What they were meant to answer now goes through the
summarizer's facets and cluster summaries, which carry the message ids that
back them.
"""

import logging
import os
from dataclasses import dataclass
from datetime import date, datetime, timezone
from zoneinfo import ZoneInfo
from typing import Any, Optional, Sequence

from dotenv import load_dotenv
from pgvector import Vector
from pgvector.psycopg import register_vector_async
from psycopg import AsyncConnection
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb
from psycopg_pool import AsyncConnectionPool

load_dotenv()

log = logging.getLogger(__name__)

# Standard libpq connection string, e.g.
#   postgresql://user:pass@host:5432/ragbot
# Read but not required at import: bot.py imports this module unconditionally,
# so demanding the variable here would stop the whole bot from starting on a
# deploy that has no database. open_pool() is the one place that truly needs it.
DATABASE_URL = os.environ.get("DATABASE_URL")

# The group's own timezone — the one clock every derived time value is stated
# in. Timestamps stay UTC in the column; this is only for the human-facing
# buckets (hour_of_day, day_of_week) and for where the summarizer cuts a day.
#
# "America/New_York", not a fixed -5 EST: the group's clocks move with DST, so
# a fixed offset would put every derived hour off by one for two thirds of the
# year. Changing this invalidates every stored hour_of_day/day_of_week and
# every day summary, since the boundaries themselves move.
CORPUS_TZ = ZoneInfo(os.environ.get("CORPUS_TZ", "America/New_York"))

# Width of the cluster embedding vectors. Pinned to what llm.embed_texts asks
# gemini-embedding-2 for via output_dimensionality; 768 is one of that model's
# recommended widths. The vector column and its HNSW index are built to this
# width, so changing it means a migration. Changing the model means
# re-embedding every cluster even when the width happens to stay the same.
EMBED_DIM = int(os.environ.get("EMBED_DIM", "768"))

# Distance operator for the future cluster search path. Cosine, matching
# gemini-embedding-2 (llm.embed_texts normalises the vectors it stores).
DISTANCE_OP = "<=>"


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------
# day_of_week (0-6) and hour_of_day (0-23) are stored as plain columns computed
# at ingestion rather than generated columns: EXTRACT over timestamptz is not
# immutable (it depends on the session time zone), so it can't back a generated
# column. Computing them once at ingest keeps the filters index-backed.

SCHEMA_DDL = f"""
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS messages (
    id                  bigserial PRIMARY KEY,

    -- Provenance / structural fields (never null; the ground truth).
    discord_message_id  bigint      NOT NULL UNIQUE,
    channel_id          bigint      NOT NULL,
    author_id           bigint      NOT NULL,
    content             text        NOT NULL,
    created_at          timestamptz NOT NULL,

    -- Anchor-based retrieval: the message this one replied to, if any.
    reply_to_message_id bigint,

    -- Derived-at-ingest temporal buckets for pattern/aggregation queries.
    day_of_week         smallint    NOT NULL,   -- 0=Monday .. 6=Sunday
    hour_of_day         smallint    NOT NULL    -- 0..23, in CORPUS_TZ
);

-- Migrations. Both columns sets below were specced, indexed, and never written
-- by anything: embeddings moved to clusters before the embedding pipeline ran,
-- and per-message classification was dropped rather than built (see module
-- doc). They are NULL in every row of every database that has them, so these
-- drops lose no data. Idempotent, like the rest of this DDL.
DROP INDEX IF EXISTS messages_embedding_idx;
ALTER TABLE messages DROP COLUMN IF EXISTS embedding;

DROP INDEX IF EXISTS messages_category_idx;
DROP INDEX IF EXISTS messages_target_idx;
DROP INDEX IF EXISTS messages_author_cat_idx;
ALTER TABLE messages DROP COLUMN IF EXISTS category;
ALTER TABLE messages DROP COLUMN IF EXISTS sentiment;
ALTER TABLE messages DROP COLUMN IF EXISTS target_person_id;

-- The server a message came from. Added 2026-08-25 with multi-server support.
-- Nullable, and deliberately not the fence: channel_id already partitions the
-- corpus correctly, since discord ids are globally unique snowflakes, and rows
-- written before this column existed have no guild until the repair walk
-- backfills them. Making reads depend on it would have hidden the whole
-- existing corpus the moment it shipped. What it buys is grouping — every
-- channel of one server — plus per-server settings and spend later.
ALTER TABLE messages ADD COLUMN IF NOT EXISTS guild_id bigint;
CREATE INDEX IF NOT EXISTS messages_guild_idx         ON messages (guild_id);

-- Selective equality filters (per-person, per-channel) hit these B-tree
-- indexes directly.
CREATE INDEX IF NOT EXISTS messages_author_idx        ON messages (author_id);
CREATE INDEX IF NOT EXISTS messages_channel_idx       ON messages (channel_id);
CREATE INDEX IF NOT EXISTS messages_created_at_idx    ON messages (created_at);
CREATE INDEX IF NOT EXISTS messages_reply_to_idx      ON messages (reply_to_message_id);
CREATE INDEX IF NOT EXISTS messages_dow_hour_idx      ON messages (day_of_week, hour_of_day);

-- TODO: keyword_search() uses ILIKE for now. For real term/name matching add
-- a pg_trgm GIN index or a tsvector column; embeddings are weak on exact names.

-- Topical clusters: contiguous stretches of one channel's messages, cut by the
-- daily summarizer where it saw the topic change. The `summary` text is what
-- gets embedded — a separate pass fills `embedding` wherever it is NULL, so
-- generation and embedding can fail and retry independently.
--
-- A cluster may span midnight: each day's run re-cuts the previous day's final
-- cluster together with the new day's messages, since a cluster cut at a day
-- boundary was cut blind to how the conversation continued.
CREATE TABLE IF NOT EXISTS clusters (
    id                  bigserial   PRIMARY KEY,
    channel_id          bigint      NOT NULL,

    -- Drill-down range, like day_summaries: discord ids, inclusive both ends.
    first_message_id    bigint      NOT NULL,
    last_message_id     bigint      NOT NULL,
    started_at          timestamptz NOT NULL,   -- created_at of first message
    ended_at            timestamptz NOT NULL,   -- created_at of last message
    message_count       integer     NOT NULL,

    topic               text,                   -- few-word label, for humans
    summary             text        NOT NULL,   -- the text that gets embedded

    embedding           vector({EMBED_DIM}),    -- NULL until the embed pass
    embed_model         text,                   -- set when embedded
    embedded_at         timestamptz,

    model               text        NOT NULL,   -- what wrote the summary
    prompt_version      text        NOT NULL,
    generated_at        timestamptz NOT NULL DEFAULT now(),

    -- Clusters for one channel never overlap; re-runs replace by span, and
    -- this catches any bug that would insert two clusters starting together.
    UNIQUE (channel_id, first_message_id)
);

CREATE INDEX IF NOT EXISTS clusters_span_idx
    ON clusters (channel_id, started_at, ended_at);
-- The embed pass walks exactly this: everything not yet embedded, in order.
CREATE INDEX IF NOT EXISTS clusters_pending_idx
    ON clusters (id) WHERE embedding IS NULL;
-- Answers the ORDER BY in similarity_search(), the planner's semantic path.
CREATE INDEX IF NOT EXISTS clusters_embedding_idx
    ON clusters USING hnsw (embedding vector_cosine_ops);

-- LLM-written summaries: the retrieval index over the corpus (specs-summaries.md).
-- Derived and regenerable — messages are the ground truth, these are a lossy
-- pointer layer. Facts cite message ids, never summary ids.
--
-- One table holds both tiers: whole days, and sub-summaries of dense stretches
-- within a day. granularity + parent_id keep reads from special-casing tiers.
CREATE TABLE IF NOT EXISTS day_summaries (
    id                  bigserial PRIMARY KEY,

    summary_date        date        NOT NULL,
    channel_id          bigint      NOT NULL,

    -- 'day' = the whole date; 'session' = one dense stretch inside that date.
    granularity         text        NOT NULL DEFAULT 'day',
    -- Set on session rows only; the day summary they were split out of.
    parent_id           bigint      REFERENCES day_summaries(id) ON DELETE CASCADE,

    -- Actual span covered. Needed for sessions, where the date alone does not
    -- locate the summary; on day rows it is the first/last message timestamp.
    started_at          timestamptz NOT NULL,
    ended_at            timestamptz NOT NULL,

    prose               text        NOT NULL,   -- read once a span is chosen
    facets              jsonb       NOT NULL DEFAULT '{{}}'::jsonb,  -- searchable

    -- Drill-down range: any summary expands to exactly the rows behind it.
    first_message_id    bigint      NOT NULL,   -- discord_message_id
    last_message_id     bigint      NOT NULL,
    message_count       integer     NOT NULL,

    -- Regeneration bookkeeping: the prompt and model will both change.
    model               text        NOT NULL,
    prompt_version      text        NOT NULL,
    generated_at        timestamptz NOT NULL DEFAULT now()
);

-- Upsert keys, split by tier: one summary per (day, channel), but a day may
-- hold many sessions. Partial unique indexes let each tier have its own
-- ON CONFLICT target while sharing the table.
CREATE UNIQUE INDEX IF NOT EXISTS day_summaries_day_key
    ON day_summaries (summary_date, channel_id) WHERE granularity = 'day';
CREATE UNIQUE INDEX IF NOT EXISTS day_summaries_session_key
    ON day_summaries (parent_id, started_at)    WHERE granularity = 'session';

CREATE INDEX IF NOT EXISTS day_summaries_date_idx   ON day_summaries (channel_id, summary_date);
CREATE INDEX IF NOT EXISTS day_summaries_parent_idx ON day_summaries (parent_id);
-- Entity/topic lookup without reading every summary. jsonb_path_ops is the
-- smaller, faster operator class; it supports @> containment, which is the
-- only facet query we need.
CREATE INDEX IF NOT EXISTS day_summaries_facets_idx
    ON day_summaries USING gin (facets jsonb_path_ops);

-- Watermark for incremental summarization: how far each channel is caught up.
-- Lets the job run on startup/timer and be safe to call repeatedly, with no
-- cron service and no gap to repair after downtime.
CREATE TABLE IF NOT EXISTS summary_state (
    channel_id          bigint      PRIMARY KEY,
    summarized_through  date        NOT NULL,
    updated_at          timestamptz NOT NULL DEFAULT now()
);

-- Days whose summarize/cluster run failed. The watermark advances PAST a
-- failed day once it is flagged here (a flagged day is retryable, so the gap
-- is not permanent), and every run retries flagged days before doing new ones.
-- Embedding failures are not flagged here — an unembedded cluster is its own
-- flag (embedding IS NULL) and is retried by the embed pass automatically.
CREATE TABLE IF NOT EXISTS summary_failures (
    channel_id          bigint      NOT NULL,
    day                 date        NOT NULL,
    stage               text        NOT NULL,   -- 'summary' | 'clusters'
    error               text,
    attempts            integer     NOT NULL DEFAULT 1,
    last_failed_at      timestamptz NOT NULL DEFAULT now(),
    PRIMARY KEY (channel_id, day)
);

-- Durable feature switches. The bot keeps these in memory to read them for
-- free, but the container is rebuilt on every deploy, crash, and restart, so
-- the in-memory copy is a cache and this table is the truth.
CREATE TABLE IF NOT EXISTS settings (
    key                 text        PRIMARY KEY,
    value               boolean     NOT NULL,
    updated_at          timestamptz NOT NULL DEFAULT now()
);

-- Estimated Gemini spend, one row per model call. In the database rather than
-- in memory for two reasons: the containers are rebuilt on every deploy and
-- crash, and the RAG bot and the test bots are separate containers sharing one
-- API key — a per-process counter would see half the spend and reset to zero
-- on restart.
--
-- Estimated, not billed. Google exposes no spend endpoint; this is tokens
-- reported by the API multiplied by the published price. See llm.MODEL_PRICES.
CREATE TABLE IF NOT EXISTS api_usage (
    id                  bigserial   PRIMARY KEY,
    model               text        NOT NULL,
    input_tokens        bigint      NOT NULL,
    output_tokens       bigint      NOT NULL,
    cost_usd            numeric(14,8) NOT NULL,
    called_at           timestamptz NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS api_usage_called_at_idx ON api_usage (called_at);
"""


# ---------------------------------------------------------------------------
# Pool lifecycle
# ---------------------------------------------------------------------------

_pool: Optional[AsyncConnectionPool] = None


async def _configure(conn) -> None:
    """Per-connection setup: teach psycopg how to adapt pgvector types."""
    await register_vector_async(conn)


async def _ensure_vector_extension() -> None:
    """Create the vector extension before the pool exists.

    Ordering trap: _configure registers pgvector's types on every pooled
    connection, and that registration fails with "vector type not found in the
    database" if the extension is not there yet — which is exactly the state of
    a fresh database, since init_db() has not run. Every connection then fails
    to configure, the pool never fills, and it surfaces as a PoolTimeout that
    says nothing about vectors.

    So this runs first, on one plain connection with no configure hook. It is
    also where a Postgres image genuinely lacking pgvector reports itself, with
    an error naming the extension instead of a 30-second timeout.
    """
    async with await AsyncConnection.connect(DATABASE_URL, autocommit=True) as conn:
        await conn.execute("CREATE EXTENSION IF NOT EXISTS vector")


async def open_pool() -> AsyncConnectionPool:
    """Create the shared async connection pool. Call once at startup.

    A long-running bot must not connect per-query — connecting is an expensive
    handshake. The pool holds a set of reusable connections that callers borrow
    and return.
    """
    global _pool
    if _pool is not None:
        return _pool

    if not DATABASE_URL:
        raise RuntimeError(
            "DATABASE_URL is not set, so there is no message store to open. "
            "Set it in the environment and restart."
        )

    await _ensure_vector_extension()

    pool = AsyncConnectionPool(
        conninfo=DATABASE_URL,
        min_size=1,
        max_size=10,
        kwargs={"row_factory": dict_row},
        configure=_configure,
        open=False,
    )
    await pool.open()
    await pool.wait()
    _pool = pool
    log.info("db pool opened (max_size=%d, embed_dim=%d)", pool.max_size, EMBED_DIM)
    return _pool


async def close_pool() -> None:
    """Close the pool on shutdown."""
    global _pool
    if _pool is not None:
        await _pool.close()
        _pool = None
        log.info("db pool closed")


def _get_pool() -> AsyncConnectionPool:
    if _pool is None:
        raise RuntimeError("db pool not opened; call open_pool() at startup")
    return _pool


async def init_db() -> None:
    """Create the extension, table, and indexes if they do not exist."""
    pool = _get_pool()
    async with pool.connection() as conn:
        await conn.execute(SCHEMA_DDL)
    log.info("db schema ensured")


async def clear_channel(channel_id: int) -> dict[str, int]:
    """Delete everything stored for one channel. Returns the rows removed.

    Scoped to a single channel on purpose: this backs a per-channel reset, and
    the other channels in the corpus have nothing to do with it.

    `settings` is deliberately untouched. It holds the durable power/RAG
    switches, which are not channel-scoped — wiping them would silently turn
    the bot off, and the caller asking to empty a channel has not asked for
    that. The schema itself is left in place; this empties rows, it does not
    drop tables.

    All the deletes share one transaction, so a reset either lands whole or
    not at all — no run can leave summaries pointing at messages that are gone.
    """
    pool = _get_pool()
    async with pool.connection() as conn:
        async with conn.cursor() as cur:
            # Summaries first. Session rows reference day rows in this same
            # table via parent_id, and deleting the whole channel's worth in
            # one statement avoids depending on which tier goes first.
            await cur.execute(
                "DELETE FROM day_summaries WHERE channel_id = %s", (channel_id,)
            )
            summaries = cur.rowcount
            await cur.execute(
                "DELETE FROM clusters WHERE channel_id = %s", (channel_id,)
            )
            clusters = cur.rowcount
            await cur.execute(
                "DELETE FROM messages WHERE channel_id = %s", (channel_id,)
            )
            messages = cur.rowcount
            # The watermark has to go too, or the summarizer believes the
            # channel is caught up through a date whose messages no longer
            # exist and never re-reads it. Same for failure flags: a flagged
            # day would be retried against messages that are gone.
            await cur.execute(
                "DELETE FROM summary_state WHERE channel_id = %s", (channel_id,)
            )
            watermarks = cur.rowcount
            await cur.execute(
                "DELETE FROM summary_failures WHERE channel_id = %s", (channel_id,)
            )
            failures = cur.rowcount

    log.info(
        "cleared channel %d: %d messages, %d summaries, %d clusters, "
        "%d watermarks, %d failure flags",
        channel_id, messages, summaries, clusters, watermarks, failures,
    )
    return {
        "messages": messages,
        "summaries": summaries,
        "clusters": clusters,
        "watermarks": watermarks,
        "failures": failures,
    }


# ---------------------------------------------------------------------------
# API spend
# ---------------------------------------------------------------------------

async def record_usage(
    *,
    model: str,
    input_tokens: int,
    output_tokens: int,
    cost_usd: float,
) -> None:
    """Append one model call's estimated cost."""
    pool = _get_pool()
    async with pool.connection() as conn:
        await conn.execute(
            "INSERT INTO api_usage (model, input_tokens, output_tokens, cost_usd) "
            "VALUES (%s, %s, %s, %s)",
            (model, input_tokens, output_tokens, cost_usd),
        )


async def total_cost_usd(since: Optional[datetime] = None) -> float:
    """Estimated spend across every recorded call, in USD.

    `since` limits it to calls after a moment — for a monthly cap rather than
    a lifetime one. Default is everything the table holds.
    """
    pool = _get_pool()
    sql = "SELECT COALESCE(SUM(cost_usd), 0) AS total FROM api_usage"
    params: list[Any] = []
    if since is not None:
        sql += " WHERE called_at >= %s"
        params.append(_as_utc(since))

    async with pool.connection() as conn:
        async with conn.cursor() as cur:
            await cur.execute(sql, params)
            row = await cur.fetchone()

    # numeric comes back as Decimal; callers compare against a plain float.
    return float(row["total"]) if row else 0.0


# ---------------------------------------------------------------------------
# Ingestion
# ---------------------------------------------------------------------------

async def upsert_message(
    *,
    discord_message_id: int,
    channel_id: int,
    author_id: int,
    content: str,
    created_at: datetime,
    reply_to_message_id: Optional[int] = None,
    guild_id: Optional[int] = None,
) -> tuple[int, bool]:
    """Insert a message (or update it if already ingested).

    Returns (row id, inserted) — `inserted` being True only when the row is
    genuinely new. The repair walk re-reads messages it already has, and that
    flag is what stops it from re-summarizing (and re-embedding) days where
    nothing was actually missing. `xmax = 0` is the standard way to ask
    Postgres which half of an upsert happened: a row this statement inserted
    has no update transaction stamped on it.

    An edit to an already-stored message reads as not-inserted, so it does not
    trigger re-summarization. That matches the rest of the bot, which has no
    edit handling at all.

    day_of_week / hour_of_day are derived here from created_at, in CORPUS_TZ —
    they exist to answer "who posts late at night", and that question is about
    the group's clock, not the server's. The timestamp itself stays UTC.
    """
    created_at = _as_utc(created_at)
    local = created_at.astimezone(CORPUS_TZ)
    day_of_week = local.weekday()               # 0=Monday
    hour_of_day = local.hour

    # COALESCE on guild_id, not EXCLUDED: the repair walk re-upserts rows that
    # already exist, and a walk that could not resolve the guild would otherwise
    # blank one that had already been filled in. Backfill only ever adds.
    sql = """
        INSERT INTO messages (
            discord_message_id, channel_id, author_id, content, created_at,
            reply_to_message_id, day_of_week, hour_of_day, guild_id
        )
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
        ON CONFLICT (discord_message_id) DO UPDATE SET
            content             = EXCLUDED.content,
            reply_to_message_id = EXCLUDED.reply_to_message_id,
            guild_id            = COALESCE(EXCLUDED.guild_id, messages.guild_id)
        RETURNING id, (xmax = 0) AS inserted
    """
    params = (
        discord_message_id, channel_id, author_id, content, created_at,
        reply_to_message_id, day_of_week, hour_of_day, guild_id,
    )
    pool = _get_pool()
    async with pool.connection() as conn:
        cur = await conn.execute(sql, params)
        row = await cur.fetchone()
    return row["id"], row["inserted"]


async def known_channel_ids(guild_id: Optional[int] = None) -> list[int]:
    """Every channel the store already holds messages for.

    This is what the startup catch-up walks: a channel enters the corpus by
    being backfilled once, deliberately, and is kept current from then on.

    With `guild_id`, only that server's channels — which is one half of
    building a Scope. It reads what is stored rather than what Discord
    currently reports, so a channel the bot has lost access to still fences
    correctly instead of silently dropping out of scope.
    """
    sql = "SELECT DISTINCT channel_id FROM messages"
    params: list[Any] = []
    if guild_id is not None:
        sql += " WHERE guild_id = %s"
        params.append(guild_id)
    pool = _get_pool()
    async with pool.connection() as conn:
        cur = await conn.execute(sql, params)
        rows = await cur.fetchall()
    return [row["channel_id"] for row in rows]


async def last_message_id(channel_id: int) -> Optional[int]:
    """Highest stored discord_message_id for a channel, or None if empty.

    Discord ids are snowflakes — they increase with time — so this doubles as
    "how far this channel is caught up", and anything above it is new.
    """
    pool = _get_pool()
    async with pool.connection() as conn:
        cur = await conn.execute(
            "SELECT MAX(discord_message_id) AS last FROM messages WHERE channel_id = %s",
            (channel_id,),
        )
        row = await cur.fetchone()
    return row["last"] if row else None


# ---------------------------------------------------------------------------
# Settings — switches that have to outlive the process
# ---------------------------------------------------------------------------

async def get_switches() -> dict[str, bool]:
    """Every stored switch. A switch that has never been set is simply absent,
    so the caller keeps its own default rather than getting a false here."""
    pool = _get_pool()
    async with pool.connection() as conn:
        cur = await conn.execute("SELECT key, value FROM settings")
        rows = await cur.fetchall()
    return {row["key"]: row["value"] for row in rows}


async def set_switch(key: str, value: bool) -> None:
    """Persist one switch. Upsert, so no row has to be seeded first."""
    pool = _get_pool()
    async with pool.connection() as conn:
        await conn.execute(
            """
            INSERT INTO settings (key, value) VALUES (%s, %s)
            ON CONFLICT (key) DO UPDATE
               SET value = EXCLUDED.value, updated_at = now()
            """,
            (key, value),
        )


# ---------------------------------------------------------------------------
# Filter building (shared by every retrieval instrument)
# ---------------------------------------------------------------------------
# Filters are passed as a plain dict so the LLM planner can compile a retrieval
# into structured constraints without this layer knowing about prompts. Values
# always go through parameters, never string interpolation.

_EQ_FIELDS = {
    "author_id",
    "channel_id",
    "day_of_week",
    "hour_of_day",
    "reply_to_message_id",
}


@dataclass(frozen=True)
class Scope:
    """The channels one request is allowed to touch. Not a filter — a fence.

    Every filter on a retrieval instrument is optional and chosen by the model,
    which is fine while the bot sits in one server: the worst a missing
    channel_id costs is a wider search. With the bot in two servers it costs
    something else entirely — a question asked in one group answered out of
    another group's private history. A model omitting an argument is ordinary
    model behaviour, so this cannot be a rule in a prompt.

    So scope is set by the code from the request that arrived, and forced into
    every query underneath. The model may narrow inside it and has no way to
    widen past it.

    Fails closed: an empty channel set matches nothing rather than everything.
    A bug that loses the scope returns no rows, which is visible and harmless,
    instead of returning someone else's chat.
    """

    channel_ids: frozenset[int]
    guild_id: Optional[int] = None

    @classmethod
    def of(cls, channel_ids, guild_id: Optional[int] = None) -> "Scope":
        return cls(frozenset(int(c) for c in channel_ids), guild_id)

    def narrowed_to(self, channel_id: Optional[int]) -> "Scope":
        """This scope restricted to one channel, or unchanged if that channel
        is outside it. Narrowing is always allowed; widening never is."""
        if channel_id is None:
            return self
        cid = int(channel_id)
        return Scope(frozenset({cid}), self.guild_id) if cid in self.channel_ids else self

    def __bool__(self) -> bool:
        return bool(self.channel_ids)

    def describe(self) -> str:
        return (f"guild={self.guild_id or '-'} "
                f"channels={len(self.channel_ids)}")


def _scope_clause(scope: Scope) -> tuple[str, list[Any]]:
    """The fence, as SQL. Always emitted, even for an empty scope."""
    if not isinstance(scope, Scope):
        raise TypeError(
            "a Scope is required — every read is fenced to the channels the "
            "request came from (db.Scope)"
        )
    # = ANY(array) rather than IN (...): one parameter whatever the channel
    # count, so the SQL text is identical for every request and Postgres can
    # reuse the plan.
    return "channel_id = ANY(%s)", [list(scope.channel_ids)]


def _build_where(
    filters: Optional[dict[str, Any]], scope: Scope,
) -> tuple[str, list[Any]]:
    """Turn a filter dict into a WHERE fragment + ordered params, fenced to
    `scope`.

    Supported keys: the equality fields in _EQ_FIELDS, plus two range pairs —
    'after' / 'before' (created_at bounds) and 'min_id' / 'max_id'
    (discord_message_id bounds).

    The scope clause is not one of them and is never optional: it is emitted
    whether or not any filter is, and a channel_id filter can only narrow
    within it. Returning a bare WHERE fence for empty filters is the point —
    there is no code path here that produces an unfenced query.
    """
    if not isinstance(scope, Scope):
        # Checked before the scope is used for anything, so a caller that
        # forgot it gets a sentence naming the problem rather than an
        # AttributeError from three lines down.
        raise TypeError(
            "a Scope is required — every read is fenced to the channels the "
            "request came from (db.Scope)"
        )
    scope = scope.narrowed_to((filters or {}).get("channel_id"))
    scope_sql, params = _scope_clause(scope)
    clauses: list[str] = [scope_sql]

    if not filters:
        return "WHERE " + scope_sql, params

    # channel_id is spent: the fence above already carries it, narrowed.
    filters = {k: v for k, v in filters.items() if k != "channel_id"}

    for field in _EQ_FIELDS:
        if field in filters and filters[field] is not None:
            clauses.append(f"{field} = %s")
            params.append(filters[field])

    if filters.get("after") is not None:
        clauses.append("created_at >= %s")
        params.append(_as_utc(filters["after"]))
    if filters.get("before") is not None:
        clauses.append("created_at < %s")
        params.append(_as_utc(filters["before"]))

    # Id bounds, and both ends are inclusive — unlike the timestamp pair above,
    # whose `before` is exclusive so consecutive day windows can abut without
    # double-counting a message. The id range exists to read a cluster's span
    # back out, and clusters store first_message_id/last_message_id as the
    # first and last messages actually in the span, so excluding either end
    # would drop a real message. discord_message_id, not the bigserial `id`:
    # the discord id is what clusters, citations, and replies all point with,
    # and its UNIQUE constraint indexes these range scans already.
    if filters.get("min_id") is not None:
        clauses.append("discord_message_id >= %s")
        params.append(filters["min_id"])
    if filters.get("max_id") is not None:
        clauses.append("discord_message_id <= %s")
        params.append(filters["max_id"])

    # clauses always holds the fence, so there is no empty-WHERE branch.
    return "WHERE " + " AND ".join(clauses), params


# ---------------------------------------------------------------------------
# Retrieval instrument 1: structured filter (pure SQL, no vectors)
# ---------------------------------------------------------------------------

async def structured_search(
    *,
    scope: Scope,
    filters: Optional[dict[str, Any]] = None,
    order_by: str = "created_at DESC",
    limit: int = 100,
) -> list[dict[str, Any]]:
    """Pure-SQL retrieval by structured constraints — often the primary path.

    order_by is a fixed whitelist of columns to keep it injection-safe.
    """
    allowed = {
        "created_at ASC", "created_at DESC",
        "author_id", "id ASC", "id DESC",
    }
    if order_by not in allowed:
        raise ValueError(f"order_by must be one of {sorted(allowed)}")

    where, params = _build_where(filters, scope)
    sql = f"SELECT * FROM messages {where} ORDER BY {order_by} LIMIT %s"
    pool = _get_pool()
    async with pool.connection() as conn:
        cur = await conn.execute(sql, [*params, limit])
        return await cur.fetchall()


# ---------------------------------------------------------------------------
# Retrieval instrument 2: keyword / exact match
# ---------------------------------------------------------------------------

async def keyword_search(
    term: str,
    *,
    scope: Scope,
    filters: Optional[dict[str, Any]] = None,
    limit: int = 50,
) -> list[dict[str, Any]]:
    """Substring match for names/specific terms where embeddings are weak.

    TODO: ILIKE is a placeholder. Swap for pg_trgm or full-text search once the
    term-matching requirements are clearer (see schema note).
    """
    where, params = _build_where(filters, scope)
    sql = (f"SELECT * FROM messages {where} AND content ILIKE %s "
           f"ORDER BY created_at DESC LIMIT %s")
    pool = _get_pool()
    async with pool.connection() as conn:
        cur = await conn.execute(sql, [*params, f"%{term}%", limit])
        return await cur.fetchall()


# ---------------------------------------------------------------------------
# Retrieval instrument 3: anchor-based (structural, not semantic)
# ---------------------------------------------------------------------------

async def replies_to(
    discord_message_id: int, *, scope: Scope, limit: int = 100,
) -> list[dict[str, Any]]:
    """Messages that were direct replies to a given message, inside `scope`.

    Fenced like every other read: an anchor id is just a number, and one from
    another server would otherwise pull that server's replies back.
    """
    scope_sql, params = _scope_clause(scope)
    pool = _get_pool()
    async with pool.connection() as conn:
        cur = await conn.execute(
            f"SELECT * FROM messages WHERE {scope_sql} AND reply_to_message_id = %s "
            f"ORDER BY created_at ASC LIMIT %s",
            (*params, discord_message_id, limit),
        )
        return await cur.fetchall()


async def messages_near(
    anchor: datetime,
    *,
    scope: Scope,
    window_minutes: int = 30,
    channel_id: Optional[int] = None,
    limit: int = 100,
) -> list[dict[str, Any]]:
    """Messages within a time window around an anchor timestamp.

    Timestamp proximity as a structural retrieval — useful for reconstructing
    the conversation surrounding a hit without any semantic assumption.
    """
    anchor = _as_utc(anchor)
    scope_sql, scope_params = _scope_clause(scope.narrowed_to(channel_id))
    clauses = [scope_sql,
               "created_at BETWEEN %s - make_interval(mins => %s) "
               "AND %s + make_interval(mins => %s)"]
    params: list[Any] = [*scope_params, anchor, window_minutes, anchor, window_minutes]
    sql = f"SELECT * FROM messages WHERE {' AND '.join(clauses)} ORDER BY created_at ASC LIMIT %s"
    params.append(limit)
    pool = _get_pool()
    async with pool.connection() as conn:
        cur = await conn.execute(sql, params)
        return await cur.fetchall()


# ---------------------------------------------------------------------------
# Retrieval instrument 4: aggregation (computed in SQL, never eyeballed)
# ---------------------------------------------------------------------------
# The doc is emphatic: pattern claims need denominators. "C mentions hiking
# less" is meaningless without C's total message volume. These return rates,
# not raw rows, so the reasoning layer never estimates counts from text.

async def message_counts_by_author(
    *,
    scope: Scope,
    filters: Optional[dict[str, Any]] = None,
) -> list[dict[str, Any]]:
    """Total messages per author within an optional filter (the denominators)."""
    where, params = _build_where(filters, scope)
    sql = f"SELECT author_id, COUNT(*) AS total FROM messages {where} GROUP BY author_id"
    pool = _get_pool()
    async with pool.connection() as conn:
        cur = await conn.execute(sql, params)
        return await cur.fetchall()


# ---------------------------------------------------------------------------
# Retrieval instrument 5: aggregation (counts, never bodies)
# ---------------------------------------------------------------------------

async def activity_stats(
    *,
    scope: Scope,
    group_by: str = "author_id",
    filters: Optional[dict[str, Any]] = None,
    limit: int = 50,
) -> list[dict[str, Any]]:
    """Message counts bucketed by one dimension, largest bucket first.

    Answers "who talks most", "when is this channel awake", "how big was March"
    without reading a single message body, so it costs one query and adds no
    rows to the model's context. The counting questions that would otherwise
    burn the retrieval budget paging through messages land here instead.

    group_by is a fixed whitelist because it is interpolated into the SQL.
    Every option is index-backed: author_id/channel_id have their own B-trees,
    day_of_week+hour_of_day share messages_dow_hour_idx, and the date buckets
    scan messages_created_at_idx.
    """
    buckets = {
        "author_id": "author_id",
        "channel_id": "channel_id",
        "day_of_week": "day_of_week",
        "hour_of_day": "hour_of_day",
        "day": "(created_at AT TIME ZONE %s)::date",
        "month": "date_trunc('month', created_at AT TIME ZONE %s)::date",
    }
    if group_by not in buckets:
        raise ValueError(f"group_by must be one of {sorted(buckets)}")
    expr = buckets[group_by]

    # The bucket expression sits in the SELECT list, ahead of the WHERE params.
    # GROUP BY 1 rather than repeating it keeps that ordering to one binding.
    params: list[Any] = [str(CORPUS_TZ)] if "%s" in expr else []
    where, where_params = _build_where(filters, scope)
    params.extend(where_params)

    sql = f"""
        SELECT {expr} AS bucket, COUNT(*) AS total
          FROM messages {where}
      GROUP BY 1
      ORDER BY total DESC
         LIMIT %s
    """
    pool = _get_pool()
    async with pool.connection() as conn:
        cur = await conn.execute(sql, [*params, limit])
        return await cur.fetchall()


# category_rate_by_author lived here. It computed a per-author rate as
# "messages with category = X over that author's total", and went with the
# column. The denominator half survives above; a numerator now has to come
# from something that actually exists in a row — a keyword match, a time
# bucket — rather than a label nothing ever wrote.


# ---------------------------------------------------------------------------
# Day summaries: the retrieval index over the corpus (specs-summaries.md)
# ---------------------------------------------------------------------------
# Derived and regenerable. Messages stay the ground truth; these are a lossy
# pointer layer, which is why nothing here ever replaces a message read.

async def messages_for_span(
    channel_id: int,
    start: datetime,
    end: datetime,
    *,
    limit: int = 5000,
) -> list[dict[str, Any]]:
    """Every message in one channel from start (inclusive) to end (exclusive),
    oldest first — the raw input to one summary.

    The limit is a blast radius, not a page size: a day that hits it is a day
    the summarizer is seeing only part of, and the caller should say so.
    """
    sql = """
        SELECT * FROM messages
         WHERE channel_id = %s AND created_at >= %s AND created_at < %s
      ORDER BY created_at ASC
         LIMIT %s
    """
    pool = _get_pool()
    async with pool.connection() as conn:
        cur = await conn.execute(sql, (channel_id, _as_utc(start), _as_utc(end), limit))
        return await cur.fetchall()


async def first_message_at(channel_id: int) -> Optional[datetime]:
    """Timestamp of the oldest stored message in a channel, or None if empty.

    Where summarization starts when a channel has no watermark yet.
    """
    pool = _get_pool()
    async with pool.connection() as conn:
        cur = await conn.execute(
            "SELECT MIN(created_at) AS first FROM messages WHERE channel_id = %s",
            (channel_id,),
        )
        row = await cur.fetchone()
    return row["first"] if row else None


async def upsert_day_summary(
    *,
    summary_date: date,
    channel_id: int,
    started_at: datetime,
    ended_at: datetime,
    prose: str,
    facets: dict[str, Any],
    first_message_id: int,
    last_message_id: int,
    message_count: int,
    model: str,
    prompt_version: str,
) -> int:
    """Write one day's summary, replacing any existing row for that day.

    Upsert rather than insert so re-running a day — after a prompt change, or
    after a backfill filled in messages that were missing the first time — is
    a normal operation instead of a duplicate-key error.
    """
    sql = """
        INSERT INTO day_summaries (
            summary_date, channel_id, granularity,
            started_at, ended_at, prose, facets,
            first_message_id, last_message_id, message_count,
            model, prompt_version
        )
        VALUES (%s, %s, 'day', %s, %s, %s, %s, %s, %s, %s, %s, %s)
        ON CONFLICT (summary_date, channel_id) WHERE granularity = 'day'
        DO UPDATE SET
            started_at       = EXCLUDED.started_at,
            ended_at         = EXCLUDED.ended_at,
            prose            = EXCLUDED.prose,
            facets           = EXCLUDED.facets,
            first_message_id = EXCLUDED.first_message_id,
            last_message_id  = EXCLUDED.last_message_id,
            message_count    = EXCLUDED.message_count,
            model            = EXCLUDED.model,
            prompt_version   = EXCLUDED.prompt_version,
            generated_at     = now()
        RETURNING id
    """
    pool = _get_pool()
    async with pool.connection() as conn:
        cur = await conn.execute(sql, (
            summary_date, channel_id,
            _as_utc(started_at), _as_utc(ended_at),
            prose, Jsonb(facets),
            first_message_id, last_message_id, message_count,
            model, prompt_version,
        ))
        row = await cur.fetchone()
    return row["id"]


async def recent_day_summaries(
    channel_id: int,
    *,
    before: date,
    days: int = 7,
) -> list[dict[str, Any]]:
    """The `days` day-summaries immediately preceding `before`, oldest first.

    This is the summarizer's whole memory of what came before — enough to
    resolve "the trip" and "he", and to carry an unresolved thread forward.
    Oldest first because that is the order it happened in.
    """
    sql = """
        SELECT * FROM (
            SELECT * FROM day_summaries
             WHERE channel_id = %s
               AND granularity = 'day'
               AND summary_date < %s
          ORDER BY summary_date DESC
             LIMIT %s
        ) recent
      ORDER BY summary_date ASC
    """
    pool = _get_pool()
    async with pool.connection() as conn:
        cur = await conn.execute(sql, (channel_id, before, days))
        return await cur.fetchall()


# ---------------------------------------------------------------------------
# Retrieval instrument 7: the summary layer (orientation, never evidence)
# ---------------------------------------------------------------------------
# The one instrument that answers a question about a stretch of time rather
# than about particular messages. A day summary was written with the whole day
# in view, so its facets — decisions, open_threads — carry a judgment about
# what mattered that no reader of thirty retrieved rows can reconstruct.
#
# Same rule as clusters: this text is model-written and is never citable. A
# summary tells you which span is worth opening; first_message_id and
# last_message_id are how you open it.

async def read_summaries(
    *,
    scope: Scope,
    channel_id: Optional[int] = None,
    after: Optional[date] = None,
    before: Optional[date] = None,
    granularity: str = "day",
    limit: int = 14,
) -> list[dict[str, Any]]:
    """Day summaries across a date range, **oldest first**.

    Chronological on purpose, and it is the only instrument that is. Every
    other read here sorts by recency or by count, because they answer "find
    me the messages that..."; this one answers "what happened over this
    stretch", and a stretch read out of order is not a story.

    `limit` truncates from the OLD end — an unbounded call returns the most
    recent summaries rather than the first ones ever written, which is the
    useful default for "what has been going on". Index: the range scan rides
    day_summaries_date_idx (channel_id, summary_date).
    """
    if granularity not in ("day", "session"):
        raise ValueError("granularity must be 'day' or 'session'")

    scope_sql, scope_params = _scope_clause(scope.narrowed_to(channel_id))
    clauses = [scope_sql, "granularity = %s"]
    params: list[Any] = [*scope_params, granularity]
    if after is not None:
        clauses.append("summary_date >= %s")
        params.append(after)
    if before is not None:
        clauses.append("summary_date <= %s")
        params.append(before)

    # Newest `limit` rows in the range, then flipped back into reading order.
    sql = f"""
        SELECT * FROM (
            SELECT id, summary_date, channel_id, granularity, started_at, ended_at,
                   prose, facets, first_message_id, last_message_id, message_count
              FROM day_summaries
             WHERE {" AND ".join(clauses)}
          ORDER BY summary_date DESC, started_at DESC
             LIMIT %s
        ) recent
      ORDER BY summary_date ASC, started_at ASC
    """
    pool = _get_pool()
    async with pool.connection() as conn:
        cur = await conn.execute(sql, [*params, limit])
        return await cur.fetchall()


async def summary_date_range(
    *, scope: Scope, channel_id: Optional[int] = None,
) -> dict[str, Any]:
    """How much summarized history exists: first date, last date, and how many.

    What a reader needs before asking for a range — "summarize last month" is
    unanswerable if the summariser has only ever covered a week, and finding
    that out by getting a short result back is guesswork.
    """
    scope_sql, params = _scope_clause(scope.narrowed_to(channel_id))
    sql = f"""
        SELECT MIN(summary_date) AS first_day,
               MAX(summary_date) AS last_day,
               COUNT(*)          AS days
          FROM day_summaries
         WHERE {scope_sql} AND granularity = 'day'
    """
    pool = _get_pool()
    async with pool.connection() as conn:
        cur = await conn.execute(sql, params)
        row = await cur.fetchone()
    return row or {"first_day": None, "last_day": None, "days": 0}


async def summary_watermark(channel_id: int) -> Optional[date]:
    """Last date this channel is summarized through, or None if never run."""
    pool = _get_pool()
    async with pool.connection() as conn:
        cur = await conn.execute(
            "SELECT summarized_through FROM summary_state WHERE channel_id = %s",
            (channel_id,),
        )
        row = await cur.fetchone()
    return row["summarized_through"] if row else None


async def set_summary_watermark(channel_id: int, through: date) -> None:
    """Advance the watermark. Only ever moves forward, so an out-of-order or
    repeated run cannot rewind a channel and cause days to be redone."""
    sql = """
        INSERT INTO summary_state (channel_id, summarized_through)
        VALUES (%s, %s)
        ON CONFLICT (channel_id) DO UPDATE
           SET summarized_through = GREATEST(summary_state.summarized_through,
                                             EXCLUDED.summarized_through),
               updated_at = now()
    """
    pool = _get_pool()
    async with pool.connection() as conn:
        await conn.execute(sql, (channel_id, through))


# ---------------------------------------------------------------------------
# Clusters — topical stretches, written by the summarizer, embedded separately
# ---------------------------------------------------------------------------

async def latest_cluster(channel_id: int) -> Optional[dict[str, Any]]:
    """The channel's most recent cluster, or None.

    This is what the next day's run re-cuts: its start message is where that
    run's clustering input begins, so the cluster gets re-drawn with the new
    day's context instead of staying cut blind at midnight.
    """
    pool = _get_pool()
    async with pool.connection() as conn:
        cur = await conn.execute(
            "SELECT * FROM clusters WHERE channel_id = %s "
            "ORDER BY first_message_id DESC LIMIT 1",
            (channel_id,),
        )
        return await cur.fetchone()


async def clusters_overlapping(
    channel_id: int, start: datetime, end: datetime
) -> list[dict[str, Any]]:
    """Clusters whose span intersects [start, end), oldest first.

    A re-run of a day must replace these whole, never trim them — so the
    caller extends its input window back to the earliest one's start.
    """
    pool = _get_pool()
    async with pool.connection() as conn:
        cur = await conn.execute(
            "SELECT * FROM clusters WHERE channel_id = %s "
            "AND started_at < %s AND ended_at >= %s "
            "ORDER BY first_message_id ASC",
            (channel_id, _as_utc(end), _as_utc(start)),
        )
        return await cur.fetchall()


async def replace_clusters(
    channel_id: int,
    window_start: datetime,
    window_end: datetime,
    clusters: Sequence[dict[str, Any]],
) -> tuple[int, int]:
    """Swap the clusters covering one window for a new set, atomically.

    Deletes every cluster intersecting [window_start, window_end), then
    inserts the new rows — one transaction, so no crash can leave the window
    half-clustered or doubly clustered. New rows have no embedding yet; the
    embed pass finds them via embedding IS NULL.

    Returns (deleted, inserted).
    """
    pool = _get_pool()
    async with pool.connection() as conn:
        async with conn.cursor() as cur:
            await cur.execute(
                "DELETE FROM clusters WHERE channel_id = %s "
                "AND started_at < %s AND ended_at >= %s",
                (channel_id, _as_utc(window_end), _as_utc(window_start)),
            )
            deleted = cur.rowcount
            for c in clusters:
                await cur.execute(
                    """
                    INSERT INTO clusters (
                        channel_id, first_message_id, last_message_id,
                        started_at, ended_at, message_count,
                        topic, summary, model, prompt_version
                    )
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    """,
                    (
                        channel_id,
                        c["first_message_id"], c["last_message_id"],
                        _as_utc(c["started_at"]), _as_utc(c["ended_at"]),
                        c["message_count"],
                        c.get("topic"), c["summary"],
                        c["model"], c["prompt_version"],
                    ),
                )
    return deleted, len(clusters)


async def unembedded_clusters(limit: int = 64) -> list[dict[str, Any]]:
    """Clusters still waiting for a vector, oldest first — the embed pass's
    work queue. Every channel's pending rows, since embedding is corpus-wide."""
    pool = _get_pool()
    async with pool.connection() as conn:
        cur = await conn.execute(
            "SELECT * FROM clusters WHERE embedding IS NULL ORDER BY id LIMIT %s",
            (limit,),
        )
        return await cur.fetchall()


async def set_cluster_embedding(
    cluster_id: int, embedding: Sequence[float], embed_model: str
) -> None:
    """Attach one cluster's vector, recording what produced it.

    Vector() for the same reason similarity_search uses it: a bare list arrives
    as double precision[]. Here the assignment cast into the vector column
    rescues it, so this worked either way — which is precisely what let the
    read path stay broken unnoticed. Both sides send the same type now.
    """
    pool = _get_pool()
    async with pool.connection() as conn:
        await conn.execute(
            "UPDATE clusters SET embedding = %s, embed_model = %s, "
            "embedded_at = now() WHERE id = %s",
            (Vector(embedding), embed_model, cluster_id),
        )


# ---------------------------------------------------------------------------
# Retrieval instrument 6: cluster similarity (semantic, and only an index)
# ---------------------------------------------------------------------------
# A hit here is a span worth reading, not a fact. Clusters carry LLM-written
# summaries, so citing one would cite something no human ever said — the caller
# drills into first_message_id..last_message_id and cites the messages it finds
# (specs-summaries.md decision 2, enforced by the ledger's id check).

async def similarity_search(
    embedding: Sequence[float],
    *,
    scope: Scope,
    channel_id: Optional[int] = None,
    limit: int = 10,
) -> list[dict[str, Any]]:
    """Clusters whose summary vector is nearest the query vector, closest first.

    Cosine distance over clusters_embedding_idx. The vector never comes from
    here: db.py has no model client (llm.py imports db, not the reverse), so
    the caller embeds the query — with is_query=True, since a query embedded as
    a document is compared against the wrong shape.

    Unembedded clusters are excluded rather than sorted last: a NULL vector has
    no distance, and letting it through would hand back rows that matched
    nothing. The `embedding IS NOT NULL` predicate also keeps this off any
    cluster the embed pass has not reached yet.
    """
    scope_sql, filter_params = _scope_clause(scope.narrowed_to(channel_id))
    where = "WHERE " + " AND ".join([scope_sql, "embedding IS NOT NULL"])

    # The query vector binds twice — once for the returned distance, once for
    # the ORDER BY that the HNSW index actually answers.
    sql = f"""
        SELECT id, channel_id, topic, summary,
               first_message_id, last_message_id,
               started_at, ended_at, message_count,
               embedding {DISTANCE_OP} %s AS distance
          FROM clusters
          {where}
      ORDER BY embedding {DISTANCE_OP} %s
         LIMIT %s
    """
    # Vector(), not the bare list. register_vector_async teaches psycopg to
    # send a pgvector Vector as `vector`; a plain Python list falls through to
    # the default dumper and arrives as `double precision[]`. Writing one of
    # those into a vector column works — pgvector defines an assignment cast —
    # so the embed pass never noticed, but `<=>` is an operator and operators
    # take no implicit cast: it fails with "operator does not exist: vector <=>
    # double precision[]". Reads and writes have to agree, so both wrap.
    query = Vector(embedding)
    pool = _get_pool()
    async with pool.connection() as conn:
        cur = await conn.execute(
            sql, [query, *filter_params, query, limit]
        )
        return await cur.fetchall()


# ---------------------------------------------------------------------------
# Summary failure flags — how a failed day stays retryable
# ---------------------------------------------------------------------------
# The watermark advances past a failed day only once it is flagged here, and
# every run retries flagged days before starting new ones — so a failure costs
# retries, never a silent permanent gap.

async def flag_summary_failure(
    channel_id: int, day: date, *, stage: str, error: str
) -> None:
    """Record that a day's run failed at `stage` ('summary' or 'clusters')."""
    pool = _get_pool()
    async with pool.connection() as conn:
        await conn.execute(
            """
            INSERT INTO summary_failures (channel_id, day, stage, error)
            VALUES (%s, %s, %s, %s)
            ON CONFLICT (channel_id, day) DO UPDATE
               SET stage = EXCLUDED.stage,
                   error = EXCLUDED.error,
                   attempts = summary_failures.attempts + 1,
                   last_failed_at = now()
            """,
            (channel_id, day, stage, error[:2000]),
        )


async def clear_summary_failure(channel_id: int, day: date) -> None:
    """Drop a day's flag after a successful run. No-op if it was never set."""
    pool = _get_pool()
    async with pool.connection() as conn:
        await conn.execute(
            "DELETE FROM summary_failures WHERE channel_id = %s AND day = %s",
            (channel_id, day),
        )


async def summary_failed(channel_id: int, day: date) -> bool:
    """Whether a day is currently flagged as failed."""
    pool = _get_pool()
    async with pool.connection() as conn:
        cur = await conn.execute(
            "SELECT 1 FROM summary_failures WHERE channel_id = %s AND day = %s",
            (channel_id, day),
        )
        return await cur.fetchone() is not None


async def failed_summary_days(channel_id: int) -> list[date]:
    """Every flagged day for one channel, oldest first — the retry queue."""
    pool = _get_pool()
    async with pool.connection() as conn:
        cur = await conn.execute(
            "SELECT day FROM summary_failures WHERE channel_id = %s ORDER BY day",
            (channel_id,),
        )
        rows = await cur.fetchall()
    return [row["day"] for row in rows]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _as_utc(dt: datetime) -> datetime:
    """Normalise to timezone-aware UTC. Naive datetimes are assumed UTC."""
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)
