"""PostgreSQL + pgvector storage layer for the Discord RAG bot.

This is scaffolding for the storage layer described in the project design doc.
The guiding idea from that doc shapes everything here:

  * ONE table holds both the structured metadata (speaker, category, sentiment,
    target-person, timestamps) AND the embedding, so a single query can filter
    and rank together instead of vector-searching and post-filtering.
  * Retrieval is layered. Vector similarity is one instrument among several.
    Structured filters, keyword/exact match, anchor-based lookup, and SQL
    aggregation are first-class and often the *primary* mechanism. Each gets
    its own function below rather than being bolted onto similarity search.
  * Classification happens at ingestion, into indexed columns, so pattern
    questions hit B-tree indexes instead of per-query LLM passes.

Nothing here decides the questions the doc leaves open. In particular:
  * EMBED_DIM is provisional — the embedding model is not chosen yet, so the
    dimension is read from the environment and defaults to a common value.
  * `embedding` is nullable: messages can be ingested and classified before
    (or without) being embedded, which keeps backfill an open question.
  * Classification columns are nullable and untrusted; the doc treats their
    noise on joke-heavy chat as an unmeasured risk, not a solved problem.
"""

import logging
import os
from datetime import datetime, timezone
from typing import Any, Optional, Sequence

from dotenv import load_dotenv
from pgvector.psycopg import register_vector_async
from psycopg.rows import dict_row
from psycopg_pool import AsyncConnectionPool

load_dotenv()

log = logging.getLogger(__name__)

# Standard libpq connection string, e.g.
#   postgresql://user:pass@host:5432/ragbot
# Read but not required at import: bot.py imports this module unconditionally,
# so demanding the variable here would stop the whole bot from starting on a
# deploy that has no database. open_pool() is the one place that truly needs it.
DATABASE_URL = os.environ.get("DATABASE_URL")

# PROVISIONAL. The embedding model is an open question in the design doc
# (local sentence-transformers vs. an API model). The vector column and its
# HNSW index are built to this width, so changing models later means a
# migration + re-embed. 768 is a common default; do not treat it as committed.
EMBED_DIM = int(os.environ.get("EMBED_DIM", "768"))

# Distance operator for similarity search. Must match how the chosen embedding
# model was trained/normalised: <=> cosine, <-> L2, <#> (neg) inner product.
DISTANCE_OP = "<=>"


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------
# day_of_week (0-6) and hour_of_day (0-23) are stored as plain columns computed
# at ingestion rather than generated columns: EXTRACT over timestamptz is not
# immutable (it depends on the session time zone), so it can't back a generated
# column. Computing them once at ingest also fits "classified at ingestion".

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
    hour_of_day         smallint    NOT NULL,   -- 0..23, in the corpus tz

    -- Ingestion-time classification. Nullable and untrusted (see module doc).
    category            text,                   -- e.g. proposal/complaint/...
    sentiment           text,
    target_person_id    bigint,                 -- who a message is "about"

    -- Nullable so a message can exist before it is embedded (backfill is open).
    embedding           vector({EMBED_DIM})
);

-- Selective equality filters (per-person, per-target) are meant to pre-filter
-- via B-tree indexes; for those, the exact scan over the narrowed set beats an
-- approximate vector index. See similarity_search() for the interaction.
CREATE INDEX IF NOT EXISTS messages_author_idx        ON messages (author_id);
CREATE INDEX IF NOT EXISTS messages_target_idx        ON messages (target_person_id);
CREATE INDEX IF NOT EXISTS messages_channel_idx       ON messages (channel_id);
CREATE INDEX IF NOT EXISTS messages_category_idx      ON messages (category);
CREATE INDEX IF NOT EXISTS messages_created_at_idx    ON messages (created_at);
CREATE INDEX IF NOT EXISTS messages_reply_to_idx      ON messages (reply_to_message_id);
CREATE INDEX IF NOT EXISTS messages_author_cat_idx    ON messages (author_id, category);
CREATE INDEX IF NOT EXISTS messages_dow_hour_idx      ON messages (day_of_week, hour_of_day);

-- Approximate index for weakly-filtered similarity search. When filters are
-- highly selective this index is deliberately bypassed (planner's choice);
-- it earns its keep on broad searches across the whole corpus.
-- TODO: revisit ef_search / m once the embedding model and data volume settle.
CREATE INDEX IF NOT EXISTS messages_embedding_idx
    ON messages USING hnsw (embedding vector_cosine_ops);

-- TODO: keyword_search() uses ILIKE for now. For real term/name matching add
-- a pg_trgm GIN index or a tsvector column; embeddings are weak on exact names.
"""


# ---------------------------------------------------------------------------
# Pool lifecycle
# ---------------------------------------------------------------------------

_pool: Optional[AsyncConnectionPool] = None


async def _configure(conn) -> None:
    """Per-connection setup: teach psycopg how to adapt pgvector types."""
    await register_vector_async(conn)


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
    category: Optional[str] = None,
    sentiment: Optional[str] = None,
    target_person_id: Optional[int] = None,
    embedding: Optional[Sequence[float]] = None,
) -> int:
    """Insert a message (or update it if already ingested). Returns the row id.

    day_of_week / hour_of_day are derived here from created_at. Classification
    and embedding are optional so ingestion, classification, and embedding can
    be separate passes (keeps backfill/embedding strategy open).
    """
    created_at = _as_utc(created_at)
    day_of_week = created_at.weekday()          # 0=Monday
    hour_of_day = created_at.hour

    sql = """
        INSERT INTO messages (
            discord_message_id, channel_id, author_id, content, created_at,
            reply_to_message_id, day_of_week, hour_of_day,
            category, sentiment, target_person_id, embedding
        )
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        ON CONFLICT (discord_message_id) DO UPDATE SET
            content             = EXCLUDED.content,
            reply_to_message_id = EXCLUDED.reply_to_message_id,
            category            = COALESCE(EXCLUDED.category, messages.category),
            sentiment           = COALESCE(EXCLUDED.sentiment, messages.sentiment),
            target_person_id    = COALESCE(EXCLUDED.target_person_id, messages.target_person_id),
            embedding           = COALESCE(EXCLUDED.embedding, messages.embedding)
        RETURNING id
    """
    params = (
        discord_message_id, channel_id, author_id, content, created_at,
        reply_to_message_id, day_of_week, hour_of_day,
        category, sentiment, target_person_id, embedding,
    )
    pool = _get_pool()
    async with pool.connection() as conn:
        cur = await conn.execute(sql, params)
        row = await cur.fetchone()
    return row["id"]


async def set_classification(
    discord_message_id: int,
    *,
    category: Optional[str] = None,
    sentiment: Optional[str] = None,
    target_person_id: Optional[int] = None,
) -> None:
    """Attach/replace ingestion-time classification for one message."""
    sql = """
        UPDATE messages
           SET category         = COALESCE(%s, category),
               sentiment        = COALESCE(%s, sentiment),
               target_person_id = COALESCE(%s, target_person_id)
         WHERE discord_message_id = %s
    """
    pool = _get_pool()
    async with pool.connection() as conn:
        await conn.execute(sql, (category, sentiment, target_person_id, discord_message_id))


async def set_embedding(discord_message_id: int, embedding: Sequence[float]) -> None:
    """Attach/replace the embedding for one message (separate pass from ingest)."""
    pool = _get_pool()
    async with pool.connection() as conn:
        await conn.execute(
            "UPDATE messages SET embedding = %s WHERE discord_message_id = %s",
            (embedding, discord_message_id),
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
    "category",
    "sentiment",
    "target_person_id",
    "day_of_week",
    "hour_of_day",
    "reply_to_message_id",
}


def _build_where(filters: Optional[dict[str, Any]]) -> tuple[str, list[Any]]:
    """Turn a filter dict into a WHERE fragment + ordered params.

    Supported keys: the equality fields in _EQ_FIELDS, plus range keys
    'after' / 'before' (created_at bounds). Returns ("", []) when empty.
    """
    if not filters:
        return "", []

    clauses: list[str] = []
    params: list[Any] = []

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

    if not clauses:
        return "", []
    return "WHERE " + " AND ".join(clauses), params


# ---------------------------------------------------------------------------
# Retrieval instrument 1: vector similarity (with structured filters)
# ---------------------------------------------------------------------------

async def similarity_search(
    query_embedding: Sequence[float],
    *,
    filters: Optional[dict[str, Any]] = None,
    limit: int = 10,
) -> list[dict[str, Any]]:
    """Rank messages by embedding distance, optionally within a filtered subset.

    Filter + rank happen in one statement (the whole point of co-locating
    metadata and vectors). When the filters are selective the planner will
    pre-filter via B-tree and scan exactly; when they are broad it can lean on
    the HNSW index. Each row includes a `distance` column (smaller = closer).
    """
    where, params = _build_where(filters)
    # Only rank rows that are actually embedded; fold that into the WHERE.
    where = f"{where} AND embedding IS NOT NULL" if where else "WHERE embedding IS NOT NULL"
    sql = f"""
        SELECT *, embedding {DISTANCE_OP} %s AS distance
          FROM messages
          {where}
        ORDER BY embedding {DISTANCE_OP} %s
         LIMIT %s
    """
    pool = _get_pool()
    async with pool.connection() as conn:
        cur = await conn.execute(sql, [query_embedding, *params, query_embedding, limit])
        return await cur.fetchall()


# ---------------------------------------------------------------------------
# Retrieval instrument 2: structured filter (pure SQL, no vectors)
# ---------------------------------------------------------------------------

async def structured_search(
    *,
    filters: Optional[dict[str, Any]] = None,
    order_by: str = "created_at DESC",
    limit: int = 100,
) -> list[dict[str, Any]]:
    """Pure-SQL retrieval by structured constraints — often the primary path.

    order_by is a fixed whitelist of columns to keep it injection-safe.
    """
    allowed = {
        "created_at ASC", "created_at DESC",
        "author_id", "category", "id ASC", "id DESC",
    }
    if order_by not in allowed:
        raise ValueError(f"order_by must be one of {sorted(allowed)}")

    where, params = _build_where(filters)
    sql = f"SELECT * FROM messages {where} ORDER BY {order_by} LIMIT %s"
    pool = _get_pool()
    async with pool.connection() as conn:
        cur = await conn.execute(sql, [*params, limit])
        return await cur.fetchall()


# ---------------------------------------------------------------------------
# Retrieval instrument 3: keyword / exact match
# ---------------------------------------------------------------------------

async def keyword_search(
    term: str,
    *,
    filters: Optional[dict[str, Any]] = None,
    limit: int = 50,
) -> list[dict[str, Any]]:
    """Substring match for names/specific terms where embeddings are weak.

    TODO: ILIKE is a placeholder. Swap for pg_trgm or full-text search once the
    term-matching requirements are clearer (see schema note).
    """
    where, params = _build_where(filters)
    joiner = "AND" if where else "WHERE"
    sql = f"SELECT * FROM messages {where} {joiner} content ILIKE %s ORDER BY created_at DESC LIMIT %s"
    pool = _get_pool()
    async with pool.connection() as conn:
        cur = await conn.execute(sql, [*params, f"%{term}%", limit])
        return await cur.fetchall()


# ---------------------------------------------------------------------------
# Retrieval instrument 4: anchor-based (structural, not semantic)
# ---------------------------------------------------------------------------

async def replies_to(discord_message_id: int, *, limit: int = 100) -> list[dict[str, Any]]:
    """Messages that were direct replies to a given message."""
    pool = _get_pool()
    async with pool.connection() as conn:
        cur = await conn.execute(
            "SELECT * FROM messages WHERE reply_to_message_id = %s "
            "ORDER BY created_at ASC LIMIT %s",
            (discord_message_id, limit),
        )
        return await cur.fetchall()


async def messages_near(
    anchor: datetime,
    *,
    window_minutes: int = 30,
    channel_id: Optional[int] = None,
    limit: int = 100,
) -> list[dict[str, Any]]:
    """Messages within a time window around an anchor timestamp.

    Timestamp proximity as a structural retrieval — useful for reconstructing
    the conversation surrounding a hit without any semantic assumption.
    """
    anchor = _as_utc(anchor)
    clauses = ["created_at BETWEEN %s - make_interval(mins => %s) "
               "AND %s + make_interval(mins => %s)"]
    params: list[Any] = [anchor, window_minutes, anchor, window_minutes]
    if channel_id is not None:
        clauses.append("channel_id = %s")
        params.append(channel_id)
    sql = f"SELECT * FROM messages WHERE {' AND '.join(clauses)} ORDER BY created_at ASC LIMIT %s"
    params.append(limit)
    pool = _get_pool()
    async with pool.connection() as conn:
        cur = await conn.execute(sql, params)
        return await cur.fetchall()


# ---------------------------------------------------------------------------
# Retrieval instrument 5: aggregation (computed in SQL, never eyeballed)
# ---------------------------------------------------------------------------
# The doc is emphatic: pattern claims need denominators. "C mentions hiking
# less" is meaningless without C's total message volume. These return rates,
# not raw rows, so the reasoning layer never estimates counts from text.

async def message_counts_by_author(
    *,
    filters: Optional[dict[str, Any]] = None,
) -> list[dict[str, Any]]:
    """Total messages per author within an optional filter (the denominators)."""
    where, params = _build_where(filters)
    sql = f"SELECT author_id, COUNT(*) AS total FROM messages {where} GROUP BY author_id"
    pool = _get_pool()
    async with pool.connection() as conn:
        cur = await conn.execute(sql, params)
        return await cur.fetchall()


async def category_rate_by_author(
    category: str,
    *,
    filters: Optional[dict[str, Any]] = None,
) -> list[dict[str, Any]]:
    """Per-author rate of a category: matching count over total, with the raw
    numbers alongside so small samples are visible (silence/low-N is weak
    evidence and must stay inspectable).
    """
    where, params = _build_where(filters)
    sql = f"""
        SELECT author_id,
               COUNT(*) FILTER (WHERE category = %s) AS matching,
               COUNT(*)                              AS total,
               COUNT(*) FILTER (WHERE category = %s)::float
                   / NULLIF(COUNT(*), 0)             AS rate
          FROM messages
          {where}
      GROUP BY author_id
    """
    pool = _get_pool()
    async with pool.connection() as conn:
        cur = await conn.execute(sql, [category, *params, category])
        return await cur.fetchall()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _as_utc(dt: datetime) -> datetime:
    """Normalise to timezone-aware UTC. Naive datetimes are assumed UTC."""
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)
