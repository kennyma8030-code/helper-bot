# Day Summaries — Spec

Replaces vector search with two retrieval paths: exact/keyword match over raw
messages, and LLM-written daily summaries that act as an index into the corpus.

Rationale: the embedding pipeline was never built — `set_embedding()` has no
caller, so every `similarity_search()` matches zero rows. At 3-person volume,
a day-level summary index is cheaper to build, inspectable when it goes wrong,
and regenerable when the prompt changes.

## Supersedes

These `[v1]` items in `specs.md` Part 2 are void:

- Metadata-filtered vector search
- pgvector (extension stays only if something else needs it)
- HNSW indexing
- Multiple embedding spaces / cross-encoder reranking (already `[later]`,
  now out of scope entirely)

Everything else in `specs.md` stands — the loop, ledger, budgets, and triage
are unchanged by this.

## Decisions

1. **Days are the summary unit.** One summary per (day, channel). Not
   sessions, not weeks. Days are trivially addressable, map to how people
   remember chat, and are the only unit whose boundaries need no tuning.
   Sub-day and multi-day tiers are deferred (see below).

2. **Summaries are an index, never evidence.** A summary points at days worth
   reading; facts cite raw messages only. Enforced in code: cited message ids
   must exist in `messages`, which a summary id never will. Rationale:
   summaries are LLM-written and lossy — a fact citing one is unverifiable.

3. **Every summary carries the message id range it covers.** `first_message_id`
   / `last_message_id` make drill-down from any summary hit mechanical, and
   let a summary be re-derived from exactly the rows that produced it.

4. **Summaries are structured, not just prose.** A `facets` jsonb column holds
   the searchable parts (entities, participants, topics); `prose` is for
   reading once a day is selected. Prose alone can only be searched by reading
   all of it.

5. **Summaries are versioned.** Every row stores `model` and `prompt_version`.
   The summary prompt will change; without this there is no way to find stale
   rows or regenerate selectively.

6. **Generation is watermark-driven, not scheduled.** A `summarized_through`
   date advances as days are written. The job summarizes from the watermark to
   yesterday, and is safe to call repeatedly. Runs on bot startup and on a
   timer inside the existing process — no cron service, and it self-heals
   after downtime instead of leaving a gap.

7. **Write summaries for recall, not readability.** Entity-dense and specific:
   names, places, proper nouns, decisions, unresolved threads. Prose narrative
   ("the group discussed weekend plans") is unretrievable — if a name isn't in
   the summary, that day is invisible.

8. **The summarizer sees the previous few days.** Otherwise references like
   "the trip" or "he" are unresolvable and summaries continue nothing.

## Schema

```sql
CREATE TABLE day_summaries (
    id                bigserial PRIMARY KEY,
    summary_date      date        NOT NULL,
    channel_id        bigint      NOT NULL,

    prose             text        NOT NULL,
    facets            jsonb       NOT NULL DEFAULT '{}'::jsonb,

    first_message_id  bigint      NOT NULL,   -- discord_message_id
    last_message_id   bigint      NOT NULL,
    message_count     integer     NOT NULL,

    model             text        NOT NULL,
    prompt_version    text        NOT NULL,
    generated_at      timestamptz NOT NULL DEFAULT now(),

    UNIQUE (summary_date, channel_id)         -- upsert key; re-runnable
);

CREATE INDEX day_summaries_date_idx   ON day_summaries (summary_date);
CREATE INDEX day_summaries_facets_idx ON day_summaries USING gin (facets jsonb_path_ops);

CREATE TABLE summary_state (
    channel_id          bigint PRIMARY KEY,
    summarized_through  date   NOT NULL
);
```

Facets shape (all optional, all arrays unless noted):

```json
{
  "participants": [111, 222],
  "entities": ["cabin", "zillow listing", "chris's brother"],
  "topics": ["trip planning", "rent"],
  "decisions": [{"what": "trip moved to may", "msg_ids": [123, 456]}],
  "open_threads": ["who is fronting the deposit"],
  "aliases_observed": {"222": ["chris", "kris"]}
}
```

## Build order

1. **Delete the vector path.** `similarity_search` + `set_embedding` from
   `db.py`; `_embed_query`, `EMBED_MODEL`, and the `similarity_search` tool
   declaration from `loop.py`; the `embedding` column, HNSW index, and
   `EMBED_DIM`/`DISTANCE_OP` from the schema; item 4 of the tool preference
   list in `PLANNER_PROMPT`.

2. **Summaries table + generation job.** Schema above, a
   `summarize_day(date, channel_id)` that returns prose + facets, the
   watermark loop, and a backfill that runs it over existing history.

3. **`read_summaries` tool.** One new entry in `TOOLS` taking a date range and
   optional channel, returning summaries for that span. This is where it
   becomes clear whether the summaries are good enough to retrieve on.

4. **Report match counts.** Every search in `_execute_call` caps at 30 rows
   (`MAX_ROWS_PER_CALL`) with no signal that more existed. Run `COUNT(*)`
   alongside each query and include `matched` vs `returned` in what
   `_render_results` sends back.

5. **Validate citation ids.** `Ledger.apply` checks only that `citations` is a
   non-empty dict, never that the ids are real. Check against `messages` and
   reject on miss (this is what makes decision 2 enforceable).

Steps 4 and 5 are independent of 1–3 and can land in any order.

## Open

- **Timezone for day boundaries.** Storage is `timestamptz` and `_as_utc()`
  normalizes to UTC, but `db.py` describes `hour_of_day` as "in the corpus
  tz". Bucketing on UTC splits late-night conversations across two summaries.
  Pick the group's local zone and commit it to a constant.

## Deferred — with triggers

- **Sub-day session summaries** → a month of real summaries shows high-volume
  days losing detail that questions actually need. Day summaries flag
  themselves (`needs_detail`) rather than being split by a message-count rule.
- **Week/month rollups** → queries routinely span more days than fit in
  context at day resolution.
- **Entity pages (wiki-style context store)** → questions need to enter by
  entity rather than by time, and scanning facets across all summaries to
  find an entity's date range is the bottleneck. Build as a rollup over
  facets, not a second ingestion path; pages append claims with message-id
  provenance rather than being rewritten.
- **Aggregation tools** → `db.py` already has `message_counts_by_author` and
  friends, unexposed in `TOOLS`, which is why `PLANNER_PROMPT` says "you
  cannot count." Exposing them (plus per-day counts stored at summary time)
  removes that restriction.
- **Alias-expanded keyword search** → `aliases_observed` accumulates enough
  that exact match is demonstrably missing people by nickname.
