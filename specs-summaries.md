# Day Summaries — Spec

> **2026-08-16:** the vector path changed shape — per-message embeddings are
> gone; the summarizer now also cuts topical clusters whose summaries are what
> gets embedded. See the Addendum at the bottom; parts of this spec it
> supersedes are marked inline.

Adds a third retrieval path alongside the existing two: LLM-written daily
summaries that act as an index into the corpus, sitting next to exact/keyword
match over raw messages and metadata-filtered vector search.

Rationale: the embedding pipeline was never built — `set_embedding()`
(`db.py:321`) has no caller, so every `similarity_search()` currently matches
zero rows. At 3-person volume, a day-level summary index is cheaper to build,
inspectable when it goes wrong, and regenerable when the prompt changes — so
summaries are the near-term bet and land first. The vector path is not
removed; it is finished (build order step 4) and kept as the last-resort
instrument it was always specced to be.

## Relationship to specs.md

Nothing in `specs.md` is void. These `[v1]` items stand as written:

- Metadata-filtered vector search
- pgvector
- HNSW indexing

Still `[later]`, unchanged: multiple embedding spaces / cross-encoder
reranking.

The loop, ledger, budgets, and triage are unchanged by this.

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

9. **Summaries and vector search are different instruments, both kept.**
   Summaries answer "which days are worth reading" — coarse, time-addressed,
   entity-dense. Vector search answers "which individual message means roughly
   this" — fine-grained, and the only path that survives when the summary
   dropped the detail being asked about (see Limitations). `PLANNER_PROMPT`'s
   preference list gains summaries as a new entry — structured filters,
   keyword, anchors, summaries, similarity last — and item 4's existing
   "last resort" framing for similarity is kept verbatim. What changes for
   similarity search is only that it stops being a no-op.

10. **A dead tool must never look like an empty result.** Until the backfill
    has run, `similarity_search` returning nothing is indistinguishable from
    "no messages match" — the model ledgers a false dead branch. Any search
    over an unpopulated index reports coverage explicitly ("0 of N messages
    embedded") rather than an empty row list. *(Applied in its strongest form
    on 2026-08-16: the dead tool was removed from the planner entirely — see
    Addendum.)*

## Limitations of LLM summaries

What this approach cannot do — the reason the vector path is kept rather than
deleted, and the list to re-read when summary retrieval disappoints.

1. **Compression is lossy, and the loss is chosen before the question
   exists.** A summary is written once and must serve every future question.
   Whatever the summarizer judged unimportant in March is unrecoverable from
   the summary in July. Vector search is the opposite trade: it compresses
   nothing and decides relevance at question time.

2. **Absence in a summary is not absence in the corpus.** This breaks exactly
   the questions that feel easiest — "has anyone ever mentioned X", "when did
   we stop talking about Y". A summary index can support "yes, here", never
   "no, never." Only a full scan (keyword) can answer negatives, and the
   planner must not read a summary miss as a dead branch.

3. **The summarizer can be wrong in ways nothing catches.** It can merge two
   conversations, attribute a statement to the wrong person, or assert
   something plausible that no message says. A wrong summary looks exactly
   like a right one. This is what decision 2 exists to contain: summaries
   route, raw messages testify.

4. **Sarcasm flattens into fact.** `PLANNER_PROMPT` already warns that this
   chat is joke-heavy. A summarizer compressing "we're definitely selling the
   car" writes down a decision that was never made — and unlike a raw message,
   the summary carries no tone for the planner to be suspicious of.

5. **Entity-resolution errors become authoritative.** Decision 8 gives the
   summarizer prior days so pronouns resolve; when it resolves one wrong, the
   error is baked into `facets` and looks like structured ground truth. Bad
   entries in `aliases_observed` are self-reinforcing.

6. **Granularity is uniform; conversation is not.** A 500-message day and a
   5-message day get one summary each. Detail loss scales with how much
   actually happened, so the busiest days — usually the ones worth
   asking about — are the most compressed. (Deferred: `needs_detail`.)

7. **Day boundaries cut real threads.** A conversation running past midnight
   is split across two summaries, neither coherent. No timezone choice
   removes this; it only moves the cut (see Open).

8. **Regeneration cost is recurring, not one-time.** Every prompt or model
   change invalidates the corpus, and re-summarizing all history is an LLM
   call per day per channel. Between changes the corpus is heterogeneous —
   which is what decision 5's `prompt_version` is for, and why partial
   regeneration is normal rather than exceptional.

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
  "continues_from": ["cabin dates, left unfixed on the 14th"],
  "aliases_observed": {"222": ["chris", "kris"]}
}
```

`open_threads` and `continues_from` are the pair that make a conversation
spanning midnight followable: a day records what it left hanging, and the next
day — which receives the previous week of summaries as context — records what
it picked back up.

## Build order

Summaries land first — they are the cheaper bet and the thing that is
actually missing. The vector path is unblocked after, not deleted.

1. **Summaries table + generation job.** Schema above, a
   `summarize_day(date, channel_id)` that returns prose + facets, the
   watermark loop, and a backfill that runs it over existing history.

2. **`read_summaries` tool.** One new entry in `TOOLS` taking a date range and
   optional channel, returning summaries for that span. This is where it
   becomes clear whether the summaries are good enough to retrieve on.

3. ~~**Stop the vector path lying.**~~ **Superseded (see Addendum):** the
   per-message vector path was removed outright — `similarity_search` and the
   planner's tool for it no longer exist, which is the stronger form of "a
   dead tool must never look like an empty result."

4. ~~**Finish the vector path.**~~ **Superseded (see Addendum):** embeddings
   now live on clusters, not messages, and the embed job exists
   (`summarize.embed_pending`, batched, resumable, driven off
   `embedding IS NULL`). The model is pinned: `gemini-embedding-2` at
   `EMBED_DIM` 768, cosine, normalized in `llm.embed_texts`.

5. **Report match counts.** Every search in `_execute_call` caps at 30 rows
   (`MAX_ROWS_PER_CALL`) with no signal that more existed. Run `COUNT(*)`
   alongside each query and include `matched` vs `returned` in what
   `_render_results` sends back.

6. **Validate citation ids.** `Ledger.apply` checks only that `citations` is a
   non-empty dict, never that the ids are real. Check against `messages` and
   reject on miss (this is what makes decision 2 enforceable).

Steps 3, 5, and 6 are independent of everything else and can land in any
order. Step 4 depends only on step 3 being the honest fallback in the
meantime.

## Open

- ~~**Timezone for day boundaries.**~~ **Settled: US Eastern.** `CORPUS_TZ` in
  `db.py` (`America/New_York`, env-overridable) is the single clock the corpus
  is stated in — it cuts the summary day *and* derives `hour_of_day` /
  `day_of_week` at ingest, which were previously UTC despite the schema
  claiming otherwise. `America/New_York` rather than a fixed -5 EST, so the
  buckets track DST the way the group's clocks do. Changing it later
  invalidates every stored bucket and every summary, since the boundaries
  themselves move.

- ~~**Which embedding model.**~~ **Settled: `gemini-embedding-001` at 768
  dimensions, cosine** (`llm.EMBED_MODEL` / `db.EMBED_DIM`). Changing either
  means re-embedding every stored cluster vector.

- ~~**Whether to embed summaries too.**~~ **Settled, in a stronger form (see
  Addendum):** LLM-written cluster summaries are the *only* thing embedded —
  the per-message vector path was removed rather than finished.

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
- **Aggregation tools** → `db.py` has `message_counts_by_author`, unexposed in
  `TOOLS`, which is why `PLANNER_PROMPT` says "you cannot count." Exposing it
  (plus per-day counts stored at summary time) removes that restriction. Note
  it is now the only aggregation function: `category_rate_by_author` was
  removed with the `category` column on 2026-08-16, so the numerator side
  needs a basis that exists in a row — a keyword match or a time bucket.
- **Alias-expanded keyword search** → `aliases_observed` accumulates enough
  that exact match is demonstrably missing people by nickname.

## Addendum — cluster embeddings (2026-08-16)

The per-message vector path was removed before it ever ran, and replaced with
embeddings over **topical clusters**. What changed and why:

1. **The daily summarize call also cuts clusters.** The same Gemini call that
   writes a day's prose + facets now returns `clusters`: contiguous stretches
   of the day's messages, split where the topic significantly changed, each
   with a few-word `topic`, a 1–3 sentence anchor-dense `summary`, and its
   boundary message ids **copied from the input lines** (never counted —
   models copy reliably and count badly). Code validates that the clusters
   exactly partition the input: ids real, in order, no gaps, no overlaps.

2. **Cluster summaries are what gets embedded**, into a dedicated `clusters`
   table (`db.py`) — not the messages, not the day prose. `messages.embedding`
   and `similarity_search` are gone; the planner keeps structured/keyword/
   anchor tools only. A cluster-similarity read path over the new table is a
   later, separate change (the HNSW index for it already exists).

3. **Generation and embedding are separate stages.** Clusters are stored with
   `embedding IS NULL`; `summarize.embed_pending()` fills every NULL in
   batches after each run. An embedding outage costs nothing — the NULL is the
   retry flag. Model pinned: `gemini-embedding-2`, 768 dims, cosine,
   normalized (`llm.embed_texts`); embed spend is recorded in `api_usage`.
   Task instructions are text prefixes (`llm.as_document` / `llm.as_query`),
   not the `task_type` field — this model accepts that field and silently
   ignores it. Each text goes in its own `Content`; passing bare strings
   returns one aggregated vector for the whole batch, verified against the
   live API on 2026-08-16.

4. **Midnight cuts are healed by re-cutting.** Each day's run extends its
   clustering input back to the start of the previous day's *final* cluster
   and replaces it — that cluster was cut blind to how the conversation
   continued, so it is re-drawn with the new day in view and may span
   midnight. Skipped when the previous day failed or had no messages. The day
   *summary* still covers exactly the calendar day; only clusters cross it.

5. **Failures are flagged, not blocking.** A failed day (model call, bad JSON,
   or invalid clusters) is recorded in `summary_failures` and the watermark
   advances past it; every later run retries flagged days before starting new
   ones. This replaces "the watermark never advances over an unsummarized
   day" — the flag is what keeps the gap from becoming permanent. If the
   prose was good but the clusters were not, the summary is stored and the
   day flagged `stage='clusters'`; the retry redoes the whole call (the
   upsert makes that harmless).

6. **Per-message classification was dropped with it.** `messages.category`,
   `sentiment`, and `target_person_id` were specced as ingestion-time labels
   and indexed, but nothing ever wrote them — the same "dead instrument that
   looks alive" problem as the empty vector column, and `specs.md` had already
   flagged their noise on sarcastic chat as unmeasured. Removed, along with
   `set_classification`, `category_rate_by_author`, their three indexes, and
   their entries in the filter whitelist. Facets and cluster summaries carry
   this now, with message ids behind them.

Known, accepted limitation: contiguous ranges cannot represent interleaved
conversations — a stretch that braids two topics becomes one cluster. Ranges
are what make validation and drill-down mechanical; topical purity is not
promised.
