# Day Summaries — Spec

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
    embedded") rather than an empty row list.

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

3. **Stop the vector path lying.** Cheap and independent of everything else:
   implement decision 10, so `similarity_search` over an unembedded corpus
   reports its coverage instead of returning an empty row list the model reads
   as "nothing matched."

4. **Finish the vector path.** Everything is in place except the one thing
   that makes it work: `set_embedding` (`db.py:321`) has no caller. Write the
   embedding job — a backfill over existing history plus an ongoing pass for
   new messages, batched, resumable, and driven off `embedding IS NULL` so it
   is safe to re-run. Two things must be settled first, because changing
   either invalidates every stored vector: pin the embedding model (`EMBED_DIM`
   at `db.py:50` is provisional at 768 and must match the chosen model's real
   output width, which `_embed_query` in `loop.py:132` already requests via
   `output_dimensionality`), and confirm `DISTANCE_OP` (`db.py:54`, currently
   cosine `<=>`) matches how that model normalizes.

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

- **Which embedding model.** Blocks build order step 4 and nothing else.
  `EMBED_DIM` (`db.py:50`) is a provisional 768 and `DISTANCE_OP` (`db.py:54`)
  a provisional cosine; both are properties of the model, not choices to make
  independently. Pinning this late is fine — pinning it twice is not, since
  the second choice means re-embedding the whole corpus.

- **Whether to embed summaries too.** `prose` is exactly the kind of text
  embeddings are good at, and a vector index over summaries would be a
  cheaper, coarser semantic search than one over every message. Not decided;
  revisit once step 4 has run and both paths are real.

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
