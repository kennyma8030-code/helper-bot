"""Shared Gemini access — one client, one call helper, reused by every module.

qa.py (persona) and loop.py (agent) both talk to Gemini; this module owns the
client so there is a single place for API-key handling, model defaults, and
call logging.
"""

import logging
import math
import os

from dotenv import load_dotenv
from google import genai

import db

load_dotenv()

log = logging.getLogger(__name__)

# One client for the whole process. Reads GEMINI_API_KEY from the environment.
client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

DEFAULT_MODEL = "gemini-3.1-flash-lite"

# The one embedding model, pinned. db.EMBED_DIM is the width this model is
# asked for; changing either means re-embedding every stored cluster vector,
# so treat them as a pair. 768 is one of this model's recommended widths, so
# the column width survived the move off gemini-embedding-001 — but the vectors
# did not, since two models' vectors are not comparable to each other.
EMBED_MODEL = "gemini-embedding-2"

# USD per 1,000,000 tokens, (input, output), standard paid tier, text input.
# From https://ai.google.dev/gemini-api/docs/pricing — checked 2026-08-16.
# These are the published list prices, so re-check them if a model is swapped
# or Google changes the rate; nothing in the code can notice a stale number.
# Embedding calls have no output tokens, hence the 0.
#
# gemini-embedding-001 stays listed after the move to -2: it prices nothing new,
# but the rate that produced the existing api_usage rows belongs next to them.
MODEL_PRICES = {
    "gemini-3.1-flash-lite": (0.25, 1.50),
    "gemini-3.5-flash": (1.50, 9.00),
    "gemini-embedding-001": (0.15, 0.0),
    "gemini-embedding-2": (0.20, 0.0),
}

# Models seen with no price entry, so the warning is logged once each rather
# than on every call.
_unpriced_warned: set[str] = set()


def estimate_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    """USD for one call, from token counts and the published price.

    An unknown model costs 0 and warns. Guessing a price would be worse: a made
    up number silently moves a spend cap in whichever direction it was wrong.
    """
    price = MODEL_PRICES.get(model)
    if price is None:
        if model not in _unpriced_warned:
            _unpriced_warned.add(model)
            log.warning(
                "no price for model %r; its calls count as $0 toward the budget. "
                "Add it to llm.MODEL_PRICES.", model,
            )
        return 0.0
    in_price, out_price = price
    return (input_tokens * in_price + output_tokens * out_price) / 1_000_000


async def track_usage(model: str, response) -> None:
    """Record what one response cost. Never raises.

    Best-effort on purpose: this runs after every model call in both bots, and
    a database that is down or absent must not turn a working reply into an
    error. A failure here means the call goes uncounted, which the budget check
    treats as the safety problem it is — see test.scheduled_session.
    """
    usage = getattr(response, "usage_metadata", None)
    if usage is None:
        return

    input_tokens = getattr(usage, "prompt_token_count", 0) or 0
    total = getattr(usage, "total_token_count", 0) or 0
    # Everything that is not input is billed at the output rate, thinking
    # tokens included — which candidates_token_count alone leaves out.
    output_tokens = max(total - input_tokens, 0) or (
        getattr(usage, "candidates_token_count", 0) or 0
    )

    try:
        await db.open_pool()
        await db.record_usage(
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=estimate_cost(model, input_tokens, output_tokens),
        )
    except Exception as e:
        log.warning("could not record api usage (%s: %s)", type(e).__name__, e)


def _normalize(vector: list[float]) -> list[float]:
    """Scale a vector to unit length.

    gemini-embedding-2 normalizes its own truncated output, so this is a no-op
    on everything it currently returns. Kept as the guarantee the column
    actually depends on: DISTANCE_OP is cosine, and an operator reading the
    table should be able to assume unit vectors without trusting a model's
    release notes. Re-normalizing a unit vector costs nothing.
    """
    norm = math.sqrt(sum(x * x for x in vector))
    return [x / norm for x in vector] if norm else vector


def as_document(text: str, title: str | None = None) -> str:
    """Prefix a text for indexing. See as_query for why the prefixes exist."""
    return f"title: {title or 'none'} | text: {text}"


def as_query(text: str, task: str = "search result") -> str:
    """Prefix a text for searching.

    gemini-embedding-2 takes its task instruction as part of the text rather
    than as an API field, and the two sides of a retrieval are prefixed
    differently on purpose: the stored summaries are documents, the string
    typed against them is a query. Google's guidance is that applying both
    prefixes measurably improves retrieval, and they only work as a pair —
    a query embedded as a document is compared against the wrong shape.
    """
    return f"task: {task} | query: {text}"


async def embed_texts(
    texts: list[str],
    *,
    is_query: bool = False,
) -> list[list[float]]:
    """Embed a batch of texts; returns one unit vector per text, in order.

    Defaults to the document side, since indexing cluster summaries is what
    calls this. A search path embedding what a user typed passes is_query=True.
    """
    prepared = [as_query(t) if is_query else as_document(t) for t in texts]

    response = await client.aio.models.embed_content(
        model=EMBED_MODEL,
        # One Content per text, which is what makes this a batch of separate
        # embeddings. Handing the same list as bare strings is a documented
        # and silent trap: this model reads that as one multimodal input and
        # returns a single aggregated vector for the whole batch.
        contents=[
            genai.types.Content(parts=[genai.types.Part.from_text(text=t)])
            for t in prepared
        ],
        config=genai.types.EmbedContentConfig(
            # No task_type. The docs say the field cannot be used with this
            # model, but the API accepts it and ignores it rather than
            # erroring — so leaving the old value in place would have looked
            # like it still worked while doing nothing. The instruction it
            # used to carry lives in the prefixes above now.
            output_dimensionality=db.EMBED_DIM,
        ),
    )
    vectors = [_normalize(list(e.values)) for e in response.embeddings]
    if len(vectors) != len(texts):
        raise ValueError(
            f"embedding API returned {len(vectors)} vectors for {len(texts)} texts"
        )

    # Spend tracking. Embed responses do not reliably carry usage metadata, so
    # fall back to a chars/4 token estimate — a rough number recorded against
    # the budget beats a real cost recorded as zero.
    usage = getattr(response, "usage_metadata", None)
    if usage is not None and getattr(usage, "prompt_token_count", None):
        await track_usage(EMBED_MODEL, response)
    else:
        # `prepared`, not `texts`: the prefixes are billed too.
        estimated = sum(len(t) for t in prepared) // 4
        try:
            await db.open_pool()
            await db.record_usage(
                model=EMBED_MODEL,
                input_tokens=estimated,
                output_tokens=0,
                cost_usd=estimate_cost(EMBED_MODEL, estimated, 0),
            )
        except Exception as e:
            log.warning("could not record embedding usage (%s: %s)",
                        type(e).__name__, e)

    return vectors


async def ask_gemini(
    prompt: str,
    message: str,
    *,
    model: str = DEFAULT_MODEL,
    web_search: bool = True,
) -> str:
    """Send `message` to Gemini under system prompt `prompt`; return the text.

    web_search=True attaches the Google Search grounding tool (the persona
    bot's behavior). Callers doing function calling should use `client`
    directly — this helper is for plain prompt->text calls.
    """
    print(f"[ask_gemini] calling Gemini, message={message!r}", flush=True)
    tools = (
        [genai.types.Tool(google_search=genai.types.GoogleSearch())]
        if web_search
        else None
    )
    try:
        response = await client.aio.models.generate_content(
            model=model,
            contents=message,
            config=genai.types.GenerateContentConfig(
                system_instruction=prompt,
                tools=tools,
            ),
        )
    except Exception as e:
        print(f"[ask_gemini] ERROR calling Gemini: {type(e).__name__}: {e}", flush=True)
        raise
    await track_usage(model, response)
    print(f"[ask_gemini] Gemini returned: {response.text!r}", flush=True)
    return response.text or ""
