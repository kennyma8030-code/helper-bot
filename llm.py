"""Shared Gemini access — one client, one call helper, reused by every module.

qa.py (persona) and loop.py (agent) both talk to Gemini; this module owns the
client so there is a single place for API-key handling, model defaults, and
call logging.
"""

import logging
import os

from dotenv import load_dotenv
from google import genai

import db

load_dotenv()

log = logging.getLogger(__name__)

# One client for the whole process. Reads GEMINI_API_KEY from the environment.
client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

DEFAULT_MODEL = "gemini-3.1-flash-lite"

# USD per 1,000,000 tokens, (input, output), standard paid tier, text input.
# From https://ai.google.dev/gemini-api/docs/pricing — checked 2026-08-15.
# These are the published list prices, so re-check them if a model is swapped
# or Google changes the rate; nothing in the code can notice a stale number.
MODEL_PRICES = {
    "gemini-3.1-flash-lite": (0.25, 1.50),
    "gemini-3.5-flash": (1.50, 9.00),
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
