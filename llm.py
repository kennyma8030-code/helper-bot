"""Shared Gemini access — one client, one call helper, reused by every module.

qa.py (persona) and loop.py (agent) both talk to Gemini; this module owns the
client so there is a single place for API-key handling, model defaults, and
call logging.
"""

import logging
import os

from dotenv import load_dotenv
from google import genai

load_dotenv()

log = logging.getLogger(__name__)

# One client for the whole process. Reads GEMINI_API_KEY from the environment.
client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

DEFAULT_MODEL = "gemini-3.5-flash"


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
    print(f"[ask_gemini] Gemini returned: {response.text!r}", flush=True)
    return response.text or ""
