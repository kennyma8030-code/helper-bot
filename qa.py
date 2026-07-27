"""Question-answering logic, independent of Discord.

Owns the persona and classifier prompts, and the full pipeline for handling
a candidate message:

    classify (is it a question?) -> answer -> enforce persona styling

The single entry point is answer_question(): give it message text, get back
either a styled reply or None (not a question / classifier output unusable).
bot.py should not need to know anything beyond that contract. Gemini access
lives in llm.py.
"""

import json

from llm import ask_gemini

PROMPT1 = """you respond to every question in lowercase only, no capitalization ever, and minimal to no punctuation — no periods, no commas unless truly needed for the sentence to parse, no exclamation points, no question marks

the tone is dismissive and completely apathetic — you answer like you could not care less whether the question was asked at all. detached, flat, indifferent. you are not annoyed, not amused, not invested in any direction. you simply provide the fact and disengage

rules for how this reads:
- answer the actual question, correctly, but say the minimum needed and stop
- keep the register measured and formal — proper words, no slang, no abbreviations, never anything like "cuz", "lol", "gonna", "idk", "tbh". write it the way it would appear in plain, correct prose (just without capitals or periods per the style rule)
- do not perform emotion of any kind — no enthusiasm, no snark, no jokes, no warmth. total apathy reads colder than insults
- no filler, no hedging, no "well" or "to be fair" or "honestly" — those imply you care how it lands
- if the question is trivial, do not remark on it — simply answer it and stop; the flatness carries the dismissiveness
- never insult the person, never explain the tone, never acknowledge you are being dismissive, never apologize
- the vibe is someone stating a fact they have no stake in and immediately moving on

length: as short as the answer allows. one sentence is usually plenty. do not pad it out. the response must always be under 2000 characters, no exceptions.

respond only with the answer itself, nothing else"""

PROMPT2 = """You classify whether a message is a question.

A message counts as a question if the sender is asking for information, clarification, confirmation, or a response — even if it lacks a question mark or standard question grammar. This includes:
- Direct questions ("what time is it")
- Indirect/implied questions ("i wonder if this thing works")
- Rhetorical-sounding but genuinely inquisitive messages ("so this is really how it works?")
- Requests phrased as statements ("tell me the score")

It does NOT count as a question if it's:
- A statement, observation, or opinion, even if it ends in "?" for emphasis ("that's crazy?")
- A greeting, reaction, or exclamation
- A command with no informational ask ("stop that")

Respond with only a JSON object in this exact format, no other text:
{"is_question": true or false}
"""


def style_response(text: str) -> str:
    """Enforce the persona in code: all lowercase, no periods (commas instead)."""
    styled = text.lower().replace(".", ",")
    # Avoid a dangling comma left where a sentence-ending period was.
    return styled.rstrip(", ").rstrip()


async def is_question(message: str) -> bool | None:
    """Classify a message. Returns True/False, or None if the classifier
    response wasn't valid JSON."""
    raw = await ask_gemini(PROMPT2, message)
    print(f"[is_question] classifier raw response: {raw!r}", flush=True)
    try:
        result = json.loads(raw)
    except json.JSONDecodeError:
        print("[is_question] classifier did not return valid JSON", flush=True)
        return None
    print(f"[is_question] parsed result: {result}", flush=True)
    return bool(result["is_question"])


async def answer_question(content: str, classify: bool = True) -> str | None:
    """Full pipeline for one message: classify, answer, style.

    Returns the styled reply, or None if the message shouldn't be answered
    (not a question, or the classifier output was unusable).

    classify=False skips the is_question gate and answers unconditionally —
    for callers where a human already decided this message deserves a reply.
    """
    if classify:
        verdict = await is_question(content)
        if not verdict:
            print("[answer_question] not treated as a question, no reply", flush=True)
            return None
        print("[answer_question] is a question, asking gemini for answer...", flush=True)
    answer = await ask_gemini(PROMPT1, content)
    print(f"[answer_question] gemini answer: {answer!r}", flush=True)
    answer = style_response(answer)
    print(f"[answer_question] styled answer: {answer!r}", flush=True)
    return answer
