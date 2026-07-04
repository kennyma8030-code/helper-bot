import os

import discord
from discord import app_commands
from dotenv import load_dotenv
from google import genai
import json

load_dotenv()

# message_content is a privileged intent — enable it in the Discord
# Developer Portal (Bot > Privileged Gateway Intents) as well.
intents = discord.Intents.default()
intents.message_content = True

client = discord.Client(intents=intents)
tree = app_commands.CommandTree(client)

# The user whose messages the bot should answer. Store the ID, not the name.
TARGET_USER_ID = int(os.environ["TARGET_USER_ID"])

# Only this user may run the /power command.
ADMIN_USER_ID = int(os.environ["ADMIN_USER_ID"])

# Optional: a guild ID to sync commands to instantly (dev). If unset, commands
# sync globally, which can take up to an hour to appear.
GUILD_ID = int(os.environ["GUILD_ID"]) if os.environ.get("GUILD_ID") else None

# Master on/off switch, toggled by /power.
bot_enabled = True

# Gemini client. Reads GEMINI_API_KEY from the environment.
gemini_client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
GEMINI_MODEL = "gemini-2.5-flash"

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

length: as short as the answer allows. one sentence is usually plenty. do not pad it out.

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
{"is_question": true or false, "confidence": a number between 0 and 1}
"""

async def ask_gemini(prompt: str, message: str) -> str:
    """Send `message` to Gemini under the given `prompt` and return the reply."""
    print(f"[ask_gemini] calling Gemini, message={message!r}", flush=True)
    try:
        response = await gemini_client.aio.models.generate_content(
            model=GEMINI_MODEL,
            contents=message,
            config=genai.types.GenerateContentConfig(
                system_instruction=prompt,
                tools=[genai.types.Tool(google_search=genai.types.GoogleSearch())],
            ),
        )
    except Exception as e:
        print(f"[ask_gemini] ERROR calling Gemini: {type(e).__name__}: {e}", flush=True)
        raise
    print(f"[ask_gemini] Gemini returned: {response.text!r}", flush=True)
    return response.text or ""


@tree.command(name="power", description="Turn the bot on or off (admin only).")
@app_commands.describe(on="True to enable, False to disable")
async def power(interaction: discord.Interaction, on: bool):
    global bot_enabled
    if interaction.user.id != ADMIN_USER_ID:
        await interaction.response.send_message(
            "NOT ALOUD!", ephemeral=True
        )
        return
    bot_enabled = on
    await interaction.response.send_message(
        f"Bot is now {'on' if on else 'off'}.", ephemeral=True
    )


@client.event
async def on_ready():
    if GUILD_ID:
        guild = discord.Object(id=GUILD_ID)
        tree.copy_global_to(guild=guild)
        await tree.sync(guild=guild)
    else:
        await tree.sync()
    print(f"Logged in as {client.user}")


@client.event
async def on_message(message):
    print(f"[on_message] from {message.author} ({message.author.id}): {message.content!r}", flush=True)

    # Ignore the bot's own messages.
    if message.author == client.user:
        print("[on_message] ignoring: own message", flush=True)
        return

    # Master switch — do nothing while disabled.
    if not bot_enabled:
        print("[on_message] ignoring: bot disabled", flush=True)
        return

    if message.author.id != TARGET_USER_ID:
        print(f"[on_message] ignoring: not target user (target={TARGET_USER_ID})", flush=True)
        return

    print("[on_message] target user matched, classifying...", flush=True)
    raw = await is_question(message.content)
    print(f"[on_message] classifier raw response: {raw!r}", flush=True)
    try:
        result = json.loads(raw)
    except json.JSONDecodeError:
        print("[on_message] aborting: classifier did not return valid JSON", flush=True)
        return
    print(f"[on_message] parsed result: {result}", flush=True)
    if result["is_question"] and result["confidence"] > .8:
        print("[on_message] is a question, asking gemini for answer...", flush=True)
        answer = await ask_gemini(PROMPT1, message.content)
        print(f"[on_message] gemini answer: {answer!r}", flush=True)
        answer = style_response(answer)
        print(f"[on_message] styled answer: {answer!r}", flush=True)
        await message.reply(answer)
        print("[on_message] reply sent", flush=True)
    else:
        print("[on_message] not treated as a question, no reply", flush=True)


def style_response(text: str) -> str:
    """Enforce the persona in code: all lowercase, no periods (commas instead)."""
    styled = text.lower().replace(".", ",")
    # Avoid a dangling comma left where a sentence-ending period was.
    return styled.rstrip(", ").rstrip()


async def is_question(message):
    return await ask_gemini(PROMPT2, message)
    

if __name__ == "__main__":
    client.run(os.environ["DISCORD_TOKEN"])
