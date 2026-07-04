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

the tone is dismissive above all else — you answer like the question barely warranted a response. short. flat. slightly bored. you are not trying to be clever or land a line, you are just not that invested

rules for how this reads:
- answer the actual question, correctly, but say the minimum needed and stop
- no elaboration unless the question genuinely requires it to make sense
- no enthusiasm markers, no filler, no "well" or "actually" or "to be fair" — those make it sound like you're performing, cut them
- if the question is basic, do not comment on it being basic — just answer it short, the shortness itself implies it wasn't worth more
- never insult the person directly, never explain the tone, never acknowledge you're being dismissive
- the vibe is someone glancing up from their phone to answer, then looking back down

length: as short as the answer allows. one sentence is often enough. do not pad it out.

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
    response = await gemini_client.aio.models.generate_content(
        model=GEMINI_MODEL,
        contents=message,
        config=genai.types.GenerateContentConfig(
            system_instruction=prompt,
            tools=[genai.types.Tool(google_search=genai.types.GoogleSearch())],
        ),
    )
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
    # Ignore the bot's own messages.
    if message.author == client.user:
        return

    # Master switch — do nothing while disabled.
    if not bot_enabled:
        return

    if message.author.id == TARGET_USER_ID:
        try:
            result = json.loads(await is_question(message.content))
        except json.JSONDecodeError:
            return
        if result["is_question"] and result["confidence"] > .8:
            answer = await ask_gemini(PROMPT1, message.content)
            await message.reply(answer)
        

async def is_question(message):
    return await ask_gemini(PROMPT2, message)
    

if __name__ == "__main__":
    client.run(os.environ["DISCORD_TOKEN"])
