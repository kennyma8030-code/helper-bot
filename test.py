import discord
from discord import app_commands
import asyncio
import contextlib
import json
import os
from llm import ask_gemini
from prompts import CONVO_PROMPT, KICKOFF_PROMPT, fill_prompt
import random
from collections import defaultdict
import time

NUMBER_OF_BOTS = 5

# Bots are identified by number only: 1_bot, 2_bot, ...
BOT_ROSTER = ", ".join(f"{i}_bot" for i in range(1, NUMBER_OF_BOTS + 1))

intents = discord.Intents.default()
intents.message_content = True

token_vars = [f"DISCORD_BOT_{i}" for i in range(NUMBER_OF_BOTS)]
args = [discord.Client(intents=intents) for _ in range(NUMBER_OF_BOTS)]

async def main():
    async with contextlib.AsyncExitStack() as stack:
        for client in args:
            await stack.enter_async_context(client)

        await asyncio.gather(
            *(client.start(os.environ[var]) for client, var in zip(args, token_vars))
        )

bot_id_map = {
    index: int(value) if (value := os.environ.get(f"{index}_bot")) else None
    for index in range(1, NUMBER_OF_BOTS + 1)
}
ADMIN_USER_ID = int(os.environ["ADMIN_USER_ID"])
GUILD_ID = int(os.environ["GUILD_ID"]) if os.environ.get("GUILD_ID") else None
past_messages = []
bot_context = defaultdict(str)
message_number = 0
prev_res = {}
test_enabled = False

# Slash command lives on bot 1's client; it controls the whole test.
tree = app_commands.CommandTree(args[0])


@tree.command(name="test", description="Turn the bot conversation test on or off (admin only).")
@app_commands.describe(on="True to start (resets the conversation), False to stop")
async def test(interaction: discord.Interaction, on: bool):
    global test_enabled, message_number, prev_res
    if interaction.user.id != ADMIN_USER_ID:
        await interaction.response.send_message("NOT ALOUD!", ephemeral=True)
        return
    test_enabled = on
    if on:
        # Fresh run: back to kickoff state.
        message_number = 0
        past_messages.clear()
        prev_res = {}
        bot_context.clear()
    await interaction.response.send_message(
        f"test is now {'on' if on else 'off'}.", ephemeral=True
    )


@args[0].event
async def on_ready():
    if GUILD_ID:
        guild = discord.Object(id=GUILD_ID)
        tree.copy_global_to(guild=guild)
        await tree.sync(guild=guild)
    else:
        await tree.sync()
    print(f"test controller logged in as {args[0].user}", flush=True)


def random_pick(message):
    # Seeded by the message id: new message -> new value; same message ->
    # same value in every client's handler, so all 5 bots agree on the pick
    # with no shared state to reset.
    return random.Random(message.id).randint(1, NUMBER_OF_BOTS)


def parse_response(text):
    """First parseable JSON object in the model's reply, or None."""
    decoder = json.JSONDecoder()
    for i, ch in enumerate(text):
        if ch == "{":
            try:
                obj, _ = decoder.raw_decode(text[i:])
            except ValueError:
                continue
            if isinstance(obj, dict):
                return obj
    return None


def build_content(index):
    """What the bot receives each turn: last 10 messages + its private notes."""
    return (
        "LAST 10 MESSAGES (oldest first):\n"
        + "\n".join(past_messages[-10:])
        + "\n\nYOUR PRIVATE CONTEXT:\n"
        + (bot_context[index] or "(none yet)")
    )


def handler_helper(client, index):
    @client.event
    async def on_message(message):
        global message_number
        global prev_res

        if not test_enabled:
            return

        if message_number == 0:
            # Kickoff: the human's opening message; bot 1 answers, others wait.
            if message.author.id != ADMIN_USER_ID or index != 1:
                return
            past_messages.append(f"human: {message.content}")
            prompt = fill_prompt(
                KICKOFF_PROMPT,
                bot_name="1_bot", bot_number=1,
                num_bots=NUMBER_OF_BOTS, bot_roster=BOT_ROSTER,
            )
        else:
            # Ignore my own messages.
            if message.author.id == bot_id_map[index]:
                return
            routing = prev_res.get("respond_to") or {}
            if routing.get("0"):
                return
            if routing.get(str(NUMBER_OF_BOTS + 1)):
                if random_pick(message) != index:
                    return
            elif not routing.get(str(index)):
                return
            prompt = fill_prompt(
                CONVO_PROMPT,
                bot_name=f"{index}_bot", bot_number=index,
                num_bots=NUMBER_OF_BOTS, bot_roster=BOT_ROSTER,
            )

        await asyncio.sleep(3)
        raw = await ask_gemini(prompt, build_content(index), web_search=False)
        parsed = parse_response(raw)
        if parsed is None or not parsed.get("message"):
            print(f"[bot {index}] unparseable response, skipping: {raw!r}", flush=True)
            return

        await message.channel.send(parsed["message"])

        # Store the response: routing for the next turn, the message into the
        # rolling window, and this bot's private notes if it rewrote them.
        prev_res = parsed
        past_messages.append(f"{index}_bot: {parsed['message']}")
        message_number += 1
        ctx = parsed.get("bot_context") or {}
        if ctx.get("edit_context"):
            bot_context[index] = str(ctx.get("new_context", ""))[:500]


for i, client in enumerate(args, start=1):
    handler_helper(client, i)


if __name__ == "__main__":
    asyncio.run(main())
