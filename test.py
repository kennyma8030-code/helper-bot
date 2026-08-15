import discord
from discord import app_commands
from discord.ext import tasks
import asyncio
import contextlib
import json
import os
from llm import ask_gemini
from prompts import CONVO_PROMPT, KICKOFF_PROMPT, OPENER_PROMPT, fill_prompt
import random
from collections import defaultdict
import time

NUMBER_OF_BOTS = 5

# Scheduled conversations: every SESSION_EVERY_HOURS, the bots talk among
# themselves for SESSION_MINUTES and then go quiet until the next one.
SESSION_EVERY_HOURS = 4
SESSION_MINUTES = 5

# Where the scheduled conversation happens. There is no triggering message to
# reply to, so the channel has to be named outright. Unset means the timer does
# nothing and /test still works by hand.
TEST_CHANNEL_ID = (
    int(os.environ["TEST_CHANNEL_ID"]) if os.environ.get("TEST_CHANNEL_ID") else None
)

# One of these is drawn per session and handed to bot 1 as its opening subject.
# Picking in code rather than asking the model for "something random" is what
# keeps sessions from all sounding alike — models reach for the same handful of
# ideas when left to invent a topic. Edit this list freely.
TOPICS = [
    "the correct way to eat a burrito",
    "whether cereal counts as soup",
    "the worst haircut you've ever had",
    "airport food, ranked",
    "people who reply 'k'",
    "the best sound in the world",
    "whether hot dogs are sandwiches",
    "an unreasonably strong opinion about socks",
    "the last thing that made you laugh out loud",
    "elevator etiquette",
    "songs that are objectively too long",
    "the ideal number of pillows",
    "what you'd name a boat",
    "grocery store self-checkout",
    "the most overrated snack",
    "whether you'd survive in the wilderness",
    "an animal that could take you in a fight",
    "the tyranny of group photos",
    "food you loved as a kid and can't stand now",
    "people who stand up the second the plane lands",
    "the best excuse for leaving a party early",
    "whether a straw has one hole or two",
    "your position on pineapple, generally",
    "the most useless thing you own",
]

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
sleep_seconds = 3.0

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


@tree.command(name="sleep", description="Set the delay before each bot reply, in seconds (admin only).")
@app_commands.describe(seconds="Delay in seconds (e.g. 3, 10, 0 for none)")
async def sleep(interaction: discord.Interaction, seconds: float):
    global sleep_seconds
    if interaction.user.id != ADMIN_USER_ID:
        await interaction.response.send_message("NOT ALOUD!", ephemeral=True)
        return
    if seconds < 0:
        await interaction.response.send_message("delay can't be negative.", ephemeral=True)
        return
    sleep_seconds = seconds
    await interaction.response.send_message(
        f"reply delay is now {seconds:g}s.", ephemeral=True
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

    # on_ready fires again on every reconnect; is_running() keeps a flaky
    # connection from stacking up a second timer on top of the first.
    if not scheduled_session.is_running():
        scheduled_session.start()


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


async def _post_opener(channel):
    """Bot 1 opens a conversation on a random topic. True if the message landed.

    This is the scheduled stand-in for the human kickoff message: same JSON
    contract, same routing, so bots 2..N carry on with the existing handler and
    never learn the conversation started itself.
    """
    topic = random.choice(TOPICS)
    print(f"[session] topic: {topic}", flush=True)

    prompt = fill_prompt(
        OPENER_PROMPT,
        bot_name="1_bot", bot_number=1,
        num_bots=NUMBER_OF_BOTS, bot_roster=BOT_ROSTER,
        topic=topic,
    )
    raw = await ask_gemini(prompt, "The chat is empty. Say the first thing.",
                           web_search=False)
    parsed = parse_response(raw)
    if parsed is None or not parsed.get("message"):
        print(f"[session] opener unparseable, skipping run: {raw!r}", flush=True)
        return False

    # State before send, for the reason in handler_helper: the gateway can
    # deliver this message to the other four clients before send() returns, and
    # they must already see message_number > 0 or they will read it as a kickoff.
    global message_number, prev_res
    prev_res = parsed
    past_messages.append(f"1_bot: {parsed['message']}")
    message_number += 1
    ctx = parsed.get("bot_context") or {}
    if ctx.get("edit_context"):
        bot_context[1] = str(ctx.get("new_context", ""))[:500]

    await channel.send(parsed["message"])
    print(f"[session] opener sent; respond_to={parsed.get('respond_to')}", flush=True)
    return True


@tasks.loop(hours=SESSION_EVERY_HOURS)
async def scheduled_session():
    """Run one timed conversation, then switch the bots off until the next."""
    global test_enabled, message_number, prev_res

    if TEST_CHANNEL_ID is None:
        print("[session] TEST_CHANNEL_ID is not set; nothing to do", flush=True)
        return
    if test_enabled:
        # A manual /test run, or an overrunning previous session. Either way,
        # starting now would clear state out from under a live conversation.
        print("[session] a conversation is already running; skipping", flush=True)
        return

    channel = args[0].get_channel(TEST_CHANNEL_ID)
    if channel is None:
        print(f"[session] channel {TEST_CHANNEL_ID} not visible to bot 1; skipping",
              flush=True)
        return

    # Same reset /test on performs: every session is a fresh conversation.
    message_number = 0
    past_messages.clear()
    prev_res = {}
    bot_context.clear()
    test_enabled = True

    print(f"[session] starting a {SESSION_MINUTES}-minute conversation", flush=True)
    if not await _post_opener(channel):
        test_enabled = False
        return

    try:
        await asyncio.sleep(SESSION_MINUTES * 60)
    finally:
        # Off even if the wait is cancelled, so a reload can never leave the
        # bots talking forever.
        test_enabled = False
    print("[session] finished", flush=True)


@scheduled_session.before_loop
async def _before_session():
    # get_channel reads the client's cache, which is empty until the gateway
    # has finished handing over the guilds.
    await args[0].wait_until_ready()


def handler_helper(client, index):
    @client.event
    async def on_message(message):
        global message_number
        global prev_res

        if not test_enabled:
            return

        print(f"[bot {index}] saw msg {message.id} from {message.author.id} "
              f"(message_number={message_number}): {message.content[:80]!r}", flush=True)

        if message_number == 0:
            # Kickoff: the human's opening message; bot 1 answers, others wait.
            if message.author.id != ADMIN_USER_ID or index != 1:
                print(f"[bot {index}] kickoff: not admin+bot1, ignoring", flush=True)
                return
            print(f"[bot {index}] kickoff: responding to human", flush=True)
            past_messages.append(f"human: {message.content}")
            prompt = fill_prompt(
                KICKOFF_PROMPT,
                bot_name="1_bot", bot_number=1,
                num_bots=NUMBER_OF_BOTS, bot_roster=BOT_ROSTER,
            )
        else:
            # Ignore my own messages.
            if message.author.id == bot_id_map[index]:
                print(f"[bot {index}] own message, ignoring", flush=True)
                return
            routing = prev_res.get("respond_to") or {}
            if routing.get("0"):
                print(f"[bot {index}] routing says nobody replies, ignoring", flush=True)
                return
            if routing.get(str(NUMBER_OF_BOTS + 1)):
                pick = random_pick(message)
                if pick != index:
                    print(f"[bot {index}] random mode picked {pick}, not me, ignoring", flush=True)
                    return
                print(f"[bot {index}] random mode picked me", flush=True)
            elif not routing.get(str(index)):
                print(f"[bot {index}] not routed to me (routing={routing}), ignoring", flush=True)
                return
            else:
                print(f"[bot {index}] routed to me, responding", flush=True)
            prompt = fill_prompt(
                CONVO_PROMPT,
                bot_name=f"{index}_bot", bot_number=index,
                num_bots=NUMBER_OF_BOTS, bot_roster=BOT_ROSTER,
            )

        await asyncio.sleep(sleep_seconds)
        raw = await ask_gemini(prompt, build_content(index), web_search=False)
        parsed = parse_response(raw)
        if parsed is None or not parsed.get("message"):
            print(f"[bot {index}] unparseable response, skipping: {raw!r}", flush=True)
            return
        # The session can end while a reply is being generated. Stop before
        # touching shared state: the next session has already cleared it, and
        # writing now would leak this conversation into that one.
        if not test_enabled:
            print(f"[bot {index}] session ended mid-reply, dropping it", flush=True)
            return
        print(f"[bot {index}] sending reply; respond_to={parsed.get('respond_to')}", flush=True)

        # Store the response BEFORE sending: the gateway can deliver our sent
        # message to the other clients before send() returns, and they must
        # see the new routing/state when it arrives, not the previous turn's.
        prev_res = parsed
        past_messages.append(f"{index}_bot: {parsed['message']}")
        message_number += 1
        ctx = parsed.get("bot_context") or {}
        if ctx.get("edit_context"):
            bot_context[index] = str(ctx.get("new_context", ""))[:500]

        await message.channel.send(parsed["message"])


for i, client in enumerate(args, start=1):
    handler_helper(client, i)


if __name__ == "__main__":
    asyncio.run(main())
