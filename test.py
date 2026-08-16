import discord
from discord import app_commands
from discord.ext import tasks
import asyncio
import contextlib
import json
import os
import db
from llm import ask_gemini
from prompts import CONVO_PROMPT, OPENER_PROMPT, fill_prompt
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

# Ceiling on estimated Gemini spend, in USD. Once the recorded total crosses
# it the scheduled conversation stops for good — it is the only thing here that
# spends money without anyone asking it to, so it is the thing with a budget.
#
# Estimated, not billed: Google has no spend endpoint, so this is tokens the
# API reported multiplied by the published price (llm.MODEL_PRICES). Treat it
# as close, not exact, and leave headroom under a real billing limit.
GEMINI_BUDGET_USD = float(os.environ.get("GEMINI_BUDGET_USD", "3"))

# The only subjects the bots talk about. One is drawn per session, handed to
# bot 1 as its opening subject, and then held for the whole conversation — the
# other four are told to steer back to it rather than follow a tangent
# somewhere else.
#
# Picking in code rather than asking the model for "something random" is what
# keeps sessions from all sounding alike — models reach for the same handful of
# ideas when left to invent a topic. Edit this list freely.
TOPICS = [
    "hiking",
    "piano",
    "soccer",
    "aquascaping",
    "classical music",
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

# The subject drawn for the conversation that is running now. Set when the
# opener is posted and read by every bot's handler for the rest of the session,
# which is what keeps all five on one subject instead of five.
current_topic = ""

# Whether the 4-hour schedule should be running at all — /schedule writes it.
# Kept in the settings table so it outlives the container: a switch that came
# back on after every deploy would be a switch you cannot trust to stay off.
# The key is prefixed because the RAG bot stores its own switches in that same
# table under "bot" and "RAG".
SCHEDULE_SWITCH_KEY = "test_schedule"
schedule_enabled = True

# Slash command lives on bot 1's client; it controls the whole test.
tree = app_commands.CommandTree(args[0])


@tree.command(name="test", description="Start or stop the bot conversation (admin only).")
@app_commands.describe(on="True to start a conversation here, False to stop")
async def test(interaction: discord.Interaction, on: bool):
    global test_enabled, message_number, prev_res
    if interaction.user.id != ADMIN_USER_ID:
        await interaction.response.send_message("NOT ALOUD!", ephemeral=True)
        return

    if not on:
        test_enabled = False
        await interaction.response.send_message("test is now off.", ephemeral=True)
        return

    channel = interaction.channel
    if not isinstance(channel, discord.abc.Messageable):
        await interaction.response.send_message(
            "Nothing can be posted here; run it from a text channel.", ephemeral=True
        )
        return

    # Writing the opener is a model call, which takes longer than the 3 seconds
    # Discord allows a command to stay silent.
    await interaction.response.defer(thinking=True, ephemeral=True)

    # Fresh run: every conversation starts from nothing.
    message_number = 0
    past_messages.clear()
    prev_res = {}
    bot_context.clear()
    test_enabled = True

    # Bot 1 opens on its own rather than waiting to be spoken to. The same call
    # the scheduled session makes, so a hand-started conversation and a timed
    # one are the same thing from the other four bots' side.
    if not await _post_opener(channel):
        test_enabled = False
        await interaction.edit_original_response(
            content="Couldn't write an opener — the model's reply didn't parse. "
                    "Nothing started; check the logs and try again."
        )
        return

    await interaction.edit_original_response(
        content=f"test is now on. Talking about: {current_topic}"
    )


@tree.command(
    name="schedule",
    description="Turn the 4-hour scheduled conversation on or off (admin only).",
)
@app_commands.describe(on="True to run a conversation every 4 hours, False to stop")
async def schedule(interaction: discord.Interaction, on: bool):
    global schedule_enabled

    if interaction.user.id != ADMIN_USER_ID:
        await interaction.response.send_message("NOT ALOUD!", ephemeral=True)
        return

    # Saving the switch and reading the spend both go to the database.
    await interaction.response.defer(thinking=True, ephemeral=True)

    schedule_enabled = on
    notes = []
    try:
        await db.open_pool()
        await db.init_db()
        await db.set_switch(SCHEDULE_SWITCH_KEY, on)
    except Exception as e:
        print(f"[schedule] could not save the switch ({type(e).__name__}: {e})",
              flush=True)
        notes.append("Couldn't save it — a redeploy will forget this.")

    if not on:
        # cancel(), not stop(). stop() is only checked after an iteration
        # finishes, so a loop sleeping between conversations would wait out the
        # rest of its four hours and hold one more conversation before noticing
        # it was switched off.
        scheduled_session.cancel()
        notes.append("A conversation running right now stops too.")
        await interaction.edit_original_response(
            content="\n".join(["The 4-hour schedule is **off**.", *notes])
        )
        return

    if TEST_CHANNEL_ID is None:
        notes.append("TEST_CHANNEL_ID isn't set, so runs will do nothing until it is.")

    spent = await _estimated_spend()
    if spent is None:
        notes.append("The database can't be read, so every run will skip itself "
                     "until it can — check DATABASE_URL.")
    elif spent >= GEMINI_BUDGET_USD:
        # Starting it here would only have it stop itself on the first run.
        notes.append(f"Not started: estimated spend is ${spent:.2f} of the "
                     f"${GEMINI_BUDGET_USD:.2f} budget. Raise GEMINI_BUDGET_USD "
                     f"and redeploy.")
        await interaction.edit_original_response(
            content="\n".join(["The 4-hour schedule is **on**.", *notes])
        )
        return
    else:
        notes.append(f"Estimated spend so far: ${spent:.4f} of "
                     f"${GEMINI_BUDGET_USD:.2f}.")

    if scheduled_session.is_running():
        notes.append("It was already running.")
    else:
        # The first iteration fires immediately, not in four hours.
        scheduled_session.start()
        notes.append("A conversation starts now, then every "
                     f"{SESSION_EVERY_HOURS} hours.")

    await interaction.edit_original_response(
        content="\n".join(["The 4-hour schedule is **on**.", *notes])
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


@tree.command(
    name="reset",
    description="Delete every message in this channel and its stored history (admin only).",
)
@app_commands.describe(
    confirm="Must be True. Every message in this channel is deleted and cannot be recovered."
)
async def reset(interaction: discord.Interaction, confirm: bool):
    """Empty one channel in both places it exists: Postgres and Discord."""
    global test_enabled, message_number, prev_res

    if interaction.user.id != ADMIN_USER_ID:
        await interaction.response.send_message("NOT ALOUD!", ephemeral=True)
        return
    if not confirm:
        await interaction.response.send_message(
            "Nothing done. Run it again with confirm:True to actually wipe the channel.",
            ephemeral=True,
        )
        return

    # Forums and categories are channels you cannot post in, and they have no
    # message history to walk. Nothing should route a command here, but the
    # type allows it and the walk below would fail with an AttributeError.
    channel = interaction.channel
    if not isinstance(channel, discord.abc.Messageable):
        await interaction.response.send_message(
            "This channel has no messages to delete; run it from a text channel.",
            ephemeral=True,
        )
        return

    # Walking a channel takes minutes and Discord drops a command that has not
    # replied within 3 seconds. defer() buys 15; a long wipe outlives even that,
    # which is why every step also prints to the log.
    await interaction.response.defer(thinking=True, ephemeral=True)

    # Stop the conversation before deleting anything. Bots that are still
    # talking would post new messages into a channel being emptied, and the
    # walk would never catch up with them.
    test_enabled = False
    message_number = 0
    past_messages.clear()
    prev_res = {}
    bot_context.clear()

    async def progress(text: str) -> None:
        # A failed edit (rate limit, or a token that expired mid-wipe) must not
        # abort a reset that is otherwise working.
        try:
            await interaction.edit_original_response(content=text)
        except discord.HTTPException:
            pass

    # The database first: it is one fast transaction that either lands or does
    # not, so its result is known before the slow part starts. A reset that is
    # interrupted later is safe to run again — both halves are idempotent.
    print(f"[/reset] wiping channel {channel.id}", flush=True)
    try:
        await db.open_pool()
        await db.init_db()
        rows = await db.clear_channel(channel.id)
        db_line = (
            f"Database: {rows['messages']} messages, {rows['summaries']} summaries, "
            f"{rows['clusters']} clusters."
        )
    except Exception as e:
        # No database is a normal state for this service — it does not need one
        # for anything else. Say so and still clear the channel.
        print(f"[/reset] db wipe failed ({type(e).__name__}: {e})", flush=True)
        db_line = f"Database: skipped ({type(e).__name__} — check DATABASE_URL)."

    await progress(f"{db_line}\nNow deleting messages…")

    # One at a time, newest first. Discord's bulk delete refuses anything over
    # 14 days old, which in a channel this bot has been sitting in is most of
    # it; deleting individually is slower but has no such cutoff. discord.py
    # waits out the rate limits on its own, so a big channel just takes a while.
    deleted = 0
    try:
        async for message in channel.history(limit=None, oldest_first=False):
            try:
                await message.delete()
            except discord.NotFound:
                # Already gone — someone else deleted it, or a previous run did.
                continue
            deleted += 1
            if deleted % 25 == 0:
                await progress(f"{db_line}\nDeleted {deleted} messages so far…")
            if deleted % 100 == 0:
                print(f"[/reset] deleted {deleted} messages", flush=True)
    except discord.Forbidden:
        print(f"[/reset] forbidden after {deleted} messages", flush=True)
        await progress(
            f"{db_line}\nDeleted {deleted} messages, then stopped: 1_bot needs the "
            f"Manage Messages and Read Message History permissions in this channel."
        )
        return

    print(f"[/reset] done: {deleted} messages deleted", flush=True)
    await progress(
        f"{db_line}\nChannel: {deleted} messages deleted. The conversation is reset "
        f"and the bots are off."
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

    # on_ready fires again on every reconnect, so this runs repeatedly. Reading
    # the switch each time is what keeps a reconnect from restarting a schedule
    # that /schedule turned off; is_running() keeps a flaky connection from
    # stacking a second timer on top of the first.
    await _load_schedule_switch()
    if not schedule_enabled:
        print("[schedule] not starting: switched off", flush=True)
    elif not scheduled_session.is_running():
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


async def _load_schedule_switch():
    """Read the stored on/off state of the schedule into schedule_enabled.

    A database that cannot be reached leaves the current value alone. The
    alternative — defaulting to on — would turn the schedule back on every time
    the database blinked, which is the one direction this switch must never
    fail in: it is what someone reaches for to stop the spending.
    """
    global schedule_enabled
    try:
        await db.open_pool()
        await db.init_db()
        stored = await db.get_switches()
    except Exception as e:
        print(f"[schedule] could not read the stored switch "
              f"({type(e).__name__}: {e}); leaving it "
              f"{'on' if schedule_enabled else 'off'}", flush=True)
        return

    # Absent means never set, which is not the same as False — the default of
    # on has to survive a fresh database.
    if SCHEDULE_SWITCH_KEY in stored:
        schedule_enabled = stored[SCHEDULE_SWITCH_KEY]
    print(f"[schedule] schedule is {'on' if schedule_enabled else 'off'}", flush=True)


async def _estimated_spend():
    """Estimated Gemini spend so far in USD, or None if it can't be read.

    None and 0.0 are different answers and the caller has to tell them apart:
    0.0 means nothing has been spent, None means the budget is unknowable right
    now — no database, or no DATABASE_URL on this service at all.
    """
    try:
        await db.open_pool()
        await db.init_db()
        return await db.total_cost_usd()
    except Exception as e:
        print(f"[session] could not read api spend ({type(e).__name__}: {e})",
              flush=True)
        return None


async def _post_opener(channel):
    """Bot 1 opens a conversation on a random topic. True if the message landed.

    This is the scheduled stand-in for the human kickoff message: same JSON
    contract, same routing, so bots 2..N carry on with the existing handler and
    never learn the conversation started itself.
    """
    global current_topic

    # Set before the call, not after: the four other bots read it to stay on
    # subject, and the gateway can hand them the opener the instant it lands.
    current_topic = random.choice(TOPICS)
    print(f"[session] topic: {current_topic}", flush=True)

    prompt = fill_prompt(
        OPENER_PROMPT,
        bot_name="1_bot", bot_number=1,
        num_bots=NUMBER_OF_BOTS, bot_roster=BOT_ROSTER,
        topic=current_topic,
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

    spent = await _estimated_spend()
    if spent is None:
        # No reading means no way to know whether the budget is gone. Skipping
        # one run costs 4 hours; running blind is how a cap gets passed. The
        # next run tries again, so a brief database outage is not permanent.
        print("[session] skipping: the spend so far could not be read", flush=True)
        return
    if spent >= GEMINI_BUDGET_USD:
        print(f"[session] estimated spend ${spent:.2f} has reached the "
              f"${GEMINI_BUDGET_USD:.2f} budget; switching the schedule off. "
              f"Raise GEMINI_BUDGET_USD and redeploy to turn it back on.",
              flush=True)
        scheduled_session.stop()
        return
    print(f"[session] estimated spend so far: ${spent:.4f} "
          f"of ${GEMINI_BUDGET_USD:.2f}", flush=True)

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
    try:
        if not await _post_opener(channel):
            return
        await asyncio.sleep(SESSION_MINUTES * 60)
    finally:
        # Off however this ends: a failed opener, a model error, or /schedule
        # off cancelling the loop mid-conversation. Any path that leaves this
        # True leaves five bots answering every message in the server with no
        # session running and nothing to switch them off.
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
            topic=current_topic,
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
