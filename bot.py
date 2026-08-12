import asyncio
import os
import time

import discord
from discord import app_commands
from discord.ext import tasks
from dotenv import load_dotenv
import db
import loop
import qa
import summarize

load_dotenv()

# message_content and members are privileged intents — enable them in the
# Discord Developer Portal (Bot > Privileged Gateway Intents) as well.
intents = discord.Intents.default()
intents.message_content = True
intents.members = True

client = discord.Client(intents=intents)
tree = app_commands.CommandTree(client)

# The user whose messages the bot should answer. Store the ID, not the name.
TARGET_USER_ID = int(os.environ["TARGET_USER_ID"])

# Only this user may run the /power command.
ADMIN_USER_ID = int(os.environ["ADMIN_USER_ID"])

# Optional: a guild ID to sync commands to instantly (dev). If unset, commands
# sync globally, which can take up to an hour to appear.
GUILD_ID = int(os.environ["GUILD_ID"]) if os.environ.get("GUILD_ID") else None

# Nickname lock: this user's nickname gets put back to NICK_LOCK_NAME shortly
# after they change it. Both must be set for the feature to do anything.
NICK_LOCK_USER_ID = (
    int(os.environ["NICK_LOCK_USER_ID"]) if os.environ.get("NICK_LOCK_USER_ID") else None
)
NICK_LOCK_NAME = os.environ.get("NICK_LOCK_NAME", "")

# How long the new nickname is allowed to stand before it goes back.
NICK_LOCK_DELAY = 200

# At most one pending reset per user. Renaming repeatedly during the countdown
# would otherwise stack up timers that all fire at once at the end.
_nick_tasks: dict[int, asyncio.Task] = {}

# on_ready fires again on every gateway reconnect, and startup work must not
# stack up overlapping runs. Holding the task also keeps it from being garbage
# collected mid-walk.
_startup_task: asyncio.Task | None = None

# Command syncing is per-process, not per-connection. tree.sync() is one of the
# most aggressively rate-limited calls Discord offers, and on_ready fires again
# on every reconnect — so syncing there turns a flaky connection into a stream
# of syncs, and eventually a 429 the login itself inherits. The command set
# cannot change while the process is alive, so once is all it can ever need; a
# deploy that changes the commands is a new process and syncs again.
_synced = False

# Feature switches, toggled by the admin commands below. This dict is only a
# cache: `settings` in Postgres is the durable copy. Railway rebuilds the
# container on every deploy, crash, and restart, so without persistence both
# switches silently fall back to their defaults and the bot's real state is
# whatever the last restart happened to leave.
#
# Both default to on. These are the values for a database that has never been
# written to and for one that cannot be reached — and in both cases recording
# is what we want, because a message not stored while the bot sat switched off
# is gone for good. Anything stored that shouldn't have been can be deleted.
switches = {"bot": True, "RAG": True}

# Ceiling on the write-through, leaving headroom inside Discord's 3s reply
# deadline. See _save_switch.
SWITCH_SAVE_TIMEOUT = 2.0


async def _load_switches() -> None:
    """Restore switch state at startup.

    Keeps the defaults above if the db is unreachable, rather than refusing to
    start: an admin who switched something off will find it off again, and a
    bot that has never been configured comes up running.
    """
    try:
        await db.open_pool()
        await db.init_db()
        stored = await db.get_switches()
    except Exception as e:
        print(
            f"[switches] could not load from db ({type(e).__name__}: {e}); "
            f"staying with {switches}",
            flush=True,
        )
        return

    for name in switches:
        if name in stored:
            switches[name] = stored[name]
    print(f"[switches] restored {switches}", flush=True)


async def _save_switch(name: str, on: bool) -> bool:
    """Write one switch through to the db. Returns whether it will survive a
    restart, so the caller can say so instead of quietly promising it.

    Time-boxed: this runs before the command replies, and Discord drops any
    command that has not answered within 3 seconds. An unreachable database
    must cost the toggle its durability, never the reply.
    """
    try:
        async with asyncio.timeout(SWITCH_SAVE_TIMEOUT):
            await db.open_pool()
            await db.set_switch(name, on)
        return True
    except Exception as e:
        print(
            f"[switches] could not save {name}={on} ({type(e).__name__}: {e})",
            flush=True,
        )
        return False


async def _powered_on(interaction: discord.Interaction) -> bool:
    """The master gate: nothing but /power works while the bot is switched off.

    Sends the refusal itself, so callers only have to bail on False.
    """
    if not switches["bot"]:
        await interaction.response.send_message(
            "The bot is off — run /power on:True first.", ephemeral=True
        )
        return False
    return True


async def _verify(
    interaction: discord.Interaction,
    name: str,
    on: bool,
    *,
    require_on: bool = True,
) -> bool:
    """Flip a switch if the caller is the admin and the bot is powered on.

    Returns whether it was allowed, so callers with a body of their own can
    bail on a rejected command. require_on=False is for /power itself, which
    has to work precisely when the bot is off.
    """
    if interaction.user.id != ADMIN_USER_ID:
        await interaction.response.send_message(
            "NOT ALOUD!", ephemeral=True
        )
        return False
    if require_on and not await _powered_on(interaction):
        return False
    switches[name] = on
    saved = await _save_switch(name, on)
    note = "" if saved else " (in memory only — the db write failed, so a restart resets it)"
    await interaction.response.send_message(
        f"{name} is now {'on' if on else 'off'}.{note}", ephemeral=True
    )
    return True


@tree.command(name="power", description="Turn the bot on or off (admin only).")
@app_commands.describe(on="True to enable, False to disable")
async def power(interaction: discord.Interaction, on: bool):
    if not await _verify(interaction, "bot", on, require_on=False):
        return
    if not on:
        # Powering down clears the modes, so switching back on is a clean slate
        # rather than whatever was left set from last time.
        switches["RAG"] = False
        await _save_switch("RAG", False)
        # The pool deliberately stays open: it is what the switches are stored
        # in, and closing it here would mean the next toggle has to reconnect
        # before it can record itself.


@tree.command(name="rag", description="Turn RAG on or off (admin only).")
@app_commands.describe(on="True to enable, False to disable")
async def rag(interaction: discord.Interaction, on: bool):
    if not await _verify(interaction, "RAG", on):
        return
    if on:
        await db.open_pool()
        await db.init_db()


@tree.command(
    name="respond",
    description="Reply to the target user's last message (admin only).",
)
@app_commands.describe(on="True to reply now, False to do nothing")
async def respond(interaction: discord.Interaction, on: bool):
    if interaction.user.id != ADMIN_USER_ID:
        await interaction.response.send_message("NOT ALOUD!", ephemeral=True)
        return
    if not await _powered_on(interaction):
        return
    if not on:
        await interaction.response.send_message("Nothing to do.", ephemeral=True)
        return

    # answer_question makes model calls, and Discord drops any command that has
    # not replied within 3 seconds. defer() buys 15 minutes.
    await interaction.response.defer(thinking=True, ephemeral=True)

    # Only the single newest message in the channel — if it isn't the target
    # user's, /respond does nothing this time rather than reaching further back.
    latest = [m async for m in interaction.channel.history(limit=1)]
    target = latest[0] if latest else None

    if target is None or target.author.id != TARGET_USER_ID or not target.content:
        await interaction.followup.send(
            "The most recent message isn't one I can respond to.", ephemeral=True
        )
        return

    answer = await qa.answer_question(target.content, False)
    if answer is None:
        await interaction.followup.send("No answer came back.", ephemeral=True)
        return

    await target.reply(answer)
    print("[/respond] reply sent", flush=True)
    await interaction.followup.send("Replied.", ephemeral=True)


@tree.command(name="ask", description="Answer a question from the chat's history.")
@app_commands.describe(ask="What you want to know")
async def ask_command(interaction: discord.Interaction, ask: str):
    if not await _powered_on(interaction):
        return

    # The loop makes many model calls and searches; Discord drops any command
    # that has not replied within 3 seconds. defer() buys 15 minutes.
    await interaction.response.defer(thinking=True)

    # Both are idempotent, so /ask works whether or not /rag is switched on.
    await db.open_pool()
    await db.init_db()

    try:
        answer = await loop.answer(ask)
    except Exception as e:
        print(f"[/ask] investigation failed: {type(e).__name__}: {e}", flush=True)
        await interaction.followup.send(
            f"That one broke something on my end ({type(e).__name__}). Check the logs."
        )
        return

    await interaction.followup.send(answer or "(the investigation produced no answer)")


async def _walk_channel(
    channel,
    *,
    limit: int | None = None,
    stop_at: int | None = None,
    progress=None,
) -> tuple[int, int, set]:
    """Import a channel's history into the store.

    Returns (stored, scanned, days) — `days` being the calendar days, in
    CORPUS_TZ, that actually gained messages, so the caller can re-summarize
    exactly those and nothing else.

    Walks newest first, so an interrupted run leaves the more useful half done.
    `stop_at` is a discord_message_id to stop at: ids are snowflakes, so
    anything at or below one already in the store is already stored, and the
    walk can end there instead of re-reading the whole channel.

    Raises discord.Forbidden if the bot cannot read the channel's history.
    """
    scanned = stored = 0
    days: set = set()

    async for m in channel.history(limit=limit):
        # Everything from here down is already in the store (see stop_at).
        if stop_at is not None and m.id <= stop_at:
            break

        scanned += 1

        # The rules on_message applies, plus one: skip empty messages. An
        # image-only post has content == "" and can never match a search.
        if m.author == client.user or not m.content:
            continue
        # Replies are their own message type — filtering to `default` alone
        # would drop exactly the messages reply_to_message_id exists for.
        if m.type not in (discord.MessageType.default, discord.MessageType.reply):
            continue

        await db.upsert_message(
            discord_message_id=m.id,
            channel_id=m.channel.id,
            author_id=m.author.id,
            content=m.content,
            created_at=m.created_at,
            reply_to_message_id=(
                m.reference.message_id if m.reference else None
            ),
        )
        stored += 1
        # The day this message belongs to on the group's clock, not the
        # server's — the same boundary the summarizer cuts on.
        days.add(m.created_at.astimezone(db.CORPUS_TZ).date())

        if progress and stored % 500 == 0:
            await progress(
                f"Backfilling {channel.mention}… {stored} stored, {scanned} scanned."
            )

    return stored, scanned, days


async def _catch_up() -> None:
    """Re-ingest whatever arrived while the process was down.

    Only channels the store already holds messages for: a channel joins the
    corpus by being backfilled once, on purpose, and is kept current from then
    on. Without this, every deploy and crash leaves a permanent hole, since
    on_message only ever sees messages sent while the bot is connected.
    """
    if not switches["RAG"]:
        print("[catch_up] skipped: RAG is off", flush=True)
        return

    try:
        await db.open_pool()
        await db.init_db()
        channel_ids = await db.known_channel_ids()
    except Exception as e:
        print(f"[catch_up] could not reach the db ({type(e).__name__}: {e})", flush=True)
        return

    for channel_id in channel_ids:
        channel = client.get_channel(channel_id)
        if channel is None or not hasattr(channel, "history"):
            # Left the server, deleted, or not visible with the current intents.
            print(f"[catch_up] {channel_id}: not reachable, skipped", flush=True)
            continue

        try:
            last_id = await db.last_message_id(channel_id)
            stored, scanned, days = await _walk_channel(channel, stop_at=last_id)
        except discord.Forbidden:
            print(f"[catch_up] {channel_id}: no Read Message History", flush=True)
            continue
        except Exception as e:
            # One bad channel must not stop the others from catching up.
            print(f"[catch_up] {channel_id}: failed ({type(e).__name__}: {e})", flush=True)
            continue

        if scanned:
            print(f"[catch_up] {channel_id}: {stored} stored, {scanned} scanned", flush=True)

        # An outage that straddled midnight leaves messages on a day already
        # summarized and marked done. Re-summarize those, or they never enter
        # the index at all.
        if stored:
            try:
                written = await summarize.summarize_backfilled(channel_id, days)
                if written:
                    print(f"[catch_up] {channel_id}: re-summarized {written} day(s)",
                          flush=True)
            except Exception as e:
                print(f"[catch_up] {channel_id}: summarize failed "
                      f"({type(e).__name__}: {e})", flush=True)

    print("[catch_up] done", flush=True)


async def _startup_maintenance() -> None:
    """Catch the store up, then start the summarizer — in that order.

    Ordering is the whole point. summarize_daily runs its body the moment it
    starts, and a day summarized before the catch-up has ingested the rest of
    that day would be written from partial history — with the watermark then
    advanced past it, so it would never be redone.
    """
    await _catch_up()
    if not summarize_daily.is_running():
        summarize_daily.start()


@tasks.loop(hours=24)
async def summarize_daily() -> None:
    """Write the day summaries the store is behind on, once a day.

    tasks.loop runs the body immediately on start and every 24h after, which is
    exactly what the watermark wants: a run on startup repairs whatever the
    downtime missed, and the timer keeps it current. Every run is idempotent —
    it summarizes from the watermark to yesterday and stops.
    """
    if not switches["RAG"]:
        print("[summarize] skipped: RAG is off", flush=True)
        return

    try:
        await db.open_pool()
        await db.init_db()
        written = await summarize.run_once()
    except Exception as e:
        # A failed run must not kill the timer; the next one retries the same
        # days, because the watermark only advances on success.
        print(f"[summarize] run failed ({type(e).__name__}: {e})", flush=True)
        return

    print(f"[summarize] wrote {written} day summaries", flush=True)


@summarize_daily.before_loop
async def _before_summarize() -> None:
    # Nothing can be summarized before the gateway is up and the channel list
    # is knowable.
    await client.wait_until_ready()


@tree.command(
    name="backfill",
    description="Import a channel's past messages into the RAG store (admin only).",
)
@app_commands.describe(
    channel="Channel to import (defaults to where you run the command)",
    limit="Only the most recent N messages. Leave empty for the whole history.",
    full="Re-walk the whole history instead of stopping at what is already stored.",
)
async def backfill(
    interaction: discord.Interaction,
    channel: discord.TextChannel | None = None,
    limit: int | None = None,
    full: bool = False,
):
    if interaction.user.id != ADMIN_USER_ID:
        await interaction.response.send_message("NOT ALOUD!", ephemeral=True)
        return
    if not await _powered_on(interaction):
        return

    channel = channel or interaction.channel

    # Walking a real channel takes minutes, and Discord drops any command that
    # has not replied within 3 seconds. defer() buys 15 minutes.
    await interaction.response.defer(thinking=True, ephemeral=True)

    # Both are idempotent, so this works whether or not /rag is switched on.
    await db.open_pool()
    await db.init_db()

    async def progress(text: str) -> None:
        # A failed progress edit (rate limit, expired token) must not abort a
        # backfill that is otherwise working.
        try:
            await interaction.edit_original_response(content=text)
        except discord.HTTPException:
            pass

    # Incremental by default: stop once the walk reaches a message already
    # stored. full=True re-reads everything, which is what you want after
    # changing what gets stored, since incremental never revisits old rows.
    stop_at = None if full else await db.last_message_id(channel.id)

    try:
        stored, scanned, days = await _walk_channel(
            channel, limit=limit, stop_at=stop_at, progress=progress
        )
    except discord.Forbidden:
        await progress(
            f"No permission to read history in {channel.mention} — "
            f"the bot needs Read Message History there."
        )
        return

    caught_up = "" if full else " (stopped at what was already stored)"
    await progress(
        f"Backfilled {channel.mention}: {stored} stored, {scanned} scanned{caught_up}. "
        f"Summarizing {len(days)} day(s)…"
    )

    # Every day this run added messages to gets a summary — a day imported
    # after its summary was written would otherwise keep the stale one.
    summarized = 0
    if stored:
        try:
            summarized = await summarize.summarize_backfilled(channel.id, days)
        except Exception as e:
            print(f"[/backfill] summarize failed ({type(e).__name__}: {e})", flush=True)
            await progress(
                f"Backfilled {channel.mention}: {stored} stored, {scanned} scanned"
                f"{caught_up}. Summarizing failed ({type(e).__name__}) — check the logs."
            )
            return

    await progress(
        f"Backfilled {channel.mention}: {stored} stored, {scanned} scanned"
        f"{caught_up}. Wrote {summarized} day summaries."
    )


async def _reset_nick_later(member: discord.Member) -> None:
    """Wait out the delay, then put the nickname back."""
    try:
        await asyncio.sleep(NICK_LOCK_DELAY)
        await member.edit(nick=NICK_LOCK_NAME)
        print(f"[nick_lock] reset {member.id} to {NICK_LOCK_NAME!r}", flush=True)
    except discord.Forbidden:
        # Either the bot lacks Manage Nicknames, or the target outranks it.
        # Discord never lets anyone rename the server owner.
        print("[nick_lock] not allowed to rename that member", flush=True)
    except discord.HTTPException as e:
        print(f"[nick_lock] reset failed: {e}", flush=True)
    finally:
        # Only clear the slot if it is still ours — a newer rename may have
        # cancelled this task and put its own there.
        if _nick_tasks.get(member.id) is asyncio.current_task():
            _nick_tasks.pop(member.id, None)


@client.event
async def on_member_update(before: discord.Member, after: discord.Member):
    if not (NICK_LOCK_USER_ID and NICK_LOCK_NAME):
        return
    if after.id != NICK_LOCK_USER_ID:
        return
    # This event also fires for role, timeout, and avatar changes.
    if before.nick == after.nick:
        return
    # Our own reset fires this event again; stop there rather than rescheduling
    # forever.
    if after.nick == NICK_LOCK_NAME:
        return
    if not switches["bot"]:
        print("[nick_lock] ignoring: bot is off", flush=True)
        return

    # A second rename during the countdown replaces the pending reset rather
    # than adding another one.
    pending = _nick_tasks.get(after.id)
    if pending:
        pending.cancel()
    _nick_tasks[after.id] = asyncio.create_task(_reset_nick_later(after))
    print(
        f"[nick_lock] {after.id} renamed to {after.nick!r}, "
        f"resetting in {NICK_LOCK_DELAY}s",
        flush=True,
    )


@client.event
async def on_ready():
    # Before syncing commands: a reconnect must not leave the bot answering
    # with a switch state that only exists in this process.
    await _load_switches()

    global _synced
    if not _synced:
        try:
            if GUILD_ID:
                guild = discord.Object(id=GUILD_ID)
                tree.copy_global_to(guild=guild)
                await tree.sync(guild=guild)
            else:
                await tree.sync()
            _synced = True
        except discord.HTTPException as e:
            # A failed sync must not take the process down. The commands
            # already registered from the last deploy keep working, and the
            # next start syncs again — whereas an exception here kills a bot
            # that is otherwise connected and able to do its job.
            print(f"[sync] failed ({e.status}): {e}", flush=True)

    print(f"Logged in as {client.user}")

    # Last, and in the background: catching up can take minutes on a long
    # outage, and on_ready must not block the gateway that whole time. Messages
    # arriving during the walk are handled by on_message as usual, and the
    # upsert makes the overlap harmless.
    global _startup_task
    if _startup_task is None or _startup_task.done():
        _startup_task = asyncio.create_task(_startup_maintenance())


@client.event
async def on_message(message):
    print(f"[on_message] from {message.author} ({message.author.id}): {message.content!r}", flush=True)

    # Ignore the bot's own messages.
    if message.author == client.user:
        print("[on_message] ignoring: own message", flush=True)
        return

    
    # The master gate: nothing runs while the bot is powered off.
    if not switches["bot"]:
        print("[on_message] ignoring: bot is off", flush=True)
        return

    if switches["RAG"]:
        # RAG defaults to on, so this now runs on a deploy whose database is
        # missing or asleep. Log and keep going: an unhandled exception here
        # fires on every single message and buries the real error.
        try:
            await db.open_pool()
            await db.upsert_message(
                discord_message_id=message.id,
                channel_id=message.channel.id,
                author_id=message.author.id,
                content=message.content,
                created_at=message.created_at,
                reply_to_message_id=(
                    message.reference.message_id if message.reference else None
                ),
            )
        except Exception as e:
            print(
                f"[on_message] not recorded ({type(e).__name__}: {e})",
                flush=True,
            )

    # No answering happens here. on_message only records messages for RAG;
    # replies are sent by /respond and /ask, on the admin's explicit request.


# How long the process is willing to sit out a rate limit before handing the
# restart back to Railway. Long enough for an ordinary window to pass, short
# enough that a container is never parked for an hour doing nothing.
MAX_RATE_LIMIT_WAIT = 900


def _retry_after(e: discord.HTTPException) -> float:
    """Seconds Discord asked us to wait, or a conservative guess.

    A Cloudflare ban answers with an HTML body rather than the usual JSON, so
    the header is the only reliable place to read this from.
    """
    try:
        return float(e.response.headers.get("Retry-After", 0)) or 60.0
    except (TypeError, ValueError):
        return 60.0


def main() -> None:
    """Start the bot, and never answer a rate limit with an instant restart.

    Railway restarts the container the moment the process exits non-zero. When
    what killed it is a 429 on login, restarting is the one response guaranteed
    to make it worse: every attempt is another request against the limit that is
    already exhausted, and Discord extends the block each time. Waiting the
    window out inside the container costs nothing and lets the restart land on
    the other side of it.
    """
    try:
        client.run(os.environ["DISCORD_TOKEN"])
    except discord.HTTPException as e:
        if e.status != 429:
            raise
        wait = min(_retry_after(e), MAX_RATE_LIMIT_WAIT)
        print(
            f"[startup] rate limited by Discord on login; waiting {wait:.0f}s "
            f"before letting the process exit so the restart is not another "
            f"request against the same limit",
            flush=True,
        )
        time.sleep(wait)
        # Non-zero on purpose: Railway's restart policy is what brings the bot
        # back, and a clean exit would leave it down for good.
        raise SystemExit(1)


if __name__ == "__main__":
    main()
