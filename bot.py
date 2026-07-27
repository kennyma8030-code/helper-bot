import os

import discord
from discord import app_commands
from dotenv import load_dotenv
import db
import loop
import qa

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

# Feature switches, toggled by the admin commands below.
switches = {"bot": False, "RAG": False}

async def _verify(interaction: discord.Interaction, name: str, on: bool) -> bool:
    """Flip a switch if the caller is the admin. Returns whether it was allowed,
    so callers with a body of their own can bail on a rejected command."""
    if interaction.user.id != ADMIN_USER_ID:
        await interaction.response.send_message(
            "NOT ALOUD!", ephemeral=True
        )
        return False
    switches[name] = on
    await interaction.response.send_message(
        f"{name} is now {'on' if on else 'off'}.", ephemeral=True
    )
    return True


@tree.command(name="power", description="Turn the bot on or off (admin only).")
@app_commands.describe(on="True to enable, False to disable")
async def power(interaction: discord.Interaction, on: bool):
    await _verify(interaction, "bot", on)


@tree.command(name="rag", description="Turn RAG on or off (admin only).")
@app_commands.describe(on="True to enable, False to disable")
async def rag(interaction: discord.Interaction, on: bool):
    if not await _verify(interaction, "RAG", on):
        return
    if on:
        await db.open_pool()
        await db.init_db()
    else:
        await db.close_pool()


@tree.command(
    name="respond",
    description="Reply to the target user's last message (admin only).",
)
@app_commands.describe(on="True to reply now, False to do nothing")
async def respond(interaction: discord.Interaction, on: bool):
    if interaction.user.id != ADMIN_USER_ID:
        await interaction.response.send_message("NOT ALOUD!", ephemeral=True)
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


@tree.command(
    name="backfill",
    description="Import a channel's past messages into the RAG store (admin only).",
)
@app_commands.describe(
    channel="Channel to import (defaults to where you run the command)",
    limit="Only the most recent N messages. Leave empty for the whole history.",
)
async def backfill(
    interaction: discord.Interaction,
    channel: discord.TextChannel | None = None,
    limit: int | None = None,
):
    if interaction.user.id != ADMIN_USER_ID:
        await interaction.response.send_message("NOT ALOUD!", ephemeral=True)
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

    scanned = stored = 0
    try:
        # Newest first, so an interrupted run leaves the more useful half done.
        async for m in channel.history(limit=limit):
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

            if stored % 500 == 0:
                await progress(
                    f"Backfilling {channel.mention}… {stored} stored, {scanned} scanned."
                )
    except discord.Forbidden:
        await progress(
            f"No permission to read history in {channel.mention} — "
            f"the bot needs Read Message History there."
        )
        return

    await progress(
        f"Backfilled {channel.mention}: {stored} stored, {scanned} scanned."
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

    
    if switches["bot"]:
        answer = await qa.answer_question(message.content, True)
        if answer is not None:
            await message.reply(answer)
            print("[on_message] reply sent", flush=True)
        return
    
    if switches["RAG"]:
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

    if message.author.id != TARGET_USER_ID:
        print(f"[on_message] ignoring: not target user (target={TARGET_USER_ID})", flush=True)
        return

    
   


if __name__ == "__main__":
    client.run(os.environ["DISCORD_TOKEN"])
