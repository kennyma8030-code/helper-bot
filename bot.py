import os

import discord
from discord import app_commands
from dotenv import load_dotenv
import db
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

async def _verify(interaction: discord.Interaction, name: str, on: bool):
    if interaction.user.id != ADMIN_USER_ID:
        await interaction.response.send_message(
            "NOT ALOUD!", ephemeral=True
        )
        return
    switches[name] = on
    await interaction.response.send_message(
        f"{name} is now {'on' if on else 'off'}.", ephemeral=True
    )


@tree.command(name="power", description="Turn the bot on or off (admin only).")
@app_commands.describe(on="True to enable, False to disable")
async def power(interaction: discord.Interaction, on: bool):
    await _verify(interaction, "bot", on)


@tree.command(name="rag", description="Turn RAG on or off (admin only).")
@app_commands.describe(on="True to enable, False to disable")
async def rag(interaction: discord.Interaction, on: bool):
    await _verify(interaction, "RAG", on)
    if on:
        await db.open_pool()
        await db.init_db()
    else:
        await db.close_pool()


@tree.command(name="ask", description="llm will answer based on context")
@app_commands.describe(ask="ask a question")
async def rag(interaction: discord.Interaction, ask: str):
    pass

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
        answer = await qa.answer_question(message.content)
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
