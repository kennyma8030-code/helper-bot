import discord
from discord import app_commands
import asyncio
import contextlib
import os
from llm import ask_gemini
import random
from collections import defaultdict

NUMBER_OF_BOTS = 5

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

ADMIN_USER_ID = int(os.environ["ADMIN_USER_ID"])
answer_randomely = {"bot_number": }
past_messages = []
bot_context = defaultdict(str)
def handler_helper(client, id):
    @client.event
    async def on_message(message):
        if message.author.id != id:
            random.randint(1, 5)

            ask_gemini()

for client in args:
    handler_helper(client)


if __name__ == "__main__":
    asyncio.run(main())