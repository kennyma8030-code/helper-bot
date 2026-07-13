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

bot_id_map = {index: os.environ.get(f"{index}_bot") for index in range(1, 6)}
ADMIN_USER_ID = int(os.environ["ADMIN_USER_ID"])
answer_randomely = {"bot_number": }
past_messages = []
bot_context = defaultdict(str)
message_number = 0
prev_res = {}
rand = 0

def handler_helper(client, index):
    @client.event
    async def on_message(message):
        global message_number
        global random
        if message_number == 0:
            return
        if message.author.id != bot_id_map{index}:
            if prev_res["respond_to"]["none"]:
                return
            elif prev_res["respond_to"]["index"]:
                return
            elif prev_res["respond_to"][str(NUMBER_OF_BOTS + 1)]:
                if rand == 0:
                    rand = random.randint(1, 5)


            ask_gemini()


for i, client in enumerate(args, start=1):
    handler_helper(client, i)


if __name__ == "__main__":
asyncio.run(main())