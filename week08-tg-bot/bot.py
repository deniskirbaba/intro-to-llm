from config import *
from aiogram import Bot, types
from aiogram import Dispatcher
from aiogram.filters.command import Command
import asyncio
import aiohttp

bot = Bot(BOT_TOKEN)
dp = Dispatcher()

url = "http://127.0.0.1:8000/genetate_text"


async def post_requets(url, data):
    async with aiohttp.ClientSession() as session:
        async with session.post(url, json=data) as respons:
            return await respons.json()


@dp.message(Command('start'))
async def start(message: types.Message):
    await message.reply("all work")

@dp.message()
async def answer(message: types.Message):
    data = {"promt": message.text}
    answer = asyncio.create_task(post_requets(url, data))
    response_json = await answer
    answer_text = response_json["text"]
    await message.reply(answer_text)


async def main():
    await dp.start_polling(bot)





if __name__ == "__main__":
    asyncio.run(main())
