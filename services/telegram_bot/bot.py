import asyncio
from aiogram import Bot, Dispatcher
from services.services_config.config import read_yaml
import logging
from services.telegram_bot.handlers import router as status_router

config = read_yaml('services/services_config/config.yaml')
bot = Bot(token=config['conf']['BOT_TOKEN'])
dp = Dispatcher()
dp.include_router(status_router)
is_running = False
bot_loop = None
bot_task = None

async def start_polling():
    global is_running, bot_loop, bot_task
    is_running = True
    bot_loop = asyncio.get_running_loop()
    
    try:
        await bot.delete_webhook(drop_pending_updates=True)
        while is_running:
            try:
                bot_task = asyncio.create_task(dp.start_polling(bot))
                await bot_task
            except asyncio.CancelledError:
                break
    finally:
        is_running = False
        if bot_task and not bot_task.done():
            bot_task.cancel()
            try:
                await bot_task
            except asyncio.CancelledError:
                pass

def run_bot():
    global bot_loop
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    bot_loop = loop
    try:
        loop.run_until_complete(start_polling())
    finally:
        loop.close()

async def stop_bot():
    global is_running, bot_loop, bot_task
    
    if not is_running:
        return
    
    logging.info("Stopping bot...")
    is_running = False
    
    if bot_task and not bot_task.done():
        bot_task.cancel()
    
    await asyncio.sleep(1)
    
    await dp.storage.close()
    await bot.session.close()
    
    logging.info("Bot stopped successfully")