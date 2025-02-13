import asyncio
import logging
import websockets
from services.services_config.config import read_yaml

# Конфигурация
config = read_yaml('services/services_config/config.yaml')
is_running = False
chat_loop = None
chat_task = None

# Пример обработчика WebSocket для чата
async def chat_handler(websocket, path):
    async for message in websocket:
        await websocket.send(f"Echo: {message}")

async def start_chat():
    global is_running, chat_loop, chat_task
    is_running = True
    chat_loop = asyncio.get_running_loop()
    
    try:
        async with websockets.serve(chat_handler, "0.0.0.0", 8765):
            logging.info("Chat server started on ws://0.0.0.0:8765")
            while is_running:
                await asyncio.sleep(1)
    except asyncio.CancelledError:
        logging.info("Chat server was cancelled")
    finally:
        is_running = False
        if chat_task and not chat_task.done():
            chat_task.cancel()
            try:
                await chat_task
            except asyncio.CancelledError:
                pass

def run_chat():
    global chat_loop
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    chat_loop = loop
    try:
        loop.run_until_complete(start_chat())
    finally:
        loop.close()

async def stop_chat():
    global is_running, chat_loop, chat_task
    
    if not is_running:
        return
    
    logging.info("Stopping chat...")
    is_running = False
    
    if chat_task and not chat_task.done():
        chat_task.cancel()
    
    await asyncio.sleep(1)
    
    logging.info("Chat stopped successfully")