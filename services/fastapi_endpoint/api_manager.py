import asyncio
import logging
from fastapi import FastAPI
from services.fastapi_endpoint.fastapi_app import app


is_running = False
api_loop = None
api_task = None

async def start_api():
    global is_running, api_loop, api_task
    is_running = True
    api_loop = asyncio.get_running_loop()
    
    try:
        import uvicorn
        config = uvicorn.Config(app, host="0.0.0.0", port=8000, log_level="info")
        server = uvicorn.Server(config)
        api_task = asyncio.create_task(server.serve())
        await api_task
    except asyncio.CancelledError:
        logging.info("API server was cancelled")
    finally:
        is_running = False
        if api_task and not api_task.done():
            api_task.cancel()
            try:
                await api_task
            except asyncio.CancelledError:
                pass

def run_api():
    global api_loop
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    api_loop = loop
    try:
        loop.run_until_complete(start_api())
    finally:
        loop.close()

async def stop_api():
    global is_running, api_loop, api_task
    
    if not is_running:
        return
    
    logging.info("Stopping API...")
    is_running = False
    
    if api_task and not api_task.done():
        api_task.cancel()
    
    await asyncio.sleep(1)
    
    logging.info("API stopped successfully")