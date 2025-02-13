from contextlib import asynccontextmanager
from fastapi import FastAPI
from models import Base, engine


@asynccontextmanager
async def lifespan(app: FastAPI):
    print('Start')
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield
    await engine.dispose()
    print('stop')

