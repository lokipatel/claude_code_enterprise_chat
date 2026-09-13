from contextlib import asynccontextmanager

from fastapi import FastAPI
from neo4j import AsyncGraphDatabase

from app.config import settings


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.neo4j_driver = AsyncGraphDatabase.driver(
        settings.neo4j_uri,
        auth=(settings.neo4j_username, settings.neo4j_password),
    )
    yield
    await app.state.neo4j_driver.close()


app = FastAPI(title="claude_code_enterprise_chat", lifespan=lifespan)


@app.get("/health")
async def health():
    return {"status": "ok"}
