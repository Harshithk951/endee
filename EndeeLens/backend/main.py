from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from core.endee_client import get_endee_client
from core.metrics_db import init_db
from routers.memory import router as memory_router
from routers.metrics import router as metrics_router
from routers.system import router as system_router


@asynccontextmanager
async def lifespan(_: FastAPI):
    await init_db()
    await get_endee_client()
    yield


app = FastAPI(title="EndeeLens API", version="1.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:3001", "http://localhost:3002"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(system_router, tags=["system"])
app.include_router(memory_router, prefix="/memory", tags=["memory"])
app.include_router(metrics_router, prefix="/metrics", tags=["metrics"])
