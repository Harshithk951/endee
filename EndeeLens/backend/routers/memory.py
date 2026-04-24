from __future__ import annotations

from fastapi import APIRouter

from models.memory import (
    MemoryRecallRequest,
    MemoryRecallResponse,
    MemoryStoreRequest,
    MemoryStoreResponse,
)
from services.memory_service import recall_agent_memory, store_agent_memory

router = APIRouter()

@router.post("/store", response_model=MemoryStoreResponse)
async def store_memory(request: MemoryStoreRequest) -> MemoryStoreResponse:
    return await store_agent_memory(request)


@router.post("/recall", response_model=MemoryRecallResponse)
async def recall_memory(request: MemoryRecallRequest) -> MemoryRecallResponse:
    return await recall_agent_memory(request)
