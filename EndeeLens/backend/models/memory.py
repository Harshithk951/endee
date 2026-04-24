from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class MemoryStoreRequest(BaseModel):
    text: str = Field(min_length=1)
    agent_id: str = Field(min_length=1)
    metadata: dict[str, Any] = Field(default_factory=dict)


class MemoryStoreResponse(BaseModel):
    id: str
    latency_ms: float
    indexes_written: list[str]


class MemoryRecallRequest(BaseModel):
    query: str = Field(min_length=1)
    agent_id: str = Field(min_length=1)
    top_k: int = Field(default=5, ge=1, le=50)


class MemoryRecallHit(BaseModel):
    id: str
    score: float | None = None
    meta: dict[str, Any] = Field(default_factory=dict)


class MemoryRecallResponse(BaseModel):
    query: str
    top_k: int
    results: list[MemoryRecallHit]
    recall_drift: float
    latency_float32_ms: float
    latency_int8_ms: float
    p99_like_latency_ms: float
    memory_savings_estimate: float
    debug: dict[str, Any]
