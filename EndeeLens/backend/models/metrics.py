from __future__ import annotations

from pydantic import BaseModel


class SystemSummary(BaseModel):
    total_memories: int
    avg_drift: float
    p99_latency_float32: float
    p99_latency_int8: float
    benchmarking_on: str = "BEIR-SciFact"


class LatencyPoint(BaseModel):
    timestamp: str
    p99_float32: float
    p99_int8: float


class DriftPoint(BaseModel):
    timestamp: str
    drift: float
    memory_savings: float


class RecentQuery(BaseModel):
    timestamp: str
    query: str
    agent_id: str
    title: str | None = None
    source: str | None = None
    scientific_domain: str | None = None
    recall_drift: float
    latency_float32_ms: float
    latency_int8_ms: float
    top_k: int


class HealthResponse(BaseModel):
    status: str
    endee_connected: bool


class SeedResponse(BaseModel):
    inserted: int
    agent_id: str
    note: str
