from __future__ import annotations

from fastapi import APIRouter, Query

from models.metrics import DriftPoint, LatencyPoint, RecentQuery, SystemSummary
from services.metrics_service import (
    get_drift_timeseries,
    get_latency_timeseries,
    get_recent_queries,
    get_system_summary,
)

router = APIRouter()

@router.get("/summary", response_model=SystemSummary)
async def get_summary() -> SystemSummary:
    return await get_system_summary()


@router.get("/latency", response_model=list[LatencyPoint])
async def get_latency(window: str = Query(default="1h")) -> list[LatencyPoint]:
    return await get_latency_timeseries(window)


@router.get("/drift", response_model=list[DriftPoint])
async def get_drift(window: str = Query(default="1h")) -> list[DriftPoint]:
    return await get_drift_timeseries(window)


@router.get("/recent-queries", response_model=list[RecentQuery])
async def recent_queries(limit: int = Query(default=10, ge=1, le=100)) -> list[RecentQuery]:
    return await get_recent_queries(limit)
