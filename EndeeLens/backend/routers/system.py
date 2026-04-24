from __future__ import annotations

from fastapi import APIRouter, Query

from core.endee_client import get_endee_client
from models.metrics import HealthResponse, SeedResponse
from services.seed_service import seed_benchmark_dataset

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    client = await get_endee_client()
    try:
        connected = await client.health()
    except Exception:
        connected = False

    return HealthResponse(status="healthy" if connected else "degraded", endee_connected=connected)


@router.post("/seed", response_model=SeedResponse)
@router.post("/system/seed", response_model=SeedResponse)
async def seed(dataset: str = Query(default="scifact")) -> SeedResponse:
    return await seed_benchmark_dataset(dataset=dataset)
