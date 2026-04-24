from __future__ import annotations

import json

from core.metrics_db import get_connection
from models.metrics import DriftPoint, LatencyPoint, RecentQuery, SystemSummary

_WINDOW_TO_LIMIT = {
    "1h": 60,
    "6h": 360,
    "24h": 1440,
}


async def get_system_summary() -> SystemSummary:
    async with get_connection() as conn:
        total_row = await (
            await conn.execute("SELECT COUNT(*) AS count FROM metrics WHERE operation = 'store'")
        ).fetchone()
        drift_row = await (
            await conn.execute(
                "SELECT COALESCE(AVG(drift), 0) AS avg_drift FROM metrics WHERE operation = 'recall'"
            )
        ).fetchone()
        p99_f32_row = await (
            await conn.execute(
                """
                SELECT COALESCE(MAX(latency_float32), 0) AS p99
                FROM (
                    SELECT latency_float32
                    FROM metrics
                    WHERE operation = 'recall' AND latency_float32 IS NOT NULL
                    ORDER BY latency_float32 ASC
                )
                """
            )
        ).fetchone()
        p99_i8_row = await (
            await conn.execute(
                """
                SELECT COALESCE(MAX(latency_int8), 0) AS p99
                FROM (
                    SELECT latency_int8
                    FROM metrics
                    WHERE operation = 'recall' AND latency_int8 IS NOT NULL
                    ORDER BY latency_int8 ASC
                )
                """
            )
        ).fetchone()

    return SystemSummary(
        total_memories=int(total_row["count"] if total_row else 0),
        avg_drift=round(float(drift_row["avg_drift"] if drift_row else 0.0), 4),
        p99_latency_float32=round(float(p99_f32_row["p99"] if p99_f32_row else 0.0), 3),
        p99_latency_int8=round(float(p99_i8_row["p99"] if p99_i8_row else 0.0), 3),
        benchmarking_on="BEIR-SciFact",
    )


async def get_latency_timeseries(window: str = "1h") -> list[LatencyPoint]:
    limit = _WINDOW_TO_LIMIT.get(window, 60)
    async with get_connection() as conn:
        cursor = await conn.execute(
            """
            SELECT timestamp, latency_float32, latency_int8
            FROM metrics
            WHERE operation = 'recall'
            ORDER BY id DESC
            LIMIT ?
            """,
            (limit,),
        )
        rows = await cursor.fetchall()

    rows = list(reversed(rows))
    return [
        LatencyPoint(
            timestamp=str(row["timestamp"]),
            p99_float32=round(float(row["latency_float32"] or 0.0), 3),
            p99_int8=round(float(row["latency_int8"] or 0.0), 3),
        )
        for row in rows
    ]


async def get_drift_timeseries(window: str = "1h") -> list[DriftPoint]:
    limit = _WINDOW_TO_LIMIT.get(window, 60)
    async with get_connection() as conn:
        cursor = await conn.execute(
            """
            SELECT timestamp, drift, memory_savings
            FROM metrics
            WHERE operation = 'recall'
            ORDER BY id DESC
            LIMIT ?
            """,
            (limit,),
        )
        rows = await cursor.fetchall()

    rows = list(reversed(rows))
    return [
        DriftPoint(
            timestamp=str(row["timestamp"]),
            drift=round(float(row["drift"] or 0.0), 4),
            memory_savings=round(float(row["memory_savings"] or 75.0), 2),
        )
        for row in rows
    ]


async def get_recent_queries(limit: int = 10) -> list[RecentQuery]:
    async with get_connection() as conn:
        cursor = await conn.execute(
            """
            SELECT timestamp, query_text, agent_id, drift, latency_float32, latency_int8, top_k, metadata_json
            FROM metrics
            WHERE operation = 'recall'
            ORDER BY id DESC
            LIMIT ?
            """,
            (limit,),
        )
        rows = await cursor.fetchall()

    response: list[RecentQuery] = []
    for row in rows:
        metadata_raw = row["metadata_json"]
        metadata: dict = {}
        if isinstance(metadata_raw, str) and metadata_raw.strip():
            try:
                metadata = json.loads(metadata_raw)
            except json.JSONDecodeError:
                metadata = {}
        response.append(
            RecentQuery(
                timestamp=str(row["timestamp"]),
                query=str(row["query_text"] or ""),
                agent_id=str(row["agent_id"] or "unknown-agent"),
                title=metadata.get("title"),
                source=metadata.get("source") or metadata.get("dataset"),
                scientific_domain=metadata.get("scientific_domain"),
                recall_drift=round(float(row["drift"] or 0.0), 4),
                latency_float32_ms=round(float(row["latency_float32"] or 0.0), 3),
                latency_int8_ms=round(float(row["latency_int8"] or 0.0), 3),
                top_k=int(row["top_k"] or 0),
            )
        )
    return response
