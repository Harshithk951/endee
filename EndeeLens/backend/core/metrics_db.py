from __future__ import annotations

from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

import aiosqlite

DB_PATH = "metrics.db"


CREATE_METRICS_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS metrics (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    operation TEXT NOT NULL,
    latency_float32 REAL,
    latency_int8 REAL,
    drift REAL,
    memory_savings REAL,
    agent_id TEXT,
    query_text TEXT,
    top_k INTEGER,
    metadata_json TEXT
);
"""


@dataclass(slots=True)
class MetricRecord:
    operation: str
    latency_float32: float | None = None
    latency_int8: float | None = None
    drift: float | None = None
    memory_savings: float | None = None
    agent_id: str | None = None
    query_text: str | None = None
    top_k: int | None = None
    metadata_json: str | None = None
    timestamp: str | None = None


@asynccontextmanager
async def get_connection():
    async with aiosqlite.connect(DB_PATH) as conn:
        conn.row_factory = aiosqlite.Row
        yield conn


async def init_db() -> None:
    async with get_connection() as conn:
        await conn.execute(CREATE_METRICS_TABLE_SQL)
        await conn.commit()


async def insert_metric(record: MetricRecord) -> int:
    ts = record.timestamp or datetime.now(timezone.utc).isoformat()
    async with get_connection() as conn:
        cursor = await conn.execute(
            """
            INSERT INTO metrics (
                timestamp, operation, latency_float32, latency_int8, drift, memory_savings,
                agent_id, query_text, top_k, metadata_json
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                ts,
                record.operation,
                record.latency_float32,
                record.latency_int8,
                record.drift,
                record.memory_savings,
                record.agent_id,
                record.query_text,
                record.top_k,
                record.metadata_json,
            ),
        )
        await conn.commit()
        return int(cursor.lastrowid)


async def list_metrics(limit: int = 100) -> list[dict[str, Any]]:
    async with get_connection() as conn:
        cursor = await conn.execute(
            """
            SELECT
                id, timestamp, operation, latency_float32, latency_int8,
                drift, memory_savings, agent_id, query_text, top_k, metadata_json
            FROM metrics
            ORDER BY id DESC
            LIMIT ?
            """,
            (limit,),
        )
        rows = await cursor.fetchall()
        return [dict(row) for row in rows]


async def delete_metrics_for_agent(agent_id: str) -> int:
    async with get_connection() as conn:
        cursor = await conn.execute(
            """
            DELETE FROM metrics
            WHERE agent_id = ?
              AND operation IN ('store', 'recall')
            """,
            (agent_id,),
        )
        await conn.commit()
        return int(cursor.rowcount or 0)


async def delete_dashboard_metrics() -> int:
    async with get_connection() as conn:
        cursor = await conn.execute(
            """
            DELETE FROM metrics
            WHERE operation IN ('store', 'recall')
            """
        )
        await conn.commit()
        return int(cursor.rowcount or 0)
