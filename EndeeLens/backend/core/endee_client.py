from __future__ import annotations

import logging
import os
from enum import Enum
from typing import Protocol, runtime_checkable

logger = logging.getLogger("endeelens.client")

FLOAT32_INDEX = "agent_memory_float32"
INT8_INDEX = "agent_memory_int8"
VECTOR_DIM = 384
SPACE_TYPE = "cosine"
SPARSE_MODEL = "endee_bm25"


class Precision(str, Enum):
    FLOAT32 = "float32"
    INT8 = "int8"


@runtime_checkable
class EndeeClientProtocol(Protocol):
    async def upsert(
        self,
        index_name: str,
        id: str,
        vector: list[float],
        sparse_indices: list[int],
        sparse_values: list[float],
        meta: dict,
    ) -> None: ...

    async def search(
        self,
        index_name: str,
        vector: list[float],
        sparse_indices: list[int],
        sparse_values: list[float],
        top_k: int,
        filter: dict | None,
    ) -> list[dict]: ...

    async def health(self) -> bool: ...


class MockEndeeClient:
    def __init__(self) -> None:
        self._store: dict[str, list[dict]] = {
            FLOAT32_INDEX: [],
            INT8_INDEX: [],
        }
        logger.warning("[MOCK] MockEndeeClient active - not connected to Endee server")

    async def upsert(
        self,
        index_name: str,
        id: str,
        vector: list[float],
        sparse_indices: list[int],
        sparse_values: list[float],
        meta: dict,
    ) -> None:
        bucket = self._store.setdefault(index_name, [])
        for i, rec in enumerate(bucket):
            if rec["id"] == id:
                bucket[i] = {
                    "id": id,
                    "vector": vector,
                    "sparseIndices": sparse_indices,
                    "sparseValues": sparse_values,
                    "meta": meta,
                }
                return
        bucket.append(
            {
                "id": id,
                "vector": vector,
                "sparseIndices": sparse_indices,
                "sparseValues": sparse_values,
                "meta": meta,
            }
        )

    async def search(
        self,
        index_name: str,
        vector: list[float],
        sparse_indices: list[int],
        sparse_values: list[float],
        top_k: int,
        filter: dict | None = None,
    ) -> list[dict]:
        import math
        import random

        bucket = self._store.get(index_name, [])

        def _dot(a: list[float], b: list[float]) -> float:
            return sum(x * y for x, y in zip(a, b))

        def _norm(a: list[float]) -> float:
            return math.sqrt(sum(x * x for x in a)) or 1e-9

        def _cosine(a: list[float], b: list[float]) -> float:
            return _dot(a, b) / (_norm(a) * _norm(b))

        def _passes_filter(meta: dict, f: dict | None) -> bool:
            if not f:
                return True
            for field, condition in f.items():
                if isinstance(condition, dict):
                    for op, val in condition.items():
                        if op == "$eq" and meta.get(field) != val:
                            return False
                elif meta.get(field) != condition:
                    return False
            return True

        scored = []
        for rec in bucket:
            if not _passes_filter(rec.get("meta", {}), filter):
                continue
            score = _cosine(vector, rec["vector"])
            scored.append({**rec, "score": score})

        # Keep benchmark demos usable when Docker/Endee is unavailable and nothing is seeded yet.
        if not scored:
            filter_agent = None
            if filter and isinstance(filter.get("agent_id"), dict):
                filter_agent = filter["agent_id"].get("$eq")
            return [
                {
                    "id": f"mock-scifact-{idx + 1}",
                    "score": round(0.95 - (idx * 0.03), 4),
                    "meta": {
                        "agent_id": filter_agent or "mock-agent",
                        "title": f"SciFact Mock Evidence {idx + 1}",
                        "source": "BEIR-SciFact",
                        "scientific_domain": random.choice(
                            ["biology", "medicine", "scientific_claim_verification"]
                        ),
                        "dataset": "beir/scifact",
                        "text": (
                            "Mock SciFact abstract used while Endee Docker is unreachable. "
                            "Seed the benchmark endpoint for real dataset-backed retrieval."
                        ),
                    },
                }
                for idx in range(top_k)
            ]

        scored.sort(key=lambda r: r["score"], reverse=True)
        return scored[:top_k]

    async def health(self) -> bool:
        return True


class RealEndeeClient:
    def __init__(self, host: str = "localhost", port: int = 8080) -> None:
        try:
            from endee import Endee  # type: ignore[import]

            self._client = Endee()
            self._client.set_base_url(f"http://{host}:{port}/api/v1")
            logger.info("[REAL] Connected to Endee server at %s:%d", host, port)
        except ImportError as exc:
            raise RuntimeError(
                "endee SDK not installed. Run: pip install endee endee-model"
            ) from exc

    async def upsert(
        self,
        index_name: str,
        id: str,
        vector: list[float],
        sparse_indices: list[int],
        sparse_values: list[float],
        meta: dict,
    ) -> None:
        import asyncio

        payload = [
            {
                "id": id,
                "vector": vector,
                "sparseIndices": sparse_indices,
                "sparseValues": sparse_values,
                "meta": meta,
            }
        ]
        await asyncio.to_thread(
            self._client.get_index(index_name).upsert,
            payload,
        )

    async def search(
        self,
        index_name: str,
        vector: list[float],
        sparse_indices: list[int],
        sparse_values: list[float],
        top_k: int,
        filter: dict | None = None,
    ) -> list[dict]:
        import asyncio

        raw = await asyncio.to_thread(
            self._client.get_index(index_name).query,
            vector=vector,
            top_k=top_k,
            filter=filter,
            sparse_indices=sparse_indices,
            sparse_values=sparse_values,
        )
        if isinstance(raw, dict):
            candidates = raw.get("result") or raw.get("results") or []
            return list(candidates or [])
        return list(raw or [])

    async def health(self) -> bool:
        import asyncio

        try:
            await asyncio.to_thread(self._client.list_indexes)
            return True
        except Exception:
            return False


async def init_indexes(client: EndeeClientProtocol) -> None:
    if isinstance(client, MockEndeeClient):
        logger.info("[MOCK] Skipping index init - mock client manages indexes in-memory")
        return

    import asyncio

    real: RealEndeeClient = client  # type: ignore[assignment]
    index_configs = [
        {
            "name": FLOAT32_INDEX,
            "dimension": VECTOR_DIM,
            "space": SPACE_TYPE,
            "sparse_model": SPARSE_MODEL,
            "precision": Precision.FLOAT32,
        },
        {
            "name": INT8_INDEX,
            "dimension": VECTOR_DIM,
            "space": SPACE_TYPE,
            "sparse_model": SPARSE_MODEL,
            "precision": Precision.INT8,
        },
    ]

    for cfg in index_configs:
        try:
            await asyncio.to_thread(
                real._client.create_index,
                name=cfg["name"],
                dimension=cfg["dimension"],
                space_type=cfg["space"],
                sparse_model=cfg["sparse_model"],
                precision=cfg["precision"].value,
            )
            logger.info(
                "[INIT] Index created: %s (precision=%s)",
                cfg["name"],
                cfg["precision"].value,
            )
        except Exception as exc:
            if "already exists" in str(exc).lower():
                logger.info("[INIT] Index already exists, skipping: %s", cfg["name"])
            else:
                logger.error("[INIT] Failed to create index %s: %s", cfg["name"], exc)
                raise


_client_singleton: EndeeClientProtocol | None = None


async def get_endee_client() -> EndeeClientProtocol:
    global _client_singleton

    if _client_singleton is not None:
        return _client_singleton

    use_mock = os.getenv("USE_MOCK_DATA", "true").strip().lower() == "true"
    if use_mock:
        _client_singleton = MockEndeeClient()
    else:
        host = os.getenv("ENDEE_HOST", "localhost")
        port = int(os.getenv("ENDEE_PORT", "8080"))
        _client_singleton = RealEndeeClient(host=host, port=port)
        await init_indexes(_client_singleton)

    return _client_singleton
