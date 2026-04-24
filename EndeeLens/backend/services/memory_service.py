from __future__ import annotations

import json
import math
import time
import uuid
from typing import Any

from core.embedders import EndeeEncoderGateway
from core.endee_client import FLOAT32_INDEX, INT8_INDEX, get_endee_client
from core.metrics_db import MetricRecord, insert_metric
from models.memory import (
    MemoryRecallHit,
    MemoryRecallRequest,
    MemoryRecallResponse,
    MemoryStoreRequest,
    MemoryStoreResponse,
)

TOP_K = 10
_encoder = EndeeEncoderGateway()


class DriftCalculator:
    @staticmethod
    def compute(float32_ids: list[str], int8_ids: list[str], top_k: int = TOP_K) -> float:
        relevance: dict[str, float] = {
            doc_id: float(top_k - rank) for rank, doc_id in enumerate(float32_ids[:top_k])
        }

        def _dcg(ranked_ids: list[str]) -> float:
            total = 0.0
            for rank, doc_id in enumerate(ranked_ids[:top_k]):
                rel = relevance.get(doc_id, 0.0)
                total += rel / math.log2(rank + 2)
            return total

        idcg = _dcg(float32_ids)
        if idcg == 0.0:
            return 1.0

        dcg_int8 = _dcg(int8_ids)
        ndcg = dcg_int8 / idcg
        drift = round(1.0 - ndcg, 6)
        return max(0.0, min(1.0, drift))


def _normalize_hits(raw_hits: list[dict[str, Any]]) -> list[MemoryRecallHit]:
    normalized: list[MemoryRecallHit] = []
    for hit in raw_hits:
        if not isinstance(hit, dict):
            continue
        item_id = str(hit.get("id", ""))
        if not item_id:
            continue
        meta = hit.get("meta") or {}
        score = hit.get("score")
        normalized.append(MemoryRecallHit(id=item_id, score=score, meta=meta))
    return normalized


async def store_agent_memory(request: MemoryStoreRequest) -> MemoryStoreResponse:
    client = await get_endee_client()
    dense_vector = _encoder.encode_dense(request.text)
    sparse_doc = _encoder.encode_document(request.text)
    memory_id = str(uuid.uuid4())
    meta = {
        "text": request.text,
        "agent_id": request.agent_id,
        "session_id": request.metadata.get("session_id", "") if request.metadata else "",
        **(request.metadata or {}),
    }

    started = time.perf_counter()
    await client.upsert(
        index_name=FLOAT32_INDEX,
        id=memory_id,
        vector=dense_vector,
        sparse_indices=sparse_doc.indices,
        sparse_values=sparse_doc.values,
        meta=meta,
    )
    await client.upsert(
        index_name=INT8_INDEX,
        id=memory_id,
        vector=dense_vector,
        sparse_indices=sparse_doc.indices,
        sparse_values=sparse_doc.values,
        meta=meta,
    )
    latency_ms = round((time.perf_counter() - started) * 1000, 3)

    await insert_metric(
        MetricRecord(
            operation="store",
            latency_float32=latency_ms,
            latency_int8=latency_ms,
            drift=0.0,
            memory_savings=0.0,
            agent_id=request.agent_id,
            query_text=request.text,
            top_k=1,
            metadata_json=json.dumps({"id": memory_id, "meta": meta}),
        )
    )

    return MemoryStoreResponse(
        id=memory_id,
        latency_ms=latency_ms,
        indexes_written=[FLOAT32_INDEX, INT8_INDEX],
    )


async def recall_agent_memory(request: MemoryRecallRequest) -> MemoryRecallResponse:
    client = await get_endee_client()
    dense_vector = _encoder.encode_dense(request.query)
    sparse_query = _encoder.encode_query(request.query)
    sparse_indices = sparse_query.indices
    sparse_values = sparse_query.values
    filter_meta = {"agent_id": {"$eq": request.agent_id}}

    start_float = time.perf_counter()
    float_hits_raw = await client.search(
        index_name=FLOAT32_INDEX,
        vector=dense_vector,
        sparse_indices=sparse_indices,
        sparse_values=sparse_values,
        top_k=request.top_k,
        filter=filter_meta,
    )
    latency_float_ms = round((time.perf_counter() - start_float) * 1000, 3)

    start_int8 = time.perf_counter()
    int8_hits_raw = await client.search(
        index_name=INT8_INDEX,
        vector=dense_vector,
        sparse_indices=sparse_indices,
        sparse_values=sparse_values,
        top_k=request.top_k,
        filter=filter_meta,
    )
    latency_int8_ms = round((time.perf_counter() - start_int8) * 1000, 3)

    float_hits = _normalize_hits(float_hits_raw)
    int8_hits = _normalize_hits(int8_hits_raw)

    float_ids = [h.id for h in float_hits[: request.top_k]]
    int8_ids = [h.id for h in int8_hits[: request.top_k]]
    recall_drift = DriftCalculator.compute(float_ids, int8_ids, request.top_k)
    top_meta = float_hits[0].meta if float_hits else {}

    memory_savings_estimate = 75.0
    p99_like_latency_ms = round(max(latency_float_ms, latency_int8_ms), 3)

    await insert_metric(
        MetricRecord(
            operation="recall",
            latency_float32=latency_float_ms,
            latency_int8=latency_int8_ms,
            drift=recall_drift,
            memory_savings=memory_savings_estimate,
            agent_id=request.agent_id,
            query_text=request.query,
            top_k=request.top_k,
            metadata_json=json.dumps(
                {
                    "query_sparse_vector": {
                        "indices": sparse_indices,
                        "values": sparse_values,
                    },
                    "float32_ranking": float_ids,
                    "int8_ranking": int8_ids,
                    "float_result_count": len(float_hits),
                    "int8_result_count": len(int8_hits),
                    "title": top_meta.get("title"),
                    "source": top_meta.get("source"),
                    "scientific_domain": top_meta.get("scientific_domain"),
                }
            ),
        )
    )

    return MemoryRecallResponse(
        query=request.query,
        top_k=request.top_k,
        results=float_hits,
        recall_drift=recall_drift,
        latency_float32_ms=latency_float_ms,
        latency_int8_ms=latency_int8_ms,
        p99_like_latency_ms=p99_like_latency_ms,
        memory_savings_estimate=memory_savings_estimate,
        debug={
            "query_tf_idf_path": "query_embed",
            "bm25_contract": "client_tf_server_idf_asymmetric",
            "query_sparse_dimensions": len(sparse_indices),
        },
    )
