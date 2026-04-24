from __future__ import annotations

import hashlib
import logging
import math
import time
from dataclasses import dataclass

logger = logging.getLogger("endeelens.embedders")

VECTOR_DIM = 384


@dataclass(frozen=True)
class SparseVector:
    indices: list[int]
    values: list[float]

    def __post_init__(self) -> None:
        if len(self.indices) != len(self.values):
            raise ValueError(
                f"SparseVector length mismatch: indices={len(self.indices)}, values={len(self.values)}"
            )
        if any(v < 0 for v in self.values):
            raise ValueError("SparseVector values must be non-negative (BM25 weights)")


class NumpyDenseStub:
    def encode(self, text: str) -> list[float]:
        import struct

        seed_bytes = hashlib.sha256(text.encode()).digest()
        raw: list[float] = []
        chunk = seed_bytes
        while len(raw) < VECTOR_DIM:
            chunk = hashlib.sha256(chunk).digest()
            raw.extend(
                struct.unpack_from("f", chunk, i)[0]
                for i in range(0, len(chunk) - 3, 4)
            )
        raw = raw[:VECTOR_DIM]
        norm = math.sqrt(sum(x * x for x in raw)) or 1e-9
        return [x / norm for x in raw]


class EndeeModelSparseEncoder:
    def __init__(self) -> None:
        try:
            from endee_model import BM25TextEmbedder  # type: ignore[import]

            self._model = BM25TextEmbedder()
            logger.info("[EMBEDDER] endee-model BM25TextEmbedder loaded (real)")
        except ImportError:
            logger.warning(
                "[EMBEDDER] endee-model not installed - falling back to sparse stub. Run: pip install endee-model"
            )
            self._model = None

    def embed_doc(self, text: str) -> SparseVector:
        t0 = time.perf_counter()
        sv = self._encode_internal(text, mode="doc")
        ms = (time.perf_counter() - t0) * 1000
        logger.debug("[EMBEDDER] embed_doc tokens=%d ms=%.2f", len(sv.indices), ms)
        return sv

    def embed_query(self, text: str) -> SparseVector:
        t0 = time.perf_counter()
        sv = self._encode_internal(text, mode="query")
        ms = (time.perf_counter() - t0) * 1000
        logger.debug("[EMBEDDER] embed_query tokens=%d ms=%.2f", len(sv.indices), ms)
        return sv

    def _encode_internal(self, text: str, mode: str) -> SparseVector:
        if self._model is not None:
            try:
                if mode == "doc":
                    result = self._model.embed(text)
                else:
                    result = self._model.query_embed(text)
                return SparseVector(indices=list(result.indices), values=list(result.values))
            except Exception as exc:
                logger.warning(
                    "[EMBEDDER] endee-model failed (%s), using stub: %s", mode, exc
                )

        return _sparse_stub(text, mode=mode)


def _sparse_stub(text: str, mode: str) -> SparseVector:
    tokens = text.lower().split()
    if not tokens:
        return SparseVector(indices=[], values=[])

    seen: dict[int, float] = {}
    for token in tokens:
        token_hash = int(hashlib.md5(token.encode()).hexdigest(), 16)
        dim = token_hash % 30000
        tf = tokens.count(token) / len(tokens)
        if mode == "doc":
            weight = math.log1p(tf * len(tokens)) + 0.5
        else:
            weight = 1.0
        seen[dim] = max(seen.get(dim, 0.0), weight)

    indices = sorted(seen.keys())
    values = [seen[i] for i in indices]
    return SparseVector(indices=indices, values=values)


class EndeeEncoderGateway:
    def __init__(self) -> None:
        self._dense = NumpyDenseStub()
        self._sparse = EndeeModelSparseEncoder()

    def encode_document(self, text: str) -> SparseVector:
        return self._sparse.embed_doc(text)

    def encode_query(self, text: str) -> SparseVector:
        return self._sparse.embed_query(text)

    def encode_dense(self, text: str) -> list[float]:
        t0 = time.perf_counter()
        vec = self._dense.encode(text)
        ms = (time.perf_counter() - t0) * 1000
        logger.debug("[EMBEDDER] encode_dense dim=%d ms=%.2f", len(vec), ms)
        return vec

    def mode_report(self) -> dict:
        sparse_real = self._sparse._model is not None
        return {
            "dense": "numpy_stub",
            "sparse_doc": "endee_model_real" if sparse_real else "sparse_stub",
            "sparse_query": "endee_model_real" if sparse_real else "sparse_stub",
            "asymmetric_bm25_enforced": True,
        }
