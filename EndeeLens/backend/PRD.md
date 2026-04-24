# PRD.md — Endee-MemScope: AI Agent Memory & Infrastructure Dashboard

> **Version:** 3.0 (Official Endee SDK Syntax) | **Deadline:** April 25, 2026
> **Target Roles:** SDE Intern + AI Intern @ Endee.io

## 1. Executive Summary
Endee-MemScope is a high-performance observability platform built to showcase the core strengths of the Endee Vector Database.
- **AI Focus:** Implements an Agentic Memory system using Endee's asymmetrical BM25 hybrid search.
- **SDE Focus:** Implements a performance benchmarking suite to measure "Quantization Drift" between FLOAT32 and INT8 precision tiers.

## 2. Tech Stack
- **Database:** `endeeio/endee-server:latest` (Docker) on Port `8080`.
- **Backend:** FastAPI, `endee` (SDK), `endee-model` (BM25), `sentence-transformers`, `aiosqlite`.
- **Frontend:** Next.js 14, Tailwind CSS, Shadcn UI, Recharts.

## 3. Core Features & Logic
### A. Hybrid Memory Storage (`/memory/store`)
- Generates 384-dim dense vectors (`all-MiniLM-L6-v2`).
- Generates sparse document vectors via `sparse_model.embed()`.
- **Dual-Write:** Every memory is written to two indexes: `agent_memory_float32` and `agent_memory_int8`.
- **Contract:** All metadata goes into the `meta` field.

### B. Intelligent Recall (`/memory/recall`)
- Uses `sparse_model.query_embed()` for asymmetric BM25 search.
- Performs hybrid search using both dense and sparse components.
- Filters queries by `agent_id` using MongoDB-style operators: `{"agent_id": {"$eq": "..."}}`.
- Calculates **Recall Drift**: `1.0 - (overlap_count / top_k)`.

### C. Observability Dashboard
- **Latency Chart:** Compares FLOAT32 vs INT8 retrieval speed.
- **Quantization Drift Plot:** Visualizes the accuracy cost of memory compression.
- **System Health:** Monitors Endee server status and memory consumption.

## 4. Endee Index Configuration
- **Dimension:** 384
- **Space Type:** `cosine`
- **Sparse Model:** `endee_bm25` (Must be enabled during index creation).

## 5. Execution Plan (36 Hours)
1. **Phase 1:** Docker setup, Backend Core (SDK, Embedders, SQLite), and CRUD APIs.
2. **Phase 2:** Frontend setup, Shadcn UI integration, and Recharts implementation.
3. **Phase 3:** Integration, Seeding data, and Performance Tuning.