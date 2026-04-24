# claude.md — AI Coding Agent Instructions

> **ROLE:** Staff Full-Stack Engineer.
> **OBJECTIVE:** Build Endee-MemScope per PRD.md.
> **STRICT RULE:** Use Endee.io native syntax only. Do not use Qdrant/Milvus patterns.

---

## 1. Environment & Config
- **Endee Port:** `8080` (REST/gRPC).
- **Docker Image:** `endeeio/endee-server:latest`.
- **Backend Dependencies:** `fastapi`, `endee`, `endee-model`, `sentence-transformers`, `aiosqlite`, `numpy`.

## 2. Endee SDK Implementation Rules

### 2.1 Initialization
```python
from endee import Endee, Precision
client = Endee(host="localhost", port=8080)

2.2 Index Creation (Hybrid Enabled)
code
Python
client.create_index(
    name="agent_memory_int8",
    dimension=384,
    space_type="cosine",
    precision=Precision.INT8,
    sparse_model="endee_bm25"
)
2.3 BM25 Asymmetric Pattern
code
Python
from endee_model import SparseModel
sparse_model = SparseModel(model_name="endee/bm25")

# For Upserting: Use .embed()
sparse_doc = sparse_model.embed(text)

# For Searching: Use .query_embed()
sparse_query = sparse_model.query_embed(query_text)
2.4 The Upsert Contract
Metadata MUST be inside a meta field. Use CamelCase for sparse fields.

code
Python
payload = [{
    "id": str(uuid.uuid4()),
    "vector": dense_vector,
    "sparseIndices": sparse_doc.indices.tolist(),
    "sparseValues": sparse_doc.values.tolist(),
    "meta": {
        "text": raw_text,
        "agent_id": agent_id,
        "created_at": timestamp
    }
}]
index.upsert(payload)
2.5 Hybrid Search & Filtering
code
Python
results = index.search(
    vector=query_dense_vec,
    sparseIndices=query_sparse_vec.indices.tolist(),
    sparseValues=query_sparse_vec.values.tolist(),
    top_k=5,
    filter={"agent_id": {"$eq": agent_id}}
)
3. Directory Structure
code
Text
/backend
  /core
    endee_client.py  # Singleton + Index creation
    embedders.py     # Dense (MiniLM) + Sparse (BM25)
    metrics_db.py    # SQLite metrics
  /routers           # memory.py, metrics.py
  main.py
/frontend
  /components        # Charts, Shadcn UI
  /app               # Next.js App Router
/docker-compose.yml

# claude.md — EndeeLens System Instructions

> **ROLE:** Staff Engineer. **PROJECT:** EndeeLens.
> **DB:** Endee.io (Port 8080). **DOCKER:** endeeio/endee-server:latest.

## 1. SDK Implementation Rules
- **Init:** `client = Endee(host="localhost", port=8080)`
- **Indexing:** Must use `sparse_model="endee_bm25"` and `Precision.INT8` vs `Precision.FLOAT32`.
- **BM25:** Use `from endee_model import SparseModel`.
- **Metadata:** Strictly use the `meta` field for all payload data.
- **Filter:** Use MongoDB-style `{"agent_id": {"$eq": "..."}}`.

## 2. Directory Structure
```text
/backend
  /core (endee_client.py, embedders.py, metrics_db.py)
  /routers (memory.py, metrics.py)
  main.py
/frontend (Next.js App Router)
/docker-compose.yml

4. Execution Protocol
Token Savings: Only output the file you are currently editing.

Phase Gate: Complete Phase 1 (Backend) and verify via /health and /seed before starting the Frontend.

Wait for Signal: Acknowledge these rules, then wait for the user to say "Start Phase 1".