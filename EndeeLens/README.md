# EndeeLens Observability Suite

Created by **Harshithk951**.

EndeeLens is a full-stack observability dashboard for monitoring memory retrieval quality and performance across quantization modes.

## What It Includes

- FastAPI backend for metrics, memory store/recall, and seeding.
- Next.js dashboard frontend with Infrastructure Noir UI.
- Drift and latency visualizations for FLOAT32 vs INT8 comparison.

## Setup

### 1) Backend

```bash
cd "/Users/pro/endee/EndeeLens/backend"
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

Backend endpoints:

- API docs: `http://localhost:8000/docs`
- Health: `http://localhost:8000/health`

### 2) Frontend

```bash
cd "/Users/pro/endee/EndeeLens/backend/frontend"
npm install
npm run dev
```

Frontend URL:

- `http://localhost:3000`

### 3) Optional Verification

```bash
cd "/Users/pro/endee/EndeeLens/backend"
source .venv/bin/activate
python verify_backend.py
```
