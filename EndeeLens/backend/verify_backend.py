from __future__ import annotations

import json
import sys
import uuid
from dataclasses import dataclass
from typing import Any

import httpx

BASE_URL = "http://localhost:8000"
TIMEOUT = 30.0


@dataclass
class CheckResult:
    name: str
    ok: bool
    detail: str


def _emit(message: str) -> None:
    sys.stdout.write(f"{message}\n")


def _print_header(title: str) -> None:
    _emit(f"\n=== {title} ===")


def _pick_seed_endpoint(client: httpx.Client) -> str | None:
    for path in ("/system/seed", "/seed"):
        try:
            response = client.post(path)
            if response.status_code == 200:
                _emit(f"Seed endpoint used: {path} (status {response.status_code})")
                return path
        except httpx.HTTPError:
            continue
    return None


def _json_or_text(response: httpx.Response) -> Any:
    try:
        return response.json()
    except Exception:  # noqa: BLE001
        return {"raw_text": response.text}


def _health_check(client: httpx.Client) -> CheckResult:
    response = client.get("/health")
    payload = _json_or_text(response)
    if not isinstance(payload, dict):
        return CheckResult("health", False, f"status={response.status_code}, invalid_response")
    connected = bool(payload.get("endee_connected", False))
    detail = f"status={payload.get('status')}, endee_connected={connected}"
    return CheckResult("health", response.status_code == 200 and connected, detail)


def _storage_test(client: httpx.Client) -> tuple[CheckResult, dict[str, Any], str]:
    unique_code = f"X-100-{uuid.uuid4().hex[:8]}"
    memory_text = f"The project code is {unique_code}"
    body = {
        "text": memory_text,
        "agent_id": "agent_001",
        "metadata": {"source": "verify_script", "trace_id": str(uuid.uuid4())},
    }
    response = client.post("/memory/store", json=body)
    payload = _json_or_text(response)
    has_latency = isinstance(payload.get("latency_ms"), (int, float))
    ok = response.status_code == 200 and has_latency
    detail = f"status={response.status_code}, latency_ms={payload.get('latency_ms')}, id={payload.get('id')}"
    return CheckResult("store", ok, detail), payload, unique_code


def _recall_test(client: httpx.Client, unique_code: str) -> tuple[CheckResult, dict[str, Any]]:
    body = {"query": "project code", "agent_id": "agent_001", "top_k": 10}
    response = client.post("/memory/recall", json=body)
    payload = _json_or_text(response)

    found = False
    for row in payload.get("results", []) if isinstance(payload, dict) else []:
        text = (row.get("meta") or {}).get("text", "")
        if unique_code in text:
            found = True
            break

    detail = (
        f"status={response.status_code}, found_unique_memory={found}, "
        f"recall_drift={payload.get('recall_drift')}, "
        f"latency_float32_ms={payload.get('latency_float32_ms')}, "
        f"latency_int8_ms={payload.get('latency_int8_ms')}, "
        f"p99_like_latency_ms={payload.get('p99_like_latency_ms')}"
    )
    ok = response.status_code == 200 and found
    return CheckResult("recall", ok, detail), payload


def _metrics_validation(client: httpx.Client) -> tuple[CheckResult, dict[str, Any], list[dict[str, Any]]]:
    summary_resp = client.get("/metrics/summary")
    latency_resp = client.get("/metrics/latency", params={"window": "1h"})
    summary = _json_or_text(summary_resp)
    latency_points = _json_or_text(latency_resp)
    if not isinstance(latency_points, list):
        latency_points = []
    ok = summary_resp.status_code == 200 and latency_resp.status_code == 200
    detail = (
        f"summary_status={summary_resp.status_code}, latency_status={latency_resp.status_code}, "
        f"total_memories={summary.get('total_memories')}, avg_drift={summary.get('avg_drift')}, "
        f"latency_points={len(latency_points)}"
    )
    return CheckResult("metrics", ok, detail), summary, latency_points


def main() -> int:
    results: list[CheckResult] = []

    try:
        with httpx.Client(base_url=BASE_URL, timeout=TIMEOUT) as client:
            _print_header("1) HEALTH CHECK")
            health = _health_check(client)
            _emit(health.detail)
            results.append(health)

            _print_header("2) SEEDING")
            seed_endpoint = _pick_seed_endpoint(client)
            if seed_endpoint:
                # Trigger one more seed call on selected endpoint to match explicit phase requirement.
                seed_response = client.post(seed_endpoint)
                seed_body = _json_or_text(seed_response)
                _emit(f"seed_response_status={seed_response.status_code}")
                _emit(f"seed_response_body={seed_body}")
                results.append(
                    CheckResult(
                        "seed",
                        seed_response.status_code == 200,
                        f"endpoint={seed_endpoint}, status={seed_response.status_code}",
                    )
                )
            else:
                _emit("seed_response_status=unavailable")
                _emit("seed_response_body={'detail': 'No 200 response from /system/seed or /seed'}")
                results.append(
                    CheckResult(
                        "seed",
                        False,
                        "endpoint_unavailable_or_failing",
                    )
                )

            _print_header("3) STORAGE TEST")
            store_result, store_payload, unique_code = _storage_test(client)
            _emit(store_result.detail)
            _emit(f"stored_payload={json.dumps(store_payload, indent=2)}")
            results.append(store_result)

            _print_header("4) RECALL & DRIFT TEST")
            recall_result, recall_payload = _recall_test(client, unique_code)
            _emit(recall_result.detail)
            _emit(f"recall_payload={json.dumps(recall_payload, indent=2)}")
            results.append(recall_result)

            _print_header("5) METRICS VALIDATION")
            metrics_result, summary, latency_points = _metrics_validation(client)
            _emit(metrics_result.detail)
            _emit(f"summary={json.dumps(summary, indent=2)}")
            _emit(f"latency_sample={json.dumps(latency_points[:5], indent=2)}")
            results.append(metrics_result)

    except httpx.HTTPError as err:
        _emit("\nBackend verification failed due to connection/runtime error:")
        _emit(str(err))
        return 2
    except Exception as err:  # noqa: BLE001
        _emit("\nBackend verification failed with unexpected error:")
        _emit(str(err))
        return 3

    _print_header("FINAL STATUS")
    failed = [r for r in results if not r.ok]
    for r in results:
        marker = "PASS" if r.ok else "FAIL"
        _emit(f"[{marker}] {r.name}: {r.detail}")

    if failed:
        _emit("\nDiagnosis:")
        for item in failed:
            if item.name == "seed":
                _emit("- Seed endpoint mismatch or router path issue (logic/router contract mismatch).")
            elif item.name in {"store", "recall"}:
                _emit("- Memory flow failed; likely router/service logic or Endee index interaction issue.")
            elif item.name == "health":
                _emit("- Endee connectivity issue or incorrect client/index bootstrap.")
            elif item.name == "metrics":
                _emit("- SQLite logging or metrics query logic issue.")
        return 1

    _emit("\nAll backend verification checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
