"""Lightweight FastAPI app for the NL-to-SQL PEFT Lab dashboard."""

from __future__ import annotations

import json
import os
from importlib.metadata import PackageNotFoundError, version
from functools import lru_cache
from pathlib import Path
from typing import Any

import httpx
import yaml
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from peft_lab.runtime_info import collect_runtime_info

ROOT_DIR = Path(__file__).resolve().parents[2]
WEB_DIR = ROOT_DIR / "web"
BASELINE_CONFIG = ROOT_DIR / "configs" / "zero_shot_wikisql_baseline.yaml"
REAL_RESULTS = ROOT_DIR / "outputs" / "baselines" / "zero_shot_wikisql.json"
REAL_RESULTS_INDEX = (
    ROOT_DIR / "benchmark_results" / "zero_shot" / "zero_shot_wikisql_index.json"
)
DEMO_RESULTS = ROOT_DIR / "sample_results" / "zero_shot_wikisql.demo.json"
BENCHMARK_MODES = [
    {
        "id": "zero-shot",
        "label": "Zero-shot baseline",
        "description": "Base models without fine-tuning",
        "result_path": REAL_RESULTS_INDEX,
        "fallback_path": DEMO_RESULTS,
    },
    {
        "id": "qlora",
        "label": "QLoRA",
        "description": "LoRA adapters with 4-bit quantization",
        "result_path": ROOT_DIR / "benchmark_results" / "qlora" / "qlora_wikisql_index.json",
        "fallback_path": None,
    },
    {
        "id": "bitfit",
        "label": "BitFit",
        "description": "Bias-only fine-tuning",
        "result_path": ROOT_DIR / "benchmark_results" / "bitfit" / "bitfit_wikisql_index.json",
        "fallback_path": None,
    },
    {
        "id": "prefix-tuning",
        "label": "Prefix Tuning",
        "description": "Trainable virtual prefix vectors",
        "result_path": ROOT_DIR / "benchmark_results" / "prefix_tuning" / "prefix_tuning_wikisql_index.json",
        "fallback_path": None,
    },
    {
        "id": "ia3",
        "label": "IA3",
        "description": "Learned activation scaling",
        "result_path": ROOT_DIR / "benchmark_results" / "ia3" / "ia3_wikisql_index.json",
        "fallback_path": None,
    },
]

app = FastAPI(title="NL-to-SQL PEFT Lab", version="0.1.0")
app.mount("/static", StaticFiles(directory=WEB_DIR), name="static")


class GenerateRequest(BaseModel):
    model_id: str
    example_index: int
    peft_method: str | None = None


@app.get("/")
def index() -> FileResponse:
    return FileResponse(WEB_DIR / "index.html")


@app.get("/api/version")
def app_version() -> dict[str, Any]:
    return {
        "version": current_version(),
        "runtime": collect_runtime_info(),
    }


@app.get("/api/models")
def models() -> dict[str, Any]:
    return {"models": [serialize_model(spec) for spec in get_model_specs()]}


@app.get("/api/benchmarks")
def benchmarks(mode: str = "zero-shot") -> dict[str, Any]:
    mode_config = get_benchmark_mode(mode)
    path = resolve_benchmark_path(mode_config)
    if path is None:
        return {
            "mode": mode_config["id"],
            "label": mode_config["label"],
            "description": mode_config["description"],
            "available": False,
            "is_demo": False,
            "source": None,
            "runtime": collect_runtime_info(),
            "benchmark": pending_benchmark_metadata(mode_config),
            "dataset": None,
            "models": [],
            "message": "Results pending",
        }

    payload = json.loads(path.read_text(encoding="utf-8"))
    payload.setdefault("runtime", collect_runtime_info())
    payload["benchmark"] = {
        **benchmark_metadata_from_payload(payload),
        **payload.get("benchmark", {}),
    }
    payload["available"] = True
    payload["label"] = mode_config["label"]
    payload["description"] = mode_config["description"]
    payload["source"] = str(path.relative_to(ROOT_DIR))
    payload["is_demo"] = path == DEMO_RESULTS
    return payload


@app.get("/api/benchmark-modes")
def benchmark_modes() -> dict[str, Any]:
    modes = []
    for mode_config in BENCHMARK_MODES:
        path = resolve_benchmark_path(mode_config)
        modes.append(
            {
                "id": mode_config["id"],
                "label": mode_config["label"],
                "description": mode_config["description"],
                "available": path is not None,
                "is_demo": path == DEMO_RESULTS,
                "source": str(path.relative_to(ROOT_DIR)) if path else None,
            }
        )
    return {"modes": modes}


@app.get("/api/examples")
async def examples(limit: int = 12, offset: int = 0) -> dict[str, Any]:
    return await proxy_ml_get("/api/examples", {"limit": limit, "offset": offset})


@app.post("/api/generate")
async def generate(request: GenerateRequest) -> dict[str, Any]:
    return await proxy_ml_post("/api/generate", request.model_dump())


@lru_cache(maxsize=1)
def get_config() -> dict[str, Any]:
    with BASELINE_CONFIG.open("r", encoding="utf-8") as config_file:
        return yaml.safe_load(config_file)


@lru_cache(maxsize=1)
def get_model_specs() -> list[dict[str, Any]]:
    return list(get_config()["models"])


def serialize_model(model_spec: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": model_spec["id"],
        "name": model_spec["name"],
        "architecture": model_spec["architecture"],
        "role": model_spec["role"],
    }


def get_benchmark_mode(mode: str) -> dict[str, Any]:
    for mode_config in BENCHMARK_MODES:
        if mode_config["id"] == mode:
            return mode_config
    raise HTTPException(status_code=404, detail="Unknown benchmark mode")


def resolve_benchmark_path(mode_config: dict[str, Any]) -> Path | None:
    candidates = [mode_config["result_path"]]
    if mode_config["id"] == "zero-shot":
        candidates.append(REAL_RESULTS)
    if mode_config["fallback_path"] is not None:
        candidates.append(mode_config["fallback_path"])
    return next((candidate for candidate in candidates if candidate.exists()), None)


def benchmark_metadata_from_payload(payload: dict[str, Any]) -> dict[str, Any]:
    dataset = payload.get("dataset") or {}
    models = payload.get("models") or []
    model_count = len(models)
    sample_size = int(dataset.get("sample_size") or infer_sample_size(models))
    generation = payload.get("generation") or {}
    return {
        "task": "NL-to-SQL",
        "mode": payload.get("mode") or "zero-shot",
        "runner": payload.get("runner") or "peft_lab.benchmark_zero_shot",
        "planned_framework": "Hugging Face LightEval custom task",
        "dataset": dataset.get("name") or "Salesforce/wikisql",
        "split": dataset.get("split") or "validation",
        "sample_size": sample_size,
        "calls_per_model": sample_size,
        "models_evaluated": model_count,
        "total_model_calls": sample_size * model_count,
        "sample_strategy": dataset.get("sample_strategy") or "unknown",
        "seed": (payload.get("experiment") or {}).get("seed"),
        "max_source_length": None,
        "max_new_tokens": None,
        "generation": generation,
        "metrics": [
            "exact_match",
            "bleu",
            "rouge_l",
            "token_f1",
            "sql_validity",
            "execution_accuracy",
            "latency_seconds_per_example",
        ],
        "evaluation_notes": [
            "Exact match compares normalized SQL against the WikiSQL reference.",
            "Valid SQL checks whether the generated query can run against the example table.",
            "Execution match compares generated SQL results against reference SQL results.",
        ],
    }


def pending_benchmark_metadata(mode_config: dict[str, Any]) -> dict[str, Any]:
    return {
        "task": "NL-to-SQL",
        "mode": mode_config["id"],
        "runner": None,
        "planned_framework": "Hugging Face LightEval custom task",
        "dataset": "Salesforce/wikisql",
        "split": "validation",
        "sample_size": None,
        "calls_per_model": None,
        "models_evaluated": 0,
        "total_model_calls": None,
        "sample_strategy": None,
        "seed": None,
        "max_source_length": None,
        "max_new_tokens": None,
        "generation": {},
        "metrics": [
            "exact_match",
            "bleu",
            "rouge_l",
            "token_f1",
            "sql_validity",
            "execution_accuracy",
            "latency_seconds_per_example",
        ],
        "evaluation_notes": [],
    }


def infer_sample_size(models: list[dict[str, Any]]) -> int:
    for model in models:
        sample_size = (model.get("metrics") or {}).get("sample_size")
        if sample_size is not None:
            return int(sample_size)
        examples = model.get("examples") or []
        if examples:
            return len(examples)
    return 0


async def proxy_ml_get(path: str, params: dict[str, Any]) -> dict[str, Any]:
    return await proxy_ml_request("GET", path, params=params)


async def proxy_ml_post(path: str, payload: dict[str, Any]) -> dict[str, Any]:
    return await proxy_ml_request("POST", path, json_payload=payload)


async def proxy_ml_request(
    method: str,
    path: str,
    params: dict[str, Any] | None = None,
    json_payload: dict[str, Any] | None = None,
) -> dict[str, Any]:
    base_url = get_ml_api_base_url()
    timeout = float(os.environ.get("ML_API_TIMEOUT_SECONDS", "900"))
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.request(
                method,
                f"{base_url}{path}",
                params=params,
                json=json_payload,
            )
    except httpx.TimeoutException as exc:
        raise HTTPException(status_code=504, detail="ML inference service timed out") from exc
    except httpx.RequestError as exc:
        raise HTTPException(status_code=502, detail=f"ML inference service unavailable: {exc}") from exc

    if response.status_code >= 400:
        raise HTTPException(
            status_code=response.status_code,
            detail=error_detail(response),
        )
    return response.json()


def get_ml_api_base_url() -> str:
    base_url = os.environ.get("ML_API_BASE_URL", "").strip().rstrip("/")
    if not base_url:
        raise HTTPException(
            status_code=503,
            detail="ML_API_BASE_URL is not configured for playground inference",
        )
    return base_url


def error_detail(response: httpx.Response) -> Any:
    try:
        payload = response.json()
    except ValueError:
        return response.text
    return payload.get("detail", payload)


def current_version() -> str:
    configured = os.environ.get("APP_VERSION")
    if configured:
        return configured
    try:
        return f"v{version('nl-to-sql-peft-lab')}"
    except PackageNotFoundError:
        return "development"
