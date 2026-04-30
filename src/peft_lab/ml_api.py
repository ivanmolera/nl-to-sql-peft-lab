"""ML inference API for the NL-to-SQL PEFT Lab playground."""

from __future__ import annotations

import time
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from peft_lab.data import build_prompt, load_wikisql_split
from peft_lab.evaluate_zero_shot import ModelSpec, generate_sql, load_model
from peft_lab.execution import execution_match, is_valid_sql
from peft_lab.runtime_info import collect_runtime_info
from peft_lab.sql import normalize_sql, render_wikisql_query
from peft_lab.wikisql import get_table

ROOT_DIR = Path(__file__).resolve().parents[2]
BASELINE_CONFIG = ROOT_DIR / "configs" / "zero_shot_wikisql_baseline.yaml"

app = FastAPI(title="NL-to-SQL PEFT Lab ML API", version="0.1.0")


class GenerateRequest(BaseModel):
    model_id: str
    example_index: int
    peft_method: str | None = None


@app.get("/health")
def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "service": "nl-to-sql-peft-lab-ml",
        "runtime": collect_runtime_info(),
    }


@app.get("/api/models")
def models() -> dict[str, Any]:
    return {"models": [serialize_model(spec) for spec in get_model_specs().values()]}


@app.get("/api/examples")
def examples(limit: int = 12, offset: int = 0) -> dict[str, Any]:
    dataset = get_dataset()
    if offset < 0:
        raise HTTPException(status_code=400, detail="offset must be >= 0")

    end = min(offset + max(1, min(limit, 50)), len(dataset))
    rows = [
        serialize_example(index, dataset[index])
        for index in range(offset, end)
    ]
    return {
        "dataset": get_config()["dataset"]["name"],
        "split": get_config()["dataset"]["split"],
        "total": len(dataset),
        "offset": offset,
        "limit": limit,
        "examples": rows,
    }


@app.post("/api/generate")
def generate(request: GenerateRequest) -> dict[str, Any]:
    model_specs = get_model_specs()
    if request.model_id not in model_specs:
        raise HTTPException(status_code=404, detail="Unknown model_id")

    dataset = get_dataset()
    if request.example_index < 0 or request.example_index >= len(dataset):
        raise HTTPException(status_code=404, detail="Unknown WikiSQL example")

    if request.peft_method and request.peft_method != "zero-shot":
        raise HTTPException(
            status_code=501,
            detail=f"PEFT method '{request.peft_method}' is not available for live inference yet",
        )

    example = dataset[request.example_index]
    model_spec = model_specs[request.model_id]
    tokenizer, model = get_loaded_model(model_spec.id)

    started_at = time.perf_counter()
    prediction = generate_sql(
        model,
        tokenizer,
        model_spec,
        example,
        get_config(),
    )
    latency = time.perf_counter() - started_at
    reference = render_wikisql_query(example)
    valid = is_valid_sql(prediction, example)
    execution_ok = execution_match(prediction, reference, example)

    return {
        "model": serialize_model(model_spec),
        "example": serialize_example(request.example_index, example),
        "prediction": prediction,
        "reference": reference,
        "exact_match": normalize_sql(prediction) == normalize_sql(reference),
        "valid_sql": valid,
        "execution_match": execution_ok,
        "latency_seconds": latency,
        "prompt": build_prompt(example),
    }


@lru_cache(maxsize=1)
def get_config() -> dict[str, Any]:
    with BASELINE_CONFIG.open("r", encoding="utf-8") as config_file:
        return yaml.safe_load(config_file)


@lru_cache(maxsize=1)
def get_dataset():
    config = get_config()
    return load_wikisql_split(
        config["dataset"]["name"],
        config["dataset"]["split"],
        limit=None,
    )


@lru_cache(maxsize=1)
def get_model_specs() -> dict[str, ModelSpec]:
    return {
        model_config["id"]: ModelSpec(**model_config)
        for model_config in get_config()["models"]
    }


@lru_cache(maxsize=3)
def get_loaded_model(model_id: str):
    model_spec = get_model_specs()[model_id]
    tokenizer = __import__("transformers").AutoTokenizer.from_pretrained(
        model_spec.name,
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = load_model(model_spec, get_config())
    model.eval()
    return tokenizer, model


def serialize_model(model_spec: ModelSpec) -> dict[str, Any]:
    return {
        "id": model_spec.id,
        "name": model_spec.name,
        "architecture": model_spec.architecture,
        "role": model_spec.role,
    }


def serialize_example(index: int, example: dict[str, Any]) -> dict[str, Any]:
    table = get_table(example)
    rows = table.get("rows", [])
    return {
        "index": index,
        "question": example["question"],
        "reference_sql": render_wikisql_query(example),
        "columns": table["header"],
        "types": table.get("types") or ["text"] * len(table["header"]),
        "row_count": len(rows),
        "sample_rows": rows[:4],
    }
