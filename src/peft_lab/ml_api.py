"""ML inference API for the NL-to-SQL PEFT Lab playground."""

from __future__ import annotations

import time
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
import torch
from fastapi import FastAPI, HTTPException
from peft import PeftModel
from pydantic import BaseModel
from transformers import AutoTokenizer

from peft_lab.data import build_prompt, load_wikisql_split
from peft_lab.evaluate_zero_shot import ModelSpec, generate_sql, load_model
from peft_lab.execution import execution_match, is_valid_sql
from peft_lab.runtime_info import collect_runtime_info
from peft_lab.sql import normalize_sql, render_wikisql_query
from peft_lab.wikisql import get_table

ROOT_DIR = Path(__file__).resolve().parents[2]
BASELINE_CONFIG = ROOT_DIR / "configs" / "zero_shot_wikisql_baseline.yaml"
QLORA_T5_ADAPTER = ROOT_DIR / "model_artifacts" / "qlora" / "t5-small-wikisql-qlora" / "adapter"
BITFIT_T5_ADAPTER = ROOT_DIR / "model_artifacts" / "bitfit" / "t5-small-wikisql-bitfit" / "adapter"

app = FastAPI(title="NL-to-SQL PEFT Lab ML API", version="0.1.1")


@dataclass
class LiveModelSpec:
    id: str
    name: str
    architecture: str
    role: str
    base_model_name: str | None = None
    adapter_path: str | None = None
    peft_method: str | None = None
    adapter_type: str | None = None

    def generation_spec(self) -> ModelSpec:
        return ModelSpec(
            id=self.id,
            name=self.base_model_name or self.name,
            architecture=self.architecture,
            role=self.role,
        )


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

    model_spec = model_specs[request.model_id]
    if request.peft_method and request.peft_method not in {"zero-shot", model_spec.peft_method}:
        raise HTTPException(
            status_code=501,
            detail=f"PEFT method '{request.peft_method}' is not available for {model_spec.id}",
        )

    example = dataset[request.example_index]
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
def get_model_specs() -> dict[str, LiveModelSpec]:
    specs = {
        model_config["id"]: LiveModelSpec(**model_config)
        for model_config in get_config()["models"]
    }
    if QLORA_T5_ADAPTER.exists():
        specs["t5-small-qlora"] = LiveModelSpec(
            id="t5-small-qlora",
            name="google-t5/t5-small + QLoRA",
            architecture="seq2seq",
            role="T5-small fine-tuned on WikiSQL with QLoRA",
            base_model_name="google-t5/t5-small",
            adapter_path=str(QLORA_T5_ADAPTER),
            peft_method="qlora",
            adapter_type="peft",
        )
    if BITFIT_T5_ADAPTER.exists():
        specs["t5-small-bitfit"] = LiveModelSpec(
            id="t5-small-bitfit",
            name="google-t5/t5-small + BitFit",
            architecture="seq2seq",
            role="T5-small fine-tuned on WikiSQL with BitFit",
            base_model_name="google-t5/t5-small",
            adapter_path=str(BITFIT_T5_ADAPTER),
            peft_method="bitfit",
            adapter_type="bitfit",
        )
    return specs


@lru_cache(maxsize=5)
def get_loaded_model(model_id: str):
    model_spec = get_model_specs()[model_id]
    tokenizer_source = model_spec.adapter_path or model_spec.base_model_name or model_spec.name
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_source,
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = load_live_model(model_spec)
    model.eval()
    return tokenizer, model


def load_live_model(model_spec: LiveModelSpec):
    if model_spec.adapter_path:
        if model_spec.architecture != "seq2seq":
            raise ValueError(f"Unsupported adapter architecture: {model_spec.architecture}")
        base_model = load_model(model_spec.generation_spec(), get_config())
        if model_spec.adapter_type == "bitfit":
            return load_bitfit_adapter(base_model, Path(model_spec.adapter_path))
        return PeftModel.from_pretrained(base_model, model_spec.adapter_path)
    return load_model(model_spec.generation_spec(), get_config())


def load_bitfit_adapter(model, adapter_path: Path):
    device = next(model.parameters()).device
    bias_state = torch.load(adapter_path / "bitfit_biases.pt", map_location=device)
    parameters = dict(model.named_parameters())
    for name, value in bias_state.items():
        if name not in parameters:
            raise ValueError(f"BitFit parameter '{name}' was not found in the base model")
        parameters[name].data.copy_(value.to(parameters[name].device))
    return model


def serialize_model(model_spec: LiveModelSpec) -> dict[str, Any]:
    return {
        "id": model_spec.id,
        "name": model_spec.name,
        "architecture": model_spec.architecture,
        "role": model_spec.role,
        "base_model_name": model_spec.base_model_name,
        "peft_method": model_spec.peft_method,
        "is_fine_tuned": model_spec.adapter_path is not None,
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
