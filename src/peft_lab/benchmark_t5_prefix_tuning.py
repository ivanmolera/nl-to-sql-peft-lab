"""Benchmark a trained T5-small Prefix Tuning adapter on WikiSQL."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import torch
from peft import PeftModel
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from peft_lab.benchmark_zero_shot import (
    aggregate_metrics,
    select_preview_records,
    select_sample_indices,
)
from peft_lab.data import load_wikisql_split
from peft_lab.evaluate_zero_shot import ModelSpec, generate_sql, load_config, set_seed
from peft_lab.execution import execution_match, is_valid_sql
from peft_lab.runtime_info import collect_runtime_info
from peft_lab.sql import normalize_sql, render_wikisql_query


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    set_seed(config["experiment"]["seed"])

    dataset = load_wikisql_split(config["dataset"]["name"], config["dataset"]["split"], limit=None)
    sample_indices = select_sample_indices(
        dataset_size=len(dataset),
        sample_size=args.max_examples or config["dataset"]["sample_size"],
        strategy=config["dataset"].get("sample_strategy", "random"),
        seed=config["experiment"]["seed"],
    )
    examples = [(index, dataset[index]) for index in sample_indices]
    model_spec = ModelSpec(
        id=config["model"]["id"],
        name=config["model"]["name"],
        architecture=config["model"]["architecture"],
        role=config["model"]["role"],
    )

    output_dir = Path(config["output"]["dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    result = benchmark_adapter(model_spec, examples, config)
    fine_tuning_metadata = build_fine_tuning_metadata(config)
    if fine_tuning_metadata:
        result["training"] = fine_tuning_metadata
    model_path = output_dir / "prefix_tuning_wikisql_t5-small.json"
    model_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"Saved Prefix Tuning model benchmark to {model_path}")

    result_without_records = {key: value for key, value in result.items() if key != "records"}
    result_without_records["result_file"] = str(model_path)
    benchmark_metadata = build_benchmark_metadata(config, sample_size=len(examples))
    if fine_tuning_metadata:
        benchmark_metadata["fine_tuning"] = fine_tuning_metadata
    payload = {
        "experiment": config["experiment"],
        "runtime": collect_runtime_info(),
        "dataset": {
            "name": config["dataset"]["name"],
            "split": config["dataset"]["split"],
            "total_examples": len(dataset),
            "sample_size": len(examples),
            "sample_strategy": config["dataset"].get("sample_strategy", "random"),
            "sample_indices": sample_indices,
        },
        "benchmark": benchmark_metadata,
        "mode": "prefix-tuning",
        "models": [result_without_records],
    }
    if fine_tuning_metadata:
        payload["training"] = fine_tuning_metadata
    output_path = Path(config["output"]["index_path"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Saved Prefix Tuning benchmark index to {output_path}")


def benchmark_adapter(
    model_spec: ModelSpec,
    examples: list[tuple[int, dict[str, Any]]],
    config: dict[str, Any],
) -> dict[str, Any]:
    print(f"Benchmarking {model_spec.name} Prefix Tuning adapter on {len(examples)} WikiSQL examples")
    load_started_at = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(config["model"]["adapter_path"])
    model = load_adapter_model(config)
    model.eval()
    load_time = time.perf_counter() - load_started_at

    records: list[dict[str, Any]] = []
    predictions: list[str] = []
    references: list[str] = []
    generation_latencies: list[float] = []
    evaluation_latencies: list[float] = []
    total_started_at = time.perf_counter()

    for position, (example_index, example) in enumerate(examples, start=1):
        print(f"[{model_spec.id}] {position}/{len(examples)} example={example_index}")
        reference_sql = render_wikisql_query(example)
        generation_started_at = time.perf_counter()
        error: str | None = None
        try:
            prediction = generate_sql(model, tokenizer, model_spec, example, config)
        except Exception as exc:  # pragma: no cover - model/runtime dependent
            prediction = ""
            error = str(exc)
        generation_latency = time.perf_counter() - generation_started_at

        evaluation_started_at = time.perf_counter()
        exact_match = normalize_sql(prediction) == normalize_sql(reference_sql)
        valid_sql = False if error else is_valid_sql(prediction, example)
        execution_ok = False if error else execution_match(prediction, reference_sql, example)
        evaluation_latency = time.perf_counter() - evaluation_started_at

        predictions.append(prediction)
        references.append(reference_sql)
        generation_latencies.append(generation_latency)
        evaluation_latencies.append(evaluation_latency)
        records.append(
            {
                "example_index": example_index,
                "question": example["question"],
                "prediction": prediction,
                "reference": reference_sql,
                "exact_match": exact_match,
                "valid_sql": valid_sql,
                "execution_match": execution_ok,
                "generation_latency_seconds": generation_latency,
                "evaluation_latency_seconds": evaluation_latency,
                "output_characters": len(prediction),
                "error": error,
            }
        )

    total_time = time.perf_counter() - total_started_at
    preview_limit = config.get("reporting", {}).get("preview_limit", 20)
    return {
        "id": model_spec.id,
        "name": f"{model_spec.name} + Prefix Tuning",
        "architecture": model_spec.architecture,
        "role": model_spec.role,
        "runtime": collect_runtime_info(),
        "metrics": aggregate_metrics(
            records=records,
            predictions=predictions,
            references=references,
            generation_latencies=generation_latencies,
            evaluation_latencies=evaluation_latencies,
            load_time=load_time,
            total_time=total_time,
        ),
        "examples": select_preview_records(records, preview_limit),
        "records": records,
    }


def load_adapter_model(config: dict[str, Any]):
    base_model = AutoModelForSeq2SeqLM.from_pretrained(config["model"]["name"])
    if torch.cuda.is_available():
        base_model.to("cuda")
    return PeftModel.from_pretrained(base_model, config["model"]["adapter_path"])


def build_benchmark_metadata(config: dict[str, Any], sample_size: int) -> dict[str, Any]:
    metric_names = [
        "exact_match",
        "bleu",
        "rouge_l",
        "token_f1",
        "sql_validity",
        "execution_accuracy",
        "latency_seconds_per_example",
        "generation_latency_p50",
        "generation_latency_p95",
        "throughput_examples_per_second",
    ]
    return {
        "task": "NL-to-SQL",
        "mode": "prefix-tuning",
        "runner": "peft_lab.benchmark_t5_prefix_tuning",
        "planned_framework": "Hugging Face LightEval custom task",
        "dataset": config["dataset"]["name"],
        "split": config["dataset"]["split"],
        "sample_size": sample_size,
        "calls_per_model": sample_size,
        "models_evaluated": 1,
        "total_model_calls": sample_size,
        "sample_strategy": config["dataset"].get("sample_strategy", "random"),
        "seed": config["experiment"]["seed"],
        "max_source_length": config["prompt"].get("max_source_length"),
        "max_new_tokens": config["prompt"].get("max_new_tokens"),
        "generation": config.get("generation", {}),
        "metrics": metric_names,
        "evaluation_notes": [
            "Exact match compares normalized SQL against the WikiSQL reference.",
            "Valid SQL checks whether the generated query can run against the example table.",
            "Execution match compares generated SQL results against reference SQL results.",
        ],
    }


def build_fine_tuning_metadata(config: dict[str, Any]) -> dict[str, Any] | None:
    fine_tuning = config.get("fine_tuning")
    if not fine_tuning:
        return None

    metadata = {
        key: value
        for key, value in fine_tuning.items()
        if key != "eval_metrics_path"
    }
    eval_metrics_path = fine_tuning.get("eval_metrics_path")
    if eval_metrics_path and Path(eval_metrics_path).exists():
        metadata["trainer_eval_metrics"] = json.loads(
            Path(eval_metrics_path).read_text(encoding="utf-8")
        )
    return metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to Prefix Tuning benchmark YAML.")
    parser.add_argument("--max-examples", type=int, default=None)
    return parser.parse_args()


if __name__ == "__main__":
    main()
