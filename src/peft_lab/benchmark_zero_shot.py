"""Benchmark selected base models on WikiSQL without fine-tuning."""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path
from typing import Any

import numpy as np

from peft_lab.data import load_wikisql_split
from peft_lab.evaluate_zero_shot import (
    ModelSpec,
    generate_sql,
    load_config,
    load_model,
    set_seed,
)
from peft_lab.execution import execution_match, is_valid_sql
from peft_lab.metrics import (
    bleu_score,
    exact_match_score,
    rouge_l_score,
    token_f1_score,
)
from peft_lab.runtime_info import collect_runtime_info
from peft_lab.sql import normalize_sql, render_wikisql_query


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    set_seed(config["experiment"]["seed"])

    dataset = load_wikisql_split(
        config["dataset"]["name"],
        config["dataset"]["split"],
        limit=None,
    )
    sample_indices = select_sample_indices(
        dataset_size=len(dataset),
        sample_size=args.max_examples or config["dataset"]["sample_size"],
        strategy=config["dataset"].get("sample_strategy", "random"),
        seed=config["experiment"]["seed"],
    )
    examples = [(index, dataset[index]) for index in sample_indices]
    selected_model_ids = set(args.model_id or [])
    model_specs = [
        ModelSpec(**model_config)
        for model_config in config["models"]
        if not selected_model_ids or model_config["id"] in selected_model_ids
    ]
    if selected_model_ids and len(model_specs) != len(selected_model_ids):
        available = ", ".join(model_config["id"] for model_config in config["models"])
        raise ValueError(f"Unknown model id. Available models: {available}")

    output_dir = Path(config["output"]["dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    model_results = []
    for model_spec in model_specs:
        result = benchmark_model(model_spec, examples, config)
        model_path = output_dir / result_filename(model_spec.id)
        model_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(f"Saved model benchmark to {model_path}")
        result_without_records = {key: value for key, value in result.items() if key != "records"}
        result_without_records["result_file"] = str(model_path)
        model_results.append(result_without_records)

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
        "benchmark": build_benchmark_metadata(
            config=config,
            model_count=len(model_specs),
            sample_size=len(examples),
        ),
        "mode": "zero-shot",
        "models": model_results,
    }
    output_path = Path(config["output"]["index_path"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Saved zero-shot benchmark index to {output_path}")


def benchmark_model(
    model_spec: ModelSpec,
    examples: list[tuple[int, dict[str, Any]]],
    config: dict[str, Any],
) -> dict[str, Any]:
    print(f"Benchmarking {model_spec.name} on {len(examples)} WikiSQL examples")

    load_started_at = time.perf_counter()
    tokenizer = __import__("transformers").AutoTokenizer.from_pretrained(
        model_spec.name,
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = load_model(model_spec, config)
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
        execution_ok = (
            False if error else execution_match(prediction, reference_sql, example)
        )
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
    metrics = aggregate_metrics(
        records=records,
        predictions=predictions,
        references=references,
        generation_latencies=generation_latencies,
        evaluation_latencies=evaluation_latencies,
        load_time=load_time,
        total_time=total_time,
    )
    preview_limit = config.get("reporting", {}).get("preview_limit", 20)
    return {
        "id": model_spec.id,
        "name": model_spec.name,
        "architecture": model_spec.architecture,
        "role": model_spec.role,
        "runtime": collect_runtime_info(),
        "metrics": metrics,
        "examples": select_preview_records(records, preview_limit),
        "records": records,
    }


def aggregate_metrics(
    records: list[dict[str, Any]],
    predictions: list[str],
    references: list[str],
    generation_latencies: list[float],
    evaluation_latencies: list[float],
    load_time: float,
    total_time: float,
) -> dict[str, float]:
    total = len(records)
    errors = sum(1 for record in records if record["error"])
    empty_outputs = sum(1 for record in records if not record["prediction"].strip())
    non_sql_outputs = sum(
        1 for record in records if "select" not in record["prediction"].lower()
    )
    failures = sum(
        1
        for record in records
        if record["error"]
        or not record["prediction"].strip()
        or "select" not in record["prediction"].lower()
    )
    execution_matches = sum(1 for record in records if record["execution_match"])
    valid_sql = sum(1 for record in records if record["valid_sql"])

    return {
        "sample_size": float(total),
        "exact_match": exact_match_score(predictions, references),
        "bleu": bleu_score(predictions, references),
        "rouge_l": rouge_l_score(predictions, references),
        "token_f1": token_f1_score(predictions, references),
        "execution_accuracy": ratio(execution_matches, total),
        "sql_validity": ratio(valid_sql, total),
        "failure_rate": ratio(failures, total),
        "error_rate": ratio(errors, total),
        "empty_output_rate": ratio(empty_outputs, total),
        "non_sql_output_rate": ratio(non_sql_outputs, total),
        "load_time_seconds": load_time,
        "total_time_seconds": total_time,
        "latency_seconds_per_example": mean(generation_latencies),
        "generation_latency_mean": mean(generation_latencies),
        "generation_latency_p50": percentile(generation_latencies, 50),
        "generation_latency_p95": percentile(generation_latencies, 95),
        "evaluation_latency_mean": mean(evaluation_latencies),
        "throughput_examples_per_second": total / total_time if total_time else 0.0,
        "output_characters_mean": mean(
            [float(record["output_characters"]) for record in records]
        ),
    }


def select_sample_indices(
    dataset_size: int,
    sample_size: int,
    strategy: str,
    seed: int,
) -> list[int]:
    sample_size = min(sample_size, dataset_size)
    if strategy == "first":
        return list(range(sample_size))
    if strategy == "random":
        rng = random.Random(seed)
        return sorted(rng.sample(range(dataset_size), sample_size))
    raise ValueError(f"Unsupported sample strategy: {strategy}")


def build_benchmark_metadata(
    config: dict[str, Any],
    model_count: int,
    sample_size: int,
) -> dict[str, Any]:
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
        "mode": "zero-shot",
        "runner": "peft_lab.benchmark_zero_shot",
        "planned_framework": "Hugging Face LightEval custom task",
        "dataset": config["dataset"]["name"],
        "split": config["dataset"]["split"],
        "sample_size": sample_size,
        "calls_per_model": sample_size,
        "models_evaluated": model_count,
        "total_model_calls": sample_size * model_count,
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


def select_preview_records(
    records: list[dict[str, Any]],
    preview_limit: int,
) -> list[dict[str, Any]]:
    failures = [record for record in records if not record["execution_match"]]
    successes = [record for record in records if record["execution_match"]]
    return (successes[: preview_limit // 2] + failures[:preview_limit])[:preview_limit]


def ratio(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def mean(values: list[float]) -> float:
    return float(np.mean(values)) if values else 0.0


def percentile(values: list[float], q: float) -> float:
    return float(np.percentile(values, q)) if values else 0.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to benchmark YAML.")
    parser.add_argument(
        "--max-examples",
        type=int,
        default=None,
        help="Override configured sample size.",
    )
    parser.add_argument(
        "--model-id",
        action="append",
        default=None,
        help="Run only the selected model id. Can be passed more than once.",
    )
    return parser.parse_args()


def result_filename(model_id: str) -> str:
    return f"zero_shot_wikisql_{model_id}.json"


if __name__ == "__main__":
    main()
