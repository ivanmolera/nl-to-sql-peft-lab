"""Evaluate selected base models on WikiSQL without fine-tuning."""

from __future__ import annotations

import argparse
import json
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from transformers import AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoTokenizer

from peft_lab.data import build_prompt, load_wikisql_split
from peft_lab.execution import execution_match, is_valid_sql
from peft_lab.metrics import (
    bleu_score,
    exact_match_score,
    rouge_l_score,
    token_f1_score,
)
from peft_lab.sql import normalize_sql, render_wikisql_query


@dataclass
class ModelSpec:
    id: str
    name: str
    architecture: str
    role: str


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    set_seed(config["experiment"]["seed"])

    dataset = load_wikisql_split(
        config["dataset"]["name"],
        config["dataset"]["split"],
        args.max_examples or config["dataset"].get("eval_limit"),
    )
    examples = [dataset[index] for index in range(len(dataset))]
    model_results = [
        evaluate_model(ModelSpec(**model_config), examples, config)
        for model_config in config["models"]
    ]

    payload = {
        "experiment": config["experiment"],
        "dataset": {
            "name": config["dataset"]["name"],
            "split": config["dataset"]["split"],
            "examples": len(examples),
        },
        "mode": "zero-shot",
        "models": model_results,
    }
    output_path = Path(config["output"]["path"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Saved baseline results to {output_path}")


def evaluate_model(
    model_spec: ModelSpec,
    examples: list[dict[str, Any]],
    config: dict[str, Any],
) -> dict[str, Any]:
    print(f"Evaluating {model_spec.name}")
    tokenizer = AutoTokenizer.from_pretrained(model_spec.name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = load_model(model_spec, config)
    model.eval()

    predictions: list[str] = []
    references: list[str] = []
    valid_count = 0
    execution_count = 0
    example_rows = []
    started_at = time.perf_counter()

    for example in examples:
        reference_sql = render_wikisql_query(example)
        prediction = generate_sql(model, tokenizer, model_spec, example, config)
        predictions.append(prediction)
        references.append(reference_sql)

        valid = is_valid_sql(prediction, example)
        execution_ok = execution_match(prediction, reference_sql, example)
        valid_count += int(valid)
        execution_count += int(execution_ok)

        if len(example_rows) < 10:
            example_rows.append(
                {
                    "question": example["question"],
                    "prediction": prediction,
                    "reference": reference_sql,
                    "exact_match": normalize_sql(prediction)
                    == normalize_sql(reference_sql),
                    "valid_sql": valid,
                    "execution_match": execution_ok,
                }
            )

    elapsed = time.perf_counter() - started_at
    total = len(examples)
    return {
        "id": model_spec.id,
        "name": model_spec.name,
        "architecture": model_spec.architecture,
        "role": model_spec.role,
        "metrics": {
            "exact_match": exact_match_score(predictions, references),
            "bleu": bleu_score(predictions, references),
            "rouge_l": rouge_l_score(predictions, references),
            "token_f1": token_f1_score(predictions, references),
            "execution_accuracy": execution_count / total if total else 0.0,
            "sql_validity": valid_count / total if total else 0.0,
            "latency_seconds_per_example": elapsed / total if total else 0.0,
        },
        "examples": example_rows,
    }


def load_model(model_spec: ModelSpec, config: dict[str, Any]):
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    common_kwargs = {
        "torch_dtype": torch_dtype,
        "trust_remote_code": True,
    }
    if torch.cuda.is_available():
        common_kwargs["device_map"] = "auto"

    if model_spec.architecture == "seq2seq":
        return AutoModelForSeq2SeqLM.from_pretrained(model_spec.name, **common_kwargs)
    if model_spec.architecture == "causal":
        return AutoModelForCausalLM.from_pretrained(model_spec.name, **common_kwargs)
    raise ValueError(f"Unsupported architecture: {model_spec.architecture}")


def generate_sql(
    model,
    tokenizer,
    model_spec: ModelSpec,
    example: dict[str, Any],
    config: dict[str, Any],
) -> str:
    prompt = build_prompt(example)
    if model_spec.architecture == "causal":
        prompt = format_causal_prompt(tokenizer, prompt)

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=config["prompt"]["max_source_length"],
    ).to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=config["prompt"]["max_new_tokens"],
            do_sample=config["generation"]["do_sample"],
            temperature=config["generation"]["temperature"]
            if config["generation"]["do_sample"]
            else None,
            pad_token_id=tokenizer.pad_token_id,
        )

    if model_spec.architecture == "causal":
        generated_tokens = outputs[0][inputs["input_ids"].shape[-1] :]
        decoded = tokenizer.decode(generated_tokens, skip_special_tokens=True)
    else:
        decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return extract_sql(decoded)


def format_causal_prompt(tokenizer, prompt: str) -> str:
    messages = [
        {
            "role": "system",
            "content": "You generate one SQLite SQL query and no explanation.",
        },
        {"role": "user", "content": prompt},
    ]
    if getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    return (
        "System: You generate one SQLite SQL query and no explanation.\n"
        f"User: {prompt}\n"
        "Assistant:"
    )


def extract_sql(text: str) -> str:
    text = text.strip()
    if "```" in text:
        chunks = [chunk.strip() for chunk in text.split("```") if chunk.strip()]
        text = next((chunk for chunk in chunks if "select" in chunk.lower()), text)
        text = text.removeprefix("sql").strip()

    select_index = text.lower().find("select")
    if select_index >= 0:
        text = text[select_index:]
    first_line = text.splitlines()[0].strip() if text else ""
    return first_line.rstrip(";")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to the baseline YAML.")
    parser.add_argument(
        "--max-examples",
        type=int,
        default=None,
        help="Override the configured evaluation size.",
    )
    return parser.parse_args()


def load_config(path: str) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as config_file:
        return yaml.safe_load(config_file)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


if __name__ == "__main__":
    main()
