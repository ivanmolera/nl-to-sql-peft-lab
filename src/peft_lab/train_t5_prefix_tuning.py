"""Train T5-small on WikiSQL with Prefix Tuning."""

from __future__ import annotations

import argparse
import json
import random
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from peft import PrefixTuningConfig, TaskType, get_peft_model
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)

from peft_lab.data import load_wikisql_split, prepare_seq2seq_dataset
from peft_lab.metrics import bleu_score, exact_match_score, rouge_l_score, token_f1_score
from peft_lab.training_utils import (
    add_best_model_metadata,
    add_training_run_metadata,
    best_metric_training_args,
    build_manual_best_peft_callback,
    parameter_efficiency_metrics,
    ResourceMonitor,
)


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    set_seed(config["experiment"]["seed"])

    output_dir = Path(config["training"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(config["model"]["name"])
    model = AutoModelForSeq2SeqLM.from_pretrained(config["model"]["name"])
    model = get_peft_model(model, build_prefix_config(config))
    if torch.cuda.is_available():
        model.to("cuda")
    model.print_trainable_parameters()
    parameter_metrics = parameter_efficiency_metrics(model)

    train_raw = load_wikisql_split(
        config["dataset"]["name"],
        config["dataset"]["train_split"],
        config["dataset"].get("train_limit"),
    )
    eval_raw = load_wikisql_split(
        config["dataset"]["name"],
        config["dataset"]["eval_split"],
        config["dataset"].get("eval_limit"),
    )
    train_dataset = prepare_seq2seq_dataset(
        train_raw,
        tokenizer,
        config["prompt"]["max_source_length"],
        config["prompt"]["max_target_length"],
    )
    eval_dataset = prepare_seq2seq_dataset(
        eval_raw,
        tokenizer,
        config["prompt"]["max_source_length"],
        config["prompt"]["max_target_length"],
    )

    best_callback = build_manual_best_peft_callback(config)
    trainer = Seq2SeqTrainer(
        model=model,
        args=build_training_args(config),
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=DataCollatorForSeq2Seq(
            tokenizer=tokenizer,
            model=model,
            label_pad_token_id=-100,
        ),
        tokenizer=tokenizer,
        compute_metrics=build_compute_metrics(tokenizer),
        callbacks=[best_callback],
    )

    resource_monitor = ResourceMonitor()
    resource_monitor.start()
    train_result = trainer.train()
    resource_metrics = resource_monitor.stop()
    metrics = best_callback.best_metrics or trainer.evaluate()
    metrics = add_best_model_metadata(metrics, trainer)
    metrics = add_training_run_metadata(
        metrics,
        train_result.metrics,
        resource_metrics,
        parameter_metrics,
    )
    save_best_adapter(best_callback.best_model_checkpoint, output_dir / "adapter", trainer)
    tokenizer.save_pretrained(output_dir / "adapter")
    save_json(output_dir / "eval_metrics.json", metrics)


def build_prefix_config(config: dict[str, Any]) -> PrefixTuningConfig:
    prefix = config["prefix_tuning"]
    return PrefixTuningConfig(
        task_type=TaskType.SEQ_2_SEQ_LM,
        num_virtual_tokens=prefix["num_virtual_tokens"],
        prefix_projection=prefix["prefix_projection"],
    )


def build_training_args(config: dict[str, Any]) -> Seq2SeqTrainingArguments:
    training = config["training"]
    return Seq2SeqTrainingArguments(
        output_dir=training["output_dir"],
        per_device_train_batch_size=training["per_device_train_batch_size"],
        per_device_eval_batch_size=training["per_device_eval_batch_size"],
        gradient_accumulation_steps=training["gradient_accumulation_steps"],
        learning_rate=training["learning_rate"],
        num_train_epochs=training["num_train_epochs"],
        logging_steps=training["logging_steps"],
        eval_steps=training["eval_steps"],
        save_steps=training.get("save_steps", training["eval_steps"]),
        save_total_limit=training.get("save_total_limit", 2),
        eval_strategy="steps",
        save_strategy="steps",
        predict_with_generate=training["predict_with_generate"],
        generation_max_length=training["generation_max_length"],
        report_to="none",
        fp16=torch.cuda.is_available() and not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
        **best_metric_training_args(training),
    )


def save_best_adapter(
    best_checkpoint: str | None,
    adapter_dir: Path,
    trainer: Seq2SeqTrainer,
) -> None:
    if best_checkpoint and Path(best_checkpoint).exists():
        if adapter_dir.exists():
            shutil.rmtree(adapter_dir)
        shutil.copytree(best_checkpoint, adapter_dir)
        return
    trainer.save_model(adapter_dir)


def build_compute_metrics(tokenizer: AutoTokenizer):
    def compute_metrics(eval_prediction) -> dict[str, float]:
        predictions, labels = eval_prediction
        if isinstance(predictions, tuple):
            predictions = predictions[0]
        predictions = sanitize_token_ids(predictions, tokenizer)
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        labels = sanitize_token_ids(labels, tokenizer)
        decoded_predictions = tokenizer.batch_decode(predictions, skip_special_tokens=True)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
        return {
            "exact_match": exact_match_score(decoded_predictions, decoded_labels),
            "bleu": bleu_score(decoded_predictions, decoded_labels),
            "rouge_l": rouge_l_score(decoded_predictions, decoded_labels),
            "token_f1": token_f1_score(decoded_predictions, decoded_labels),
        }

    return compute_metrics


def sanitize_token_ids(token_ids: Any, tokenizer: AutoTokenizer) -> np.ndarray:
    ids = np.asarray(token_ids)
    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = tokenizer.eos_token_id or 0
    vocab_size = len(tokenizer)
    return np.where((ids >= 0) & (ids < vocab_size), ids, pad_token_id)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to the experiment YAML.")
    return parser.parse_args()


def load_config(path: str) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as config_file:
        return yaml.safe_load(config_file)


def save_json(path: Path, payload: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as output_file:
        json.dump(payload, output_file, indent=2, sort_keys=True)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


if __name__ == "__main__":
    main()
