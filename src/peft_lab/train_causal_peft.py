"""Train decoder-only models on WikiSQL with PEFT."""

from __future__ import annotations

import argparse
import json
import random
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from peft import (
    IA3Config,
    LoraConfig,
    PrefixTuningConfig,
    TaskType,
    get_peft_model,
    prepare_model_for_kbit_training,
)
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments,
)

from peft_lab.data import load_wikisql_split, prepare_causal_completion_dataset
from peft_lab.training_utils import (
    ResourceMonitor,
    add_best_model_metadata,
    add_training_run_metadata,
    best_metric_training_args,
    best_model_training_args,
    build_early_stopping_callbacks,
    build_manual_best_peft_callback,
    parameter_efficiency_metrics,
)


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    set_seed(config["experiment"]["seed"])

    output_dir = Path(config["training"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(
        config["model"]["name"],
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    model = load_model(config)
    model.config.use_cache = False
    method = config["peft"]["method"]
    if method == "bitfit":
        enable_bitfit_parameters(model)
    else:
        model = get_peft_model(model, build_peft_config(config))
    if config["training"].get("gradient_checkpointing", False):
        model.gradient_checkpointing_enable()
        enable_input_require_grads(model)
    if torch.cuda.is_available() and method != "qlora":
        model.to("cuda")
    print_trainable_parameters(model)
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
    train_dataset = prepare_causal_completion_dataset(
        train_raw,
        tokenizer,
        config["prompt"]["max_source_length"],
        config["prompt"]["max_target_length"],
    )
    eval_dataset = prepare_causal_completion_dataset(
        eval_raw,
        tokenizer,
        config["prompt"]["max_source_length"],
        config["prompt"]["max_target_length"],
    )

    callbacks = build_callbacks(config)
    trainer = Trainer(
        model=model,
        args=build_training_args(config),
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=CausalCompletionCollator(tokenizer),
        tokenizer=tokenizer,
        callbacks=callbacks,
    )

    resource_monitor = ResourceMonitor()
    resource_monitor.start()
    train_result = trainer.train()
    resource_metrics = resource_monitor.stop()

    best_callback = next(
        (callback for callback in callbacks if hasattr(callback, "best_metrics")),
        None,
    )
    metrics = (
        best_callback.best_metrics
        if best_callback is not None and best_callback.best_metrics
        else trainer.evaluate()
    )
    metrics = add_best_model_metadata(metrics, trainer)
    metrics = add_training_run_metadata(
        metrics,
        train_result.metrics,
        resource_metrics,
        parameter_metrics,
    )

    adapter_dir = output_dir / "adapter"
    if method == "bitfit":
        save_bitfit_adapter(model, tokenizer, adapter_dir, config)
    elif best_callback is not None:
        save_best_adapter(best_callback.best_model_checkpoint, adapter_dir, trainer)
        tokenizer.save_pretrained(adapter_dir)
    else:
        trainer.save_model(adapter_dir)
        tokenizer.save_pretrained(adapter_dir)
    save_json(output_dir / "eval_metrics.json", metrics)


def load_model(config: dict[str, Any]):
    model_name = config["model"]["name"]
    method = config["peft"]["method"]
    common_kwargs = {"trust_remote_code": True}
    if method == "qlora":
        if not torch.cuda.is_available():
            raise RuntimeError("QLoRA requires CUDA for bitsandbytes 4-bit training.")
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16
            if torch.cuda.is_bf16_supported()
            else torch.float16,
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=quantization_config,
            device_map="auto",
            **common_kwargs,
        )
        return prepare_model_for_kbit_training(model)

    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    return AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
        **common_kwargs,
    )


def build_peft_config(config: dict[str, Any]):
    method = config["peft"]["method"]
    if method == "qlora":
        qlora = config["qlora"]
        return LoraConfig(
            r=qlora["r"],
            lora_alpha=qlora["alpha"],
            lora_dropout=qlora["dropout"],
            target_modules=qlora["target_modules"],
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )
    if method == "ia3":
        ia3 = config["ia3"]
        return IA3Config(
            task_type=TaskType.CAUSAL_LM,
            target_modules=ia3["target_modules"],
            feedforward_modules=ia3["feedforward_modules"],
        )
    if method == "prefix-tuning":
        prefix = config["prefix_tuning"]
        return PrefixTuningConfig(
            task_type=TaskType.CAUSAL_LM,
            num_virtual_tokens=prefix["num_virtual_tokens"],
            prefix_projection=prefix["prefix_projection"],
        )
    raise ValueError(f"Unsupported PEFT method: {method}")


def build_callbacks(config: dict[str, Any]) -> list:
    if config["peft"]["method"] == "prefix-tuning":
        return [build_manual_best_peft_callback(config)]
    return build_early_stopping_callbacks(config)


def build_training_args(config: dict[str, Any]) -> TrainingArguments:
    training = config["training"]
    best_args = (
        best_metric_training_args(training)
        if config["peft"]["method"] == "prefix-tuning"
        else best_model_training_args(training)
    )
    return TrainingArguments(
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
        report_to="none",
        fp16=torch.cuda.is_available() and not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
        remove_unused_columns=False,
        **best_args,
    )


@dataclass
class CausalCompletionCollator:
    tokenizer: Any

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        pad_token_id = self.tokenizer.pad_token_id
        if pad_token_id is None:
            pad_token_id = self.tokenizer.eos_token_id or 0
        max_length = max(len(feature["input_ids"]) for feature in features)
        batch = {"input_ids": [], "attention_mask": [], "labels": []}
        for feature in features:
            padding = max_length - len(feature["input_ids"])
            batch["input_ids"].append(feature["input_ids"] + [pad_token_id] * padding)
            batch["attention_mask"].append(feature["attention_mask"] + [0] * padding)
            batch["labels"].append(feature["labels"] + [-100] * padding)
        return {key: torch.tensor(value, dtype=torch.long) for key, value in batch.items()}


def enable_bitfit_parameters(model) -> None:
    for name, parameter in model.named_parameters():
        parameter.requires_grad = "bias" in name.lower()


def enable_input_require_grads(model) -> None:
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
        return

    input_embeddings = model.get_input_embeddings()

    def make_inputs_require_grad(_module, _inputs, output):
        output.requires_grad_(True)

    input_embeddings.register_forward_hook(make_inputs_require_grad)


def print_trainable_parameters(model) -> None:
    if hasattr(model, "print_trainable_parameters"):
        model.print_trainable_parameters()
        return
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    total = sum(parameter.numel() for parameter in model.parameters())
    ratio = 100 * trainable / total if total else 0
    print(f"trainable params: {trainable:,} || all params: {total:,} || trainable%: {ratio:.4f}")


def save_bitfit_adapter(model, tokenizer, adapter_dir: Path, config: dict[str, Any]) -> None:
    adapter_dir.mkdir(parents=True, exist_ok=True)
    bias_state = {
        name: parameter.detach().cpu()
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }
    torch.save(bias_state, adapter_dir / "bitfit_biases.pt")
    tokenizer.save_pretrained(adapter_dir)
    save_json(
        adapter_dir / "bitfit_config.json",
        {
            "base_model_name_or_path": config["model"]["name"],
            "peft_type": "BITFIT",
            "target": "parameters whose name contains 'bias'",
            "trainable_parameters": sorted(bias_state.keys()),
        },
    )


def save_best_adapter(best_checkpoint: str | None, adapter_dir: Path, trainer: Trainer) -> None:
    if best_checkpoint and Path(best_checkpoint).exists():
        if adapter_dir.exists():
            shutil.rmtree(adapter_dir)
        shutil.copytree(best_checkpoint, adapter_dir)
        return
    trainer.save_model(adapter_dir)


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
