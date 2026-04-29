"""Train T5-small on WikiSQL with QLoRA."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)

from peft_lab.data import load_wikisql_split, prepare_seq2seq_dataset
from peft_lab.metrics import exact_match_score


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    set_seed(config["experiment"]["seed"])

    if not torch.cuda.is_available():
        raise RuntimeError("QLoRA requires CUDA for bitsandbytes 4-bit training.")

    output_dir = Path(config["training"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(config["model"]["name"])
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16
        if torch.cuda.is_bf16_supported()
        else torch.float16,
    )
    model = AutoModelForSeq2SeqLM.from_pretrained(
        config["model"]["name"],
        quantization_config=quantization_config,
        device_map="auto",
    )
    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, build_lora_config(config))
    model.print_trainable_parameters()

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

    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        label_pad_token_id=-100,
    )
    training_args = build_training_args(config)
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=build_compute_metrics(tokenizer),
    )

    trainer.train()
    metrics = trainer.evaluate()
    trainer.save_model(output_dir / "adapter")
    tokenizer.save_pretrained(output_dir / "adapter")
    save_json(output_dir / "eval_metrics.json", metrics)


def build_lora_config(config: dict[str, Any]) -> LoraConfig:
    qlora = config["qlora"]
    return LoraConfig(
        r=qlora["r"],
        lora_alpha=qlora["alpha"],
        lora_dropout=qlora["dropout"],
        target_modules=qlora["target_modules"],
        bias="none",
        task_type=TaskType.SEQ_2_SEQ_LM,
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
        save_steps=training["save_steps"],
        save_total_limit=training["save_total_limit"],
        eval_strategy="steps",
        save_strategy="steps",
        predict_with_generate=training["predict_with_generate"],
        generation_max_length=training["generation_max_length"],
        report_to="none",
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
    )


def build_compute_metrics(tokenizer: AutoTokenizer):
    def compute_metrics(eval_prediction) -> dict[str, float]:
        predictions, labels = eval_prediction
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_predictions = tokenizer.batch_decode(
            predictions,
            skip_special_tokens=True,
        )
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
        return {"exact_match": exact_match_score(decoded_predictions, decoded_labels)}

    return compute_metrics


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
    torch.cuda.manual_seed_all(seed)


if __name__ == "__main__":
    main()
