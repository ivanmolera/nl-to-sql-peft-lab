"""Dataset loading and tokenization for WikiSQL NL-to-SQL experiments."""

from __future__ import annotations

from typing import Any

from datasets import Dataset, load_dataset
from transformers import PreTrainedTokenizerBase

from peft_lab.sql import render_wikisql_query
from peft_lab.wikisql import get_table


def build_prompt(example: dict[str, Any]) -> str:
    table = get_table(example)
    columns = _format_columns(table)
    return (
        "Translate the question into SQL for the given table.\n"
        f"Table columns: {columns}\n"
        f"Question: {example['question']}\n"
        "SQL:"
    )


def load_wikisql_split(
    dataset_name: str,
    split: str,
    limit: int | None = None,
) -> Dataset:
    dataset = load_dataset(dataset_name, split=split, trust_remote_code=True)
    if limit:
        dataset = dataset.select(range(min(limit, len(dataset))))
    return dataset


def prepare_seq2seq_dataset(
    dataset: Dataset,
    tokenizer: PreTrainedTokenizerBase,
    max_source_length: int,
    max_target_length: int,
) -> Dataset:
    def tokenize_batch(batch: dict[str, list[Any]]) -> dict[str, Any]:
        prompts = [build_prompt(_row_from_batch(batch, index)) for index in range(len(batch["question"]))]
        targets = [
            render_wikisql_query(_row_from_batch(batch, index))
            for index in range(len(batch["question"]))
        ]

        model_inputs = tokenizer(
            prompts,
            max_length=max_source_length,
            truncation=True,
        )
        labels = tokenizer(
            text_target=targets,
            max_length=max_target_length,
            truncation=True,
        )
        model_inputs["labels"] = labels["input_ids"]
        model_inputs["target_sql"] = targets
        return model_inputs

    return dataset.map(tokenize_batch, batched=True)


def prepare_causal_completion_dataset(
    dataset: Dataset,
    tokenizer: PreTrainedTokenizerBase,
    max_source_length: int,
    max_target_length: int,
) -> Dataset:
    def tokenize_batch(batch: dict[str, list[Any]]) -> dict[str, Any]:
        input_ids = []
        attention_mask = []
        labels = []
        targets = []

        eos_token_id = tokenizer.eos_token_id
        for index in range(len(batch["question"])):
            row = _row_from_batch(batch, index)
            prompt = format_causal_prompt(tokenizer, build_prompt(row))
            target = render_wikisql_query(row)
            prompt_ids = tokenizer(
                prompt,
                add_special_tokens=True,
                max_length=max_source_length,
                truncation=True,
            )["input_ids"]
            target_ids = tokenizer(
                f" {target}",
                add_special_tokens=False,
                max_length=max_target_length,
                truncation=True,
            )["input_ids"]
            if eos_token_id is not None:
                target_ids = target_ids + [eos_token_id]

            ids = prompt_ids + target_ids
            input_ids.append(ids)
            attention_mask.append([1] * len(ids))
            labels.append([-100] * len(prompt_ids) + target_ids)
            targets.append(target)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "target_sql": targets,
        }

    return dataset.map(
        tokenize_batch,
        batched=True,
        remove_columns=dataset.column_names,
    )


def format_causal_prompt(tokenizer: PreTrainedTokenizerBase, prompt: str) -> str:
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


def _format_columns(table: dict[str, Any]) -> str:
    headers = table["header"]
    types = table.get("types") or ["text"] * len(headers)
    return " | ".join(
        f"{header} ({column_type})"
        for header, column_type in zip(headers, types)
    )


def _row_from_batch(batch: dict[str, list[Any]], index: int) -> dict[str, Any]:
    return {key: value[index] for key, value in batch.items()}
