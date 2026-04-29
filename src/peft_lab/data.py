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


def _format_columns(table: dict[str, Any]) -> str:
    headers = table["header"]
    types = table.get("types") or ["text"] * len(headers)
    return " | ".join(
        f"{header} ({column_type})"
        for header, column_type in zip(headers, types)
    )


def _row_from_batch(batch: dict[str, list[Any]], index: int) -> dict[str, Any]:
    return {key: value[index] for key, value in batch.items()}
