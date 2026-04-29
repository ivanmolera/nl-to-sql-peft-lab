"""Evaluation helpers."""

from __future__ import annotations

from peft_lab.sql import normalize_sql


def exact_match_score(predictions: list[str], references: list[str]) -> float:
    if not references:
        return 0.0

    matches = sum(
        normalize_sql(prediction) == normalize_sql(reference)
        for prediction, reference in zip(predictions, references)
    )
    return matches / len(references)
