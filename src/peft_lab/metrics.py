"""Evaluation helpers."""

from __future__ import annotations

import math
import re
from collections import Counter

from peft_lab.sql import normalize_sql


def exact_match_score(predictions: list[str], references: list[str]) -> float:
    if not references:
        return 0.0

    matches = sum(
        normalize_sql(prediction) == normalize_sql(reference)
        for prediction, reference in zip(predictions, references)
    )
    return matches / len(references)


def bleu_score(predictions: list[str], references: list[str]) -> float:
    """Mean smoothed BLEU-4 over normalized SQL strings."""
    return mean_score(
        sentence_bleu(sql_tokens(prediction), sql_tokens(reference))
        for prediction, reference in zip(predictions, references)
    )


def rouge_l_score(predictions: list[str], references: list[str]) -> float:
    """Mean ROUGE-L F1 over normalized SQL strings."""
    return mean_score(
        rouge_l_f1(sql_tokens(prediction), sql_tokens(reference))
        for prediction, reference in zip(predictions, references)
    )


def token_f1_score(predictions: list[str], references: list[str]) -> float:
    """Mean bag-of-token F1 over normalized SQL strings."""
    return mean_score(
        token_f1(sql_tokens(prediction), sql_tokens(reference))
        for prediction, reference in zip(predictions, references)
    )


def sql_tokens(value: str) -> list[str]:
    normalized = normalize_sql(value).lower()
    return re.findall(r"\"[^\"]+\"|'[^']+'|\w+|<>|<=|>=|!=|[=<>(),*+-]", normalized)


def sentence_bleu(
    prediction_tokens: list[str],
    reference_tokens: list[str],
    max_order: int = 4,
) -> float:
    if not prediction_tokens or not reference_tokens:
        return 0.0

    precisions = []
    for order in range(1, max_order + 1):
        prediction_counts = ngram_counts(prediction_tokens, order)
        reference_counts = ngram_counts(reference_tokens, order)
        overlap = sum(
            min(count, reference_counts[ngram])
            for ngram, count in prediction_counts.items()
        )
        total = sum(prediction_counts.values())
        precisions.append((overlap + 1.0) / (total + 1.0) if total else 1.0)

    brevity_penalty = 1.0
    if len(prediction_tokens) < len(reference_tokens):
        brevity_penalty = math.exp(1.0 - len(reference_tokens) / len(prediction_tokens))

    return brevity_penalty * math.exp(
        sum(math.log(precision) for precision in precisions) / max_order
    )


def rouge_l_f1(prediction_tokens: list[str], reference_tokens: list[str]) -> float:
    if not prediction_tokens or not reference_tokens:
        return 0.0
    lcs = lcs_length(prediction_tokens, reference_tokens)
    precision = lcs / len(prediction_tokens)
    recall = lcs / len(reference_tokens)
    return f1(precision, recall)


def token_f1(prediction_tokens: list[str], reference_tokens: list[str]) -> float:
    if not prediction_tokens or not reference_tokens:
        return 0.0
    prediction_counts = Counter(prediction_tokens)
    reference_counts = Counter(reference_tokens)
    overlap = sum((prediction_counts & reference_counts).values())
    precision = overlap / len(prediction_tokens)
    recall = overlap / len(reference_tokens)
    return f1(precision, recall)


def ngram_counts(tokens: list[str], order: int) -> Counter[tuple[str, ...]]:
    return Counter(
        tuple(tokens[index : index + order])
        for index in range(0, max(0, len(tokens) - order + 1))
    )


def lcs_length(left: list[str], right: list[str]) -> int:
    previous = [0] * (len(right) + 1)
    for left_token in left:
        current = [0]
        for index, right_token in enumerate(right, start=1):
            if left_token == right_token:
                current.append(previous[index - 1] + 1)
            else:
                current.append(max(previous[index], current[-1]))
        previous = current
    return previous[-1]


def f1(precision: float, recall: float) -> float:
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def mean_score(values) -> float:
    values = list(values)
    return sum(values) / len(values) if values else 0.0
