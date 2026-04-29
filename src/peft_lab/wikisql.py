"""Compatibility helpers for WikiSQL dataset records."""

from __future__ import annotations

from typing import Any


def get_table(example: dict[str, Any]) -> dict[str, Any]:
    if isinstance(example.get("table"), dict):
        return example["table"]

    return {
        "header": example["header"],
        "types": example.get("types") or ["text"] * len(example["header"]),
        "rows": _extract_rows(example.get("rows", [])),
        "name": example.get("name", "table"),
    }


def get_sql_annotation(example: dict[str, Any]) -> dict[str, Any]:
    if isinstance(example.get("sql"), dict):
        return example["sql"]

    return {
        "sel": example["sel"],
        "agg": example["agg"],
        "conds": example.get("conds", []),
    }


def _extract_rows(rows: Any) -> list[list[Any]]:
    if isinstance(rows, dict) and "feature" in rows:
        return rows["feature"]
    return rows
