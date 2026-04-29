"""WikiSQL canonical SQL rendering."""

from __future__ import annotations

import re
from typing import Any

from peft_lab.wikisql import get_sql_annotation, get_table

AGGREGATORS = ["", "MAX", "MIN", "COUNT", "SUM", "AVG"]
OPERATORS = ["=", ">", "<", "OP"]


def normalize_sql(sql: str) -> str:
    """Normalize generated SQL for rough exact-match comparisons."""
    sql = sql.strip().rstrip(";")
    sql = re.sub(r"\s+", " ", sql)
    sql = re.sub(r"\s*,\s*", ", ", sql)
    sql = re.sub(r"\s*\(\s*", "(", sql)
    sql = re.sub(r"\s*\)", ")", sql)
    return sql.strip().lower()


def quote_identifier(identifier: Any) -> str:
    value = str(identifier).strip().replace('"', '""')
    return f'"{value}"'


def quote_literal(value: Any) -> str:
    if value is None:
        return "NULL"
    if isinstance(value, (int, float)):
        return str(value)

    text = str(value).strip()
    if _looks_number_like(text):
        return text
    return "'" + text.replace("'", "''") + "'"


def render_wikisql_query(example: dict[str, Any]) -> str:
    """Render the structured WikiSQL annotation into a deterministic SQL string."""
    table = get_table(example)
    headers = table["header"]
    sql = get_sql_annotation(example)

    selected_column = quote_identifier(headers[sql["sel"]])
    aggregator = AGGREGATORS[sql["agg"]]
    select_expression = (
        f"{aggregator}({selected_column})" if aggregator else selected_column
    )

    query = f"SELECT {select_expression} FROM table"
    conditions = _iter_conditions(sql.get("conds", []))
    rendered_conditions = [
        f"{quote_identifier(headers[column_index])} {operator} {quote_literal(value)}"
        for column_index, operator_index, value in conditions
        for operator in [OPERATORS[operator_index]]
    ]
    if rendered_conditions:
        query += " WHERE " + " AND ".join(rendered_conditions)

    return query


def _iter_conditions(conditions: Any) -> list[tuple[int, int, Any]]:
    if isinstance(conditions, dict):
        columns = conditions.get("column_index", [])
        operators = conditions.get("operator_index", [])
        values = conditions.get("condition", [])
        return [
            (int(column), int(operator), value)
            for column, operator, value in zip(columns, operators, values)
        ]

    return [
        (int(condition[0]), int(condition[1]), condition[2])
        for condition in conditions
    ]


def _looks_number_like(value: str) -> bool:
    return bool(re.fullmatch(r"[-+]?\d+(\.\d+)?", value))
