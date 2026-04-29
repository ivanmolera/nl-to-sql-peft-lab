"""SQLite execution utilities for WikiSQL examples."""

from __future__ import annotations

import re
import sqlite3
from typing import Any

from peft_lab.wikisql import get_table

TABLE_NAME = "wikisql_table"


def execution_match(predicted_sql: str, reference_sql: str, example: dict[str, Any]) -> bool:
    """Compare predicted and reference SQL by executing both against the WikiSQL table."""
    try:
        predicted_rows = execute_wikisql(predicted_sql, example)
        reference_rows = execute_wikisql(reference_sql, example)
    except sqlite3.Error:
        return False
    return predicted_rows == reference_rows


def is_valid_sql(sql: str, example: dict[str, Any]) -> bool:
    try:
        execute_wikisql(sql, example)
    except sqlite3.Error:
        return False
    return True


def execute_wikisql(sql: str, example: dict[str, Any]) -> list[tuple[Any, ...]]:
    table = get_table(example)
    headers = table["header"]
    rows = table["rows"]

    connection = sqlite3.connect(":memory:")
    try:
        create_columns = ", ".join(
            f"{quote_identifier(header)} TEXT" for header in headers
        )
        connection.execute(f"CREATE TABLE {TABLE_NAME} ({create_columns})")

        placeholders = ", ".join(["?"] * len(headers))
        insert_sql = f"INSERT INTO {TABLE_NAME} VALUES ({placeholders})"
        connection.executemany(insert_sql, rows)

        executable_sql = rewrite_table_name(sql)
        return list(connection.execute(executable_sql))
    finally:
        connection.close()


def rewrite_table_name(sql: str) -> str:
    sql = sql.strip().rstrip(";")
    return re.sub(r"\bfrom\s+table\b", f"FROM {TABLE_NAME}", sql, flags=re.IGNORECASE)


def quote_identifier(identifier: Any) -> str:
    return '"' + str(identifier).replace('"', '""') + '"'
