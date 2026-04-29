import unittest

from peft_lab.execution import execution_match, is_valid_sql
from peft_lab.sql import render_wikisql_query


class WikiSqlExecutionTest(unittest.TestCase):
    def test_execution_match_for_rendered_reference(self):
        example = {
            "question": "What is the population of Spain?",
            "table": {
                "header": ["Country", "Population"],
                "types": ["text", "number"],
                "rows": [["Spain", "48"], ["France", "68"]],
            },
            "sql": {
                "sel": 1,
                "agg": 0,
                "conds": {
                    "column_index": [0],
                    "operator_index": [0],
                    "condition": ["Spain"],
                },
            },
        }
        sql = render_wikisql_query(example)

        self.assertTrue(is_valid_sql(sql, example))
        self.assertTrue(execution_match(sql, sql, example))


if __name__ == "__main__":
    unittest.main()
