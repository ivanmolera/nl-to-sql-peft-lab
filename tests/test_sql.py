import unittest

from peft_lab.sql import normalize_sql, render_wikisql_query


class WikiSqlRenderingTest(unittest.TestCase):
    def test_render_wikisql_query_with_condition(self):
        example = {
            "question": "What is the population of Spain?",
            "table": {
                "header": ["Country", "Population"],
                "types": ["text", "number"],
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

        self.assertEqual(
            render_wikisql_query(example),
            'SELECT "Population" FROM table WHERE "Country" = \'Spain\'',
        )

    def test_normalize_sql_ignores_case_spacing_and_trailing_semicolon(self):
        left = ' select  MAX ( "Points" ) from table ; '
        right = 'SELECT MAX("Points") FROM table'

        self.assertEqual(normalize_sql(left), normalize_sql(right))


if __name__ == "__main__":
    unittest.main()
