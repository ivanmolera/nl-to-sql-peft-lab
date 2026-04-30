import unittest

from peft_lab.metrics import bleu_score, rouge_l_score, token_f1_score


class GenerativeMetricsTest(unittest.TestCase):
    def test_identical_sql_scores_one(self):
        prediction = 'SELECT "name" FROM table WHERE "year" = 2020'
        reference = 'SELECT "name" FROM table WHERE "year" = 2020'

        self.assertAlmostEqual(bleu_score([prediction], [reference]), 1.0)
        self.assertAlmostEqual(rouge_l_score([prediction], [reference]), 1.0)
        self.assertAlmostEqual(token_f1_score([prediction], [reference]), 1.0)

    def test_empty_prediction_scores_zero(self):
        reference = 'SELECT "name" FROM table'

        self.assertEqual(bleu_score([""], [reference]), 0.0)
        self.assertEqual(rouge_l_score([""], [reference]), 0.0)
        self.assertEqual(token_f1_score([""], [reference]), 0.0)


if __name__ == "__main__":
    unittest.main()
