import unittest
from importlib.util import find_spec

if find_spec("datasets") is not None:
    from peft_lab.benchmark_zero_shot import aggregate_metrics, build_benchmark_metadata


@unittest.skipIf(find_spec("datasets") is None, "datasets is not installed")
class BenchmarkZeroShotTest(unittest.TestCase):
    def test_failure_rate_counts_failed_examples_once(self):
        records = [
            {
                "prediction": "",
                "error": "generation failed",
                "execution_match": False,
                "valid_sql": False,
                "output_characters": 0,
            },
            {
                "prediction": "SELECT count(*) FROM table",
                "error": None,
                "execution_match": True,
                "valid_sql": True,
                "output_characters": 26,
            },
        ]

        metrics = aggregate_metrics(
            records=records,
            predictions=["", "SELECT count(*) FROM table"],
            references=["SELECT name FROM table", "SELECT count(*) FROM table"],
            generation_latencies=[1.0, 2.0],
            evaluation_latencies=[0.1, 0.2],
            load_time=3.0,
            total_time=4.0,
        )

        self.assertEqual(metrics["failure_rate"], 0.5)

    def test_build_benchmark_metadata_counts_model_calls(self):
        metadata = build_benchmark_metadata(
            config={
                "experiment": {"seed": 42},
                "dataset": {
                    "name": "Salesforce/wikisql",
                    "split": "validation",
                    "sample_strategy": "random",
                },
                "prompt": {"max_source_length": 384, "max_new_tokens": 96},
                "generation": {"do_sample": False, "temperature": 0.0},
            },
            model_count=3,
            sample_size=30,
        )

        self.assertEqual(metadata["calls_per_model"], 30)
        self.assertEqual(metadata["total_model_calls"], 90)
        self.assertEqual(metadata["max_new_tokens"], 96)


if __name__ == "__main__":
    unittest.main()
