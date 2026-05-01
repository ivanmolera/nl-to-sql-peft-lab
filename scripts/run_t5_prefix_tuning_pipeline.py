"""Run T5-small Prefix Tuning training, benchmark it, and optionally upload artifacts."""

from __future__ import annotations

import os
import subprocess

from pipeline_utils import artifact_paths, upload_artifacts


DEFAULT_TRAIN_CONFIG = "configs/t5_small_wikisql_prefix_tuning_full.yaml"
DEFAULT_BENCHMARK_CONFIG = "configs/t5_small_wikisql_prefix_tuning_benchmark.yaml"


def main() -> None:
    train_config = os.environ.get("TRAIN_CONFIG", DEFAULT_TRAIN_CONFIG)
    benchmark_config = os.environ.get("BENCHMARK_CONFIG", DEFAULT_BENCHMARK_CONFIG)

    run(["python3", "-m", "peft_lab.train_t5_prefix_tuning", "--config", train_config])
    run(["python3", "-m", "peft_lab.benchmark_t5_prefix_tuning", "--config", benchmark_config])

    gcs_output_uri = os.environ.get("GCS_OUTPUT_URI")
    if gcs_output_uri:
        upload_artifacts(gcs_output_uri, artifact_paths(train_config, benchmark_config))


def run(command: list[str]) -> None:
    print(f"Running: {' '.join(command)}", flush=True)
    subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
