"""Run T5-small IA3 training, benchmark it, and optionally upload artifacts."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

from google.cloud import storage


DEFAULT_TRAIN_CONFIG = "configs/t5_small_wikisql_ia3_full.yaml"
DEFAULT_BENCHMARK_CONFIG = "configs/t5_small_wikisql_ia3_benchmark.yaml"
ARTIFACT_PATHS = [
    Path("outputs/t5-small-wikisql-ia3"),
    Path("benchmark_results/ia3"),
]


def main() -> None:
    train_config = os.environ.get("TRAIN_CONFIG", DEFAULT_TRAIN_CONFIG)
    benchmark_config = os.environ.get("BENCHMARK_CONFIG", DEFAULT_BENCHMARK_CONFIG)

    run(["python3", "-m", "peft_lab.train_t5_ia3", "--config", train_config])
    run(["python3", "-m", "peft_lab.benchmark_t5_ia3", "--config", benchmark_config])

    gcs_output_uri = os.environ.get("GCS_OUTPUT_URI")
    if gcs_output_uri:
        upload_artifacts(gcs_output_uri)


def run(command: list[str]) -> None:
    print(f"Running: {' '.join(command)}", flush=True)
    subprocess.run(command, check=True)


def upload_artifacts(gcs_output_uri: str) -> None:
    bucket_name, prefix = parse_gcs_uri(gcs_output_uri)
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    for root in ARTIFACT_PATHS:
        if not root.exists():
            continue
        for path in root.rglob("*"):
            if not path.is_file():
                continue
            destination = "/".join(
                part.strip("/")
                for part in [prefix, str(path)]
                if part.strip("/")
            )
            print(f"Uploading {path} to gs://{bucket_name}/{destination}", flush=True)
            bucket.blob(destination).upload_from_filename(path)


def parse_gcs_uri(uri: str) -> tuple[str, str]:
    if not uri.startswith("gs://"):
        raise ValueError("GCS_OUTPUT_URI must start with gs://")
    without_scheme = uri.removeprefix("gs://")
    bucket, _, prefix = without_scheme.partition("/")
    if not bucket:
        raise ValueError("GCS_OUTPUT_URI must include a bucket")
    return bucket, prefix


if __name__ == "__main__":
    main()
