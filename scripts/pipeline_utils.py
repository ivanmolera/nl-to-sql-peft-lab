"""Shared helpers for Vertex AI pipeline wrapper scripts."""

from __future__ import annotations

from pathlib import Path

import yaml
from google.cloud import storage


def artifact_paths(train_config: str, benchmark_config: str) -> list[Path]:
    paths: list[Path] = []
    train = load_yaml(train_config)
    benchmark = load_yaml(benchmark_config)
    training_output = train.get("training", {}).get("output_dir")
    benchmark_output = benchmark.get("output", {}).get("dir")
    for value in [training_output, benchmark_output]:
        if value:
            paths.append(Path(value))
    return paths


def upload_artifacts(gcs_output_uri: str, paths: list[Path]) -> None:
    bucket_name, prefix = parse_gcs_uri(gcs_output_uri)
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    for root in paths:
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


def load_yaml(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as config_file:
        return yaml.safe_load(config_file)
