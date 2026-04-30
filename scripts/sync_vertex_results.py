"""Download Vertex AI training artifacts that should be bundled into the web UI."""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path


DEFAULT_GCS_RUN_URI = (
    "gs://nl-sql-peft-lab-ivan-0429-training/runs/t5-small-wikisql-qlora-v0.1.0"
)
DEFAULT_ARTIFACTS = [
    "benchmark_results/qlora",
    "outputs/t5-small-wikisql-qlora",
]


def main() -> None:
    args = parse_args()
    for artifact in args.artifact:
        download_artifact(args.gcs_run_uri, artifact, args.project)


def download_artifact(gcs_run_uri: str, artifact: str, project: str | None) -> None:
    source = f"{gcs_run_uri.rstrip('/')}/{artifact.strip('/')}"
    destination = Path(artifact).parent
    destination.mkdir(parents=True, exist_ok=True)

    command = ["gcloud", "storage", "cp", "--recursive", source, str(destination)]
    if project:
        command.extend(["--project", project])

    print(f"Syncing {source} -> {destination}", flush=True)
    subprocess.run(command, check=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--gcs-run-uri",
        default=DEFAULT_GCS_RUN_URI,
        help="GCS URI used as GCS_OUTPUT_URI by the Vertex AI training job.",
    )
    parser.add_argument(
        "--project",
        default="nl-sql-peft-lab-ivan-0429",
        help="Google Cloud project used by gcloud storage.",
    )
    parser.add_argument(
        "--artifact",
        action="append",
        default=DEFAULT_ARTIFACTS,
        help="Artifact path inside the run URI. Can be passed multiple times.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    main()
