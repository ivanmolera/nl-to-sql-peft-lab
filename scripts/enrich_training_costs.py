"""Add reproducible Vertex AI training cost estimates to benchmark JSON files."""

from __future__ import annotations

import argparse
import json
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml


ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_PRICING = ROOT_DIR / "configs" / "pricing" / "gcp_vertex_ai_europe_west4.yaml"
DEFAULT_RESULTS = [
    ROOT_DIR / "benchmark_results" / "qlora" / "qlora_wikisql_index.json",
    ROOT_DIR / "benchmark_results" / "bitfit" / "bitfit_wikisql_index.json",
    ROOT_DIR / "benchmark_results" / "prefix_tuning" / "prefix_tuning_wikisql_index.json",
    ROOT_DIR / "benchmark_results" / "ia3" / "ia3_wikisql_index.json",
    ROOT_DIR / "benchmark_results" / "v0_2" / "qlora" / "qlora_wikisql_index.json",
    ROOT_DIR / "benchmark_results" / "v0_2" / "bitfit" / "bitfit_wikisql_index.json",
    ROOT_DIR / "benchmark_results" / "v0_2" / "prefix_tuning" / "prefix_tuning_wikisql_index.json",
    ROOT_DIR / "benchmark_results" / "v0_2" / "ia3" / "ia3_wikisql_index.json",
]

MODEL_SLUGS = {
    "t5-small": "t5-small",
    "smollm2": "smollm2",
    "qwen2-5": "qwen2-5",
    "qwen2.5": "qwen2-5",
    "gpt2": "gpt2",
}

PEFT_SLUGS = {
    "qlora": "qlora",
    "bitfit": "bitfit",
    "prefix-tuning": "prefix-tuning",
    "prefix_tuning": "prefix-tuning",
    "ia3": "ia3",
}


@dataclass(frozen=True)
class JobMetadata:
    name: str | None
    display_name: str | None
    state: str | None
    start_time: str | None
    end_time: str | None
    machine_type: str | None
    accelerator_type: str | None
    accelerator_count: int
    boot_disk_type: str | None
    boot_disk_size_gb: int | None
    replica_count: int


def main() -> None:
    args = parse_args()
    pricing = load_yaml(args.pricing)
    job_index = discover_vertex_jobs(args) if args.discover_vertex_jobs else {}
    explicit_jobs = load_job_manifest(args.job_manifest) if args.job_manifest else {}

    changed = False
    for path in args.result:
        if not path.exists():
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        if enrich_payload(payload, pricing, job_index, explicit_jobs):
            changed = True
            if args.dry_run:
                print(f"Would update {relative(path)}")
            else:
                path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
                print(f"Updated {relative(path)}")
        if sync_model_result_files(path, payload, args.dry_run):
            changed = True

    if not changed:
        print("No benchmark files needed cost enrichment.")


def enrich_payload(
    payload: dict[str, Any],
    pricing: dict[str, Any],
    job_index: dict[tuple[str, str], JobMetadata],
    explicit_jobs: dict[str, JobMetadata],
) -> bool:
    changed = False
    for model in payload.get("models", []):
        training = model.get("training") or {}
        trainer_metrics = training.get("trainer_eval_metrics") or {}
        resources = trainer_metrics.get("resource_metrics") or {}
        if not training and not resources:
            continue

        model_slug = model_slug_for(model)
        peft_slug = peft_slug_for(training, payload)
        job = explicit_jobs.get(model.get("id")) or job_index.get((model_slug, peft_slug))
        cost = build_cost_estimate(model, training, resources, pricing, job)
        if not cost:
            continue
        if training.get("cost_estimate") != cost:
            training["cost_estimate"] = cost
            model["training"] = training
            changed = True

    return changed


def sync_model_result_files(index_path: Path, payload: dict[str, Any], dry_run: bool) -> bool:
    changed = False
    for model in payload.get("models", []):
        cost = ((model.get("training") or {}).get("cost_estimate")) or {}
        result_file = model.get("result_file")
        if not cost or not result_file:
            continue
        for result_path in resolve_result_paths(index_path, result_file):
            result_payload = json.loads(result_path.read_text(encoding="utf-8"))
            training = result_payload.setdefault("training", {})
            if training.get("cost_estimate") == cost:
                continue
            training["cost_estimate"] = cost
            changed = True
            if dry_run:
                print(f"Would update {relative(result_path)}")
            else:
                result_path.write_text(json.dumps(result_payload, indent=2) + "\n", encoding="utf-8")
                print(f"Updated {relative(result_path)}")
    return changed


def resolve_result_paths(index_path: Path, result_file: str) -> list[Path]:
    path = Path(result_file)
    if path.is_absolute():
        return [path] if path.exists() else []
    candidates = [ROOT_DIR / path, index_path.parent / path.name]
    existing: list[Path] = []
    for candidate in candidates:
        if candidate.exists() and candidate not in existing:
            existing.append(candidate)
    return existing


def build_cost_estimate(
    model: dict[str, Any],
    training: dict[str, Any],
    resources: dict[str, Any],
    pricing: dict[str, Any],
    job: JobMetadata | None,
) -> dict[str, Any] | None:
    runtime = model.get("runtime") or {}
    duration_seconds, duration_source = billable_duration_seconds(job, resources)
    if duration_seconds is None:
        return None

    machine_type = first_present(
        job.machine_type if job else None,
        runtime.get("machine_type"),
        training.get("machine_type"),
    )
    accelerator_type = first_present(
        job.accelerator_type if job else None,
        runtime.get("accelerator_type"),
        resources.get("cuda_device_name"),
    )
    accelerator_count = int(first_present(job.accelerator_count if job else None, runtime.get("accelerator_count"), 0) or 0)
    boot_disk_type = first_present(job.boot_disk_type if job else None, training.get("boot_disk_type"), "pd-ssd")
    boot_disk_size_gb = int(first_present(job.boot_disk_size_gb if job else None, training.get("boot_disk_size_gb"), 100) or 0)
    replica_count = int(first_present(job.replica_count if job else None, 1) or 1)

    machine_price = hourly_price(pricing, "machine_hourly_usd", machine_type)
    accelerator_price = hourly_price(pricing, "accelerator_hourly_usd", accelerator_type)
    disk_price = hourly_price(pricing, "disk_hourly_usd_per_gb", boot_disk_type)
    if machine_price is None and accelerator_price is None and disk_price is None:
        return None

    hours = duration_seconds / 3600
    machine_cost = hours * (machine_price or 0) * replica_count
    accelerator_cost = hours * (accelerator_price or 0) * accelerator_count * replica_count
    disk_cost = hours * (disk_price or 0) * boot_disk_size_gb * replica_count
    total = machine_cost + accelerator_cost + disk_cost

    return {
        "provider": pricing.get("provider"),
        "region": pricing.get("region") or runtime.get("region"),
        "currency": pricing.get("currency", "USD"),
        "pricing_model": pricing.get("pricing_model", "on_demand"),
        "cost_type": "estimated",
        "status": "complete" if job is None or job.state == "JOB_STATE_SUCCEEDED" else "partial",
        "job_name": job.name if job else None,
        "job_display_name": job.display_name if job else None,
        "job_state": job.state if job else None,
        "job_start_time": job.start_time if job else None,
        "job_end_time": job.end_time if job else None,
        "duration_source": duration_source,
        "billable_duration_seconds": round(duration_seconds, 3),
        "billable_duration_hours": round(hours, 6),
        "machine_type": machine_type,
        "machine_hourly_usd": machine_price,
        "machine_cost_usd": round(machine_cost, 6),
        "accelerator_type": accelerator_type,
        "accelerator_count": accelerator_count,
        "accelerator_hourly_usd": accelerator_price,
        "accelerator_cost_usd": round(accelerator_cost, 6),
        "boot_disk_type": boot_disk_type,
        "boot_disk_size_gb": boot_disk_size_gb,
        "boot_disk_hourly_usd_per_gb": disk_price,
        "boot_disk_cost_usd": round(disk_cost, 6),
        "replica_count": replica_count,
        "estimated_total_usd": round(total, 6),
        "pricing_source": pricing.get("source"),
        "notes": pricing.get("notes", []),
    }


def billable_duration_seconds(job: JobMetadata | None, resources: dict[str, Any]) -> tuple[float | None, str | None]:
    if job and job.start_time:
        start = parse_time(job.start_time)
        end = parse_time(job.end_time) if job.end_time else datetime.now(timezone.utc)
        return max(0.0, (end - start).total_seconds()), "vertex_ai_job_time"
    if resources.get("training_wall_time_seconds") is not None:
        return float(resources["training_wall_time_seconds"]), "training_resource_monitor"
    if resources.get("training_wall_time_minutes") is not None:
        return float(resources["training_wall_time_minutes"]) * 60, "training_resource_monitor"
    return None, None


def discover_vertex_jobs(args: argparse.Namespace) -> dict[tuple[str, str], JobMetadata]:
    command = [
        "gcloud",
        "ai",
        "custom-jobs",
        "list",
        f"--region={args.region}",
        f"--project={args.project}",
        f"--limit={args.job_limit}",
        "--sort-by=~createTime",
        "--format=json",
    ]
    if args.job_filter:
        command.append(f"--filter={args.job_filter}")
    jobs = json.loads(subprocess.check_output(command, text=True))
    index: dict[tuple[str, str], JobMetadata] = {}
    for raw_job in jobs:
        metadata = metadata_from_job(raw_job)
        key = key_from_display_name(metadata.display_name)
        if key is None:
            continue
        current = index.get(key)
        if current is None or is_better_job(metadata, current):
            index[key] = metadata
    return index


def load_job_manifest(path: Path) -> dict[str, JobMetadata]:
    payload = load_yaml(path)
    jobs: dict[str, JobMetadata] = {}
    for item in payload.get("jobs", []):
        model_id = item["model_id"]
        job = item.get("job") or {}
        jobs[model_id] = metadata_from_job(job)
    return jobs


def metadata_from_job(job: dict[str, Any]) -> JobMetadata:
    worker_pool = ((job.get("jobSpec") or {}).get("workerPoolSpecs") or [{}])[0]
    machine = worker_pool.get("machineSpec") or {}
    disk = worker_pool.get("diskSpec") or {}
    return JobMetadata(
        name=job.get("name"),
        display_name=job.get("displayName"),
        state=job.get("state"),
        start_time=job.get("startTime"),
        end_time=job.get("endTime"),
        machine_type=machine.get("machineType"),
        accelerator_type=machine.get("acceleratorType"),
        accelerator_count=int(machine.get("acceleratorCount") or 0),
        boot_disk_type=disk.get("bootDiskType"),
        boot_disk_size_gb=int(disk.get("bootDiskSizeGb") or 0) or None,
        replica_count=int(worker_pool.get("replicaCount") or 1),
    )


def is_better_job(candidate: JobMetadata, current: JobMetadata) -> bool:
    if candidate.state == "JOB_STATE_SUCCEEDED" and current.state != "JOB_STATE_SUCCEEDED":
        return True
    if candidate.start_time and current.start_time:
        return candidate.start_time > current.start_time
    return False


def key_from_display_name(display_name: str | None) -> tuple[str, str] | None:
    text = (display_name or "").lower()
    if "wikisql" not in text or "early-stopping" not in text:
        return None
    model = next((slug for token, slug in MODEL_SLUGS.items() if token in text), None)
    peft = next((slug for token, slug in PEFT_SLUGS.items() if token in text), None)
    if not model or not peft:
        return None
    return model, peft


def model_slug_for(model: dict[str, Any]) -> str:
    text = " ".join(str(value) for value in [model.get("id"), model.get("name"), model.get("base_model_name")] if value).lower()
    for token, slug in MODEL_SLUGS.items():
        if token in text:
            return slug
    return text


def peft_slug_for(training: dict[str, Any], payload: dict[str, Any]) -> str:
    text = normalize_price_key(str(training.get("technique") or payload.get("mode") or "")).replace("_", "-")
    for token, slug in PEFT_SLUGS.items():
        if token.replace("_", "-") in text:
            return slug
    return text


def hourly_price(pricing: dict[str, Any], section: str, key: str | None) -> float | None:
    if not key:
        return None
    table = pricing.get(section) or {}
    if key in table:
        return float(table[key])
    normalized_key = normalize_price_key(key)
    for candidate, value in table.items():
        if normalize_price_key(candidate) == normalized_key:
            return float(value)
    return None


def normalize_price_key(value: str) -> str:
    return value.lower().replace(" ", "_").replace("-", "_")


def first_present(*values: Any) -> Any:
    for value in values:
        if value is not None and value != "":
            return value
    return None


def parse_time(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as config_file:
        return yaml.safe_load(config_file)


def relative(path: Path) -> str:
    return str(path.relative_to(ROOT_DIR) if path.is_relative_to(ROOT_DIR) else path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", default="nl-sql-peft-lab-ivan-0429")
    parser.add_argument("--region", default="europe-west4")
    parser.add_argument("--pricing", type=Path, default=DEFAULT_PRICING)
    parser.add_argument("--result", type=Path, action="append", default=DEFAULT_RESULTS)
    parser.add_argument("--discover-vertex-jobs", action="store_true")
    parser.add_argument("--job-limit", type=int, default=100)
    parser.add_argument("--job-filter", default='displayName:"wikisql"')
    parser.add_argument("--job-manifest", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    main()
