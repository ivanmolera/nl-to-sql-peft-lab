"""Shared helpers for PEFT training runs."""

from __future__ import annotations

import os
import resource
import subprocess
import threading
import time
from typing import Any

import torch
from transformers import EarlyStoppingCallback, TrainerCallback


class ResourceMonitor:
    """Collect lightweight process and GPU metrics during a training run."""

    def __init__(self, interval_seconds: float = 5.0) -> None:
        self.interval_seconds = interval_seconds
        self.started_at: float | None = None
        self.process_started_at: float | None = None
        self.finished_at: float | None = None
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self.gpu_samples: list[dict[str, float]] = []

    def start(self) -> None:
        self.started_at = time.perf_counter()
        self.process_started_at = time.process_time()
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        self._thread = threading.Thread(target=self._sample_loop, daemon=True)
        self._thread.start()

    def stop(self) -> dict[str, Any]:
        self.finished_at = time.perf_counter()
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=self.interval_seconds + 1)

        wall_seconds = (
            self.finished_at - self.started_at
            if self.started_at is not None
            else None
        )
        process_seconds = (
            time.process_time() - self.process_started_at
            if self.process_started_at is not None
            else None
        )
        cpu_count = os.cpu_count() or 1
        cpu_percent = (
            (process_seconds / wall_seconds / cpu_count) * 100
            if wall_seconds and process_seconds is not None
            else None
        )

        payload: dict[str, Any] = {
            "training_wall_time_seconds": wall_seconds,
            "training_wall_time_minutes": wall_seconds / 60 if wall_seconds else None,
            "process_cpu_time_seconds": process_seconds,
            "cpu_count": cpu_count,
            "cpu_utilization_estimated_percent": cpu_percent,
            "process_max_rss_mb": max_rss_mb(),
            "gpu_samples": len(self.gpu_samples),
        }
        payload.update(torch_cuda_metrics())
        payload.update(aggregate_gpu_samples(self.gpu_samples))
        return payload

    def _sample_loop(self) -> None:
        while not self._stop_event.is_set():
            sample = query_gpu_sample()
            if sample:
                self.gpu_samples.append(sample)
            self._stop_event.wait(self.interval_seconds)


def build_early_stopping_callbacks(config: dict[str, Any]) -> list[TrainerCallback]:
    early_stopping = config.get("training", {}).get("early_stopping", {})
    if not early_stopping.get("enabled", False):
        return []
    return [
        EarlyStoppingCallback(
            early_stopping_patience=early_stopping.get("patience", 3),
            early_stopping_threshold=early_stopping.get("threshold", 0.0),
        )
    ]


def best_model_training_args(training: dict[str, Any]) -> dict[str, Any]:
    keys = [
        "load_best_model_at_end",
        "metric_for_best_model",
        "greater_is_better",
    ]
    return {
        key: training[key]
        for key in keys
        if key in training
    }


def add_best_model_metadata(metrics: dict[str, Any], trainer) -> dict[str, Any]:
    payload = dict(metrics)
    payload["best_model_checkpoint"] = trainer.state.best_model_checkpoint
    payload["best_metric"] = trainer.state.best_metric
    payload["global_step"] = trainer.state.global_step
    return payload


def add_training_run_metadata(
    metrics: dict[str, Any],
    train_metrics: dict[str, Any],
    resource_metrics: dict[str, Any],
) -> dict[str, Any]:
    payload = dict(metrics)
    payload["train_metrics"] = train_metrics
    payload["resource_metrics"] = resource_metrics
    return payload


def max_rss_mb() -> float:
    max_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if os.uname().sysname == "Darwin":
        return max_rss / (1024 * 1024)
    return max_rss / 1024


def torch_cuda_metrics() -> dict[str, Any]:
    if not torch.cuda.is_available():
        return {
            "cuda_available": False,
            "cuda_device_count": 0,
            "cuda_max_memory_allocated_mb": None,
            "cuda_max_memory_reserved_mb": None,
        }
    return {
        "cuda_available": True,
        "cuda_device_count": torch.cuda.device_count(),
        "cuda_device_name": torch.cuda.get_device_name(0),
        "cuda_max_memory_allocated_mb": torch.cuda.max_memory_allocated() / (1024 * 1024),
        "cuda_max_memory_reserved_mb": torch.cuda.max_memory_reserved() / (1024 * 1024),
    }


def query_gpu_sample() -> dict[str, float] | None:
    command = [
        "nvidia-smi",
        "--query-gpu=utilization.gpu,memory.used,memory.total",
        "--format=csv,noheader,nounits",
    ]
    try:
        completed = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
            timeout=3,
        )
    except (FileNotFoundError, subprocess.SubprocessError):
        return None

    first_line = completed.stdout.strip().splitlines()[0] if completed.stdout.strip() else ""
    if not first_line:
        return None
    values = [value.strip() for value in first_line.split(",")]
    if len(values) < 3:
        return None
    return {
        "gpu_utilization_percent": float(values[0]),
        "gpu_memory_used_mb": float(values[1]),
        "gpu_memory_total_mb": float(values[2]),
    }


def aggregate_gpu_samples(samples: list[dict[str, float]]) -> dict[str, Any]:
    if not samples:
        return {
            "gpu_utilization_mean_percent": None,
            "gpu_utilization_peak_percent": None,
            "gpu_memory_used_mean_mb": None,
            "gpu_memory_used_peak_mb": None,
            "gpu_memory_total_mb": None,
        }

    utilization = [sample["gpu_utilization_percent"] for sample in samples]
    memory_used = [sample["gpu_memory_used_mb"] for sample in samples]
    return {
        "gpu_utilization_mean_percent": sum(utilization) / len(utilization),
        "gpu_utilization_peak_percent": max(utilization),
        "gpu_memory_used_mean_mb": sum(memory_used) / len(memory_used),
        "gpu_memory_used_peak_mb": max(memory_used),
        "gpu_memory_total_mb": samples[-1]["gpu_memory_total_mb"],
    }
