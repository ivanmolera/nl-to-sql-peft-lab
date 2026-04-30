"""Runtime metadata for reproducible benchmark reports."""

from __future__ import annotations

import os
import platform
import sys
from datetime import datetime, timezone
from typing import Any


def collect_runtime_info() -> dict[str, Any]:
    info = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "runtime": os.environ.get("APP_RUNTIME", "unknown"),
        "container_image": os.environ.get("APP_DOCKER_IMAGE", "unknown"),
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "cpu_count": os.cpu_count(),
        "memory_total_gb": total_memory_gb(),
        "training_service": os.environ.get("TRAINING_SERVICE"),
        "region": os.environ.get("TRAINING_REGION") or os.environ.get("CLOUD_ML_REGION"),
        "machine_type": os.environ.get("MACHINE_TYPE"),
        "accelerator_type": os.environ.get("ACCELERATOR_TYPE"),
        "accelerator_count": env_int("ACCELERATOR_COUNT"),
        "cloud_run": {
            "service": os.environ.get("K_SERVICE"),
            "revision": os.environ.get("K_REVISION"),
            "configuration": os.environ.get("K_CONFIGURATION"),
        },
    }
    info.update(torch_info())
    return info


def torch_info() -> dict[str, Any]:
    try:
        import torch
    except ImportError:
        return {
            "torch_version": None,
            "cuda_available": False,
            "cuda_version": None,
            "device": "unavailable",
        }

    cuda_available = torch.cuda.is_available()
    device = torch.cuda.get_device_name(0) if cuda_available else "cpu"
    return {
        "torch_version": torch.__version__,
        "cuda_available": cuda_available,
        "cuda_version": torch.version.cuda,
        "device": device,
        "cuda_device_count": torch.cuda.device_count() if cuda_available else 0,
    }


def total_memory_gb() -> float | None:
    cgroup_limit = cgroup_memory_limit_bytes()
    if cgroup_limit is not None:
        return round(cgroup_limit / (1024**3), 2)
    try:
        pages = os.sysconf("SC_PHYS_PAGES")
        page_size = os.sysconf("SC_PAGE_SIZE")
    except (AttributeError, OSError, ValueError):
        return None
    return round((pages * page_size) / (1024**3), 2)


def env_int(name: str) -> int | None:
    raw_value = os.environ.get(name)
    if raw_value is None:
        return None
    try:
        return int(raw_value)
    except ValueError:
        return None


def cgroup_memory_limit_bytes() -> int | None:
    candidates = [
        "/sys/fs/cgroup/memory.max",
        "/sys/fs/cgroup/memory/memory.limit_in_bytes",
    ]
    for path in candidates:
        try:
            raw_value = open(path, encoding="utf-8").read().strip()
        except OSError:
            continue
        if raw_value == "max":
            continue
        try:
            value = int(raw_value)
        except ValueError:
            continue
        if 0 < value < 1 << 60:
            return value
    return None
