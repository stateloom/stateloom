"""Hardware detection and model recommendations for local inference."""

from __future__ import annotations

import logging
import os
import platform
import shutil
import subprocess
from dataclasses import dataclass
from typing import Any

from stateloom.local.models import MODEL_CATALOG

logger = logging.getLogger("stateloom.local.hardware")

_HEADROOM_GB = 2.0


@dataclass
class HardwareInfo:
    ram_gb: float = 0.0
    disk_free_gb: float = 0.0
    gpu_name: str = ""
    gpu_vram_gb: float = 0.0
    os_name: str = ""
    arch: str = ""
    cpu_count: int = 0


def detect_hardware() -> HardwareInfo:
    """Detect available hardware for local model inference."""
    info = HardwareInfo(
        os_name=platform.system(),
        arch=platform.machine(),
        cpu_count=os.cpu_count() or 1,
    )

    # Disk free space
    try:
        usage = shutil.disk_usage("/")
        info.disk_free_gb = usage.free / (1024**3)
    except Exception:
        pass

    # RAM detection
    info.ram_gb = _detect_ram()

    # GPU detection
    gpu_name, gpu_vram = _detect_gpu(info)
    info.gpu_name = gpu_name
    info.gpu_vram_gb = gpu_vram

    return info


def _detect_ram() -> float:
    """Detect total system RAM in GB."""
    # Try psutil first (most accurate)
    try:
        import psutil  # type: ignore[import-untyped]

        return float(psutil.virtual_memory().total / (1024**3))
    except ImportError:
        pass

    # macOS / Linux fallback via os.sysconf
    try:
        pages = os.sysconf("SC_PHYS_PAGES")
        page_size = os.sysconf("SC_PAGE_SIZE")
        if pages > 0 and page_size > 0:
            return (pages * page_size) / (1024**3)
    except (ValueError, OSError, AttributeError):
        pass

    return 0.0


def _detect_gpu(info: HardwareInfo) -> tuple[str, float]:
    """Detect GPU name and VRAM. Returns (name, vram_gb)."""
    system = info.os_name

    # Apple Silicon — unified memory, estimate 75% available for GPU
    if system == "Darwin" and info.arch == "arm64":
        available = info.ram_gb * 0.75
        return ("Apple Silicon (unified)", available)

    # NVIDIA — try nvidia-smi
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0 and result.stdout.strip():
            line = result.stdout.strip().split("\n")[0]
            parts = line.split(", ")
            name = parts[0].strip()
            vram_mb = float(parts[1].strip()) if len(parts) > 1 else 0.0
            return (name, vram_mb / 1024)
    except (FileNotFoundError, subprocess.TimeoutExpired, Exception):
        pass

    return ("", 0.0)


def recommend_models(hardware: HardwareInfo | None = None) -> list[dict[str, Any]]:
    """Recommend models based on available hardware.

    Returns models sorted by tier, filtered to fit in available memory
    with headroom.
    """
    if hardware is None:
        hardware = detect_hardware()

    # Use GPU VRAM if available, otherwise RAM
    available = hardware.gpu_vram_gb if hardware.gpu_vram_gb > 0 else hardware.ram_gb
    max_model_size = available - _HEADROOM_GB

    if max_model_size <= 0:
        return []

    recommended = [
        {**model, "fits_in_memory": True}
        for model in MODEL_CATALOG
        if model["size_gb"] <= max_model_size
    ]

    # Sort by tier order, then size
    tier_order = {"ultra-light": 0, "light": 1, "medium": 2, "heavy": 3}
    recommended.sort(key=lambda m: (tier_order.get(m["tier"], 99), m["size_gb"]))

    return recommended
