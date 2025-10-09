"""
Utility helpers to keep notebook measurements consistent across the repo.

All helpers default to CPU-first assumptions and degrade gracefully when optional
dependencies such as PyTorch or psutil are unavailable.
"""
from __future__ import annotations

import csv
import datetime as _dt
import json
import os
import platform
import sys
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, Optional

try:
    import psutil  # type: ignore
except ImportError:  # pragma: no cover
    psutil = None

try:
    import torch  # type: ignore
except ImportError:  # pragma: no cover
    torch = None

try:
    import transformers  # type: ignore
except ImportError:  # pragma: no cover
    transformers = None

try:
    import accelerate  # type: ignore
except ImportError:  # pragma: no cover
    accelerate = None

try:
    import resource  # type: ignore
except ImportError:  # pragma: no cover
    resource = None


BENCHMARK_HEADER = (
    "hardware,os,framework,precision,task,model_id,dataset,"
    "sequence_or_image_res,batch,peak_ram_mb,peak_vram_mb,load_time_s,"
    "ttfb_s,tokens_per_s_or_images_per_s,runtime_versions,repo_commit,"
    "notebook_path,timestamp_utc"
)


def _ru_maxrss_mb() -> Optional[float]:
    if resource is None:
        return None
    usage = resource.getrusage(resource.RUSAGE_SELF)
    value = usage.ru_maxrss
    if value == 0:
        return None
    if sys.platform.startswith("darwin"):
        return value / (1024 * 1024)
    return value / 1024


def _psutil_peak_mb() -> Optional[float]:
    if psutil is None:
        return None
    process = psutil.Process(os.getpid())
    try:
        peak = getattr(process.memory_info(), "peak_wset", None)
        if peak:
            return peak / (1024 * 1024)
    except psutil.Error:  # pragma: no cover
        pass
    try:
        rss = process.memory_info().rss
        return rss / (1024 * 1024)
    except psutil.Error:  # pragma: no cover
        return None


def detect_env() -> Dict[str, str]:
    """Return a compact description of hardware, OS, and key library versions."""
    hardware = platform.machine()
    if torch and torch.cuda.is_available():
        device_name = torch.cuda.get_device_name(0)
        hardware = f"{hardware} | CUDA: {device_name}"
    elif torch and getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        hardware = f"{hardware} | Apple Silicon (MPS)"
    runtime_versions = {
        "python": platform.python_version(),
    }
    if torch:
        runtime_versions["torch"] = torch.__version__
        runtime_versions["cuda"] = torch.version.cuda or "cpu"
    if transformers:
        runtime_versions["transformers"] = transformers.__version__
    if accelerate:
        runtime_versions["accelerate"] = accelerate.__version__
    return {
        "hardware": hardware,
        "os": platform.platform(),
        "runtime_versions": json.dumps(runtime_versions, sort_keys=True),
    }


@dataclass
class Recorder:
    """Utility passed into run functions so they can mark timings and counts."""

    unit: str = "tokens"
    _start: float = field(default_factory=time.perf_counter)
    _ttfb: Optional[float] = None
    _items: int = 0

    def mark_first_token(self) -> None:
        if self._ttfb is None:
            self._ttfb = time.perf_counter() - self._start

    def add_items(self, count: int) -> None:
        self._items += int(count)

    @property
    def ttfb(self) -> Optional[float]:
        return self._ttfb

    @property
    def items_per_second(self) -> Optional[float]:
        elapsed = time.perf_counter() - self._start
        if elapsed <= 0 or self._items <= 0:
            return None
        return self._items / elapsed

    def reset_timer(self) -> None:
        self._start = time.perf_counter()
        self._ttfb = None
        self._items = 0


def measure_memory_speed(
    run_fn: Callable[[Recorder], None],
    setup_fn: Optional[Callable[[], None]] = None,
    warmup_runs: int = 2,
) -> Dict[str, Optional[float]]:
    """
    Execute `run_fn` while capturing peak memory and throughput.

    The provided `run_fn` receives a Recorder instance it can use to
    mark the first-token latency and emitted items. The helper handles
    warmup runs to stabilise kernels before measuring.
    """
    recorder = Recorder()
    if setup_fn:
        setup_fn()

    for _ in range(max(0, warmup_runs)):
        recorder.reset_timer()
        run_fn(recorder)

    recorder.reset_timer()

    if torch and torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

    peak_before = _ru_maxrss_mb() or _psutil_peak_mb() or 0.0
    start_time = time.perf_counter()
    run_fn(recorder)
    elapsed = time.perf_counter() - start_time

    peak_ram = max(_ru_maxrss_mb() or 0.0, _psutil_peak_mb() or 0.0, peak_before)

    peak_vram = None
    if torch and torch.cuda.is_available():
        torch.cuda.synchronize()
        peak_vram = torch.cuda.max_memory_allocated() / (1024 * 1024)
    elif torch and getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        peak_vram = None  # no direct API; record as None

    throughput = recorder.items_per_second()

    return {
        "wall_time_s": elapsed,
        "ttfb_s": recorder.ttfb,
        "throughput_per_s": throughput,
        "peak_ram_mb": peak_ram,
        "peak_vram_mb": peak_vram,
        "unit": recorder.unit,
    }


def append_benchmark_row(**fields: str) -> None:
    """
    Append a benchmark row to `benchmarks/matrix.csv`, creating the file with
    the canonical header if required.
    """
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    matrix_path = os.path.join(repo_root, "benchmarks", "matrix.csv")

    os.makedirs(os.path.dirname(matrix_path), exist_ok=True)

    file_exists = os.path.isfile(matrix_path)
    if not file_exists:
        with open(matrix_path, "w", encoding="utf-8", newline="") as f:
            f.write(f"{BENCHMARK_HEADER}\n")

    env = detect_env()
    timestamp = _dt.datetime.utcnow().isoformat(timespec="seconds")

    merged = {
        "hardware": env["hardware"],
        "os": env["os"],
        "framework": fields.get("framework", "transformers"),
        "precision": fields.get("precision", "fp32"),
        "task": fields.get("task", "unknown"),
        "model_id": fields.get("model_id", "unknown"),
        "dataset": fields.get("dataset", "unknown"),
        "sequence_or_image_res": fields.get("sequence_or_image_res", "n/a"),
        "batch": fields.get("batch", "1"),
        "peak_ram_mb": fields.get("peak_ram_mb", ""),
        "peak_vram_mb": fields.get("peak_vram_mb", ""),
        "load_time_s": fields.get("load_time_s", ""),
        "ttfb_s": fields.get("ttfb_s", ""),
        "tokens_per_s_or_images_per_s": fields.get("tokens_per_s_or_images_per_s", ""),
        "runtime_versions": env["runtime_versions"],
        "repo_commit": fields.get("repo_commit", ""),
        "notebook_path": fields.get("notebook_path", ""),
        "timestamp_utc": timestamp,
    }

    with open(matrix_path, "a", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(merged.values())

