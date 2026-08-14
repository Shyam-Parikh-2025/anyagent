"""benchmark.py - measures this machine's real CPU/GPU compute throughput and
PCIe transfer bandwidth, on top of the capacity numbers HardwareProfiler already
reports (VRAM/RAM size, in hardware.py). Feeds llmadapt's offload solver with
numbers a static size ratio can't give it: how fast this machine can move data
and do math, not just how much of it it has.

Zero required third-party dependencies, matching the rest of llmadapt. Where a
real measurement benefits from numpy/torch, both are used opportunistically if
already installed on the host and the code falls back to a conservative
stdlib-only estimate otherwise - never a hard install requirement.
"""

import hashlib
import json
import os
import platform
import random
import subprocess
import time
from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional

from .hardware import HardwareProfiler

DEFAULT_CACHE_DIR = os.path.join(os.path.expanduser("~"), ".llmadapt")
CACHE_FILENAME = "benchmark_cache.json"
DEFAULT_CACHE_TTL_SECONDS = 7 * 24 * 3600  # 1 week - drivers/hardware rarely change more often


def _random_matrix(n: int):
    return [[random.random() for _ in range(n)] for _ in range(n)]


def _matmul_pure_python(a, b, n: int):
    # Deliberately naive triple loop - this only needs to be a consistent,
    # comparable-across-machines proxy when numpy isn't installed, not fast.
    result = [[0.0] * n for _ in range(n)]
    for i in range(n):
        row_a = a[i]
        row_r = result[i]
        for k in range(n):
            aik = row_a[k]
            row_b = b[k]
            for j in range(n):
                row_r[j] += aik * row_b[j]
    return result


@dataclass
class BenchmarkResult:
    """Everything the offload solver needs about this machine's real
    throughput, not just its capacity."""

    os: str
    cpu_name: str
    cpu_cores: int
    cpu_gflops: float
    cpu_gflops_measured_with: str  # "numpy" or "pure_python"
    system_ram_gb: float
    gpu_name: str
    gpu_vram_gb: float
    gpu_pcie_gen: Optional[int]
    gpu_pcie_link_width: Optional[int]
    pcie_bandwidth_gbps: float
    pcie_bandwidth_measured: bool
    gpu_tflops: float
    gpu_tflops_measured: bool
    fingerprint: str
    timestamp: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class HardwareBenchmark:
    """Runs a short (roughly 1-3s) real benchmark of CPU compute, GPU compute
    (when measurable), and PCIe transfer bandwidth, and caches the result to
    disk keyed by a hardware fingerprint so repeat launches on the same
    machine skip the measurement pass.

    This is the missing piece behind MoELayerOffloader's offload math: today
    that solver only knows how big the model and the VRAM/RAM pools are. This
    class tells it how fast this specific machine can actually move data
    across PCIe and do CPU/GPU math, which is what the split point actually
    depends on.
    """

    _PCIE_GBPS_PER_LANE = {1: 0.25, 2: 0.5, 3: 0.985, 4: 1.969, 5: 3.938}
    _PCIE_EFFICIENCY = 0.75  # real sustained transfers run well under theoretical peak

    # Coarse, deliberately conservative FP16-TFLOPS-by-VRAM-tier fallback for
    # when torch isn't installed to measure the GPU directly. This is a rough
    # placeholder for the solver's ranking logic, not a precise spec lookup -
    # a real measurement (torch, if present) always overrides it.
    _GPU_TFLOPS_BY_VRAM_TIER = [
        (8.0, 20.0),
        (16.0, 40.0),
        (24.0, 80.0),
        (float("inf"), 100.0),
    ]

    # ---- name / link lookups -----------------------------------------------

    @staticmethod
    def _cpu_name() -> str:
        try:
            if platform.system() == "Windows":
                return platform.processor() or "unknown-cpu"
            elif platform.system() == "Linux":
                with open("/proc/cpuinfo") as f:
                    for line in f:
                        if line.lower().startswith("model name"):
                            return line.split(":", 1)[1].strip()
            elif platform.system() == "Darwin":
                out = subprocess.check_output(
                    ["sysctl", "-n", "machdep.cpu.brand_string"], stderr=subprocess.DEVNULL
                )
                return out.decode("utf-8").strip()
        except Exception:
            pass
        return platform.processor() or "unknown-cpu"

    @staticmethod
    def _gpu_name() -> str:
        try:
            res = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                stderr=subprocess.DEVNULL,
            ).decode("utf-8").strip()
            if res:
                return res.split("\n")[0].strip()
        except Exception:
            pass
        return "none"

    @staticmethod
    def _gpu_pcie_info():
        """Returns (gen, link_width) queried from nvidia-smi, or (None, None)
        if unavailable (no NVIDIA GPU, driver not installed, etc.)."""
        try:
            out = subprocess.check_output(
                [
                    "nvidia-smi",
                    "--query-gpu=pcie.link.gen.max,pcie.link.width.max",
                    "--format=csv,noheader,nounits",
                ],
                stderr=subprocess.DEVNULL,
            ).decode("utf-8").strip()
            if out:
                gen_str, width_str = out.split("\n")[0].split(",")
                return int(gen_str.strip()), int(width_str.strip())
        except Exception:
            pass
        return None, None

    # ---- PCIe bandwidth -----------------------------------------------------

    @classmethod
    def _estimate_pcie_bandwidth_gbps(cls, gen: Optional[int], width: Optional[int]) -> float:
        if not gen or not width:
            return 4.0  # conservative fallback when the link itself can't be detected
        per_lane = cls._PCIE_GBPS_PER_LANE.get(gen, cls._PCIE_GBPS_PER_LANE[3])
        return round(per_lane * width * cls._PCIE_EFFICIENCY, 2)

    @staticmethod
    def _measure_pcie_bandwidth_gbps_torch() -> Optional[float]:
        """Opportunistic real measurement if torch+CUDA happen to already be
        installed. Not a dependency llmadapt requires - only used when the
        host already has it (e.g. because the user also runs a PyTorch-based
        stack), otherwise the caller falls back to the link-speed estimate."""
        try:
            import torch  # local, opportunistic import - not a hard dependency

            if not torch.cuda.is_available():
                return None
            size_mb = 256
            n_floats = (size_mb * 1024 * 1024) // 4
            staging = torch.empty(n_floats, dtype=torch.float32, pin_memory=True)
            torch.cuda.synchronize()
            iters = 5
            start = time.perf_counter()
            for _ in range(iters):
                staging.to("cuda", non_blocking=True)
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - start
            if elapsed <= 0:
                return None
            total_gb = (size_mb * iters) / 1024
            return round(total_gb / elapsed, 2)
        except Exception:
            return None

    # ---- GPU compute ---------------------------------------------------------

    @classmethod
    def _estimate_gpu_tflops(cls, vram_gb: float) -> float:
        for tier_vram, tflops in cls._GPU_TFLOPS_BY_VRAM_TIER:
            if vram_gb <= tier_vram:
                return tflops
        return cls._GPU_TFLOPS_BY_VRAM_TIER[-1][1]

    @staticmethod
    def _measure_gpu_tflops_torch() -> Optional[float]:
        try:
            import torch

            if not torch.cuda.is_available():
                return None
            n = 4096
            a = torch.randn(n, n, dtype=torch.float16, device="cuda")
            b = torch.randn(n, n, dtype=torch.float16, device="cuda")
            torch.cuda.synchronize()
            iters = 10
            start = time.perf_counter()
            for _ in range(iters):
                a @ b
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - start
            if elapsed <= 0:
                return None
            flops = 2 * (n ** 3) * iters
            return round(flops / elapsed / 1e12, 2)
        except Exception:
            return None

    # ---- CPU compute ---------------------------------------------------------

    @staticmethod
    def _measure_cpu_gflops(time_budget_s: float = 1.0):
        """Returns (gflops, method). This is a relative proxy for comparing
        this machine's CPU-side offload speed against its own GPU numbers -
        NOT a predictor of llama.cpp's actual achieved throughput, which uses
        hand-tuned AVX/NEON kernels far faster than either path here."""
        try:
            import numpy as np

            n = 256
            a = np.random.rand(n, n).astype(np.float32)
            b = np.random.rand(n, n).astype(np.float32)
            flops_per_call = 2 * (n ** 3)
            start = time.perf_counter()
            count = 0
            while time.perf_counter() - start < time_budget_s:
                np.dot(a, b)
                count += 1
            elapsed = time.perf_counter() - start
            if elapsed <= 0 or count == 0:
                return 1.0, "numpy"
            return round((flops_per_call * count) / elapsed / 1e9, 2), "numpy"
        except ImportError:
            n = 48
            a = _random_matrix(n)
            b = _random_matrix(n)
            flops_per_call = 2 * (n ** 3)
            start = time.perf_counter()
            count = 0
            while time.perf_counter() - start < time_budget_s:
                _matmul_pure_python(a, b, n)
                count += 1
            elapsed = time.perf_counter() - start
            if elapsed <= 0 or count == 0:
                return 0.01, "pure_python"
            return round((flops_per_call * count) / elapsed / 1e9, 4), "pure_python"

    # ---- fingerprint + cache ---------------------------------------------------

    @staticmethod
    def _fingerprint(specs: Dict[str, Any], gpu_name: str) -> str:
        raw = (
            f"{specs['os']}|{specs['cpu_cores']}|{HardwareBenchmark._cpu_name()}|"
            f"{gpu_name}|{specs['system_ram_gb']}|{specs['gpu_vram_gb']}"
        )
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]

    @staticmethod
    def _cache_path(cache_dir: str) -> str:
        return os.path.join(cache_dir, CACHE_FILENAME)

    @classmethod
    def _load_cache(cls, fingerprint: str, ttl_s: int, cache_dir: str) -> Optional[BenchmarkResult]:
        path = cls._cache_path(cache_dir)
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            return None
        if data.get("fingerprint") != fingerprint:
            return None
        if time.time() - data.get("timestamp", 0) > ttl_s:
            return None
        try:
            return BenchmarkResult(**data)
        except TypeError:
            return None  # stale schema from an older llmadapt version - re-measure

    @classmethod
    def _save_cache(cls, result: BenchmarkResult, cache_dir: str) -> None:
        try:
            os.makedirs(cache_dir, exist_ok=True)
            with open(cls._cache_path(cache_dir), "w", encoding="utf-8") as f:
                json.dump(result.to_dict(), f, indent=2)
        except Exception:
            pass  # caching is a convenience, never fatal to the benchmark itself

    # ---- public API ---------------------------------------------------------

    @classmethod
    def run(
        cls,
        force: bool = False,
        cpu_time_budget_s: float = 1.0,
        cache_ttl_s: int = DEFAULT_CACHE_TTL_SECONDS,
        cache_dir: str = DEFAULT_CACHE_DIR,
    ) -> BenchmarkResult:
        """Runs (or loads a cached copy of) the hardware benchmark.

        force=True always re-measures, ignoring any cached result.
        cpu_time_budget_s controls how long the CPU matmul loop runs (an
        accuracy-vs-speed knob - the whole pass, GPU included, targets a few
        seconds total on a typical machine).
        cache_dir lets callers (and tests) point the cache somewhere other
        than the user's home directory.
        """
        specs = HardwareProfiler.inspect()
        gpu_name = cls._gpu_name()
        fingerprint = cls._fingerprint(specs, gpu_name)

        if not force:
            cached = cls._load_cache(fingerprint, cache_ttl_s, cache_dir)
            if cached is not None:
                return cached

        no_gpu = gpu_name == "none" or specs["gpu_vram_gb"] <= 0.0

        if no_gpu:
            # Honest zeros, not a guessed fallback number - there's nothing to
            # offload to and no link to measure.
            gen, width = None, None
            pcie_bandwidth, measured_pcie = 0.0, False
            gpu_tflops, measured_tflops = 0.0, False
        else:
            gen, width = cls._gpu_pcie_info()
            measured_pcie_val = cls._measure_pcie_bandwidth_gbps_torch()
            measured_pcie = measured_pcie_val is not None
            pcie_bandwidth = measured_pcie_val if measured_pcie else cls._estimate_pcie_bandwidth_gbps(gen, width)

            measured_tflops_val = cls._measure_gpu_tflops_torch()
            measured_tflops = measured_tflops_val is not None
            gpu_tflops = measured_tflops_val if measured_tflops else cls._estimate_gpu_tflops(specs["gpu_vram_gb"])

        cpu_gflops, cpu_method = cls._measure_cpu_gflops(cpu_time_budget_s)

        result = BenchmarkResult(
            os=specs["os"],
            cpu_name=cls._cpu_name(),
            cpu_cores=specs["cpu_cores"],
            cpu_gflops=cpu_gflops,
            cpu_gflops_measured_with=cpu_method,
            system_ram_gb=specs["system_ram_gb"],
            gpu_name=gpu_name,
            gpu_vram_gb=specs["gpu_vram_gb"],
            gpu_pcie_gen=gen,
            gpu_pcie_link_width=width,
            pcie_bandwidth_gbps=pcie_bandwidth,
            pcie_bandwidth_measured=measured_pcie,
            gpu_tflops=gpu_tflops,
            gpu_tflops_measured=measured_tflops,
            fingerprint=fingerprint,
            timestamp=time.time(),
        )
        cls._save_cache(result, cache_dir)
        return result


if __name__ == "__main__":
    print("Running llmadapt hardware benchmark (first run may take a few seconds)...\n")
    r = HardwareBenchmark.run()
    print(f"OS:              {r.os}")
    print(f"CPU:             {r.cpu_name} ({r.cpu_cores} cores)")
    print(f"CPU throughput:  {r.cpu_gflops} GFLOPS (measured with {r.cpu_gflops_measured_with})")
    print(f"System RAM:      {r.system_ram_gb} GB")
    print(f"GPU:             {r.gpu_name}")
    if r.gpu_name == "none":
        print("GPU VRAM:        n/a - no dedicated GPU detected, offload solver will target CPU-only")
    else:
        print(f"GPU VRAM:        {r.gpu_vram_gb} GB")
        print(f"PCIe link:       gen {r.gpu_pcie_gen or '?'} x{r.gpu_pcie_link_width or '?'}")
        print(f"PCIe bandwidth:  {r.pcie_bandwidth_gbps} GB/s ({'measured' if r.pcie_bandwidth_measured else 'estimated'})")
        print(f"GPU throughput:  {r.gpu_tflops} TFLOPS ({'measured' if r.gpu_tflops_measured else 'estimated'})")
    print(f"\nFingerprint: {r.fingerprint}  (cached at {DEFAULT_CACHE_DIR}/{CACHE_FILENAME})")
