import os
import time
import ctypes
import platform
import subprocess
import threading
from typing import Dict, Any, Optional


class ResourceQuota:
    """Defines strict memory, compute, and hardware quotas for agent execution."""

    def __init__(
        self,
        mode: str = "auto",
        max_ram_gb: Optional[float] = None,
        max_vram_gb: Optional[float] = None,
        cpu_cores: Optional[int] = None,
        ttl_seconds: int = 120
    ):
        self.mode = mode.lower()  # "auto" or "manual"
        self.max_ram_gb = max_ram_gb
        self.max_vram_gb = max_vram_gb
        self.cpu_cores = cpu_cores
        self.ttl_seconds = ttl_seconds

    def to_dict(self) -> Dict[str, Any]:
        return {
            "mode": self.mode,
            "max_ram_gb": self.max_ram_gb,
            "max_vram_gb": self.max_vram_gb,
            "cpu_cores": self.cpu_cores,
            "ttl_seconds": self.ttl_seconds
        }


class HardwareProfiler:
    """Standard-library host hardware profiler for System RAM, VRAM, and CPU cores."""

    @staticmethod
    def get_system_ram_gb(show_warning:bool = False) -> float:
        """Retrieves total system RAM in Gigabytes cross-platform."""
        try:
            if platform.system() == "Windows":
                class MEMORYSTATUSEX(ctypes.Structure):
                    _fields_ = [
                        ("dwLength", ctypes.c_ulong),
                        ("dwMemoryLoad", ctypes.c_ulong),
                        ("ullTotalPhys", ctypes.c_ulonglong),
                        ("ullAvailPhys", ctypes.c_ulonglong),
                        ("ullTotalPageFile", ctypes.c_ulonglong),
                        ("ullAvailPageFile", ctypes.c_ulonglong),
                        ("ullTotalVirtual", ctypes.c_ulonglong),
                        ("ullAvailVirtual", ctypes.c_ulonglong),
                        ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
                    ]

                stat = MEMORYSTATUSEX()
                stat.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
                ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat))
                return round(stat.ullTotalPhys / (1024**3), 2)
            
            elif platform.system() == "Linux":
                with open("/proc/meminfo", "r") as f:
                    for line in f:
                        if "MemTotal" in line:
                            mem_kb = int(line.split()[1])
                            return round(mem_kb / (1024**2), 2)
            
            elif platform.system() == "Darwin":
                out = subprocess.check_output(["sysctl", "-n", "hw.memsize"]).strip()
                return round(int(out) / (1024**3), 2)
        except Exception:
            pass
        if show_warning:
            print("Hardware Profiler Warning: Unable to determine system RAM. Defaulting to 8GB.")
        return 8.0  # Safe fallback estimate

    @staticmethod
    def get_vram_gb(show_warning:bool = False) -> float:
        """Queries GPU VRAM capacity in Gigabytes using standard system utilities."""
        try:
            # NVIDIA CUDA check
            res = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=memory.total", "--format=csv,nounits,noheader"],
                stderr=subprocess.DEVNULL
            ).decode("utf-8").strip()
            if res:
                vram_mb = float(res.split("\n")[0])
                return round(vram_mb / 1024, 2)
        except Exception:
            pass

        try:
            # macOS Apple Silicon Unified Memory fallback
            if platform.system() == "Darwin":
                return HardwareProfiler.get_system_ram_gb() * 0.75  # ~75% usable for Metal
        except Exception:
            pass
        
        if show_warning:
            print("Hardware Profiler Warning: Unable to determine GPU VRAM. Defaulting to 0GB.")
        return 0.0  # Default to CPU execution if no dedicated VRAM detected

    @classmethod
    def inspect(cls) -> Dict[str, Any]:
        return {
            "os": platform.system(),
            "cpu_cores": os.cpu_count() or 1,
            "system_ram_gb": cls.get_system_ram_gb(),
            "gpu_vram_gb": cls.get_vram_gb()
        }


class MoELayerOffloader:
    """Calculates MoE layer distribution across GPU VRAM and System RAM (DDR5)."""

    @staticmethod
    def calculate_offload(
        total_layers: int,
        num_experts: int,
        model_size_gb: float,
        quota: ResourceQuota
    ) -> Dict[str, Any]:
        specs = HardwareProfiler.inspect()
        avail_vram = quota.max_vram_gb if (quota.mode == "manual" and quota.max_vram_gb) else specs["gpu_vram_gb"]
        avail_ram = quota.max_ram_gb if (quota.mode == "manual" and quota.max_ram_gb) else specs["system_ram_gb"]

        # Reserve safety buffer for KV-cache context memory
        vram_budget = max(0.0, avail_vram - 1.5)

        if avail_vram <= 0.0 or model_size_gb <= 0.0:
            return {
                "gpu_layers": 0,
                "offload_experts_to_ram": True,
                "vram_allocated_gb": 0.0,
                "ram_allocated_gb": min(model_size_gb, avail_ram)
            }

        vram_ratio = min(1.0, vram_budget / model_size_gb)
        gpu_layers = int(total_layers * vram_ratio)
        offload_experts = gpu_layers < total_layers

        return {
            "gpu_layers": gpu_layers,
            "cpu_layers": total_layers - gpu_layers,
            "offload_experts_to_ram": offload_experts,
            "vram_allocated_gb": round(vram_ratio * model_size_gb, 2),
            "ram_allocated_gb": round(max(0.0, model_size_gb - (vram_ratio * model_size_gb)), 2)
        }


class LocalModelSingleton:
    """Thread-safe active model lock ensuring only one local model binary is loaded at a time."""

    _instance_lock = threading.Lock()
    _active_model_name: Optional[str] = None
    _active_process: Optional[Any] = None
    _last_active_time: float = 0.0
    _timer_thread: Optional[threading.Thread] = None

    @classmethod
    def acquire_model(cls, model_name: str, launch_func, quota: ResourceQuota):
        with cls._instance_lock:
            now = time.time()
            if cls._active_model_name == model_name and cls._active_process is not None:
                cls._last_active_time = now
                return cls._active_process

            # Unload any currently active local model before loading the new binary
            cls._unload_active_unlocked()

            print(f"[Hardware Engine] Loading local model binary: '{model_name}'...")
            cls._active_process = launch_func()
            cls._active_model_name = model_name
            cls._last_active_time = now

            # Start TTL background monitor thread
            if quota.ttl_seconds > 0:
                cls._start_ttl_monitor(quota.ttl_seconds)

            return cls._active_process

    @classmethod
    def _unload_active_unlocked(cls):
        if cls._active_model_name:
            print(f"[Hardware Engine] Unloading inactive model: '{cls._active_model_name}' to free VRAM/RAM.")
            if cls._active_process and hasattr(cls._active_process, "kill"):
                try:
                    cls._active_process.kill()
                except Exception:
                    pass
            cls._active_process = None
            cls._active_model_name = None

    @classmethod
    def unload(cls):
        with cls._instance_lock:
            cls._unload_active_unlocked()

    @classmethod
    def _start_ttl_monitor(cls, ttl_seconds: int):
        def monitor():
            while True:
                time.sleep(5)
                with cls._instance_lock:
                    if cls._active_model_name is None:
                        break
                    if time.time() - cls._last_active_time >= ttl_seconds:
                        print(f"[Hardware Engine] TTL Timeout ({ttl_seconds}s) reached. Releasing VRAM/RAM.")
                        cls._unload_active_unlocked()
                        break

        cls._timer_thread = threading.Thread(target=monitor, daemon=True)
        cls._timer_thread.start()