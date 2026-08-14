"""selector.py - decides which locally-installed model to use when the caller
asks for "auto" local model selection, using real hardware numbers from
benchmark.py instead of a guess.

Two responsibilities kept deliberately separate:

1. ModelCatalog - finds what's actually installed (Ollama / LM Studio / local
   Hugging Face snapshots), and lets the caller register additional
   candidates by hand - including ones that aren't installed yet, so auto
   mode can tell the user what to install instead of silently ignoring the
   name or picking something else behind their back.
2. LocalModelSelector - ranks the catalog's candidates against a
   BenchmarkResult and picks the best fit.

The ranking here is a fit-tier heuristic (does it fit in VRAM outright, does
it fit once some of it spills to RAM, does it not fit at all) nudged by real
CPU/PCIe throughput - it is NOT a predicted tokens/sec number. A real tokens/sec
estimate needs each model's active-parameter count (crucial for MoE models,
where only a handful of experts fire per token), which needs the GGUF metadata
parser this module doesn't have yet. Once that lands, swap _score_candidate's
internals for a real cost-model call into the offload solver - the public
surface here (select_best / SelectionResult) is built so callers don't need to
change when that happens.
"""

import glob
import json
import os
import platform
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from .benchmark import BenchmarkResult
from .hardware import ResourceQuota

VRAM_SAFETY_BUFFER_GB = 1.5  # same convention as MoELayerOffloader's KV-cache reserve


@dataclass
class LocalModelCandidate:
    """One local model llmadapt knows about, installed or not."""

    name: str
    provider: str  # "ollama" | "lm-studio" | "hf"
    size_gb: Optional[float] = None
    path: Optional[str] = None
    installed: bool = True
    install_hint: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "provider": self.provider,
            "size_gb": self.size_gb,
            "path": self.path,
            "installed": self.installed,
            "install_hint": self.install_hint,
        }


@dataclass
class ScoredCandidate:
    candidate: LocalModelCandidate
    tier: str  # "gpu_resident" | "offloaded" | "infeasible"
    gpu_resident_fraction: float
    score: float
    note: str


@dataclass
class SelectionResult:
    chosen: Optional[LocalModelCandidate]
    reason: str
    needs_install: Optional[LocalModelCandidate] = None
    install_hint: Optional[str] = None
    ranked: List[ScoredCandidate] = field(default_factory=list)


def _install_hint(provider: str, name: str) -> str:
    if provider == "ollama":
        return f"ollama pull {name}"
    if provider == "lm-studio":
        return f"Search for '{name}' in LM Studio's model browser and download it, then re-run discovery."
    if provider in ("hf", "vllm"):
        return f"huggingface-cli download {name}"
    return f"Install '{name}' for provider '{provider}' before it can be selected."


# ---- discovery ---------------------------------------------------------------


def _ollama_base_path() -> str:
    home = os.path.expanduser("~")
    if platform.system() == "Windows":
        return os.path.join(home, ".ollama", "models")
    if platform.system() == "Linux":
        system_path = "/usr/share/ollama/.ollama/models"
        if os.path.exists(system_path):
            return system_path
        return os.path.join(home, ".ollama", "models")
    return os.path.join(home, ".ollama", "models")  # macOS


def _scan_ollama(base_path: Optional[str] = None) -> List[LocalModelCandidate]:
    base_path = base_path or _ollama_base_path()
    found: List[LocalModelCandidate] = []
    manifest_root = os.path.join(base_path, "manifests", "registry.ollama.ai", "library")
    if not os.path.isdir(manifest_root):
        return found
    try:
        for model_name in os.listdir(manifest_root):
            model_dir = os.path.join(manifest_root, model_name)
            if not os.path.isdir(model_dir):
                continue
            for tag in os.listdir(model_dir):
                manifest_file = os.path.join(model_dir, tag)
                if not os.path.isfile(manifest_file):
                    continue
                try:
                    with open(manifest_file, "r", encoding="utf-8") as f:
                        manifest_data = json.load(f)
                    digest = None
                    for layer in manifest_data.get("layers", []):
                        if layer.get("mediaType") == "application/vnd.ollama.image.model":
                            digest = layer.get("digest")
                            break
                    if not digest:
                        continue
                    hash_prefix, hash_body = digest.split(":", 1)
                    blob_path = os.path.join(base_path, "blobs", f"{hash_prefix}-{hash_body}")
                    if not os.path.exists(blob_path):
                        continue
                    size_gb = round(os.path.getsize(blob_path) / (1024 ** 3), 4)
                    found.append(LocalModelCandidate(
                        name=f"{model_name}:{tag}", provider="ollama",
                        size_gb=size_gb, path=blob_path, installed=True,
                    ))
                except Exception:
                    continue  # one bad manifest shouldn't stop the whole scan
    except Exception:
        pass
    return found


def _scan_lm_studio(base_path: Optional[str] = None) -> List[LocalModelCandidate]:
    base_path = base_path or os.path.join(os.path.expanduser("~"), ".lmstudio", "models")
    found: List[LocalModelCandidate] = []
    if not os.path.isdir(base_path):
        return found
    try:
        for gguf_path in glob.glob(os.path.join(base_path, "**", "*.gguf"), recursive=True):
            rel = os.path.relpath(gguf_path, base_path)
            name = rel.replace(os.sep, "/")
            size_gb = round(os.path.getsize(gguf_path) / (1024 ** 3), 4)
            found.append(LocalModelCandidate(
                name=name, provider="lm-studio", size_gb=size_gb, path=gguf_path, installed=True,
            ))
    except Exception:
        pass
    return found


def _scan_hf(base_path: Optional[str] = None) -> List[LocalModelCandidate]:
    base_path = base_path or os.path.join(os.path.expanduser("~"), ".cache", "huggingface", "hub")
    found: List[LocalModelCandidate] = []
    if not os.path.isdir(base_path):
        return found
    try:
        for entry in os.listdir(base_path):
            if not entry.startswith("models--"):
                continue
            repo_name = entry[len("models--"):].replace("--", "/")
            snapshots_dir = os.path.join(base_path, entry, "snapshots")
            if not os.path.isdir(snapshots_dir):
                continue
            snapshots = sorted(os.listdir(snapshots_dir))
            if not snapshots:
                continue
            latest = os.path.join(snapshots_dir, snapshots[-1])
            total_bytes = 0
            for root, _dirs, files in os.walk(latest):
                for fname in files:
                    try:
                        total_bytes += os.path.getsize(os.path.join(root, fname))
                    except OSError:
                        continue
            if total_bytes == 0:
                continue
            found.append(LocalModelCandidate(
                name=repo_name, provider="hf",
                size_gb=round(total_bytes / (1024 ** 3), 4),
                path=latest, installed=True,
            ))
    except Exception:
        pass
    return found


_SCANNERS = {"ollama": _scan_ollama, "lm-studio": _scan_lm_studio, "hf": _scan_hf}


def discover_installed(
    providers: Tuple[str, ...] = ("ollama", "lm-studio", "hf"),
    base_paths: Optional[Dict[str, str]] = None,
) -> List[LocalModelCandidate]:
    base_paths = base_paths or {}
    found: List[LocalModelCandidate] = []
    for p in providers:
        scanner = _SCANNERS.get(p)
        if scanner:
            found.extend(scanner(base_paths.get(p)))
    return found


class ModelCatalog:
    """What's installed (via discovery) plus whatever the caller registers by
    hand. Discovery re-scans on each call rather than caching - these are
    cheap directory listings, not file reads, and staying live means a model
    downloaded mid-session shows up without restarting anything."""

    def __init__(
        self,
        providers: Tuple[str, ...] = ("ollama", "lm-studio", "hf"),
        base_paths: Optional[Dict[str, str]] = None,
    ):
        self.providers = tuple(providers)
        self.base_paths = base_paths or {}
        self._registered: Dict[str, LocalModelCandidate] = {}

    def _discover(self) -> List[LocalModelCandidate]:
        return discover_installed(self.providers, self.base_paths)

    def register(
        self,
        name: str,
        provider: str,
        size_gb: Optional[float] = None,
        path: Optional[str] = None,
        installed: Optional[bool] = None,
    ) -> LocalModelCandidate:
        """Add a candidate by hand - to widen what auto mode considers beyond
        what discovery found, or to declare a model the user wants that isn't
        installed yet, so selection can guide them to install it instead of
        silently ignoring the name.

        If installed isn't given explicitly, it's inferred: if a matching
        model turns up in discovery, the registration adopts that real entry
        (so a stale hand-typed size_gb doesn't shadow the true one);
        otherwise it's registered as not-installed with an install hint.
        """
        if installed is None:
            discovered_match = next(
                (c for c in self._discover() if c.name == name and c.provider == provider), None
            )
            candidate = discovered_match or LocalModelCandidate(
                name=name, provider=provider, size_gb=size_gb, path=path,
                installed=False, install_hint=_install_hint(provider, name),
            )
        else:
            candidate = LocalModelCandidate(
                name=name, provider=provider, size_gb=size_gb, path=path,
                installed=installed,
                install_hint=None if installed else _install_hint(provider, name),
            )
        self._registered[f"{provider}:{name}"] = candidate
        return candidate

    def unregister(self, name: str, provider: str) -> None:
        self._registered.pop(f"{provider}:{name}", None)

    def list_candidates(self, include_uninstalled: bool = True) -> List[LocalModelCandidate]:
        merged: Dict[str, LocalModelCandidate] = {}
        for c in self._discover():
            merged[f"{c.provider}:{c.name}"] = c
        for key, c in self._registered.items():
            merged[key] = c  # explicit registration wins - lets a caller override size/path
        results = list(merged.values())
        if not include_uninstalled:
            results = [c for c in results if c.installed]
        return results

    def find(self, name: str, provider: Optional[str] = None) -> Optional[LocalModelCandidate]:
        for c in self.list_candidates(include_uninstalled=True):
            if c.name == name and (provider is None or c.provider == provider):
                return c
        return None


# ---- selection ----------------------------------------------------------------


class LocalModelSelector:
    """Ranks a catalog's candidates against a BenchmarkResult and picks the
    best fit for 'auto' local mode."""

    @staticmethod
    def _budgets(benchmark: BenchmarkResult, quota: Optional[ResourceQuota]) -> Tuple[float, float]:
        avail_vram = (
            quota.max_vram_gb if (quota and quota.mode == "manual" and quota.max_vram_gb)
            else benchmark.gpu_vram_gb
        )
        avail_ram = (
            quota.max_ram_gb if (quota and quota.mode == "manual" and quota.max_ram_gb)
            else benchmark.system_ram_gb
        )
        vram_budget = max(0.0, avail_vram - VRAM_SAFETY_BUFFER_GB)
        return vram_budget, avail_ram

    @classmethod
    def _score_candidate(
        cls,
        candidate: LocalModelCandidate,
        benchmark: BenchmarkResult,
        quota: Optional[ResourceQuota],
        min_gpu_resident_fraction: float,
    ) -> ScoredCandidate:
        vram_budget, avail_ram = cls._budgets(benchmark, quota)
        size_gb = candidate.size_gb or 0.0

        if size_gb <= 0:
            return ScoredCandidate(candidate, "infeasible", 0.0, -1.0, "unknown size - can't evaluate fit")

        if size_gb <= vram_budget:
            # Prefer the biggest model that still fits fully resident - bigger
            # usually means a stronger model, and it costs nothing extra here.
            score = 1000.0 + size_gb
            return ScoredCandidate(candidate, "gpu_resident", 1.0, score, "fits entirely in VRAM")

        total_capacity = vram_budget + avail_ram
        if size_gb <= total_capacity:
            gpu_fraction = (vram_budget / size_gb) if size_gb > 0 else 0.0

            # Real CPU/PCIe throughput nudges how much offload this specific
            # machine can tolerate - a machine with strong CPU+PCIe numbers can
            # bear a lower gpu_fraction than a weak one. Bounded adjustment,
            # not a tokens/sec prediction (see module docstring).
            cpu_strength = min(1.0, benchmark.cpu_gflops / 200.0) if benchmark.cpu_gflops else 0.0
            pcie_strength = min(1.0, benchmark.pcie_bandwidth_gbps / 16.0) if benchmark.pcie_bandwidth_gbps else 0.0
            tolerance_bonus = 0.15 * ((cpu_strength + pcie_strength) / 2.0)
            required_fraction = max(0.0, min_gpu_resident_fraction - tolerance_bonus)

            if gpu_fraction >= required_fraction:
                score = 500.0 + (gpu_fraction * 100) + size_gb
                note = f"fits with ~{round((1 - gpu_fraction) * 100)}% offloaded to RAM"
                return ScoredCandidate(candidate, "offloaded", gpu_fraction, score, note)

            note = (
                f"fits but only {round(gpu_fraction * 100)}% GPU-resident - below this "
                f"machine's acceptable threshold ({round(required_fraction * 100)}%), likely too slow"
            )
            return ScoredCandidate(candidate, "infeasible", gpu_fraction, -1.0, note)

        return ScoredCandidate(
            candidate, "infeasible", 0.0, -1.0,
            f"needs {size_gb}GB, only {round(total_capacity, 1)}GB VRAM+RAM available",
        )

    @classmethod
    def select_best(
        cls,
        benchmark: BenchmarkResult,
        catalog: ModelCatalog,
        requested_name: Optional[str] = None,
        requested_provider: Optional[str] = None,
        quota: Optional[ResourceQuota] = None,
        min_gpu_resident_fraction: float = 0.5,
    ) -> SelectionResult:
        """If requested_name is given, tries to select exactly that model -
        and if it isn't installed, returns install guidance instead of
        silently picking something else. If requested_name is None (pure auto
        mode), ranks every installed candidate and picks the best fit."""
        if requested_name is not None:
            match = catalog.find(requested_name, requested_provider)
            if match is None:
                hint = _install_hint(requested_provider or "ollama", requested_name)
                placeholder = LocalModelCandidate(
                    name=requested_name, provider=requested_provider or "ollama",
                    installed=False, install_hint=hint,
                )
                return SelectionResult(
                    chosen=None,
                    reason=f"'{requested_name}' isn't registered or installed anywhere llmadapt looked.",
                    needs_install=placeholder, install_hint=hint,
                )
            if not match.installed:
                return SelectionResult(
                    chosen=None, reason=f"'{requested_name}' is known but not installed.",
                    needs_install=match, install_hint=match.install_hint,
                )
            scored = cls._score_candidate(match, benchmark, quota, min_gpu_resident_fraction)
            if scored.tier == "infeasible":
                return SelectionResult(
                    chosen=None,
                    reason=f"'{requested_name}' is installed but doesn't fit this machine: {scored.note}",
                    ranked=[scored],
                )
            return SelectionResult(chosen=match, reason=scored.note, ranked=[scored])

        installed = catalog.list_candidates(include_uninstalled=False)
        scored = [cls._score_candidate(c, benchmark, quota, min_gpu_resident_fraction) for c in installed]
        scored.sort(key=lambda s: s.score, reverse=True)
        feasible = [s for s in scored if s.tier != "infeasible"]

        if not feasible:
            uninstalled = [c for c in catalog.list_candidates(include_uninstalled=True) if not c.installed]
            if not installed and uninstalled:
                # Nothing installed at all, but the caller registered names
                # they want - point at the first one instead of just failing.
                pick = uninstalled[0]
                return SelectionResult(
                    chosen=None, reason="No local models are installed yet.",
                    needs_install=pick, install_hint=pick.install_hint, ranked=scored,
                )
            return SelectionResult(
                chosen=None, reason="No installed local model fits this machine acceptably.", ranked=scored,
            )

        best = feasible[0]
        return SelectionResult(chosen=best.candidate, reason=best.note, ranked=scored)
