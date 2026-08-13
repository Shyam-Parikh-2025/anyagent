from typing import Dict, Any, Optional, Tuple
from .hardware import HardwareProfiler, ResourceQuota
from .compressor import CompressionPolicy


class RoleRank:
    C_SUITE = "C_SUITE"
    MANAGER = "MANAGER"
    SENIOR = "SENIOR"
    JUNIOR = "JUNIOR"
    INTERN = "INTERN"


class ModelRouter:
    """Evaluates tasks, checks hardware constraints, and maps LLMs dynamically."""

    @classmethod
    def check_hardware_safety(
        cls, 
        requested_model: str, 
        estimated_ram_gb: float, 
        quota: ResourceQuota
    ) -> Tuple[bool, str]:
        """Validates if a local model binary exceeds host system memory thresholds."""
        specs = HardwareProfiler.inspect()
        max_allowed_ram = (
            quota.max_ram_gb 
            if (quota.mode == "manual" and quota.max_ram_gb is not None) 
            else (specs["system_ram_gb"] * 0.50)
        )

        if estimated_ram_gb > max_allowed_ram:
            msg = (
                f"Model '{requested_model}' (~{estimated_ram_gb}GB RAM) exceeds "
                f"usable threshold of {max_allowed_ram:.2f}GB RAM."
            )
            return False, msg
        return True, ""

    @classmethod
    def allocate_model(
        cls,
        rank: str,
        policy: Dict[str, Any],
        model_map: Dict[str, str],
        task_prompt: Optional[str] = None,
        quota: Optional[ResourceQuota] = None
    ) -> str:
        """Determines the target LLM based on corporate rank, cost/speed policy, and custom mapping."""
        # 1. Check explicit custom rank mappings passed by user
        if rank in model_map:
            selected_model = model_map[rank]
            if quota and "ollama" in selected_model.lower():
                # Estimate 5GB for standard 8B local models as a reference check
                is_safe, warning = cls.check_hardware_safety(selected_model, 5.0, quota)
                if not is_safe and policy.get("auto_fallback", True):
                    fallback = model_map.get("fallback", "gemini-2.5-flash")
                    print(f"[Router Warning]: {warning} Falling back to '{fallback}'.")
                    return fallback
            return selected_model

        # 2. Policy-driven routing heuristics
        cost_priority = policy.get("cost_priority", "medium")

        if cost_priority == "high" and rank in [RoleRank.JUNIOR, RoleRank.INTERN]:
            return model_map.get("local_default", "ollama/llama3.1:8b")
        elif rank in [RoleRank.C_SUITE, RoleRank.MANAGER]:
            return model_map.get("frontier_default", "claude-3-5-sonnet-20241022")

        return model_map.get("fast_default", "gemini-2.5-flash")

    @classmethod
    def allocate_compression_policy(cls, rank: str) -> CompressionPolicy:
        """Rank -> CompressionPolicy, same shape as allocate_model but for tool-output
        compression instead of model choice. Whoever builds an Agent for a given rank calls
        this alongside allocate_model to get both.

        Intern/Junior are the local-tier "worker" ranks - tightest context windows, and the
        ones directly executing raw tool calls, so compression is on with a tight budget and
        (deliberately) no summarizer, per the model-thrash reasoning in CompressionPolicy's
        own docstring - a local worker agent shouldn't trigger a second local model load just
        to shrink its own tool output. Senior gets compression on too but with more headroom.
        Manager/C-Suite run on frontier APIs with much bigger context windows and are more
        likely consuming already-delegated/summarized results than raw tool spew, so they're
        left disabled by default - construct a CompressionPolicy by hand and pass it to
        set_compression_policy() if a specific agent needs different behavior than its rank's
        default.
        """
        if rank in (RoleRank.INTERN, RoleRank.JUNIOR):
            return CompressionPolicy(enabled=True, max_chars=1500, min_chars_to_bother=300)
        if rank == RoleRank.SENIOR:
            return CompressionPolicy(enabled=True, max_chars=4000, min_chars_to_bother=1000)
        return CompressionPolicy(enabled=False)  # MANAGER, C_SUITE, and any unrecognized rank