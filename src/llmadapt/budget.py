"""budget.py - Phase 3: percentage-based token budget allocation across a
Company, with a recursive "ask my manager/siblings for spare capacity"
reallocation protocol, and a hard company-wide ceiling that always requires
human sign-off (via Company.on_escalation) to cross, regardless of internal
reallocation.

Real usage numbers now come from Agent.total_tokens_used (core.py) - each
provider's actual usage/usageMetadata is captured per response, not
estimated. That was the prerequisite this module was blocked on through
Phase 0-2.

Deliberately simple for v1, per the architecture roadmap: fixed percentage
shares by rank (scaled by an optional per-employee importance weight), not a
full water-filling optimizer across the whole org - that's a real scheduling
problem, worth targeting once this simpler version proves the escalation path
end to end. Checked BEFORE an employee starts a new top-level task, not
enforced mid-generation - llmadapt has no way to interrupt a request already
in flight with a real provider, so this is a pre-flight gate + governance
ledger, not a hard runtime cutoff mid-response.

**Not all tokens are equal**, and since Phase 4 that stopped being theoretical:
an intern on a local Ollama model and a C-suite on a frontier API model both
spend "tokens", and this ledger charged them at identical rates. A company that
is half local and half API therefore had a budget that did not mean what it
looked like - the free half consumed the same ceiling as the expensive half.
`CostModel` is the answer, and it is opt-in: leave it off and every existing
config behaves byte-identically.
"""

import logging
from typing import Any, Dict, Optional, Sequence

logger = logging.getLogger(__name__)

# Default per-rank share of the company's total_token_budget, before any
# per-employee importance scaling. Reflects the "don't waste the expensive
# employees' time" principle from the original plan: cheaper, more numerous
# worker tiers get more of the raw token budget because they do more of the
# actual token-churning work; C-suite/GM interactions are rarer and shorter.
# Override via Company(rank_budget_shares=...) if a different split fits your
# org better - these are a sensible default, not a law.
DEFAULT_RANK_BUDGET_SHARES: Dict[str, float] = {
    "C_SUITE": 0.05,
    "GENERAL_MANAGER": 0.08,
    "MANAGER": 0.12,
    "SENIOR": 0.20,
    "JUNIOR": 0.30,
    "INTERN": 0.20,
    "VOLUNTEER": 0.05,
}


def normalize_shares(shares: Dict[str, float]) -> Dict[str, float]:
    """Rescales an arbitrary rank->share dict so its values sum to 1.0, so a
    caller can pass rough relative weights (e.g. {"MANAGER": 2, "JUNIOR": 5})
    instead of having to hand-compute exact percentages."""
    total = sum(shares.values())
    if total <= 0:
        return dict(shares)
    return {k: v / total for k, v in shares.items()}


# Provider names that mean "the weights are on this machine", so the tokens are
# free. These are selector.py's names plus the two transports policy.py binds
# them to. Matched as substrings of a lowercased provider string, since a
# caller may write "ollama" or "Ollama (local)".
LOCAL_PROVIDER_HINTS: Sequence[str] = (
    "ollama", "lm-studio", "lmstudio", "llama.cpp", "llamacpp", "vllm", "local", "hf",
)

# A base_url pointing at this machine also means local, whatever the provider
# string says - policy.py's DEFAULT_LOCAL_BINDINGS routes lm-studio and vllm
# through provider="openai" at a localhost port, so the provider name alone
# would read those as paid API traffic.
LOCAL_URL_HINTS: Sequence[str] = ("localhost", "127.0.0.1", "0.0.0.0", "::1")


class CostModel:
    """Turns "what is this employee actually running on" into a multiplier on
    the tokens they spend, so the ledger can charge cost-equivalent tokens
    rather than raw ones.

    **Why this and not a second price table.** Phase 4 already built the only
    pricing knowledge in the library - `ApiModelSpec.blended_cost_per_1k()` and
    the catalog the user supplies - and `PolicyDecision.estimated_cost_per_1k`
    is already stored on every routed employee as `employee.model_decision`.
    Adding a parallel per-rank weight table would mean a second set of numbers
    to hand-maintain and keep in sync with which employee is on which model,
    which is precisely the staleness Phase 4's own docstring warns about. So
    this reads the numbers that already exist, in this order:

      1. `hire(cost_weight=...)` - an explicit override always wins.
      2. `employee.model_decision.estimated_cost_per_1k` - what the Phase 4
         policy actually decided, the most accurate source available.
      3. `model_map[rank]["cost_per_1k"]` - a plain number on the user's own
         model_map, for a company that pins providers by hand and never
         attaches a ModelPolicy at all. This is the "use the model dict the
         user provides" path, and it means cost weighting does not require
         Phase 4 to be in play.
      4. A lookup of the employee's live provider/model in the API catalog,
         for an employee whose model was set literally but happens to be a
         model the catalog knows the price of.
      5. Local detection - an Ollama/LM-Studio/vLLM provider, or any base_url
         on this machine - costs nothing to call, so it weighs `local_weight`
         (0.0 by default: free tokens consume no budget).
      6. `default_weight` (1.0) when nothing above knows. Deliberately not 0 -
         an unknown model is far more likely to be a paid one than a free one,
         and the failure directions are not symmetric: guessing 1.0 makes the
         budget slightly conservative, guessing 0.0 makes it unenforceable.

    **What a weight of 1.0 means.** Cost per 1k is divided by `baseline_per_1k`
    to get the multiplier, so 1.0 is "a typical model in your catalog" - the
    median blended cost across it, computed once at construction and then
    frozen. Frozen because a baseline that moved as employees were hired would
    make the ledger non-monotonic: the same spend would cross the ceiling or
    not depending on who was hired afterwards, and a budget report from an hour
    ago would no longer reconcile.

    Nothing here is a spend forecast. It is a *ratio* between models, built on
    the same hand-entered catalog Phase 4 flags as going stale, and it inherits
    that caveat wholesale.
    """

    def __init__(
        self,
        baseline_per_1k: Optional[float] = None,
        api_catalog: Optional[Any] = None,
        model_map: Optional[Dict[str, Dict[str, Any]]] = None,
        local_weight: float = 0.0,
        default_weight: float = 1.0,
    ):
        self.api_catalog = api_catalog
        self.model_map = model_map or {}
        self.local_weight = max(0.0, float(local_weight))
        self.default_weight = max(0.0, float(default_weight))
        self.baseline_per_1k = self._resolve_baseline(baseline_per_1k)
        self._explanations: Dict[str, str] = {}

    def _resolve_baseline(self, given: Optional[float]) -> float:
        """The cost that counts as weight 1.0. Explicit, else the catalog's
        median, else 1.0 (which makes weights literal dollars-per-1k - odd
        units, but a defined and documented fallback rather than a crash)."""
        if given is not None and given > 0:
            return float(given)
        costs = sorted(
            cost for cost in (self._catalog_costs()) if cost > 0
        )
        if not costs:
            return 1.0
        middle = len(costs) // 2
        return costs[middle] if len(costs) % 2 else (costs[middle - 1] + costs[middle]) / 2

    def _catalog_costs(self) -> Sequence[float]:
        if self.api_catalog is None:
            return ()
        try:
            return [spec.blended_cost_per_1k() for spec in self.api_catalog.all()]
        except Exception:  # pragma: no cover - a duck-typed catalog that isn't one
            return ()

    @staticmethod
    def _looks_local(provider: Optional[str], base_url: Optional[str]) -> bool:
        text = (provider or "").strip().lower()
        url = (base_url or "").strip().lower()
        if any(hint in url for hint in LOCAL_URL_HINTS):
            return True
        return any(hint in text for hint in LOCAL_PROVIDER_HINTS)

    def _catalog_cost_for(self, provider: Optional[str], model: Optional[str]) -> Optional[float]:
        if self.api_catalog is None or not model:
            return None
        try:
            spec = self.api_catalog.find(model, provider)
        except Exception:  # pragma: no cover - duck-typed catalog
            return None
        return spec.blended_cost_per_1k() if spec is not None else None

    def weight_for(self, employee: Any) -> float:
        """The multiplier for one employee, by the order in the class
        docstring. Also records a one-line explanation, so `budget_report()`
        can say why an employee is charged what it is - the same "the decision
        stays answerable later" rule Phase 4's PolicyDecision.reason follows.
        """
        override = getattr(employee, "cost_weight", None)
        if override is not None:
            self._explanations[employee.name] = f"explicit cost_weight={float(override):g}"
            return max(0.0, float(override))

        agent = getattr(employee, "agent", None)
        provider = getattr(agent, "provider", None)
        model = getattr(agent, "model", None)
        base_url = getattr(agent, "base_url", None)

        decision = getattr(employee, "model_decision", None)
        cost = getattr(decision, "estimated_cost_per_1k", None) if decision is not None else None
        source = "the Phase 4 policy decision"

        if not cost:
            rank_config = self.model_map.get(getattr(employee, "rank", ""), {}) or {}
            configured = rank_config.get("cost_per_1k")
            if configured:
                cost, source = float(configured), "model_map[rank]['cost_per_1k']"

        if not cost:
            found = self._catalog_cost_for(provider, model)
            if found:
                cost, source = found, "a match in the API catalog"

        if cost:
            weight = float(cost) / self.baseline_per_1k
            self._explanations[employee.name] = (
                f"~${cost:.5f}/1k from {source}, / ${self.baseline_per_1k:.5f} baseline"
            )
            return max(0.0, weight)

        if self._looks_local(provider, base_url):
            self._explanations[employee.name] = (
                f"provider {provider!r} runs locally, so its tokens are free"
            )
            return self.local_weight

        self._explanations[employee.name] = (
            f"no price known for {provider!r}/{model!r}, charged at the default weight"
        )
        return self.default_weight

    def charged(self, employee: Any, raw_tokens: int) -> int:
        """Raw tokens converted to cost-equivalent tokens for the ledger."""
        return int(round(raw_tokens * self.weight_for(employee)))

    def explain(self, employee: Any) -> str:
        """Why `employee` weighs what it does. Empty until `weight_for` has
        run for them at least once."""
        self.weight_for(employee)
        return self._explanations.get(getattr(employee, "name", ""), "")


class BudgetLedger:
    """Tracks allocation (not spend - that lives on each Employee's Agent via
    total_tokens_used) for one Company. Kept as its own class rather than more
    Company methods, since this is a genuinely separate concern with its own
    state - same reasoning as observability.py being split out."""

    def __init__(
        self,
        total_token_budget: Optional[int],
        rank_shares: Optional[Dict[str, float]] = None,
        cost_model: Optional[CostModel] = None,
    ):
        self.total_token_budget = total_token_budget
        self.rank_shares = normalize_shares(rank_shares or DEFAULT_RANK_BUDGET_SHARES)
        # None = charge raw tokens, exactly as before Phase 3's cost weighting
        # existed. Set it and the ledger's unit becomes cost-equivalent tokens.
        self.cost_model = cost_model
        # Extra allocation granted to a specific employee beyond their base
        # rank share - via sibling/manager reallocation or an escalation
        # grant. Never negative; this only ever adds headroom, it doesn't
        # literally debit anyone else's ledger (see company.py's
        # _try_reallocate for why that simplification is safe here).
        self._bonus_allocation: Dict[str, int] = {}

    def base_allocation(self, rank: str, importance: float = 0.5) -> Optional[int]:
        """None means unlimited (no total_token_budget set on the Company).
        importance in [0, 1] scales the rank's base share from 0.5x (0.0) to
        1.5x (1.0) - a blunt but honest v1 knob; true per-task sizing (e.g.
        from an estimated output length) is future work, not implemented."""
        if self.total_token_budget is None:
            return None
        share = self.rank_shares.get(rank, 0.0)
        scaled = share * (0.5 + max(0.0, min(1.0, importance)))
        return int(self.total_token_budget * scaled)

    def allocated_for(self, rank: str, name: str, importance: float = 0.5) -> Optional[int]:
        base = self.base_allocation(rank, importance)
        if base is None:
            return None
        return base + self._bonus_allocation.get(name, 0)

    def remaining(self, rank: str, name: str, importance: float, spent: int) -> Optional[int]:
        allocated = self.allocated_for(rank, name, importance)
        if allocated is None:
            return None
        return allocated - spent

    def grant_bonus(self, name: str, amount: int) -> None:
        if amount <= 0:
            return
        self._bonus_allocation[name] = self._bonus_allocation.get(name, 0) + amount

    def charged_spend(self, employee: Any) -> int:
        """What this employee's spend counts as against the budget.

        Raw `Agent.total_tokens_used` with no cost model, cost-equivalent
        tokens with one. Every budget check goes through here rather than
        reading `total_tokens_used` directly, so there is exactly one place
        where the ledger's unit is decided.
        """
        raw = employee.agent.total_tokens_used
        if self.cost_model is None:
            return raw
        return self.cost_model.charged(employee, raw)

    def report(self, employees: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        """employees: Company.employees (name -> Employee). Returns a plain
        dict per employee - allocated/spent/remaining - for Company.budget_report().

        With a cost model attached, `spent` is the charged figure the gate
        actually compares against, and `raw_tokens` carries the unweighted
        count beside it. Both, deliberately: the weighted number is the one
        that explains a budget decision, and the raw one is the only figure
        that reconciles against a provider's own usage page.
        """
        out = {}
        for name, employee in employees.items():
            raw = employee.agent.total_tokens_used
            spent = self.charged_spend(employee)
            allocated = self.allocated_for(employee.rank, employee.name, employee.importance)
            entry: Dict[str, Any] = {
                "rank": employee.rank,
                "allocated": allocated,  # None means unlimited
                "spent": spent,
                "remaining": None if allocated is None else allocated - spent,
            }
            if self.cost_model is not None:
                entry["raw_tokens"] = raw
                entry["cost_weight"] = round(self.cost_model.weight_for(employee), 4)
                entry["cost_basis"] = self.cost_model.explain(employee)
            out[name] = entry
        return out
