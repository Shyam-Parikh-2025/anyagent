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
"""

from typing import Any, Dict, Optional

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


class BudgetLedger:
    """Tracks allocation (not spend - that lives on each Employee's Agent via
    total_tokens_used) for one Company. Kept as its own class rather than more
    Company methods, since this is a genuinely separate concern with its own
    state - same reasoning as observability.py being split out."""

    def __init__(self, total_token_budget: Optional[int], rank_shares: Optional[Dict[str, float]] = None):
        self.total_token_budget = total_token_budget
        self.rank_shares = normalize_shares(rank_shares or DEFAULT_RANK_BUDGET_SHARES)
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

    def report(self, employees: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        """employees: Company.employees (name -> Employee). Returns a plain
        dict per employee - allocated/spent/remaining - for Company.budget_report()."""
        out = {}
        for name, employee in employees.items():
            spent = employee.agent.total_tokens_used
            allocated = self.allocated_for(employee.rank, employee.name, employee.importance)
            out[name] = {
                "rank": employee.rank,
                "allocated": allocated,  # None means unlimited
                "spent": spent,
                "remaining": None if allocated is None else allocated - spent,
            }
        return out
