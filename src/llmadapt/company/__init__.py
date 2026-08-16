"""llmadapt.company - the multi-agent hierarchy: Employee/Team/Company,
config-driven Agent creation per rank (with per-employee overrides),
delegation wired as a normal tool call, and tool-iteration AND token-budget
failures both escalating up the reporting chain instead of raising straight to
the caller.

(Phase 2) queryable activity/tool-call logs plus org-chart rendering
(ascii/mermaid/svg) - see observability.py.

(Phase 3) real percentage-based token budget governance - see budget.py.

(Phase 4) "mode" in model_map / hire() is real: with a ModelPolicy attached
(see policy.py), a rank or an individual employee can be routed automatically
between local and API models, steered by an effort/priority hint that is
accepted at hire time and per task.

(Phase 5) Skills, personalities, org templates and org-chart palettes all come
from ONE named-preset registry (presets.py) - hire(skills=, personality=),
build_from_template(), render_org_chart(palette=). Team's `reviewer` field
actually reviews; see Team's docstring for the decision.

(Phase 6) run_structured(task, strategy="direct"|"plan"|"stub_fill") routes a
task through delegation.py's decomposition strategies.

(Phase 7) Company(log_compaction=LogCompactionPolicy(...)) compacts the
activity/tool-call logs after every completed top-level task - see
compaction.py.

(Phase 8) builder.set_company_up(mode="text"|"gui") builds a whole Company
from a spec, a schema-driven tool call, or an interactive node-graph editor.

**This used to be a single 1100-line module and is now a package.** The split
was flagged in the file itself since Phase 0 and deferred through the Phase 4-8
run only because moving files mid-phase makes every phase's diff unreadable.
Only `company.py`'s own classes moved; the eight modules that orbit it
(`budget`, `observability`, `policy`, `presets`, `delegation`, `compaction`,
`builder`, `gui`) stayed exactly where they were, so
`from llmadapt.policy import ModelPolicy` is untouched.

Every name that used to be importable from `llmadapt.company` is re-exported
here, so `from llmadapt.company import Company, Employee, Team` resolves
exactly as it did before the split. `tests/test_import_paths.py` asserts that
for every public export, from both the package root and the submodule, so this
guarantee cannot rot silently.

Layout:
    escalation.py  EscalationEvent / EscalationDecision / EscalationUnresolved
    employee.py    Employee
    team.py        Team + the review-loop prompts
    company.py     Company
"""

from .company import POLICY_MODES, Company, PausedRun, PolicyMode, mode
from .employee import Employee
from .escalation import (
    ESCALATION_PENDING,
    EscalationDecision,
    EscalationEvent,
    EscalationUnresolved,
    RunPaused,
    always_approve,
    always_decline,
    default_on_escalation,
)
from .team import REVIEW_MODES, ReviewMode, Team, review

# `mode`/`review` are reachable here (`from llmadapt.company import mode`) and
# from their owning submodule (`llmadapt.company.company.mode` /
# `llmadapt.company.team.review`) - deliberately NOT from the bare top-level
# `llmadapt` package. See PolicyMode/ReviewMode's own docstrings for why,
# same reasoning router.py's `role` uses for RoleRank.

# Private names that were importable from the old single-file `company.py`, and
# stay importable from the package so nothing reaching for them broke in the
# split - a test may monkeypatch a review prompt or call `_looks_approved`
# directly, and silently breaking that is exactly what a "no import path
# changed" promise exists to prevent.
from .escalation import _EMERGENCY_BUDGET_DRAW_FRACTION, _EMERGENCY_GRANT_ITERATIONS
from .team import _REVIEW_PROMPT, _REVISION_PROMPT, _looks_approved

# Referencing them here is what makes the re-export deliberate rather than
# something a linter reads as five stray imports. Preferred over a `# noqa`
# (flake8 only - Pylance keeps reporting) or the `X as X` alias form (pyright
# and mypy only - pyflakes keeps reporting): a plain tuple is understood by
# every checker there is, and it doubles as the list of what this promise
# actually covers. Deliberately not in `__all__` - these stay reachable, not
# advertised, and `import *` should not pull them in.
_PRIVATE_REEXPORTS = (
    _EMERGENCY_BUDGET_DRAW_FRACTION,
    _EMERGENCY_GRANT_ITERATIONS,
    _REVIEW_PROMPT,
    _REVISION_PROMPT,
    _looks_approved,
)

__all__ = [
    "Company",
    "Employee",
    "Team",
    "PausedRun",
    "ESCALATION_PENDING",
    "RunPaused",
    "EscalationEvent",
    "EscalationDecision",
    "EscalationUnresolved",
    "default_on_escalation",
    "always_decline",
    "always_approve",
    "PolicyMode",
    "mode",
    "POLICY_MODES",
    "ReviewMode",
    "review",
    "REVIEW_MODES",
]
