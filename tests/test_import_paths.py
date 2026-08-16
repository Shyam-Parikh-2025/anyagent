"""test_import_paths.py - the guard on the company/ package split.

`company.py` became `company/` in this session. The whole promise of that
split was that *no import path changed* - `from llmadapt.company import
Company` had to keep resolving, and the eight modules orbiting company.py
(policy, budget, presets, ...) had to stay exactly where they were so
`from llmadapt.policy import ModelPolicy` never becomes
`from llmadapt.company.policy import ModelPolicy`.

A promise like that rots silently: nothing else in the suite fails if a
re-export is dropped from `company/__init__.py`, because every other test
imports from `llmadapt` directly. So this file asserts it explicitly, for
every public export, from both the package root and the owning submodule.

Same style as the rest of the suite: plain asserts, plain prints, no pytest.
"""

import importlib
import importlib.util
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import llmadapt

checks = 0


def check(label, condition):
    global checks
    assert condition, label
    checks += 1
    print(f"PASS: {label}")


# --- 1. every advertised export actually resolves from the package root ----
missing = [name for name in llmadapt.__all__ if not hasattr(llmadapt, name)]
check(
    f"all {len(llmadapt.__all__)} names in llmadapt.__all__ import from the package root",
    not missing,
)

# --- 2. the pre-split company paths still resolve --------------------------
# These are the exact import statements a caller wrote before the split. If any
# of them stops working, the split broke its own contract.
from llmadapt.company import (  # noqa: E402
    Company,
    Employee,
    EscalationDecision,
    EscalationEvent,
    EscalationUnresolved,
    Team,
)

check("llmadapt.company still exports Company/Employee/Team", all(
    isinstance(obj, type) for obj in (Company, Employee, Team)
))
check("llmadapt.company still exports the escalation types", all(
    isinstance(obj, type) for obj in (EscalationEvent, EscalationDecision, EscalationUnresolved)
))

from llmadapt.company import always_approve, always_decline, default_on_escalation  # noqa: E402

check("llmadapt.company exports the ready-made escalation handlers too", all(
    callable(obj) for obj in (default_on_escalation, always_decline, always_approve)
))
check("llmadapt.default_on_escalation is the same function as llmadapt.company's",
      llmadapt.default_on_escalation is default_on_escalation)
check("llmadapt.builder.default_on_escalation is the same function too - one definition, not two",
      importlib.import_module("llmadapt.builder").default_on_escalation is default_on_escalation)

# The same objects, not copies - a shim that rebuilt the class would silently
# break `isinstance` and `except EscalationUnresolved` across module boundaries.
check("llmadapt.Company is llmadapt.company.Company (one class, not a copy)",
      llmadapt.Company is Company)
check("llmadapt.EscalationUnresolved is the same exception class",
      llmadapt.EscalationUnresolved is EscalationUnresolved)

# The carve-out that lets a declined escalation escape a delegate_to_<name>
# tool call. `except EscalationUnresolved` and the `except ToolControlFlow` in
# ToolRegistry.execute have to be talking about the same class object across
# the core/company boundary - if a shim ever rebuilt either one, the decline
# would silently go back to being stringified into a tool result and the run
# would carry on past a human saying no.
from llmadapt.core import ToolControlFlow  # noqa: E402

check("llmadapt.ToolControlFlow is llmadapt.core.ToolControlFlow (one class, not a copy)",
      llmadapt.ToolControlFlow is ToolControlFlow)
check("EscalationUnresolved subclasses ToolControlFlow, so ToolRegistry.execute re-raises it",
      issubclass(EscalationUnresolved, ToolControlFlow))
check("ToolControlFlow is still an Exception, so `except Exception` at the top of a run catches it",
      issubclass(ToolControlFlow, Exception))

# --- 3. the orbiting modules did NOT move ---------------------------------
# The split moved company.py's own classes only. If a later refactor pulls
# policy.py (or any sibling) under company/, this fails loudly rather than
# quietly forcing every caller to rewrite their imports.
SIBLINGS = {
    "llmadapt.budget": ["BudgetLedger"],
    "llmadapt.observability": ["EventLog"],
    "llmadapt.policy": ["ModelPolicy", "ApiModelCatalog", "PolicyDecision"],
    "llmadapt.presets": ["PresetRegistry", "PresetBundle", "compose_system_instruction"],
    "llmadapt.delegation": ["stub_and_fill", "plan_then_execute",
                            "stub_and_fill_async", "plan_then_execute_async"],
    "llmadapt.compaction": ["LogCompactionPolicy"],
    "llmadapt.state": ["CompanyState", "StateMismatch", "capture_state", "apply_state"],
    "llmadapt.builder": ["CompanySpec", "build_company", "make_company_via_gui", "quick_company",
                         "preset_descriptions", "company_help"],
    "llmadapt.help": ["company_help", "HELP_VERSION"],
    "llmadapt.core": ["Agent", "Conversation", "ToolRegistry", "ToolControlFlow",
                      "run_coroutine_blocking", "loop_local_lock"],
    "llmadapt.compressor": ["ContextCompressor", "HistoryCompactionPolicy"],
    "llmadapt.router": ["ModelRouter", "RoleRank", "role"],
}
for module_name, names in SIBLINGS.items():
    module = importlib.import_module(module_name)
    absent = [n for n in names if not hasattr(module, n)]
    check(f"{module_name} still exists at the top level with {', '.join(names)}", not absent)

import llmadapt.router as router_module  # noqa: E402
import llmadapt.company.company as company_module  # noqa: E402
import llmadapt.company.team as team_module  # noqa: E402

check("router.role is llmadapt.RoleRank - a second name, not a second definition",
      router_module.role is llmadapt.RoleRank)
check("role.SENIOR reads the same string as RoleRank.SENIOR",
      router_module.role.SENIOR == llmadapt.RoleRank.SENIOR == "SENIOR")
check("role is NOT re-exported from the bare top-level llmadapt package - opt in via "
      "llmadapt.router (or llmadapt.company for mode/review) required",
      not hasattr(llmadapt, "role") and "role" not in llmadapt.__all__)

check("company.mode is llmadapt.PolicyMode - a second name, not a second definition",
      company_module.mode is llmadapt.PolicyMode)
check("mode.AUTO/.LOCAL/.API match company.company.POLICY_MODES",
      (llmadapt.PolicyMode.AUTO, llmadapt.PolicyMode.LOCAL, llmadapt.PolicyMode.API)
      == company_module.POLICY_MODES == ("auto", "local", "api"))
check("mode is reachable from llmadapt.company but NOT from the bare top-level llmadapt package",
      llmadapt.company.mode is llmadapt.PolicyMode
      and not hasattr(llmadapt, "mode") and "mode" not in llmadapt.__all__)

check("team.review is llmadapt.ReviewMode - a second name, not a second definition",
      team_module.review is llmadapt.ReviewMode)
check("review.CRITIQUE/.APPEND/.OFF match company.team.REVIEW_MODES",
      (llmadapt.ReviewMode.CRITIQUE, llmadapt.ReviewMode.APPEND, llmadapt.ReviewMode.OFF)
      == team_module.REVIEW_MODES == ("critique", "append", "off"))
check("review is reachable from llmadapt.company but NOT from the bare top-level llmadapt package",
      llmadapt.company.review is llmadapt.ReviewMode
      and not hasattr(llmadapt, "review") and "review" not in llmadapt.__all__)

# builder.py and gui.py must both read the *same* REVIEW_MODES/POLICY_MODES
# objects rather than each hardcoding their own copy of the three-item list -
# the exact duplication ReviewMode/REVIEW_MODES was added to close off.
gui_module = importlib.import_module("llmadapt.gui")
builder_module = importlib.import_module("llmadapt.builder")
check("gui.py and builder.py both import the one REVIEW_MODES, not separate literals",
      gui_module.REVIEW_MODES is builder_module.REVIEW_MODES is team_module.REVIEW_MODES)
check("gui.py and builder.py both import the one POLICY_MODES, not separate literals",
      gui_module.POLICY_MODES is builder_module.POLICY_MODES is company_module.POLICY_MODES)

check(
    "no sibling module was moved under llmadapt.company",
    not any(
        importlib.util.find_spec(f"llmadapt.company.{name.split('.')[-1]}")
        for name in ("llmadapt.policy", "llmadapt.budget", "llmadapt.presets")
    ),
)

# --- 4. the packages' own submodules are reachable by their real paths -----
for sub in ("company", "employee", "team", "escalation"):
    module = importlib.import_module(f"llmadapt.company.{sub}")
    check(f"llmadapt.company.{sub} imports cleanly", module is not None)

for sub in ("registry", "skills", "personalities", "palettes", "org_templates",
            "bundle", "compose"):
    module = importlib.import_module(f"llmadapt.presets.{sub}")
    check(f"llmadapt.presets.{sub} imports cleanly", module is not None)

# --- 5. presets.py became a package too, on the same contract --------------
# Half of it was content rather than mechanism, and content is what grows. The
# promise is identical to the company/ split: nothing a caller wrote changed.
from llmadapt.presets import (  # noqa: E402
    ORG_TEMPLATES,
    PALETTES,
    PERSONALITIES,
    SKILLS,
    OrgTemplate,
    Palette,
    Personality,
    Preset,
    PresetBundle,
    PresetRegistry,
    RoleSpec,
    Skill,
    compose_system_instruction,
    default_bundle,
    skill_hints,
)

check("every pre-split llmadapt.presets name still imports from the package root",
      all(obj is not None for obj in (
          Preset, PresetRegistry, PresetBundle, Skill, Personality, Palette,
          OrgTemplate, RoleSpec, SKILLS, PERSONALITIES, PALETTES, ORG_TEMPLATES,
          compose_system_instruction, skill_hints, default_bundle)))

check("the registries are the same objects the submodules hold, not copies",
      SKILLS is importlib.import_module("llmadapt.presets.skills").SKILLS
      and PALETTES is importlib.import_module("llmadapt.presets.palettes").PALETTES)

check("the mechanism lives alone in registry.py, with no catalog beside it",
      not hasattr(importlib.import_module("llmadapt.presets.registry"), "BUILTIN_SKILLS"))

# --- 6. every __all__ must be honest --------------------------------------
# The failure this catches is invisible to every other test in the suite: a
# module whose __all__ names things it does not define imports fine, runs fine,
# and only breaks on `from mod import *` - or in an editor, which is where it
# actually surfaced. presets/compose.py inherited the whole package's __all__
# when presets.py was split, listing fourteen names it had never heard of.
import pkgutil  # noqa: E402

modules = ["llmadapt"] + [m.name for m in pkgutil.walk_packages(llmadapt.__path__, "llmadapt.")]
dishonest = []
star_failures = []
for module_name in sorted(modules):
    module = importlib.import_module(module_name)
    declared = getattr(module, "__all__", None)
    if declared is None:
        continue
    missing = [n for n in declared if not hasattr(module, n)]
    if missing:
        dishonest.append(f"{module_name}: {missing}")
    try:
        exec(f"from {module_name} import *", {})
    except Exception as e:  # pragma: no cover - the thing being guarded against
        star_failures.append(f"{module_name}: {e!r}")

check(f"every __all__ across all {len(modules)} modules names only things that exist",
      not dishonest)
check("`from <module> import *` works for every module in the package",
      not star_failures)

# The private re-exports the company/ split kept are reachable but deliberately
# not advertised - `import *` must not pull an underscore name in.
import llmadapt.company as company_pkg  # noqa: E402

check("private names survived the split and are still reachable",
      all(hasattr(company_pkg, n) for n in
          ("_looks_approved", "_REVIEW_PROMPT", "_REVISION_PROMPT",
           "_EMERGENCY_GRANT_ITERATIONS", "_EMERGENCY_BUDGET_DRAW_FRACTION")))
check("...but they are not in __all__, so `import *` leaves them out",
      not any(n.startswith("_") for n in company_pkg.__all__))


print(f"\nOK - {checks} checks")
