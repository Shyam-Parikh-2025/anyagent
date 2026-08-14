# company.py roadmap — status and forward-looking design notes

Living reference for the multi-agent hierarchy work in `llmadapt`. Update this
alongside the code as phases land, so a future session (or a different AI
picking this up) doesn't have to reconstruct the plan from chat scrollback.

## Done

**Phase 0-1 — skeleton, delegation, escalation** (`company.py`, `router.py`)
`RoleRank` extended with `GENERAL_MANAGER`/`VOLUNTEER` + `ORDER` (seniority,
not capability). `Employee`/`Team`/`Company` classes. `Company.hire(name,
rank, reports_to=...)` builds an `Agent` from `model_map[rank]` and — if
`reports_to` is given — registers a `delegate_to_<name>` tool directly on the
manager's `Agent`, so delegation is just the normal tool-calling loop, no new
plumbing. Tool-iteration overflow escalates up the `reports_to` chain to
`Company._handle_iteration_escalation`, which tries the
`emergency_iteration_reserve` first (measured in *tool iterations*, unrelated
to real token cost) and only then calls the `on_escalation(event) ->
EscalationDecision` callback. `0` emergency budget means always ask first —
the deliberate cost-control default. (Renamed from `emergency_token_budget`/
`_emergency_remaining` in Phase 3 once a *real* token-denominated reserve
existed and needed the less ambiguous name.)

**Phase 2 — observability** (`observability.py`)
`EventLog` — live queryable wrapper (`by_employee`, `by_kind`, `since`) over
`Company.activity_log` / `Company.tool_call_log`, exposed via
`Company.activity()` / `Company.tool_calls()`. `Company.render_org_chart(fmt=
"ascii"|"mermaid"|"svg", theme="light"|"dark")` — the SVG renderer is a pure
Python tree layout (no graphviz/matplotlib dependency), colored with the
dataviz skill's fixed-order categorical palette (one hue per rank tier,
validated for colorblind-safe adjacent contrast; text always in ink tokens,
never the rank color itself).

**Phase 3 — budget governance** (`budget.py`, `company.py`, `core.py`)
`core.py`'s `Agent` now captures real per-provider usage on every response
(`_extract_usage`/`_record_usage`, called from `process_response` so both
sync and async paths get it) — `total_tokens_used`/`usage_log`, not a proxy
unit anymore. `budget.py`'s `BudgetLedger` does fixed percentage-of-total
splits by rank (`DEFAULT_RANK_BUDGET_SHARES`), scaled 0.5x-1.5x by a
per-employee `importance` (0.0-1.0, now a real `hire(..., importance=...)`
param). `Company._budget_gate()` runs before every `Employee.run()` — a
pre-flight check, not a mid-generation cutoff (llmadapt has no way to
interrupt a request already in flight with a real provider), checked in this
order: (1) the company-wide `total_token_budget` hard ceiling — once total
spend reaches it, *every* further request needs `on_escalation`, no internal
reallocation can cross it; (2) the employee's own rank share — if exhausted,
`_try_reallocate()` walks the `reports_to` chain for a manager with slack to
lend (a simplified, non-water-filling protocol — it grants a bonus rather
than literally debiting the lender, which is safe precisely because the hard
ceiling above is the real backstop) before escalating. `budget_exhausted`
escalations reuse the same `EscalationEvent`/`on_escalation` machinery as
`tool_iteration_limit` ones, with their own `emergency_budget_tokens` reserve
(skipped entirely once the hard ceiling itself is hit, same 0-means-always-ask
default as the iteration reserve) and `EscalationDecision.extra_token_budget`
for human-approved recovery — approving a *hard-ceiling* escalation actually
raises `total_token_budget` by the granted amount, since that's the only way
forward once the whole company is out of room. `Company.total_tokens_spent()`
and `Company.budget_report()` expose the live numbers. Per-employee
`provider`/`model`/`api_key` overrides landed on `hire()` at the same time
(open question #1 below — resolved yes, do it now).

Deliberately still simple, per the original v1 scope: fixed rank shares +
recursive ask-your-manager reallocation, not a global water-filling optimizer
— that's real scheduling, worth targeting once this simpler version has run
for a while.

## Not built yet

**Phase 4 — auto mode model policy (local vs. API)**
Builds on `benchmark.py`/`selector.py`/`router.py.allocate_local_auto`, which
today only rank among *local* candidates. Phase 4 adds the local-vs-API
decision itself — needs a cost/specialty table for API models (start with a
small user-supplied config, not live-pricing fetch — real scope trap
otherwise). **New requirement to design in from the start:** the selection
must accept an optional per-employee/per-task *effort or priority hint*
("needs effort", "keep it cheap") as an input, not just rank — don't bolt
this on after Phase 4 ships once the API shape is already fixed.

**Phase 5 — skills/personality presets, default org templates, color palettes**
A **single named-preset registry pattern**, reused identically across skills,
personality presets, team/company templates, and now color palettes (per this
session's note: ready-made palettes + user-added custom ones, referenced by
name — same mechanism as skills/teams, not a bespoke system bolted onto
`observability.py`). Presets determine system-instruction templating. Skills
can carry constraints ("don't use library X"). Default org templates (e.g.
"small coding team": 1 reviewer + 1 C-suite + 1 manager) scale with task size
when the user opts in.

**Phase 6 — stub-and-fill delegation**
Higher tiers emit function signatures + docstrings + type hints, cheaper
tiers fill bodies. Fixed by rank in v1 (not a dynamically-decided
decomposition-depth algorithm — no clean formula for that; treat it as a
research-adjacent stretch goal, not a v1 requirement).

**Phase 7 — log compaction**
The `compressor.py`-based "keep broad concepts, drop specifics" cleanup pass
over `activity_log`/`tool_call_log` after a task completes. Logs are recorded
and queryable now (Phase 2); nothing prunes or summarizes them yet.

**Phase 8 — `set_company_up(mode="text"|"gui")` (new, this session)**
Two builds under one name, different scope entirely:

- *Text/schema mode*: for an AI caller. Should reuse `core.py`'s existing
  `ToolRegistry` schema-auto-generation (from type hints + docstring) rather
  than inventing a parallel schema format — expose it as a normal tool with
  an auto-derived schema, return value is the structured "what got built"
  the calling AI needs. Depends on Phase 5's registries existing (a schema
  needs to reference skills/teams/colors by name), so this comes after.
- *GUI mode*: a real interactive node-graph editor — click-to-connect
  employees, multi-select + right-click bulk actions ("connect all to one
  manager"), a left panel for per-employee model/API assignment, a left menu
  for skill/team/color presets. This is not a Python function, it's a small
  app: most likely a self-contained local web page (vanilla JS + SVG/canvas,
  Python's `http.server` for a tiny local save/load API, opened via
  `webbrowser.open()`) to stay in step with llmadapt's zero-dependency
  ethos rather than pulling in a frontend framework. Meaningfully bigger than
  everything built so far combined — its own phase, tackled last.

## Open questions to resolve before Phase 4 code

1. Phase 4's local-vs-API decision needs an effort/priority hint input (see
   above) — worth checking whether that hint should also feed
   `BudgetLedger.importance` (an expensive/high-effort task arguably deserves
   a bigger allocation too), or whether keeping "how much budget" and "how
   capable a model" as separate knobs is the right call. Not resolved yet;
   flagging so Phase 4 doesn't quietly redefine what `importance` means.
2. Ready-made color palettes + custom palette support for `render_org_chart`
   (Phase 3/4-adjacent idea, explicitly deferred by the user until Phase 5's
   named-preset registry exists) — still open, still fine to leave for Phase 5.
