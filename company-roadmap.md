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

**Phase 4 — auto mode model policy (local vs. API)** (`policy.py`, `company.py`)
`policy.py` adds the layer above `selector.py`/`router.allocate_local_auto`:
those rank among *local* candidates, this decides whether to be local at all.

`ApiModelSpec`/`ApiModelCatalog` are the user-supplied cost/specialty table
(per-1k input+output price, a coarse hand-set `capability` 0-1, free-form
`specialties` tags, optional `context_window`). `DEFAULT_API_CATALOG` seeds
five common models so a bare `ModelPolicy()` works, but **every seed entry
carries a `note` saying the numbers are hand-entered and will go stale** —
there is deliberately no live-pricing fetch (the flagged scope trap). Prices
are collapsed to one comparable number via `blended_cost_per_1k(output_ratio=
0.75)`; that ratio is an assumption about agent traffic (short prompts, long
completions), stated as such — it's a ranking aid, not a spend forecast.

**The effort hint is an input everywhere, from the start** —
`ModelPolicy.decide(effort=)`, `Company.hire(effort=)`, `Company.run(effort=)`.
Three levels only (`cheap`/`balanced`/`effort`) with a small fixed alias table
("needs effort", "keep it cheap", "high", …); three because a user cannot
calibrate seven levels against a catalog they wrote by hand. An unrecognized
string falls back to the rank default rather than raising, because in Phase 8
this value arrives from an LLM filling in a tool schema and a typo shouldn't
take a running company down. `DEFAULT_RANK_EFFORT` supplies the fallback per
rank, mirroring `router.allocate_compression_policy`'s rank table.

**The effort hint is what moves the local/API threshold** — that's the actual
mechanism, not a separate knob bolted next to it:
- `cheap` — any feasible local model wins, offloaded or not (free beats fast);
  API only if nothing local fits.
- `balanced` — local wins *only* at the `gpu_resident` tier (fits entirely in
  VRAM). A heavily offloaded model is slow enough that a cheap API call is the
  better trade, so it routes to the API and says so in the decision's `reason`.
- `effort` — straight to the API catalog without scoring local at all; local
  only if the API catalog is empty.

Within the API catalog the same hint picks the ranking rule: cheapest blended
cost / best capability-per-dollar / highest capability. An unmatched specialty
tag widens to the whole catalog with an explicit note rather than failing.

`PolicyDecision` is returned for every outcome including failure
(`kind="unavailable"`) rather than raising — same philosophy as
`SelectionResult.needs_install`. It carries the full `reason` string and the
raw `SelectionResult` it was based on, and is stored on `Employee.
model_decision` + logged to `activity_log` as `kind="model_policy"`, so "why
is this employee on this model?" stays answerable months later.

Failure modes are deliberately asymmetric: `mode="local"` **fails closed**
(`unavailable` + install hint) because an explicit local request should never
silently start spending money; `allow_api_fallback=True` opts in. `mode="auto"`
falls back to the API, since that's the whole point of auto. A policy that
can't route anything at `hire()` time is **logged and ignored** (the literal
`model_map` values are used) rather than fatal — a routing miss shouldn't
abort building an org chart halfway; it resurfaces as a real provider error on
first request.

`DEFAULT_LOCAL_BINDINGS` bridges selector.py's provider names (which describe
*where weights live*) to `Agent.change_api`'s transports: ollama → the Ollama
endpoint; lm-studio → `openai` + localhost:1234; hf/vllm → `openai` +
localhost:8000, with the decision's reason explicitly warning that raw HF
weights on disk are not an endpoint.

Per-task (not just per-employee) effort is real: `Company.run(task, effort=)`
calls `Company.reassign_model()`, which re-decides and swaps the endpoint via
`Agent.change_api()` — keeping conversation history, registered tools
(including every `delegate_to_*`), and usage counters. The hint applies to the
entry point only; delegates keep their own standing hints, since a manager
finding a task hard doesn't mean every intern touching it needs a frontier
model. Flagged honestly in the docstring: mid-conversation provider swaps on a
long tool-heavy history aren't hardened, the intended use is a fresh task.

**Resolved: open question #1 — effort does NOT feed `BudgetLedger.importance`.**
They answer different questions (how capable a model vs. how large a token
slice), and coupling them automatically would quietly redefine what
`importance` means in Phase 3's ledger and make budget reports unreadable.
Instead `policy.suggested_importance(effort)` is an explicit opt-in bridge the
caller passes into `hire(importance=...)` themselves. Explicit beats implicit
where the ledger is the thing standing between this library and a surprise
bill.

Also fixed while getting the suite green: `core.py`'s
`process_openai_custom_response` raised a raw `IndexError` on an empty
`choices` list instead of the clear `RuntimeError` the Gemini path already
produced (`test_full.py` was asserting the latter). Added `tests/run_all.py`,
a dependency-free whole-suite runner (skips `test_trial_own.py`, an
interactive REPL rather than a test).

**Phase 5 — one named-preset registry** (`presets.py`, `company.py`, `observability.py`)
`PresetRegistry` is the mechanism, instantiated four times — `SKILLS`,
`PERSONALITIES`, `PALETTES`, `ORG_TEMPLATES`. Same class, same API
(`get`/`register`/`resolve`/`names`/`copy`) for all four. That was the explicit
correction to honor: palettes are picked by name through the *same* mechanism
as skills, not a bespoke colour system inside `observability.py`, and org
templates are not a bespoke system inside `company.py`.

What committing to one mechanism actually bought:
- `resolve()` takes a **name or the object**, so `hire(skills=["python"])` and
  `hire(skills=[Skill(...)])` both work and a one-off preset needs no
  registration (passing an object deliberately does *not* register it).
- `names()` is where Phase 8's schema enums come from — one method, four kinds,
  via `PresetBundle.names()`/`describe()`.
- `PresetBundle.fork()` gives a Company private membership, so registering a
  custom skill can't leak into another Company in the same process (or another
  test). Presets are treated as immutable value objects, so a fork isolates
  membership only.
- `get()` raises listing what *is* available and never returns None — a typo'd
  skill name would otherwise silently produce an employee with no skills.
  `register()` refuses to shadow an existing name without `overwrite=True`.

`compose_system_instruction()` templates base → role line → personality →
skills → constraints, in a **fixed order**, because a system instruction that
reshuffles between runs makes prompt caching useless and behavior changes
impossible to attribute. Constraints are pooled across skills and deduplicated.
With no presets it returns `""`, so a pre-Phase-5 hire is byte-identical.

Skills carry a `specialty`/`effort` hint that feeds **Phase 4's policy** — but
only as a fallback; an explicit `hire(specialty=…)` always wins. That is what
makes the two phases one system rather than two.

Org templates (`OrgTemplate`/`RoleSpec`) are declarative — roles refer to each
other by key, and `validate()` catches dangling reporting lines, cycles, and
duplicate leads before anything is half-built. **Scaling is opt-in**:
`build_from_template(name, size=…)` takes the size as an argument and never
infers it from the task text, because inferring it means guessing at spend on
the user's behalf — the thing Phase 3 exists to prevent. Roles declare
`count_by_size`, so "large" multiplies worker tiers while leaving oversight
roles alone, and a role whose manager doesn't exist at a smaller size reports
to that manager's manager instead (so one template works at every size without
a rewritten small-only copy). Built-ins: `solo`, `small-coding-team` (the
roadmap's reference shape), `research-pod`, `writing-desk`.

Palettes: `dataviz` (the pre-Phase-5 colours, still the default, so every
existing `render_org_chart()` call renders identically), plus `grayscale`,
`ocean`, `ember`. **Only `dataviz` claims the colourblind-safe validation** —
the others are stylistic and say so, rather than implying validation by sitting
in the same list.

**Resolves open question #3 — `Team.reviewer` now actually reviews.** The
decision (it needed one, not just an implementation): `review_mode="critique"`
is the default — the lead drafts, the reviewer either signs off or lists what
must change, and a requested change goes back to the lead. Chosen because it is
the only option where the reviewer's opinion can change the delivered output,
which is what "reviewer" means outside this codebase. The alternatives are kept
as modes rather than discarded: `"append"` (reviewer independently answers, both
returned labelled — a second opinion, not a review) and `"off"` (exactly the
pre-Phase-5 lead-only behavior).

Two deliberate bounds on that loop, both cost-driven:
- `max_review_rounds` defaults to **1**, counting *send-backs*. Two agents can
  disagree forever and each round costs real tokens — an unbounded critique
  loop is precisely the hazard Phase 3 exists to prevent.
- There is always **one more review than revision**, so the final revision is
  looked at before it ships. Without that pass a revised draft would go out
  carrying a critique written about the previous version.
- If an objection still stands when the rounds run out, the work ships **with
  the outstanding critique appended** rather than the objection being silently
  dropped or an exception raised. The caller gets both and decides.

Flagged simplification: approval detection is a *convention* — the reviewer is
asked to reply exactly `APPROVED` and `_looks_approved()` checks the opening of
the reply (rejecting obvious negations). A structured tool call would be
stricter but would force every reviewer onto a tool-calling-capable model,
which the cheap local tiers this library targets often are not. A reviewer that
buries "APPROVED" mid-critique will be misread.

## Not built yet

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

## Open questions

1. ~~Should the effort hint feed `BudgetLedger.importance`?~~ **Resolved in
   Phase 4: no.** Kept as separate knobs, with `policy.suggested_importance()`
   as an explicit opt-in bridge. Reasoning in the Phase 4 entry above.
2. Ready-made color palettes + custom palette support for `render_org_chart`
   (deferred until Phase 5's named-preset registry exists) — addressed by
   Phase 5's palette registry.
3. ~~**`Team.reviewer` is structural only**~~ **Resolved in Phase 5** — see the
   Phase 5 entry for the decision and its bounds. Original note: — `Team.__init__` stores a reviewer
   and the class comment promises "at least one reviewer by default", but
   `Team.run()` just calls `self.lead.run(...)` and the reviewer is never
   invoked. Needs a *decision* on what review means (reviewer also gets the
   task and their output is appended? reviewer critiques the lead's output and
   can send it back?), not just an implementation. Folded into Phase 5, which
   owns team templates.
4. **"Plan-then-execute for complex tasks"** was in the original spec alongside
   stub-and-fill and skills, but had no home in Phases 4-8. Folded into Phase 6
   — stub-and-fill is itself a form of planning, and doing both in one module
   avoids two competing task-decomposition systems.
