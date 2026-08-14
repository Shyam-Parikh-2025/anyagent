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

**Phase 6 — task decomposition: stub-and-fill + plan-then-execute** (`delegation.py`)
Both strategies in one module, deliberately. They are the same idea at
different granularities — a more capable agent decides the *shape*, cheaper
agents fill it in — and splitting them would create two competing
decomposition systems, which is what the roadmap warned about when it folded
"plan-then-execute" (open question #4) into this phase.

*stub-and-fill*: the architect emits a module of signatures + docstrings +
type hints with `...` bodies; implementers fill one body each; results are
reassembled. **Reuses `ContextCompressor.code_to_stub()`** rather than writing
a second AST layer — it already produces exactly the signature+docstring+`...`
shape, so it normalizes the architect's output into the interface implementers
see (a chatty architect that half-implements something still hands over a clean
interface).

*plan-then-execute*: the planner emits a numbered plan, steps run in order, and
each step sees a **compressed digest** of earlier results
(`compress_tool_output`, the same machinery tool results already use) rather
than their full text — feeding every prior step's output into every later step
is how a five-step plan becomes a context-window failure. `synthesize=True`
gives the planner a final pass; set it False when the last step is already the
deliverable.

**Fixed by rank in v1**, as decided: `DEFAULT_ARCHITECT_RANKS` (most senior
first — architecting is the leverage point, a bad interface makes every filled
body wrong) and `DEFAULT_IMPLEMENTER_RANKS` (cheapest first — bodies are the
high-volume, low-leverage part, which is the economic point). Implementers get
stubs round-robin, deliberately *not* "best implementer per stub", because
ranking stubs by difficulty needs exactly the dynamic-decomposition judgment
this phase decided not to fake.

Failure handling is the bulk of the design, since both strategies are driven by
model output that will sometimes be malformed. Nothing raises for a
model-quality failure — everything lands on the result object and in the
activity log:
- unparseable architect module → `parse_error`, and **no implementer is
  called** (nobody pays for an unusable plan)
- implementer returns another stub, or renames the function → that stub is
  reported `unfilled` rather than silently accepted; a surviving `...` would
  produce a syntactically valid module that does nothing, the worst failure
  mode available here
- unfilled stubs stay **visible in the assembled module** under an
  `# UNFILLED (reason):` comment rather than being dropped, so the artifact
  says where it is incomplete
- architect returns no stubs → its answer is returned verbatim rather than
  inventing stubs to justify the strategy
- a plan that doesn't parse → the whole task runs as a single step with
  `degraded=True`, rather than failing
- a step that raises → recorded on that step; the remaining steps still run

`MAX_STUBS` (20) and `MAX_PLAN_STEPS` (12) are runaway guards in the same
spirit as Phase 3's budget ceiling and Phase 5's `max_review_rounds`: each stub
and each step is a real model call, and a model asked to decompose a vague task
will happily produce forty.

`Company.run_structured(task, strategy=…)` is the entry point; the structured
strategies return result *objects*, not strings, because "did every step
succeed?" and "which stubs came back unfilled?" are the questions a caller
actually has. `effort`/`specialty` work as in `run()`, re-routing the
architect/planner through the Phase 4 policy.

Stated limits: nothing here executes or tests the generated code —
assembly is checked with `compile()` for **syntax only**, so a confidently
wrong body assembles cleanly. An implementer that keeps the function name but
changes the parameter list is accepted as-is. Plan-step parsing is a
numbered-list regex over prose; models are good at numbered lists and bad at
guarantees, which is why the no-match path degrades instead of erroring.

**Phase 7 — log compaction** (`compaction.py`, `company.py`)
The "keep broad concepts, drop specifics" pass over `activity_log` /
`tool_call_log`, run automatically at the end of every completed top-level task
(`run()` / `run_structured()`) — a task boundary is exactly when the specifics
of *how* it got done stop mattering and the shape is what you keep.

**Reuses `compressor.py` twice over**, as instructed — not a second compression
system:
- *the code*: `compress_tool_output` shrinks result previews,
  `_dedupe_repeated_lines` collapses repetition before summarizing,
  `token_estimate` measures what was saved (so the numbers are comparable with
  what `CompressionPolicy` budgets against elsewhere).
- *the pattern*: `LogCompactionPolicy` is deliberately shaped like
  `HistoryCompactionPolicy` — same three modes (`off`/`algorithmic`/`agent`),
  same untouched `keep_recent` tail, same trigger threshold, same
  summarizer-optional contract, same "a typo'd mode degrades to the free/safe
  path", same "agent mode falls back to algorithmic if the summarizer fails, so
  choosing it can never break a run". Anyone who has configured history
  compaction already knows how to configure this.

Algorithmic mode collapses old activity events into one rollup carrying counts
per kind, the employees involved, and the time span — every individual task
string dropped. Tool calls group by `(employee, tool_name)` into one row with
call count, total/mean duration and one compressed example preview, which is
the useful residue of a thousand identical `delegate_to_worker` calls.

**The one genuinely new decision: what compaction may not touch.** Compaction
is lossy and irreversible, and these logs are the only record of where the
money went. `ALWAYS_KEEP_KINDS` protects escalations, escalation decisions,
emergency-reserve draws and budget reallocations at *every* mode regardless of
age; tool-call entries that recorded an **error** are likewise never collapsed,
since an error is the specific thing you go back to a log to find. The rule for
what belongs on that list: if losing the event would make "who approved this
spend?" or "what went wrong?" unanswerable after the fact, it stays.

Compaction assigns **in place** (`log[:] = …`) rather than rebinding, because
Phase 2's `EventLog` deliberately wraps the company's list live — rebinding
would leave any `EventLog` a caller already holds pointed at the stale list.
The policy's own methods never mutate the list passed to them; only
`compact_company()` writes.

**Phase 8 (text/schema mode) — `set_company_up(mode="text")`** (`builder.py`)
The AI-callable company builder. The design constraint driving everything: the
*calling model* must be able to see its options and fill them in correctly.

The schema is **generated by `core.ToolRegistry`** from `set_up_company`'s type
hints and docstring — not a hand-written parallel format, as instructed. The
payoff is concrete: the tool an agent calls and the schema it reads come from
the same function, so they cannot drift (a test asserts they match). Because
`ToolRegistry`'s generator maps flat scalars, `set_up_company` takes **flat
scalar parameters** plus one `employees_json` escape hatch for org shapes no
template covers — a nested-object schema would have meant writing a second
schema generator, the exact thing this phase was told not to do.

`company_options()` pulls the valid names for every preset kind straight from
Phase 5's `PresetBundle.names()`, and `company_setup_schema()` injects them as
`enum`s and into the parameter descriptions. **This is why text mode had to
wait for Phase 5**: a schema saying `template: str` is useless to a model,
while one carrying the actual template, skill and personality names is fillable
in one shot. It reads the registries live, so a forked bundle's custom presets
show up too.

`CompanySpec`/`EmployeeSpec` are the serializable format (the same one `gui.py`
saves and loads, so either half of Phase 8 can consume the other's output).
`validate()` **returns** a list of problems rather than raising on the first —
the caller is often a model filling in a schema, and "here are the four things
wrong" beats a traceback. Every message lists what *is* allowed. It catches
unknown fields, unknown ranks/skills/personalities/templates/palettes,
duplicate names, dangling `reports_to` (including against names a template will
create, since a spec may extend a template), and reporting cycles.

Three safety decisions, all in the same direction — an AI must not be able to
spend money by accident:
- Building **runs nothing**. `set_up_company` returns the org, never a result.
  A tool that could build *and* start a company is one hallucinated argument
  away from a bill.
- With no `model_map`, every rank defaults to **local Ollama** — the only
  provider needing no API key — so an unconfigured build cannot spend.
- With no `on_escalation`, the default handler **declines**, matching the
  0-means-always-ask reserves from Phases 0-3: "no human attached" must mean
  stop, never approve-yourself.

Runtime concerns (`model_map`, `on_escalation`, `model_policy`, `presets`,
`log_compaction`) are arguments to `build_company`, deliberately *not* spec
fields: they're callables and policy objects, they don't serialize, and putting
them in the spec would let an AI-authored JSON blob name a provider and a key.
The spec describes the org; the caller supplies the runtime. Every default
taken is reported in `BuildResult.warnings` rather than applied silently.

**Phase 8 (GUI mode) — the node-graph company editor** (`gui.py`, `gui_assets.py`)
The half the roadmap called "meaningfully bigger than everything built so far
combined", and it is: a small local web app, not a Python function.
`http.server` serves one self-contained page plus a small JSON API, opened with
`webbrowser.open()`. Vanilla JS and SVG — no framework, no CDN, no build step,
so the zero-dependency rule holds on both sides of the wire.

**The division of labour is the design.** The browser owns *interaction*
(dragging, rubber-band and shift multi-select, the right-click bulk actions);
Python owns *truth* (which preset names exist, whether a design is valid, what
a template expands to). The page hardcodes no rank, skill, template or palette
— it fetches them from Phase 5's registries at load, so adding a skill in
`presets.py` makes it appear in the editor with no front-end change. A test
asserts the JS contains none of those names.

Implemented: click-to-connect (click the report, then the manager), shift-click
and rubber-band multi-select, right-click bulk actions including "connect all N
to one manager", "add a skill to all", duplicate and delete, a left panel for
per-employee rank / manager / skills / personality / effort / policy mode /
provider / model / importance, a left menu for templates, palettes and review
settings, live validation against the server, auto-layout, and drag-to-move
with positions persisted in `CompanySpec.layout` (which `build_company()`
ignores entirely — it exists so reopening a design doesn't scramble the
picture).

Guards, none of them incidental:
- The server binds **127.0.0.1 only** and every request must carry a random
  per-session token issued at launch, so another local process — or a page in
  another tab — can't drive the editor by guessing the port. Not an auth
  system; the minimum that keeps an unauthenticated local port from being a
  foot-gun.
- **API keys are not editable and never enter a saved design**, and the page
  says so where a user would look for the field. They come from the
  `model_map` passed to `build_company()`, so a design file can be shared or
  committed safely — the same rule text mode set.
- Loading a template **expands it into concrete employees** rather than storing
  a template reference: once nodes are dragged around, "this is the
  small-coding-team template" stops being true, and a spec claiming a template
  it no longer matches is worse than one that just lists its employees.
- A build that fails validation **leaves the editor open** instead of returning
  a broken company; only a successful build releases the waiting caller.
- The GUI builds an org and **runs nothing** — identical to text mode, for
  identical reasons.

Testing, honestly split: `tests/test_gui.py` (30 checks, in the suite) drives
every route over real HTTP and statically checks the page — one script block,
no external loads, balanced structure, no hardcoded vocabulary, every fetch
carrying the token, every called route actually served, user text escaped
before insertion. `tests/browser_smoke.py` (**not** in the suite, and its
filename deliberately doesn't match the runner's pattern) drives a headless
Chromium through the DOM behaviour that genuinely can't be covered otherwise —
template load, add, rename, drag, connect mode, the right-click menu, the
cycle guard, check and build — failing on any console error. It needs
playwright, which llmadapt does not depend on and shouldn't acquire for one
module. It was run during this phase and passes with no console errors.

## Not built yet

Nothing from the original Phase 0-8 plan remains. What's left is the
deliberately-deferred work called out in the entries above:

- **Budget governance v2** — a real global water-filling optimizer instead of
  Phase 3's fixed rank shares + ask-your-manager reallocation. Flagged since
  Phase 3 as worth targeting once the simpler version has proven the escalation
  path end to end. It has now run through five more phases without needing to
  change, which is some evidence the simple version is load-bearing enough.
- **Dynamic decomposition depth** (Phase 6's stretch goal) — deciding how far
  to break a task down from the task itself, rather than fixing architect and
  implementer tiers by rank. Still no clean formula; still research-adjacent.
- **Verifying generated code** — Phase 6 checks assembled modules with
  `compile()` for syntax only. Actually running or testing what the
  implementers wrote is the obvious next increment, and the one that would
  make stub-and-fill trustworthy rather than merely useful.
- **Mid-generation budget enforcement** — Phase 3's gate is pre-flight, because
  llmadapt cannot interrupt a request already in flight. Streaming responses
  would make a real mid-response cutoff possible.
- **A `company/` package split** — `company.py` has been carrying a note since
  Phase 0 that it should become a package once enough neighbours landed. Six
  modules now orbit it (`budget`, `observability`, `policy`, `presets`,
  `delegation`, `compaction`, `builder`, `gui`), so the split is due; it was
  left alone during this run because moving files mid-phase would have made
  every phase's diff unreadable.

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
4. ~~**"Plan-then-execute for complex tasks"**~~ **Resolved in Phase 6** — built
   alongside stub-and-fill in `delegation.py`. Original note: was in the original spec alongside
   stub-and-fill and skills, but had no home in Phases 4-8. Folded into Phase 6
   — stub-and-fill is itself a form of planning, and doing both in one module
   avoids two competing task-decomposition systems.
