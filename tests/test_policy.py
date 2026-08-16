# Phase 4 tests - policy.py (local-vs-API model policy) and its company.py
# wiring. Same style as the other test_*.py files: plain asserts + prints, no
# pytest, no network. Layers, bottom-up:
#   (1) effort-hint normalization + the explicit importance bridge
#   (2) ApiModelSpec / ApiModelCatalog economics and ranking
#   (3) ModelPolicy.decide() across modes and effort levels, with a fake
#       local catalog + a synthetic BenchmarkResult so no real hardware is
#       touched and the local/API threshold is exercised deterministically
#   (4) Company integration - hire() routing, per-task effort on run(),
#       reassign_model(), and the activity-log trail
import json
import time

from llmadapt.benchmark import BenchmarkResult
from llmadapt.company import Company, EscalationDecision
from llmadapt.policy import (
    DEFAULT_RANK_EFFORT,
    EFFORT_BALANCED,
    EFFORT_CHEAP,
    EFFORT_EFFORT,
    ApiModelCatalog,
    ApiModelSpec,
    ModelPolicy,
    normalize_effort,
    suggested_importance,
)
from llmadapt.presets import SKILLS
from llmadapt.router import RoleRank
from llmadapt.selector import LocalModelCandidate, ModelCatalog


class FakeResponder:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def __call__(self, payload, headers):
        self.calls.append(json.loads(json.dumps(payload)))
        return self.responses.pop(0)


def text_response(text):
    return {"content": [{"type": "text", "text": text}]}


def fake_benchmark(vram_gb=24.0, ram_gb=64.0):
    """A synthetic BenchmarkResult - the local selector only reads a handful
    of these fields, but the dataclass requires all of them."""
    return BenchmarkResult(
        os="linux", cpu_name="test-cpu", cpu_cores=8, cpu_gflops=100.0,
        cpu_gflops_measured_with="pure_python", system_ram_gb=ram_gb,
        gpu_name="test-gpu", gpu_vram_gb=vram_gb, gpu_pcie_gen=4,
        gpu_pcie_link_width=16, pcie_bandwidth_gbps=16.0, pcie_bandwidth_measured=True,
        gpu_tflops=50.0, gpu_tflops_measured=False, fingerprint="test", timestamp=time.time(),
    )


def local_catalog(*candidates):
    """A ModelCatalog scanning no providers (providers=() disables discovery),
    holding exactly the candidates given - so these tests never depend on what
    happens to be installed on the machine running them."""
    catalog = ModelCatalog(providers=())
    for candidate in candidates:
        catalog.register(
            name=candidate.name, provider=candidate.provider,
            size_gb=candidate.size_gb, path=candidate.path, installed=candidate.installed,
        )
    return catalog


def small_local(name="llama3.1:8b", size_gb=5.0):
    return LocalModelCandidate(name=name, provider="ollama", size_gb=size_gb,
                               path="/fake/path", installed=True)


def huge_local(name="llama3.1:70b", size_gb=40.0):
    return LocalModelCandidate(name=name, provider="ollama", size_gb=size_gb,
                               path="/fake/path", installed=True)


CHEAP_API = ApiModelSpec(name="tiny-api", provider="openai", cost_per_1k_input=0.0001,
                         cost_per_1k_output=0.0004, capability=0.50, specialties=("code",))
MID_API = ApiModelSpec(name="mid-api", provider="openai", cost_per_1k_input=0.001,
                       cost_per_1k_output=0.004, capability=0.75, specialties=("code", "reasoning"))
STRONG_API = ApiModelSpec(name="strong-api", provider="anthropic", cost_per_1k_input=0.003,
                          cost_per_1k_output=0.015, capability=0.95, specialties=("reasoning", "vision"))


def api_catalog(*specs):
    return ApiModelCatalog(specs=specs, include_defaults=False)


# ---------------------------------------------------------------------------
# Layer 1: effort hints
# ---------------------------------------------------------------------------

assert normalize_effort("cheap") == EFFORT_CHEAP
assert normalize_effort("keep it cheap") == EFFORT_CHEAP
assert normalize_effort("needs effort") == EFFORT_EFFORT
assert normalize_effort("  HIGH  ") == EFFORT_EFFORT
assert normalize_effort("balanced") == EFFORT_BALANCED
print("PASS 1: effort aliases normalize to the three canonical levels")

# Unknown strings fall back to the rank default rather than raising - this
# value can arrive from an LLM filling in a tool schema (Phase 8).
assert normalize_effort("wibble", RoleRank.JUNIOR) == DEFAULT_RANK_EFFORT[RoleRank.JUNIOR]
assert normalize_effort(None, RoleRank.C_SUITE) == EFFORT_EFFORT
assert normalize_effort(None, RoleRank.INTERN) == EFFORT_CHEAP
assert normalize_effort(None, None) == EFFORT_BALANCED
assert normalize_effort(None, "NOT_A_RANK") == EFFORT_BALANCED
print("PASS 2: unknown/missing effort falls back to rank default, then balanced")

# The importance bridge exists but is opt-in only (roadmap open question #1).
assert suggested_importance("cheap") < suggested_importance("balanced") < suggested_importance("needs effort")
assert 0.0 <= suggested_importance("cheap") <= 1.0
assert 0.0 <= suggested_importance("needs effort") <= 1.0
print("PASS 3: suggested_importance is a monotonic, opt-in bridge to BudgetLedger")


# ---------------------------------------------------------------------------
# Layer 2: API catalog economics
# ---------------------------------------------------------------------------

spec = ApiModelSpec(name="x", provider="openai", cost_per_1k_input=0.001, cost_per_1k_output=0.005)
# default output_ratio 0.75 -> 0.25*0.001 + 0.75*0.005
assert abs(spec.blended_cost_per_1k() - (0.00025 + 0.00375)) < 1e-12
assert abs(spec.blended_cost_per_1k(output_ratio=0.0) - 0.001) < 1e-12
assert abs(spec.blended_cost_per_1k(output_ratio=1.0) - 0.005) < 1e-12
print("PASS 4: blended_cost_per_1k mixes input/output prices by the assumed ratio")

catalog = api_catalog(CHEAP_API, MID_API, STRONG_API)
assert len(catalog) == 3
assert catalog.find("mid-api") is MID_API
assert catalog.find("mid-api", "anthropic") is None
assert {s.name for s in catalog.matching("code")} == {"tiny-api", "mid-api"}
assert catalog.matching("nonexistent-tag") == []
print("PASS 5: ApiModelCatalog find/matching filter by name, provider, and specialty")

# register() replaces same provider+name rather than duplicating, so a user
# correcting a seed default's price gets one row, not two.
override = ApiModelSpec(name="mid-api", provider="openai", cost_per_1k_input=0.5,
                        cost_per_1k_output=0.5, capability=0.75)
catalog.register(override)
assert len(catalog) == 3
assert catalog.find("mid-api").cost_per_1k_input == 0.5
catalog.unregister("mid-api")
assert len(catalog) == 2 and catalog.find("mid-api") is None
print("PASS 6: register() overrides in place; unregister() removes")

# The seed defaults exist so a bare ModelPolicy() is usable, and every entry
# is honestly flagged as possibly-stale config rather than live pricing.
seeded = ApiModelCatalog()
assert len(seeded) > 0
assert all(s.note for s in seeded.all()), "seed specs should carry a staleness note"
print("PASS 7: DEFAULT_API_CATALOG is populated and every entry is flagged as a seed default")


# ---------------------------------------------------------------------------
# Layer 3: ModelPolicy.decide()
# ---------------------------------------------------------------------------

policy = ModelPolicy(api_catalog=api_catalog(CHEAP_API, MID_API, STRONG_API), allow_local=False)

assert [s.name for s in policy.rank_api_models(EFFORT_CHEAP)][0] == "tiny-api"
assert [s.name for s in policy.rank_api_models(EFFORT_EFFORT)][0] == "strong-api"
# balanced maximizes capability per dollar: tiny-api is 0.50 / 0.000325
# ~= 1538, mid ~= 0.75/0.00325 ~= 231, strong ~= 0.95/0.0117 ~= 81.
assert [s.name for s in policy.rank_api_models(EFFORT_BALANCED)][0] == "tiny-api"
print("PASS 8: API ranking rule changes with the effort hint (cheapest / best value / strongest)")

decision = policy.decide(rank=RoleRank.SENIOR, effort="needs effort", specialty="reasoning")
assert decision.kind == "api" and decision.model == "strong-api"
assert decision.provider == "anthropic"
assert decision.estimated_cost_per_1k > 0
assert "effort" in decision.reason
print("PASS 9: specialty + effort together select the strongest matching API model")

# A specialty nothing carries widens to the full catalog with an explicit note
# instead of returning nothing usable.
decision = policy.decide(rank=RoleRank.SENIOR, effort="cheap", specialty="underwater-basket-weaving")
assert decision.kind == "api" and decision.model == "tiny-api"
assert "underwater-basket-weaving" in decision.reason and "whole catalog" in decision.reason
print("PASS 10: an unmatched specialty widens the search and says so in the reason")

# An empty catalog with no local option is 'unavailable', not a crash.
empty = ModelPolicy(api_catalog=api_catalog(), allow_local=False)
decision = empty.decide(rank=RoleRank.JUNIOR)
assert decision.kind == "unavailable" and not decision.ok
print("PASS 11: no routable model returns kind='unavailable' rather than raising")

# --- the local/API threshold ---
# A small model that fits entirely in VRAM ('gpu_resident').
fits_policy = ModelPolicy(
    api_catalog=api_catalog(CHEAP_API, MID_API, STRONG_API),
    local_catalog=local_catalog(small_local()),
    benchmark=fake_benchmark(vram_gb=24.0),
)
decision = fits_policy.decide(rank=RoleRank.JUNIOR, effort="cheap")
assert decision.kind == "local" and decision.model == "llama3.1:8b"
assert decision.provider == "ollama" and decision.estimated_cost_per_1k == 0.0
print("PASS 12: effort=cheap takes a feasible local model over any API model")

decision = fits_policy.decide(rank=RoleRank.SENIOR, effort="balanced")
assert decision.kind == "local", decision.reason
assert "VRAM" in decision.reason
print("PASS 13: effort=balanced accepts a local model that is fully GPU-resident")

decision = fits_policy.decide(rank=RoleRank.C_SUITE, effort="needs effort")
assert decision.kind == "api" and decision.model == "strong-api"
assert "before considering local" in decision.reason
print("PASS 14: effort=effort goes straight to the API catalog without scoring local")

# Now a model too big for VRAM but fitting in VRAM+RAM -> 'offloaded' tier.
offload_policy = ModelPolicy(
    api_catalog=api_catalog(CHEAP_API, MID_API, STRONG_API),
    local_catalog=local_catalog(huge_local(size_gb=20.0)),
    benchmark=fake_benchmark(vram_gb=12.0, ram_gb=64.0),
)
decision = offload_policy.decide(rank=RoleRank.JUNIOR, effort="cheap")
assert decision.kind == "local", decision.reason
print("PASS 15: effort=cheap still takes an offloaded local model - free beats fast")

decision = offload_policy.decide(rank=RoleRank.SENIOR, effort="balanced")
assert decision.kind == "api", decision.reason
assert decision.considered_local and "not fully" in decision.reason
print("PASS 16: effort=balanced rejects an offloaded local model in favor of a cheap API call")

# ...unless there is no API catalog to fall back to, in which case take it and say so.
no_api = ModelPolicy(
    api_catalog=api_catalog(),
    local_catalog=local_catalog(huge_local(size_gb=20.0)),
    benchmark=fake_benchmark(vram_gb=12.0, ram_gb=64.0),
)
decision = no_api.decide(rank=RoleRank.SENIOR, effort="balanced")
assert decision.kind == "local" and "no API models are configured" in decision.reason
decision = no_api.decide(rank=RoleRank.C_SUITE, effort="needs effort")
assert decision.kind == "local" and "fell back to local" in decision.reason
print("PASS 17: with an empty API catalog, local is used and the fallback is stated explicitly")

# mode='local' with nothing usable does NOT silently start spending money.
nothing_fits = ModelPolicy(
    api_catalog=api_catalog(CHEAP_API),
    local_catalog=local_catalog(huge_local(size_gb=500.0)),
    benchmark=fake_benchmark(vram_gb=8.0, ram_gb=16.0),
)
decision = nothing_fits.decide(rank=RoleRank.JUNIOR, mode="local")
assert decision.kind == "unavailable", decision.reason
assert decision.considered_local
# ...but the caller can opt into the fallback explicitly.
decision = nothing_fits.decide(rank=RoleRank.JUNIOR, mode="local", allow_api_fallback=True)
assert decision.kind == "api" and decision.model == "tiny-api"
print("PASS 18: mode='local' fails closed by default; allow_api_fallback=True opts into API")

# mode='auto' with no usable local model DOES fall back to the API (the whole
# point of auto), and records that it looked.
decision = nothing_fits.decide(rank=RoleRank.JUNIOR, mode="auto", effort="cheap")
assert decision.kind == "api" and decision.considered_local
print("PASS 19: mode='auto' falls back to the API when nothing local fits")

# mode='api' never touches the local path, even with a local model available.
decision = fits_policy.decide(rank=RoleRank.JUNIOR, effort="cheap", mode="api")
assert decision.kind == "api" and not decision.considered_local
print("PASS 20: mode='api' skips local scoring entirely")

# A pinned model that lives in the API catalog wins outright.
decision = fits_policy.decide(rank=RoleRank.JUNIOR, effort="cheap", requested_model="strong-api")
assert decision.kind == "api" and decision.model == "strong-api"
assert "requested explicitly" in decision.reason
print("PASS 21: an explicitly requested catalog model is honored over the effort heuristic")

# LM Studio binds to the OpenAI transport with an explicit local base_url.
lms = LocalModelCandidate(name="mistral-7b", provider="lm-studio", size_gb=4.0,
                          path="/fake", installed=True)
lms_policy = ModelPolicy(
    api_catalog=api_catalog(), local_catalog=local_catalog(lms), benchmark=fake_benchmark(),
)
decision = lms_policy.decide(rank=RoleRank.JUNIOR, effort="cheap")
assert decision.kind == "local" and decision.provider == "openai"
assert decision.base_url and "1234" in decision.base_url
print("PASS 22: a non-Ollama local provider binds to its OpenAI-compatible local endpoint")

try:
    fits_policy.decide(rank=RoleRank.JUNIOR, mode="sideways")
    assert False, "expected ValueError for an unknown mode"
except ValueError as e:
    assert "sideways" in str(e)
print("PASS 23: an unknown mode raises a clear ValueError")


# ---------------------------------------------------------------------------
# Layer 4: Company integration
# ---------------------------------------------------------------------------

MODEL_MAP = {
    RoleRank.C_SUITE: {"provider": "anthropic", "model": "claude-x", "api_key": "k", "mode": "auto"},
    RoleRank.JUNIOR: {"provider": "anthropic", "model": None, "api_key": "k", "mode": "auto"},
    RoleRank.INTERN: {"provider": "anthropic", "model": "pinned-model", "api_key": "k"},
}


def make_company(policy=None, **kwargs):
    def on_escalation(event):
        return EscalationDecision(approve=False)

    return Company(name="Policy Co", model_map=MODEL_MAP, on_escalation=on_escalation,
                   model_policy=policy, **kwargs)


# Without a policy, nothing about hiring changes (pre-Phase-4 behavior intact).
plain = make_company()
emp = plain.hire("NoPolicy", RoleRank.INTERN)
assert emp.agent.model == "pinned-model"
assert emp.model_decision is None
print("PASS 24: with no ModelPolicy attached, model_map values are still taken literally")

company_policy = ModelPolicy(
    api_catalog=api_catalog(CHEAP_API, MID_API, STRONG_API),
    local_catalog=local_catalog(small_local()),
    benchmark=fake_benchmark(vram_gb=24.0),
)
co = make_company(company_policy)

# JUNIOR: mode=auto, no explicit model -> rank default effort 'cheap' -> local.
junior = co.hire("Jun", RoleRank.JUNIOR)
assert junior.model_decision is not None and junior.model_decision.kind == "local"
assert junior.agent.model == "llama3.1:8b" and junior.agent.provider == "ollama"
print("PASS 25: hire() routes a JUNIOR through the policy to a local model by rank default")

# C_SUITE: mode=auto but provider AND model are both explicit -> policy skipped.
boss = co.hire("Boss", RoleRank.C_SUITE)
assert boss.model_decision is None and boss.agent.model == "claude-x"
print("PASS 26: an explicit provider+model pair skips the policy - it fills gaps, not overrules")

# INTERN has no mode at all -> policy not consulted.
intern = co.hire("Intern", RoleRank.INTERN)
assert intern.model_decision is None and intern.agent.model == "pinned-model"
print("PASS 27: a rank with no 'mode' is untouched by the policy")

# An explicit effort at hire time overrides the rank default.
eager = co.hire("Eager", RoleRank.JUNIOR, effort="needs effort", specialty="reasoning")
assert eager.model_decision.kind == "api" and eager.agent.model == "strong-api"
assert eager.agent.provider == "anthropic"
assert eager.effort == "needs effort" and eager.specialty == "reasoning"
print("PASS 28: hire(effort=...) overrides the rank default and reaches the API catalog")

# effort does NOT quietly change importance (roadmap open question #1).
assert eager.importance == 0.5, "effort must not silently redefine BudgetLedger importance"
explicit = co.hire("Explicit", RoleRank.JUNIOR, effort="needs effort",
                   importance=suggested_importance("needs effort"))
assert explicit.importance > 0.5
print("PASS 29: effort leaves importance alone unless suggested_importance() is passed explicitly")

# Every routing decision leaves a trail in the activity log.
policy_events = co.activity().by_kind("model_policy")
assert len(policy_events) >= 3
assert all(e.get("reason") for e in policy_events)
assert any(e["employee"] == "Jun" and e["decision"] == "local" for e in policy_events)
print("PASS 30: each policy decision is recorded in the activity log with its reason")

# reassign_model() swaps the endpoint in place, keeping tools and history.
worker = co.hire("Worker", RoleRank.JUNIOR)
assert worker.agent.provider == "ollama"
worker.agent.add_tool(lambda note: "ok")
tools_before = set(worker.agent.tool_registry.schemas)
worker.agent.conversation.add_user_msg("earlier turn")
decision = co.reassign_model(worker, effort="needs effort")
assert decision.kind == "api" and worker.agent.model == "strong-api"
assert worker.agent.provider == "anthropic"
assert set(worker.agent.tool_registry.schemas) == tools_before
assert any(m.get("content") == "earlier turn" for m in worker.agent.conversation.history)
assert worker.effort == "needs effort"
print("PASS 31: reassign_model() swaps the endpoint while keeping tools and conversation history")

# Per-task effort on run(): the entry point is re-routed for this call only.
runner = make_company(company_policy)
lead = runner.hire("Lead", RoleRank.JUNIOR)
assert lead.agent.provider == "ollama"
lead.agent._send_request = FakeResponder([text_response("done")])
result = runner.run("a hard one", effort="needs effort")
assert result == "done"
assert lead.agent.model == "strong-api", lead.agent.model
print("PASS 32: run(effort=...) re-routes the entry point for that task")

# A no-policy company ignores the per-task hint instead of failing.
plain2 = make_company()
worker2 = plain2.hire("Solo", RoleRank.INTERN)
worker2.agent._send_request = FakeResponder([text_response("fine")])
assert plain2.run("something", effort="needs effort") == "fine"
print("PASS 33: run(effort=...) is a harmless no-op when no ModelPolicy is attached")

# A policy that can't route anything is logged, not fatal - the literal
# model_map values are used and the miss shows up in the activity log.
dead_policy = ModelPolicy(api_catalog=api_catalog(), allow_local=False)
dead_co = make_company(dead_policy)
survivor = dead_co.hire("Survivor", RoleRank.JUNIOR)
assert survivor.model_decision.kind == "unavailable"
assert survivor.agent.provider == "anthropic"  # fell back to the model_map values
assert any(e["decision"] == "unavailable" for e in dead_co.activity().by_kind("model_policy"))
print("PASS 34: an unroutable policy decision is logged and falls back, not fatal at hire time")

# ---------------------------------------------------------------------------
# Typo handling, and the company-wide default routing mode
# ---------------------------------------------------------------------------
# An unrecognized effort hint still falls back rather than raising (Phase 4's
# decision - this value comes from an LLM filling a schema), but it no longer
# does so silently: a one-character typo used to quietly return the LOWEST
# tier with nothing anywhere saying why.

typo_policy = ModelPolicy(api_catalog=api_catalog())
resolved, note = typo_policy.resolve_effort_explained("balnced", RoleRank.JUNIOR)
assert resolved == DEFAULT_RANK_EFFORT[RoleRank.JUNIOR], resolved
assert "not recognized" in note and "'balanced'" in note, note
clean, clean_note = typo_policy.resolve_effort_explained("balanced", RoleRank.JUNIOR)
assert clean == EFFORT_BALANCED and clean_note == ""
print("PASS 35: an unrecognized effort still falls back, but says so and suggests the real name")

typo_decision = typo_policy.decide(rank=RoleRank.JUNIOR, effort="balnced")
assert "not recognized" in typo_decision.reason, typo_decision.reason
assert typo_decision.effort == DEFAULT_RANK_EFFORT[RoleRank.JUNIOR]
assert "not recognized" not in typo_policy.decide(rank=RoleRank.JUNIOR, effort="cheap").reason
print("PASS 36: the note reaches PolicyDecision.reason, so the log explains the model choice")

try:
    SKILLS.get("pyhton")
    assert False, "expected a KeyError"
except KeyError as e:
    assert "did you mean 'python'?" in str(e), str(e)
print("PASS 37: a misspelled preset name suggests the nearest real one too")

# default_policy_mode: what was missing was saying "this company is local" once.
# A rank whose model_map entry names neither a mode nor a full provider+model,
# so nothing narrower than the company default has an opinion.
UNOPINIONATED_MAP = {
    RoleRank.SENIOR: {"provider": "anthropic", "api_key": "k"},
    RoleRank.JUNIOR: {"provider": "anthropic", "api_key": "k", "mode": "auto"},
}
mode_policy = ModelPolicy(api_catalog=api_catalog(), allow_local=False)
local_co = Company(name="Local Co", model_map=UNOPINIONATED_MAP,
                   on_escalation=lambda e: EscalationDecision(approve=False),
                   model_policy=mode_policy, default_policy_mode="local")
inherited = local_co.hire("Inheritor", RoleRank.SENIOR)
assert local_co.resolve_policy_mode(None, None) == "local"
assert any(e["mode"] == "local" for e in local_co.activity().by_kind("model_policy")), \
    "the hire should have been routed with the company default"
# The rank that DOES name a mode keeps it - the default is a fallback, not an override.
local_co.hire("Opinionated", RoleRank.JUNIOR)
assert [e["mode"] for e in local_co.activity().by_kind("model_policy")] == ["local", "auto"]
print("PASS 38: hires inherit default_policy_mode when nothing narrower names one")

assert local_co.resolve_policy_mode("api", None) == "api"
assert local_co.resolve_policy_mode(None, "auto") == "auto"
print("PASS 39: hire(mode=) beats model_map[rank]['mode'] beats the company default")

# A per-task re-route must not be able to cross the line the company drew -
# reassign_model used to default to "auto" outright.
local_co.reassign_model(inherited, effort="cheap")
modes = [e["mode"] for e in local_co.activity().by_kind("model_policy")]
assert modes[-1] == "local", modes
print("PASS 40: reassign_model inherits the company default instead of forcing 'auto'")

try:
    Company(name="Bad", model_map=UNOPINIONATED_MAP,
            on_escalation=lambda e: EscalationDecision(approve=False),
            model_policy=mode_policy, default_policy_mode="loclal")
    assert False, "expected a ValueError"
except ValueError as e:
    assert "did you mean 'local'?" in str(e), str(e)
print("PASS 41: a bad company-wide default raises at construction rather than silently routing")


print("\nAll Phase 4 model-policy checks passed.")
