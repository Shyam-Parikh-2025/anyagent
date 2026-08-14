# Phase 8 (text/schema mode) tests - builder.py. Plain asserts + prints, no
# pytest, no network. Layers, bottom-up:
#   (1) CompanySpec round-tripping and validation (the part an AI gets wrong)
#   (2) build_company - templates, hand-written employees, and the safe defaults
#   (3) the ToolRegistry-generated schema and its preset enrichment
#   (4) set_up_company as an actual tool an Agent can call
import json

from llmadapt.builder import (
    DEFAULT_MODEL_MAP,
    BuildResult,
    CompanySpec,
    EmployeeSpec,
    build_company,
    company_options,
    company_setup_schema,
    default_on_escalation,
    register_company_builder,
    set_company_up,
    set_up_company,
)
from llmadapt.company import Company, EscalationEvent
from llmadapt.core import Agent, ToolRegistry
from llmadapt.presets import ORG_TEMPLATES, TASK_SIZES, default_bundle
from llmadapt.router import RoleRank


# ---------------------------------------------------------------------------
# Layer 1: the spec
# ---------------------------------------------------------------------------

spec = CompanySpec(
    name="ACME",
    employees=[
        EmployeeSpec(name="Grace", rank=RoleRank.C_SUITE),
        EmployeeSpec(name="Ada", rank=RoleRank.SENIOR, reports_to="Grace", skills=["python"]),
    ],
)
assert spec.validate() == []
round_tripped = CompanySpec.from_json(spec.to_json())
assert round_tripped.to_dict() == spec.to_dict()
print("PASS 1: a CompanySpec round-trips through JSON unchanged")

bad = CompanySpec(name="", template="no-such-template", size="enormous", palette="chartreuse",
                  review_mode="vibes")
problems = bad.validate()
assert len(problems) >= 5
joined = " ".join(problems)
for expected in ("name", "no-such-template", "enormous", "chartreuse", "vibes"):
    assert expected in joined, expected
# every message should say what IS allowed, not just what isn't
assert "Available" in joined or "one of" in joined
print("PASS 2: validate() reports every problem at once, each listing valid options")

dup = CompanySpec(name="X", employees=[EmployeeSpec(name="A"), EmployeeSpec(name="A")])
assert any("duplicate" in p for p in dup.validate())
print("PASS 3: duplicate employee names are caught before hiring raises")

ghost = CompanySpec(name="X", employees=[EmployeeSpec(name="A", reports_to="Nobody")])
assert any("Nobody" in p for p in ghost.validate())
print("PASS 4: a reports_to naming someone outside the company is caught")

cycle = CompanySpec(name="X", employees=[
    EmployeeSpec(name="A", reports_to="B"), EmployeeSpec(name="B", reports_to="A")])
assert any("cycle" in p for p in cycle.validate())
print("PASS 5: a reporting cycle in a hand-written spec is caught")

unknown_bits = CompanySpec(name="X", employees=[
    EmployeeSpec(name="A", rank="WIZARD", skills=["telepathy"], personality="grumpy")])
problems = unknown_bits.validate()
assert any("WIZARD" in p for p in problems)
assert any("telepathy" in p for p in problems)
assert any("grumpy" in p for p in problems)
print("PASS 6: unknown rank, skill and personality names are each reported with the valid set")

assert any("template or at least one employee" in p for p in CompanySpec(name="Empty").validate())
print("PASS 7: a spec describing nobody is rejected")

try:
    EmployeeSpec.from_dict({"name": "A", "salary": 100})
    assert False, "expected an unknown-field error"
except ValueError as e:
    assert "salary" in str(e)
try:
    CompanySpec.from_dict({"name": "A", "headquarters": "Mars"})
    assert False, "expected an unknown-field error"
except ValueError as e:
    assert "headquarters" in str(e)
print("PASS 8: unknown fields are rejected rather than silently ignored")


# ---------------------------------------------------------------------------
# Layer 2: building
# ---------------------------------------------------------------------------

result = build_company(CompanySpec(name="Templated", template="small-coding-team", size="small"))
assert result.ok and isinstance(result.company, Company)
assert set(result.company.employees) == {"Chief", "Manager", "Reviewer", "Developer"}
assert result.company.teams and result.company.teams[0].reviewer.name == "Reviewer"
print("PASS 9: build_company builds a whole org from a template name")

result = build_company(spec)
assert result.ok
assert result.company.employees["Ada"].reports_to.name == "Grace"
assert result.company.employees["Ada"].skills == ["python"]
print("PASS 10: build_company hires hand-written employees and wires reporting lines")

# Order-independent: a manager listed after their report still resolves.
reversed_spec = CompanySpec(name="Rev", employees=[
    EmployeeSpec(name="Junior", rank=RoleRank.JUNIOR, reports_to="Boss"),
    EmployeeSpec(name="Boss", rank=RoleRank.C_SUITE),
])
result = build_company(reversed_spec)
assert result.ok and result.company.employees["Junior"].reports_to.name == "Boss"
print("PASS 11: employees can be listed in any order - reporting lines resolve regardless")

# A spec can extend a template rather than replace it.
extended = CompanySpec(name="Ext", template="solo",
                       employees=[EmployeeSpec(name="Helper", rank=RoleRank.INTERN, reports_to="Worker")])
result = build_company(extended)
assert result.ok and "Worker" in result.company.employees and "Helper" in result.company.employees
assert result.company.employees["Helper"].reports_to.name == "Worker"
print("PASS 12: a spec can extend a template with extra employees")

failed = build_company(CompanySpec(name="Bad", template="nope"))
assert not failed.ok and failed.company is None and failed.problems
print("PASS 13: an invalid spec returns problems and builds nothing at all - never half an org")

# The safe defaults, and the fact that they are announced rather than silent.
result = build_company(CompanySpec(name="Defaults", template="solo"))
warnings = " ".join(result.warnings)
assert "model_map" in warnings and "on_escalation" in warnings and "total_token_budget" in warnings
assert result.company.employees["Worker"].agent.provider == "ollama", \
    "with no model_map, a company built by an AI must not be able to spend money"
assert all(v["provider"] == "ollama" for v in DEFAULT_MODEL_MAP.values())
print("PASS 14: unconfigured builds default to local-only and warn about every default taken")

decision = default_on_escalation(
    EscalationEvent(kind="budget_exhausted", employee_name="A", rank="SENIOR", message="m"))
assert decision.approve is False and decision.note
print("PASS 15: the default escalation handler declines - no human attached means stop, not approve")

# Runtime concerns stay out of the serializable spec.
spec_keys = set(CompanySpec(name="x").to_dict())
for runtime_key in ("model_map", "on_escalation", "model_policy", "api_key", "presets"):
    assert runtime_key not in spec_keys, runtime_key
print("PASS 16: the spec describes the org only - no callables, providers or keys serialize into it")

report = build_company(CompanySpec(name="Reported", template="small-coding-team")).to_dict()
assert report["ok"] and report["company"] == "Reported"
assert len(report["employees"]) == 4 and report["teams"]
assert "Chief" in report["org_chart"]
json.dumps(report)  # must be serializable for a tool result
print("PASS 17: BuildResult.to_dict() is a JSON-serializable report including an ASCII org chart")


# ---------------------------------------------------------------------------
# Layer 3: the generated schema
# ---------------------------------------------------------------------------

options = company_options()
assert set(options) >= {"skills", "personalities", "palettes", "org_templates",
                        "ranks", "sizes", "review_modes"}
assert options["org_templates"] == ORG_TEMPLATES.names()
assert options["sizes"] == list(TASK_SIZES)
assert options["ranks"] == list(RoleRank.ORDER)
print("PASS 18: company_options() reports every valid name, sourced from Phase 5's registries")

schema = company_setup_schema()
assert schema["name"] == "set_up_company"
assert "does not run any task" in schema["description"]
properties = schema["parameters"]["properties"]
assert set(properties) >= {"name", "template", "size", "palette", "total_token_budget",
                           "employees_json", "review_mode"}
assert schema["parameters"]["required"] == ["name"], schema["parameters"]["required"]
print("PASS 19: the schema is auto-derived from the function's hints and docstring")

# It really is ToolRegistry's generator, not a hand-written parallel format.
plain = ToolRegistry()
plain.register(set_up_company)
generated = plain.schemas["set_up_company"]
assert generated["parameters"]["properties"].keys() == properties.keys()
assert generated["description"] == schema["description"]
print("PASS 20: the schema comes from ToolRegistry's own generator, so it cannot drift from the tool")

# ...enriched with the live preset names, which is what makes it usable.
assert properties["template"]["enum"] == ORG_TEMPLATES.names()
assert properties["size"]["enum"] == list(TASK_SIZES)
assert "small-coding-team" in properties["template"]["description"]
assert "SENIOR" in properties["employees_json"]["description"]
assert "python" in properties["employees_json"]["description"]
print("PASS 21: preset names are injected as enums and into descriptions, so a model can fill it in")

assert properties["total_token_budget"]["type"] == "integer"
assert properties["max_review_rounds"]["type"] == "integer"
print("PASS 22: numeric parameters carry the right JSON type")


# ---------------------------------------------------------------------------
# Layer 4: calling it as a tool
# ---------------------------------------------------------------------------

reply = json.loads(set_up_company(name="Tooled", template="small-coding-team", size="medium"))
assert reply["ok"] and reply["company"] == "Tooled"
assert len(reply["employees"]) == 6, [e["name"] for e in reply["employees"]]
assert "company" not in reply or isinstance(reply["company"], str)
print("PASS 23: set_up_company returns a JSON report of what got built")

reply = json.loads(set_up_company(name="X", template="does-not-exist"))
assert reply["ok"] is False and reply["problems"]
print("PASS 24: a bad tool call comes back as a problem report, not an exception")

reply = json.loads(set_up_company(name="Custom", employees_json=json.dumps([
    {"name": "Boss", "rank": "C_SUITE"},
    {"name": "Coder", "rank": "JUNIOR", "reports_to": "Boss", "skills": ["python"]},
])))
assert reply["ok"] and {e["name"] for e in reply["employees"]} == {"Boss", "Coder"}
coder = next(e for e in reply["employees"] if e["name"] == "Coder")
assert coder["skills"] == ["python"] and coder["reports_to"] == "Boss"
print("PASS 25: employees_json is the escape hatch for org shapes no template covers")

reply = json.loads(set_up_company(name="X", employees_json="{not json"))
assert reply["ok"] is False and any("not valid JSON" in p for p in reply["problems"])
reply = json.loads(set_up_company(name="X", employees_json='{"name": "solo"}'))
assert reply["ok"] is False and any("array" in p for p in reply["problems"])
print("PASS 26: malformed employees_json is reported clearly instead of raising")

# The flat-scalar sentinels: "" means no template, 0 means no ceiling.
result = set_company_up(mode="text", name="Sentinels", template="", employees_json=json.dumps(
    [{"name": "Solo", "rank": "SENIOR"}]), total_token_budget=0)
assert result.ok and result.spec.template is None
assert result.company.budget.total_token_budget is None
result = set_company_up(mode="text", name="Capped", template="solo", total_token_budget=5000)
assert result.company.budget.total_token_budget == 5000
print("PASS 27: empty-string template and 0 budget are the documented 'none' sentinels")

# set_company_up also takes a spec directly, and dispatches on mode.
result = set_company_up(mode="text", spec={"name": "FromSpec", "template": "solo"})
assert result.ok and result.company.name == "FromSpec"
result = set_company_up(mode="text", spec=CompanySpec(name="FromObject", template="solo"))
assert result.ok and result.company.name == "FromObject"
try:
    set_company_up(mode="telepathy", name="X")
    assert False, "expected an unknown-mode error"
except ValueError as e:
    assert "telepathy" in str(e)
print("PASS 28: set_company_up accepts a spec dict or object and rejects an unknown mode")

# Building must never run anything - that is the whole safety property.
result = set_company_up(mode="text", name="Idle", template="small-coding-team")
assert result.company.total_tokens_spent() == 0
assert not [e for e in result.company.activity() if e["kind"] == "task_start"]
print("PASS 29: building a company runs no tasks and spends nothing")

# An Agent can be given the builder in one line and gets the enriched schema.
agent = Agent(provider="anthropic", api_key="k")
register_company_builder(agent)
assert "set_up_company" in agent.tool_registry.schemas
assert agent.tool_registry.schemas["set_up_company"]["parameters"]["properties"]["template"]["enum"]
tool_output = agent.tool_registry.execute("set_up_company", {"name": "ViaTool", "template": "solo"})
assert json.loads(tool_output)["ok"]
print("PASS 30: register_company_builder wires the tool onto an Agent and it executes end to end")

# A forked preset bundle's custom names reach the schema, since it reads the
# registries live rather than a snapshot.
from llmadapt.presets import Skill  # noqa: E402

bundle = default_bundle().fork()
bundle.skills.register(Skill(name="private-skill", instructions="secret"))
assert "private-skill" in company_options(bundle)["skills"]
assert "private-skill" in company_setup_schema(bundle)["parameters"]["properties"]["employees_json"]["description"]
print("PASS 31: a forked bundle's custom presets show up in the options and the schema")

print("\nAll Phase 8 text/schema-mode checks passed.")
