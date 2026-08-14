# Phase 5 tests - presets.py (the one named-preset registry pattern) and its
# wiring into company.py and observability.py. Plain asserts + prints, no
# pytest, no network. Layers, bottom-up:
#   (1) PresetRegistry itself - the mechanism all four kinds share
#   (2) each kind: skills, personalities, palettes, org templates
#   (3) system-instruction composition
#   (4) Company integration - hire(skills=/personality=), build_from_template,
#       render_org_chart(palette=), and the Team review loop
import json

from llmadapt.company import Company, EscalationDecision, Team
from llmadapt.observability import resolve_palette
from llmadapt.presets import (
    ORG_TEMPLATES,
    PALETTES,
    PERSONALITIES,
    SKILLS,
    TASK_SIZES,
    OrgTemplate,
    Palette,
    Personality,
    PresetBundle,
    PresetRegistry,
    RoleSpec,
    Skill,
    compose_system_instruction,
    default_bundle,
    skill_hints,
)
from llmadapt.router import RoleRank


class FakeResponder:
    """Returns scripted replies; if it runs out, it repeats the last one, so a
    review loop that takes an extra turn doesn't IndexError the test."""

    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []
        self._last = None

    def __call__(self, payload, headers):
        self.calls.append(json.loads(json.dumps(payload)))
        if self.responses:
            self._last = self.responses.pop(0)
        return self._last


def text_response(text):
    return {"content": [{"type": "text", "text": text}]}


MODEL_MAP = {
    rank: {"provider": "anthropic", "model": "claude-x", "api_key": "k"}
    for rank in RoleRank.ORDER
}


def make_company(**kwargs):
    def on_escalation(event):
        return EscalationDecision(approve=False)

    return Company(name="Preset Co", model_map=MODEL_MAP, on_escalation=on_escalation, **kwargs)


# ---------------------------------------------------------------------------
# Layer 1: the shared mechanism
# ---------------------------------------------------------------------------

reg = PresetRegistry("thing", Skill)
a = reg.register(Skill(name="a", instructions="do a"))
assert len(reg) == 1 and "a" in reg and reg.get("a") is a
assert reg.names() == ["a"]
print("PASS 1: PresetRegistry registers and looks up by name")

try:
    reg.register(Skill(name="a", instructions="different"))
    assert False, "expected a duplicate-name error"
except ValueError as e:
    assert "already registered" in str(e)
reg.register(Skill(name="a", instructions="different"), overwrite=True)
assert reg.get("a").instructions == "different"
print("PASS 2: duplicate names are refused unless overwrite=True is explicit")

try:
    reg.get("nope")
    assert False, "expected KeyError"
except KeyError as e:
    assert "nope" in str(e) and "Available" in str(e)
print("PASS 3: an unknown name raises listing what IS available, never returns None")

try:
    reg.register(Personality(name="wrong"))
    assert False, "expected TypeError"
except TypeError as e:
    assert "Skill" in str(e)
print("PASS 4: a registry refuses presets of the wrong kind")

# resolve() takes a name OR an object - the object needs no registration.
one_off = Skill(name="never-registered", instructions="just this once")
assert reg.resolve("a") is reg.get("a")
assert reg.resolve(one_off) is one_off
assert "never-registered" not in reg, "resolving an object must not register it"
print("PASS 5: resolve() accepts a name or an unregistered object, without side effects")

fork = reg.copy()
fork.register(Skill(name="only-in-fork"))
assert "only-in-fork" in fork and "only-in-fork" not in reg
print("PASS 6: copy() isolates membership so a fork can't leak into the shared registry")

reg.remove("a")
assert "a" not in reg
print("PASS 7: remove() drops a preset")

# All four built-in registries are the same class with the same API - that is
# the whole design claim of this phase.
for registry in (SKILLS, PERSONALITIES, PALETTES, ORG_TEMPLATES):
    assert isinstance(registry, PresetRegistry)
    assert len(registry) > 0
    assert registry.names() == sorted(registry.names())
    assert all("name" in d and "description" in d for d in registry.describe())
print("PASS 8: skills, personalities, palettes and org templates are all the same registry type")


# ---------------------------------------------------------------------------
# Layer 2: the four kinds
# ---------------------------------------------------------------------------

python_skill = SKILLS.get("python")
assert python_skill.constraints and python_skill.specialty == "code"
assert all(s.description for s in SKILLS.all()), "every built-in skill needs a description"
print("PASS 9: built-in skills carry instructions, constraints and a Phase 4 specialty hint")

assert all(p.instructions for p in PERSONALITIES.all())
print("PASS 10: built-in personalities all carry instructions")

dataviz = PALETTES.get("dataviz")
assert dataviz.colorblind_validated is True
assert len(dataviz.ranks_light) == len(RoleRank.ORDER)
assert len(dataviz.ranks_dark) == len(RoleRank.ORDER)
# Only the dataviz palette claims validation - the stylistic ones must not.
assert [p.name for p in PALETTES.all() if p.colorblind_validated] == ["dataviz"]
print("PASS 11: the dataviz palette covers every rank and is the only one claiming validation")

light = dataviz.color_for_rank(RoleRank.C_SUITE, "light")
dark = dataviz.color_for_rank(RoleRank.C_SUITE, "dark")
assert light != dark and light.startswith("#")
assert dataviz.color_for_rank("NOT_A_RANK", "light") in dataviz.ranks_light
print("PASS 12: palettes are theme-aware and fall back for an unknown rank instead of crashing")

template = ORG_TEMPLATES.get("small-coding-team")
small = template.roles_for("small")
large = template.roles_for("large")
assert len(large) > len(small), "large should activate roles small doesn't have"
dev = template.role("dev")
assert dev.count_for("small") == 1 and dev.count_for("large") == 4
assert template.role("intern").count_for("small") == 0
print("PASS 13: org template roles scale by declared per-size counts")

for name in ORG_TEMPLATES.names():
    ORG_TEMPLATES.get(name).validate()
print("PASS 14: every built-in org template validates (keys, reporting lines, single lead)")

bad = OrgTemplate(name="bad", roles=(RoleSpec(key="a", rank=RoleRank.SENIOR, reports_to="ghost"),))
try:
    bad.validate()
    assert False, "expected a validation error for a dangling reports_to"
except ValueError as e:
    assert "ghost" in str(e)

cyclic = OrgTemplate(name="cyc", roles=(
    RoleSpec(key="a", rank=RoleRank.SENIOR, reports_to="b"),
    RoleSpec(key="b", rank=RoleRank.SENIOR, reports_to="a"),
))
try:
    cyclic.validate()
    assert False, "expected a cycle error"
except ValueError as e:
    assert "cycle" in str(e)
print("PASS 15: template validation catches dangling reporting lines and cycles")

try:
    template.roles_for("gigantic")
    assert False, "expected an unknown-size error"
except ValueError as e:
    assert "gigantic" in str(e)
assert TASK_SIZES == ("small", "medium", "large")
print("PASS 16: an unknown task size raises rather than silently returning nothing")


# ---------------------------------------------------------------------------
# Layer 3: system-instruction composition
# ---------------------------------------------------------------------------

assert compose_system_instruction() == ""
print("PASS 17: no presets composes to an empty instruction, not a header skeleton")

composed = compose_system_instruction(
    base="You work at ACME.", personality="concise", skills=["python", "code-review"],
)
assert composed.startswith("You work at ACME.")
assert "## How you work" in composed and "## Your skills" in composed
assert "## Hard constraints" in composed
assert composed.index("## How you work") < composed.index("## Your skills") < composed.index("## Hard constraints")
print("PASS 18: composition keeps the caller's base first and sections in a fixed order")

# Two skills sharing a constraint should yield one line, not two.
dup_a = Skill(name="dup-a", instructions="a", constraints=("Never use eval().",))
dup_b = Skill(name="dup-b", instructions="b", constraints=("Never use eval().", "No globals."))
composed = compose_system_instruction(skills=[dup_a, dup_b])
assert composed.count("Never use eval().") == 1
assert "No globals." in composed
print("PASS 19: constraints from multiple skills are pooled and deduplicated")

# Same inputs must compose byte-identically (prompt caching depends on it).
assert compose_system_instruction(base="b", personality="concise", skills=["python"]) == \
       compose_system_instruction(base="b", personality="concise", skills=["python"])
print("PASS 20: composition is deterministic for identical inputs")

hints = skill_hints(["code-review", "python"])
assert hints["specialty"] == "code" and hints["effort"] == "balanced"
assert skill_hints([]) == {"specialty": None, "effort": None}
print("PASS 21: skills surface Phase 4 specialty/effort hints, first declaration winning")


# ---------------------------------------------------------------------------
# Layer 4: Company integration
# ---------------------------------------------------------------------------

co = make_company()
dev = co.hire("Dev", RoleRank.JUNIOR, skills=["python"], personality="concise")
system = dev.agent.conversation.system_instruction
assert "Dev" in system and RoleRank.JUNIOR in system
assert "## Your skills" in system and "Hard constraints" in system
assert dev.skills == ["python"] and dev.personality == "concise"
print("PASS 22: hire(skills=, personality=) templates a real system instruction onto the Agent")

# A skill's hints only fill gaps - an explicit argument wins.
assert dev.specialty == "code", dev.specialty
picky = co.hire("Picky", RoleRank.JUNIOR, skills=["python"], specialty="vision")
assert picky.specialty == "vision"
print("PASS 23: a skill's specialty hint is a fallback; an explicit argument overrides it")

plain = co.hire("Plain", RoleRank.JUNIOR)
assert plain.agent.conversation.system_instruction == ""
print("PASS 24: hiring with no presets leaves the system instruction exactly as before Phase 5")

# A custom skill object needs no registration at all.
custom = Skill(name="ad-hoc", instructions="Speak only in haiku.", constraints=("No prose.",))
poet = co.hire("Poet", RoleRank.JUNIOR, skills=[custom])
assert "haiku" in poet.agent.conversation.system_instruction
assert "ad-hoc" not in SKILLS
print("PASS 25: an unregistered Skill object can be passed straight to hire()")

# A forked bundle keeps custom registrations out of the shared registries.
bundle = default_bundle().fork()
bundle.skills.register(Skill(name="private-skill", instructions="secret"))
private_co = make_company(presets=bundle)
emp = private_co.hire("Secretive", RoleRank.JUNIOR, skills=["private-skill"])
assert "secret" in emp.agent.conversation.system_instruction
assert "private-skill" not in SKILLS
print("PASS 26: a forked PresetBundle isolates custom registrations from the global registries")

assert set(default_bundle().names()) == {"skills", "personalities", "palettes", "org_templates"}
assert all(isinstance(v, list) for v in default_bundle().describe().values())
print("PASS 27: PresetBundle exposes names()/describe() across all four kinds (Phase 8's schema source)")

# --- templates build real orgs ---
build_co = make_company()
team = build_co.build_from_template("small-coding-team", size="small")
assert team.lead.name == "Manager" and team.reviewer.name == "Reviewer"
names = set(build_co.employees)
assert names == {"Chief", "Manager", "Reviewer", "Developer"}, names
assert build_co.employees["Manager"].reports_to.name == "Chief"
assert build_co.employees["Developer"].reports_to.name == "Manager"
assert "code-review" in build_co.employees["Reviewer"].skills
print("PASS 28: build_from_template builds the org, wires reporting lines, and applies role presets")

big_co = make_company()
big_team = big_co.build_from_template("small-coding-team", size="large")
devs = [n for n in big_co.employees if n.startswith("Developer")]
assert len(devs) == 4, devs
assert "Intern 1" in big_co.employees and "Intern 2" in big_co.employees
assert len(big_co.employees) > len(names)
print("PASS 29: size='large' scales the worker roles and activates size-gated ones")

# Two instantiations in one Company need only a prefix.
big_co.build_from_template("solo", name_prefix="B-")
assert "B-Worker" in big_co.employees
print("PASS 30: name_prefix lets the same Company hold more than one instantiated template")

# Every built-in template builds at every size without raising.
for template_name in ORG_TEMPLATES.names():
    for size in TASK_SIZES:
        probe = make_company()
        built = probe.build_from_template(template_name, size=size)
        assert built.lead is not None
        assert all(e.reports_to is None or e.reports_to.name in probe.employees
                   for e in probe.employees.values())
print("PASS 31: every built-in template builds cleanly at every size, with valid reporting lines")

built_events = build_co.activity().by_kind("team_built")
assert built_events and built_events[0]["template"] == "small-coding-team"
print("PASS 32: building from a template is recorded in the activity log")

# --- palettes through the same registry ---
chart_co = make_company()
chart_co.hire("Solo", RoleRank.C_SUITE)
default_svg = chart_co.render_org_chart(fmt="svg")
grayscale_svg = chart_co.render_org_chart(fmt="svg", palette="grayscale")
assert default_svg != grayscale_svg
assert PALETTES.get("grayscale").ranks_light[0] in grayscale_svg
assert default_svg == chart_co.render_org_chart(fmt="svg", palette="dataviz")
print("PASS 33: render_org_chart(palette=) resolves by name and defaults to the pre-Phase-5 colors")

mermaid = chart_co.render_org_chart(fmt="mermaid", palette="ocean")
assert PALETTES.get("ocean").ranks_light[0] in mermaid
print("PASS 34: the mermaid renderer honors the palette too")

custom_palette = Palette(name="mine", ranks_light=("#111111",) * 7, ranks_dark=("#eeeeee",) * 7)
assert "#111111" in chart_co.render_org_chart(fmt="svg", palette=custom_palette)
assert resolve_palette(None).name == "dataviz"
try:
    chart_co.render_org_chart(fmt="svg", palette="no-such-palette")
    assert False, "expected an unknown-palette error"
except KeyError as e:
    assert "no-such-palette" in str(e)
print("PASS 35: an unregistered Palette object works; an unknown palette name raises clearly")


# ---------------------------------------------------------------------------
# Layer 4b: the Team review loop (the previously-dormant reviewer)
# ---------------------------------------------------------------------------

review_co = make_company()
lead = review_co.hire("Lead", RoleRank.MANAGER)
reviewer = review_co.hire("Rev", RoleRank.SENIOR, reports_to=lead)

lead.agent._send_request = FakeResponder([text_response("draft answer")])
reviewer.agent._send_request = FakeResponder([text_response("APPROVED")])
team = Team("T", lead=lead, reviewer=reviewer)
assert team.run("do the thing", company=review_co) == "draft answer"
assert review_co.activity().by_kind("review_approved")
print("PASS 36: an approving reviewer returns the lead's draft untouched")

lead2 = review_co.hire("Lead2", RoleRank.MANAGER)
rev2 = review_co.hire("Rev2", RoleRank.SENIOR, reports_to=lead2)
lead2.agent._send_request = FakeResponder([text_response("first draft"), text_response("revised draft")])
rev2.agent._send_request = FakeResponder([
    text_response("Please fix the second paragraph."), text_response("APPROVED"),
])
team2 = Team("T2", lead=lead2, reviewer=rev2)
result = team2.run("do the thing", company=review_co)
assert result == "revised draft", result
assert review_co.activity().by_kind("review_revised")
print("PASS 37: a critique sends the work back and the revised draft is what ships")

# max_review_rounds counts send-backs, so a revision is always reviewed before
# it ships - one more review than revision.
assert len(rev2.agent._send_request.calls) == 2
assert len(lead2.agent._send_request.calls) == 2
print("PASS 37b: the final revision gets its own review pass before shipping")

# Out of rounds with the objection standing: deliver work AND objection.
lead3 = review_co.hire("Lead3", RoleRank.MANAGER)
rev3 = review_co.hire("Rev3", RoleRank.SENIOR, reports_to=lead3)
lead3.agent._send_request = FakeResponder([text_response("d1"), text_response("d2")])
rev3.agent._send_request = FakeResponder([text_response("Still wrong.")])
team3 = Team("T3", lead=lead3, reviewer=rev3, max_review_rounds=1)
result = team3.run("do the thing", company=review_co)
assert "d2" in result and "Unresolved reviewer note" in result and "Still wrong." in result
assert review_co.activity().by_kind("review_unresolved")
print("PASS 38: an unresolved objection ships alongside the work instead of being dropped")

# review_mode='append' returns both answers, labelled.
lead4 = review_co.hire("Lead4", RoleRank.MANAGER)
rev4 = review_co.hire("Rev4", RoleRank.SENIOR, reports_to=lead4)
lead4.agent._send_request = FakeResponder([text_response("lead take")])
rev4.agent._send_request = FakeResponder([text_response("reviewer take")])
team4 = Team("T4", lead=lead4, reviewer=rev4, review_mode="append")
result = team4.run("do the thing", company=review_co)
assert "lead take" in result and "reviewer take" in result and "(reviewer)" in result
print("PASS 39: review_mode='append' returns both answers labelled, as a second opinion")

# review_mode='off' is exactly the pre-Phase-5 behavior.
lead5 = review_co.hire("Lead5", RoleRank.MANAGER)
rev5 = review_co.hire("Rev5", RoleRank.SENIOR, reports_to=lead5)
lead5.agent._send_request = FakeResponder([text_response("only mine")])
rev5.agent._send_request = FakeResponder([text_response("never called")])
team5 = Team("T5", lead=lead5, reviewer=rev5, review_mode="off")
assert team5.run("x", company=review_co) == "only mine"
assert rev5.agent._send_request.calls == []
print("PASS 40: review_mode='off' preserves the pre-Phase-5 lead-only behavior exactly")

try:
    Team("bad", lead=lead5, reviewer=rev5, review_mode="vibes")
    assert False, "expected an unknown review_mode error"
except ValueError as e:
    assert "vibes" in str(e)
print("PASS 41: an unknown review_mode raises at construction")

# A team with no reviewer behaves exactly as before, and a built template's
# team actually runs its review loop end to end.
tmpl_co = make_company()
tmpl_team = tmpl_co.build_from_template("writing-desk", size="small")
for employee in tmpl_co.employees.values():
    employee.agent._send_request = FakeResponder([text_response("APPROVED")])
tmpl_team.lead.agent._send_request = FakeResponder([text_response("the copy")])
assert tmpl_team.review_mode == "critique"
print("PASS 42: a template-built team comes back with its review loop configured")

print("\nAll Phase 5 preset-registry checks passed.")
