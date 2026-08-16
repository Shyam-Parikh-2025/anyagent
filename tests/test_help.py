# Test made with AI to speed up validation, same style as test_builder.py:
# plain asserts + prints, no pytest, fully offline.
#
# Covers the "I don't know what to pass" entry points: company_help(),
# set_company_up(mode="help"), set_up_company(help_only=True), and the preset
# descriptions that make an options list choosable rather than just valid.
import json

from llmadapt import (
    PALETTES, PERSONALITIES, SKILLS, ORG_TEMPLATES,
    company_help, company_options, preset_descriptions, set_company_up, set_up_company,
)
from llmadapt.presets import default_bundle

# --- 1. every selectable preset explains itself ----------------------------
# The gap this closes: a checkbox list of bare names like "prompt-engineering"
# is valid input nobody can choose from confidently.
descriptions = preset_descriptions()
for kind, registry in (("skills", SKILLS), ("personalities", PERSONALITIES),
                       ("palettes", PALETTES), ("org_templates", ORG_TEMPLATES)):
    assert set(descriptions[kind]) == set(registry.names()), kind
    for name, text in descriptions[kind].items():
        assert text and text.strip(), f"{kind}/{name} has no description"
        assert len(text) > 15, f"{kind}/{name} description is too thin to help: {text!r}"
print(f"PASS 1: all {sum(len(v) for v in descriptions.values())} selectable presets carry a real description")


# --- 2. company_options() carries them, without changing its old shape -----
options = company_options()
assert isinstance(options["skills"], list) and isinstance(options["skills"][0], str), \
    "existing readers expect a plain list of names"
assert options["descriptions"]["skills"]["python"], options["descriptions"]["skills"].get("python")
print("PASS 2: company_options() gained descriptions without breaking its list-of-names shape")


# --- 3. company_help() is a complete, self-contained rundown ---------------
doc = company_help()
for key in ("summary", "concepts", "safety_defaults", "choices", "running_work",
            "gotchas", "examples", "next_steps", "help_version"):
    assert key in doc, f"help document is missing {key!r}"
assert len(doc["concepts"]) >= 8
assert len(doc["examples"]) >= 3
for example in doc["examples"]:
    assert example["title"] and "import" in example["code"], example
# Every choice is a name AND what it does - the whole point of this over
# company_options().
for kind in ("skills", "personalities", "palettes", "org_templates"):
    entries = doc["choices"][kind]
    assert entries and all(e["name"] and e["description"] for e in entries), kind
assert doc["choices"]["ranks"] and doc["choices"]["sizes"]
print(f"PASS 3: company_help() describes {len(doc['concepts'])} concepts and every option with its meaning")


# --- 4. the safety defaults are stated, not implied ------------------------
# These are the four that surprise people, and the reason a rundown exists at
# all rather than just a list of argument names.
safety = " ".join(f"{item['rule']} {item['detail']}" for item in doc["safety_defaults"]).lower()
assert "local" in safety and "cannot spend money" in safety
assert "decline" in safety
assert "0 means always ask a human" in safety
assert "no ceiling" in safety
assert "runs nothing" in safety or "never runs" in safety
print("PASS 4: the rundown spells out the safety defaults rather than leaving them to be inferred")


# --- 5. pausing and resuming is documented where someone would look --------
running = " ".join(f"{item['call']} {item['what']}" for item in doc["running_work"])
assert "run_resumable" in running and "resume" in running
assert "run_async" in running
assert "save_state" in running
print("PASS 5: the rundown covers the async, resumable and snapshot entry points")


# --- 6. markdown form, for a human or a model being handed context ---------
text = company_help(as_text=True)
assert text.startswith("# Building a company with llmadapt")
assert "## Safety defaults" in text and "## Examples" in text
assert "```python" in text
for name in ("small-coding-team", "python", "concise", "dataviz"):
    assert name in text, f"{name!r} missing from the markdown rundown"
assert len(text.splitlines()) > 100
print(f"PASS 6: company_help(as_text=True) renders a {len(text.splitlines())}-line markdown rundown")


# --- 7. reachable from set_company_up and from the AI-callable tool --------
assert set_company_up(mode="help") == doc
assert set_company_up(mode="help", as_text=True) == text

tool_output = set_up_company(name="ignored", template="also ignored", help_only=True)
parsed = json.loads(tool_output)
assert parsed["help_version"] == doc["help_version"]
assert parsed["choices"]["skills"], "the tool's help must carry the catalog too"
# help_only ignores everything else and builds nothing - no company in there.
assert "company" not in parsed and "ok" not in parsed
print("PASS 7: the rundown is reachable via set_company_up(mode='help') and set_up_company(help_only=True)")


# --- 8. an unknown mode still says what the real ones are ------------------
try:
    set_company_up(mode="magic")
    assert False, "an unknown mode should be rejected"
except ValueError as e:
    assert "'text'" in str(e) and "'gui'" in str(e) and "'help'" in str(e), str(e)
print("PASS 8: an unknown mode names all three real ones, including the new one")


# --- 9. it follows a forked bundle, not just the global registries --------
bundle = default_bundle().fork()
from llmadapt import Skill  # noqa: E402

bundle.skills.register(Skill(name="basket-weaving", description="Weaves baskets, allegedly.",
                             instructions="Weave a basket."))
forked = company_help(bundle=bundle)
names = [entry["name"] for entry in forked["choices"]["skills"]]
assert "basket-weaving" in names
assert "basket-weaving" not in [e["name"] for e in company_help()["choices"]["skills"]], \
    "a forked bundle's custom skill leaked into the global catalog"
print("PASS 9: the rundown reflects a forked bundle's custom presets without leaking them")

print("\nAll checks passed.")
