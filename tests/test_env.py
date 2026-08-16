"""test_env.py - .env loading and layered API-key resolution.

The bug this module exists to fix was invisible: `core.py` called
`os.getenv("GEMINI_API_KEY")`, the key was sitting in a `.env` at the project
root, and nothing in `src/` ever opened that file. So the assertions that
matter here are about the *paths a key can arrive by* and the precedence
between them - not about parsing for its own sake.

No key is ever printed. Same style as the rest of the suite: plain asserts,
plain prints, no pytest.
"""

import os
import shutil
import tempfile

from llmadapt.env import (
    env_value,
    find_env_file,
    key_env_candidates,
    load_env,
    parse_env_text,
    resolve_api_key,
)

checks = 0
TMP = tempfile.mkdtemp(prefix="llmadapt-env-")


def check(label, condition):
    global checks
    assert condition, label
    checks += 1
    print(f"PASS: {label}")


# --- 1. parsing ------------------------------------------------------------
# Every one of these forms appears in a real .env somewhere, and the user's own
# file uses the spaces-around-= form that a naive split would misread as a key
# named "GEMINI_API_KEY " (with a trailing space) that never matches anything.

SAMPLE = """
# a comment line
GEMINI_API_KEY = sk-spaces
ANTHROPIC_API_KEY=sk-plain
export OPENAI_API_KEY=sk-exported
QUOTED="sk-quoted"
SINGLE='sk-single'
TRAILING=sk-trailing # inline comment
HASH_IN_VALUE="sk-with#hash"

not a valid line at all
EMPTY=
"""
parsed = parse_env_text(SAMPLE)

check("spaces around '=' are handled (the form the project's own .env uses)",
      parsed["GEMINI_API_KEY"] == "sk-spaces")
check("a bare KEY=value works", parsed["ANTHROPIC_API_KEY"] == "sk-plain")
check("an 'export' prefix is accepted", parsed["OPENAI_API_KEY"] == "sk-exported")
check("double and single quotes are stripped",
      parsed["QUOTED"] == "sk-quoted" and parsed["SINGLE"] == "sk-single")
check("an inline comment is stripped from an unquoted value",
      parsed["TRAILING"] == "sk-trailing")
check("but a '#' inside a quoted value is part of the secret, not a comment",
      parsed["HASH_IN_VALUE"] == "sk-with#hash")
check("comments and blank lines are ignored", "# a comment line" not in parsed)
check("an unparseable line is skipped rather than raising", len(parsed) == 8)

# --- 2. finding the file ---------------------------------------------------
# Upward search, because running a script from tests/ or scripts/ is normal and
# the file lives at the project root.

root = os.path.join(TMP, "project")
deep = os.path.join(root, "a", "b", "c")
os.makedirs(deep)
with open(os.path.join(root, ".env"), "w", encoding="utf-8") as handle:
    handle.write("TOGETHER_API_KEY = sk-from-file\nMYVENDOR_API_KEY=sk-vendor\n")

check("a .env is found from a subdirectory by walking upward",
      find_env_file(start=deep) == os.path.join(root, ".env"))
check("no .env anywhere above returns None, rather than raising",
      find_env_file(start=tempfile.gettempdir(), filename=".definitely-not-here") is None)

loaded = load_env(path=os.path.join(root, ".env"), force=True)
check("load_env returns what it read", sorted(loaded) == ["MYVENDOR_API_KEY", "TOGETHER_API_KEY"])

# The file must never be pushed into the process environment - importing a
# library should not be able to change the environment of the program that
# imported it.
check("loading a .env does NOT write into os.environ",
      "TOGETHER_API_KEY" not in os.environ)
check("but the value is still resolvable through env_value",
      env_value("TOGETHER_API_KEY") == "sk-from-file")

# A real environment variable always beats the file.
os.environ["TOGETHER_API_KEY"] = "sk-from-shell"
check("os.environ wins over the .env file",
      env_value("TOGETHER_API_KEY") == "sk-from-shell")
del os.environ["TOGETHER_API_KEY"]

# --- 3. resolution order ---------------------------------------------------

os.environ.pop("OPENAI_API_KEY", None)
os.environ["OPENAI_API_KEY"] = "sk-openai-account"
os.environ["OPENROUTER_API_KEY"] = "sk-openrouter-account"

key, source = resolve_api_key(provider="openai", explicit="sk-explicit")
check("an explicit key always wins", key == "sk-explicit" and source == "passed explicitly")

key, source = resolve_api_key(provider="openai")
check("provider fallback still works exactly as before",
      key == "sk-openai-account" and "OPENAI_API_KEY" in source)

# The case the layering exists for: OpenRouter, Together, Groq and a local vLLM
# are all provider="openai" to the transport, and do not share an account.
key, source = resolve_api_key(provider="openai", alias="openrouter")
check("an alias is checked before the provider, so same-protocol endpoints differ",
      key == "sk-openrouter-account" and "OPENROUTER_API_KEY" in source)

key, source = resolve_api_key(provider="openai", key_env="MYVENDOR_API_KEY")
check("an explicit key_env name beats both, and reads from the .env file",
      key == "sk-vendor" and "from a .env file" in source)

check("candidate names are reported most-specific-first",
      key_env_candidates(provider="openai", alias="openrouter", key_env="CUSTOM")
      == ["CUSTOM", "OPENROUTER_API_KEY", "OPENAI_API_KEY"])

# An alias with punctuation still produces a legal variable name.
check("an alias is normalized into a legal env var name",
      key_env_candidates(alias="my-vendor.v2") == ["MY_VENDOR_V2_API_KEY"])

# --- 4. the missing-key path ----------------------------------------------
# "No key" on its own tells you nothing about which name to set, which is
# exactly the confusion this whole module exists to remove.

os.environ.pop("NOSUCH_API_KEY", None)
key, source = resolve_api_key(provider="nosuch")
check("a missing key returns None rather than raising", key is None)
check("...and the reason names every variable that was tried",
      "NOSUCH_API_KEY" in source and "tried" in source)

try:
    resolve_api_key(provider="nosuch", required=True)
    assert False, "expected a ValueError"
except ValueError as e:
    check("required=True raises, and the message says what to set",
          "NOSUCH_API_KEY" in str(e))

# --- 5. through the Agent and the Company ---------------------------------

from llmadapt import Agent, Company, EscalationDecision  # noqa: E402
from llmadapt.router import RoleRank  # noqa: E402

agent = Agent(provider="openai", model="x/y", key_alias="openrouter")
check("Agent resolves an aliased key at construction",
      agent.api_key == "sk-openrouter-account")

local = Agent(provider="ollama", model="llama3.1:8b")
check("a local provider needs no key and does not report a missing one",
      local.api_key is None and "not needed" in local.api_key_source)

MODEL_MAP = {
    RoleRank.C_SUITE: {"provider": "anthropic", "model": "claude-opus-4-6", "api_key": "sk-direct"},
    RoleRank.JUNIOR: {"provider": "openai", "model": "x/y", "alias": "openrouter"},
    RoleRank.INTERN: {"provider": "ollama", "model": "llama3.1:8b"},
}
company = Company(name="Env Co", model_map=MODEL_MAP,
                  on_escalation=lambda e: EscalationDecision(approve=False))
for name, rank in (("Boss", RoleRank.C_SUITE), ("Kid", RoleRank.JUNIOR), ("Local", RoleRank.INTERN)):
    company.hire(name, rank)

sources = company.key_sources()
check("model_map's alias reaches the hired Agent",
      company.employees["Kid"].agent.api_key == "sk-openrouter-account")
check("key_sources reports provenance for every employee", len(sources) == 3)
check("key_sources never returns a key itself",
      all("key" not in v or v.get("has_key") in (True, False) for v in sources.values())
      and not any("sk-" in str(v) for v in sources.values()))
check("key_sources lists the variables tried, most specific first",
      sources["Kid"]["tried"] == ["OPENROUTER_API_KEY", "OPENAI_API_KEY"])

shutil.rmtree(TMP, ignore_errors=True)
print(f"\nOK - {checks} checks")
