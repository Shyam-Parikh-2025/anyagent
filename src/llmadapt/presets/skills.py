"""presets/skills.py - the Skill preset and the built-in catalog.

Skills are *prompt-level* capabilities: instructions plus hard constraints
templated into an employee's system instruction. A skill cannot register a
tool or change the tool-calling loop - "do not use library X" is a sentence in
a prompt, not an enforced sandbox, and nothing here verifies compliance.

Two rules the catalog holds to, because a preset library is only useful if its
entries are predictable:

- **One skill, one job.** `commenting` is separate from `python` because
  commenting rules are not about Python, and folding them in means copying
  them into the next language's skill. Compose instead:
  `hire(skills=["python", "commenting"])`.
- **Declare the Phase 4 hints honestly.** `specialty` and `effort` steer model
  routing, and they are a *fallback* - an explicit `hire(effort=...)` always
  wins. A skill that claims `effort="effort"` is asking every employee holding
  it to be routed to an expensive model, so only the ones where that is really
  true say it.
"""

from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence, Tuple

from .registry import Preset, PresetRegistry

@dataclass
class Skill(Preset):
    """A capability an employee is told they have.

    `instructions` is prose spliced into the system instruction.
    `constraints` are hard "don't" rules, collected across every skill an
    employee holds and rendered as one deduplicated list - so two skills both
    saying "no third-party dependencies" produce one line, not two.

    `specialty` and `effort` are the Phase 4 hook: a skill can declare that
    work of this kind wants a model tagged e.g. "code", or that it is
    inherently high-effort. `Company.hire()` uses them only as a *fallback*
    when the caller didn't pass their own - an explicit argument always beats
    a skill's suggestion.
    """

    instructions: str = ""
    constraints: Sequence[str] = ()
    specialty: Optional[str] = None
    effort: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        out = super().to_dict()
        out.update({
            "instructions": self.instructions,
            "constraints": list(self.constraints),
            "specialty": self.specialty,
            "effort": self.effort,
        })
        return out


BUILTIN_SKILLS: Tuple[Skill, ...] = (
    Skill(
        name="python",
        description="Writes idiomatic, well-typed Python.",
        instructions=(
            "Write idiomatic Python 3. Use type hints on public functions, prefer the standard "
            "library over new dependencies, and keep functions small enough to test."
        ),
        constraints=("Do not add a third-party dependency without saying why it is unavoidable.",),
        specialty="code",
    ),
    # Commenting rules are their own skill rather than a paragraph inside
    # "python", because they are not about Python. The same rule - explain the
    # non-obvious, leave the obvious alone - is what you want from an employee
    # writing SQL, a shell script or a Terraform file, and folding it into one
    # language's skill means copying it into the next language's skill too.
    # Composition is what the registry is for: hire(skills=["python",
    # "commenting"]). The cost is one more name at hire time; the gain is that
    # hire(skills=["python"]) still means only "write good Python" for a
    # caller who wants terse output, and that constraints from both are pooled
    # and deduplicated by compose_system_instruction() automatically.
    Skill(
        name="commenting",
        description="Comments and documents code for the next reader, without noise.",
        instructions=(
            "Include docstrings and comments where they help, but do not over-comment obvious "
            "code. Keep them clear and concise, and reserve them for the non-obvious: why this "
            "approach, what the caller has to know, what would break. Do not restate what the "
            "line already says."
        ),
        constraints=("Do not add a comment that only repeats the code it sits above.",),
        specialty="code",
    ),
    Skill(
        name="code-review",
        description="Reviews code for correctness, clarity, and risk.",
        instructions=(
            "Review code for correctness first, then clarity, then style. Point at specific lines. "
            "Say plainly when something is fine - do not invent problems to seem thorough."
        ),
        constraints=("Do not rewrite the whole file when a targeted fix would do.",),
        specialty="code",
        effort="balanced",
    ),
    Skill(
        name="research",
        description="Gathers and cites evidence before concluding.",
        instructions=(
            "Gather evidence before concluding. Distinguish what a source states from what you "
            "infer, and say when the evidence is thin rather than filling the gap with confidence."
        ),
        constraints=("Do not present an inference as a cited fact.",),
        specialty="reasoning",
        effort="effort",
    ),
    Skill(
        name="writing",
        description="Produces clear prose for a stated audience.",
        instructions=(
            "Write for the stated audience. Lead with the conclusion, keep sentences short, and "
            "cut any sentence that does not carry information."
        ),
        constraints=("Do not pad with filler introductions or restate the prompt back.",),
        specialty="writing",
    ),
    Skill(
        name="data-analysis",
        description="Analyzes tabular/numeric data and states its limits.",
        instructions=(
            "State the question, the method, and the limitations of the data before the answer. "
            "Show the numbers you relied on."
        ),
        constraints=("Do not report a correlation as a cause.",),
        specialty="reasoning",
        effort="balanced",
    ),
    Skill(
        name="sql",
        description="Writes and reviews SQL that is correct before it is clever.",
        instructions=(
            "Write standard SQL unless a dialect is named, and say which dialect you assumed. "
            "State the grain of the result - one row per what - before the query. Prefer an "
            "explicit JOIN and an explicit column list over SELECT *, and check that every "
            "aggregate has the GROUP BY it needs."
        ),
        constraints=(
            "Do not write an UPDATE or DELETE without a WHERE clause, even as an example.",
            "Do not claim a query is optimized without saying what you assumed about the indexes.",
        ),
        specialty="code",
    ),
    Skill(
        name="testing",
        description="Writes tests that would actually fail if the code were wrong.",
        instructions=(
            "Write tests that can fail. Cover the boundary and the error path, not only the "
            "happy case, and assert on behaviour rather than on incidental structure. Name "
            "each test after the property it establishes."
        ),
        constraints=(
            "Do not write a test whose assertions would pass against an empty implementation.",
            "Do not mock the thing under test.",
        ),
        specialty="code",
        effort="balanced",
    ),
    Skill(
        name="debugging",
        description="Finds the actual cause instead of the first plausible one.",
        instructions=(
            "Find the cause before proposing a fix. State what you expected, what happened, and "
            "the smallest input that reproduces it. When the evidence does not distinguish two "
            "explanations, say so and name the observation that would."
        ),
        constraints=("Do not propose a fix while the cause is still a guess - say it is a guess.",),
        specialty="code",
        effort="balanced",
    ),
    Skill(
        name="refactoring",
        description="Changes structure without changing behaviour.",
        instructions=(
            "Change structure, not behaviour. Make one kind of change at a time and say what "
            "guarantees it preserves. If a refactor would change an observable result, stop and "
            "flag it as a behaviour change instead of folding it in quietly."
        ),
        constraints=("Do not mix a behaviour change into a refactor without calling it out.",),
        specialty="code",
    ),
    Skill(
        name="api-design",
        description="Designs interfaces around the caller, and names the compatibility cost.",
        instructions=(
            "Design from the caller's side: what they have, what they want back, what they must "
            "not have to know. Make the common case short and the dangerous case explicit. Say "
            "what each choice costs in backwards compatibility."
        ),
        constraints=("Do not add a parameter that changes the meaning of the others without saying so.",),
        specialty="code",
        effort="balanced",
    ),
    Skill(
        name="security-review",
        description="Looks for exploitable flaws, and rates them honestly.",
        instructions=(
            "Look for the classes that actually bite: injection, unvalidated input crossing a "
            "trust boundary, secrets in code or logs, missing authorization, unsafe "
            "deserialization. For each finding give the concrete path an attacker takes. Rank by "
            "exploitability, not by how alarming it sounds."
        ),
        constraints=(
            "Do not report a theoretical issue as exploitable without the path that exploits it.",
            "Do not include a real secret, or a real token, in your output.",
        ),
        specialty="reasoning",
        effort="effort",
    ),
    Skill(
        name="devops",
        description="Automates and operates, with the failure path thought through first.",
        instructions=(
            "Prefer reproducible steps over clever ones. State what the change does on a machine "
            "that is already in a bad state, and how it is rolled back. Say which steps are "
            "destructive before listing them."
        ),
        constraints=(
            "Do not give a command that deletes data without saying exactly what it deletes.",
            "Do not assume credentials or network access that were not stated.",
        ),
        specialty="code",
        effort="balanced",
    ),
    Skill(
        name="technical-docs",
        description="Documents what a reader needs to act, not everything that exists.",
        instructions=(
            "Write for someone trying to do a thing, not for completeness. Lead with the task, "
            "show the smallest working example, then the options. Document what is guaranteed "
            "and what is incidental, so a reader knows what they may rely on."
        ),
        constraints=("Do not document behaviour you have not verified from the code.",),
        specialty="writing",
    ),
    Skill(
        name="summarization",
        description="Compresses without quietly changing what was said.",
        instructions=(
            "Preserve the claims, the numbers and the uncertainty. Say who claimed what when the "
            "source matters. Prefer dropping a whole point to shortening it into something "
            "stronger than the original said."
        ),
        constraints=(
            "Do not resolve a disagreement in the source by summarizing only one side.",
            "Do not state a hedged claim as a flat one.",
        ),
        specialty="writing",
    ),
    # --- Additional built-ins ----------------------------------------------
    # Same two rules as above: one skill one job, and honest Phase 4 hints.
    # Anything claiming effort="effort" is asking to be routed to an expensive
    # model every time it is worn, so only the ones where that is really true
    # say it.
    Skill(
        name="javascript",
        description="Writes modern, dependency-light JavaScript.",
        instructions=(
            "Write modern JavaScript (ES2020+). Prefer built-ins and platform APIs over packages, "
            "handle the async paths explicitly, and say which runtime you are targeting when it "
            "changes the answer."
        ),
        constraints=("Do not reach for a framework when a few lines of plain JS would do.",),
        specialty="code",
    ),
    Skill(
        name="frontend-ui",
        description="Builds interfaces that work before they are decorated.",
        instructions=(
            "Get the structure and the states right first: loading, empty, error, and too much "
            "data. Use semantic markup, keep layout responsive by default, and describe the "
            "interaction, not just the appearance."
        ),
        constraints=("Do not ship an interactive element that cannot be reached by keyboard.",),
        specialty="code",
    ),
    Skill(
        name="accessibility",
        description="Checks work against what people using assistive technology actually get.",
        instructions=(
            "Review for real assistive-technology use: focus order, names and roles for every "
            "control, contrast, and whether meaning survives without colour or sound. Name the "
            "specific barrier and who it stops, rather than citing a guideline number alone."
        ),
        constraints=(
            "Do not call something accessible because it passes an automated checker - say what "
            "was and was not checked by hand.",
        ),
    ),
    Skill(
        name="performance-tuning",
        description="Makes things faster on evidence, not on instinct.",
        instructions=(
            "Measure before changing anything, and say what you measured. Find the actual "
            "bottleneck rather than the suspicious-looking code, change one thing at a time, and "
            "report the before and after."
        ),
        constraints=(
            "Do not claim a speedup without a measurement.",
            "Do not trade correctness for speed without flagging it as that trade.",
        ),
        specialty="code",
        effort="effort",
    ),
    Skill(
        name="database-design",
        description="Designs schemas that survive the second year.",
        instructions=(
            "Model the entities and their real relationships before the tables. Say what each key "
            "and index is for, where the data will grow, and which constraint enforces each rule "
            "you are relying on."
        ),
        constraints=("Do not propose a migration without saying how it behaves on a table already full of rows.",),
        specialty="code",
    ),
    Skill(
        name="planning",
        description="Breaks work into steps that can actually be handed out.",
        instructions=(
            "Turn the task into ordered steps that each have a clear finish line. Name what has to "
            "happen before what, mark the steps that can run at the same time, and flag the one "
            "step most likely to be wrong."
        ),
        constraints=("Do not produce a step nobody could tell was finished.",),
        specialty="reasoning",
    ),
    Skill(
        name="requirements-gathering",
        description="Finds out what was actually wanted before anything is built.",
        instructions=(
            "Separate what was asked for from what is needed. Write down the assumptions you are "
            "making, list the questions whose answers would change the design, and state what is "
            "explicitly out of scope."
        ),
        constraints=("Do not invent a requirement to fill a gap - mark the gap as a question.",),
    ),
    Skill(
        name="fact-checking",
        description="Verifies claims and says how confident it is, and why.",
        instructions=(
            "Check each claim against a source and say which source. Separate what is verified "
            "from what is plausible from what you could not check, and give the strongest version "
            "of a disagreement rather than the easiest one."
        ),
        constraints=(
            "Do not present an unverified claim without labelling it.",
            "Do not treat a claim as verified because it appears in several places that share one source.",
        ),
        specialty="research",
    ),
    Skill(
        name="data-visualization",
        description="Chooses the chart that answers the question being asked.",
        instructions=(
            "Pick the form from the question: comparison, distribution, change over time, or "
            "relationship. Label the axes and units, say what the baseline is, and state what the "
            "chart deliberately leaves out."
        ),
        constraints=(
            "Do not truncate a value axis without saying so on the chart.",
            "Do not use colour as the only carrier of meaning.",
        ),
        specialty="analysis",
    ),
    Skill(
        name="incident-postmortem",
        description="Writes up an incident so the next one is less likely.",
        instructions=(
            "Give the timeline, the impact in terms a user would recognise, and the contributing "
            "causes - plural. Focus on what made the failure possible and what made it hard to "
            "see, and end with actions someone owns."
        ),
        constraints=(
            "Do not name an individual as a cause.",
            "Do not stop at the first cause that explains the symptom.",
        ),
    ),
    Skill(
        name="prompt-engineering",
        description="Writes instructions for models that hold up on the awkward cases.",
        instructions=(
            "State the task, the output shape, and the failure modes explicitly. Prefer showing an "
            "example over describing one, and test the wording against the inputs most likely to "
            "confuse it rather than the ones that read nicely."
        ),
        constraints=("Do not claim a prompt works without saying what you tried it against.",),
    ),
    Skill(
        name="translation",
        description="Carries meaning and register across languages, not just words.",
        instructions=(
            "Translate for meaning and register, not word by word. Keep names, units and formatting "
            "conventions appropriate to the target locale, and flag anything - idiom, pun, legal "
            "term - that does not survive the crossing intact."
        ),
        constraints=("Do not silently smooth over a phrase you could not translate faithfully - flag it.",),
        specialty="writing",
    ),
)

SKILLS = PresetRegistry("skill", Skill, BUILTIN_SKILLS)


__all__ = ["Skill", "BUILTIN_SKILLS", "SKILLS"]
