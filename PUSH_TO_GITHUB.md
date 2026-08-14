# Pushing this overnight run to GitHub

**Why this file exists.** The plan was for the cloud session to `git push`
after every phase. It couldn't: the sandbox routes git through a proxy that
refuses to inject credentials for any repo outside the session's authorized
set, so `Shyam-Parikh-2025/llmadapt` was rejected with a 403 at the proxy — the
token you provided never even reached GitHub. Nothing was wrong with the token.

So the run fell back to the other option in the handoff doc: full git history
kept in the cloud clone, every phase committed there, and every changed file
synced to this machine after each phase via the device bridge. **The files in
this folder are already up to date** — this file is only about getting the
*commit history* onto GitHub.

Also worth knowing: the GitHub repo was several sessions behind this folder
when the run started (it had no `company.py`, `budget.py`, `observability.py`,
`selector.py` or `benchmark.py`). The first commit of the run brings it level.

---

## Option 1 — restore the history from the bundle (recommended)

`llmadapt-overnight.bundle` sits next to this file. It is a complete git
repository in one file: 9 commits, the whole run.

```bash
cd "C:\Users\HP\OneDrive\Coding Projects\Basics Genai"
git clone llmadapt\llmadapt-overnight.bundle llmadapt-pushed
cd llmadapt-pushed
git remote set-url origin https://github.com/Shyam-Parikh-2025/llmadapt.git
git push origin main
```

Check it first if you like — `git log --oneline` in that clone shows one commit
per phase, each with the reasoning in its message.

Once you're happy, `llmadapt-pushed` is the canonical copy; you can delete the
old folder or just keep working in the new one.

## Option 2 — commit this folder as one commit

If you don't care about the per-phase history and just want the code on GitHub:

```bash
cd "C:\Users\HP\OneDrive\Coding Projects\Basics Genai\llmadapt"
git init                       # only if this folder isn't a git repo yet
git remote add origin https://github.com/Shyam-Parikh-2025/llmadapt.git
git add -A
git commit -m "Phases 4-8: model policy, presets, delegation, compaction, company builder"
git push -u origin main
```

You lose the per-phase commit messages this way, but `company-roadmap.md` has
the same reasoning in more detail.

---

## Revoke the token

The fine-grained PAT you pasted into the chat should be revoked once you've
pushed: GitHub → Settings → Developer settings → Personal access tokens. It was
never used successfully (the proxy blocked it) and was not written to any file
in this repo — but it did appear in a chat transcript, so revoke it anyway.

## Verify before you push

```bash
cd "C:\Users\HP\OneDrive\Coding Projects\Basics Genai\llmadapt"
PYTHONPATH=src python3 tests/run_all.py
```

Expected: `SUITE PASSED: 14 files, 275 checks`.

On Windows PowerShell:

```powershell
$env:PYTHONPATH="src"; python tests\run_all.py
```
