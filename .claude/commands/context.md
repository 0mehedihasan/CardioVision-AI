---
description: Load the project context — what CardioVision AI is, what exists, and what does not.
---

Read the project context before doing anything else, then summarise it back.

1. Read `.claude/memory/project.md` and `.claude/memory/current-state.md`.
2. Read `.claude/memory/models.md` and `.claude/memory/architecture.md`.
3. Skim `.claude/rules/` — five files: `project.md`, `medical-ai.md`,
   `code-quality.md`, `data-and-privacy.md`, `git-and-models.md`.
4. Check the context against the repository for anything that has drifted:
   `src/cardiovision/config.py` is the source of truth for constants and metrics,
   and `MODALITY_STATUS` is the source of truth for what exists.

Then report, in under 20 lines:

- the three trained models and the one language model
- what has **no** model behind it (learned fusion, clinical risk)
- the strict layering rule and why it is load-bearing
- anything in `.claude/` that no longer matches the code

Do not restate the whole context. Do not start implementing anything.
