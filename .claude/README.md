# `.claude/` — project context

Project-local context for Claude Code working in the CardioVision AI repository.
Committed to Git and shared: it describes **this repository**, not any developer.

> [!IMPORTANT]
> Repository code and configuration are the source of truth. The .claude context
> describes the repository and must be updated when the implementation changes.

A stale context file is worse than no context file, because it will be believed.

---

## Layout

| Path | Contents |
| --- | --- |
| [`memory/`](memory/) | what the project is, what exists, and what does not |
| [`rules/`](rules/) | constraints that hold across every change |
| [`skills/`](skills/) | working notes for the six workflows that actually exist |
| [`commands/`](commands/) | repeatable requests — context, status, test, review, models, test-cases |
| [`agents/`](agents/) | role descriptions scoped to real parts of this repository |

### `memory/`

| File | Contents |
| --- | --- |
| `project.md` | purpose, scope, and the line between CardioVision AI and the broader platform vision |
| `architecture.md` | layers, the dependency rules and why each is load-bearing |
| `models.md` | the three trained models, their real metrics, and what has no model |
| `data.md` | the three datasets, their splits and their licences |
| `current-state.md` | COMPLETED / IN PROGRESS / EXISTS BUT NEEDS VERIFICATION / NOT IMPLEMENTED |
| `decisions.md` | choices made and the reason; unrecorded ones say "Not established in repository." |
| `deployment.md` | full local, two processes, one CI workflow — and an inventory of what does not exist |

### `rules/`

| File | Covers |
| --- | --- |
| `project.md` | inspect before changing, do not invent functionality, preserve the architecture, no unnecessary refactoring, never retrain unasked, no absolute paths |
| `medical-ai.md` | research ≠ clinical validation, no diagnostic claims, preserve uncertainty, absence ≠ normality, no fabricated evidence, human in the loop |
| `code-quality.md` | package structure, layer separation, reuse, add tests, avoid dependencies |
| `data-and-privacy.md` | never commit patient data, dataset licences, patient-level splits, leakage, secrets |
| `git-and-models.md` | Git LFS, no fake placeholders, exact checkpoint filenames, verify `.gitignore`, do not auto-commit |

---

## Start here

1. [`memory/project.md`](memory/project.md) — what this is.
2. [`memory/current-state.md`](memory/current-state.md) — what is real today.
3. [`rules/medical-ai.md`](rules/medical-ai.md) — the constraints that matter most in
   a clinical UI.
4. The skill for the modality you are touching.

Or run [`/context`](commands/context.md).

---

## The three facts most easily got wrong

1. **Three models are trained** — echo (UNet++ / EfficientNet-B3), CCTA
   (Small3DUNet), ECG (ECGResNet1D) — plus MedGemma for narrative text. There is **no
   learned fusion model and no clinical-risk model**;
   `src/cardiovision/fusion/` is a deterministic evidence aggregator, and
   `MODALITY_STATUS` in `src/cardiovision/config.py` is the single source of truth for
   what exists.
2. **CardioVision AI is not the OpenMedImaging platform.** That is a separate, future,
   general vision. Do not merge the two concepts, and do not build toward it here.
3. **Every metric is a dataset-level figure on a narrow task.** None of the models has
   regulatory clearance or prospective validation. CCTA is n = 3. ECG `HYP` gets
   roughly two of three positive calls wrong.

---

## Keeping this honest

- Nothing here may contain secrets, API keys, tokens, passwords, personal developer
  memory, private conversation history or patient information.
- `.claude/local/` is gitignored for anything local or private. The tracked directory
  holds reusable project context only.
- When the implementation changes, update the affected context file in the same
  change. When a statement here cannot be verified against the repository, delete it
  rather than softening it.
- Related documentation lives in [`../docs/`](../docs/):
  [`architecture.md`](../docs/architecture.md), [`models.md`](../docs/models.md),
  [`api.md`](../docs/api.md), [`verification.md`](../docs/verification.md).
