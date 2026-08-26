# Rules — working in this repository

These are constraints, not suggestions. They exist because this repository has
already been through several rounds where the alternative was tried.

## 1. Inspect before changing

Read the module before editing it. `src/cardiovision/config.py` in particular
carries provenance comments naming the notebook cell that produced each number;
those comments are the only record of where a constant came from.

Concretely, before changing anything:

- read the file you are editing, in full, not just the function
- check `config.py` for a constant that already covers what you were about to
  hardcode
- check `.claude/memory/current-state.md` for whether the thing you are "fixing"
  is a known deliberate choice
- for a metric, find its `SOURCE OF TRUTH` comment before touching the value

## 2. Do not invent functionality

If a capability does not exist, no file may imply that it does.

| Do not | Instead |
| --- | --- |
| Add a "risk score" | there is no risk model; `MODALITY_STATUS["clinical"]["available"]` is `False` |
| Add a learned fusion output | `fusion/` is deterministic software; say so |
| Fill a missing model output with a plausible number | return the status that says it was not analysed |
| Add an endpoint for a model that does not exist | add nothing; the UI reads `/api/health` |
| Document a Docker/cloud/CI-deploy setup | none exists; see `memory/deployment.md` |

`MODALITY_STATUS` in `config.py` is the single source of truth for what is real.
The frontend reads it through `/api/health` so the UI cannot advertise a
capability the backend lacks. Change it there, never in two places.

## 3. Preserve the existing architecture

The layering in `memory/architecture.md` is load-bearing, not stylistic:

- `config.py` must not `import torch` at module scope
- `preprocessing/` must not import `inference/`
- `rendering/` must not import torch
- `fusion/` must not import a model
- `api/` routers validate, dispatch and serialise — they do not compute

Breaking any of these makes the test suite unrunnable on a machine without torch,
which is most machines, including CI.

## 4. Avoid unnecessary refactoring

Do not restructure the repository. Do not rename modules for tidiness. Do not
introduce an abstraction layer whose only justification is a hypothetical future
modality or the broader OpenMedImaging platform vision.

**CardioVision AI is not OpenMedImaging.** It is cardiovascular only: three
modalities, one local operator account, one SQLite case store. There is no plugin
system, no modality registry, no multi-tenancy, no remote inference — and adding
one is out of scope here.

Every new file must have a real purpose. Do not create a file to make the
structure look industrial.

## 5. Do not retrain, ever, unless explicitly asked

The three checkpoints are fixed pretrained artefacts. Do not retrain them, do not
add a training pipeline, do not fine-tune, and do not replace a checkpoint with
one you produced. The notebooks in `notebooks/` are the training *record*; they
are documentation, not a build step.

## 6. Do not hardcode absolute paths

No `/Users/...`, no `/kaggle/...`, no `/root/...`, no `/content/...`. Use
`config.PROJECT_ROOT` and the paths derived from it, or an environment variable.
`CARDIOVISION_HOME` overrides the root.

## 7. Preserve the honesty behaviour

Every constraint listed in `rules/medical-ai.md` is already implemented and
already asserted by a test. Do not relax one to make an output look cleaner.

## 8. Keep this context in sync

Repository code and configuration are the source of truth. When you change what
the repository does, update the `.claude` file that describes it in the same
change. A stale context file is worse than no context file, because it will be
believed.
