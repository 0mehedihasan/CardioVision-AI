# Contributing to CardioVision AI

Thanks for taking an interest. This document covers how to get the project
running, the one rule that matters more than the rest, and what a reviewable
change looks like here.

---

## The one rule

> **Never make the software claim more certainty than the model has.**

This is a clinical interface. Every number on screen has to be traceable to a
measurement, and every measured weakness has to stay visible at the point of
use. Concretely, that means a contribution must not:

- invent output for a modality that has no trained model (CCTA, clinical risk
  scoring and multimodal fusion are **empty notebooks** — the UI says so, and it
  keeps saying so until weights exist);
- report a macro average without the per-class numbers it hides (`HYP`
  precision is 0.361 — roughly two in three positive hypertrophy calls are
  wrong, against 0.83–0.92 AP for the other four classes);
- render a saliency map from a gradient that failed to compute (an all-zero map
  looks exactly like a real one);
- present "no class reached the threshold" as a normal ECG (`NORM` is one of the
  five independent outputs and was not called either);
- describe segmentation as diagnosis, or a dataset-level Dice score as
  per-case confidence.

If a change removes a caveat, the pull request needs to explain why the caveat
was wrong — not why it was inconvenient.

---

## Getting set up

```bash
git clone https://github.com/0mehedihasan/CardioVision-AI.git
cd CardioVision-AI

# The checkpoints are Git LFS objects. Without this they arrive as ~130-byte
# pointer files and the loader reports a corrupt checkpoint.
git lfs install && git lfs pull

python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

cp .env.example .env        # optional; every value has a default

cardiovision check          # what loaded, what did not, and why
cardiovision serve --skip medgemma
```

```bash
cd frontend
npm install
npm run dev                 # http://localhost:5173
```

Default sign-in is `medexpert` / `1111`. Override with `CARDIOVISION_USER` and
`CARDIOVISION_PASSWORD`.

### Working without the ML stack

`cardiovision serve --skip medgemma --skip echo --skip ecg` starts the API with
no weights loaded. `/api/health` reports each model as unavailable and the UI
shows that state, which is also the fastest way to work on the frontend.

---

## Running the checks

```bash
python tests/test_case_lifecycle.py     # storage, schema, prompt, frontend wiring
python tests/test_ecg_architecture.py   # module vs the real checkpoint, tensor by tensor
python tests/test_ecg_pipeline.py       # decode, filter, resample, normalise
python tests/test_ecg_rendering.py      # SVG strip and attribution chart
python tests/test_ecg_reporting.py      # thresholds, operating points, honesty rules

pytest                                  # all of the above
ruff check src tests
```

```bash
cd frontend
npm run lint
npm run build                           # the only thing that parses the JSX
```

**Read the last few lines of each suite.** They state what was actually
exercised and what was stubbed. The suites run without torch, without SciPy and
without the weights, which makes them fast and portable — and means a green run
is not evidence that a forward pass produced the right numbers.

| Suite | Runs for real | Stubbed or skipped |
|---|---|---|
| `test_case_lifecycle` | SQLite store, migrations, prompt text, static frontend assertions | HTTP layer (no FastAPI needed) |
| `test_ecg_architecture` | Checkpoint parsed without torch, shapes compared | Forward pass; skips if the LFS pointer is unresolved |
| `test_ecg_pipeline` | Decoding, lead matching, framing | Filter/resample maths when SciPy is absent |
| `test_ecg_rendering` | Full SVG generation, parsed back and asserted | nothing |
| `test_ecg_reporting` | Threshold logic, model card, context builder | torch (`tests/torch_stub.py`) |

If your change touches numerical behaviour, run it on a machine with torch and
SciPy installed and say so in the pull request.

---

## Project layout

```
src/cardiovision/          installable package
  api/                     FastAPI app, routers, deps, schemas
  inference/               echo, ecg, medgemma — one module per model
  preprocessing/           image_io (PNG/JPEG/NIfTI/DICOM), ecg_io (WFDB/CSV/NPY)
  rendering/               server-side SVG and PNG figures
  services/                auth, SQLite case store, prompt construction
  config.py                single source of truth for paths, classes, metrics
frontend/src/              React app (Vite)
tests/                     executable verification suites
notebooks/                 the training record — 02 and 03 produced the weights
models/                    checkpoints (LFS); ccta/ and fusion/ are empty
docs/                      architecture, models, API reference
```

### Where things belong

- **A new constant, class name, threshold or published metric** → `config.py`.
  Nowhere else. It is imported by the renderers, the store, the context builder
  and the tests, none of which may touch a tensor.
- **A new model** → a module in `inference/`, a router, an entry in
  `MODALITY_STATUS`, and a `--skip` flag. Loading must fail soft: one missing
  checkpoint costs exactly one modality.
- **Anything shown to a clinician** → the component that renders it also renders
  its caveat. Do not put the caveat only in a docstring.

---

## Style

**Python** — `ruff check src tests` must pass. 88 columns, except in the
renderers and `config.py`, where wide tables of constants stay readable
(`E501` is off). Type hints on anything public.

**JavaScript** — `npm run lint` must pass. Functional components, hooks, no
class components. No `localStorage` for anything session-shaped; the auth token
lives in `sessionStorage` so it dies with the tab.

**Comments explain _why_.** The codebase is deliberately heavy on rationale and
light on narration. `# increment the counter` is noise; a comment recording that
per-lead normalisation removes absolute voltage — and that this is why the model
cannot apply millimetre hypertrophy criteria — is the reason the next person
does not "fix" it.

**Commits** — imperative subject under ~72 characters, body explaining the
reasoning. Reference the notebook or the measurement when a number changes.

---

## Pull requests

1. Branch from `main`.
2. Make the change, with tests for anything that has an invariant worth keeping.
3. Run the suites and both frontend commands.
4. Open the PR with: what changed, why, what you ran, and **what you could not
   verify**. The last one is not a weakness in a PR here; it is the part
   reviewers rely on.

CI runs the Python suites on 3.10 / 3.11 / 3.12, `ruff`, `oxlint` and the Vite
build. It does not load weights or run a forward pass, for the reasons above.

---

## Reporting a problem

Open an issue with the model card values from the UI (they are printed on every
result panel), the input format, and what you expected instead. For a wrong
prediction, include the sampling rate and lead order — a scrambled lead order
classifies happily and silently, and it is the single most common cause of a
confident wrong answer.

**Do not attach patient data to an issue.** Not a de-identified export either.
Describe the shape of the input instead, or reproduce it with a synthetic
record.

### Security

Report anything security-relevant privately rather than in a public issue. Note
that the threat model here is narrow by design: one shared password, no TLS,
localhost binding. Those are documented limitations, not vulnerabilities — a
report that the app is insecure when bound to `0.0.0.0` will be closed with a
pointer to this line.

---

## Licence

Contributions are accepted under the [MIT licence](LICENSE). The dataset and
vendor-model terms noted at the bottom of that file are unaffected by it.
