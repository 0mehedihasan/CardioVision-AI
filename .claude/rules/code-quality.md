# Rules — code quality

## 1. Follow the existing package structure

Everything importable lives under `src/cardiovision/`. There is no `backend/`
directory and no top-level module; a `src/` layout with `pyproject.toml` is the
shape, and `pip install -e .` is how it is used.

```
src/cardiovision/
  config.py         every constant, every published metric, every path
  cli.py            cardiovision serve | check
  preprocessing/    uploaded bytes -> arrays + provenance
  inference/        checkpoints, forward passes, saliency
  rendering/        arrays -> PNG / SVG bytes
  fusion/           deterministic evidence + report schema and prompt
  services/         auth, SQLite case store, prompt context
  api/              app, deps, schemas, routers/
```

Put a new file in the layer that matches its job. If it does not fit a layer, that
is a signal to reconsider the design, not to add a layer.

## 2. Keep the layers separate

| Layer | May import | Must not |
| --- | --- | --- |
| `config` | stdlib only (torch **inside** `select_device()`) | import torch at module scope, import anything from the project |
| `preprocessing` | numpy, format libraries, `config` | import `inference` |
| `inference` | torch, numpy, `config`, `preprocessing` | render, or touch HTTP |
| `rendering` | numpy, PIL, matplotlib, `config` | import torch |
| `fusion` | `config`, `inference.medgemma` (narrative only) | import a segmentation or classification model |
| `services` | stdlib, sqlite3, `config` | import a model |
| `api` | everything above | contain arithmetic |

Arithmetic in a router is arithmetic no test suite can reach. Move it down a layer
and assert it there.

## 3. Reuse the existing utilities

Before writing a helper, check whether one exists:

| Need | Use |
| --- | --- |
| a constant, threshold, class name, colour, metric | `config.py` — do not redefine |
| PNG encoding, colour maps, overlay compositing | `rendering/primitives.py` |
| an upload read with a size limit | the shared `read_upload` used by the routers |
| session enforcement on a route | `api/deps.py::require_session` |
| the text block given to MedGemma | `services/case_context.py` |
| a status string for a modality | `fusion/schema.py` — the vocabulary is fixed |

Duplicating a constant is how a threshold silently changes in one place and not
the other.

## 4. Add tests for new logic

Every suite in `tests/` is an **executable script**, not a pytest module, and new
suites follow the same idiom. The shape, using `tests/test_ecg_reporting.py` as the
model:

```python
#!/usr/bin/env python3
"""
One line on what this covers, then the run command:

    python3 tests/test_<name>.py
"""
HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(HERE))

import torch_stub
STUBBED_TORCH = torch_stub.install()

FAILURES: list[str] = []
CHECKS = 0

def check(label, condition, detail=""): ...     # prints [PASS] / records failure
```

with numbered sections (`print("\n=== 1. ... ===")`) and a footer that prints the
stub NOTE when `STUBBED_TORCH`, then either the failure list and `sys.exit(1)` or
`ALL <n> CHECKS PASSED`.

Then register it in **both** places: the `SUITES` tuple in `tests/test_all.py`
(cheapest-first) and a step in `.github/workflows/ci.yml`.

Write checks that are exact arithmetic or pure geometry wherever possible, so they
run with numpy alone. `tests/torch_stub.py` constructs models without torch and
makes every numeric entry point **raise**; if your new code needs a stubbed layer,
add it to the stub in that spirit — never make the stub return a plausible number.

**Never weaken a test to make it pass.** If behaviour legitimately changed, fix the
assertion to match the new truth; do not delete the check and do not loosen it into
vacuity. Read the code to find its actual output rather than guessing at the
expected string.

## 5. Avoid unnecessary dependencies

The dependency list in `pyproject.toml` uses lower bounds, not pins — a pin on
torch in an application whose point is running on whatever accelerator the operator
has is a portability bug, not a reproducibility feature.

Before adding a dependency, ask whether numpy, PIL, matplotlib or the stdlib
already covers it. In particular:

- there is **no dotenv loader**, deliberately — one fewer place a password can be
  picked up silently
- SciPy and nibabel are used but the code has explicit **no-SciPy branches**;
  preserve them, they are what lets the pipeline degrade instead of failing
- do not add a web framework, ORM, task queue, or logging framework

If you must add one, pin it to an exact or lower-bounded version, prefer something
actively maintained, and say why in the same change.

## 6. Style

- `ruff check src tests` must pass; CI runs it on 3.10, 3.11 and 3.12
- `from __future__ import annotations` at the top of new modules
- type annotations on public functions
- explicit errors that name the file or value that failed — a loader must fail with
  the path it tried to open, not with a bare `FileNotFoundError`
- comments explain *why*, not *what*; the existing provenance comments are the
  house style and are worth matching
- no bare `except:`; catch what you can handle and let the rest surface
- tracebacks go to the server log, never to the client
