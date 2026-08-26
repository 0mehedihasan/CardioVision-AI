---
name: testing
description: Works on the verification suites and CI — extends coverage, never weakens a check.
---

# Testing

## Responsibility

Keep the 761 checks meaningful and honest. Every suite is an **executable script**,
not a pytest module: it stubs whatever part of the ML stack is missing, prints a
per-assertion report, and exits non-zero on the first failure — so a red run names the
broken invariant in the log rather than in a traceback.

## Scope

`tests/` — `test_all.py` (the pytest entry point and the cheapest-first runner),
`torch_stub.py`, `checkpoint_reader.py`, and the seven suites; plus
`.github/workflows/ci.yml` and `[tool.pytest.ini_options]` in `pyproject.toml`.

| Suite | Checks |
| --- | --- |
| `test_case_lifecycle.py` | 207 |
| `test_report_evidence.py` | 134 |
| `test_ccta_pipeline.py` | 108 |
| `test_ecg_pipeline.py` | 114 |
| `test_ecg_rendering.py` | 100 |
| `test_ecg_reporting.py` | 53 |
| `test_ecg_architecture.py` | 45 |

## What to hold to

- **Never weaken a test to make it pass.** If behaviour changed on purpose, fix the
  assertion to match the new truth and keep the check. Never delete a check.
- A new suite is registered in **both** places: the `SUITES` tuple in
  `tests/test_all.py`, cheapest-first, and a step in `.github/workflows/ci.yml`.
- `python_files` in `pyproject.toml` points only at `test_all.py` on purpose. pytest
  would otherwise import the suites as modules and run their top-level assertions
  during collection, where the architecture suite's deliberate `sys.exit(0)` on an
  unresolved LFS pointer reads as a collection error rather than the skip it is.
- The torch stub must keep **raising loudly** from every numeric entry point
  (`from_numpy`, `sigmoid`, `load`, `tensor`, all of `torch.nn.functional`). A stub
  that returned zeros would let a suite report a green forward pass that never
  happened.
- Every suite prints an honest footer when the stub was installed: no forward pass
  ran, the inputs were synthetic.
- Test data is synthetic or a temporary file. Never a real study, never `data/`.
- Pin the invariants that fail silently: resample direction, `available_modalities`
  order (`ccta`, `echo`, `ecg` — not alphabetical), CCTA metrics as dicts, unticked
  risk factors as unknown, no EF as a value, the six forbidden phrases, the presence
  thresholds, `max_probability` over the whole map, volume arithmetic, path-traversal
  rejection.

## Verification

```bash
pytest -q
python3 tests/test_all.py
ruff check src tests
```

State what a green run does not prove: no forward pass, no checkpoint load, no HTTP
request, no SciPy or nibabel arithmetic. `docs/verification.md` holds the full gap
table; keep it current.

## Boundaries

Does not change application behaviour to make a test pass. Does not commit.
