# Verification status

What has been executed, what has only been inspected, and what has not been
checked at all. This file exists so that "the tests pass" is never mistaken for
"the model works".

---

## How to run it

```bash
pytest -q                                  # the documented wrapper
python3 tests/test_case_lifecycle.py       # or any suite directly
```

Every suite is an **executable script**, not a pytest module. Each stubs whatever
part of the ML stack is missing, prints a per-assertion report, and exits
non-zero on the first failure — so a red run names the broken invariant in the log
rather than in a traceback, and every suite runs on a machine where torch was
never installed.

`tests/test_all.py` is the pytest entry point. `[tool.pytest.ini_options]
python_files` in `pyproject.toml` points only at it, because pytest would
otherwise import the suites as modules and execute their top-level assertions
during collection — where the architecture suite's deliberate `sys.exit(0)` on an
unresolved LFS pointer reads as a collection error rather than as the skip it is.

`tests/torch_stub.py` constructs the models without torch: enough of `torch.nn`
to build the 1-D ECG ResNet, the 2-D echo encoder and the 3-D CCTA U-Net and to
inspect parameter names and shapes. Every member of `torch.nn.functional`, and
every numeric entry point (`from_numpy`, `sigmoid`, `load`, `tensor`, …), **raises
loudly** rather than returning something plausible. A stub that quietly returned
zeros would let a suite report a green forward pass that never happened.

---

## The suites

761 checks across seven suites. `tests/test_all.py` orders them cheapest-first,
so a broken shared foundation surfaces before four suites fail on top of it.

| Suite | Checks | What it executes |
| --- | --- | --- |
| `test_case_lifecycle.py` | 207 | a real temporary SQLite file: schema and migrations, 500 unique case IDs with no collision, age derivation at the awkward edges, PNG magic numbers on the bytes actually written, path-traversal rejection on the image route, search, cascade delete, token expiry, sliding renewal, login lockout |
| `test_report_evidence.py` | 134 | the evidence layer and report assembly over fixtures shaped exactly like router responses: status vocabulary, `available_modalities` ordering, typed uncertainties, the full report key set, the recommendation kinds, and a forbidden-phrase sweep |
| `test_ccta_pipeline.py` | 108 | format detection by magic bytes, resample geometry, HU windowing, window-start offsets, the window budget and partial coverage, quantification, the finding dict, slice selection, the model card, declared status, unloaded refusal, the input summary |
| `test_ecg_pipeline.py` | 114 | signal loading, WFDB formats 16 and 212 and declared byte offsets, resampling, filtering, normalisation, lead-order reporting |
| `test_ecg_rendering.py` | 100 | the 12-lead waveform figure with saliency shading |
| `test_ecg_reporting.py` | 53 | the operating point, weak-class flagging, and the honesty rules in the response |
| `test_ecg_architecture.py` | 45 | the constructed architecture against the checkpoint's own parameter names and shapes |
| **Total** | **761** | |

Each suite prints an honest footer: when the torch stub was installed, it says so
and states that no forward pass ran and that the inputs were synthetic — the same
honesty rule the application code follows, applied to the tests.

### Notable invariants pinned

- **CCTA resampling direction.** 256 voxels at 0.5 mm is 128 mm of tissue, which
  is 128 voxels at 1 mm. Resampling to a coarser grid **reduces** the count.
  Getting this backwards would scale every reported volume by 8×.
- **`available_modalities` order.** `ccta`, `echo`, `ecg` — the
  `EVIDENCE_MODALITIES` order, not alphabetical (`'ecg' < 'echo'`).
- **CCTA metrics are dicts.** `{mean, sd, min, max}` per metric, asserted for
  each. A bare float over three cases reads as a stable estimate.
- **Unticked risk factors are unknown.** The uncertainty says
  "UNKNOWN, not absent … left blank", never "not reported".
- **No ejection fraction as a value.** EF may appear only inside echo
  `limitations` and an uncertainty detail; it must never appear inside a finding's
  `measurement`.
- **Forbidden phrases.** `consistent with`, `confirms`, `corroborates`,
  `suggestive of`, `risk score` and `rules out` are asserted absent from every
  cross-modal observation. Co-occurrence only.
- **Presence thresholds.** 50 pixels for echo, 500 voxels for CCTA, surfaced
  rather than hidden.
- **`max_probability`** is over the whole probability map, including voxels
  outside the mask.
- **Volume arithmetic.** 1000 voxels of 1 mm³ = 1000 mm³ = 1 mL.
- **A WFDB byte offset is skipped, not decoded.** A header declaring `16+24` — the
  PTB-XL packaging the ECG model was trained from — puts a 24-byte MATLAB header
  before the samples. Reading it as signal does not fail: it yields one fabricated
  sample per lead (up to 124 mV, ~100× physiological) and drops the last real one,
  which the robust normalisation then scales everything else against.

---

## What CI does and does not prove

`.github/workflows/ci.yml` runs, on Python 3.10 / 3.11 / 3.12:

1. `ruff check src tests`
2. each suite as a script, in cheapest-first order
3. `pytest -q` — the documented wrapper, so it cannot rot
4. an installed-package import check (`pip install --no-deps -e .`, then import
   `cardiovision` and `cardiovision.config`) — this catches a module that resolves
   in a checkout but not in a wheel

and, in a separate job, `npm ci`, `npm run lint` (oxlint) and `npm run build`
(vite) — the only place the JSX is actually parsed and compiled.

Checkout runs with `lfs: false`. The architecture suite detects an unresolved
pointer file and **skips with a message** instead of failing, so no run burns LFS
bandwidth on a 46 MB checkpoint.

> [!IMPORTANT]
> A green tick means the wiring, the schema, the prompt construction, the figure
> rendering and the architecture-vs-checkpoint comparison hold.
>
> **It does not mean a forward pass ran.** Numerical model behaviour has to be
> verified on a machine with the real stack installed.

---

## Not verified

Stated plainly rather than implied to be covered.

| Gap | Consequence | How to close it |
| --- | --- | --- |
| **No forward pass has been executed** in the verification environment; torch is absent | segmentation and classification *numbers* are unverified there — only the arithmetic around them is checked | install the real stack, start the backend, and upload a frame, a volume and a record |
| **No checkpoint has been loaded** | the loaders themselves are unexercised beyond the architecture comparison against parameter names and shapes | `git lfs pull`, then `GET /api/models/{echo,ccta,ecg}` |
| **No HTTP request has been executed**; FastAPI, pydantic and uvicorn are absent and uninstallable there | the routers are verified against the store, auth and fusion modules they call, not over the wire | run the server and exercise the endpoints, or add an httpx/TestClient suite once the dependency is available |
| **SciPy and nibabel are absent** | CCTA resampling, connected-component counting and the ECG bandpass arithmetic are unexecuted; the CCTA suite covers the *no-SciPy* branch instead | run the suites in the real environment |
| **`oxlint` and `vite build` cannot run** in the sandbox — the npm registry is blocked and `node_modules` holds only darwin-arm64 bindings with no JS parser | frontend files are bracket-balanced but not lint-checked or compiled locally; **CI does compile them** | `cd frontend && npm ci && npm run lint && npm run build` |
| **git-lfs is absent** in the sandbox | every `.pt`/`.pth` reads as modified; never `git add -A` or `git commit -a` there | `git lfs install` |
| **MedGemma output is not asserted** | narrative text is not a testable invariant. What *is* tested is the prompt: which fields it contains, which it withholds, and that the structured report stands without it | use `?include_prompt=true` and read it |
| **No clinical validation, no prospective study, no reader study** | none of the three models has an established clinical performance characteristic | out of scope for this repository |

---

## Sample inputs, and what a run on them proves

Two cases per modality are tracked under `samples/` — CAMUS, MedHK23/CCA and PTB-XL,
38 files, 181 MB — so the acceptance pass below needs no downloads.
`samples/README.md` documents them; `LICENSE` records the attribution each dataset
requires.

They were originally committed inside `data/cases/`, which is `CASE_FILES_DIR` — the
directory the case store writes real patient files into — and were moved out with
`git mv`, so history is intact. `data/` is now runtime state only:

```bash
git ls-files data/          # only data/.gitignore
```

> [!IMPORTANT]
> `samples/ccta/CCTA_CASE_01{4,5}` are **test-split** cases, and two of the same three
> the published CCTA Dice and the 0.60 threshold were measured on. Running them is a
> reproduction, not an independent evaluation. Which split the echo and ECG samples
> came from is not recorded in this repository, so they are not held out either.
>
> No suite reads `samples/`. They are for the manual pass, and a comparison against
> the shipped ground truth on two cases is an anecdote, not a metric.

---

## Manual acceptance pass

The one step no suite replaces. On a machine with the real stack:

```bash
git lfs pull
pip install -e .
cardiovision serve --port 8000
# second terminal
cd frontend && npm ci && npm run dev
```

Then, signed in as the operator account:

1. `GET /api/health` — all three models `loaded`, `MODALITY_STATUS` as expected.
2. Upload an echo frame. Check the reported device, the orientation note, and that
   areas are cm² for NIfTI/DICOM and percent-of-field for PNG/JPEG. Rotate 90° and
   confirm total field area is unchanged.
3. Upload a CCTA volume. Confirm `n=3` travels with the Dice, that the coverage
   note appears if the window budget was short, and that the Grad-CAM panel is a
   real overlay rather than a flat map.
4. Upload a 12-lead record. Confirm every probability is returned, that the
   operating point is stated, and that a positive `HYP` carries its weak-class
   note.
5. `POST /api/report?include_prompt=true`. Read the prompt: no patient name, no
   MRN, unticked risk factors described as unknown, weak models stated.
6. Save the case, reload the app, reopen it. Confirm the re-run control is hidden
   on a restored case and that no finding count is stale in the sidebar.
7. Restart the backend and confirm the old token is rejected.

