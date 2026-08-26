# Current state

A snapshot of what works, what is half-wired, what is untested and what does not
exist. Written 2026-08-26 against the working tree, not against the README —
**the README is out of date and contradicts this file; trust the code.**

> **Notebook ≠ application.** `notebooks/` contains research pipelines that ran
> once on Kaggle against absolute paths that do not exist on any other machine.
> A capability demonstrated in a notebook is *not* a capability of the
> application. Below, "COMPLETED" means reachable through
> `src/cardiovision/`, not "shown in a notebook".

---

## COMPLETED

**Three trained models load and are served.**

| | |
|---|---|
| CCTA lumen segmentation | `inference/ccta.py`, `POST /api/analyze/ccta`, `GET /api/models/ccta` |
| Echo 4-class segmentation | `inference/echo.py`, `POST /api/analyze/echo`, `GET /api/models/echo` |
| ECG 5-class classification | `inference/ecg.py`, `POST /api/analyze/ecg`, `GET /api/models/ecg` |
| MedGemma narrative | `inference/medgemma.py`, used by `/api/report` and `/api/clinical-question` |

**Backend.** One FastAPI application, `src/cardiovision/api/app.py`, 8 routers,
**20 routes**. Models load independently in the lifespan; a failure costs exactly
one modality and is reported through `/api/health` rather than crashing startup.
`cardiovision serve` and `cardiovision check` both work as console scripts.

**Input handling.** PNG/JPEG, NIfTI and DICOM for echo; NIfTI and DICOM series
for CCTA; WFDB, CSV and NumPy for ECG. Size ceilings are enforced before decode
(`MAX_UPLOAD_BYTES`, `MAX_CCTA_UPLOAD_BYTES`, `MAX_CCTA_VOXELS`).

**Explainability.** 3-D Grad-CAM for CCTA, input-gradient saliency for echo and
ECG, each rendered server-side to PNG. When attribution cannot be computed the
API returns *nothing* rather than a zero map — a blank heat map reads as
"the model looked nowhere", which is a different claim.

**Authentication.** Single fixed account, opaque bearer tokens, 8-hour TTL,
`services/auth.py`. Everything except `/`, `/api/health` and `/api/auth/login`
requires a session.

**Case persistence.** SQLite at `data/cardiovision.db`, `services/database.py`.
Create, update, list, search, load, delete; stored renders served back per case;
additive-only schema migrations.

**Evidence aggregation.** `fusion/` builds a structured `CaseEvidence` from
whatever modalities were actually run — no model, no language model, fully
deterministic. `POST /api/evidence` answers correctly on a server where nothing
loaded.

**Report generation.** `POST /api/report` produces a structured report plus a
MedGemma narrative constrained by the evidence, with unavailable modalities and
partial coverage named explicitly in the prompt.

**Test suites.** 5 executable suites, **502 checks**, registered in
`tests/test_all.py` and in CI.

**Frontend.** Login gate, patient form, case list, echo view, ECG clinical view,
mask canvas, explainability view, case assistant. 12 files, ~10 100 lines.

---

## IN PROGRESS

**The CCTA frontend has not caught up with the CCTA backend.** This is the single
largest inconsistency in the repository right now.

| Layer | State |
|---|---|
| `inference/ccta.py`, `/api/analyze/ccta`, `/api/models/ccta` | Done |
| `frontend/src/api.js` | **No `analyzeCcta`.** Also no `integratedEvidence`, no `generateReport`. |
| `frontend/src/App.jsx` line ~1617 | CCTA tab renders `<PendingModel>` with the note *"There is no trained CCTA model yet, so nothing is inferred from CT data."* — **false** |
| `frontend/src/App.jsx` line ~1219 | `{ label: "CCTA", active: false, unavailable: true }` — hardcoded |
| `frontend/src/App.jsx` line ~2346 | `<Metric label="CCTA" value="No model" />` — hardcoded |
| `navItems` | 4 sections. There is no integrated-report section for `/api/report`. |

So `/api/evidence` and `/api/report` are implemented and **unreachable from the
UI**. A user cannot currently produce a CCTA reading or an integrated report
without calling the API directly.

**One test is failing, honestly.** `tests/test_case_lifecycle.py` — 203 of 204
checks pass; the failure is the check named *"but the untrained modalities still
are"*, which asserts `"Coronary CT angiography: not available" in ecg_text`. That
assertion encoded the old world in which CCTA had no model. It is now wrong.
**Fix the assertion to match the new truth; do not weaken the test and do not
delete the check.**

---

## EXISTS BUT NEEDS VERIFICATION

**No forward pass has ever been executed in this environment.** torch is not
installed here. Architecture, parameter counts, tensor shapes and checkpoint
contents were all verified by reading — `tests/checkpoint_reader.py` parses the
zip/pickle structure without torch — but no model has produced a prediction on
this machine. The reported metrics come from the checkpoints and the notebooks.
They have not been independently reproduced here.

**No HTTP request has ever been executed.** fastapi, pydantic and uvicorn are not
installed and cannot be installed here. Every route was verified by reading the
router modules. Route existence, method, auth dependency and response shape are
established from source; **runtime behaviour is not.**

**Suite-level stubbing, per the suites' own closing notes:**

| Suite | Checks | Result | What was stubbed |
|---|---|---|---|
| `test_case_lifecycle.py` | 204 | **203 pass, 1 fail** | — |
| `test_ecg_pipeline.py` | 100 | pass | SciPy absent → `bandpass_filter` and `resample_ecg` ran against numpy stand-ins, so **the filter arithmetic itself is uncovered** |
| `test_ecg_rendering.py` | 100 | pass | nothing — figures were generated and parsed for real |
| `test_ecg_reporting.py` | 53 | pass | torch absent; probabilities fed in were synthetic |
| `test_ecg_architecture.py` | 45 | pass | torch absent → `nn` stub; names, shapes and parameter count real, no forward pass |

**Also unverified here:** CCTA resampling (needs nibabel), DICOM reading (needs
pydicom), MedGemma loading (needs transformers, 8.6 GB of weights), frontend lint
and build (`oxlint` and `vite` cannot run — the npm registry is unreachable and
`node_modules` holds darwin-arm64 bindings only), and Git LFS state (git-lfs is
not installed, so every `.pt`/`.pth` reads as modified).

> **Consequence for anyone working here.** `git add -A` and `git commit -a` will
> stage 200+ MB of checkpoints as if they were edited. Stage explicit paths.

---

## NOT IMPLEMENTED

**Clinical risk scoring.** `MODALITY_STATUS["clinical"]` is `available: False`.
There is no model, no checkpoint and no notebook. The UI marks it unavailable.
Structured clinical fields *are* collected and *are* passed to the report prompt
as context — that is data entry, not a risk model.

**Multimodal fusion as a learned model.** `MODALITY_STATUS["fusion"]` is
`available: False`. `notebooks/04_Multimodal_Fusion.ipynb` is **0 bytes**.
`models/fusion/` is an **empty directory**. What exists under
`src/cardiovision/fusion/` is deterministic aggregation software — it combines
findings that other models produced and never learns or infers anything. Do not
describe it as a fusion model.

**Not present at all:**

- No training code in `src/` — the package serves models, it does not fit them
- No multimodal dataset, and no patient shared across the three cohorts
- No `docs/` content (the directory is empty, though `CONTRIBUTING.md` line 122
  advertises it)
- No deployment tooling: no Dockerfile, no compose file, no cloud config
- No user management, roles or audit log — one shared account
- No TLS, and no encryption at rest for the case database
- No DICOM cine frame selector — a multi-frame study uses a chosen frame
- No external validation, and no demographic performance breakdown

---

## Known-stale documentation

These files still describe an earlier state of the project. They are listed so
nobody treats them as authoritative, **not** as a to-do that this context file
authorises fixing.

| File | Stale claim |
|---|---|
| `README.md` | One trained model; "ECG \| No pipeline exists"; "CCTA \| Not trained"; references `backend/config.py` and `uvicorn main:app`, neither of which exists |
| `src/cardiovision/__init__.py` | Docstring: "Two models are trained and serving", CCTA untrained |
| `pyproject.toml` | Description omits CCTA; no `authors`; `urls` point at `github.com/mehedi/...` |
| `.github/ISSUE_TEMPLATE/config.yml` | Two `github.com/mehedi/...` URLs |
| `.github/ISSUE_TEMPLATE/feature_request.yml` | "Three of the four notebooks" |
| `LICENSE` | "Copyright (c) 2026 CardioVision AI contributors" — no named author |
| `frontend/package.json` | `"name": "frontend"`, `"version": "0.0.0"` |
| `frontend/src/App.jsx` | The CCTA `PendingModel` note and the two hardcoded "no model" strings above |

---

## Git

58 tracked files, 43 untracked entries, 22 showing as modified (most of that is
the LFS artefact described above). **Nothing in the recent work has been
committed.** HEAD is `8488f79` *"Add CardioVision CCTA model weights"*.

Most of `src/cardiovision/` is untracked: the whole of `fusion/`, `rendering/`,
`api/routers/`, `inference/ccta.py`, `inference/ecg.py`, `preprocessing/ccta_io.py`,
`preprocessing/ecg_io.py`, `services/__init__.py`, `cli.py`, `pyproject.toml`,
`LICENSE`, `CONTRIBUTING.md`, `.github/`, and four of the test files. A clone of
`origin` today would not contain a working application.
