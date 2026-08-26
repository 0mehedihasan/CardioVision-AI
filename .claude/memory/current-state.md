# Current state

A snapshot of what works, what is half-wired, what is untested and what does not
exist. Written 2026-08-26, updated 2026-08-27, against the working tree. The README
has since been rewritten against the code, so the two should now agree — where they
do not, **trust the code**.

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

**Test suites.** 7 executable suites, **761 checks**, registered in
`tests/test_all.py` and in CI.

**Two UI clients, one core.** `src/cardiovision/analysis.py` holds one pipeline per
modality — decode, forward pass, saliency, render, payload, archive — with no HTTP in
it. The FastAPI routers are thin wrappers over it (`AnalysisError` → `as_http_error`),
and `streamlit_app.py` calls the same functions in-process. No medical logic is
duplicated between the clients.

**Streamlit client.** One file, `streamlit_app.py`, `pip install -e ".[streamlit]"`,
seven sections (Dashboard, CCTA, Echocardiography, ECG, AI Assistant, Sample Cases,
About / Developer). No sign-in by requirement — a research and demonstration
surface, not a second application. Models load lazily behind `st.cache_resource`
and honour the same `CARDIOVISION_SKIP_*` flags. **It persists nothing**: no
`case_id` reaches `analyze_*`, `store.connect()` is never called, and results live
in `st.session_state` only — case management, patient records and the SQLite store
belong to the authenticated React client, which is untouched. MedGemma is
optional; `medgemma_state()` checks the gitignored weights directory and the
Dashboard and AI Assistant report its absence, with the deterministic report and
every measurement unaffected.

**Sample inputs.** `samples/` (`config.SAMPLES_DIR`) — two cases per modality from
CAMUS, MedHK23/CCA and PTB-XL, 181 MB tracked. The Streamlit Sample Cases section
runs them through the *same* `analyze_*` call as an upload; CAMUS `_gt` label maps
are shown as dataset annotation, never as a model output. An illustration, not an
evaluation set.

**Frontend.** Login gate, patient form, case list, echo / ECG / CCTA result views,
mask canvas, explainability view, case assistant and the integrated report.
`App.jsx` (3 125 lines) + `api.js` (700) + nine components in
`frontend/src/components/` (3 548).

---

## IN PROGRESS

Nothing is half-wired. The CCTA gap recorded here earlier — backend done, frontend
still rendering `<PendingModel>` and a hardcoded "No model" metric — is closed:
`api.js` exports `analyzeCcta`, `integratedEvidence` and `generateReport`, and
`navItems` has five sections ending in **Integrated report**.

`<PendingModel>` survives in exactly one place, `App.jsx:2978`, for the clinical-risk
form — which genuinely has no model. That is the component doing its job.

All seven suites pass (761 checks). The `test_case_lifecycle.py` failure recorded
here earlier — the assertion `"Coronary CT angiography: not available" in ecg_text`,
which encoded the pre-CCTA world — was fixed by correcting the assertion, not by
weakening it.

Still open, and small:

| Item | Why it is still open |
|---|---|
| Echo 256×256 resize note | The echo preprocessing squares a non-square frame, so a reported area inherits an anisotropic distortion that is not surfaced in the response |
| DICOM cine frame selector | `frame` exists on the API; the UI does not expose a picker, so a multi-frame loop always reads frame 0 from the browser |
| `cardiovision check` in the README quick start | The command exists and is the fastest way to see what loaded; the quick start does not mention it |

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
| `test_case_lifecycle.py` | 207 | pass | nothing — a real temporary SQLite file |
| `test_report_evidence.py` | 134 | pass | torch absent; the modality results fed in were fixtures, not model output |
| `test_ccta_pipeline.py` | 108 | pass | SciPy and nibabel absent → the **no-SciPy** resample branch ran; the masks fed in were synthetic |
| `test_ecg_pipeline.py` | 114 | pass | SciPy absent → `bandpass_filter` and `resample_ecg` ran against numpy stand-ins, so **the filter arithmetic itself is uncovered** |
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
- No deployment tooling: no Dockerfile, no compose file, no cloud config
- No user management, roles or audit log — one shared account
- No TLS, and no encryption at rest for the case database
- No DICOM cine frame selector — a multi-frame study uses a chosen frame
- No external validation, and no demographic performance breakdown

---

## Documentation state

`README.md` and `docs/` were rewritten against the code as it stands: `docs/architecture.md`
(layers and dependency rules), `docs/models.md` (every metric, copied from `config.py`),
`docs/api.md` (routes, parameters, status codes) and `docs/verification.md` (what the
761 checks prove and what they do not). Attribution was applied across `pyproject.toml`,
`LICENSE`, `CITATION.cff`, the issue templates and `frontend/package.json` (with
`package-lock.json` kept in sync so `npm ci` still resolves).

Two things to keep in mind rather than trust:

| Risk | Note |
|---|---|
| Drift | `src/cardiovision/config.py` is the source of truth for every constant and metric. If a documentation table disagrees with it, the table is stale and `config.py` wins |
| Verification claims | `docs/verification.md` is the honest record of what has and has not been executed. Do not upgrade a "not verified" row without running the thing |

---

## Git

134 tracked files at the time of writing, including all of `src/cardiovision/`, the
suites, `.github/` and six files under `.claude/memory/`. HEAD is `df14043`.

Uncommitted work exists in the tree — the documentation rewrite, the attribution
pass, and the rest of `.claude/`. **Nothing is committed automatically**; staging and
committing is the developer's decision.

> [!WARNING]
> Six `models/**` paths show as modified purely because git-lfs is not installed in
> this environment: a `.pt`/`.pth` reads as ~130 bytes of pointer text rather than
> weights. **Never run `git add -A` or `git commit -a` here** — it would commit
> pointer text over real weights. Stage named paths.

Verify before trusting these counts; they are a snapshot, and
`git status --short` is authoritative.
