# Architecture — as actually built

> Line counts and paths below were read from the tree. Re-run
> `wc -l $(find src -name '*.py')` if they look stale.

## Top level

```
CardioVision-AI/
├── src/cardiovision/     the application (installable package, 40 modules, 13 851 lines)
├── frontend/             Vite + React 19 single-page app
├── models/               weights and training artefacts (Git LFS)
├── notebooks/            the training record — NOT the application
├── tests/                executable verification suites
├── data/                 runtime case database + per-case files (gitignored)
├── samples/              two tracked sample inputs per modality (see rules/data-and-privacy)
├── docs/                 api.md, architecture.md, models.md, verification.md
├── streamlit_app.py       the second UI client — ONE file, by requirement
├── pyproject.toml         packaging, deps, ruff + pytest config
└── requirements.txt       contains only `-e .` — deps live in pyproject
```

There is **one** backend. `backend/` no longer exists; anything referring to
`backend/config.py` or `uvicorn main:app` is stale documentation.

## Package layout — `src/cardiovision/`

| Module | Lines | Responsibility |
| --- | --- | --- |
| `config.py` | 649 | **Single source of truth.** Paths, device, model constants, published metrics, `MODALITY_STATUS`, upload limits, allowed suffixes |
| `cli.py` | 192 | `cardiovision serve`, `cardiovision check` |
| `__main__.py` | 5 | `python -m cardiovision` |
| `preprocessing/` | 2 467 | `ccta_io.py`, `ecg_io.py`, `image_io.py` — uploaded file to model input |
| `inference/` | 2 632 | `ccta.py`, `echo.py`, `ecg.py`, `medgemma.py` — one model each |
| `rendering/` | 1 394 | `ccta.py`, `echo.py`, `ecg.py`, `primitives.py` — server-side PNG/SVG |
| `fusion/` | 1 969 | `evidence.py`, `report.py`, `schema.py` — deterministic aggregation |
| `services/` | 2 266 | `auth.py`, `database.py`, `case_context.py` |
| `analysis.py` | 665 | **The shared application core.** decode → forward pass → saliency → render → payload → archive, per modality, with no HTTP in it. `AnalysisError` carries the status code the API maps |
| `api/` | 1 984 | `app.py`, `deps.py`, `schemas.py`, `routers/` (8 routers) |

Import direction is one-way: `api` → `services`/`fusion`/`inference`/`rendering`
→ `preprocessing` → `config`. **`services/` never imports `api/`**, which is what
lets `tests/test_case_lifecycle.py` run without FastAPI installed.

`analysis.py` sits between them: above `preprocessing`/`inference`/`rendering`/
`services`, below `api`. It must not import `api` and must not raise
`HTTPException` — it is what lets `streamlit_app.py` run the identical pipeline
in-process while the routers keep their status codes. Two UI clients, one
implementation of each pipeline.

`config.py` imports torch **only inside `select_device()`**, so the renderers,
the case store, the context builder and the tests do not depend on the ML stack.

## Inference — the shared wrapper pattern

Every model module follows the same shape, and new ones must:

```python
class XSegmenter:
    is_loaded: bool          # property
    load_error: str | None   # why it is not loaded
    def load(self) -> None   # idempotent, threading.Lock guarded
    def describe(self) -> dict   # model card: architecture, metrics, limits
    def analyze(self, ...) -> XAnalysis   # structured result with .to_dict()

x_segmenter = XSegmenter()    # module-level singleton
```

- **Lazy.** Weights load on first use or at startup, never at import.
- **Independent.** `api/app.py::_load_model` catches the modality's own
  `*ModelUnavailable` as a warning and anything else with a traceback, so one
  bad checkpoint costs exactly one modality.
- **Skippable.** `CARDIOVISION_SKIP_CCTA|ECHO|ECG|MEDGEMMA` keep a model out of
  memory; `cardiovision serve --skip medgemma` sets them.

Public instances: `ccta_segmenter`, `echo_segmenter`, `ecg_classifier`,
`medgemma`.

## API surface — 20 routes

| Method | Path | Auth | Router |
| --- | --- | --- | --- |
| GET | `/` | public | `health.py` |
| GET | `/api/health` | public | `health.py` |
| POST | `/api/auth/login` | public | `auth.py` |
| POST | `/api/auth/logout` | session | `auth.py` |
| GET | `/api/auth/session` | session | `auth.py` |
| GET | `/api/models/ccta` | session | `ccta.py` |
| POST | `/api/analyze/ccta` | session | `ccta.py` |
| GET | `/api/models/echo` | session | `echo.py` |
| POST | `/api/analyze/echo` | session | `echo.py` |
| GET | `/api/models/ecg` | session | `ecg.py` |
| POST | `/api/analyze/ecg` | session | `ecg.py` |
| POST | `/api/evidence` | session | `report.py` |
| POST | `/api/report` | session | `report.py` |
| GET | `/api/cases/{id}/report` | session | `report.py` |
| POST | `/api/clinical-question` | session | `qa.py` |
| GET | `/api/cases` | session | `cases.py` |
| POST | `/api/cases` | session | `cases.py` |
| GET | `/api/cases/{id}` | session | `cases.py` |
| DELETE | `/api/cases/{id}` | session | `cases.py` |
| GET | `/api/cases/{id}/images/{name}` | session | `cases.py` |

Only the banner, `/api/health` and login are public — health stays public so the
login screen can report backend and model state before anyone signs in.

Shared dependencies live in `api/deps.py`: `require_session`, `require_store`,
`bearer_token`, `read_upload` (size-limited upload reader).

`routers/__init__.py` records the one ordering constraint: any literal path that
is a sibling of `/api/cases/{case_id}` must be registered before it, or the
literal is swallowed as an ID.

## Data flow — an analysis request

1. `POST /api/analyze/<modality>` with a multipart upload.
2. `deps.read_upload` enforces `MAX_UPLOAD_BYTES` (200 MB) or
   `MAX_CCTA_UPLOAD_BYTES` (800 MB) and the modality's allowed suffixes.
3. `preprocessing/*_io.py` decodes and normalises. **It never guesses** — a
   missing pixel spacing is reported missing, not assumed to be 1 mm.
4. `inference/*.py` runs the model under `no_grad`/`inference_mode` except where
   gradients are the explanation, and returns a dataclass.
5. `rendering/*.py` produces PNG/SVG figures as base64 data URLs. A render
   failure degrades to a note, never to a 500.
6. The router returns `analysis.to_dict()` plus figures, and optionally archives
   both against `case_id`.

## Evidence and report flow

`fusion/evidence.py::build_case_evidence(case)` takes the **client-held case
object** — the same shape the frontend has on screen — and returns a
`CaseEvidence` with `available_modalities`, `missing_modalities`, per-modality
status from a four-state vocabulary, `cross_modal` observations, and
`uncertainties`.

Because that input is client-supplied, every field is treated as untrusted:
`_mapping`, `_items` and `_number` coerce malformed values to "not supplied"
instead of raising, so one bad field cannot take down the request.

`fusion/report.py::build_report(evidence)` produces the report document, and
`build_report_prompt(evidence)` produces the `{context, question}` pair handed to
MedGemma. MedGemma receives **structured evidence only** and cannot override a
model output.

## Storage — `services/database.py`

- SQLite at `data/cardiovision.db`, images and uploads as files under
  `data/cases/`.
- Schema changes are **additive only**, applied through `_MIGRATIONS`.
- Per-modality column groups are guarded independently in the `ON CONFLICT`
  clause, so a metadata-only save cannot blank another modality's findings.
- `report_json` is written by a separate `save_report()` and is deliberately
  absent from `save()`'s conflict list, so a routine case save cannot overwrite
  a generated report.
- Stored figure keys are namespaced (`ccta_*`, `ecg_*`) via `_FIGURE_ALIASES`
  because the CCTA and echo renderers both emit an `overlay` key.

## Frontend — `frontend/src/`

| File | Lines | Role |
| --- | --- | --- |
| `App.jsx` | 2 618 | All state, four nav sections, the analysis tabs |
| `App.css` | 4 768 | Whole design system |
| `api.js` | 546 | Token-aware client; 401 drops to the login screen |
| `components/EchoResult.jsx` | 521 | Echo findings + figures |
| `components/EcgResult.jsx` | 687 | Full clinical ECG view |
| `components/CaseList.jsx` | 185 | Saved cases |
| `components/PatientForm.jsx` | 159 | Patient identity fields |
| `components/Login.jsx` | 226 | Sign-in + backend status |
| `components/MaskCanvas.jsx` | 175 | Client-side mask overlay |
| `components/PendingModel.jsx` | 48 | "Model not yet trained" panel |

Nav sections: `01 Patient case`, `02 Analysis`, `03 Explainability`,
`04 Case assistant`. Analysis tabs: overview, echo, ccta, ecg, clinical.

Tokens live in `sessionStorage`, not `localStorage` — a workstation left open
should not still be signed in tomorrow. Authenticated images are fetched as
blobs and wrapped in object URLs, because an `<img>` tag cannot send an
`Authorization` header and a token in the query string would land in the access
log.

## Tests — `tests/`

| File | Lines | What it verifies |
| --- | --- | --- |
| `test_case_lifecycle.py` | 1 213 | Case store round-trip against a real SQLite file, auth, plus static assertions on `App.jsx` |
| `test_ecg_rendering.py` | 738 | The SVG the API returns: geometry, mappings, caveats |
| `test_ecg_pipeline.py` | 696 | Four ECG readers, WFDB 16/212 and declared byte offsets, every rejection path |
| `test_report_evidence.py` | 688 | The evidence layer and report assembly, including the forbidden-phrase sweep |
| `test_ccta_pipeline.py` | 528 | Volume loading, resample geometry, windowing, quantification, model card |
| `test_ecg_architecture.py` | 358 | `ECGResNet1D` vs the checkpoint, parameter by parameter |
| `test_ecg_reporting.py` | 337 | Probability ordering, positives, caveats, lead ranking |
| `torch_stub.py` | 257 | Constructs modules without torch; computes nothing |
| `checkpoint_reader.py` | 182 | Reads a `.pt` zip without torch — shapes and metadata only |
| `test_all.py` | 71 | pytest wrapper: one test per suite, run as a subprocess |

The suites are **executable scripts**, not pytest modules, so they run on a
machine with no torch and print a per-assertion report.
`[tool.pytest.ini_options] python_files = ["test_all.py"]` exists so pytest does
not import them at collection time.

## Model organisation — `models/`

One directory per modality, holding the checkpoint plus the training artefacts
that document it. `models/` is data, not code: nothing in `src/` writes there.
See `memory/models.md` for exact filenames and sizes.
