# Architecture

How the pieces fit, which direction dependencies are allowed to point, and where
to put a new modality. For what the models are, see [`models.md`](models.md);
for endpoint contracts, [`api.md`](api.md).

---

## Layers

```
uploaded bytes
      │
      ▼
preprocessing/     decode, validate, resample, normalise  →  arrays + provenance
      │
      ▼
inference/         load checkpoint, forward pass, saliency →  findings + metrics
      │
      ├──────────────▶ rendering/    arrays → PNG / SVG bytes
      │
      ▼
fusion/            three analyse responses → structured evidence → report
      │
      ▼
analysis.py        one pipeline per modality: decode → forward → saliency →
      │            render → response payload → archive.  No HTTP in it.
      ├────────────────────────────────┐
      ▼                                ▼
api/               FastAPI routers     streamlit_app.py   in-process client
      │            HTTP, auth,                            (one file)
      ▼            serialisation
frontend/src/      React 19 + Vite
```

`services/` sits beside these rather than inside them: auth, the SQLite case
store, and prompt-context construction are used by the routers, not by the model
code.

`analysis.py` is the seam that makes two UI clients possible without two
implementations. Both clients call the same functions and get the same payload;
`AnalysisError` carries the status code, so the API keeps its 413/415/422/503
distinctions while a client that speaks no HTTP still gets the message.

`config.py` is underneath everything and imports nothing from the project.

## Dependency rules

These are load-bearing, not stylistic. Breaking one makes the test suite
unrunnable on a machine without torch, which is most machines.

| Rule | Reason |
| --- | --- |
| `config.py` must not `import torch` at module scope | it is read by the renderers, the case store, the context builder and every test, none of which touch a tensor. A top-level import would take the whole API down at import time on a machine without torch, instead of at the point a model is needed. `select_device()` imports inside the function and sets `TORCH_AVAILABLE`. |
| `preprocessing/` must not import `inference/` | preprocessing is verifiable arithmetic; keeping it model-free is what lets the suites check resample geometry and HU windowing with numpy alone |
| `rendering/` must not import torch | renderers take arrays. They are checked against real `.npy` artefacts in `models/`, with no forward pass |
| `fusion/` must not import a model | there is no fusion model. `fusion/` reads finished analyse responses; the only model it touches is MedGemma, and only for narrative text over evidence already computed |
| `api/` must not compute | routers validate, dispatch, serialise and handle errors. Arithmetic in a router is arithmetic no suite can reach |
| `analysis.py` must not import `api/` or raise `HTTPException` | it is the shared core. The moment it knows what a request is, the Streamlit client needs a FastAPI install to run a forward pass |
| A UI client must not reimplement a pipeline | `frontend/` and `streamlit_app.py` format and display. Duplicated medical serialisation is how two clients start disagreeing about the same case |
| Nothing may hardcode an absolute path | `PROJECT_ROOT` walks up from the installed package looking for `pyproject.toml` or `models/`; `CARDIOVISION_HOME` overrides it |

---

## Module map

| Module | Responsibility |
| --- | --- |
| `config.py` | paths, device selection, every model constant, every published metric, `MODALITY_STATUS`, upload limits, allowed suffixes |
| `cli.py` | `cardiovision serve`; maps `--skip <name>` onto the `CARDIOVISION_SKIP_*` variables the app reads at startup |
| `preprocessing/image_io.py` | echo: PNG/JPEG/NIfTI/DICOM decode, frame selection, rotation and mirror, pixel-spacing bookkeeping |
| `preprocessing/ccta_io.py` | CCTA: format detection by magic bytes, 1 mm resample, HU window, `LoadedVolume` provenance |
| `preprocessing/ecg_io.py` | ECG: WFDB/CSV/NPY/JSON load, 500→100 Hz resample, bandpass, per-lead robust normalisation, lead-order reporting |
| `inference/echo.py` | UNet++ load, forward pass, per-structure quantification, input-gradient saliency |
| `inference/ccta.py` | Small3DUNet load, sliding-window inference with a window budget, Grad-CAM, `_quantify` |
| `inference/ecg.py` | ECGResNet1D load, multi-label sigmoid at threshold 0.5, per-lead attribution, weak-class flagging |
| `inference/medgemma.py` | local generation for Q&A and report narrative |
| `rendering/primitives.py` | shared PNG encoding, colour maps, overlay compositing |
| `rendering/echo.py` | original / mask / overlay / saliency / combined frames |
| `rendering/ccta.py` | slice selection (`_best_index`), slice views, Grad-CAM overlay |
| `rendering/ecg.py` | 12-lead waveform with saliency shading |
| `fusion/schema.py` | the status vocabulary, `CrossModalObservation`, `Uncertainty`, `EVIDENCE_MODALITIES` |
| `fusion/evidence.py` | per-modality evidence, clinical normalisation, cross-modal observations, uncertainties, recommendations |
| `fusion/report.py` | report schema assembly and the MedGemma report prompt |
| `services/auth.py` | fixed operator account, salted hash, constant-time compare, in-memory sessions, lockout |
| `services/database.py` | SQLite case store, migrations, denormalised list columns, image files |
| `services/case_context.py` | the text block handed to MedGemma; withholds name and MRN |
| `analysis.py` | `analyze_echo` / `analyze_ccta` / `analyze_ecg` and the `ensure_*_model` gates — the shared core both UI clients call |
| `api/app.py` | app construction, CORS, lifespan model loading |
| `api/deps.py` | `require_session` and shared dependencies |
| `api/schemas.py` | request/response models |
| `api/routers/*` | `auth`, `health`, `echo`, `ccta`, `ecg`, `qa`, `report`, `cases` |

---

## Request paths

### An analysis

```
POST /api/analyze/{echo|ccta|ecg}
  1. require_session          bearer token or 401
  2. size + suffix check      streamed, checked as it arrives
  3. preprocessing            bytes -> array, with provenance recorded
  4. model lock acquired      FastAPI runs sync endpoints in a threadpool
  5. forward pass             timed INSIDE the lock
  6. saliency                 or omitted entirely if the gradient is unavailable
  7. quantification           against the presence threshold for that modality
  8. rendering                arrays -> base64 PNGs
  9. persistence              written to the case if case_id was passed
```

Timing is measured inside the lock so a request that queued behind another does
not report its waiting time as compute time.

Steps 1 and 2 are the router's; steps 3 to 9 are `analysis.py::analyze_<name>`.
`streamlit_app.py` enters at step 3 with bytes it read from disk, which is why an
uploaded case and a sample case produce the same payload rather than two
lookalikes.

### A report

```
POST /api/report
  1. resolve the case         from the request body, or from storage by case_id
  2. build_case_evidence()    deterministic; no model
  3. build_report_prompt()    the exact text, returnable via ?include_prompt
  4. MedGemma narrative       skippable via ?include_summary=false
  5. assemble                 fixed schema, schema_version "1.0"
  6. optionally save          ?save=true
```

Step 2 completes with or without a language model, and the structured report is
byte-identical either way. Only `ai_summary` depends on MedGemma; when it fails,
`ai_summary_error` is populated and the rest of the report still stands.

### Model loading

Each model loads independently in the app lifespan, so a MedGemma failure does
not take segmentation down. A skipped or failed model is reported unavailable by
`/api/health`, and its panel in the UI says so.

Device preference is MPS → CUDA → CPU. If a saliency backward pass is unsupported
on MPS, the entire forward pass for that request falls back to CPU and the
response reports `fell back from mps`.

---

## Adding a modality

In order, because each step depends on the last:

1. **`config.py`** — architecture constants, preprocessing constants, the real
   held-out metrics with their provenance comment, the presence threshold, the
   checkpoint path, allowed suffixes, upload limit, and a `MODALITY_STATUS`
   entry. If there are no weights, `available: False` and a `note` saying so.
2. **`preprocessing/<name>_io.py`** — decode and normalise, returning the array
   plus enough provenance for the response to say where the numbers came from.
3. **`inference/<name>.py`** — a loader that fails with a message naming the file
   it tried to open, a forward pass, quantification against the threshold, and
   a `describe()` model card built from `config`.
4. **`rendering/<name>.py`** — arrays to PNG, reusing `primitives.py`.
5. **`fusion/evidence.py`** — a `_<name>_evidence` function, plus the modality in
   `EVIDENCE_MODALITIES`. Decide explicitly which keys gate `analysed`.
6. **`analysis.py`** — an `analyze_<name>` assembling the response payload and an
   `ensure_<name>_model` gate. This is the only place the pipeline exists.
7. **`api/routers/<name>.py`** — `GET /api/models/<name>` and
   `POST /api/analyze/<name>`, both thin: read the upload, call `analyze_<name>`,
   map `AnalysisError` through `as_http_error`. Register in `app.py`.
8. **`services/database.py`** — a migration adding the columns, plus the
   denormalised list columns. Move all columns for a modality together or not at
   all.
9. **`tests/test_<name>_*.py`** — arithmetic and geometry that runs without
   torch, registered in `tests/test_all.py` and as a CI step.
10. **`frontend/src/components/<Name>Result.jsx`** — and read
   `MODALITY_STATUS` through `/api/health` rather than assuming availability.
11. **`streamlit_app.py`** — a section calling the same `analyze_<name>`. Display
   only; if you find yourself computing there, step 6 is incomplete.

Do not skip step 1. Constants discovered later end up duplicated in two files
with different values, which is how a threshold silently changes.

