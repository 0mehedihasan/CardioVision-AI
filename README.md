<div align="center">

# CardioVision AI

**A locally deployed cardiovascular AI workstation — echo, CCTA and ECG, with a deterministic cross-modal evidence layer.**

[![CI](https://github.com/0mehedihasan/CardioVision-AI/actions/workflows/ci.yml/badge.svg)](https://github.com/0mehedihasan/CardioVision-AI/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue)](pyproject.toml)
[![Node](https://img.shields.io/badge/node-22-green)](frontend/package.json)
[![License: MIT](https://img.shields.io/badge/license-MIT-yellow)](LICENSE)
[![Checks](https://img.shields.io/badge/verification%20checks-761-brightgreen)](tests/)
[![Not for clinical use](https://img.shields.io/badge/status-research%20only-red)](#-not-for-clinical-use)

</div>

> [!CAUTION]
> **Not for clinical use.** No regulatory clearance, no prospective validation, no
> clinical trial. Three trained models, all evaluated on small held-out splits —
> the CCTA split is **three cases**. Every output is a research artefact for a
> qualified clinician to read, not a diagnosis.

---

## Contents

- [About CardioVision AI](#-about-cardiovision-ai)
- [Current capabilities](#-current-capabilities)
- [Not for clinical use](#-not-for-clinical-use)
- [Models](#-models)
- [Quick start](#-quick-start)
- [Configuration](#-configuration)
- [API](#-api)
- [Technology stack](#-technology-stack)
- [Architecture](#-architecture)
- [Testing](#-testing)
- [Honesty constraints](#-honesty-constraints)
- [Project status](#-project-status)
- [Documentation](#-documentation)
- [Citation](#-citation)
- [Developer](#-developer)
- [License](#-license)

---

## 🫀 About CardioVision AI

CardioVision AI is a **research prototype**: a single-machine workstation that
runs three trained cardiovascular models over three modalities and files the
results as local case records.

| | |
| --- | --- |
| **Purpose** | Read echo, CCTA and 12-lead ECG with trained models, then present the output with its uncertainty attached |
| **Deployment** | Fully local — FastAPI backend, React frontend, SQLite case store, on-device language model |
| **Data flow** | Nothing leaves the machine. No cloud inference, no telemetry, no external API call at runtime |
| **Intended use** | Research, engineering and education. Every output is material for a qualified clinician to read |
| **Not** | A diagnostic device, a clinically validated system, or a multi-user hospital deployment |

Three properties are deliberate and are asserted by the test suite: a capability
that does not exist is reported as unavailable rather than approximated; a
structure below the presence threshold is reported as *not identified* rather
than absent; and a weak metric travels with the number it qualifies, in the API,
in the UI and in the language-model prompt.

---

## 🩻 Current capabilities

| Capability | State | Where |
| --- | --- | --- |
| Echo 4-class structure segmentation | ✅ Trained, serving | `POST /api/analyze/echo` |
| CCTA coronary lumen segmentation | ✅ Trained, serving | `POST /api/analyze/ccta` |
| 12-lead ECG 5-class screening | ✅ Trained, serving | `POST /api/analyze/ecg` |
| Saliency / Grad-CAM for all three | ✅ Real gradients | in each analyse response |
| Cross-modal evidence aggregation | ✅ Deterministic software, **no model** | `POST /api/evidence` |
| Integrated narrative report | ✅ MedGemma over real findings | `POST /api/report` |
| Clinical Q&A with case context | ✅ MedGemma, local | `POST /api/clinical-question` |
| Sign-in + SQLite case records | ✅ Local, unencrypted | `/api/auth/*`, `/api/cases/*` |
| Clinical **risk model** | ❌ Not trained | reported as unavailable |
| Learned multimodal **fusion** | ❌ Not trained | reported as unavailable |

Everything runs on one machine. No image, no signal, no clinical value and no
question leaves it.

`src/cardiovision/config.py::MODALITY_STATUS` is the single source of truth for
that table. The frontend reads it through `/api/health`, so the UI cannot
advertise a capability the backend does not have.

---

## ⚠️ Not for clinical use

Read this before interpreting any number the app produces.

- **No clearance, no validation.** Nothing here has been through a regulatory
  process or a prospective study.
- **The CCTA test split is three cases.** Dice `0.60 ± 0.12`, HD95 `82–131 mm`.
  A mean over three volumes is not a performance estimate; it is an anecdote
  with error bars.
- **Hypertrophy on ECG is unreliable.** Precision `0.361` at the shipped
  threshold — roughly two in three positive `HYP` calls are wrong. The API, the
  UI and the language-model prompt all say so rather than averaging it into the
  macro score.
- **Dataset metrics are not per-case confidence.** Every metric shipped is a
  dataset-level figure and is labelled as such wherever it is displayed.
- **There is no fusion model.** `src/cardiovision/fusion/` aggregates evidence
  deterministically. It never combines findings into a score, and never claims
  one modality corroborates another.
- **Echo measures anatomy, not function.** No ejection fraction, no strain, no
  stenosis grade, no calcium score.

---

## 🧠 Models

### Echocardiography — UNet++ / EfficientNet-B3

| | |
| --- | --- |
| Architecture | `UnetPlusPlus`, `timm-efficientnet-b3` encoder, 1×256×256 in, 4 classes out |
| Dataset | CAMUS — 500 patients, 2000 image/mask pairs |
| Split | **patient-level** 350 / 75 / 75 (1400 / 300 / 300 pairs), disjointness asserted |
| Test Dice / IoU | **0.9044** / **0.8282** |
| Per-class Dice | LV cavity 0.9379 · myocardium 0.8759 · left atrium 0.8994 |
| Best epoch | 30 (early stopping fired at 38), Tesla T4 |
| Saliency | input-gradient magnitude w.r.t. mean LV-cavity probability — **not** Grad-CAM |
| Checkpoint | `models/echo/cardiovision_echo_unetplusplus_best.pth` |

Validation Dice/IoU are deliberately **not** hardcoded — they are read back out
of the checkpoint at load time, so they cannot drift from the weights they
describe, and they are displayed apart from the test metrics because the
validation split steered early stopping.

> [!IMPORTANT]
> **Orientation.** The model learned CAMUS NIfTI arrays as stored: sector **apex
> left**, beam opening right. Conventional displays are apex-**up** — a quarter
> turn out of distribution. The backend never guesses: it reports the mismatch,
> marks the result provisional in the prompt, and offers 0/90/180/270° plus a
> mirror, re-running the model on the original file each time.

### CCTA — Small3DUNet

| | |
| --- | --- |
| Architecture | `Small3DUNet`, base 16 ch, 1 in / **1 logit per voxel** (sigmoid, not softmax), 1 401 265 params |
| Dataset | MedHK23/CCA — 20 volumes |
| Split | 14 train / 3 validation / **3 test**, at case level |
| Test Dice | **0.5996** (sd 0.1182, min 0.4929, max 0.7266) |
| Test HD95 | **109.51 mm** (sd 25.08, min 82.20, max 131.50) |
| Preprocessing | resample to 1 mm³, HU clipped `[-1000, 1000]` → `[-1, 1]`, pad `-1.0` |
| Inference | 96³ sliding window, 50 % overlap, threshold **0.60**, ≤600 windows |
| Saliency | Grad-CAM at `enc3.block[-1]` |
| Checkpoint | `models/ccta/best_3d_unet_cca_v2.pth` |

> [!WARNING]
> Every CCTA metric is reported as a **spread** (`mean`/`sd`/`min`/`max`), never
> as a bare float, because a single number over three cases reads as a stable
> estimate. Output is a lumen mask only: no stenosis grading, no calcium score,
> no vessel labelling.

### ECG — ECGResNet1D

| | |
| --- | --- |
| Architecture | 1-D residual CNN, 12 × 1000 in, 5 classes out, dropout 0.30, 3 884 165 params |
| Classes | `NORM` `MI` `STTC` `CD` `HYP` (multi-label sigmoid, threshold 0.5) |
| Dataset | PTB-XL — 21 837 records |
| Split | **patient-level** 14 957 / 3 199 / 3 232 records (13 031 / 2 793 / 2 793 patients) |
| Macro AUROC / AP / F1 | **0.9125** / 0.7804 / 0.7086 |
| Best epoch | 14 (validation macro AUROC 0.9203) |
| Preprocessing | resample 500 → 100 Hz, 0.5–40 Hz order-4 bandpass, per-lead robust median/IQR, clip ±10 |
| Saliency | per-lead gradient attribution, computed per request |
| Checkpoint | `models/ecg/cardioVision_ptbxl_ecg_resnet1d_full.pt` |

| Class | AUROC | AP | F1 | Precision | Recall | Prevalence |
| --- | --- | --- | --- | --- | --- | --- |
| NORM | 0.9498 | 0.9211 | 0.8585 | 0.8009 | 0.9251 | 0.4171 |
| MI | 0.9173 | 0.8294 | 0.7357 | 0.6865 | 0.7925 | 0.2565 |
| STTC | 0.9303 | 0.8126 | 0.7305 | 0.6166 | 0.8959 | 0.2438 |
| CD | 0.9329 | 0.8612 | 0.7668 | 0.7113 | 0.8318 | 0.2373 |
| **HYP** | **0.8323** | **0.4777** | **0.4516** | **0.3614** | 0.6020 | 0.1259 |

### MedGemma 1.5 4B IT

Local language model, `models/medgemma-1.5-4b-it/` (~8.6 GB of safetensors). It
answers case-level questions and writes the narrative section of the integrated
report. It is given the **real** model output as context; patient name and MRN
are withheld from every prompt, while age, sex, study date and notes are sent.

---

## 🚀 Quick start

### Requirements

- Python **3.10 / 3.11 / 3.12**
- Node **22**
- [Git LFS](https://git-lfs.com) — the checkpoints are LFS objects
- ~10 GB free for weights (MedGemma is 8.6 GB of it)

### 1. Clone and fetch the weights

```bash
git clone https://github.com/0mehedihasan/CardioVision-AI.git
cd CardioVision-AI
git lfs install
git lfs pull
```

Expected after `git lfs pull`:

```
models/echo/cardiovision_echo_unetplusplus_best.pth
models/ccta/best_3d_unet_cca_v2.pth
models/ecg/cardioVision_ptbxl_ecg_resnet1d_full.pt
models/medgemma-1.5-4b-it/                 # ~8.6 GB of safetensors
```

> [!WARNING]
> Never replace a missing checkpoint with a placeholder file of the same name.
> Preserve the exact filenames above — `config.py` resolves them literally, and
> `models/ccta/latest_3d_unet_cca_v2.pth` is a **later but worse** epoch that
> also carries a different selected threshold. See
> [`.claude/rules/git-and-models.md`](.claude/rules/git-and-models.md).

### 2. Backend

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e .
cardiovision serve --port 8000
```

Iterating on the imaging code? Skip the 8.6 GB language model:

```bash
cardiovision serve --skip medgemma            # repeat --skip for more
```

Check what actually loaded before blaming the app:

```bash
cardiovision check          # each checkpoint's size or absence, torch, resolved device
```

It separates "the checkpoint is missing" from "the server is broken" — which
otherwise both surface as a `503` in the UI. It does not start a server.

### 3. Frontend

```bash
cd frontend
npm ci
npm run dev
```

Open <http://localhost:5173> and sign in.

### 4. Sign in

```
username: medexpert
password: 1111
```

Both are overridable without touching code — do that before this sits on
anything but your own laptop:

```bash
CARDIOVISION_USER=someone CARDIOVISION_PASSWORD='a real password' cardiovision serve
```

The check is server-side. Every route that touches a model or a patient record
requires a bearer token, so hitting the API directly returns `401` rather than
data. Tokens are 32 random bytes from `secrets`, held in memory only, expiring
8 h after last use with a sliding renewal; restarting signs everyone out. The
password is salted with a per-process random value, hashed, and compared with
`hmac.compare_digest`; username and password are both compared unconditionally,
so a wrong username takes exactly as long as a wrong password. Five failures
lock the account for five minutes. The browser holds its token in
`sessionStorage`, so it dies with the tab.

> [!CAUTION]
> One shared password, no TLS, localhost binding, and an **unencrypted** SQLite
> database. These are documented limitations of a single-workstation research
> tool, not vulnerabilities. Do not bind to `0.0.0.0`, and do not put
> `data/cardiovision.db` in a synced folder.

### 5. First analysis

Two sample cases per modality ship with the repository, so there is nothing to
download before the first run:

| Modality | File to upload |
| --- | --- |
| Echo | `samples/echo/patient0001/patient0001_4CH_ED.nii.gz` |
| CCTA | `samples/ccta/CCTA_CASE_014/ccta_image.nii.gz` |
| ECG | `samples/ecg/HR00001.hea` — send `HR00001.mat` as the companion file |

See [`samples/README.md`](samples/README.md) for what each one is, its licence,
and which split it came from. Two of them are dataset test cases, which makes a
run on them a reproduction rather than an independent evaluation.

---

## 🔧 Configuration

Every value has a working default; nothing is required. See
[`.env.example`](.env.example) for the annotated template. There is no dotenv
loader on purpose — one fewer dependency, one fewer place a password can be
picked up silently.

| Variable | Default | Purpose |
| --- | --- | --- |
| `CARDIOVISION_USER` | `medexpert` | Operator username |
| `CARDIOVISION_PASSWORD` | `1111` | Operator password |
| `CARDIOVISION_HOME` | auto-detected | Where `models/` and `data/` live |
| `CARDIOVISION_CORS_ORIGINS` | `http://localhost:5173,http://127.0.0.1:5173` | Allowed origins; no wildcard |
| `CARDIOVISION_SKIP_ECHO` | unset | Keep the echo model out of memory |
| `CARDIOVISION_SKIP_CCTA` | unset | Keep the CCTA model out of memory |
| `CARDIOVISION_SKIP_ECG` | unset | Keep the ECG model out of memory |
| `CARDIOVISION_SKIP_MEDGEMMA` | unset | Keep MedGemma out of memory (saves ~8.6 GB) |
| `VITE_API_BASE_URL` | `http://127.0.0.1:8000` | Compiled into the bundle; set in `frontend/.env.local` |

Device selection prefers **MPS → CUDA → CPU**. If a saliency backward pass is
unsupported on MPS, that request's whole forward pass falls back to CPU and the
response says `fell back from mps` rather than misreporting the device. Each
model loads independently, so a MedGemma failure does not take segmentation down.

### Upload limits

| Path | Limit | Accepted |
| --- | --- | --- |
| Echo | 200 MB | PNG, JPEG, NIfTI, DICOM (`?frame=N` for cine) |
| CCTA | 800 MB, ≤200 M voxels | NIfTI (`.nii`, `.nii.gz`), DICOM series in a `.zip` |
| ECG | 200 MB | WFDB (`.hea`+`.dat`, or a `.zip` of both), `.csv`/`.txt`/`.tsv`, `.npy`, `.json` |

PNG and JPEG carry no pixel spacing, so echo areas come back as a percentage of
the image field. NIfTI and DICOM carry spacing, so those give real cm². On a
quarter turn, row and column spacing are swapped with the pixels, so cm² stays
correct — the test suite verifies it.

---

## 🔌 API

```
GET    /                                 service banner
GET    /api/health                       load state + which modalities are real

POST   /api/auth/login                   username + password -> bearer token
GET    /api/auth/session                 validate the browser's token
POST   /api/auth/logout                  revoke it

GET    /api/models/echo                  echo model card
GET    /api/models/ccta                  CCTA model card
GET    /api/models/ecg                   ECG model card

POST   /api/analyze/echo                 segmentation from an uploaded frame
POST   /api/analyze/ccta                 lumen segmentation from a volume
POST   /api/analyze/ecg                  5-class screening from a 12-lead record

POST   /api/evidence                     deterministic cross-modal evidence
POST   /api/report                       integrated report (?include_summary,
                                         ?include_prompt, ?save)
GET    /api/cases/{id}/report            a stored report
POST   /api/clinical-question            MedGemma Q&A, optionally with context

GET    /api/cases                        summaries, newest first, ?search=
POST   /api/cases                        create or update
GET    /api/cases/{id}                   one full case
DELETE /api/cases/{id}                   remove it, with its images
GET    /api/cases/{id}/images/{name}     a stored PNG
```

Everything except `/`, `/api/health`, `/api/auth/login` and `/api/auth/logout`
needs `Authorization: Bearer <token>`. Logout is open on purpose: signing out
with an already-expired token should quietly succeed rather than `401`.

Notable response shapes:

- **`POST /api/analyze/echo`** accepts `frame`, `rotate` (0/90/180/270 CCW),
  `flip`, `include_mask` and `case_id`. It returns server-rendered base64 PNGs
  (original, mask, overlay, saliency, saliency overlay, combined) **and** the raw
  class mask as a flat row-major array with its colour and name maps, so the
  frontend can draw its own canvas with per-class visibility toggles.
- **`POST /api/evidence`** involves no language model at all. Its
  `model_versions` values are objects, and its `fusion` entry states
  `"No learned fusion model exists in this project."`
- **`POST /api/report`** returns a fixed schema (`schema_version` `"1.0"`) with
  `ai_summary_scope`, `recommendations_scope` and a `disclaimer`. Pass
  `include_prompt=true` to get back the exact text MedGemma was given, so a
  reader can check the narrative claims nothing the evidence did not.
- **`GET /api/cases/{id}`** returns image *endpoints*, not inline base64 —
  inlining six PNGs would make every case fetch several megabytes. The frontend
  fetches them with the same bearer token and makes blob URLs, so the token never
  appears in a URL and cannot end up in uvicorn's access log.

Full field-by-field reference: [`docs/api.md`](docs/api.md).

---

## 🧰 Technology stack

Every version below is a lower bound from [`pyproject.toml`](pyproject.toml) or
[`frontend/package.json`](frontend/package.json), not a pin. Pinning torch in an
application whose point is running on whatever accelerator the operator has is a
portability bug, not a reproducibility feature.

| Layer | Used |
| --- | --- |
| **Backend** | Python ≥ 3.10 · FastAPI ≥ 0.115 · Uvicorn ≥ 0.30 · Pydantic ≥ 2.7 |
| **Deep learning** | PyTorch ≥ 2.3 · segmentation-models-pytorch ≥ 0.3.3 · timm ≥ 0.9.7 |
| **Language model** | MedGemma 4B-IT via Transformers ≥ 4.45 · accelerate · safetensors · sentencepiece — loaded from disk, run locally |
| **Medical I/O** | nibabel (NIfTI) · pydicom + pylibjpeg (DICOM, incl. compressed transfer syntaxes) · a WFDB reader written in-repo for PTB-XL |
| **Numerics & rendering** | NumPy ≥ 1.26 · SciPy ≥ 1.11 · Pillow ≥ 10.3 · Matplotlib — all rendering is server-side PNG/SVG |
| **Frontend** | React 19 · Vite 8 · plain CSS custom properties · oxlint — no UI framework, no component library, no state library |
| **Storage** | SQLite through the stdlib `sqlite3`; rendered images on disk under `data/cases/` — no ORM |
| **Weights** | Git LFS for the three served checkpoints; MedGemma downloaded separately and gitignored |
| **Tests & CI** | 7 executable suites (761 checks) runnable without torch via `tests/torch_stub.py` · pytest wrapper · GitHub Actions on 3.10/3.11/3.12 · ruff |

Deliberately absent: no Docker image, no ORM, no task queue, no logging
framework, no dotenv loader, no cloud deployment target. The SciPy and nibabel
paths keep explicit no-dependency branches so the pipeline degrades with a clear
message instead of failing obscurely.

---

## 🏗 Architecture

```
CardioVision-AI/
├── src/cardiovision/
│   ├── config.py            single source of truth: paths, device, constants, real metrics
│   ├── cli.py               `cardiovision serve`
│   ├── preprocessing/       uploaded bytes -> model input  (image_io, ccta_io, ecg_io)
│   ├── inference/           the models + saliency          (echo, ccta, ecg, medgemma)
│   ├── rendering/           server-side PNG/SVG            (echo, ccta, ecg, primitives)
│   ├── fusion/              deterministic evidence + report schema and prompt
│   ├── services/            auth, SQLite case store, prompt context
│   └── api/                 FastAPI app, deps, schemas, 8 routers
├── frontend/src/            React 19 + Vite; App.jsx, api.js, 9 components
├── notebooks/               01 CCTA · 02 Echo · 03 ECG · 04 Fusion (fusion is empty)
├── models/                  checkpoints (Git LFS) + training artefacts per modality
├── samples/                 2 sample cases per modality (CAMUS · MedHK23/CCA · PTB-XL)
├── tests/                   7 executable verification suites, 761 checks
├── docs/                    architecture · models · api · verification
└── .claude/                 project-local Claude Code context (committed)
```

Layering is strict and worth preserving: **preprocessing** never imports
inference, **rendering** never imports torch, **fusion** never imports a model,
and **config** never imports torch at module scope — which is why the renderers,
the case store and the whole test suite run on a machine where torch was never
installed.

### The evidence layer has no model

`fusion/` is deliberately not a network. It reads the three analyse responses and
emits structured observations under a fixed status vocabulary — `analysed`,
`not_provided`, `provided_not_analysed`, `no_model`, each with a stated meaning —
plus typed uncertainties. Its cross-modal observations always carry
`inference: "none"`, and the phrases *consistent with*, *confirms*,
*corroborates*, *suggestive of*, *rules out* and *risk score* are asserted absent
from its output by [`tests/test_report_evidence.py`](tests/test_report_evidence.py).

### Case records

Each study is a row in `data/cardiovision.db`, with rendered PNGs and the source
upload under `data/cases/<case-id>/`. The backend writes a `.gitignore` inside
`data/` at startup covering the database and `cases/`, so a fresh clone is protected
before anyone thinks about it. It only writes that file if it is absent, so an older
checkout may be missing a rule — `git ls-files data/` should list nothing but
`data/.gitignore`.

A case holds name, MRN, date of birth, sex, study date, referring clinician,
free-text notes, the clinical form, the full payload for each modality, the image
files and the MedGemma transcript. Age is **derived** from the date of birth on
every read rather than stored, because an age typed in once is wrong a year
later.

A case row is created the moment an analysis starts, so the backend has somewhere
to file the upload, and the result is written as soon as it lands. MedGemma
answers are folded in too, but only when a case already exists — asking a general
question with no patient entered does not silently create one.

---

## 🧪 Testing

```bash
pytest -q                                  # the documented wrapper
python3 tests/test_case_lifecycle.py       # or any suite directly
```

Each suite is an **executable script**, not a pytest module. That is deliberate:
each stubs whatever part of the ML stack is missing, prints a per-assertion
report, and exits non-zero on the first failure — so a red run names the broken
invariant in the log instead of in a traceback, and every suite runs on a machine
where torch was never installed. `tests/test_all.py` is the pytest entry point
and runs the same scripts as subprocesses.

| Suite | Checks | Covers |
| --- | --- | --- |
| `test_case_lifecycle.py` | 207 | SQLite schema, IDs, age edges, PNG bytes, path traversal, search, cascade delete, token expiry, lockout |
| `test_report_evidence.py` | 134 | evidence aggregation, status vocabulary, uncertainties, report schema, forbidden phrases |
| `test_ccta_pipeline.py` | 108 | format detection, resample geometry, HU windowing, window grid, budget, quantification, model card |
| `test_ecg_pipeline.py` | 114 | signal loading, WFDB formats 16/212 and byte offsets, resampling, filtering, normalisation, lead order |
| `test_ecg_rendering.py` | 100 | 12-lead waveform rendering with saliency |
| `test_ecg_reporting.py` | 53 | operating point, weak-class flagging, honesty rules |
| `test_ecg_architecture.py` | 45 | architecture vs checkpoint parameter names and shapes |
| **Total** | **761** | |

CI runs the whole set on Python 3.10/3.11/3.12 plus `ruff`, an installed-package
import check, `oxlint` and `vite build` — **without** LFS, so no run burns
bandwidth on a 46 MB checkpoint. A green tick therefore means the wiring, schema,
prompt construction, figure rendering and architecture-vs-checkpoint comparison
hold. **It does not mean a forward pass ran.**

> [!NOTE]
> What the sandbox could not execute is stated plainly in
> [`docs/verification.md`](docs/verification.md) rather than implied to be
> covered: no forward pass, no checkpoint load, no HTTP request, and no
> SciPy/nibabel numeric path.

---

## 🩺 Honesty constraints

These are deliberate. Preserve them if you extend the project.

| Constraint | Why |
| --- | --- |
| Echo structure reported present only above **50 mask pixels**; CCTA above **500 voxels** | below that it is argmax noise, and "Not identified" must never read as a clinical statement of absence |
| An unticked risk-factor box is **unknown**, not denied | the form defaults those fields to `false`, so `false` means the clinician never touched the box. An earlier version emitted "not reported", handing the model a negative history nobody had taken |
| Saliency hidden entirely when the gradient is unavailable | an all-zero gradient still renders as a smooth, convincing picture — worse than showing nothing |
| Input-gradient attribution is never called Grad-CAM | it is not Grad-CAM; the CCTA Grad-CAM is, and they are labelled separately |
| CCTA metrics are always `{mean, sd, min, max}` | a bare float over three cases reads as a stable estimate |
| `HYP` positives are flagged in the response, the UI and the prompt | precision 0.361 — it must never be averaged away into the macro AUROC |
| Timing measured **inside** the model lock | a request that queued behind another must not report its wait as compute time |
| Device fallback reported (`fell back from mps`) | quietly reporting the wrong device is a silent lie about provenance |
| A restored case hides the re-run control | the original file is on disk under the case, not in the browser; a button that cannot re-analyse would misstate what the app can do at that moment |
| Name and MRN withheld from every prompt | a clinical answer can use age and sex; it cannot use an identifier |
| No ejection fraction, anywhere, ever | the echo model outlines anatomy; it measures no function. EF appears only as a disclaimer, never as a value |

---

## 📋 Project status

**Active research prototype, version 4.0.0.** Implemented functionality is
listed in [Current capabilities](#-current-capabilities) and is verified by the
suites in [`tests/`](tests/). The table below is the honest boundary.

### Implemented and serving

| Area | Detail |
| --- | --- |
| Three trained models | Echo (UNet++/EfficientNet-B3), CCTA (Small3DUNet), ECG (1-D ResNet) — all loaded from checkpoints in `models/` |
| Explainability | Input-gradient attribution for echo, Grad-CAM for CCTA and ECG lead attribution — real gradients, hidden entirely when unavailable |
| Deterministic evidence layer | `fusion/` — structured cross-modal observations under a fixed status vocabulary, **no learned model** |
| Local narrative reporting | MedGemma over real findings, with the prompt inspectable via `?include_prompt=true` |
| Case management | Sign-in, SQLite records, rendered images on disk, MedGemma transcript per case |

### Not implemented

| Area | Status |
| --- | --- |
| Clinical risk scoring | **No model.** Fields are collected as context only; `MODALITY_STATUS["clinical"]` is `available: False` |
| Learned multimodal fusion | **No model.** `notebooks/04_Multimodal_Fusion.ipynb` is empty; `fusion/` is deterministic software |
| Training / fine-tuning code | Not in `src/`. The notebooks are the training *record*, not a build step |
| Multi-user access, roles, audit log | One shared local operator account |
| TLS, encryption at rest | Neither. Bind to localhost and keep the database off synced folders |
| DICOM cine frame selection in the UI | `frame` exists on the API; the browser does not expose a picker yet |
| Deployment tooling | No Dockerfile, no compose file, no cloud configuration |
| External validation | None. No prospective study, no reader study, no demographic performance breakdown |

### Known limitations in what *is* implemented

- **CCTA was evaluated on three cases.** Dice 0.60, HD95 82–131 mm. Three
  observations support no confidence interval; the mask is a contrast-density
  highlight to review, not a verified coronary tree.
- **ECG `HYP` is weak** — precision 0.361 at the 0.5 threshold, so roughly two in
  three positive calls are wrong. It is never reported as a standalone finding.
- **Echo reads one frame.** It outlines anatomy; it measures no function — no
  ejection fraction, no strain, no volumes over a cycle.
- **A case stores one analysis per modality.** Covering several echo views means
  separate cases.
- **Multimodal pairing is filing, not physiology.** An echo and an ECG in one
  case are two studies an operator filed together; cross-modal observations carry
  `inference: "none"` for exactly that reason.

---

## 📚 Documentation

| Document | Contents |
| --- | --- |
| [`docs/architecture.md`](docs/architecture.md) | layering rules, request paths, where to add a modality |
| [`docs/models.md`](docs/models.md) | full model cards, checkpoint contents, threshold provenance |
| [`docs/api.md`](docs/api.md) | every endpoint, parameter and response field |
| [`docs/verification.md`](docs/verification.md) | what is tested, what is not, and why |
| [`CONTRIBUTING.md`](CONTRIBUTING.md) | development setup, style, security reporting |
| [`.claude/README.md`](.claude/README.md) | project-local Claude Code context |

### Datasets

| Modality | Dataset | Note |
| --- | --- | --- |
| Echo | [CAMUS](https://www.creatis.insa-lyon.fr/Challenge/camus/) | 500 patients; respect the original licence |
| CCTA | MedHK23/CCA | 20 volumes |
| ECG | [PTB-XL](https://physionet.org/content/ptb-xl/) | 21 837 records; PhysioNet terms apply |

No training or validation split is redistributed here, and no patient data is
committed. What does ship is **two sample cases per modality** under
[`samples/`](samples/README.md), tracked on purpose so every upload path can be
exercised without fetching a dataset first — with the attribution each dataset
requires, and with a note on which split each came from. All three splits are at
**patient/case level**, and that must be preserved by anyone retraining.

---

## 🤝 Contributing

Read [`CONTRIBUTING.md`](CONTRIBUTING.md) first. The short version:

1. Inspect before changing — this codebase encodes a lot of hard-won provenance in comments.
2. Never weaken a test to make it pass. Fix the assertion to match the new truth; do not delete the check.
3. Do not invent a capability. If a model does not exist, `MODALITY_STATUS` says so and the UI must too.
4. Large weights go through Git LFS. Never commit a placeholder under a checkpoint's real filename.
5. Report security issues privately, not in a public issue.

---

## 📖 Citation

```bibtex
@software{hasan_cardiovision_ai,
  author  = {Hasan, Md. Mehedi},
  title   = {{CardioVision AI}: a locally deployed cardiovascular AI workstation},
  version = {4.0.0},
  license = {MIT},
  url     = {https://github.com/0mehedihasan/CardioVision-AI}
}
```

Machine-readable metadata: [`CITATION.cff`](CITATION.cff). There is no paper and
no DOI — citing this repository cites software, not a validated clinical result.

---

## 👤 Developer

**Md. Mehedi Hasan**
Software Developer & AI Engineer

| | |
| --- | --- |
| **Affiliation** | Department of Computer Science and Engineering, Bangladesh University of Business and Technology (BUBT), Dhaka, Bangladesh |
| **Research lab** | Advanced Machine Intelligence Research Lab (AMIR Lab) |
| **Technical areas** | Machine Learning · Deep Learning · Explainable AI · Medical Imaging · Medical AI · Bioinformatics · Graph Neural Networks |
| **GitHub** | [@0mehedihasan](https://github.com/0mehedihasan) |
| **Repository** | [0mehedihasan/CardioVision-AI](https://github.com/0mehedihasan/CardioVision-AI) |

CardioVision AI is developed and maintained as an independent research project.
It has no institutional endorsement, no clinical partner and no regulatory
status.

---

## 📄 License

[MIT](LICENSE). The licence grants no clinical warranty and confers no
regulatory status — read the notice at the bottom of the file.
