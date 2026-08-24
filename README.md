# CardioVision AI

A locally deployed cardiovascular AI platform. Everything runs on your own
machine — no image, no clinical value and no question ever leaves it.

One imaging model is trained and serving predictions today: a UNet++ /
EfficientNet-B3 network that segments four cardiac structures from
echocardiography. Alongside it, MedGemma 1.5 4B IT answers case-level clinical
questions locally, and can be given the real segmentation output as context.

Access is behind a sign-in, and every study is filed as a patient case record in
a local SQLite database, so a completed analysis survives closing the tab.

The CCTA, clinical-risk and multimodal-fusion pipelines are not trained yet.
Their notebooks are still empty, so the interface says "Model not yet trained"
for those modalities rather than showing a number. Nothing in this codebase
invents a result it does not have.

---

## What is actually implemented

The echocardiography path is real end to end. You upload a frame, the backend
decodes it, runs the trained checkpoint, and returns per-structure areas, an
attribution map, and the raw class mask.

| Modality | Status |
| --- | --- |
| Echocardiography | Trained. 4-class segmentation, serving. |
| Clinical data | Collected in the UI, passed to MedGemma as text context. No risk model. |
| CCTA | Not trained. `notebooks/01_CCTA_Training.ipynb` is empty. |
| ECG | No pipeline exists. |
| Multimodal fusion | Not trained. `notebooks/04_Multimodal_Fusion.ipynb` is empty. |

`backend/config.py` holds `MODALITY_STATUS` as the single source of truth. The
frontend reads it through `/api/health`, so the UI cannot advertise a
capability the backend does not have. Flip `available` to `True` there as each
pipeline lands.

---

## Signing in

A single fixed operator account, since this is a one-workstation tool:

```
username: medexpert
password: 1111
```

Both are overridable without touching code, which is what you should do before
this sits on anything but your own laptop:

```bash
CARDIOVISION_USER=someone CARDIOVISION_PASSWORD='a real password' \
  uvicorn main:app --reload --port 8000
```

The check is enforced on the server, not in the browser. Every route that
touches a model or a patient record requires a bearer token, so opening the API
directly gets you a 401 rather than the data. Tokens are 32 random bytes from
`secrets`, held in memory only, and expire eight hours after their last use
— restarting the backend signs everyone out. The password is never stored in
plaintext or compared with `==`: it is salted with a per-process random value,
hashed, and checked with `hmac.compare_digest`, and both the username and
password are compared unconditionally so a wrong username takes exactly as long
as a wrong password. Five failed attempts lock the account for five minutes.

The browser keeps its token in `sessionStorage`, not `localStorage`, so it dies
with the tab. A 401 from any request anywhere in the app returns you to the
login screen with a reason instead of failing silently.

---

## Case records

Each study is a case row in `data/cardiovision.db`, with its rendered PNGs and
the source upload written under `data/cases/<case-id>/`. Both are excluded by
`data/*` in the root `.gitignore`, and the backend writes a second `.gitignore`
inside `data/` at startup so a fresh clone is protected before anyone thinks
about it. **The database is not encrypted** — do not put it in a synced folder.

A case holds the patient's name, medical record number, date of birth, sex,
study date, referring clinician and free-text notes, alongside the clinical
form, the full segmentation payload, the image files and the MedGemma
transcript. Age is derived from the date of birth on every read rather than
stored, because an age typed in once is wrong a year later.

The name and MRN are deliberately withheld from the MedGemma prompt. Age, sex,
study date and notes are sent, because a clinical answer can use them; an
identifier cannot.

Saving happens on its own in the places where losing work would hurt: a case
row is created the moment you start an analysis, so the backend has somewhere to
file the upload, and the result is written as soon as it lands. Each answer from
MedGemma is folded into the record too, but only when a case already exists —
asking a general cardiology question with no patient entered does not silently
create one. Everything else is the explicit Save button, and the header warns
before you discard unsaved changes by starting or opening another case.

If the database cannot be opened, the UI says so and names the reason rather
than dropping saves quietly, and analysis still works — you simply get no
record of it.

---

## The echo model

UNet++ with a `timm-efficientnet-b3` encoder, one input channel, four output
classes, 256×256 input. Trained on CAMUS: 500 patients and 2000 image-mask
pairs, split at **patient** level into 350/75/75 patients (1400/300/300 pairs).
Patient disjointness was asserted in the notebook, so the test numbers carry no
leakage. Training ran on a Kaggle Tesla T4; early stopping fired at epoch 38 and
the best checkpoint is epoch 30.

Held-out test performance is Dice 0.9044 and IoU 0.8282, with per-class test
Dice of 0.9379 for the LV cavity, 0.8759 for myocardium and 0.8994 for the left
atrium. These are dataset-level figures describing the model overall — the UI
labels them as such, because they are not a confidence score for any individual
segmentation.

Validation Dice and IoU are deliberately **not** hardcoded anywhere. They are
read back out of the checkpoint at load time, so they can never drift away from
the weights they describe. They are also displayed separately from the test
metrics, since the validation split steered training through early stopping and
checkpoint selection and is therefore not an independent estimate.

### Explainability

The saliency map is an input-gradient attribution: the absolute gradient of the
mean LV-cavity probability with respect to the input image, matching
`TARGET_CLASS = 1` in the training notebook. It is not Grad-CAM, and the UI does
not call it that. If the gradient cannot be computed the saliency tabs are
hidden entirely rather than filled with an empty heatmap — an all-zero gradient
still renders as a smooth, convincing-looking picture, which is worse than
showing nothing.

### Orientation — read this before judging a result

The model learned on CAMUS NIfTI arrays exactly as stored, which puts the
ultrasound sector's **apex on the left** with the beam opening rightward. You
can confirm this in the notebook's own saved figure,
`models/echo/__results___0_129.png`.

Conventional echo displays are apex-**up**. That is a quarter turn away from the
training distribution, so an exported PNG, JPEG or DICOM frame is out of
distribution until it is rotated.

The backend never guesses. It reports the mismatch, flags the result as
provisional in the MedGemma prompt, and offers 0°/90°/180°/270° rotation plus a
horizontal mirror in the UI. Each change re-runs the model on the original file.
A silent auto-rotation would leave no way to distinguish a real segmentation
from a lucky one.

### Supported inputs

PNG and JPEG work but carry no pixel spacing, so areas come back as a percentage
of the image field rather than in cm². NIfTI and DICOM carry spacing, so those
give real areas in cm². Multi-frame DICOM cine loops are handled — pass
`?frame=N` to pick a frame. On a quarter turn the row and column spacing are
swapped along with the pixels, so the cm² figure stays correct; total field area
is rotation-invariant, which the test suite verifies.

---

## Running it

You need the model weights in place first:

```
models/echo/cardiovision_echo_unetplusplus_best.pth
models/medgemma-1.5-4b-it/          (~8.6 GB of safetensors)
```

Backend:

```bash
cd backend
python3 -m venv .venv && source .venv/bin/activate
pip install -r ../requirements.txt
uvicorn main:app --reload --port 8000
```

Frontend, in a second terminal:

```bash
cd frontend
npm install
npm run dev
```

Then open http://localhost:5173 and sign in as `medexpert` / `1111`.

The two models load independently, so a MedGemma failure no longer takes echo
segmentation down with it. While iterating on the imaging pipeline you can skip
the 8.6 GB language model entirely:

```bash
CARDIOVISION_SKIP_MEDGEMMA=1 uvicorn main:app --reload --port 8000
```

`CARDIOVISION_SKIP_ECHO=1` does the same for the segmentation model. Device
selection prefers MPS, then CUDA, then CPU. If the saliency backward pass is
unsupported on MPS the whole forward pass falls back to CPU for that request
and the response says so — the UI shows "fell back from mps" rather than
quietly reporting the wrong device.

---

## API

```
GET    /                              service banner
GET    /api/health                    load state + which modalities are real
POST   /api/auth/login                username + password -> bearer token
GET    /api/auth/session              validate the token held by the browser
POST   /api/auth/logout               revoke it
GET    /api/models/echo               echo model card (architecture + metrics)
POST   /api/analyze/echo              segmentation from an uploaded image
POST   /api/clinical-question         MedGemma Q&A, optionally with case context
GET    /api/cases                     case summaries, newest first, ?search=
POST   /api/cases                     create or update a case
GET    /api/cases/{id}                one full case
DELETE /api/cases/{id}                remove it, with its images
GET    /api/cases/{id}/images/{name}  a stored PNG
```

Everything except `/`, `/api/health`, `/api/auth/login` and `/api/auth/logout`
needs `Authorization: Bearer <token>`. Logout is open on purpose: signing out
with an already-expired token should quietly succeed rather than 401.

`POST /api/analyze/echo` accepts `frame`, `rotate` (0/90/180/270, CCW), `flip`
and `include_mask`. It returns server-rendered base64 PNGs (original, mask,
overlay, saliency, saliency overlay, combined) **and** the raw class mask as a
flat row-major array with its colour and name maps, so the frontend can draw its
own canvas with per-class visibility toggles. Pass `case_id` and the source
upload is archived under that case.

A stored case does not carry its images inline — `GET /api/cases/{id}` returns
image *endpoints* instead, because inlining six base64 PNGs would make every
case fetch several megabytes. The frontend requests them with the same bearer
token and turns them into blob URLs; the token never appears in a URL, so it
cannot end up in uvicorn's access log.

Uploads are capped at 200 MB, streamed and checked as they arrive, since DICOM
cine loops are large.

---

## Honesty constraints built into the code

These are deliberate and worth preserving if you extend the project.

A structure is only reported present once at least 50 mask pixels carry its
label; below that it is argmax noise. That threshold is surfaced through the API
and explained in the UI, so "Not identified" can never be misread as a clinical
statement that the structure is absent.

An unticked risk-factor checkbox is treated as **unknown**, not as a denial. The
form ships those fields defaulting to false, so `false` means the clinician never
touched the box — it does not mean the patient was asked and said no. An earlier
version emitted these to MedGemma as "not reported", which handed the model a
negative history nobody had ever taken and which it would then repeat back as
fact. The prompt now says explicitly: treat these as unknown, do not describe
the patient as having denied them.

Inference timing is measured inside the model lock, so a request that queued
behind another one does not report its waiting time as compute time. FastAPI
runs sync endpoints in a threadpool, hence the lock.

Dataset-level accuracy is never rendered next to a single prediction without a
label saying which it is.

A restored case cannot pretend to be a live one. Reopening a stored study shows
its findings and images, but the orientation re-run control is hidden, because
the original file is on disk under the case and not in the browser — offering a
re-analysis button that cannot re-analyse would be a lie about what the app can
do at that moment. Restored MedGemma answers likewise do not offer a "show case
context" toggle, since the context that produced them was not preserved.

---

## Verification status

The torch-free code paths were executed for real against genuine notebook
arrays, not just inspected: the image loader across all four rotations, spacing
swaps and rejection of invalid angles; the case-context builder across seven
scenarios; the renderers against `models/echo/xai_prediction_mask.npy` and
`xai_lv_cavity_saliency.npy`; and the case store and auth layer end to end
against a real SQLite file — schema, 500 unique case IDs with no collision, age
derivation at the awkward edges, PNG magic numbers on the bytes actually
written, path-traversal rejection on the image route, search, cascade delete,
token expiry, sliding renewal and the login lockout.

On top of that, all nine backend modules compile, the API response shape was
cross-checked in both directions against every field the frontend reads, the
stylesheet parses cleanly (508 rules) with every one of the 196 referenced
`cv-` classes having a rule, and all eight JSX files are structurally balanced.

The case-lifecycle half of that is checked in and runnable, against a temporary
database so it never touches real records:

```bash
cd backend && python3 test_case_lifecycle.py     # 98 checks, no torch needed
```

That suite caught a real bug worth recording: the case list reads
denormalised `structures_found` and `echo_filename` columns so listing never has
to parse JSON, and the upsert was protecting `echo_json` with `COALESCE` but
overwriting those two unconditionally. Editing a patient's name on an analysed
case would have left the sidebar advertising "Echo · 0/3" with a blank filename
while the segmentation sat intact one click away. All four echo columns now move
together or not at all.

What could **not** be checked in the sandbox: actual model inference, because
torch is not installed there and the checkpoint wants your Mac's MPS device; the
HTTP layer, because FastAPI could not be installed either, so the endpoints were
verified against the store and auth modules they call rather than over the wire;
and `oxlint` / `vite build`, whose native binaries were installed for macOS.
Run those on your machine — start the two dev servers above, sign in, and upload
a frame. That is the one step still outstanding.
