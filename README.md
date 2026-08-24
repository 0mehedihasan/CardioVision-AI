# CardioVision AI

A locally deployed cardiovascular AI platform. Everything runs on your own
machine — no image, no clinical value and no question ever leaves it.

One imaging model is trained and serving predictions today: a UNet++ /
EfficientNet-B3 network that segments four cardiac structures from
echocardiography. Alongside it, MedGemma 1.5 4B IT answers case-level clinical
questions locally, and can be given the real segmentation output as context.

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

Then open http://localhost:5173.

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
GET  /                        service banner
GET  /api/health              load state + which modalities are real
GET  /api/models/echo         echo model card (architecture + metrics)
POST /api/analyze/echo        segmentation from an uploaded image
POST /api/clinical-question   MedGemma Q&A, optionally with case context
```

`POST /api/analyze/echo` accepts `frame`, `rotate` (0/90/180/270, CCW), `flip`
and `include_mask`. It returns server-rendered base64 PNGs (original, mask,
overlay, saliency, saliency overlay, combined) **and** the raw class mask as a
flat row-major array with its colour and name maps, so the frontend can draw its
own canvas with per-class visibility toggles.

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

---

## Verification status

The torch-free code paths were executed for real against genuine notebook
arrays, not just inspected: the image loader across all four rotations, spacing
swaps and rejection of invalid angles; the case-context builder across seven
scenarios; the renderers against `models/echo/xai_prediction_mask.npy` and
`xai_lv_cavity_saliency.npy`. On top of that, all seven backend modules compile,
the API response shape was cross-checked in both directions against all 51
fields the frontend reads, the stylesheet parses cleanly (417 rules) with every
referenced `cv-` class having a rule, and all five JSX files are structurally
balanced.

What could **not** be checked in the sandbox: actual model inference, because
torch is not installed there and the checkpoint wants your Mac's MPS device;
and `oxlint` / `vite build`, whose native binaries were installed for macOS.
Run those on your machine — start the two dev servers above and upload a frame.
That is the one step still outstanding.
