# Model cards

Three trained models and one language model. Every number below is copied from
`src/cardiovision/config.py`, which is itself annotated with the notebook cell it
came from. If a number here and a number there disagree, `config.py` wins and
this file is stale.

> [!CAUTION]
> None of these models has regulatory clearance or prospective validation. Each
> metric is a **dataset-level** figure describing the model overall. It is not a
> confidence score for any individual prediction, and the UI labels it as such
> wherever it appears.

---

## Contents

- [Echocardiography — UNet++ / EfficientNet-B3](#echocardiography--unet--efficientnet-b3)
- [CCTA — Small3DUNet](#ccta--small3dunet)
- [ECG — ECGResNet1D](#ecg--ecgresnet1d)
- [MedGemma 1.5 4B IT](#medgemma-15-4b-it)
- [There is no fusion model](#there-is-no-fusion-model)
- [There is no clinical risk model](#there-is-no-clinical-risk-model)
- [Checkpoint hygiene](#checkpoint-hygiene)

---

## Echocardiography — UNet++ / EfficientNet-B3

| Field | Value |
| --- | --- |
| Architecture | `UnetPlusPlus` (`segmentation_models_pytorch`) |
| Encoder | `timm-efficientnet-b3` |
| Input | 1 × 256 × 256 |
| Output | 4 classes (0 background, 1 LV cavity, 2 myocardium, 3 left atrium) |
| Checkpoint | `models/echo/cardiovision_echo_unetplusplus_best.pth` |
| Source notebook | `notebooks/02_Echo_Training.ipynb` |
| Trained on | Tesla T4 (Kaggle) |

### Data and split

| Field | Value |
| --- | --- |
| Dataset | CAMUS |
| Size | 500 patients, 2000 image/mask pairs |
| Split | **patient-level** 350 / 75 / 75 → 1400 / 300 / 300 pairs |
| Leakage | patient disjointness asserted in the notebook |

### Held-out test metrics

| Metric | Value |
| --- | --- |
| Dice | **0.9044** |
| IoU | **0.8282** |
| Dice, LV cavity | 0.9379 |
| Dice, myocardium | 0.8759 |
| Dice, left atrium | 0.8994 |
| Early stopping | epoch 38 |
| Best checkpoint | epoch 30 |

Validation Dice and IoU are **not** hardcoded. They are read out of the
checkpoint at load time so they cannot drift from the weights, and they are
displayed apart from the test metrics because the validation split steered early
stopping and checkpoint selection — it is not an independent estimate.

### Preprocessing

| Step | Detail |
| --- | --- |
| Resize | 256 × 256 |
| Channels | 1 (grayscale) |
| Spacing | preserved from NIfTI/DICOM; absent for PNG/JPEG |

### Explainability

Input-gradient attribution: the absolute gradient of the mean LV-cavity
probability with respect to the input image, matching `ECHO_SALIENCY_CLASS = 1`
in the training notebook. **This is not Grad-CAM and is never labelled as such.**
If the gradient cannot be computed, the saliency tabs are hidden entirely rather
than filled with an empty heatmap — an all-zero gradient still renders as a
smooth, convincing picture, which is worse than showing nothing.

### Orientation

`ECHO_TRAINING_ORIENTATION = "sector apex left, beam opening right"`. That is
CAMUS NIfTI array order as stored, confirmed by the notebook's own saved figure
`models/echo/__results___0_129.png`.

Conventional echo displays are apex-**up** — a quarter turn out of distribution.
The backend never guesses: it reports the mismatch, marks the result provisional
in the MedGemma prompt, and offers 0/90/180/270° plus a horizontal mirror, each
re-running the model on the original file. A silent auto-rotation would leave no
way to tell a real segmentation from a lucky one.

### Reported findings

A structure is reported present only above `ECHO_PRESENCE_THRESHOLD_PX = 50`
labelled pixels; below that it is argmax noise. The threshold is surfaced through
the API and explained in the UI, so "Not identified" cannot be misread as a
clinical statement that the structure is absent.

Areas come back in cm² when the source carries pixel spacing (NIfTI, DICOM) and
as a percentage of the image field otherwise (PNG, JPEG). On a quarter turn, row
and column spacing swap along with the pixels, so cm² stays correct; total field
area is rotation-invariant, which the suite verifies.

> [!IMPORTANT]
> This model outlines anatomy. It does **not** measure ejection fraction, strain,
> volumes over a cardiac cycle, or any functional index. EF appears in the
> codebase only inside a disclaimer, never as a value — asserted by
> `tests/test_report_evidence.py`.

---

## CCTA — Small3DUNet

| Field | Value |
| --- | --- |
| Architecture | `Small3DUNet` (3-D U-Net) |
| Base channels | 16 |
| Input | 1 channel |
| Output | **1 logit per voxel** — sigmoid, not softmax |
| Parameters | 1 401 265 |
| Grad-CAM layer | `enc3.block[-1]` |
| Checkpoint | `models/ccta/best_3d_unet_cca_v2.pth` |
| Source notebook | `notebooks/01_CCTA_Training.ipynb` |
| Trained on | Tesla T4 (Kaggle) |

### Data and split

| Field | Value |
| --- | --- |
| Dataset | MedHK23/CCA |
| Size | 20 annotated volumes, all 832 × 832 × 576 at 0.5 mm isotropic |
| Split | **case-level** 14 train / 3 validation / **3 test** |
| Test case IDs | 9, 14, 15 |
| Mean foreground ratio | 0.001110 — lumen is ~0.11 % of voxels, which is why Dice, not accuracy, is reported |

### Held-out test metrics — n = 3

Every CCTA metric is a spread, never a bare float. The standard deviation below
is the spread of **three numbers**.

| Metric | Mean | SD | Min | Max |
| --- | --- | --- | --- | --- |
| Dice | **0.5996** | 0.1182 | 0.4929 | 0.7266 |
| IoU | 0.4351 | 0.1241 | 0.3270 | 0.5707 |
| Sensitivity | 0.6157 | 0.0874 | 0.5232 | 0.6969 |
| Precision | 0.5878 | 0.1527 | 0.4658 | 0.7590 |
| HD95 (mm) | **109.5094** | 25.0774 | 82.1985 | 131.4994 |

| Field | Value |
| --- | --- |
| Best epoch | 11 |
| Best validation Dice | 0.6505 |
| Threshold | 0.60 |

> [!WARNING]
> **Read the test size.** Three observations support no confidence interval and
> no claim of generalisation. Wherever the Dice is displayed, `n=3` is displayed
> with it.
>
> **Read the HD95 row.** 82–131 mm means the predicted surface has outlier
> components most of a heart-width away from the annotation. Overlap is moderate;
> geometric fidelity is **not** established. This mask locates contrast-filled
> lumen. It is not a verified coronary tree.

### `CCTA_WEAK_NOTES` — surfaced in the API, the UI and the prompt

1. Dice 0.60 means roughly two fifths of the annotated lumen volume is missed or
   over-called; read the mask as a contrast-density highlight to review.
2. The model has exactly **one** foreground class. It does not grade stenosis,
   compute a calcium score, assign a CAD-RADS category, or label which vessel a
   voxel belongs to. Any such statement must come from the reader.
3. Twenty volumes, one public dataset, all at 0.5 mm isotropic. Behaviour on a
   different scanner, contrast protocol or slice thickness is **unmeasured**.

### Preprocessing

| Step | Detail |
| --- | --- |
| Format detection | magic bytes first — `PK\x03\x04` → zip, `\x1f\x8b` → NIfTI, a 348-byte LE/BE header → NIfTI, `DICM` at byte 128 → DICOM — then suffix fallback, else refuse |
| Resample | to 1.0 × 1.0 × 1.0 mm, trilinear (order 1), shape via `round` |
| Intensity | HU clipped to `[-1000, 1000]`, scaled to `[-1, 1]` |
| Pad value | `-1.0` (i.e. −1000 HU, air) |

Resampling to a **coarser** grid reduces the voxel count: 256 voxels at 0.5 mm is
128 mm of tissue, which is 128 voxels at 1 mm. The suite pins this, because
getting it backwards silently scales every reported volume.

### Sliding-window inference

| Field | Value |
| --- | --- |
| Patch | 96 × 96 × 96 |
| Overlap | 0.50 |
| Batch size | 2 |
| Threshold | **0.60** |
| Window budget | 600 |

`compute_starts(length, patch, overlap)` returns `[0]` when the axis is no longer
than the patch; otherwise it strides by `max(1, int(patch × (1 − overlap)))` and
appends the trailing `length − patch` start when the stride would miss the end.
When a volume needs more than 600 windows, `_plan_windows` shrinks the analysed
region proportionally by `scale = (budget / total) ** (1/3)`, centres the crop,
and the response says the coverage was partial — it does not quietly analyse a
subvolume and present it as the whole study.

### Reported findings

A lumen finding is reported present only above
`CCTA_PRESENCE_THRESHOLD_VOXELS = 500`. Volume is voxel count × voxel volume
(1000 voxels of 1 mm³ = 1 mL). `max_probability` is taken over the whole
probability map including voxels **outside** the thresholded mask, because a
near-threshold voxel is still worth seeing.

### Shipped example artefacts — not patient output

`models/ccta/case_{9,14,15}_xai.png` and
`case_{9,14,15}_gradcam_full_resampled.nii.gz` are the notebook's figures for the
three dataset test cases. They are provenance. They must never be rendered as a
patient result.

---

## ECG — ECGResNet1D

| Field | Value |
| --- | --- |
| Architecture | 1-D residual CNN |
| Input | 12 leads × 1000 samples (10 s at 100 Hz) |
| Output | 5 classes, multi-label sigmoid |
| Dropout | 0.30 |
| Parameters | 3 884 165 |
| Threshold | 0.5 |
| Checkpoint | `models/ecg/cardioVision_ptbxl_ecg_resnet1d_full.pt` |
| Source notebook | `notebooks/03_ECG.ipynb` |

### Classes

`ECG_CLASS_NAMES = ("NORM", "MI", "STTC", "CD", "HYP")` — this is the
**checkpoint's own column order**. Reordering it silently reassigns every
prediction, which is why it is read back out of the checkpoint at load time and
cross-checked against `config.py`.

| Code | Label | Meaning |
| --- | --- | --- |
| `NORM` | Normal ECG | no abnormality in the superclasses below |
| `MI` | Myocardial infarction | pattern consistent with prior or acute MI |
| `STTC` | ST/T change | ST-segment or T-wave abnormality |
| `CD` | Conduction disturbance | e.g. bundle branch or AV block |
| `HYP` | Hypertrophy | ventricular or atrial hypertrophy pattern |

### Leads

`I, II, III, aVR, aVL, aVF, V1, V2, V3, V4, V5, V6`. Input in another order is
**not** silently reordered — the loader reports the order it found, so a mismatch
is visible.

### Data and split

| Field | Value |
| --- | --- |
| Dataset | PTB-XL |
| Size | 21 837 records discovered |
| Split | **patient-level** 14 957 / 3 199 / 3 232 records |
| Patients | 13 031 / 2 793 / 2 793, disjointness asserted |
| Loss | multi-label `BCEWithLogitsLoss` with `pos_weight` |
| Best epoch | 14, by validation macro AUROC 0.9203 |

### Held-out test metrics

| Macro metric | Value |
| --- | --- |
| AUROC | **0.9125** |
| AP | 0.7804 |
| F1 | 0.7086 |
| Precision | 0.6353 |
| Recall | 0.8095 |

| Class | AUROC | AP | F1 | Precision | Recall | Test prevalence |
| --- | --- | --- | --- | --- | --- | --- |
| NORM | 0.9498 | 0.9211 | 0.8585 | 0.8009 | 0.9251 | 0.4171 |
| MI | 0.9173 | 0.8294 | 0.7357 | 0.6865 | 0.7925 | 0.2565 |
| STTC | 0.9303 | 0.8126 | 0.7305 | 0.6166 | 0.8959 | 0.2438 |
| CD | 0.9329 | 0.8612 | 0.7668 | 0.7113 | 0.8318 | 0.2373 |
| **HYP** | **0.8323** | **0.4777** | **0.4516** | **0.3614** | 0.6020 | 0.1259 |

> [!WARNING]
> **Read the HYP row.** Macro AUROC 0.9125 looks reassuring and is not the whole
> story. Hypertrophy sits at AP 0.478 with precision 0.361 at the 0.5 threshold,
> against 0.83–0.92 AP for the others — roughly **two in three positive HYP calls
> are wrong**. Treat a positive HYP as a prompt to look at the tracing, never as a
> finding. This is `ECG_WEAK_CLASSES`, and it is surfaced in the API response, the
> UI and the MedGemma prompt instead of being averaged away.

The threshold is 0.5 because the training run reported F1, precision and recall at
that operating point; moving it invalidates those numbers. That is why the API
returns **every probability**, not only the calls.

### Preprocessing

| Step | Detail |
| --- | --- |
| Source rate | 500 Hz (PTB-XL high-resolution records) |
| Target rate | 100 Hz |
| Duration | 10 s → 1000 samples |
| Bandpass | 0.5–40 Hz, order 4 |
| Clip | ±10 |
| Normalisation | per-lead robust median / IQR |

Mirrored in `models/ecg/preprocessing_config.json`.

### Explainability

Per-lead gradient attribution, computed **at request time from the uploaded
record**. The per-recording ranking shown in the UI is that computation.

`models/ecg/lead_importance.csv` and `ECG_EXAMPLE_LEAD_IMPORTANCE` are
`saliency.mean(axis=1)` for **one** test record, `HR00025` — not an average over
the test set. The filename invites the opposite reading. It is provenance for the
shipped example figure and must never be rendered as "which leads this model
relies on".

---

## MedGemma 1.5 4B IT

| Field | Value |
| --- | --- |
| Path | `models/medgemma-1.5-4b-it/` (~8.6 GB of safetensors) |
| Max new tokens | 256 |
| Used for | case-level Q&A, and the narrative section of the integrated report |
| Skip with | `CARDIOVISION_SKIP_MEDGEMMA=1` or `cardiovision serve --skip medgemma` |

It runs locally; no prompt leaves the machine. It is given the **real** model
output as context — never a fabricated finding.

**Withheld from every prompt:** patient name, medical record number.
**Sent:** age (derived from date of birth), sex, study date, free-text notes, the
clinical form, and the structured findings. A clinical answer can use those; an
identifier cannot.

Prompt honesty rules encoded in `services/case_context.py` and `fusion/report.py`:

- unticked risk-factor boxes are described as **unknown**, with an explicit
  instruction not to describe the patient as having denied them
- weak classes and weak models are stated in the prompt, not left for the model
  to infer
- an echo result whose orientation did not match the training distribution is
  marked provisional
- `?include_prompt=true` on `/api/report` returns the exact text, so a reader can
  check the narrative claims nothing the evidence did not
- when generation fails, `ai_summary_error` is populated and the structured report
  stands on its own

Not distributed with this repository. Download it and accept the Google Health AI
Developer Foundations terms at that point.

---

## There is no fusion model

`src/cardiovision/fusion/` is a **deterministic software aggregation layer**. It
is not a network, it was not trained, and `MODALITY_STATUS["fusion"]["available"]`
is `False`. `notebooks/04_Multimodal_Fusion.ipynb` is empty.

`model_versions["fusion"]` in every report reads:

```json
{
  "model": null,
  "task": "deterministic software evidence aggregation",
  "note": "No learned fusion model exists in this project."
}
```

What it does: reads the finished analyse responses, emits per-modality evidence
under a fixed status vocabulary (`analysed`, `not_provided`,
`provided_not_analysed`, `no_model` — each with a stated meaning), records typed
uncertainties, and notes cross-modal **co-occurrence**.

What it will not do: combine findings into a score, weight one modality against
another, or say that one modality supports another. Every
`CrossModalObservation` carries `inference: "none"`, and the phrases *consistent
with*, *confirms*, *corroborates*, *suggestive of*, *rules out* and *risk score*
are asserted absent from its output.

---

## There is no clinical risk model

`MODALITY_STATUS["clinical"]["available"]` is `False`. There is no
clinical-risk notebook. Clinical values typed into the form are passed to the
language model as **text context only** — no risk score is computed anywhere.

An unticked risk-factor checkbox means **unknown**, not denied: the form ships
those fields defaulting to `false`, so `false` means the clinician never touched
the box. An earlier version emitted them as "not reported", handing the model a
negative history nobody had taken, which it then repeated back as fact.

---

## Checkpoint hygiene

| Rule | Reason |
| --- | --- |
| Preserve the exact filenames in `config.py` | they are resolved literally; a rename surfaces as "checkpoint not found" |
| Never commit a placeholder under a checkpoint's real filename | a zero-byte or dummy file makes the loader fail deep inside `torch.load` instead of at a clear "missing weights" message, and can be committed by accident as a real change |
| Weights go through **Git LFS** | see `.gitattributes` |
| `models/ccta/latest_3d_unet_cca_v2.pth` is **not** the model to load | it is a later but **worse** epoch, and it carries a different selected threshold. Loading it would silently change the operating point. `config.py` points at `best_3d_unet_cca_v2.pth` |
| `models/echo/cardiovision_echo_unetplusplus_last.pth` and `models/ecg/best_ecg_resnet1d.pt` are likewise not the served weights | kept as training provenance only |
| Validation metrics are read from the checkpoint, not hardcoded | so they cannot drift from the weights they describe |
| `tests/test_ecg_architecture.py` compares the constructed architecture against the checkpoint's parameter names and shapes | a constant changed in `config.py` without a matching checkpoint fails loudly instead of loading a mis-wired model |

