---
name: ecg-inference
description: Run or modify 12-lead ECG classification — ECGResNet1D, five multi-label superclasses at threshold 0.5, per-lead attribution.
---

# ECG inference

## Purpose

Screen a 10-second 12-lead recording for five diagnostic superclasses
(`NORM MI STTC CD HYP`), multi-label sigmoid at threshold 0.5, and attribute the
call across leads.

## When to use

- Adding or changing a record format, the resample, the bandpass or the normalisation
- Changing the class handling, the threshold reporting or the weak-class note
- Working on `POST /api/analyze/ecg` or `GET /api/models/ecg`
- Debugging a lead-order or sampling-frequency complaint

## Relevant files

| File | Role |
| --- | --- |
| `src/cardiovision/config.py` | `ECG_CHECKPOINT_PATH`, `ECG_CLASS_NAMES`, `ECG_CLASS_LABELS`, `ECG_LEAD_NAMES`, `ECG_THRESHOLD`, `ECG_WEAK_CLASSES`, `ECG_TEST_METRICS`, `ECG_EXAMPLE_LEAD_IMPORTANCE`, preprocessing constants, `ALLOWED_ECG_SUFFIXES` |
| `src/cardiovision/preprocessing/ecg_io.py` | WFDB/CSV/NPY/JSON/zip load, 500→100 Hz resample, 0.5–40 Hz bandpass, per-lead robust median/IQR normalisation, lead-order reporting |
| `src/cardiovision/inference/ecg.py` | `ECGResNet1D`, `ConvBlock`, `EcgClassifier`, `ClassPrediction`, `LeadAttribution`, `EcgAnalysis`, `EcgModelUnavailable` |
| `src/cardiovision/rendering/ecg.py` | 12-lead waveform with saliency shading, lead-attribution chart |
| `src/cardiovision/api/routers/ecg.py` | the route, its parameters and its error codes |
| `models/ecg/preprocessing_config.json` | the notebook's mirror of the preprocessing constants |
| `tests/test_ecg_pipeline.py`, `test_ecg_rendering.py`, `test_ecg_reporting.py`, `test_ecg_architecture.py` | 298 checks |

## Expected inputs

| Parameter | Notes |
| --- | --- |
| `file` | `.hea .dat .mat .csv .txt .tsv .npy .json .zip` |
| `companions` | other files of the same recording (e.g. the `.dat` or `.mat` for a `.hea`) — matched by **filename**, not order, and not suffix-checked |
| `sampling_frequency` | 0–10000 Hz, for formats that do not record it (CSV, NPY). Getting it wrong rescales the recording in time, so the loader reports whatever it used |
| `target_class` | which class the saliency explains; must be one of `NORM MI STTC CD HYP`, else `422`. Defaults to the highest-probability class |
| `include_figures` | 12-lead strip and lead-attribution chart, roughly 160 KB of SVG |
| `case_id` | archive the upload |

## Expected outputs

- **Every** class probability, not only the calls above 0.5, each with its label
- The operating point the published precision and recall belong to
- A positive `HYP` flagged with its weak-class note in the response itself
- Per-lead attribution computed at request time from the uploaded record
- The lead order as found, the sampling frequency used, and the device

## Important constraints

- `ECG_CLASS_NAMES = ("NORM", "MI", "STTC", "CD", "HYP")` is the **checkpoint's own
  column order**. Reordering it silently reassigns every prediction. It is read back
  out of the checkpoint at load time and cross-checked against `config.py`; keep that
  check.
- Lead order is **reported, not corrected**. A non-standard order is surfaced so a
  mismatch is visible; silent reordering hides a broken upload.
- The threshold is 0.5 because the training run reported F1, precision and recall at
  that operating point. Moving it invalidates those numbers — which is why every
  probability is returned instead.
- **HYP is weak.** AUROC 0.8323, AP 0.4777, precision 0.3614 at threshold 0.5 —
  roughly two in three positive HYP calls are wrong, against 0.83–0.92 AP for the
  other four. Macro AUROC 0.9125 hides this. `ECG_WEAK_CLASSES` must reach the API
  response, the UI and the MedGemma prompt.
- `models/ecg/lead_importance.csv` and `ECG_EXAMPLE_LEAD_IMPORTANCE` are
  `saliency.mean(axis=1)` for **one** test record, `HR00025` — not an average over
  the test set. The filename invites the opposite reading. Provenance for the shipped
  example figure only; never label it "which leads this model relies on".
- `best_ecg_resnet1d.pt` is not the served checkpoint;
  `cardioVision_ptbxl_ecg_resnet1d_full.pt` is. Note the capital V.

## Verification steps

```bash
python3 tests/test_ecg_pipeline.py
python3 tests/test_ecg_rendering.py
python3 tests/test_ecg_reporting.py
python3 tests/test_ecg_architecture.py   # skips cleanly on an unresolved LFS pointer
```

The architecture suite compares the constructed network against the checkpoint's own
parameter names and shapes, so run it after touching any architecture constant. SciPy
is absent in the sandbox, so the bandpass arithmetic is unexecuted there.
