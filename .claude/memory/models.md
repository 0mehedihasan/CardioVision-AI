# Models — exactly what is present

> Sizes are bytes on disk from `find models -type f -printf '%s'`. Constants,
> architectures and metrics come from `src/cardiovision/config.py`, which cites
> the notebook that produced each number. **Do not restate a metric that is not
> in `config.py` or in a file under `models/`.**

Framework for all three trained models: **PyTorch**. Checkpoint format for all
three: `torch.save` zip archive (`data.pkl` plus tensor storages), tracked with
**Git LFS** (`.gitattributes` covers `*.pt`, `*.pth`, `*.ckpt`, `*.pkl`,
`*.joblib`, `*.safetensors`, `*.bin`).

---

## 1. CCTA — coronary lumen segmentation

| Field | Value |
| --- | --- |
| Loaded checkpoint | `models/ccta/best_3d_unet_cca_v2.pth` — 16 879 575 B |
| Later epoch (**not loaded, not equivalent**) | `models/ccta/latest_3d_unet_cca_v2.pth` — 16 879 987 B — `epoch=15`, `current_val_dice=0.6222`, **`selected_threshold=0.8`** |
| Config constant | `CCTA_CHECKPOINT_PATH` |
| Architecture | `Small3DUNet(in_channels=1, out_channels=1, base=16)` in `inference/ccta.py` |
| Parameters | 1 401 265 |
| Output | **One** logit per voxel, sigmoid — not softmax |
| Classes | `0 Background`, `1 Coronary artery lumen` |
| Operating threshold | `0.60` (`selected_threshold` in the checkpoint is `0.6000000000000002`; the loader reads it from the file) |
| Checkpoint contents | `epoch`, `model_state_dict` (50 tensors), `optimizer_state_dict`, `scheduler_state_dict`, `scaler_state_dict`, `best_val_dice`, `current_val_dice`, `selected_threshold`, `config` |
| Preprocessing | resample to 1 mm isotropic (order 1), clip HU to [-1000, 1000], scale to [-1, 1]; pad value `-1.0` equals air |
| Inference | 96³ sliding windows, 50 % overlap, batch 2, capped at `CCTA_MAX_WINDOWS = 600` |
| Explainability | 3-D Grad-CAM on `enc3.block[-1]` |
| Presence cutoff | `CCTA_PRESENCE_THRESHOLD_VOXELS = 500` (≈0.5 mL at 1 mm) |

**Held-out test — n = 3 (cases 9, 14, 15).** From `models/ccta/test_metrics.csv`:

| | dice | iou | sensitivity | precision | hd95 (mm) |
| --- | --- | --- | --- | --- | --- |
| mean | 0.5996 | 0.4351 | 0.6157 | 0.5878 | 109.51 |
| sd | 0.1182 | 0.1241 | 0.0874 | 0.1527 | 25.08 |
| min | 0.4929 | 0.3270 | 0.5232 | 0.4658 | 82.20 |
| max | 0.7266 | 0.5707 | 0.6969 | 0.7590 | 131.50 |

> **Hazard.** `latest_*.pth` is not a spare copy of `best_*.pth`. It is a later,
> worse epoch whose recorded operating point is 0.8 rather than 0.6. Loading it
> would silently change the threshold. `CCTA_CHECKPOINT_PATH` points at
> `best_3d_unet_cca_v2.pth`; leave it there.

**This model does NOT produce:** stenosis grading, calcium score, CAD-RADS
category, vessel labelling, or any per-vessel identity. Three test cases support
no confidence interval and no claim of generalisation. `CCTA_WEAK_NOTES` carries
this into the API, the UI and the prompt — do not weaken it.

Supporting artefacts in `models/ccta/`:

| File | Bytes | What it is |
| --- | --- | --- |
| `test_metrics.csv` | 405 | The three test rows above — source of truth for the metrics |
| `cardiovision_cca_split.csv` | 196 | Case-level split: 14 train / 3 validation / 3 test |
| `cardiovision_cca_index.csv` | 7 156 | Per-case shape, spacing, foreground ratio. **Contains `/root/.cache/huggingface/...` Kaggle paths** — training provenance, never runtime config |
| `__huggingface_repos__.json` | 1 205 | Dataset pin: `MedHK23/CCA`, commit `a78045d4546ec9f52484920d66152db7f31f84a1` |
| `case_{9,14,15}_xai.png` | ~0.7 MB each | Notebook Grad-CAM figures for the three test cases |
| `case_{9,14,15}_gradcam_full_resampled.nii.gz` | ~3.5 MB each | The same, as volumes |

The `case_*` artefacts are **dataset test cases, not patient output**
(`CCTA_EXAMPLE_ARTIFACTS`). Never render them as a result.

---

## 2. Echo — cardiac structure segmentation

| Field | Value |
| --- | --- |
| Loaded checkpoint | `models/echo/cardiovision_echo_unetplusplus_best.pth` — 159 764 920 B |
| Second copy (not loaded) | `models/echo/cardiovision_echo_unetplusplus_last.pth` — 159 764 920 B |
| Checkpoint contents | `model_state_dict` (706 tensors), `optimizer_state_dict`, `epoch=30`, `val_dice=0.9115`, `val_iou=0.8392`, `image_size=256`, `num_classes=4`, `architecture`, `encoder` |
| Config constant | `ECHO_CHECKPOINT_PATH` |
| Architecture | `segmentation_models_pytorch.UnetPlusPlus`, encoder `timm-efficientnet-b3`, `encoder_weights=None` |
| Input | 1 channel, 256 × 256 |
| Classes | `0 Background`, `1 LV cavity`, `2 Myocardium`, `3 Left atrium` |
| Explainability | Input-gradient saliency for `ECHO_SALIENCY_CLASS = 1` (LV cavity) |
| Presence cutoff | `ECHO_PRESENCE_THRESHOLD_PX = 50` |
| Training orientation | **Sector apex LEFT, beam opening right** (`ECHO_TRAINING_ORIENTATION`) — a conventional apex-up display is a quarter turn away, which is why the UI offers rotate/flip |

**Held-out test** (CAMUS, 75 patients / 300 pairs, patient-level split): dice
`0.9044`, iou `0.8282`; per class LV cavity `0.9379`, myocardium `0.8759`, left
atrium `0.8994`. `val_dice`/`val_iou` are **not** hardcoded — they are read out
of the checkpoint at load time so they cannot drift from the weights.

**This model does NOT produce:** ejection fraction, wall motion, valve
assessment, or any diagnosis. It segments one frame of one view; nothing is
measured across the cardiac cycle.

Supporting artefacts: `training_history.csv` (4 707 B),
`xai_lv_cavity_saliency.npy` (262 272 B), `xai_prediction_mask.npy`
(524 416 B), `__results___0_123.png`, `__results___0_129.png` (the notebook
figure that documents the training orientation),
`__huggingface_repos__.json` (encoder pin: `smp-hub/timm-efficientnet-b3.imagenet`).

---

## 3. ECG — 12-lead multi-label screening

| Field | Value |
| --- | --- |
| Loaded checkpoint | `models/ecg/cardioVision_ptbxl_ecg_resnet1d_full.pt` — 15 583 493 B |
| Other checkpoint (**not loaded**) | `models/ecg/best_ecg_resnet1d.pt` — 46 684 587 B — the mid-training checkpoint: same 66 tensors plus `optimizer_state_dict`, `scheduler_state_dict`, `epoch=14` |
| Config constant | `ECG_CHECKPOINT_PATH` |
| Architecture | `ECGResNet1D` — hand-written 1-D residual CNN in `inference/ecg.py` |
| Parameters | 3 884 165 |
| Input | 12 leads × 1000 samples (10 s at 100 Hz) |
| Classes | `NORM`, `MI`, `STTC`, `CD`, `HYP` — independent sigmoids, order read back from the checkpoint |
| Threshold | `0.5` |
| Preprocessing | bandpass 0.5–40 Hz order 4, resample to 100 Hz, fix length 1000, clip [-10, 10], per-lead robust median/IQR |
| Explainability | Input-gradient saliency, `abs(d logit / d sample)`, plus per-lead attribution computed per request |
| Checkpoint contents | `model_state_dict` (66 tensors), `target_classes`, `lead_names`, `input_channels`, `input_length`, `target_fs`, `preprocessing_config`, `model_config`, `test_metrics`, `best_validation_macro_AUROC=0.9203`, `best_epoch=14` |

**Held-out test** (PTB-XL, 3 232 records / 2 793 patients, patient-level split):
macro AUROC `0.9125`, macro AP `0.7804`, macro F1 `0.7086`.

| Class | AUROC | AP | F1 | Precision | Recall | Prevalence |
| --- | --- | --- | --- | --- | --- | --- |
| NORM | 0.9498 | 0.9211 | 0.8585 | 0.8009 | 0.9251 | 0.4171 |
| MI | 0.9173 | 0.8294 | 0.7357 | 0.6865 | 0.7925 | 0.2565 |
| STTC | 0.9303 | 0.8126 | 0.7305 | 0.6166 | 0.8959 | 0.2438 |
| CD | 0.9329 | 0.8612 | 0.7668 | 0.7113 | 0.8318 | 0.2373 |
| **HYP** | **0.8323** | **0.4777** | **0.4516** | **0.3614** | **0.6020** | 0.1259 |

**Read the HYP row.** Precision 0.36 means roughly two in three positive
hypertrophy calls are wrong. `ECG_WEAK_CLASSES["HYP"]` exists for this and is
surfaced in the response, the UI and the prompt. Do not average it away.

**This model does NOT produce:** heart rate, rhythm, PR/QRS/QT intervals, axis,
infarct localisation, acute-versus-old distinction, or atrial fibrillation (not
one of the five classes). Per-lead normalisation removes absolute voltage, so no
millivolt amplitude can be attributed to it.

**The packaging it was trained from.** `notebooks/03_ECG.ipynb` read PTB-XL in the
PhysioNet Challenge repackaging: a WFDB `.hea` header beside a **`.mat`** signal
file, with the format field declared as `16+24`. The `+24` is a byte offset — a
24-byte MATLAB v4 header sits before the samples. MATLAB v4 is column-major, so a
12 x N `val` matrix is byte-identical to WFDB's interleaved format-16 layout once
that header is skipped. `preprocessing/ecg_io.py` parses the offset and skips it;
reading it as signal does not raise, it fabricates one sample per lead (up to
~124 mV, roughly 100x physiological) and drops the last real one. Both suffixes
are in `ALLOWED_ECG_SUFFIXES`, and `tests/test_ecg_pipeline.py` section 17 pins
the behaviour.

Config files in `models/ecg/`, all real and all consistent with `config.py`:

| File | Bytes | Contents |
| --- | --- | --- |
| `model_config.json` | 590 | architecture, input shape, classes, params, AdamW, lr 1e-3, BCEWithLogitsLoss + pos_weight |
| `preprocessing_config.json` | 531 | 500→100 Hz, 10 s, 12 leads, bandpass, clip, seed 42 |
| `pipeline_config.json` | 2 281 | End-to-end record: record/patient counts per split, classes, leads, sampling rates, hyperparameters, best epoch, test metrics |
| `test_metrics.json` | 1 115 | The per-class table above, full precision |
| `class_distribution.csv` | 652 | Label counts |
| `lead_importance.csv` | 188 | **One record only** (HR00025). See `ECG_EXAMPLE_LEAD_IMPORTANCE` — never present it as "which leads this model relies on" |
| `preprocessing_failures.csv` | 1 | A single newline — no preprocessing failures were recorded |
| `HR00025_saliency.csv` / `.png` | 137 KB / 1.5 MB | The shipped example saliency figure |

---

## 4. MedGemma — local language model

| Field | Value |
| --- | --- |
| Path | `models/medgemma-1.5-4b-it/` (`MEDGEMMA_PATH`) |
| Display name | `MedGemma 1.5 4B IT` |
| Weights | `model-00001-of-00002.safetensors` 4 961 251 752 B + `model-00002-of-00002.safetensors` 3 639 026 128 B ≈ **8.6 GB** |
| Loader | `transformers.AutoModelForImageTextToText` |
| Generation | `MAX_NEW_TOKENS = 256` |
| Tracked in Git? | **No.** `models/medgemma-1.5-4b-it/` is gitignored — vendor weights, a `huggingface-cli download` away |

MedGemma **never overrides a model output**. It receives structured evidence from
`fusion/report.py` or `services/case_context.py` and writes prose over it.

---

## 5. Models that do NOT exist

| Claimed capability | Reality |
| --- | --- |
| Clinical risk model | No weights, no notebook, `MODALITY_STATUS["clinical"].available == False` |
| Multimodal fusion model | `models/fusion/` is an **empty directory**; `notebooks/04_Multimodal_Fusion.ipynb` is **0 bytes** |
| Stenosis / CAD-RADS / calcium scoring | Not produced by any model here |
| Ejection fraction / wall motion | Not produced by any model here |
| Rhythm or interval measurement | Not produced by any model here |

If asked to "enable" any of these: it needs weights, not wiring. Say so.

---

## Loading a checkpoint safely

- Read metadata **without torch** using `python3 tests/checkpoint_reader.py <path>`
  before writing code against a checkpoint.
- `tests/test_ecg_architecture.py` compares the module's `state_dict()` names and
  shapes against the checkpoint. It exits 0 with a message when the LFS pointer
  is unresolved, so CI stays green without pulling 46 MB.
- Without git-lfs, a `.pt`/`.pth` arrives as ~130 bytes of pointer text and the
  loader reports a corrupt checkpoint. `git lfs install && git lfs pull`.
