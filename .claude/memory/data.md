# Data

Every dataset named here is one this repository can point at a real source for.
There are no others. Three datasets were used for training. **No training or
validation split ships here and no patient data is committed** — what does ship
is two sample cases per modality under `samples/`, tracked on purpose so the
upload paths can be exercised (§4).

---

## 1. CCTA — MedHK23/CCA

| | |
|---|---|
| Source | Hugging Face dataset, `snapshot_download(repo_id="MedHK23/CCA", repo_type="dataset")` |
| Pinned commit | `a78045d4546ec9f52484920d66152db7f31f84a1` (recorded in `models/ccta/__huggingface_repos__.json`) |
| Format | NIfTI `.nii.gz`, paired `train/images/{n}.nii.gz` + `train/labels/{n}.nii.gz` |
| Size | **20 volumes** |
| Native geometry | `(832, 832, 576)` voxels at `0.5 mm` isotropic spacing — every volume identical |
| Labels | One binary foreground channel: coronary artery lumen. No vessel identity, no lesion class. |
| Class imbalance | mean foreground ratio **0.001110** — roughly 1 voxel in 900 is lumen |

**Split** — `models/ccta/cardiovision_cca_split.csv`, 20 rows:

| Split | Cases |
|---|---|
| train | 14 |
| validation | 3 |
| test | 3 — case ids **9, 14, 15** |

This dataset has one volume per case and no patient identifier beyond the case
id, so case-level splitting *is* patient-level splitting here. That is a
property of the dataset, not a shortcut.

**Preprocessing (from `notebooks/01_CCTA_Training.ipynb`)**

- Resample to `1.0 mm` isotropic with `nibabel.processing.resample_from_to`
- Clip intensities to `[-1000, 1000]` HU; NaN/±inf mapped to the window edges
- Training patches: `96³`, 16 per case, **45% foreground-centred, 30% hard
  negative**, remainder background; augmentation is geometric only
- Validation and test: full-volume sliding window, overlap `0.50`, batch 2
- Loss: Dice + Tversky + `BCEWithLogitsLoss`; AdamW at `2e-4`,
  `ReduceLROnPlateau`, AMP on, 15 epochs, seed 42
- The decision threshold was searched **on validation only**, never on test

**Limitations.** Twenty volumes total and **three test cases**. Every reported
CCTA number is an n=3 estimate — see the spread in `.claude/memory/models.md`.
Single source, single acquisition geometry, no external validation.

---

## 2. Echocardiography — CAMUS

| | |
|---|---|
| Source | CAMUS, read from a Kaggle mirror at `camus-extracted/camus/database_nifti` |
| Format | NIfTI `.nii`, `patientXXXX_{2CH,4CH}_{ED,ES}.nii` + `..._gt.nii` masks |
| Size | **500 patients**, asserted in the notebook |
| Sampling | 2 views (2CH, 4CH) × 2 phases (ED, ES) = 4 image/mask pairs per patient → **2 000 pairs** |
| Labels | 4 classes: `0` background, `1` LV cavity, `2` myocardium, `3` left atrium |

**Split** — patient-level, `train_test_split(test_size=0.30, random_state=42)`
then the remainder halved:

| Split | Patients | Pairs |
|---|---|---|
| train | 350 | 1 400 |
| validation | 75 | 300 |
| test | 75 | 300 |

The split is taken over **patient ids** and then applied to pairs, so all four
frames of a patient land in the same split. No patient appears on both sides of
the boundary.

**Preprocessing**

- Image resized to `256 × 256` **bilinear**; mask resized **nearest**
- Single channel input
- **No augmentation** — no flips, rotations or intensity jitter
- Loss: Dice + `CrossEntropyLoss`; AdamW at `2e-4`, weight decay `1e-4`,
  `CosineAnnealingLR(T_max=50)`, batch 8, AMP on, seed 42
- 50 epochs planned; the loaded checkpoint is **epoch 30**

**Limitations.** A CAMUS sector is not square, so the `256 × 256` resize
changes the aspect ratio; the model learned that distortion and inference
reproduces it, which is why aspect-ratio handling must stay consistent between
training and serving. Training saw only apex-left sectors — see
`ECHO_TRAINING_ORIENTATION` in `config.py`. Absence of augmentation means
robustness to flips, gain changes and probe rotation is **untested**, not
established.

---

## 3. ECG — PTB-XL

| | |
|---|---|
| Source | PTB-XL v1, read from a Kaggle mirror at `physionet/ptbxl-electrocardiography-database/ptb-xl/1` |
| Records discovered | 21 837 |
| Matched to metadata | 21 799 (of 21 799 metadata rows) |
| With clinical text | 21 388 |
| Labels | 5 diagnostic **superclasses**, multi-label: `NORM`, `MI`, `STTC`, `CD`, `HYP` |

**Split** — by patient, then records follow:

| Split | Patients | Records |
|---|---|---|
| train | 13 031 | 14 957 |
| validation | 2 793 | 3 199 |
| test | 2 793 | 3 232 |

Patients outnumber-per-record, i.e. some patients contributed more than one
recording; splitting on the patient id is what keeps those recordings from
straddling the boundary. **Preserve this** if the split is ever regenerated.

**Class prevalence** (`models/ecg/class_distribution.csv`) — stable across
splits, which is the point of checking it:

| Class | train | validation | test |
|---|---|---|---|
| NORM | 0.4246 | 0.4283 | 0.4171 |
| MI | 0.2549 | 0.2588 | 0.2565 |
| STTC | 0.2484 | 0.2285 | 0.2438 |
| CD | 0.2280 | 0.2254 | 0.2373 |
| HYP | **0.1237** | **0.1225** | **0.1259** |

`HYP` is the rarest class in every split and also the weakest-performing one.
Those two facts are related and both belong in any statement about HYP.

**Preprocessing** (`models/ecg/preprocessing_config.json`, mirrored in
`config.py`)

- Resample `500 Hz` → `100 Hz`; 10 seconds → **1 000 samples**
- 12 leads in a fixed order: I, II, III, aVR, aVL, aVF, V1–V6
- Butterworth bandpass `0.5–40 Hz`, order 4
- Per-lead **robust** normalisation: median and IQR, not mean and standard
  deviation
- Clip to `[-10, 10]` after normalisation
- Seed 42
- `models/ecg/preprocessing_failures.csv` is a single newline — **no record
  failed preprocessing**

**Training.** Batch 64, up to 40 epochs, early-stopping patience 8, lr `1e-3`,
weight decay `1e-4`, dropout `0.30`, decision threshold `0.5`. Best epoch 14.

**Limitations.** Single-centre German recordings collected 1989–1996; the label
set is the 5-superclass grouping, not the 71 diagnostic statements, so the model
cannot name a specific rhythm or infarct territory. No external validation.

---

## 4. What data actually lives in this repository

`data/` is runtime state only. Sample inputs live in `samples/` and are tracked
on purpose — see below.

| Path | State |
|---|---|
| `data/.gitignore` | Tracked, and the **only** tracked file under `data/` (`git ls-files data/`) |
| `data/cardiovision.db`, `-wal`, `-shm` | Runtime SQLite case store — gitignored, contains patient data, unencrypted |
| `data/cases/` | `CASE_FILES_DIR`: where the store writes each case's renders and source upload at runtime. Gitignored (`cases/`). Absent in a fresh clone; the store creates it |
| `data/sample/` | Does not exist, despite the `!data/sample/` negation in the root `.gitignore` |
| `samples/` | **Tracked dataset samples**, 181 MB, 38 files — see the table below |

### Sample inputs — `samples/`

Two cases per modality, taken from the training datasets, so every upload path can
be exercised without fetching anything. Documented in `samples/README.md`.

| Path | Dataset | Files | Size |
|---|---|---|---|
| `samples/ccta/CCTA_CASE_01{4,5}/{ccta_image,ground_truth}.nii.gz` | MedHK23/CCA | 4 | 165 MB (97 MB + 70 MB images) |
| `samples/echo/patient000{1,2}/` | CAMUS | 30 | 16 MB, gzipped from 189 MB |
| `samples/ecg/HR0000{1,2}.{hea,mat}` | PTB-XL | 4 | 248 KB |

Four facts about them that matter:

1. **They used to live in `data/cases/`** — committed there before the `cases/`
   ignore rule existed, i.e. inside `CASE_FILES_DIR`, the directory the store writes
   real patient files into. Moved out with `git mv`, so history is intact.
2. **They are tracked deliberately.** The developer's decision, recorded here so it is
   not re-litigated as a mistake. `LICENSE` must therefore not claim MedHK23/CCA is
   undistributed, and PTB-XL's attribution requirement applies to the repository.
3. **The CAMUS frames were gzipped** (`.nii` → `.nii.gz`) to get 189 MB down to 16 MB.
   Both suffixes are in `ALLOWED_ECHO_SUFFIXES`, so nothing else changed.
4. **The ECG pair is the packaging the model was trained from** — `.hea` + `.mat`,
   format `16+24`. See `.claude/memory/models.md` and `tests/test_ecg_pipeline.py` §17.

**Which split each sample came from**, because it decides what a run on it can be
claimed to show:

| Sample | Split | Consequence |
|---|---|---|
| `CCTA_CASE_014`, `CCTA_CASE_015` | **test** (`models/ccta/cardiovision_cca_split.csv`: test = 9, 14, 15) | never trained on — but two of the same three cases the published Dice and the 0.60 threshold came from, so a run here is a reproduction, not an independent evaluation |
| `HR00001`, `HR00002` | **Not established in repository** — the per-record split lives in the notebook's `*_metadata.csv`, which is not shipped | do not claim these are held out |
| `patient0001`, `patient0002` | **Not established in repository** | do not claim these are held out |

They are dataset samples, **not** results and **not** an evaluation set. A comparison
against the shipped `_gt` files on two cases is an anecdote.

`data/.gitignore` carries its own warning: *"Patient data. Never commit this,
and do not sync it to cloud storage — the database is not encrypted."* Take it
literally.

**Training-provenance artefacts that do ship:**

| Path | What it is |
|---|---|
| `models/ccta/cardiovision_cca_index.csv` | Per-case shape, spacing, foreground voxels and ratio for all 20 volumes. The `image`/`label` columns hold absolute `/root/.cache/huggingface/...` paths from the training machine. **Provenance only — never read this at runtime.** |
| `models/ccta/cardiovision_cca_split.csv` | The 14/3/3 assignment |
| `models/ccta/test_metrics.csv` | Per-case test metrics for cases 9, 14, 15 |
| `models/ccta/case_{9,14,15}_xai.png`, `..._gradcam_full_resampled.nii.gz` | Grad-CAM renders for the three held-out cases |
| `models/ecg/HR00025_saliency.{csv,png}` | Saliency for one test record |
| `models/ecg/*.json` | Preprocessing, model, pipeline and test-metric records |

These are derived artefacts about held-out cases, not the cases themselves. No
source image or recording is present.

---

## 5. Test data

**No verification suite reads any real data, and none needs to.** They build
synthetic arrays and drive the models through `tests/torch_stub.py` when torch is
absent. That is deliberate: it keeps the suite runnable on a clean clone with no
downloads and no patient data.

`samples/` is for the manual acceptance pass, not for the suites — do not point a
test at it. Two of its files (`CCTA_CASE_01{4,5}`) *are* dataset test cases, which
makes them useful for a sanity check and useless as an independent metric: the
published CCTA numbers and threshold were measured on them.

To create *real* test cases, re-fetch from the pinned sources above and take
only the rows the split files assign to `test`. Do not sample from train or
validation, and do not invent inputs — a fabricated volume tells you nothing
about the model and everything about the fabricator.

---

## 6. Cross-cutting limitations

- **No multimodal dataset exists.** CCTA, CAMUS and PTB-XL are three unrelated
  cohorts. **No patient appears in more than one of them.** Any case that
  combines modalities in this application was assembled by a human operator, and
  the software must say so rather than implying a joint acquisition.
- **No external validation** for any of the three models.
- **No demographic breakdown** of performance was computed for any modality.
- Label vocabularies are narrow by construction: CCTA has one foreground class,
  echo has three structures, ECG has five superclasses. The models cannot report
  anything outside those vocabularies, and neither should the application.
