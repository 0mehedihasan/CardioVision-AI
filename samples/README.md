# Sample inputs

Small, real recordings from the three public datasets the models were trained
on — enough to exercise every upload path end to end without hunting for data
first.

> [!IMPORTANT]
> These are **dataset samples, not patient records, and not results.** They live
> here rather than in `data/`, because `data/cases/` is `CASE_FILES_DIR` — the
> directory the case store writes real patient files into. Nothing under `data/`
> is ever committed.

---

## Contents

| Path | Dataset | Files | Size | Feeds |
| --- | --- | --- | --- | --- |
| `ccta/CCTA_CASE_014`, `ccta/CCTA_CASE_015` | MedHK23/CCA | `ccta_image.nii.gz` + `ground_truth.nii.gz` | 165 MB | `POST /api/analyze/ccta` |
| `echo/patient0001`, `echo/patient0002` | CAMUS | 2CH + 4CH, ED/ES/half-sequence, each with `_gt`, plus `Info_*.cfg` | 16 MB | `POST /api/analyze/echo` |
| `ecg/HR00001`, `ecg/HR00002` | PTB-XL | `.hea` + `.mat` | 248 KB | `POST /api/analyze/ecg` |

The `_gt` / `ground_truth` files are the datasets' own expert annotations. They
are useful for eyeballing a prediction against a reference; they are **not** a
test set, and a comparison on two cases is an anecdote, not a metric. The
published numbers in `docs/models.md` come from the notebooks' held-out splits.

---

## Using them

```bash
# ECG — WFDB is two files, so the signal file goes as a companion
curl -X POST http://127.0.0.1:8000/api/analyze/ecg \
  -H "Authorization: Bearer $TOKEN" \
  -F file=@samples/ecg/HR00001.hea \
  -F companions=@samples/ecg/HR00001.mat

# Echo — a single frame, apex left, so a conventional apex-up view needs rotate=90
curl -X POST http://127.0.0.1:8000/api/analyze/echo \
  -H "Authorization: Bearer $TOKEN" \
  -F file=@samples/echo/patient0001/patient0001_4CH_ED.nii.gz

# CCTA — a whole volume; expect tens of seconds and a large response
curl -X POST http://127.0.0.1:8000/api/analyze/ccta \
  -H "Authorization: Bearer $TOKEN" \
  -F file=@samples/ccta/CCTA_CASE_014/ccta_image.nii.gz
```

Notes on the formats, all verified against these files:

- **The ECG pair is the packaging the model was trained from.** `notebooks/03_ECG.ipynb`
  read a `WFDB/` directory of `.hea` + `.mat` pairs through the `wfdb` package.
  These headers declare their format as `16+24` — a 24-byte MATLAB v4 header
  before the samples. The in-app reader now honours that offset; before it did,
  it read those 24 bytes as one fabricated sample (up to 124 mV, ~100× anything
  physiological) and dropped the last real one. `tests/test_ecg_pipeline.py` §17
  pins it.
- **The CAMUS frames are stored gzipped** (`.nii.gz`, not the dataset's plain
  `.nii`) purely to keep 189 MB out of the repository; 16 MB carries the same
  voxels. Both suffixes are in `ALLOWED_ECHO_SUFFIXES`.
- **The CCTA volumes are 97 MB and 70 MB**, the larger one close to GitHub's
  100 MB per-file warning. Do not add more of them here.

---

## Licences and attribution

Each dataset keeps its own terms; the repository's MIT licence covers its code
only and relicenses none of the below. See `LICENSE`.

| Dataset | Terms | Obligation |
| --- | --- | --- |
| CAMUS (echo) | research use — University of Lyon, CREATIS | cite the dataset paper; each `echo/patient*/MANDATORY_CITATION.md` carries the required text and ships with the samples |
| PTB-XL (ECG) | PhysioNet, Open Data Commons Attribution Licence v1.0 | **attribution required** |
| MedHK23/CCA (CCTA) | the source's own terms | check them before redistributing anything derived from these volumes |

CAMUS: Leclerc et al., *Deep learning for segmentation using an open large-scale
dataset in 2D echocardiography*, IEEE TMI 38(9):2198–2210, 2019.
doi:10.1109/TMI.2019.2900516

---

## Adding to this directory

- Keep it small. This is a functional smoke-test set, not a distribution
  channel: two cases per modality is the intent.
- Never put a real patient study here, gzipped or not. Runtime patient files
  belong in `data/cases/`, which is ignored.
- Check the licence first. If a dataset's terms do not permit redistribution,
  document how to fetch it instead of committing it.
