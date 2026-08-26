---
description: Find and describe the inputs actually available for manual testing. Do not invent test data.
---

Report what can actually be fed to this application right now. **Do not invent test
data, do not fabricate a patient, and do not generate a synthetic study and present
it as a study.**

```bash
ls -R samples | head -50
du -sh samples/*
git ls-files data/          # only data/.gitignore should appear
ls -la models/echo models/ccta models/ecg
```

Then report four groups, kept clearly separate:

1. **Accepted formats per modality**, from `config.py`:
   - echo — `.png .jpg .jpeg .nii .nii.gz .gz .dcm .dicom`, ≤200 MB
   - CCTA — `.nii .nii.gz .gz` or a `.zip` of one DICOM series, ≤800 MB and
     ≤200 M voxels
   - ECG — `.hea .dat .mat .csv .txt .tsv .npy .json .zip`; a `.hea` needs its `.dat` or `.mat` as a
     companion, matched by filename
2. **Dataset samples under `samples/`.** The only real modality inputs available
   locally, and enough for one pass per modality:
   - `samples/ccta/CCTA_CASE_01{4,5}/` — MedHK23/CCA volume + ground truth. Both are
     **test-split** cases (`models/ccta/cardiovision_cca_split.csv`), and two of the
     same three the published Dice was measured on — a reproduction, not an
     independent evaluation.
   - `samples/echo/patient000{1,2}/` — CAMUS, 2CH and 4CH, ED/ES/half-sequence with
     ground truth, gzipped. `MANDATORY_CITATION.md` ships alongside and its citation
     is required.
   - `samples/ecg/HR0000{1,2}.{hea,mat}` — PTB-XL in the `.hea` + `.mat` packaging the
     model was trained from. Upload the `.hea` and send the `.mat` as a companion.
   Which split the echo and ECG samples came from is **not established in
   repository**, so do not describe them as held out. Read `samples/README.md` before
   reporting on them, and do not add more — the largest CCTA volume is 97 MB against
   GitHub's 100 MB limit.
3. **Artefacts under `models/`, and what they are not.** `case_{9,14,15}_xai.png`,
   `case_*_gradcam_full_resampled.nii.gz`, `HR00025_saliency.*`, `xai_*.npy`,
   `__results___*.png` and the CSV/JSON metric dumps are **notebook provenance** for
   dataset cases. They are renderer fixtures at most. They are not patient studies,
   they are not upload inputs, and they must never be displayed as a result.
4. **Where more inputs come from.** CAMUS (CREATIS), MedHK23/CCA and PTB-XL
   (PhysioNet, ODC-By — attribution required) must be obtained from their sources
   under their own terms. Nothing under `data/` is an input: `data/cases/` is
   `CASE_FILES_DIR`, where the store writes real patient files, and it is ignored.
   `data/sample/` does not exist, despite the `!data/sample/` negation in the root
   `.gitignore`.

If nothing suitable exists for the modality being tested, say so plainly and stop.
Suggesting a fabricated file would produce a plausible-looking result with no
provenance, which is the exact failure this project is built to avoid. The synthetic
arrays inside the suites are fine for arithmetic and are already there — they are not
clinical inputs.
