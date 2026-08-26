---
name: ccta-inference
description: Run or modify CCTA coronary lumen segmentation — Small3DUNet, sliding-window inference over a resampled volume, one foreground class.
---

# CCTA inference

## Purpose

Segment contrast-filled coronary lumen from a CCTA volume and report its volume,
component structure and probability spread. One foreground class, sigmoid, threshold
0.60.

## When to use

- Adding or changing a volume format, the resample step or the HU window
- Changing the sliding window, the window budget or the coverage reporting
- Changing volumetry or the finding dictionary
- Working on `POST /api/analyze/ccta` or `GET /api/models/ccta`

## Relevant files

| File | Role |
| --- | --- |
| `src/cardiovision/config.py` | `CCTA_CHECKPOINT_PATH`, `CCTA_PATCH_SIZE`, `CCTA_INFERENCE_OVERLAP`, `CCTA_INFERENCE_BATCH_SIZE`, `CCTA_THRESHOLD`, `CCTA_MAX_WINDOWS`, `CCTA_PRESENCE_THRESHOLD_VOXELS`, `CCTA_TEST_METRICS`, `CCTA_WEAK_NOTES`, `ALLOWED_CCTA_SUFFIXES`, `MAX_CCTA_UPLOAD_BYTES`, `MAX_CCTA_VOXELS` |
| `src/cardiovision/preprocessing/ccta_io.py` | magic-byte format detection, 1 mm resample, HU window, `LoadedVolume` provenance |
| `src/cardiovision/inference/ccta.py` | `Small3DUNet`, `ConvBlock3D`, `compute_starts`, `_plan_windows`, `CctaSegmenter`, `LumenFinding`, `CctaAnalysis`, `CctaModelUnavailable` |
| `src/cardiovision/rendering/ccta.py` | `_best_index` slice selection, slice views, Grad-CAM overlay |
| `src/cardiovision/api/routers/ccta.py` | the route, its parameters and its error codes |
| `tests/test_ccta_pipeline.py` | 108 checks, runs without torch, SciPy or nibabel |

## Expected inputs

| Parameter | Notes |
| --- | --- |
| `file` | `.nii`, `.nii.gz`, `.gz`, or a `.zip` holding one DICOM series. ≤800 MB, ≤200 M voxels |
| `max_windows` | 1–4000, default `CCTA_MAX_WINDOWS` (600) |
| `include_gradcam` | 3-D Grad-CAM over the patch with the most predicted lumen; one extra backward pass |
| `include_figures` | slice, overlay, probability and projection panels — roughly 1–2 MB of PNG |
| `case_id` | archive the upload |

Format is detected by **magic bytes before suffix**: `PK\x03\x04` → zip,
`\x1f\x8b` → gzipped NIfTI, a 348-byte LE/BE header → NIfTI, `DICM` at byte 128 →
DICOM. An unrecognised file is refused with `415` naming the accepted extensions; a
volume too large to resample returns `413`.

## Expected outputs

One lumen finding: `name, present, voxels, volume_ml, fraction_of_analysed,
percent_of_analysed, mean_probability, max_probability, components,
largest_component_fraction` — plus the input summary, the coverage statement, the
declared status, and the figures.

## Important constraints

- **Resampling to a coarser grid reduces the voxel count.** 256 voxels at 0.5 mm is
  128 mm of tissue, which is 128 voxels at 1 mm. Getting this backwards scales every
  reported volume by 8× and nothing else complains. The suite pins the direction.
- `compute_starts(length, patch, overlap)` returns `[0]` when the axis is no longer
  than the patch; otherwise it strides by `max(1, int(patch * (1 - overlap)))` and
  appends the trailing `length - patch` start when the stride would miss the end.
- When a volume needs more than the budget, `_plan_windows` shrinks the analysed
  region by `scale = (budget / total) ** (1/3)`, centres the crop, and the response
  **says the coverage was partial**. Never return a subvolume as if it were the
  whole study.
- Volume arithmetic: voxel count × voxel volume. 1000 voxels of 1 mm³ = 1000 mm³ =
  1 mL.
- `max_probability` is taken over the **whole** probability map, including voxels
  outside the thresholded mask — a near-threshold voxel is still worth seeing.
- Presence requires more than `CCTA_PRESENCE_THRESHOLD_VOXELS = 500` voxels.
- **Every CCTA metric is a `{mean, sd, min, max}` dict, never a bare float, and
  `n=3` travels with it.** Three observations support no confidence interval and no
  claim of generalisation. HD95 spans 82–131 mm: overlap is moderate, geometric
  fidelity is not established.
- One foreground class. No stenosis grade, no calcium score, no CAD-RADS category,
  no vessel labelling. All three `CCTA_WEAK_NOTES` must reach the API, the UI and
  the prompt.
- `models/ccta/case_{9,14,15}_xai.png` and `case_*_gradcam_full_resampled.nii.gz`
  are the notebook's figures for the three dataset test cases. They are provenance
  and must never be rendered as a patient result.
- `models/ccta/latest_3d_unet_cca_v2.pth` is a later but **worse** epoch with a
  different selected threshold. `config.py` points at `best_3d_unet_cca_v2.pth`.

## Verification steps

```bash
python3 tests/test_ccta_pipeline.py
```

The suite covers the **no-SciPy** branch, because SciPy is absent in the sandbox —
resampling and connected-component counting are unexecuted there. On a machine with
the real stack, upload a volume and confirm `n=3` travels with the Dice, that the
coverage note appears when the budget was short, and that the Grad-CAM panel is a
real overlay rather than a flat map.
