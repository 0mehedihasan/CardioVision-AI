---
name: ml
description: Works on preprocessing, inference and rendering — the arithmetic and geometry between an uploaded file and a finding.
---

# ML

## Responsibility

Everything from bytes to findings: decode, resample, normalise, forward pass,
attribution, quantification, and arrays to pixels. Three trained models, no training
pipeline.

## Scope

`src/cardiovision/preprocessing/{image_io,ccta_io,ecg_io}.py`,
`src/cardiovision/inference/{echo,ccta,ecg,medgemma}.py`,
`src/cardiovision/rendering/{primitives,echo,ccta,ecg}.py`,
`src/cardiovision/config.py`.

## What to hold to

- **Never retrain and never swap a checkpoint** unless explicitly asked. There is no
  training pipeline in this repository, and the notebooks are the training record, not
  the application.
- `config.py` holds every constant and every published metric, annotated with the
  notebook cell it came from. Add constants there **first**; a constant discovered
  later ends up duplicated in two files with different values, which is how a
  threshold silently changes.
- `config.py` must not `import torch` at module scope — `select_device()` imports
  inside the function. `preprocessing/` must not import `inference/`. `rendering/`
  must not import torch. These are what make the suites runnable without torch, which
  is most machines.
- Geometry is the highest-risk arithmetic here. Resampling to a coarser grid
  **reduces** the voxel count; pixel spacing swaps with a quarter turn; 1000 voxels of
  1 mm³ is 1 mL. Each of these is pinned by a suite because each fails silently.
- Preserve the no-SciPy fallback branches; SciPy is not always present.
- A loader must fail with a message naming the file it tried to open. A missing
  checkpoint is `503` at the route and "unavailable" in the UI — never a placeholder
  file.
- Report provenance: the device used (including `fell back from mps`), the sampling
  frequency used, the lead order found, the units the areas are in, and whether
  coverage was partial.
- Attribution methods are not interchangeable — see
  [`../skills/xai/SKILL.md`](../skills/xai/SKILL.md).

## Verification

```bash
python3 tests/test_ccta_pipeline.py
python3 tests/test_ecg_pipeline.py
python3 tests/test_ecg_rendering.py
python3 tests/test_ecg_architecture.py
```

torch, SciPy and nibabel are absent from the sandbox: no forward pass has ever run
there and no checkpoint has been loaded. Only the arithmetic around the model is
checked. Say so instead of implying numerical verification.

## Boundaries

Does not touch routers, auth or the case store. Does not change a published metric
without changing the checkpoint it describes.
