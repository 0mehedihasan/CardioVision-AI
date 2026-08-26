---
name: echo-inference
description: Run or modify echocardiography segmentation — UNet++ / EfficientNet-B3 over a single 2-D frame, four classes, per-structure areas.
---

# Echo inference

## Purpose

Segment four cardiac structures (background, LV cavity, myocardium, left atrium)
from one echocardiography frame, quantify each, and return the frame with its
overlays. This is the only modality that accepts an ordinary image file.

## When to use

- Adding or changing an echo upload format, frame selection, rotation or mirroring
- Changing how structure areas are computed or reported
- Debugging an orientation complaint, a wrong cm² figure, or a missing saliency tab
- Working on `POST /api/analyze/echo` or `GET /api/models/echo`

## Relevant files

| File | Role |
| --- | --- |
| `src/cardiovision/config.py` | `ECHO_CHECKPOINT_PATH`, `ECHO_IMAGE_SIZE`, `ECHO_CLASS_NAMES`, `ECHO_PRESENCE_THRESHOLD_PX`, `ECHO_SALIENCY_CLASS`, `ECHO_TRAINING_ORIENTATION`, `ECHO_TEST_METRICS`, `ALLOWED_ECHO_SUFFIXES`, `MAX_UPLOAD_BYTES` |
| `src/cardiovision/preprocessing/image_io.py` | decode PNG/JPEG/NIfTI/DICOM, frame selection, rotate/flip, pixel-spacing bookkeeping |
| `src/cardiovision/inference/echo.py` | `EchoSegmenter.load / analyze / describe / _quantify / _forward_with_saliency`, `StructureFinding`, `EchoAnalysis`, `EchoModelUnavailable` |
| `src/cardiovision/rendering/echo.py` | original / mask / overlay / saliency / combined PNGs |
| `src/cardiovision/api/routers/echo.py` | the route, its parameters and its error codes |
| `frontend/src/components/` | the echo result panel |

## Expected inputs

| Parameter | Notes |
| --- | --- |
| `file` | `.png .jpg .jpeg .nii .nii.gz .gz .dcm .dicom`, ≤200 MB |
| `frame` | int ≥0; frame index for a multi-frame DICOM cine loop |
| `rotate` | must be one of 0, 90, 180, 270 — anything else is `422` |
| `flip` | mirror horizontally before inference |
| `include_mask` | return the raw class mask for client-side rendering |
| `case_id` | archive the upload against a case |

## Expected outputs

- One finding per structure: name, present, pixel count, area in **cm² when the
  source carries pixel spacing** (NIfTI, DICOM) and **percent of the image field**
  otherwise (PNG, JPEG), plus the presence threshold used
- Base64 PNGs: original, mask, overlay, saliency, saliency overlay, combined
- With `include_mask`, the raw class mask as a flat row-major array with colour and
  name maps
- The device actually used, including `fell back from mps`
- An orientation note when the input did not match the training distribution

## Important constraints

- **Nothing is rotated by default.** `ECHO_TRAINING_ORIENTATION` is
  `"sector apex left, beam opening right"` — CAMUS NIfTI array order as stored.
  Conventional displays are apex-**up**, a quarter turn out of distribution. Report
  the mismatch and offer the rotations; never auto-rotate. A silent correction
  removes the only way to tell a real segmentation from a lucky one.
- A structure is present only above `ECHO_PRESENCE_THRESHOLD_PX = 50` labelled
  pixels. Below that it is argmax noise. Surface the threshold; "Not identified" is
  not a clinical statement of absence.
- On a quarter turn, row and column spacing swap with the pixels, so cm² stays
  correct and total field area is rotation-invariant. The suite pins this.
- **This model outlines anatomy.** No ejection fraction, no strain, no volumes over
  a cardiac cycle, no functional index. EF may appear only inside a limitation
  string, never as a `measurement`.
- Saliency keys are **absent**, not empty, when the gradient could not be computed.
- Timing is measured inside the model lock, so a queued request does not report its
  wait as compute time.

## Verification steps

```bash
python3 tests/test_report_evidence.py     # echo evidence + the no-EF assertion
python3 tests/test_all.py                 # or the whole set
```

Then, on a machine with the real stack: upload a frame, check the reported device
and orientation note, confirm areas are cm² for NIfTI/DICOM and percent-of-field for
PNG/JPEG, rotate 90° and confirm the total field area is unchanged.
