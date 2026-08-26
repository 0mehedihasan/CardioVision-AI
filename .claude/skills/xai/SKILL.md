---
name: xai
description: Work on explainability — echo input-gradient saliency, CCTA 3-D Grad-CAM, ECG per-lead attribution, and the figures that render them.
---

# Explainability

## Purpose

Three different attribution methods, one per modality. They are not
interchangeable and must not be labelled as each other.

| Modality | Method | Target |
| --- | --- | --- |
| Echo | **input-gradient attribution** — absolute gradient of the mean LV-cavity probability w.r.t. the input | `ECHO_SALIENCY_CLASS = 1` |
| CCTA | **3-D Grad-CAM** at layer `enc3.block[-1]` | the patch with the most predicted lumen |
| ECG | **per-lead gradient attribution**, computed at request time | `target_class`, defaulting to the top class |

## When to use

- Changing how any attribution is computed, targeted or normalised
- Changing a saliency figure, colour map or overlay compositing
- Debugging a missing saliency tab, a flat Grad-CAM panel, or an MPS fallback

## Relevant files

| File | Role |
| --- | --- |
| `src/cardiovision/inference/echo.py` | `_forward_with_saliency` |
| `src/cardiovision/inference/ccta.py` | Grad-CAM hook and the patch choice |
| `src/cardiovision/inference/ecg.py` | per-lead attribution, `LeadAttribution` |
| `src/cardiovision/rendering/primitives.py` | `apply_jet`, `jet_hex`, `to_png_data_url`, `to_svg_data_url`, `xml_text`. Overlay blending is **not** here — each of `rendering/echo.py` and `rendering/ccta.py` has its own private `_blend` |
| `src/cardiovision/rendering/echo.py`, `ccta.py`, `ecg.py` | the figures |
| `src/cardiovision/config.py` | `ECHO_SALIENCY_CLASS`, the Grad-CAM layer name |

## Expected inputs and outputs

Renderers take **arrays**, never tensors, and `rendering/` must not import torch —
that is what lets the rendering suites run against real `.npy` artefacts in `models/`
with no forward pass. Attribution maps arrive as numpy arrays already detached and
reduced.

Outputs are base64 PNG (echo, CCTA) or SVG (ECG), plus the raw attribution where the
frontend needs to draw its own canvas.

## Important constraints

- **The echo method is not Grad-CAM and must never be labelled as such.** It is an
  input gradient. Calling it Grad-CAM in a tooltip, a docstring or a report is a
  false statement about the method.
- **If the gradient cannot be computed, hide the output entirely.** The echo saliency
  keys are *absent* rather than empty, and the UI hides the tabs. An all-zero
  gradient still renders as a smooth, convincing picture, which is worse than showing
  nothing.
- Attribution is not evidence of correctness. A confident-looking heatmap over a
  wrong mask is still a wrong mask. Never phrase a saliency figure as support for a
  finding.
- If a backward pass is unsupported on MPS, the **entire forward pass** for that
  request falls back to CPU and the response reports `fell back from mps`. Do not
  mix devices within one request, and do not swallow the fallback.
- Grad-CAM costs one extra backward pass; `include_gradcam=false` must remain a real
  opt-out.
- The shipped example figures (`models/ccta/case_*_xai.png`,
  `models/ecg/HR00025_saliency.png`, `models/echo/__results___0_129.png`) are
  notebook provenance for three dataset cases and one record. They are never a
  patient result and never a population statement.

## Verification steps

```bash
python3 tests/test_ecg_rendering.py     # 100 checks on the 12-lead figure with shading
python3 tests/test_ccta_pipeline.py     # slice selection and the figure plumbing
```

No forward pass runs in the sandbox, so the *numbers* inside an attribution map are
unverified there — only the shapes, the colour handling and the absent-vs-empty
behaviour. On a real machine, confirm the CCTA Grad-CAM panel is a genuine overlay
rather than a flat map, and that removing gradient support hides the echo tabs rather
than filling them.
