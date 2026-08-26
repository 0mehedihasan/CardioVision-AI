# Project — CardioVision AI

> Verified against the repository. Code and configuration are the source of
> truth; if this file disagrees with `src/cardiovision/config.py`, the config
> wins and this file is stale.

## What it is

A **locally deployed cardiovascular AI research and software prototype**. Three
independently trained deep-learning models analyse three modalities, a
deterministic software layer collects what they reported, and a local
vision-language model (MedGemma 1.5 4B IT) writes a structured narrative over
that collected evidence. A React single-page app drives it.

Everything runs on one machine. There is no outbound network call anywhere in
`src/cardiovision/` at inference time — the echo encoder is rebuilt with
`encoder_weights=None` precisely so nothing is downloaded.

## Purpose

| Goal | How the repository serves it |
| --- | --- |
| Demonstrate multimodal cardiovascular inference end to end | Three trained checkpoints, three `/api/analyze/*` endpoints, one UI |
| Keep model weakness visible instead of averaged away | `ECG_WEAK_CLASSES`, `CCTA_WEAK_NOTES`, per-class metrics surfaced through the API and the prompt |
| Stay reproducible from the training record | Every metric in `config.py` cites the notebook that produced it |
| Stay honest about what is *not* trained | `MODALITY_STATUS` is the single source of truth, reported through `/api/health` |

This is a **research and software prototype**. It is not a medical device, has no
regulatory clearance, and has not been validated on any data outside the public
held-out splits described in `memory/data.md`.

## Current scope — three trained modalities

| Modality | Trained model | What it outputs |
| --- | --- | --- |
| Coronary CT angiography (CCTA) | `Small3DUNet` | Binary coronary **lumen** mask of a CT volume |
| Echocardiography (echo) | `UnetPlusPlus` / `timm-efficientnet-b3` | 4-class anatomical segmentation (LV cavity, myocardium, left atrium) |
| Electrocardiography (ECG) | `ECGResNet1D` | 5-class multi-label screening (NORM, MI, STTC, CD, HYP) |

Two further entries exist in `MODALITY_STATUS` with `available: False` and have
**no model at all**:

- **Clinical risk** — clinical values are collected in the form and passed to
  MedGemma as text context. There is no risk model and no clinical-risk notebook.
- **Multimodal fusion** — `notebooks/04_Multimodal_Fusion.ipynb` is a **0-byte
  file** and `models/fusion/` is an **empty directory**.

What exists instead of fusion is `src/cardiovision/fusion/`, which is
**deterministic software, not a network**: it aggregates already-computed model
outputs, records which modalities were missing, and lists uncertainties. Every
cross-modal item it emits is tagged `inference: "none"` and phrased as
co-occurrence. See `rules/medical-ai.md`.

## Relationship to OpenMedImaging

**Keep these separate.** OpenMedImaging is a broader future platform vision.
CardioVision AI is **not** that platform, must not be described as it, and must
not be extended toward it inside this repository.

- This repository is cardiovascular only: three modalities, one local operator
  account, one SQLite case store.
- Nothing here is a general medical-imaging framework. There is no plugin system,
  no modality registry, no multi-tenancy, no remote inference.
- Do not add abstraction whose only justification is a future platform. See
  `rules/project.md`.

## The pipeline as built

```
CCTA volume ─┐
Echo frame  ─┼─> per-modality inference (independent, lazy-loaded)
ECG record  ─┘        │
Clinical form ────────┤
                      v
        fusion/evidence.py    deterministic aggregation
                      v
        fusion/report.py      structured report + prompt
                      v
        inference/medgemma.py narrative over the structured evidence
                      v
        React frontend
```

Each stage degrades independently. A missing checkpoint costs exactly one
modality, and `/api/evidence` needs no model at all — it aggregates results that
were already computed, so it answers correctly on a server where nothing loaded.

## Non-negotiables

1. **No training happens in this repository.** The three checkpoints are fixed
   pretrained artefacts. Do not retrain, do not add a training pipeline, do not
   generate synthetic paired multimodal data.
2. **No fabricated output.** If a model does not produce a number, no layer above
   it may invent one.
3. **Absence is not normality.** Unanalysed regions, blank form fields and
   below-threshold structures are reported as unknown, never as normal.
