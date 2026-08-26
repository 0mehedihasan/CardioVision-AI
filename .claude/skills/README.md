# Skills

Task-specific working notes for the six workflows this repository actually
supports. Each is a real path through code that exists; none describes a planned
feature.

| Skill | Use it when |
| --- | --- |
| [`echo-inference`](echo-inference/SKILL.md) | working on echocardiography segmentation — upload, orientation, per-structure areas |
| [`ccta-inference`](ccta-inference/SKILL.md) | working on CCTA coronary lumen segmentation — volume loading, sliding window, volumetry |
| [`ecg-inference`](ecg-inference/SKILL.md) | working on 12-lead ECG classification — signal loading, the five superclasses, the HYP caveat |
| [`xai`](xai/SKILL.md) | touching saliency or Grad-CAM in any modality, or the figures that render them |
| [`test-case-validation`](test-case-validation/SKILL.md) | working on the case store, auth, or the report/evidence layer |
| [`model-verification`](model-verification/SKILL.md) | checking that a checkpoint loads and that its metrics match `config.py` |

## Deliberately absent

There is no `fusion-training` skill, no `clinical-risk` skill and no
`train-model` skill, because there is no learned fusion model, no clinical-risk
model, and no training pipeline in this repository. `notebooks/04_Multimodal_Fusion.ipynb`
is empty. Do not add a skill for a workflow that does not exist — a skill file is
read as a statement that the workflow is supported.

## Shared rules for every skill

1. Read the module before changing it. `src/cardiovision/config.py` is the single
   source of truth for every constant and metric.
2. Do not retrain, and do not swap a checkpoint.
3. Every suite runs without torch. Keep new logic testable without a forward pass.
4. Uncertainty is a feature. See [`../rules/medical-ai.md`](../rules/medical-ai.md).
