---
description: Report every model in the repository with its real metrics and its stated weaknesses.
---

Report the model inventory. Read `src/cardiovision/config.py` first — it is the
source of truth, annotated with the notebook cell each number came from — then
`.claude/memory/models.md` and `docs/models.md` for context.

For each of the three trained models give: architecture, checkpoint filename and
whether it resolves on disk, dataset, split unit and sizes, held-out test metrics,
threshold, and the stated weaknesses. Then MedGemma. Then state what has **no**
model.

Points that must appear, because averaging them away is the failure mode:

- **CCTA is n = 3.** Every metric is a `{mean, sd, min, max}` spread over three test
  cases (IDs 9, 14, 15). Dice mean 0.5996, and HD95 spans 82–131 mm — overlap is
  moderate, geometric fidelity is not established. Report all three
  `CCTA_WEAK_NOTES`.
- **ECG HYP is weak.** Macro AUROC 0.9125 hides AP 0.4777 and precision 0.3614 at
  threshold 0.5 — roughly two in three positive HYP calls are wrong.
- **Echo outlines anatomy, not function.** No EF, no strain, no volumes.
- **There is no learned fusion model and no clinical-risk model.**
  `MODALITY_STATUS["fusion"]` and `["clinical"]` are `available: False`;
  `notebooks/04_Multimodal_Fusion.ipynb` is empty; `src/cardiovision/fusion/` is a
  deterministic aggregator.
- Validation metrics are read from the checkpoint at load time and are **not** an
  independent estimate — the validation split steered early stopping and checkpoint
  selection.
- Every metric is a **dataset-level** figure, never a confidence score for an
  individual prediction.

Flag any disagreement between `config.py`, `docs/models.md` and `.claude/memory/models.md`.
`config.py` wins; the others are stale and should be corrected.
