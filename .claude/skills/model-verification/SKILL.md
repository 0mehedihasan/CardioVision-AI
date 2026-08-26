---
name: model-verification
description: Check that a checkpoint exists, loads, and matches the architecture and metrics declared in config.py — with or without torch installed.
---

# Model verification

## Purpose

Answer three questions before trusting anything a model says: does the checkpoint
resolve, does its parameter set match the constructed architecture, and do the
metrics reported by the API match `config.py` and the notebook they came from.

## When to use

- After changing any architecture constant in `config.py`
- After a checkpoint is added, moved or replaced
- When `/api/health` reports a model unavailable and the reason is unclear
- Before quoting a metric anywhere — README, docs, UI, prompt

## Relevant files

| File | Role |
| --- | --- |
| `src/cardiovision/config.py` | the checkpoint paths, every architecture constant, every published metric with the notebook cell it came from, `MODALITY_STATUS` |
| `src/cardiovision/cli.py` | `cardiovision check` — reports which checkpoints and which device would load |
| `tests/checkpoint_reader.py` | reads a `.pt`/`.pth` **without torch**, recovering tensor shapes and metadata by unpickling with a custom `find_class` |
| `tests/torch_stub.py` | enough of `torch.nn` to construct the pure-torch networks — `ECGResNet1D` is actually instantiated, `Small3DUNet` is imported; the echo model needs `segmentation_models_pytorch` and is not built under the stub. Every numeric entry point raises loudly |
| `tests/test_ecg_architecture.py` | 45 checks comparing the constructed architecture against the checkpoint's own parameter names and shapes |
| `src/cardiovision/api/routers/health.py` | what `/api/health` reports |

## Expected inputs

A checkpoint file on disk. If git-lfs is missing it will be ~130 bytes of pointer
text — check for that first, because every downstream error will be a confusing
serialisation failure.

## Expected outputs

```bash
cardiovision check                              # which checkpoints resolve, which device
python3 tests/checkpoint_reader.py models/ecg/cardioVision_ptbxl_ecg_resnet1d_full.pt
python3 tests/test_ecg_architecture.py
curl -s localhost:8000/api/health
curl -s -H "Authorization: Bearer $T" localhost:8000/api/models/ccta
```

`checkpoint_reader.py` returns shapes and metadata only. Nothing in it can produce a
prediction, and it is not a substitute for loading the model properly.

## Important constraints

- **Validation metrics are read out of the checkpoint at load time, not hardcoded**,
  so they cannot drift from the weights. They are displayed apart from the test
  metrics because the validation split steered early stopping and checkpoint
  selection — it is not an independent estimate. Do not merge the two.
- `config.py` is the source of truth. If `docs/models.md` and `config.py` disagree,
  the docs are stale.
- Preserve the exact checkpoint filenames; they are resolved literally. See
  [`../rules/git-and-models.md`](../rules/git-and-models.md) for the second-copy
  hazards, in particular `latest_3d_unet_cca_v2.pth` (later epoch, worse, different
  threshold).
- Never substitute a placeholder for a missing checkpoint. Absence must stay absence
  so `/api/health` can report it and the UI panel can say so.
- A green test run does **not** mean a forward pass happened. Under the torch stub
  nothing numeric executes, and each suite prints a footer saying so.
- `MODALITY_STATUS` is the single source of truth for what exists. `fusion` and
  `clinical` are `available: False` and must stay that way until weights exist.
- The ECG class order is the checkpoint's own column order and is cross-checked
  against `config.py` at load time. Keep that check.

## Verification steps

1. `git lfs pull`, then confirm each checkpoint is its real size, not ~130 bytes.
2. `cardiovision check` — every intended model resolves, the device is the expected
   one.
3. `python3 tests/test_ecg_architecture.py` — parameter names and shapes agree.
4. `pip install -e . && cardiovision serve`, then `GET /api/health` and
   `GET /api/models/{echo,ccta,ecg}`; compare every number against `config.py`.
5. Upload one input per modality. This is the only step that proves a forward pass
   ran. Nothing in CI does it — see [`../../docs/verification.md`](../../docs/verification.md).
