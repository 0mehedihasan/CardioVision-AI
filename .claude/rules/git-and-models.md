# Rules — git and model weights

## 1. Large weights go through Git LFS

`.gitattributes` routes `*.pth *.pt *.ckpt *.pkl *.joblib *.safetensors *.bin`
through LFS. The three served checkpoints are **tracked**, not ignored, because a
clone that installs cleanly and then cannot load a model is worse than a clone that
refuses to install.

```bash
git lfs install          # once per machine
git lfs pull             # in an existing clone
```

Without git-lfs, a `.pt`/`.pth` arrives as ~130 bytes of pointer text and the loader
reports a corrupt checkpoint. Two consequences for any agent working here:

- **Never run `git add -A` or `git commit -a` on a machine without git-lfs.** Every
  weight file reads as modified and a commit would replace real weights with pointer
  text — or pointer text with a truncated blob. Stage named files instead.
- A suite that needs a checkpoint must **detect the pointer file and skip with a
  message**, the way `tests/test_ecg_architecture.py` does. CI checks out with
  `lfs: false` on purpose so no run burns LFS bandwidth on a 46 MB file.

MedGemma never reaches LFS. `.gitattributes` does route `*.safetensors` and `*.bin`
through the filter, but `models/medgemma-1.5-4b-it/` — plus `*.safetensors` and `*.bin`
— are **gitignored**, so the filter never engages on them. It is 8.6 GB of vendor
weights under Google's terms, a `huggingface-cli download` away.

## 2. Never replace a large model file with a fake placeholder

Not a zero-byte file, not a text note, not a small random tensor, not under the real
filename.

**Why:** a dummy under a checkpoint's real name makes the loader fail deep inside
`torch.load` with a serialisation error instead of at a clear "checkpoint not found"
message — and it can be committed by accident as a legitimate change, because the
path is expected to exist.

If weights are missing, leave them missing. The loader already reports the file it
tried to open, `/api/health` reports the model as unavailable, and the UI panel says
so. That chain only works if absence stays absence.

## 3. Preserve the exact checkpoint filenames

These paths are resolved literally from `src/cardiovision/config.py`. A rename
surfaces as "checkpoint not found", nothing more helpful.

| Constant | File |
| --- | --- |
| `ECHO_CHECKPOINT_PATH` | `models/echo/cardiovision_echo_unetplusplus_best.pth` |
| `CCTA_CHECKPOINT_PATH` | `models/ccta/best_3d_unet_cca_v2.pth` |
| `ECG_CHECKPOINT_PATH` | `models/ecg/cardioVision_ptbxl_ecg_resnet1d_full.pt` |
| `MEDGEMMA_PATH` | `models/medgemma-1.5-4b-it/` (untracked) |

Note the capital V in `cardioVision_ptbxl_ecg_resnet1d_full.pt`. It is not a typo to
tidy; it is the filename on disk, and case-insensitive filesystems will hide the
breakage until someone deploys on Linux.

### The second copies are not the served weights

| File | What it is |
| --- | --- |
| `models/ccta/latest_3d_unet_cca_v2.pth` | a **later but worse** epoch, carrying a **different selected threshold**. Loading it would silently move the operating point the published metrics belong to |
| `models/echo/cardiovision_echo_unetplusplus_last.pth` | last epoch, kept as training provenance |
| `models/ecg/best_ecg_resnet1d.pt` | 46 MB training-time artefact, kept as provenance |

Do not "upgrade" `config.py` to a `latest_*` file because the name sounds newer. If
a checkpoint is ever swapped, the metrics in `config.py` and the tables in
`docs/models.md` must be replaced in the same change — a stale metric attached to new
weights is a fabricated result.

Validation metrics are read out of the checkpoint at load time rather than hardcoded,
so they cannot drift from the weights they describe. Keep it that way.

`tests/test_ecg_architecture.py` compares the constructed architecture against the
checkpoint's own parameter names and shapes, so a constant edited in `config.py`
without matching weights fails loudly instead of loading a mis-wired model. Run it
after any architecture-constant change.

## 4. Do not commit generated datasets or generated output

| Do not commit | Why |
| --- | --- |
| `data/cardiovision.db` (+ `-wal`, `-shm`) | patient records |
| `data/cases/` | rendered patient PNGs and original uploads |
| `frontend/dist/` | build output; gitignored, and `.gitattributes` marks it `linguist-generated` |
| `__pycache__/`, `*.egg-info/`, `.pytest_cache/`, `.ruff_cache/` | build and tool artefacts |
| `.DS_Store` | noise; gitignored, but several already exist untracked under `models/` |
| a resampled volume, a preprocessed array cache, a downloaded dataset | regenerable, and often large |

The notebook artefacts already under `models/` (`case_*_xai.png`,
`case_*_gradcam_full_resampled.nii.gz`, `HR00025_saliency.*`, the CSV/JSON metric
dumps) are tracked **provenance** for figures in the documentation. They are dataset
outputs, not patient results, and must never be rendered as one. Do not add more
without a stated reason.

## 5. Verify `.gitignore` before committing

```bash
git status --short                     # nothing under data/, no .DS_Store, no dist/
git check-ignore -v data/cardiovision.db data/cases
git diff --stat                        # no .pt/.pth in the list unless intended
```

`data/.gitignore` is written at startup by `CaseStore._write_gitignore()` and
only when the file is absent, so an older copy is never upgraded in place. Check it
against that function after changing either.

## 6. Committing

- **Do not commit automatically.** Stage and describe; the developer commits.
- Stage named paths, never `-A` or `-a` — see §1.
- Do not touch git config, do not force-push, do not reset --hard, do not rewrite
  history.
- One concern per commit. A checkpoint swap and a UI change are two commits.
