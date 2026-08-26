---
description: Report the current state of the repository — working tree, models on disk, and what is verified.
---

Report the current state. Read, do not change.

```bash
git status --short
git log --oneline -10
ls -la models/echo models/ccta models/ecg
du -sh data 2>/dev/null
```

Then report:

1. **Working tree** — modified, staged and untracked paths. If git-lfs is not
   installed, every `.pt`/`.pth` reads as modified; say so rather than treating it as
   a real change, and never suggest `git add -A`.
2. **Checkpoints** — which of the three resolve at their real size, and which are
   ~130-byte LFS pointer text. MedGemma is untracked by design; report whether
   `models/medgemma-1.5-4b-it/` is present.
3. **Patient data** — confirm nothing under `data/` is tracked or stageable
   (`git check-ignore -v data/cardiovision.db data/cases`).
4. **Verified vs unverified** — quote `.claude/memory/current-state.md` for what is
   COMPLETED versus EXISTS BUT NEEDS VERIFICATION, and `docs/verification.md` for
   what no test can prove in the current environment.

Do not commit. Do not clean anything up.
