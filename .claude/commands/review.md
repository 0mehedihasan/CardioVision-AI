---
description: Review a change against this project's layering, honesty and privacy rules.
---

Review the current diff. Read the changed files in full, not just the hunks.

```bash
git status --short
git diff
git diff --cached
```

Check, in this order — the first three are the ones that silently produce a wrong
medical statement:

1. **Honesty.** Does anything claim more than the model measured? Look for a
   diagnosis, `confirms`, `consistent with`, `rules out`, `suggestive of`,
   `corroborates`, `risk score`, a stenosis grade, a CAD-RADS category, a calcium
   score, an ejection fraction as a value, or a dataset metric presented as a per-case
   confidence. Are the weak-class and weak-model notes still surfaced? Is an unticked
   risk factor still described as unknown rather than denied?
2. **Privacy.** Any patient name, MRN, date of birth or DICOM remnant in a fixture,
   docstring, example or commit message? Is name/MRN still withheld from every prompt?
   Anything under `data/` newly stageable? Any secret, token or password?
3. **Uncertainty preserved.** CCTA metrics still `{mean, sd, min, max}` with `n=3`?
   Every ECG probability still returned? Validation metrics still read from the
   checkpoint rather than hardcoded? Saliency still *absent* rather than zeroed when
   the gradient fails?
4. **Layering.** `config.py` importing no torch at module scope and nothing from the
   project except `cardiovision.__version__`; `preprocessing/` not importing `inference/`; `rendering/` not
   importing torch; `fusion/` not importing a model; no arithmetic in a router.
5. **Tests.** Is new logic covered? Was any assertion weakened or deleted? Is a new
   suite registered in **both** `tests/test_all.py` and `.github/workflows/ci.yml`?
6. **Hygiene.** No absolute paths (`/Users/`, `/kaggle/`, `/root/`, `/content/`); no
   new dependency without a reason; no `.pt`/`.pth` staged unintentionally; no
   filesystem path in an API error response; checkpoint filenames unchanged.
7. **Scope.** Does the diff do only what was asked? Flag unrequested refactoring.

Report findings by severity, each with the file and line and the concrete
consequence. Do not fix anything unless asked. Do not commit.
