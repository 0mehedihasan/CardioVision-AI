## What changed

<!-- One or two sentences. -->

## Why

<!-- The reasoning. If this changes a number, name the notebook, checkpoint or
     measurement it came from. -->

## What I ran

- [ ] `python tests/test_case_lifecycle.py`
- [ ] `python tests/test_ecg_architecture.py`
- [ ] `python tests/test_ecg_pipeline.py`
- [ ] `python tests/test_ecg_rendering.py`
- [ ] `python tests/test_ecg_reporting.py`
- [ ] `ruff check src tests`
- [ ] `cd frontend && npm run lint && npm run build`
- [ ] Loaded a real study through the running app

## What I could not verify

<!-- Required, and not a weakness. "No torch on this machine, so no forward pass
     ran" is exactly the kind of thing reviewers need to know. Write "nothing"
     only if you really did exercise every path this touches. -->

## Clinical honesty checklist

- [ ] No output is produced for a modality that has no trained model
- [ ] Any averaged metric is shown alongside the per-class numbers it hides
- [ ] No explainability figure is rendered from a computation that failed
- [ ] Nothing new implies a diagnosis, a risk score, or per-case confidence
- [ ] Caveats live in the component that renders the number, not only in a docstring
- [ ] No patient data in the diff, the tests, or the screenshots
