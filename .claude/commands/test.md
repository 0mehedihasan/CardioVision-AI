---
description: Run the existing verification suites and report failures honestly.
---

Run the existing suites. Do not write new ones for this command, and **never modify
a test to make it pass**.

```bash
pytest -q                                  # the documented wrapper
python3 tests/test_all.py                  # or the runner directly
```

If a suite fails, run it alone for the per-assertion report:

```bash
python3 tests/test_case_lifecycle.py       # 207 checks, real temporary SQLite file
python3 tests/test_report_evidence.py      # 134 checks, evidence + report assembly
python3 tests/test_ccta_pipeline.py        # 108 checks
python3 tests/test_ecg_pipeline.py         # 114 checks
python3 tests/test_ecg_rendering.py        # 100 checks
python3 tests/test_ecg_reporting.py        #  53 checks
python3 tests/test_ecg_architecture.py     #  45 checks; skips on an LFS pointer
```

Report:

- the total (**761** when everything runs) and the number that actually executed
- every failure by the **invariant it names**, not just the line number
- whether the torch stub was installed — each suite prints an honest footer saying
  no forward pass ran and the inputs were synthetic
- which suites skipped, and why

Then state plainly what the run does **not** prove. A green run means the wiring, the
schema, the prompt construction, the figure rendering and the
architecture-vs-checkpoint comparison hold. It does not mean a forward pass happened.
See `docs/verification.md`.

If a test fails because behaviour changed on purpose, fix the assertion to match the
new truth. Do not weaken the check and do not delete it.
