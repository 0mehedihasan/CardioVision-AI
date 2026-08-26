---
name: test-case-validation
description: Work on the case store, auth, and the evidence/report layer — the parts verified against a real temporary database and fixture responses.
---

# Case and report validation

## Purpose

Everything between a finished analysis and a saved case: the SQLite store, the
session layer, the deterministic evidence aggregation and the report schema. This is
the largest verified surface in the repository — 341 of the 761 checks.

## When to use

- Adding a column, a migration or a denormalised list field
- Changing auth, token lifetime, lockout or the session check
- Changing the evidence status vocabulary, an uncertainty, a cross-modal observation
  or a recommendation
- Changing the report key set or the MedGemma prompt construction

## Relevant files

| File | Role |
| --- | --- |
| `src/cardiovision/services/database.py` | store, migrations, denormalised columns, per-case image files, `_write_gitignore` |
| `src/cardiovision/services/auth.py` | fixed operator account, salted hash, constant-time compare, in-memory sessions, lockout |
| `src/cardiovision/services/case_context.py` | the text block handed to MedGemma; withholds name and MRN |
| `src/cardiovision/fusion/schema.py` | status vocabulary, `CrossModalObservation`, `Uncertainty`, `EVIDENCE_MODALITIES` |
| `src/cardiovision/fusion/evidence.py` | per-modality evidence, clinical normalisation, cross-modal observations, uncertainties, recommendations |
| `src/cardiovision/fusion/report.py` | report assembly and the report prompt |
| `src/cardiovision/api/routers/cases.py`, `report.py`, `qa.py` | the routes |
| `tests/test_case_lifecycle.py` | 207 checks against a real temporary SQLite file |
| `tests/test_report_evidence.py` | 134 checks over fixtures shaped exactly like router responses |

## Expected inputs

Fixtures, not patients. `test_case_lifecycle.py` creates a **temporary** database
file; `test_report_evidence.py` builds dictionaries shaped like real analyse
responses. Never point a test at `data/`, and never copy a real study into a fixture.

## Expected outputs

- Case rows whose denormalised list columns agree with the stored payload
- Evidence with `available_modalities` in `EVIDENCE_MODALITIES` order — `ccta`,
  `echo`, `ecg`, **not** alphabetical
- A report with the fixed key set, `schema_version` `"1.0"`, and the
  `interpretation` recommendation **last**

## Important constraints

- **All columns belonging to one modality move together or not at all.** An earlier
  version protected the payload with `COALESCE` while overwriting the denormalised
  list columns unconditionally, so editing a patient's name left the sidebar
  advertising a stale finding count with the real result one click away.
- Age is **derived** from the date of birth on every read, never stored. An age typed
  in once is wrong a year later.
- The four statuses — `analysed`, `not_provided`, `provided_not_analysed`, `no_model`
  — each ship with their own `STATUS_MEANING` string, so a client never has to guess.
- Every `CrossModalObservation` carries `inference: "none"`. The phrases *consistent
  with*, *confirms*, *corroborates*, *suggestive of*, *rules out* and *risk score* are
  asserted absent. Co-occurrence only.
- An unticked risk-factor checkbox means **UNKNOWN, left blank** — never "not
  reported", which reads as a negative history nobody took.
- `patient.patient_id` reads `mrn` and falls back to `patientId`. Neither the name nor
  the MRN is ever sent to the language model.
- The structured report is identical with or without MedGemma. When generation fails,
  `ai_summary_error` is populated and the rest stands.
- `GET /api/cases/{id}` returns image **endpoints**, not inline base64; `name` on the
  image route is validated against path traversal, and the suite tests that directly.
- Logout is deliberately open, so signing out with an expired token succeeds quietly.
- **Never weaken a test to make it pass.** If behaviour changed on purpose, fix the
  assertion to match the new truth and keep the check.

## Verification steps

```bash
python3 tests/test_case_lifecycle.py
python3 tests/test_report_evidence.py
pytest -q                                 # the documented wrapper
```

Register any new suite in **both** the `SUITES` tuple in `tests/test_all.py`
(cheapest-first) and a step in `.github/workflows/ci.yml`. No HTTP request is
executed in the sandbox — FastAPI, pydantic and uvicorn are absent — so the routers
are verified against the modules they call, not over the wire.
