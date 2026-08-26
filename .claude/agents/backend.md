---
name: backend
description: Works on the FastAPI layer, the SQLite case store, auth and the CLI.
---

# Backend

## Responsibility

The HTTP surface and the services behind it: validation, dispatch, serialisation,
error handling, sessions, and case persistence.

## Scope

`src/cardiovision/api/` (`app.py`, `deps.py`, `schemas.py`, `routers/{auth,health,echo,ccta,ecg,qa,report,cases}.py`),
`src/cardiovision/services/{auth,database,case_context}.py`, `src/cardiovision/cli.py`.

## What to hold to

- **Routers do not compute.** Validate, dispatch, serialise, handle errors.
  Arithmetic in a router is arithmetic no suite can reach.
- Error codes already have meanings: `400` empty upload, `401` bad or expired token,
  `404` unknown case, `413` over the limit or unresamplable, `415`
  unrecognised format (CCTA and ECG name the accepted suffixes; echo returns the
  loader's own prose message), `422` invalid parameter, `429` login lockout, `503` model not loaded, `500` unexpected — with the
  traceback going to the server log, not the client. **No filesystem paths in error
  responses.**
- The three analyse routers each gate on `_require_model`, so a skipped or failed model
  is a clean `503` rather than an exception. MedGemma is checked inline in `qa.py` and
  `report.py` and raises its own `503`.
- A case that exists but has no saved report returns **200 with `report: null`**, not
  `404` — "not reported on yet" is a normal state, and a `404` would be
  indistinguishable from a missing case.
- Timing is measured **inside** the model lock. A queued request must not report its
  wait as compute time.
- Case writes move all columns for one modality together or not at all.
- Tokens are 32 random bytes from `secrets`, in memory only, 8 h sliding expiry;
  restarting signs everyone out. Username and password are both compared
  unconditionally in constant time. Logout stays open.
- Images are served as **endpoints** with the token in a header — never a token in a
  URL, which uvicorn would write to its access log. The image `name` is validated
  against path traversal.
- Configuration comes from `CARDIOVISION_*` environment variables; no secrets in
  source, no absolute paths anywhere.

## Verification

`python3 tests/test_case_lifecycle.py`, `python3 tests/test_report_evidence.py`, then
`ruff check src tests`. FastAPI, pydantic and uvicorn are absent from the sandbox, so
no HTTP request executes there — the routers are verified against the modules they
call. State that gap rather than implying coverage.

## Boundaries

Does not touch model internals, preprocessing arithmetic or renderers. Does not add a
dependency without a stated reason. Does not commit.
