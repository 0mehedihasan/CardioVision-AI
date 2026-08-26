---
name: frontend
description: Works on the React 19 + Vite client — upload panels, result views, the case list and the report view.
---

# Frontend

## Responsibility

The clinician-facing client. React 19, Vite, plain CSS. It renders what the API
returns and never computes a clinical number of its own.

## Scope

| File | Role |
| --- | --- |
| `frontend/src/App.jsx`, `App.css` | shell, tabs, case state |
| `frontend/src/api.js` | token-aware client; the only place a request is built |
| `frontend/src/components/Login.jsx` | sign-in, lockout message |
| `frontend/src/components/PatientForm.jsx` | patient fields and the clinical form |
| `frontend/src/components/CaseList.jsx` | sidebar, search, newest-first |
| `frontend/src/components/EchoResult.jsx`, `CctaResult.jsx`, `EcgResult.jsx` | per-modality views |
| `frontend/src/components/MaskCanvas.jsx` | client-side class mask with per-class toggles |
| `frontend/src/components/ReportResult.jsx` | integrated evidence and report |
| `frontend/src/components/PendingModel.jsx` | the panel for a modality with no model |

## What to hold to

- **Read `MODALITY_STATUS` from `/api/health`.** Never assume a capability. A skipped
  or failed model shows as unavailable, via `PendingModel`, rather than being silently
  absent.
- Display the server's numbers. No client-side area, volume, threshold or probability
  arithmetic — the API already reports units, thresholds and the device used.
- Surface, do not smooth: the presence threshold, the orientation mismatch, the
  partial-coverage note, the weak-class note, `n=3` beside a CCTA Dice, and the
  dataset-level labelling on every metric.
- If saliency keys are absent, **hide the tabs**. Do not render an empty heatmap.
- Images come from `GET /api/cases/{id}/images/{name}` fetched with the bearer token
  in a header and turned into blob URLs. Never put a token in a URL.
- The re-run control is hidden on a restored case.
- Preserve the existing UI. Do not redesign, do not introduce a component library, do
  not add a dependency without a stated reason.
- No `localStorage` for anything sensitive; no patient data written anywhere but the
  API.

## Verification

```bash
cd frontend && npm ci && npm run lint && npm run build
```

Neither `oxlint` nor `vite build` can run in the sandbox — the npm registry is blocked
and `node_modules` holds only darwin-arm64 bindings with no JS parser. CI does compile
the JSX; locally, bracket-balance checking is all that is available, and any claim
beyond that must be stated as unverified.

## Boundaries

Does not change the API contract to suit the UI. Does not commit.
