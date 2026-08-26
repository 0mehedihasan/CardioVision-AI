# API reference

Base URL defaults to `http://127.0.0.1:8000`. All responses are JSON. All routes
except `/`, `/api/health`, `/api/auth/login` and `/api/auth/logout` require
`Authorization: Bearer <token>`.

Logout is deliberately open: signing out with an already-expired token should
quietly succeed rather than return `401`.

Interactive docs are served at `/docs` while the backend is running.

---

## Contents

- [Auth](#auth)
- [Health and model cards](#health-and-model-cards)
- [Analysis](#analysis)
- [Evidence and reports](#evidence-and-reports)
- [Clinical Q&A](#clinical-qa)
- [Cases](#cases)
- [Errors](#errors)

---

## Auth

### `POST /api/auth/login`

```bash
curl -X POST http://127.0.0.1:8000/api/auth/login \
  -H 'Content-Type: application/json' \
  -d '{"username": "medexpert", "password": "1111"}'
```

Returns a bearer token. Tokens are 32 random bytes from `secrets`, held in memory
only, expiring 8 h after last use with a sliding renewal. Restarting the backend
signs everyone out.

Five failed attempts lock the account for five minutes. Username and password are
both compared unconditionally and in constant time, so a wrong username takes
exactly as long as a wrong password.

### `GET /api/auth/session`

Validates the token the browser is holding. Used on page load so a stale tab
returns to the login screen with a reason instead of failing on its first real
request.

### `POST /api/auth/logout`

Revokes the token. Open route; succeeds for an expired token.

---

## Health and model cards

### `GET /`

Service banner. Open route.

### `GET /api/health`

Open route. Reports which models actually loaded, and `MODALITY_STATUS` — the
single source of truth for what exists. The frontend reads this rather than
assuming a capability, so a skipped or failed model is shown as unavailable
instead of being silently absent.

### `GET /api/models/echo` · `GET /api/models/ccta` · `GET /api/models/ecg`

The model card for one modality: architecture, preprocessing, published metrics
with their provenance, thresholds, and any weak-class or weak-model notes.

CCTA metrics arrive as objects `{mean, sd, min, max}`, not floats. `n=3` travels
with them.

---

## Analysis

All three are `multipart/form-data`, all three accept `case_id` to archive the
upload against a case, and all three report the device actually used — including
`fell back from mps` when a saliency backward pass forced a CPU pass.

Timing is measured **inside** the model lock, so a request that queued behind
another does not report its wait as compute time.

### `POST /api/analyze/echo`

| Parameter | Type | Default | Notes |
| --- | --- | --- | --- |
| `file` | file | required | PNG, JPEG, NIfTI, DICOM. ≤200 MB |
| `frame` | int ≥0 | none | frame index for multi-frame DICOM cine loops |
| `rotate` | int | `0` | counter-clockwise degrees; **must** be 0, 90, 180 or 270, else `422` |
| `flip` | bool | `false` | mirror horizontally before inference |
| `include_mask` | bool | `true` | include the raw class mask for client-side rendering |
| `case_id` | str ≤100 | none | archive the upload against this case |

Nothing is rotated by default. The model was trained apex-**left**; conventional
apex-up displays need a quarter turn, and the response says when the orientation
did not match rather than guessing.

Returns per-structure findings (area in cm² when the source carries spacing, else
percent of field), the presence threshold used, server-rendered base64 PNGs
(original, mask, overlay, saliency, saliency overlay, combined), and — when
`include_mask` — the raw class mask as a flat row-major array with its colour and
name maps, so the frontend can draw its own canvas with per-class toggles.

Saliency keys are **absent** rather than empty when the gradient could not be
computed.

### `POST /api/analyze/ccta`

| Parameter | Type | Default | Notes |
| --- | --- | --- | --- |
| `file` | file | required | `.nii`, `.nii.gz`, or a `.zip` of one DICOM series. ≤800 MB, ≤200 M voxels |
| `max_windows` | int 1–4000 | `600` | sliding-window budget |
| `include_gradcam` | bool | `true` | 3-D Grad-CAM over the patch with the most predicted lumen; one extra backward pass |
| `include_figures` | bool | `true` | slice, overlay, probability and projection panels — roughly 1–2 MB of PNG |
| `case_id` | str ≤100 | none | archive the upload |

When the budget is short of full coverage, the analysis covers a **centred crop**
and the response says so. It does not return a partial mask as if it were
complete.

Findings carry `name, present, voxels, volume_ml, fraction_of_analysed,
percent_of_analysed, mean_probability, max_probability, components,
largest_component_fraction`. `max_probability` is over the whole probability map,
including voxels outside the thresholded mask.

Format is detected by magic bytes before suffix; an unrecognised file is refused
with `415` naming the accepted extensions. A volume too large to resample returns
`413` with the suggestion to crop to the cardiac region.

### `POST /api/analyze/ecg`

| Parameter | Type | Default | Notes |
| --- | --- | --- | --- |
| `file` | file | required | `.hea`, `.dat`, `.mat`, `.csv`, `.txt`, `.tsv`, `.npy`, `.json`, `.zip` |
| `companions` | file[] | `[]` | other files of the same recording (e.g. the `.dat` or `.mat` for a `.hea`); matched by **filename**, not order, and not suffix-checked |
| `sampling_frequency` | float 0–10000 | none | source Hz for formats that do not record it (CSV, NPY). Getting it wrong rescales the recording in time, so the loader reports whatever it used |
| `target_class` | str | highest-probability class | which class the saliency explains; must be one of `NORM MI STTC CD HYP`, else `422` |
| `include_figures` | bool | `true` | 12-lead strip and lead-attribution chart, roughly 160 KB of SVG |
| `case_id` | str ≤100 | none | archive the upload |

Returns **every** class probability, not only the calls above the 0.5 threshold,
together with the operating point the published precision/recall belong to. A
positive `HYP` is flagged with its weak-class note in the response itself.

Lead order is reported as found; a non-standard order is surfaced, not silently
reordered.

---

## Evidence and reports

Both accept the same body: either an inline case object, or `case_id` to load one
from storage. The response reports which it used as `case_source`
(`"request"` or `"storage"`). An unknown `case_id` returns `404`.

### `POST /api/evidence`

Deterministic aggregation. **No language model is involved.**

Top-level fields include `available_modalities` (in `EVIDENCE_MODALITIES` order —
`ccta`, `echo`, `ecg` — not sorted), `missing_modalities` (which may contain
`"clinical"`), per-modality evidence, `cross_modal`, `uncertainties`,
`recommendations` and `model_versions`.

| Status | Meaning |
| --- | --- |
| `analysed` | the modality was uploaded and a model ran on it |
| `not_provided` | nothing was uploaded |
| `provided_not_analysed` | a file exists on the case but no result does |
| `no_model` | there is no trained model for this modality |

Each status ships with its own `STATUS_MEANING` string, so a client never has to
guess what it implies.

`uncertainties` entries have a `kind`, a `detail` and a `severity` of `note` or
`warning`. `cross_modal` observations have kinds `coverage`,
`pairing_provenance`, `co_occurrence`, `contradiction` or `negative_result`, and
**always** carry `inference: "none"`.

`model_versions` values are objects `{model, task, dataset}`. The `fusion` entry
is:

```json
{
  "model": null,
  "task": "deterministic software evidence aggregation",
  "note": "No learned fusion model exists in this project."
}
```

### `POST /api/report`

| Parameter | Type | Default | Notes |
| --- | --- | --- | --- |
| `include_summary` | bool | `true` | have MedGemma write the narrative. The structured report is identical either way |
| `include_prompt` | bool | `false` | return the exact text MedGemma was given, so a reader can check the narrative claims nothing the evidence did not |
| `save` | bool | `false` | store the finished report against the case |

Response keys, in order:

```
schema_version         "1.0"
case_id
generated_at
generated_by
patient                {name, patient_id, sex, date_of_birth, study_date}
modality_results
clinical_context
integrated_evidence
uncertainties
ai_summary
ai_summary_error       populated instead of ai_summary when generation fails
ai_summary_scope
recommendations
recommendations_scope
model_versions
disclaimer
```

`patient.patient_id` reads `mrn` and falls back to `patientId`. Neither the name
nor the MRN is ever sent to the language model.

`recommendations` entries have a `kind` of `analysis`, `input`, `capability`,
`coverage` or `interpretation`. The `interpretation` item is always **last**, and
it states that none of the models was validated for clinical use.

### `GET /api/cases/{case_id}/report`

The stored report for a case, if one was saved. A case that exists but has never
been reported on returns `200` with `report: null` — "not reported on yet" is a
normal state, and a `404` would be indistinguishable from a missing case.

---

## Clinical Q&A

### `POST /api/clinical-question`

MedGemma, locally, optionally with case context. Sends age, sex, study date,
notes, the clinical form and the structured findings. **Never** sends the patient
name or MRN.

The answer is folded into the case record — but only when a case already exists.
Asking a general cardiology question with no patient entered does not silently
create one.

---

## Cases

### `GET /api/cases`

Summaries, newest first. `?search=` filters. The list reads denormalised columns
so it never has to parse the stored JSON.

### `POST /api/cases`

Create or update. Patient fields, the clinical form, findings and the transcript.
Age is **derived** from the date of birth on every read rather than stored,
because an age typed in once is wrong a year later.

All columns belonging to one modality move together or not at all — an earlier
version protected the payload with `COALESCE` while overwriting the denormalised
list columns unconditionally, so editing a patient's name left the sidebar
advertising a stale finding count with the real result one click away.

### `GET /api/cases/{case_id}`

One full case — with image **endpoints**, not inline base64. Inlining six PNGs
would make every case fetch several megabytes. The frontend fetches them with the
same bearer token and makes blob URLs, so the token never appears in a URL and
cannot end up in uvicorn's access log.

### `DELETE /api/cases/{case_id}`

Removes the row and its files.

### `GET /api/cases/{case_id}/images/{name}`

One stored PNG. `name` is validated against path traversal — the suite tests this
directly.

---

## Errors

| Code | When |
| --- | --- |
| `400` | the uploaded file was empty |
| `401` | missing, invalid or expired bearer token |
| `404` | no such case |
| `413` | upload over the limit, or a volume too large to resample |
| `415` | unrecognised file format; the detail names the accepted extensions |
| `422` | invalid parameter — e.g. `rotate` not in {0, 90, 180, 270}, or an unknown `target_class` |
| `429` | account locked after five failed logins |
| `503` | the model for that route is not loaded (skipped, or failed to load) |
| `500` | unexpected; the traceback goes to the server log, not to the client |

Filesystem paths are not returned in error responses.

