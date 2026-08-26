# Decisions

Architectural decisions that are **evident from the repository** — each one is
traceable to a file. Where a decision cannot be read off the code, this file says
so in those words rather than guessing.

---

## Packaging and layout

**One installable package under `src/`.** `pyproject.toml` sets
`package-dir = { "" = "src" }` with `packages.find where = ["src"]`, and
`[project.scripts]` exposes `cardiovision = "cardiovision.cli:main"`. The former
`backend/` directory no longer exists. Consequence: the application is imported
as `cardiovision.*` and CI verifies the *installed* package imports, so a module
that only works from the repository root is a bug.

**Dependency lower bounds, not pins.** `pyproject.toml` says why: pinning torch
in an application meant to run on whatever accelerator the operator has "is a
portability bug, not a reproducibility feature", and is how an Apple Silicon
install ends up on a CPU wheel. Anyone wanting byte-identical environments is
directed to generate a lock file from a working install.

**MIT licence.** Declared in both `LICENSE` and `pyproject.toml`.

---

## Model loading

**Module-level singletons with lazy, lock-guarded loading.** Every inference
module ends with `x_segmenter = XSegmenter()`. `load()` is idempotent and guarded
by a `threading.Lock`, and the object exposes `is_loaded` / `load_error` rather
than raising at import. Rationale visible in the code: weights are hundreds of
megabytes, importing must stay cheap, and two concurrent requests must not both
start a load.

**Models load independently and failure is non-fatal.**
`api/app.py::_load_model` catches the modality's own unavailability exception and
prints a warning. The docstring states the reason: "the echo segmenter is no use
for an ECG and vice versa, so one missing checkpoint should cost exactly one
modality." An unexpected exception still gets a traceback, because that is a bug
rather than a missing file.

**Load state is reported, not inferred.** `/api/health` overwrites the declared
`available` flag for the three trained modalities with the model's real
`is_loaded`. Its docstring gives the reason: otherwise "a checkpoint that failed
to load shows as a capability the UI will offer and the backend will then
refuse."

**`--skip` exists to avoid loading MedGemma.** `cli.py` maps friendly names onto
`CARDIOVISION_SKIP_*` env vars, because skipping 8.6 GB while iterating on the
imaging pipeline is worth one flag.

---

## Honesty as an architectural constraint

These are design decisions, not stylistic ones, and reversing any of them changes
what the software claims.

**Absence is reported as absence.** The report prompt and the evidence layer name
unavailable modalities and un-analysed inputs explicitly, so the language model
cannot fill the gap. Partial CCTA coverage is stated as a percentage with the
sentence that the remainder "must not be described as normal."

**Missing attribution renders nothing.** When saliency cannot be computed the API
omits it rather than returning zeros. A blank heat map is a claim — "the model
looked nowhere" — and a different one from "we could not measure where it
looked."

**Presence thresholds instead of raw argmax.** `ECHO_PRESENCE_THRESHOLD_PX` (50)
and `CCTA_PRESENCE_THRESHOLD_VOXELS` (500) exist because a handful of scattered
argmax pixels is noise, and reporting it as a detected structure is a false
positive with a confident face on it.

**Operating-point context travels with every positive call.** A positive ECG
class is accompanied by the measured precision at that threshold, expressed as a
false-call rate. The weakest class stays visible in the per-class table rather
than being hidden behind a macro average.

**The scope disclaimer is generated, not decorative.** `fusion/report.py` emits
"CardioVision AI is a research prototype. It is not a medical device…" as part of
the report structure.

---

## The fusion layer

**Deterministic aggregation, not a learned model.** `src/cardiovision/fusion/`
combines findings that the three models already produced. It contains no
weights, consults no model, and `models/fusion/` is an empty directory.
`health.py` states the reasoning inline: reporting fusion as a loadable model
"would be the exact overstatement this endpoint exists to prevent."

**`/api/evidence` has no model dependency.** `api/app.py` says it "answers
correctly on a server where nothing loaded", because it aggregates results that
were already computed. This is why the evidence layer is separate from inference
rather than a method on the segmenters.

**Cross-cohort pairing is declared, never implied.** Because no patient exists in
more than one of the three datasets, any multimodal case was assembled by an
operator. The evidence layer states this rather than presenting the modalities as
a joint study.

---

## API surface

**Three public routes, everything else gated.** `/`, `/api/health` and
`/api/auth/login` are public; the health docstring explains why — the login
screen must report backend and model state *before* anyone signs in. The storage
block in `/api/health` is deliberately a count, not a case list, because a case
list is patient data.

**`create_app()` is a factory, not a module-level-only singleton.** Its docstring
gives the reason: models are process-wide singletons but routing, middleware and
dependency overrides are not, "and sharing them is how one test's override leaks
into the next." The module-level `app` exists for `uvicorn
cardiovision.api.app:app`.

**Route list in the module docstring is maintained by hand; the banner's is
generated.** `root()` notes that generating it means it "cannot go stale."

**CORS defaults to the two Vite dev origins**, overridable through
`CARDIOVISION_CORS_ORIGINS`. The comment states the principle: anything else "is
a deployment change, not a code change."

**Server-side rendering to PNG.** `rendering/` produces figures on the server
rather than shipping raw arrays to the browser. Colours are duplicated in
`config.py` and the frontend and documented as kept in sync, so overlays agree
between server PNGs and the client canvas.

---

## Configuration

**One config module, environment-overridable, no absolute paths.**
`src/cardiovision/config.py` derives `PROJECT_ROOT` and every artefact path from
it, and each checkpoint path has an env override. No `/Users/...` or `/kaggle/...`
path is read at runtime. The one file that still contains absolute training paths
— `models/ccta/cardiovision_cca_index.csv` — is provenance, not configuration.

**Metrics live in config, sourced from the artefacts.** Test metrics are recorded
in `config.py` so the API can report them without loading a checkpoint; echo
validation figures are read from the checkpoint itself.

---

## Authentication

**One shared account, opaque bearer tokens, in-memory sessions.**
`services/auth.py` uses `secrets.token_urlsafe(32)` for tokens, a per-process
random salt with `hmac.compare_digest` for credential comparison, an
8-hour TTL (`AUTH_SESSION_TTL_SECONDS`), and a login throttle. Credentials come
from `AUTH_DEFAULT_USERNAME` / `AUTH_DEFAULT_PASSWORD`, overridable by
`CARDIOVISION_USER` / `CARDIOVISION_PASSWORD`.

**This is explicitly an access gate, not security.** The codebase says so in two
places. `cli.py` defaults `--host` to `127.0.0.1` and prints a notice when bound
elsewhere, because "there is one shared password and no transport encryption, so
binding to 0.0.0.0 puts patient data on the network behind an access gate that
was never meant to be one." The lifespan prints the same warning when the default
password is in use. Sessions are in-memory, so a restart logs everyone out — an
accepted trade for a single-operator local tool.

---

## Storage

**SQLite, one local file, never committed.** `data/.gitignore` ignores the
database and its WAL/SHM sidecars and warns that it is unencrypted and must not
be synced to cloud storage.

**Additive-only migrations.** `services/database.py` adds columns and never drops
or rewrites them, with independently guarded `ON CONFLICT` groups so one failed
group does not abort the rest. Reports are saved through a separate
`save_report()` path. Stored figure names are namespaced through
`_FIGURE_ALIASES`.

**Storage failure is non-fatal.** The lifespan reports it and continues, because
analysis works without persistence.

---

## Weights in Git

**Checkpoints are tracked through Git LFS; MedGemma is ignored.**
`.gitattributes` explains the choice: a clone without the weights "installs
cleanly and then cannot load a model", so pointers plus `git lfs pull` is better
than absence. MedGemma is excluded deliberately at 8.6 GB — it is a third-party
model, re-downloadable from its source.

**Notebooks are marked `linguist-documentation`**, and `frontend/dist/**`
`linguist-generated`, because "the notebooks are the training record, not the
application."

---

## Tests and CI

**Suites are executable scripts, with pytest as a thin wrapper.**
`pyproject.toml` sets `python_files = ["test_all.py"]` and explains why: the
suites run their checks at module level so they work "on a machine where torch
was never installed", and letting pytest collect them directly turns a deliberate
`sys.exit(0)` for an unresolved LFS pointer into a collection error rather than
the skip it is.

**`tests/torch_stub.py` exists so the suite runs without torch.** CI installs
only `numpy`, `ruff` and `pytest` — deliberately not torch — which is why the
suites must degrade to stubs and say so in their output.

**CI checks out with `lfs: false`.** The suites are expected to detect an
unresolved pointer and skip rather than fail, which also means the skip path is
exercised on every run.

**Three Python versions (3.10, 3.11, 3.12) plus a separate frontend job** running
`npm ci`, `npm run lint` (oxlint) and `npm run build`.

---

## Frontend

**Vite + React, plain CSS, no state-management library, oxlint.** Visible from
`frontend/package.json`, `vite.config.js`, `.oxlintrc.json` and the absence of
any store. State is local to `App.jsx` and passed down.

**Four navigation sections** — patient case, analysis, explainability, case
assistant. There is deliberately no section for a capability that has no model.

---

## Not established in repository

- **Intended hosting or deployment target beyond a local machine.** Not
  established in repository.
- **Release, versioning or changelog policy** (the version is `4.0.0` in three
  places; how it is incremented is) not established in repository.
- **Model retraining cadence or trigger.** Not established in repository.
- **Multi-user, role or audit-log design.** Not established in repository.
- **Regulatory or clinical-validation pathway.** Not established in repository.
- **Database encryption or backup strategy.** Not established in repository.
- **Choice of `4.0.0` as the version number** — no `1.x`–`3.x` history exists in
  this repository. Not established in repository.
- **Whether `data/sample/` is meant to hold committed sample inputs.** The
  directory is empty and untracked. Not established in repository.
