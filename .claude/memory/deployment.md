# Deployment — CardioVision AI

> Verified against the repository. Code and configuration are the source of
> truth; if this file disagrees with `src/cardiovision/config.py`,
> `src/cardiovision/cli.py` or `.github/workflows/ci.yml`, they win and this file
> is stale.

## What deployment means here

**Full local, single workstation.** Two processes on one machine: a uvicorn
backend and a Vite dev server (or a static build of it). Nothing else.

There is no cloud infrastructure in this repository. Do not invent any, and do
not document any that does not exist.

| Thing | State |
| --- | --- |
| `Dockerfile` | **does not exist** |
| `docker-compose.yml` | **does not exist** |
| Kubernetes / Helm manifests | **do not exist** |
| Terraform / CloudFormation / CDK | **do not exist** |
| `Makefile` | **does not exist** |
| Cloud model hosting, inference endpoints, GPU cluster config | **do not exist** |
| Reverse proxy / TLS / nginx config | **does not exist** |
| Database server | **none** — SQLite file on disk |
| Message queue, cache, object store | **none** |
| Monitoring, tracing, metrics exporter | **none** |
| GitHub Actions | **one workflow**, `.github/workflows/ci.yml` — CI only, no deploy job |

If a future task needs any of the above, it has to be built and then documented.
Writing it here first would make this file a wish list rather than a description.

## Processes

### Backend

```bash
pip install -e .
cardiovision serve --port 8000
```

`cardiovision serve` (`src/cardiovision/cli.py`) is a thin wrapper over
`uvicorn.run("cardiovision.api.app:app", ...)`. Flags: `--host` (default
`127.0.0.1`), `--port` (8000), `--reload`, `--skip MODEL` (repeatable),
`--log-level`.

`cardiovision check` reports which checkpoints are present and which device would
be selected, then exits. It exists to separate "the checkpoint is missing" from
"the server is broken", which otherwise both present as a 503 in the UI. Run it
first after a fresh clone.

Binding to anything other than `127.0.0.1` / `localhost` / `::1` prints a notice
to stderr about the shared password and absent TLS. The CLI still allows it — it
warns rather than refuses — so the notice is the only guard.

### Frontend

```bash
cd frontend
npm ci
npm run dev      # Vite dev server on :5173
npm run build    # static build into frontend/dist/
```

`VITE_API_BASE_URL` is compiled into the bundle at build time (default
`http://127.0.0.1:8000`), so it belongs in `frontend/.env.local`, not in the
backend environment. There is no server in this repository that serves
`frontend/dist/` — building it produces static files and nothing is configured to
host them.

## Model loading at startup

Models load in the FastAPI lifespan, each independently. A failure costs exactly
one modality; a MedGemma failure does not take segmentation down. A skipped or
failed model is reported unavailable by `/api/health`, and the UI panel says so.

| Variable | Effect |
| --- | --- |
| `CARDIOVISION_SKIP_ECHO` | keep the echo model out of memory |
| `CARDIOVISION_SKIP_CCTA` | keep the CCTA model out of memory |
| `CARDIOVISION_SKIP_ECG` | keep the ECG model out of memory |
| `CARDIOVISION_SKIP_MEDGEMMA` | keep MedGemma out of memory (~8.6 GB, most of the cold start) |

`cardiovision serve --skip medgemma` sets these. Any truthy value works
(`1`, `true`, `yes`, `on`).

Device selection is **MPS → CUDA → CPU**, decided in `config.select_device()`,
which imports torch inside the function so a torch-free machine can still read
configuration. If a saliency backward pass is unsupported on MPS, that request's
whole forward pass falls back to CPU and the response reports
`fell back from mps` rather than misstating the device.

## Configuration surface

Every value has a working default; nothing is required. `.env.example` is the
annotated template. **There is no dotenv loader in the backend, on purpose** — one
fewer dependency, and one fewer place a password can be picked up silently. Export
the variables yourself or `set -a && . ./.env && set +a`.

| Variable | Default | Purpose |
| --- | --- | --- |
| `CARDIOVISION_USER` | `medexpert` | operator username |
| `CARDIOVISION_PASSWORD` | `1111` | operator password |
| `CARDIOVISION_HOME` | auto-detected | where `models/` and `data/` live |
| `CARDIOVISION_CORS_ORIGINS` | `http://localhost:5173,http://127.0.0.1:5173` | allowed origins; wildcard not accepted |
| `CARDIOVISION_SKIP_*` | unset | see above |
| `VITE_API_BASE_URL` | `http://127.0.0.1:8000` | frontend build-time only |

`PROJECT_ROOT` walks up from the installed package looking for `pyproject.toml` or
`models/`, which is correct for a checkout and for `pip install -e .`. Set
`CARDIOVISION_HOME` when installed non-editable or when weights live on another
volume. A wrong root surfaces as "checkpoint not found".

## Persistent state on disk

| Path | Contents | Tracked? |
| --- | --- | --- |
| `data/cardiovision.db` | SQLite: patient records, findings, transcripts | no — `data/*` is gitignored |
| `data/cases/<case-id>/` | rendered PNGs and the original uploads | no |
| `models/*/…pth`, `…pt` | the three checkpoints | yes, via **Git LFS** |
| `models/medgemma-1.5-4b-it/` | ~8.6 GB of safetensors | **no** — gitignored; download separately |

The backend writes a second `.gitignore` inside `data/` at startup, so a fresh
clone is protected before anyone thinks about it.

**The database is not encrypted.** Do not put it in a synced folder. There is no
backup mechanism in this repository; copying `data/` is the whole story.

## Deployment security posture — stated, not hidden

These are documented limitations of a single-workstation research tool, not
vulnerabilities to be filed as such:

- one shared operator account, one shared password
- no TLS anywhere; plain HTTP on localhost
- tokens in memory only, so a restart signs everyone out
- unencrypted SQLite database holding patient details
- no audit log, no rate limiting beyond the five-attempt login lockout
- no multi-user separation, no roles, no per-record access control

Consequence: **do not bind to `0.0.0.0`** and do not put this on a shared network.
Anyone who can reach the port can reach the case database behind one password.

## CI

`.github/workflows/ci.yml`, on push and PR to `main` plus `workflow_dispatch`,
with `concurrency` cancelling superseded runs and `permissions: contents: read`.

- **python job**, matrix 3.10 / 3.11 / 3.12: installs only `numpy`, `ruff`,
  `pytest`; runs `ruff check src tests`; runs each suite as a script; runs
  `pytest -q`; then `pip install --no-deps -e .` and imports `cardiovision` and
  `cardiovision.config` to catch a module that resolves in a checkout but not in a
  wheel.
- **frontend job**: `npm ci`, `npm run lint` (oxlint), `npm run build` (vite) —
  the only place the JSX is actually parsed and compiled.

Checkout uses `lfs: false`. The architecture suite detects an unresolved LFS
pointer and skips with a message, so no run burns bandwidth on a 46 MB
checkpoint. Consequently a green tick does **not** mean a forward pass ran.

There is **no deployment job**, no release automation, no container publish, and
no environment secrets configured in the workflow.
