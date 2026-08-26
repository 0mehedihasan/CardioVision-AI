# Rules — data and privacy

## 1. Never commit private medical data

Nothing patient-identifiable enters this repository. Not in a test fixture, not in
a docstring, not in an example payload, not in a screenshot, not in a commit
message.

| Path | State |
| --- | --- |
| `data/cardiovision.db` (+ `-wal`, `-shm`) | patient records — ignored by `data/.gitignore` |
| `data/cases/` | `CASE_FILES_DIR` — where the store writes each case's rendered PNGs and original upload. Ignored by `data/.gitignore` (`cases/`) |
| `models/medgemma-1.5-4b-it/` | ~8.6 GB of vendor weights — ignored by the root `.gitignore`, downloaded separately |

`CaseStore._write_gitignore()` writes `data/.gitignore` at startup, so a
fresh clone is protected before anyone thinks about it. It writes the file **only if
it does not already exist**, which means an older `data/.gitignore` from a previous
version is never upgraded in place — one such checkout was missing the `cases/`
line, leaving rendered patient images untracked-but-committable. If that function's
content changes, check existing checkouts by hand.

### `samples/` is the place for test inputs, `data/` is not

Two sample cases per modality are tracked under `samples/` — 38 files, 181 MB, from
CAMUS, MedHK23/CCA and PTB-XL. That is deliberate, and `samples/README.md` documents
it. They were originally committed inside `data/cases/`, i.e. inside `CASE_FILES_DIR`
itself, and were moved out with `git mv`.

| Do | Don't |
|---|---|
| Put a new sample input in `samples/<modality>/` | Put one anywhere under `data/` — it is runtime state and ignored |
| Check the dataset's licence before adding | Add a fourth CCTA volume; the largest is already 97 MB against GitHub's 100 MB limit |
| Keep it to a couple of cases per modality | Treat `samples/` as an evaluation set — see `.claude/memory/data.md` §4 for which split each came from |

> [!WARNING]
> **An ignore rule does nothing to an already-tracked path**, and `git check-ignore`
> stays silent about one, so `git status` looks clean while committed files sit in an
> "ignored" directory. That is how the samples ended up inside `CASE_FILES_DIR`. Use
> the index, not the status, to check:

```bash
git ls-files data/                            # the real test: only .gitignore should appear
git check-ignore -v --no-index data/cases     # confirms the rule itself exists
git status --short data/
```

Do not remove that startup behaviour, and do not remove `!data/.gitignore` from the
root `.gitignore` — it is what keeps the nested rules tracked.

Before any commit that touches fixtures or documentation, check that no real name,
MRN, date of birth, accession number, institution name or DICOM header remnant went
with it. Test fixtures use obviously synthetic values (`CV-0001`, generic names).

**The database is not encrypted.** Do not put it in a synced folder, do not attach
it to an issue, do not copy it into `models/` or anywhere tracked.

## 2. Respect dataset licences

No training or validation split is redistributed here. Two sample cases per
modality are (`samples/`), which means each dataset's terms apply to **this
repository**, not only to whoever downloads the dataset. Keep `LICENSE` consistent
with that. Three datasets are used and each has its own terms:

| Dataset | Modality | Terms |
| --- | --- | --- |
| CAMUS | echo | research use, University of Lyon (CREATIS). Cite the dataset paper if you publish results from the echo checkpoint |
| MedHK23/CCA | CCTA | check the source's terms before redistributing anything derived from it |
| PTB-XL | ECG | PhysioNet, Open Data Commons Attribution Licence v1.0 — **attribution required** |
| MedGemma weights | — | Google Health AI Developer Foundations terms; not distributed here, accept them at download |

The MIT licence on this repository covers its own code and does not relicense any
of the above. `LICENSE` records this; keep it accurate if a dataset is added.

Do not add a script that downloads a dataset to a hardcoded path, and do not
commit a cached index file that embeds someone's local cache directory.

## 3. Preserve patient-level splitting

All three splits are at patient or case level, and every one was asserted in its
notebook. This is the single most destructible property in the project: a
frame-level or record-level reshuffle would inflate every metric and the number
would still look plausible.

| Modality | Split | Unit |
| --- | --- | --- |
| Echo | 350 / 75 / 75 patients → 1400 / 300 / 300 pairs | **patient** |
| CCTA | 14 / 3 / 3 | **case** |
| ECG | 13 031 / 2 793 / 2 793 patients → 14 957 / 3 199 / 3 232 records | **patient** |

If anyone ever retrains — which this repository does not do — the split must stay at
the same unit, and disjointness must be asserted, not assumed.

## 4. Prevent data leakage

- Never evaluate on data the model trained on, and never quote a training-set
  number as a performance figure.
- Validation metrics are **not** independent: the validation split steered early
  stopping and checkpoint selection. They are displayed separately from test metrics
  for that reason. Do not merge the two.
- Do not average a validation figure and a test figure into one number.
- Do not tune a threshold on the test split. The shipped thresholds (0.5 for ECG,
  0.60 for CCTA) are the operating points the published precision and recall belong
  to; moving one invalidates those numbers, which is why the API returns every
  probability rather than only the calls.
- A shipped example artefact is not evaluation data. `models/ecg/lead_importance.csv`
  is one record; `models/ccta/case_*_xai.png` are three dataset test cases.

## 5. Keep test data separate and synthetic

- The verification suites build their own inputs: synthetic arrays, a temporary
  SQLite file, and fixtures shaped like router responses. No suite reads a patient
  record, and `test_case_lifecycle.py` writes to a temporary database so it never
  touches real records.
- Renderer suites use the notebook artefacts under `models/` as fixtures. Those are
  dataset outputs, not patient data — but they are still not results, and a suite
  must not present them as one.
- Do not add a fixture by copying a real study. Generate one.
- Do not point a test at `data/` in a developer's checkout.

## 6. Identifiers and prompts

Patient name and MRN are withheld from every language-model prompt. Age is
**derived** from the date of birth on every read rather than stored, because an age
typed in once is wrong a year later.

`GET /api/cases/{id}` returns image **endpoints**, not inline base64, and the
frontend fetches them with the bearer token in a header — so the token never
appears in a URL and cannot end up in uvicorn's access log. Preserve that; a
query-string token is a credential written to disk in plaintext.

## 7. Secrets

- No secrets, API keys, tokens or passwords in tracked files — including in
  `.claude/`, which is shared project context.
- `.env` is gitignored; `.env.example` is the tracked, value-free template.
- The default operator credentials (`medexpert` / `1111`) are documented, not
  secret. They exist so the app runs out of the box and must be overridden with
  `CARDIOVISION_USER` and `CARDIOVISION_PASSWORD` before this sits anywhere but a
  personal machine.
- Do not return filesystem paths in API error responses.
- If local or private Claude state is ever needed, put it in `.claude/local/`,
  which is gitignored. The tracked `.claude/` directory holds reusable project
  context only.
