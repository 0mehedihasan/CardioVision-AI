# Rules — medical AI honesty

This repository handles cardiac imaging and produces text a clinician might read.
Every rule below is already implemented in the code and asserted by a test. Do not
relax one to make an output look cleaner or a UI look more confident.

## 1. Research is not clinical validation

Say "held-out test split", never "clinically validated". None of the three models
has regulatory clearance, prospective evaluation, or a reader study. The
distinction to keep:

| Established | Not established |
| --- | --- |
| Overlap on a specific public held-out split | performance on any other data |
| Metrics at one stated operating point | behaviour at a different threshold |
| Behaviour on the acquisition settings in the training set | behaviour on another scanner, protocol or slice thickness |

CCTA is the sharp case: **three test cases** (9, 14, 15). Three observations
support no confidence interval and no claim of generalisation. Wherever the CCTA
Dice appears, `n=3` appears with it.

## 2. Never claim diagnostic capability

These models classify and segment. They do not diagnose.

| Forbidden | Why |
| --- | --- |
| "diagnoses", "rules out", "confirms" | none of the three has the validation to support it |
| "consistent with", "suggestive of", "corroborates" | these are clinical inferences; the software does not make them |
| any "risk score" | no risk model exists |
| CAD-RADS, stenosis grade, calcium score | the CCTA model has one foreground class; it does none of these |
| ejection fraction, strain, chamber volumes over a cycle | the echo model outlines anatomy on a single frame; it measures no function |
| a stated confidence for an individual case | every published metric is dataset-level |

`tests/test_report_evidence.py` asserts that `consistent with`, `confirms`,
`corroborates`, `suggestive of`, `rules out` and `risk score` never appear in a
cross-modal observation. Keep that list intact and extend it rather than pruning
it.

## 3. Preserve uncertainty — do not average weakness away

Weakness must travel with the number, in the API response, in the UI and in the
language-model prompt.

- **`HYP` on ECG**: precision 0.361, AP 0.478 at the 0.5 threshold. Roughly two in
  three positive calls are wrong. This is `ECG_WEAK_CLASSES` and must never be
  reported as a finding on its own.
- **CCTA as a whole**: Dice 0.60 on three cases, HD95 82–131 mm. This is
  `CCTA_WEAK_NOTES`, and the mask is a contrast-density highlight to review, not a
  verified coronary tree.
- **CCTA metrics are always `{mean, sd, min, max}`**, never a bare float. A single
  number over three cases reads as a stable estimate.
- **Return every probability**, not only the calls above threshold, and state the
  operating point the published precision and recall belong to.
- **Validation metrics** are read out of the checkpoint at load time, never
  hardcoded, and are displayed apart from test metrics because the validation split
  steered early stopping.

## 4. Absence is not normality

This is the rule most easily broken by accident.

| Situation | Correct reporting |
| --- | --- |
| A structure below the presence threshold (50 px echo, 500 voxels CCTA) | "not identified", with the threshold stated — never "absent" |
| A modality not uploaded | status `not_provided` |
| A file uploaded but no result stored | status `provided_not_analysed` |
| No trained model for the modality | status `no_model` |
| A region outside the analysed crop | say the coverage was partial; do not present a subvolume as the whole study |
| An unticked risk-factor checkbox | **unknown**, not denied |

The checkbox case has bitten this project before. The form ships those fields
defaulting to `false`, so `false` means the clinician never touched the box. An
earlier version emitted them to the language model as "not reported", which handed
it a negative history nobody had taken and which it repeated back as fact. The
prompt now states explicitly that these are UNKNOWN, not absent, and that the
patient must not be described as having denied them.

## 5. Do not fabricate medical evidence

- No synthetic findings, no placeholder measurements, no example output presented
  as a patient result.
- The shipped artefacts under `models/` are **provenance, not patients**:
  `models/ccta/case_{9,14,15}_xai.png` and the matching Grad-CAM volumes are
  dataset test cases; `models/ecg/lead_importance.csv` is one record (`HR00025`),
  not an average over the test set; `models/echo/xai_*.npy` are notebook outputs
  used as renderer fixtures. None may be rendered as a case result.
- Do not fabricate multimodal pairing. If a case has an echo and an ECG, they are
  two studies filed together by an operator — the software does not assert they
  describe the same physiological state, and cross-modal observations carry
  `inference: "none"` for exactly this reason.
- Saliency is hidden entirely when the gradient is unavailable, because an
  all-zero gradient still renders as a smooth, convincing picture. Missing keys,
  not empty ones.
- Input-gradient attribution is **not** Grad-CAM. Echo uses the former, CCTA the
  latter, and they are labelled separately.

## 6. Human in the loop, always

Every output is material for a qualified clinician to read, and the code says so:

- a `disclaimer` field on every report
- `ai_summary_scope` and `recommendations_scope` bounding what the narrative was
  allowed to cover
- the final recommendation, always last, stating that none of the models was
  validated for clinical use
- `?include_prompt=true` on `/api/report`, so a reader can check the narrative
  claims nothing the evidence did not
- `ai_summary_error` populated when generation fails, with the structured report
  standing on its own

Do not add an auto-accept path, a "confident enough" shortcut, or a UI state that
presents a model output as a signed finding.

## 7. Identifiers stay out of prompts

Patient name and medical record number are withheld from every language-model
prompt. Age (derived from date of birth, never stored), sex, study date, notes and
the clinical form are sent, because a clinical answer can use them and an
identifier cannot.
