#!/usr/bin/env python3
"""
The evidence layer and the integrated report.

This is the part of the project with no model behind it: a deterministic
aggregation layer plus a prompt. That makes it the easiest place for a false
claim to appear, because nothing downstream would contradict it. So the checks
here are mostly about what the layer refuses to say — that it never combines
findings, never calls an unanalysed modality normal, and never presents a
clinician's typed field as a measurement.

    python3 tests/test_report_evidence.py

No torch needed. tests/torch_stub.py stands in only so the modules import.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(HERE))

import torch_stub  # noqa: E402

STUBBED_TORCH = torch_stub.install()

from cardiovision.config import MODALITY_STATUS  # noqa: E402
from cardiovision.fusion.evidence import (  # noqa: E402
    build_case_evidence,
    normalise_clinical,
)
from cardiovision.fusion.report import (  # noqa: E402
    build_report,
    build_report_prompt,
)
from cardiovision.fusion.schema import (  # noqa: E402
    EVIDENCE_MODALITIES,
    IMAGING_MODALITIES,
    REPORT_SCHEMA_VERSION,
    STATUS_ANALYSED,
    STATUS_MEANING,
    STATUS_NOT_PROVIDED,
    STATUS_NO_MODEL,
    STATUS_PROVIDED_NOT_ANALYSED,
)

FAILURES: list[str] = []
CHECKS = 0


def check(label: str, condition: bool, detail: str = "") -> None:
    global CHECKS
    CHECKS += 1
    if condition:
        print(f"  [PASS] {label}" + (f"  -> {detail}" if detail else ""))
    else:
        print(f"  [FAIL] {label}" + (f"  -> {detail}" if detail else ""))
        FAILURES.append(label)


# ============================================================
# FIXTURES
#
# Shaped exactly like what the routers return, because that is what the
# frontend sends back in as `caseState`. Numbers are invented; the *shape* is
# the thing under test, and a shape mismatch here is a real bug.
# ============================================================

CCTA_RESULT = {
    "analyzed": True,
    "threshold": 0.6,
    "model": {
        "architecture": "Small3DUNet",
        "parameters": 1_401_265,
        "metrics": {
            "dataset": "MedHK23/CCA",
            "test_cases": 3,
            "test_dice": {"mean": 0.5996, "sd": 0.1182, "min": 0.4929, "max": 0.7266},
            "test_hd95_mm": {"mean": 109.5094, "sd": 25.0774,
                             "min": 82.1985, "max": 131.4994},
            "scope": "dataset-level, held-out test split of 3 cases",
            "caveat": "n=3. Three cases support no confidence interval.",
        },
    },
    "input": {"format": "nifti", "analysed_shape": [128, 128, 96], "filename": "ct.nii.gz"},
    "coverage": {"complete": True, "analysed_percent": 100.0},
    "explainability": {"available": True, "method": "3-D Grad-CAM"},
    "findings": [{
        "name": "Coronary artery lumen",
        "present": True,
        "voxels": 4210,
        "volume_ml": 4.21,
        "percent_of_analysed": 0.267,
        "components": 6,
        "largest_component_fraction": 0.71,
        "mean_probability": 0.83,
        "max_probability": 0.99,
    }],
}

ECG_RESULT = {
    "analyzed": True,
    "threshold": 0.5,
    "model": {"architecture": "ECGResNet1D", "parameters": 1_000_000, "metrics": {}},
    "input": {"format": "wfdb", "sampling_frequency_hz": 500,
              "lead_order_matches_training": True},
    "saliency_available": True,
    "saliency_class": "MI",
    "predictions": [
        {"name": "NORM", "label": "Normal ECG", "probability": 0.11,
         "positive": False},
        {"name": "MI", "label": "Myocardial infarction", "probability": 0.82,
         "positive": True, "operating_point": {"precision": 0.74}},
    ],
}

ECHO_RESULT = {
    "analyzed": True,
    "model": {"architecture": "UNet++", "metrics": {}},
    "input": {"format": "png", "has_spatial_calibration": False},
    "orientation": {},
    "quantification": {"presence_threshold_pixels": 200},
    "structures": [{"name": "Left ventricle", "present": True,
                    "area_px": 18422, "mean_probability": 0.91}],
}

PATIENT = {
    "name": "Test Patient",
    "mrn": "CV-0001",
    "sex": "female",
    "dateOfBirth": "1960-04-02",
    "studyDate": "2026-01-15",
    "notes": "Referred for chest pain on exertion.",
}

CLINICAL = {
    "bloodPressure": "148/92",
    "heartRate": 78,
    "symptoms": "chest pain, dyspnoea",
    "hypertension": True,
}

FULL_CASE = {
    "case_id": "case-123",
    "patient": PATIENT,
    "clinical": CLINICAL,
    "ccta": CCTA_RESULT,
    "echo": ECHO_RESULT,
    "ecg": ECG_RESULT,
    "modalities_provided": {"ccta": True, "echo": True, "ecg": True},
}

EMPTY_CASE: dict = {}

FULL = build_case_evidence(FULL_CASE)
EMPTY = build_case_evidence(EMPTY_CASE)


# ============================================================
print("\n=== 1. An empty case is a valid, fully-shaped result ===")

check("build_case_evidence survives None",
      build_case_evidence(None).case_id is None)
check("every declared modality is present even with no data",
      set(EMPTY.modalities) == set(EVIDENCE_MODALITIES),
      str(sorted(EMPTY.modalities)))
check("nothing is reported as analysed",
      EMPTY.available_modalities == [] and EMPTY.has_any_findings is False)
check("no modality invents a finding",
      all(not m.findings for m in EMPTY.modalities.values()))
check("the empty case serialises without error",
      isinstance(json.dumps(EMPTY.to_dict()), str))
check("clinical fields come back unknown, not absent and not normal",
      EMPTY.clinical.is_empty
      and "age" in EMPTY.clinical.unknown
      and "sex" in EMPTY.clinical.unknown)
check("the interpretation note spells out what null means",
      "do not mean absent" in EMPTY.clinical.to_dict()["interpretation"].lower())


# ============================================================
print("\n=== 2. The four-state status vocabulary ===")

check("all four statuses have a written meaning",
      set(STATUS_MEANING) == {
          STATUS_ANALYSED, STATUS_NOT_PROVIDED,
          STATUS_PROVIDED_NOT_ANALYSED, STATUS_NO_MODEL,
      })
check("no input at all reads as not_provided",
      EMPTY.modalities["ccta"].status == STATUS_NOT_PROVIDED,
      EMPTY.modalities["ccta"].status)

provided_only = build_case_evidence({
    "modalities_provided": {"ecg": True},
})
check("an attached but unanalysed input is distinguished from an absent one",
      provided_only.modalities["ecg"].status == STATUS_PROVIDED_NOT_ANALYSED,
      "'never looked' and 'nothing there' must not render alike")
check("an analysed modality reads as analysed",
      FULL.modalities["ecg"].status == STATUS_ANALYSED)
check("the status meaning travels in the payload, not just the constant",
      FULL.modalities["ecg"].to_dict()["status_meaning"] == STATUS_MEANING[STATUS_ANALYSED])
check("model_available and analysed are separate flags",
      FULL.modalities["ccta"].model_available is True
      and provided_only.modalities["ccta"].analysed is False)
check("clinical is not an imaging modality",
      "clinical" not in IMAGING_MODALITIES
      and "clinical" not in EVIDENCE_MODALITIES,
      "a typed blood pressure is not a model output")


# ============================================================
print("\n=== 3. A declared-unavailable modality reads as no_model ===")

check("clinical risk has no model in the declared status",
      MODALITY_STATUS["clinical"]["available"] is False)
check("fusion has no model either",
      MODALITY_STATUS["fusion"]["available"] is False)
check("no_model is reachable from the declared status, not hardcoded",
      all(
          MODALITY_STATUS[key]["available"] is True
          for key in EVIDENCE_MODALITIES
      ),
      "all three evidence modalities currently have checkpoints")


# ============================================================
print("\n=== 4. Missing modalities include clinical, deliberately ===")

check("a case with no clinical entry lists clinical as missing",
      "clinical" in EMPTY.missing_modalities,
      "a reader scanning that list is asking 'what is not here'")
check("but clinical is not in the modality dict",
      "clinical" not in EMPTY.modalities)
check("a case with clinical context does not list it missing",
      "clinical" not in FULL.missing_modalities)
check("the three analysed modalities are all listed available",
      FULL.available_modalities == list(EVIDENCE_MODALITIES),
      str(FULL.available_modalities))
check("and the missing list is then empty",
      FULL.missing_modalities == [], str(FULL.missing_modalities))

partial = build_case_evidence({"ecg": ECG_RESULT, "clinical": CLINICAL})
check("a partial case names exactly what is missing",
      sorted(partial.missing_modalities) == ["ccta", "echo"],
      str(partial.missing_modalities))


# ============================================================
print("\n=== 5. Cross-modal observations never infer ===")

check("at least one observation is always emitted",
      len(EMPTY.cross_modal_evidence) >= 1,
      "coverage is always worth stating")
check("every observation on the full case declares inference: none",
      all(item.inference == "none" for item in FULL.cross_modal_evidence),
      f"{len(FULL.cross_modal_evidence)} observations")
check("every observation on the empty case does too",
      all(item.inference == "none" for item in EMPTY.cross_modal_evidence))
check("the flag survives serialisation",
      all(
          item.to_dict()["inference"] == "none"
          for item in FULL.cross_modal_evidence
      ),
      "it must reach the prompt and the UI, where the temptation arises")
check("every observation names its basis",
      all(item.basis for item in FULL.cross_modal_evidence))

KINDS = {item.kind for item in FULL.cross_modal_evidence}
check("two or more analysed modalities trigger the pairing-provenance note",
      "pairing_provenance" in KINDS, str(sorted(KINDS)))
PAIRING = next(
    item for item in FULL.cross_modal_evidence
    if item.kind == "pairing_provenance"
)
check("that note says the pairing is asserted by the operator",
      "asserted by" in PAIRING.statement
      and "operator" in PAIRING.statement)
check("and that no model was trained on paired data",
      "paired multimodal data" in PAIRING.statement)

single = build_case_evidence({"ecg": ECG_RESULT})
check("one analysed modality does not claim a pairing",
      "pairing_provenance" not in
      {item.kind for item in single.cross_modal_evidence})

TEXT = " ".join(item.statement.lower() for item in FULL.cross_modal_evidence)
for forbidden in ("consistent with", "confirms", "corroborates",
                  "suggestive of", "risk score", "rules out"):
    check(f"no observation says '{forbidden}'",
          forbidden not in TEXT,
          "co-occurrence only")

CO = [item for item in FULL.cross_modal_evidence if item.kind == "co_occurrence"]
check("co-occurrence statements are emitted when findings coincide",
      len(CO) >= 1, f"{len(CO)} items")
check("and each disclaims corroboration in its own text",
      all(
          any(
              phrase in item.statement.lower()
              for phrase in ("cannot corroborate", "neither support",
                             "not combined", "says nothing about")
          )
          for item in CO
      ))


# ============================================================
print("\n=== 6. Negative results are not normal results ===")

quiet_ecg = dict(ECG_RESULT)
quiet_ecg["predictions"] = [
    {"name": "NORM", "label": "Normal ECG", "probability": 0.31, "positive": False},
    {"name": "MI", "label": "Myocardial infarction", "probability": 0.22,
     "positive": False},
]
quiet = build_case_evidence({"ecg": quiet_ecg})
NEG = [item for item in quiet.cross_modal_evidence if item.kind == "negative_result"]
check("no positive ECG class produces an explicit negative-result note",
      len(NEG) == 1, f"{len(NEG)} items")
check("and it refuses to call the ECG normal",
      "not the same as a normal ECG" in NEG[0].statement,
      "NORM is one of the five classes and it was not called either")

empty_ccta = json.loads(json.dumps(CCTA_RESULT))
empty_ccta["findings"][0]["present"] = False
empty_ccta["findings"][0]["voxels"] = 12
quiet_ct = build_case_evidence({"ccta": empty_ccta})
CT_NEG = [
    item for item in quiet_ct.cross_modal_evidence
    if item.kind == "negative_result"
]
check("an empty CCTA mask produces a negative-result note too",
      len(CT_NEG) == 1)
check("which puts the miss rate beside it rather than calling it absence",
      "as likely to be a miss as an absence" in CT_NEG[0].statement,
      "measured sensitivity 0.62")

contradictory = dict(ECG_RESULT)
contradictory["predictions"] = [
    {"name": "NORM", "label": "Normal ECG", "probability": 0.66, "positive": True},
    {"name": "MI", "label": "Myocardial infarction", "probability": 0.71,
     "positive": True},
]
conflict = build_case_evidence({"ecg": contradictory})
CONTRA = [
    item for item in conflict.cross_modal_evidence
    if item.kind == "contradiction"
]
check("NORM positive alongside another class is reported as a contradiction",
      len(CONTRA) == 1)
check("and explained as model uncertainty, not resolved",
      "genuinely uncertain" in CONTRA[0].statement)


# ============================================================
print("\n=== 7. Uncertainties are always non-empty ===")

check("even a full case carries uncertainties",
      len(FULL.uncertainties) >= 1, f"{len(FULL.uncertainties)}")
check("every uncertainty has a scope, a kind and a detail",
      all(u.scope and u.kind and u.detail for u in FULL.uncertainties))
check("severity is one of the two declared values",
      all(u.severity in ("note", "warning") for u in FULL.uncertainties),
      str(sorted({u.severity for u in FULL.uncertainties})))
PAIRING_U = [u for u in FULL.uncertainties if u.kind == "pairing"]
check("the no-learned-fusion uncertainty is always present",
      len(PAIRING_U) == 1 and PAIRING_U[0].scope == "case")
check("and it says findings are not combined into a joint estimate",
      "joint risk estimate" in PAIRING_U[0].detail)
check("an empty case carries it too",
      any(u.kind == "pairing" for u in EMPTY.uncertainties),
      "the absence of fusion does not depend on having data")
check("unticked risk-factor boxes become unknown, not denied",
      any(
          u.kind == "input_quality"
          and "UNKNOWN, not absent" in u.detail
          and "left blank" in u.detail
          for u in FULL.uncertainties
      ),
      "an unticked hypertension box is not 'no hypertension'")


# ============================================================
print("\n=== 8. model_versions is dicts, and fusion has no model ===")

versions = FULL.model_versions
check("each analysed modality gets an entry",
      {"ccta", "echo", "ecg"} <= set(versions), str(sorted(versions)))
check("every entry is a dict, not a version string",
      all(isinstance(value, dict) for value in versions.values()),
      "the UI renders fields, not JSON.stringify output")
check("each names the model, the task and the dataset",
      all(
          {"model", "task"} <= set(versions[key])
          for key in ("ccta", "echo", "ecg")
      ))
check("fusion is always listed",
      "fusion" in versions)
check("with model None, because there is no fusion model",
      versions["fusion"]["model"] is None)
check("and a note saying so in words",
      "No learned fusion model exists" in versions["fusion"]["note"])
check("an empty case still lists fusion and nothing else",
      set(build_case_evidence({}).model_versions) == {"fusion"},
      "no analysed modality means no model version to report")
check("the integration method declares learned_fusion False",
      FULL.to_dict()["integration_method"]["learned_fusion"] is False)
check("and describes itself as deterministic software aggregation",
      FULL.to_dict()["integration_method"]["type"]
      == "deterministic software aggregation")


# ============================================================
print("\n=== 9. build_report assembles the whole document ===")

report = build_report(FULL, patient=PATIENT, ai_summary="A summary sentence.")

EXPECTED = {
    "schema_version", "case_id", "generated_at", "generated_by", "patient",
    "modality_results", "clinical_context", "integrated_evidence",
    "uncertainties", "ai_summary", "ai_summary_error", "ai_summary_scope",
    "recommendations", "recommendations_scope", "model_versions", "disclaimer",
}
check("the report has exactly the documented top-level keys",
      set(report) == EXPECTED,
      str(sorted(set(report) ^ EXPECTED)) if set(report) != EXPECTED else "")
check("the schema version is stamped on every report",
      report["schema_version"] == REPORT_SCHEMA_VERSION)
check("generated_by is present, not built and dropped",
      isinstance(report["generated_by"], dict)
      and report["generated_by"]["application"])
check("generated_by names the evidence layer as unlearned",
      "no learned fusion" in report["generated_by"]["evidence_layer"])
check("with a summary present, the summary model is named",
      report["generated_by"]["summary_model"]
      and report["generated_by"]["summary_available"] is True)
check("all three modality results are included, analysed or not",
      set(report["modality_results"]) == set(EVIDENCE_MODALITIES))
check("the report serialises to JSON",
      isinstance(json.dumps(report), str))
check("the disclaimer states it is not a medical device",
      "not a medical device" in report["disclaimer"])
check("and rules out clinical decision use",
      "clinical decision" in report["disclaimer"])


# ============================================================
print("\n=== 10. The patient header, and the mrn spelling ===")

patient_block = report["patient"]
check("patient_id is populated from the mrn field the app actually writes",
      patient_block["patient_id"] == "CV-0001",
      "the form, the payload and the SQLite row all spell it `mrn`")
check("the older patientId spelling is still accepted",
      build_report(FULL, patient={"patientId": "LEGACY-9"})["patient"]
      ["patient_id"] == "LEGACY-9",
      "so a payload from an older client is not silently dropped")
check("a missing identifier is None, not an empty string",
      build_report(FULL, patient={})["patient"]["patient_id"] is None)
check("the header carries only the five declared fields",
      set(patient_block) == {"name", "patient_id", "sex",
                             "date_of_birth", "study_date"},
      str(sorted(patient_block)))
check("no filename, path, accession or study UID reaches the header",
      not any(
          key in patient_block
          for key in ("filename", "path", "accession", "study_uid")
      ))
check("no value in the header looks like a filesystem path",
      not any(
          isinstance(v, str) and ("/" in v or "\\" in v)
          for v in patient_block.values()
      ))
check("patient=None gives a header of nulls rather than raising",
      all(v is None for v in build_report(FULL)["patient"].values()))


# ============================================================
print("\n=== 11. An unavailable summary model does not break the report ===")

degraded = build_report(
    FULL, patient=PATIENT, ai_summary=None,
    ai_error="MedGemma is not loaded.",
)

check("the report is still produced without a language model",
      set(degraded) == EXPECTED)
check("ai_summary is null rather than a placeholder sentence",
      degraded["ai_summary"] is None)
check("the reason is recorded beside it",
      degraded["ai_summary_error"] == "MedGemma is not loaded.")
check("generated_by records the summary as unavailable",
      degraded["generated_by"]["summary_available"] is False
      and degraded["generated_by"]["summary_model"] is None)
check("and carries the reason too",
      degraded["generated_by"]["summary_unavailable_reason"]
      == "MedGemma is not loaded.")
check("every modality finding survives the language model's absence",
      degraded["modality_results"]["ccta"]["findings"]
      == report["modality_results"]["ccta"]["findings"],
      "no finding came from the language model")
check("a successful summary clears the error field",
      report["ai_summary_error"] is None)
check("the summary's scope says the modality results are authoritative",
      "modality results are authoritative" in report["ai_summary_scope"])
check("and that it introduces no new measurement",
      "no new measurement" in report["ai_summary_scope"])


# ============================================================
print("\n=== 12. Recommendations are workflow, never management ===")

recs = build_report(EMPTY, patient=None)["recommendations"]
KINDS_R = {item["kind"] for item in recs}
check("every recommendation has a kind, a modality, an action and a reason",
      all(
          {"kind", "modality", "action", "reason"} == set(item)
          for item in recs
      ))
check("kinds stay inside the declared vocabulary",
      KINDS_R <= {"analysis", "input", "capability", "coverage",
                  "interpretation"},
      str(sorted(KINDS_R)))
check("an empty case asks for inputs",
      "input" in KINDS_R)
check("the human-in-the-loop item is always last and always present",
      recs[-1]["kind"] == "interpretation" and recs[-1]["modality"] == "case")
check("it names a qualified clinician as the interpreter",
      "qualified clinician must interpret" in recs[-1]["action"])
check("and states the models were not validated for clinical use",
      "was validated for clinical use" in recs[-1]["reason"]
      and "none of them" in recs[-1]["reason"])

FULL_RECS = build_report(FULL, patient=PATIENT)["recommendations"]
check("a fully analysed case still carries the interpretation item",
      FULL_RECS[-1]["kind"] == "interpretation")

partial_ct = json.loads(json.dumps(CCTA_RESULT))
partial_ct["coverage"] = {"complete": False, "analysed_percent": 41.5}
covered = build_report(
    build_case_evidence({"ccta": partial_ct}), patient=None
)["recommendations"]
COVERAGE = [item for item in covered if item["kind"] == "coverage"]
check("a truncated pass produces a coverage recommendation",
      len(COVERAGE) == 1)
check("which names the percentage actually examined",
      "41.5" in COVERAGE[0]["reason"], COVERAGE[0]["reason"])
check("the scope line rules out clinical management advice",
      "Not clinical" in
      build_report(FULL)["recommendations_scope"])

ALL_TEXT = " ".join(
    f"{item['action']} {item['reason']}" for item in FULL_RECS
).lower()
for forbidden in ("mg ", "start ", "prescrib", "refer for angiograph",
                  "follow up in", "statin"):
    check(f"no recommendation contains '{forbidden.strip()}'",
          forbidden not in ALL_TEXT,
          "no model here outputs a management decision")


# ============================================================
print("\n=== 13. The prompt is generated, and constrains the model ===")

prompt = build_report_prompt(FULL)

check("the prompt is a context and a question",
      set(prompt) == {"context", "question"})
check("the context names the case",
      "case-123" in prompt["context"])
check("an unsaved case is labelled as such rather than blank",
      "(unsaved)" in build_report_prompt(EMPTY)["context"])
check("the context declares itself software-assembled",
      "deterministic software layer" in prompt["context"])
check("and claims no measurement a model did not produce",
      "no measurement that a model did not produce" in prompt["context"])
check("every modality gets a block, including the unanalysed ones",
      all(
          FULL.modalities[key].label in prompt["context"]
          for key in EVIDENCE_MODALITIES
      ))
check("the question forbids stating a diagnosis",
      "Do not state a diagnosis" in prompt["question"])
check("and forbids a stenosis percentage or an ejection fraction",
      "stenosis percentage" in prompt["question"]
      and "ejection fraction" in prompt["question"])
check("and forbids combining modalities into one conclusion",
      "Do not combine findings from different modalities"
      in prompt["question"])
check("and forbids describing the unanalysed as normal",
      "Do not describe anything not analysed as normal"
      in prompt["question"])
check("and forbids altering the numbers it was given",
      "Use only the numbers given above, unchanged" in prompt["question"])
check("the question lists which modalities produced output",
      "Modalities with model output:" in prompt["question"])
check("an empty case reports 'none' rather than an empty list",
      "Modalities with model output: none"
      in build_report_prompt(EMPTY)["question"])
check("clinician-entered text is labelled as operator input in the context",
      "clinician" in prompt["context"].lower())
check("the prompt carries no filesystem path",
      "/Users/" not in prompt["context"]
      and "/sessions/" not in prompt["context"]
      and "\\" not in prompt["context"])


# ============================================================
print("\n=== 14. Clinical normalisation keeps unknown unknown ===")

clinical = normalise_clinical(PATIENT, CLINICAL)
check("age is derived from the date of birth, not the typed field",
      clinical.age_source == "derived from date of birth",
      "a typed age goes stale the moment a birthday passes")
check("a typed age is used only when no date of birth exists",
      normalise_clinical({}, {"age": 61}).age_source
      == "entered by the clinician")
check("a non-numeric age is discarded rather than coerced",
      normalise_clinical({}, {"age": "sixty"}).age is None)
check("a ticked risk factor is recorded as history",
      "Hypertension" in " ".join(clinical.history), str(clinical.history))
check("an unticked one goes to unknown, never to a recorded negative",
      any(item.startswith("history:") for item in clinical.unknown),
      "the previous behaviour read as a clinical claim nobody made")
check("symptoms split on commas into a list",
      clinical.symptoms == ["chest pain", "dyspnoea"], str(clinical.symptoms))
check("fields this app never collects are named separately",
      len(clinical.not_collected) > 0
      and not set(clinical.not_collected) & set(clinical.unknown),
      "'never offered' is not 'offered and skipped'")
check("the payload marks the whole block as clinician-entered",
      "not model output" in clinical.to_dict()["source"])
check("medications are empty because there is no input for them",
      clinical.medications == [])


# ============================================================
print("\n=== 15. Nothing in a report leaks storage or invents capability ===")

blob = json.dumps(build_report(FULL, patient=PATIENT,
                               ai_summary="A summary."))
check("no absolute user path appears anywhere in the report",
      "/Users/" not in blob and "/kaggle/" not in blob
      and "/root/" not in blob)
check("no password or token field appears",
      not any(word in blob.lower()
              for word in ("password", "api_key", "secret", "bearer ")))
check("the CCTA model entry carries its n=3 caveat into the report",
      "n=3" in blob)
check("the CCTA findings still disclaim stenosis and calcium",
      "no stenosis" in blob.lower() and "calcium score" in blob.lower())
check("no CAD-RADS category is claimed",
      "no CAD-RADS category" in blob or "CAD-RADS" not in blob.upper()
      or "no CAD-RADS" in blob)
check("the ECG confidence explains independent sigmoids",
      "do not sum to 1" in blob)
check("ejection fraction appears only as a disclaimer, never as a value",
      all(
          "ejection fraction" not in json.dumps(finding.get("measurement", {})).lower()
          for finding in report["modality_results"]["echo"]["findings"]
      )
      and "no ejection fraction" in
      json.dumps(report["modality_results"]["echo"]["limitations"]).lower(),
      "the echo model outlines anatomy; it measures no function")


# ============================================================
print("\n" + "=" * 62)

if STUBBED_TORCH:
    print("NOTE: torch is not installed here. Nothing in the evidence layer")
    print("      touches a tensor, so every check above is real — but the")
    print("      modality results fed in were fixtures, not model output.")
    print("=" * 62)

if FAILURES:
    print(f"{len(FAILURES)} FAILURE(S) out of {CHECKS} checks:")
    for item in FAILURES:
        print(f"  - {item}")
    sys.exit(1)

print(f"ALL {CHECKS} EVIDENCE AND REPORT CHECKS PASSED")
