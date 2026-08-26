"""
CardioVision AI — structured report assembly and the MedGemma prompt.

Two jobs, deliberately in one file because they must not drift apart:

``build_report_prompt`` turns a ``CaseEvidence`` object into the text MedGemma
sees. Nothing user-typed is interpolated except the free-text clinical fields,
and those are labelled as operator-entered. Every number in the prompt came out
of a model.

``build_report`` assembles the report that the API returns and the UI renders.
The report is complete and valid *without* MedGemma: the structured sections are
built from the evidence, and the language model contributes one field,
``ai_summary``. If MedGemma is unavailable the report still renders with every
model finding intact. That ordering is the point — the summary is the part of
the report that can be wrong in ways nobody notices, so it is the part that is
optional.

Recommendations here are workflow steps, never clinical management advice. No
model in this project outputs a management decision, so the report does not
contain one.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Optional

from cardiovision.config import APP_NAME, APP_VERSION, MEDGEMMA_NAME
from cardiovision.fusion.schema import (
    REPORT_SCHEMA_VERSION,
    STATUS_NO_MODEL,
    STATUS_NOT_PROVIDED,
    STATUS_PROVIDED_NOT_ANALYSED,
    CaseEvidence,
    EVIDENCE_MODALITIES,
    ModalityEvidence,
)

__all__ = ["build_report", "build_report_prompt"]


_UNKNOWN = "NOT RECORDED"


# ============================================================
# PROMPT — CONTEXT RENDERING
# ============================================================


def _fmt(value: Any, suffix: str = "") -> str:
    if value is None or value == "":
        return _UNKNOWN
    if isinstance(value, float):
        return f"{value:g}{suffix}"
    return f"{value}{suffix}"


def _fmt_list(values: list[str]) -> str:
    return ", ".join(values) if values else _UNKNOWN


def _clinical_block(evidence: CaseEvidence) -> list[str]:
    clinical = evidence.clinical
    lines = ["CLINICAL CONTEXT (entered by the clinician, not model output):"]

    if clinical.is_empty:
        lines.append(
            "  Nothing was recorded. Age, sex, symptoms and history are all "
            "unknown for this case."
        )
        return lines

    age = _UNKNOWN if clinical.age is None else f"{clinical.age} years"
    if clinical.age is not None and clinical.age_source:
        age = f"{age} ({clinical.age_source})"

    lines.extend([
        f"  Age: {age}",
        f"  Sex: {_fmt(clinical.sex)}",
        f"  Blood pressure: {_fmt(clinical.blood_pressure)}",
        f"  Heart rate: {_fmt(clinical.heart_rate, ' bpm')}",
        f"  Symptoms: {_fmt_list(clinical.symptoms)}",
        f"  Relevant history: {_fmt_list(clinical.history)}",
    ])

    if clinical.study_date:
        lines.append(f"  Study date: {clinical.study_date}")

    if clinical.notes:
        lines.append(f"  Clinician notes: {clinical.notes}")

    blank = [
        name.split(":", 1)[-1] for name in clinical.unknown
    ]
    if blank:
        lines.append(
            f"  Not recorded either way: {', '.join(blank)}. These are "
            "UNKNOWN, because the form was left blank. Do not describe them as "
            "absent, denied or normal."
        )

    if clinical.not_collected:
        lines.append(
            f"  Never collected by this application: "
            f"{', '.join(clinical.not_collected)}. Their absence carries no "
            "clinical meaning."
        )

    return lines


def _findings_lines(evidence: ModalityEvidence) -> list[str]:
    lines: list[str] = []

    for finding in evidence.findings:
        name = finding.get("label") or finding.get("name") or "finding"
        observed = finding.get("observed")
        confidence = finding.get("confidence") or {}
        measurement = finding.get("measurement") or {}

        if evidence.key == "ecg":
            probability = confidence.get("probability")
            verdict = "POSITIVE" if observed else "below threshold"
            line = f"    {name}: {verdict}"
            if isinstance(probability, (int, float)):
                line += f", model probability {probability:.3f}"
            precision = confidence.get("measured_precision")
            if isinstance(precision, (int, float)):
                line += (
                    f" (measured precision at this threshold {precision:.2f}, "
                    f"so roughly {(1 - precision) * 100:.0f}% of positive calls "
                    "on the test set were wrong)"
                )
            lines.append(line)
            if finding.get("caveat"):
                lines.append(f"      Caveat: {finding['caveat']}")
            continue

        if not observed:
            lines.append(f"    {name}: not identified in this input")
            if finding.get("absence_meaning"):
                lines.append(f"      {finding['absence_meaning']}")
            continue

        parts: list[str] = []
        if measurement.get("area_cm2"):
            parts.append(f"area {measurement['area_cm2']} cm2 (calibrated)")
        elif measurement.get("percent_of_field") is not None:
            parts.append(
                f"{measurement['percent_of_field']}% of the image field "
                "(relative, not calibrated)"
            )
        if measurement.get("volume_ml") is not None:
            parts.append(f"volume {measurement['volume_ml']} mL")
        if measurement.get("percent_of_analysed") is not None:
            parts.append(
                f"{measurement['percent_of_analysed']}% of the analysed volume"
            )
        if measurement.get("connected_components") is not None:
            parts.append(
                f"{measurement['connected_components']} connected components"
            )

        probability = confidence.get("mean_probability")
        if isinstance(probability, (int, float)):
            parts.append(f"mean model output {probability:.3f}")

        lines.append(
            f"    {name}: identified"
            + (f" — {'; '.join(parts)}" if parts else "")
        )

    return lines


def _metric_scalar(value: Any) -> Optional[float]:
    """
    One number out of a metric that may be a scalar or a spread.

    CCTA reports its metrics as ``{"mean": ..., "sd": ..., "min": ..., "max": ...}``
    because three test cases have a range worth showing; echo and ECG report
    plain floats. Unwrapping here means the prompt does not silently drop a
    metric whose shape it did not expect.
    """
    if isinstance(value, dict):
        value = value.get("mean")
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def _modality_block(evidence: ModalityEvidence) -> list[str]:
    header = f"{evidence.key.upper()} — {evidence.label}:"

    if evidence.status == STATUS_NO_MODEL:
        return [
            header,
            "  NO MODEL. No trained model for this modality exists in this "
            "project. Produce no findings for it.",
        ]

    if evidence.status == STATUS_NOT_PROVIDED:
        return [
            header,
            "  NOT PROVIDED. No input was supplied, so nothing was analysed. "
            "Produce no findings for it and do not infer any from the other "
            "modalities.",
        ]

    if evidence.status == STATUS_PROVIDED_NOT_ANALYSED:
        return [
            header,
            "  NOT ANALYSED. An input exists but no model has been run on it. "
            "Produce no findings for it.",
        ]

    model = evidence.model or {}
    lines = [
        header,
        f"  ANALYSED by {model.get('name', 'the trained model')}.",
        f"  What this model outputs: {evidence.task}",
    ]

    if model.get("dataset"):
        performance: list[str] = []
        for label, key in (
            ("Dice", "test_dice"),
            ("macro AUROC", "macro_AUROC"),
            ("mean HD95", "test_hd95_mm"),
        ):
            value = _metric_scalar(model.get(key))
            if value is not None:
                unit = " mm" if key == "test_hd95_mm" else ""
                performance.append(f"{label} {value:g}{unit}")

        cohort = (
            model.get("test_patients")
            or model.get("test_records")
            or model.get("test_cases")
        )
        line = f"  Held-out performance on {model['dataset']}"
        if cohort:
            line += f" ({cohort} cases)"
        if performance:
            line += f": {', '.join(performance)}"
        lines.append(line + ".")
        lines.append(
            "  That is a cohort average, not the probability that this "
            "particular output is correct."
        )

    if model.get("metric_caveat"):
        lines.append(f"  Metric caveat: {model['metric_caveat']}")

    findings = _findings_lines(evidence)
    if findings:
        lines.append("  Findings reported by the model:")
        lines.extend(findings)
    else:
        lines.append(
            "  The model ran and reported no findings above its reporting "
            "threshold. This is not a normal result; it means nothing crossed "
            "the cutoff."
        )

    if evidence.coverage and evidence.coverage.get("complete") is False:
        lines.append(
            f"  PARTIAL COVERAGE: only "
            f"{evidence.coverage.get('analysed_percent', '?')}% of the input was "
            "examined. The remainder was not looked at and must not be "
            "described as normal."
        )

    for limitation in evidence.limitations:
        lines.append(f"  Limitation: {limitation}")

    return lines


def _cross_modal_block(evidence: CaseEvidence) -> list[str]:
    lines = [
        "OBSERVATIONS ACROSS MODALITIES (co-occurrence only — no model "
        "combined these):"
    ]

    for item in evidence.cross_modal_evidence:
        lines.append(f"  [{item.kind}] {item.statement}")

    lines.append(
        "  Every line above is a listing of things observed together. None of "
        "them is an inference, and none may be presented as one modality "
        "confirming another."
    )
    return lines


def _uncertainty_block(evidence: CaseEvidence) -> list[str]:
    lines = ["UNCERTAINTIES — carry the relevant ones into your answer:"]

    seen: set[str] = set()
    for item in evidence.uncertainties:
        if item.detail in seen:
            continue
        seen.add(item.detail)
        marker = "!" if item.severity == "warning" else "-"
        lines.append(f"  {marker} [{item.scope}] {item.detail}")

    return lines


def build_report_prompt(evidence: CaseEvidence) -> dict[str, str]:
    """
    Render structured evidence into the context and question MedGemma receives.

    Returns ``{"context": ..., "question": ...}``. The context is generated
    entirely from ``evidence``; the only free text that reaches the model is the
    clinician's own symptoms and notes, and those arrive under a heading that
    identifies them as operator-entered rather than measured.
    """
    lines: list[str] = [
        f"CASE {evidence.case_id or '(unsaved)'}",
        "",
        "This context was assembled by a deterministic software layer from the "
        "outputs of independently trained models. It contains no measurement "
        "that a model did not produce.",
        "",
    ]

    lines.extend(_clinical_block(evidence))
    lines.append("")

    for key in EVIDENCE_MODALITIES:
        lines.extend(_modality_block(evidence.modalities[key]))
        lines.append("")

    lines.extend(_cross_modal_block(evidence))
    lines.append("")
    lines.extend(_uncertainty_block(evidence))

    analysed = ", ".join(evidence.available_modalities) or "none"
    absent = ", ".join(evidence.missing_modalities) or "none"

    question = f"""
Write a structured summary of this case for the clinician who ordered it.

Modalities with model output: {analysed}
Modalities with no output: {absent}

Cover, in this order:
1. What was analysed and what was not, naming every modality with no output.
2. What each model that ran actually reported, in its own terms and with its
   own numbers.
3. Which of the stated uncertainties most affect how this case should be read.
4. What a clinician would still need in order to answer a clinical question
   here, given that the models above measure anatomy and screening categories
   rather than function or severity.

Constraints on the summary:
- Do not state a diagnosis, a severity grade, a stenosis percentage, an
  ejection fraction, a risk score or a probability of disease. No model here
  produces any of those.
- Do not combine findings from different modalities into a single conclusion,
  and do not say one modality confirms another.
- Do not describe anything not analysed as normal or as ruled out.
- Use only the numbers given above, unchanged.
- Write prose. No headings, no bullet points, no numbered lists.
""".strip()

    return {"context": "\n".join(lines), "question": question}


# ============================================================
# WORKFLOW RECOMMENDATIONS
# ============================================================


def _recommendations(evidence: CaseEvidence) -> list[dict[str, str]]:
    """
    Next steps for completing the analysis — workflow, never clinical advice.

    Each item says what the software would need to report more, and why. None
    of them recommends a test, a treatment or a follow-up interval: no model in
    this project outputs a management decision, so the report must not contain
    one dressed up as a recommendation.
    """
    items: list[dict[str, str]] = []

    for key in EVIDENCE_MODALITIES:
        modality = evidence.modalities[key]

        if modality.status == STATUS_NOT_PROVIDED:
            items.append({
                "kind": "input",
                "modality": key,
                "action": f"Upload a {modality.label.lower()} input to analyse it.",
                "reason": (
                    "This modality has a trained model available but no input "
                    "was supplied, so it contributed nothing to this report."
                ),
            })
        elif modality.status == STATUS_PROVIDED_NOT_ANALYSED:
            items.append({
                "kind": "analysis",
                "modality": key,
                "action": f"Run the {modality.label.lower()} analysis.",
                "reason": "An input is attached to this case but no model has run on it.",
            })
        elif modality.status == STATUS_NO_MODEL:
            items.append({
                "kind": "capability",
                "modality": key,
                "action": (
                    f"{modality.label}: nothing to do — this capability does not "
                    "exist in CardioVision."
                ),
                "reason": "No trained model for this modality exists in this project.",
            })

        if modality.coverage and modality.coverage.get("complete") is False:
            items.append({
                "kind": "coverage",
                "modality": key,
                "action": (
                    "Re-run this volume with a larger window budget, or crop to "
                    "the region of interest, to analyse the part that was skipped."
                ),
                "reason": (
                    f"Only {modality.coverage.get('analysed_percent', '?')}% of "
                    "the input was examined."
                ),
            })

    if evidence.clinical.is_empty:
        items.append({
            "kind": "input",
            "modality": "clinical",
            "action": "Record age, sex, symptoms and history for this case.",
            "reason": (
                "No clinical context was entered, so nothing in this report can "
                "be read against the patient's presentation."
            ),
        })
    elif evidence.clinical.unknown:
        items.append({
            "kind": "input",
            "modality": "clinical",
            "action": (
                "Complete the blank clinical fields: "
                f"{', '.join(name.split(':', 1)[-1] for name in evidence.clinical.unknown)}."
            ),
            "reason": (
                "Blank fields are recorded as unknown, which is weaker than a "
                "recorded negative."
            ),
        })

    items.append({
        "kind": "interpretation",
        "modality": "case",
        "action": (
            "A qualified clinician must interpret this case. Read every finding "
            "above alongside the images it came from."
        ),
        "reason": (
            "CardioVision is a research prototype. Its models measure anatomy "
            "and screening categories; none of them grades severity, and none "
            "of them was validated for clinical use."
        ),
    })

    return items


# ============================================================
# REPORT ASSEMBLY
# ============================================================


def _patient_block(patient: Optional[dict[str, Any]]) -> dict[str, Any]:
    """
    Identity fields for the report header.

    Only what the case actually holds. No filesystem path, no archive name and
    no accession or study UID reaches the report, per the project's rule that
    responses do not leak storage layout.
    """
    patient = patient or {}

    return {
        "name": patient.get("name") or None,
        "patient_id": patient.get("patientId") or None,
        "sex": patient.get("sex") or None,
        "date_of_birth": patient.get("dateOfBirth") or None,
        "study_date": patient.get("studyDate") or None,
    }


def build_report(
    evidence: CaseEvidence,
    patient: Optional[dict[str, Any]] = None,
    ai_summary: Optional[str] = None,
    ai_error: Optional[str] = None,
) -> dict[str, Any]:
    """
    Assemble the structured clinical AI report.

    ``ai_summary`` is optional by design. When MedGemma is unavailable the
    report is returned complete apart from that one field, with ``ai_error``
    recorded next to it, because every finding in the report came from a
    modality model and none of them depends on the language model.
    """
    generated_at = datetime.now(timezone.utc).isoformat(timespec="seconds")

    generated_by: dict[str, Any] = {
        "application": APP_NAME,
        "version": APP_VERSION,
        "evidence_layer": "deterministic software aggregation (no learned fusion)",
        "summary_model": MEDGEMMA_NAME if ai_summary else None,
        "summary_available": bool(ai_summary),
    }

    if ai_error and not ai_summary:
        generated_by["summary_unavailable_reason"] = ai_error

    return {
        "schema_version": REPORT_SCHEMA_VERSION,
        "case_id": evidence.case_id,
        "generated_at": generated_at,
        "patient": _patient_block(patient),
        "modality_results": {
            key: evidence.modalities[key].to_dict()
            for key in EVIDENCE_MODALITIES
        },
        "clinical_context": evidence.clinical.to_dict(),
        "integrated_evidence": {
            "available_modalities": list(evidence.available_modalities),
            "missing_modalities": list(evidence.missing_modalities),
            "cross_modal_evidence": [
                item.to_dict() for item in evidence.cross_modal_evidence
            ],
            "integration_method": evidence.to_dict()["integration_method"],
        },
        "uncertainties": [item.to_dict() for item in evidence.uncertainties],
        "ai_summary": ai_summary,
        "ai_summary_error": ai_error if not ai_summary else None,
        "ai_summary_scope": (
            "Written by a language model from the structured evidence in this "
            "report. It introduces no new measurement and does not override any "
            "model output. Where it disagrees with the modality results above, "
            "the modality results are authoritative."
        ),
        "recommendations": _recommendations(evidence),
        "recommendations_scope": (
            "Workflow steps for completing this analysis. Not clinical "
            "management advice — no model in this project produces one."
        ),
        "model_versions": dict(evidence.model_versions),
        "disclaimer": (
            f"{APP_NAME} is a research prototype. It is not a medical device, "
            "it has no regulatory clearance, and no output in this report may "
            "be used as the basis of a clinical decision."
        ),
    }
