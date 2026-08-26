"""
CardioVision AI — evidence and report data structures.

One place that defines the shape of an integrated case, so the API schema, the
prompt builder, the frontend contract and the tests all agree. Changing a field
name here is a breaking change to the report; that is the point of it being one
file rather than five dictionaries built in five modules.

Every dataclass has a ``to_dict`` because the API returns plain JSON and pydantic
models would put a second, drifting definition of these shapes in
``api/schemas.py``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

# Bumped when a field is renamed or removed. Stored on saved reports so an old
# report can be recognised as old rather than silently misread.
REPORT_SCHEMA_VERSION = "1.0"

# Every modality the evidence layer knows about, in the order the target
# architecture lists them: CCTA -> Echo -> ECG -> clinical.
EVIDENCE_MODALITIES = ("ccta", "echo", "ecg")

# The subset backed by an imaging or signal model. "clinical" is deliberately
# not here: clinical data is operator-entered context, not model output, and
# treating it as a modality result is how a typed blood pressure ends up
# presented as a finding.
IMAGING_MODALITIES = ("ccta", "echo", "ecg")


# ============================================================
# STATUS VOCABULARY
# ============================================================
#
# Four distinct states that a naive `available: true/false` collapses into two.
# The collapse is exactly the failure this project keeps guarding against: "no
# finding" and "never looked" must not render the same way.

STATUS_ANALYSED = "analysed"
STATUS_NOT_PROVIDED = "not_provided"
STATUS_PROVIDED_NOT_ANALYSED = "provided_not_analysed"
STATUS_NO_MODEL = "no_model"

STATUS_MEANING = {
    STATUS_ANALYSED: "A trained model ran on the input and produced the findings below.",
    STATUS_NOT_PROVIDED: "No input was supplied for this modality, so nothing was analysed.",
    STATUS_PROVIDED_NOT_ANALYSED: "An input was supplied but analysis has not been run on it.",
    STATUS_NO_MODEL: "No trained model exists for this modality in this project.",
}


# ============================================================
# CLINICAL CONTEXT
# ============================================================


@dataclass
class ClinicalContext:
    """
    Operator-entered clinical information, normalised.

    ``None`` and ``[]`` here mean NOT RECORDED. They never mean absent, normal
    or denied. ``not_collected`` names the fields this application has no input
    for at all, which is a different thing again from a field the clinician left
    blank, and ``unknown`` names the ones that were offered and skipped.
    """

    age: Optional[int] = None
    age_source: Optional[str] = None
    sex: Optional[str] = None
    blood_pressure: Optional[str] = None
    heart_rate: Optional[int] = None
    symptoms: list[str] = field(default_factory=list)
    history: list[str] = field(default_factory=list)
    medications: list[str] = field(default_factory=list)
    notes: Optional[str] = None
    study_date: Optional[str] = None
    recorded: list[str] = field(default_factory=list)
    unknown: list[str] = field(default_factory=list)
    not_collected: list[str] = field(default_factory=list)

    @property
    def is_empty(self) -> bool:
        return not self.recorded

    def to_dict(self) -> dict[str, Any]:
        return {
            "age": self.age,
            "age_source": self.age_source,
            "sex": self.sex,
            "blood_pressure": self.blood_pressure,
            "heart_rate": self.heart_rate,
            "symptoms": list(self.symptoms),
            "history": list(self.history),
            "medications": list(self.medications),
            "notes": self.notes,
            "study_date": self.study_date,
            "recorded_fields": list(self.recorded),
            "unknown_fields": list(self.unknown),
            "not_collected_fields": list(self.not_collected),
            "source": "clinician-entered; not model output",
            "interpretation": (
                "null and [] mean NOT RECORDED. They do not mean absent, "
                "normal or denied."
            ),
        }


# ============================================================
# MODALITY EVIDENCE
# ============================================================


@dataclass
class ModalityEvidence:
    """What one modality contributed to this case, including nothing."""

    key: str
    label: str
    status: str
    model_available: bool
    input_provided: bool
    analysed: bool
    model: Optional[dict[str, Any]] = None
    task: Optional[str] = None
    findings: list[dict[str, Any]] = field(default_factory=list)
    measurements: dict[str, Any] = field(default_factory=dict)
    confidence: dict[str, Any] = field(default_factory=dict)
    explainability: dict[str, Any] = field(default_factory=dict)
    coverage: dict[str, Any] = field(default_factory=dict)
    limitations: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "modality": self.key,
            "label": self.label,
            "status": self.status,
            "status_meaning": STATUS_MEANING.get(self.status, ""),
            "model_available": self.model_available,
            "input_provided": self.input_provided,
            "analysed": self.analysed,
            "task": self.task,
            "model": self.model,
            "findings": list(self.findings),
            "measurements": dict(self.measurements),
            "confidence": dict(self.confidence),
            "explainability": dict(self.explainability),
            "coverage": dict(self.coverage),
            "limitations": list(self.limitations),
            "notes": list(self.notes),
        }


# ============================================================
# CROSS-MODAL OBSERVATIONS
# ============================================================


@dataclass
class CrossModalObservation:
    """
    Two or more things that were observed together.

    ``inference`` is always ``"none"``. It is a field rather than a docstring so
    that it travels with the data into the prompt, the API response and the UI,
    where the temptation to read a relationship into a pair of findings actually
    arises.
    """

    kind: str
    statement: str
    modalities: list[str]
    basis: str
    inference: str = "none"

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "statement": self.statement,
            "modalities": list(self.modalities),
            "basis": self.basis,
            "inference": self.inference,
        }


# ============================================================
# UNCERTAINTY
# ============================================================


@dataclass
class Uncertainty:
    """One reason a reader should trust something below less than it looks."""

    scope: str          # a modality key, "clinical", or "case"
    kind: str           # model_limitation | input_quality | coverage | contradiction | pairing
    detail: str
    severity: str = "note"      # note | warning

    def to_dict(self) -> dict[str, Any]:
        return {
            "scope": self.scope,
            "kind": self.kind,
            "detail": self.detail,
            "severity": self.severity,
        }


# ============================================================
# CASE EVIDENCE
# ============================================================


@dataclass
class CaseEvidence:
    """The deterministic integration of everything known about one case."""

    case_id: Optional[str]
    modalities: dict[str, ModalityEvidence]
    clinical: ClinicalContext
    available_modalities: list[str]
    missing_modalities: list[str]
    cross_modal_evidence: list[CrossModalObservation]
    uncertainties: list[Uncertainty]
    model_versions: dict[str, Any] = field(default_factory=dict)
    generated_at: Optional[str] = None
    schema_version: str = REPORT_SCHEMA_VERSION

    @property
    def has_any_findings(self) -> bool:
        return any(
            evidence.analysed for evidence in self.modalities.values()
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "case_id": self.case_id,
            "generated_at": self.generated_at,
            "modalities": {
                key: evidence.to_dict()
                for key, evidence in self.modalities.items()
            },
            "clinical_context": self.clinical.to_dict(),
            "available_modalities": list(self.available_modalities),
            "missing_modalities": list(self.missing_modalities),
            "cross_modal_evidence": [
                item.to_dict() for item in self.cross_modal_evidence
            ],
            "uncertainties": [item.to_dict() for item in self.uncertainties],
            "model_versions": dict(self.model_versions),
            "integration_method": {
                "type": "deterministic software aggregation",
                "learned_fusion": False,
                "note": (
                    "No multimodal model exists in this project. Findings are "
                    "collected side by side; they are not weighted, combined "
                    "or reconciled by any trained network, and no joint risk "
                    "score is produced."
                ),
            },
        }
