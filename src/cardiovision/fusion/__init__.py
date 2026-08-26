"""
CardioVision AI — multimodal evidence integration.

NOT a fusion model. There is no trained multimodal network in this project and
none is being added: ``notebooks/04_Multimodal_Fusion.ipynb`` is empty and
``MODALITY_STATUS["fusion"]`` says so. What lives here is a deterministic
software layer that collects what the three independent models actually
reported, records what was not available, and states the uncertainties — with
no learned weighting, no combined risk score and no cross-modal conclusion.

Why that boundary matters
-------------------------
The three models were trained on three unrelated public datasets (CAMUS,
PTB-XL, MedHK23/CCA). No patient appears in more than one of them, and nothing
has ever been trained on paired multimodal data. So a statement of the form
"the CT and the ECG together indicate X" has no model behind it anywhere in
this repository. Every cross-modal item this module emits is therefore tagged
``inference: "none"`` and is phrased as co-occurrence: two things were observed,
and their relationship is for the reader to judge.
"""

from cardiovision.fusion.evidence import (
    build_case_evidence,
    normalise_clinical,
)
from cardiovision.fusion.report import (
    build_report,
    build_report_prompt,
)
from cardiovision.fusion.schema import (
    CaseEvidence,
    ClinicalContext,
    CrossModalObservation,
    EVIDENCE_MODALITIES,
    IMAGING_MODALITIES,
    ModalityEvidence,
    REPORT_SCHEMA_VERSION,
    Uncertainty,
)

__all__ = [
    "CaseEvidence",
    "ClinicalContext",
    "CrossModalObservation",
    "EVIDENCE_MODALITIES",
    "IMAGING_MODALITIES",
    "ModalityEvidence",
    "REPORT_SCHEMA_VERSION",
    "Uncertainty",
    "build_case_evidence",
    "build_report",
    "build_report_prompt",
    "normalise_clinical",
]
