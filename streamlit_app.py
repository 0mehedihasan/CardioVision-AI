"""
CardioVision AI — Streamlit interface.

A second UI client over the same application core the FastAPI backend and the
React frontend use. Nothing medical is implemented here: every decode, forward
pass, saliency map, figure, evidence table and narrative comes from
``src/cardiovision/``, through :mod:`cardiovision.analysis` and the same model
singletons the API serves. This file is layout, widgets and session state.

    streamlit run streamlit_app.py

Two deliberate differences from the React client:

**No sign-in.** This is a research and demonstration surface, so it has no login
form and no session tokens. It reaches the core in-process rather than over HTTP,
so it neither uses nor weakens the authentication the API enforces on every one
of its own routes.

**Nothing loads until it is asked for.** The three checkpoints and MedGemma are
several gigabytes between them. Each is loaded on first use behind
``st.cache_resource`` and then reused for the life of the server process — the
dashboard reports load state without triggering a load.

**Nothing is persisted.** The React client owns case management: patient records,
the SQLite store and the case lifecycle belong to the authenticated application.
This interface has no login, so it writes nothing — an analysis lives in
``st.session_state`` for as long as the browser tab does and is gone afterwards.
It opens no database, which is also what makes it safe to run somewhere the
patient store must not exist.
"""

from __future__ import annotations

import base64
import json
import os
import sys
from pathlib import Path
from typing import Any, Optional

import streamlit as st

# A checkout that has not been `pip install -e .`'d still has src/ next to this
# file. Relative to __file__, never an absolute path.
_SRC = Path(__file__).resolve().parent / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from cardiovision.analysis import (  # noqa: E402  (after the path bootstrap)
    AnalysisError,
    analyze_ccta,
    analyze_ecg,
    analyze_echo,
)
from cardiovision.config import (  # noqa: E402
    ALLOWED_CCTA_SUFFIXES,
    ALLOWED_ECG_SUFFIXES,
    ALLOWED_ECHO_SUFFIXES,
    APP_NAME,
    APP_VERSION,
    CCTA_MAX_WINDOWS,
    CCTA_WEAK_NOTES,
    DEVICE,
    ECG_CLASS_NAMES,
    ECG_THRESHOLD,
    ECHO_TRAINING_ORIENTATION,
    MEDGEMMA_PATH,
    MODALITY_STATUS,
    SAMPLES_DIR,
)
from cardiovision.fusion import build_case_evidence, build_report  # noqa: E402
from cardiovision.fusion.report import build_report_prompt  # noqa: E402
from cardiovision.inference.ccta import (  # noqa: E402
    CctaModelUnavailable,
    ccta_segmenter,
)
from cardiovision.inference.ecg import EcgModelUnavailable, ecg_classifier  # noqa: E402
from cardiovision.inference.echo import (  # noqa: E402
    EchoModelUnavailable,
    echo_segmenter,
)
from cardiovision.inference.medgemma import (  # noqa: E402
    MedGemmaUnavailable,
    medgemma,
)
from cardiovision.services.case_context import build_case_context  # noqa: E402

# ============================================================
# PAGE
# ============================================================

st.set_page_config(
    page_title=f"{APP_NAME} — research interface",
    page_icon="🫀",
    layout="wide",
)

DISCLAIMER = (
    "Research software. None of the three models has regulatory clearance, "
    "prospective evaluation or a reader study, and nothing here is a diagnosis. "
    "Every output is material for a qualified clinician to review."
)

SECTIONS = (
    "Dashboard",
    "CCTA",
    "Echocardiography",
    "ECG",
    "AI Assistant",
    "Sample Cases",
    "About / Developer",
)


# ============================================================
# LAZY RESOURCES
# ============================================================
#
# One entry per model: the singleton, the exception that means "not available"
# rather than "broken", the environment flag that skips it, and a label for the
# message. The same objects, flags and semantics as api/app.py's startup —
# loaded on demand instead of eagerly, because a Streamlit reader who only wants
# the ECG section should not wait for 8.6 GB of language model.

_MODELS: dict[str, tuple[Any, type[Exception], str, str]] = {
    "ccta": (
        ccta_segmenter,
        CctaModelUnavailable,
        "CARDIOVISION_SKIP_CCTA",
        "CCTA segmentation model",
    ),
    "echo": (
        echo_segmenter,
        EchoModelUnavailable,
        "CARDIOVISION_SKIP_ECHO",
        "echo segmentation model",
    ),
    "ecg": (
        ecg_classifier,
        EcgModelUnavailable,
        "CARDIOVISION_SKIP_ECG",
        "ECG classification model",
    ),
    "medgemma": (
        medgemma,
        MedGemmaUnavailable,
        "CARDIOVISION_SKIP_MEDGEMMA",
        "clinical language model",
    ),
}


def _truthy(value: Optional[str]) -> bool:
    return (value or "").strip().lower() in ("1", "true", "yes", "on")


@st.cache_resource(show_spinner=False)
def load_model(key: str) -> tuple[Optional[Any], Optional[str]]:
    """
    Load one model once per server process. Returns ``(model, error)``.

    A missing checkpoint is a reportable state, not an exception to propagate:
    one absent file must cost exactly one modality, which is the same rule the
    API's startup follows.
    """
    model, expected, flag, label = _MODELS[key]

    if _truthy(os.environ.get(flag)):
        return None, f"The {label} is disabled because {flag} is set."

    try:
        model.load()
    except expected as error:
        return None, str(error)
    except Exception as error:                          # pragma: no cover
        return None, f"Unexpected {label} failure: {error}"

    return model, None


def medgemma_state() -> tuple[bool, Optional[str]]:
    """
    Whether the language model can be used, without loading it.

    ``MEDGEMMA_PATH`` is ~8.6 GB of vendor weights under Google's terms, so it is
    gitignored and absent from every clone until someone downloads it — and it is
    absent from any hosted deployment that builds from the repository. Checking
    the directory rather than calling ``load()`` lets the AI Assistant say so up
    front instead of failing on the first question.
    """
    if _truthy(os.environ.get("CARDIOVISION_SKIP_MEDGEMMA")):
        return False, "disabled because CARDIOVISION_SKIP_MEDGEMMA is set"

    if medgemma.is_loaded:
        return True, None

    if not MEDGEMMA_PATH.is_dir() or not any(MEDGEMMA_PATH.iterdir()):
        return False, "the weights are not present in this deployment"

    return True, None


# ============================================================
# HELPERS
# ============================================================


def show_data_url(url: str, caption: Optional[str] = None) -> None:
    """
    Render one figure from the core.

    Every renderer returns a base64 data URL — PNG for the image panels, SVG for
    the ECG strip — so the two are told apart by the media type rather than by
    which section is drawing them.
    """
    header, _, encoded = url.partition(",")

    try:
        payload = base64.b64decode(encoded)
    except Exception:                                   # pragma: no cover
        st.caption("This figure could not be decoded.")
        return

    if "svg" in header:
        # White backing: the strips are drawn for a light background and the
        # gridlines vanish against Streamlit's dark theme.
        st.markdown(
            '<div style="background:#ffffff;padding:8px;border-radius:6px">'
            + payload.decode("utf-8")
            + "</div>",
            unsafe_allow_html=True,
        )

        if caption:
            st.caption(caption)

        return

    try:
        st.image(payload, caption=caption, use_container_width=True)
    except TypeError:                                   # pragma: no cover
        # Older Streamlit spells the same argument differently.
        st.image(payload, caption=caption)


def show_notes(notes: Any) -> None:
    """Provenance notes from the loader, the model and the renderer."""
    items = [str(note) for note in (notes or [])]

    if not items:
        return

    with st.expander(f"Provenance notes ({len(items)})"):
        for note in items:
            st.markdown(f"- {note}")


def case_state() -> dict[str, Any]:
    """
    The working case, in exactly the shape ``fusion`` and ``case_context`` read.

    Held in session state so the AI Assistant and the integrated report see the
    analyses this browser session produced, without either of them re-running a
    model. It is never written to disk: ``patient`` and ``clinical`` stay empty
    here because collecting demographics belongs to the authenticated client, and
    the evidence layer reports both as not provided rather than as normal.
    """
    if "case" not in st.session_state:
        st.session_state.case = {
            "patient": {},
            "clinical": {},
            "ccta": None,
            "echo": None,
            "ecg": None,
            # Which modalities a file was supplied for. The evidence layer uses
            # this to tell "no file" apart from "file but never analysed", and
            # neither of those is a normal finding.
            "modalities_provided": {},
        }

    return st.session_state.case


def record_analysis(modality: str, payload: dict[str, Any]) -> None:
    """Attach one analysis to the working case."""
    case = case_state()
    case[modality] = payload
    case["modalities_provided"][modality] = True


def mark_provided(modality: str) -> None:
    """A file arrived for this modality, whether or not it was analysed."""
    case_state()["modalities_provided"][modality] = True


def suffix_help(suffixes: set[str]) -> str:
    return ", ".join(sorted(suffixes))


def upload_types(suffixes: set[str]) -> list[str]:
    """
    Streamlit's ``type=`` wants extensions without the dot, and it matches only
    the final one — so ``.nii.gz`` has to be offered as ``gz``.
    """
    types: set[str] = set()

    for suffix in suffixes:
        types.add(suffix.lstrip(".").split(".")[-1])

    return sorted(types)


# ============================================================
# SAMPLE DISCOVERY
# ============================================================
#
# Discovered from the tracked samples/ directory rather than hardcoded, so a
# clone without the large volumes reports them as absent instead of failing on
# a path that is not there. Only names and paths are collected here; no file is
# read until an analysis is actually requested.


@st.cache_data(show_spinner=False)
def discover_samples() -> dict[str, list[dict[str, Any]]]:
    """Sample inputs on disk, by modality."""
    found: dict[str, list[dict[str, Any]]] = {"ccta": [], "echo": [], "ecg": []}

    if not SAMPLES_DIR.is_dir():
        return found

    # ---- CCTA: one directory per case, volume plus expert annotation ----
    ccta_root = SAMPLES_DIR / "ccta"

    if ccta_root.is_dir():
        for case_dir in sorted(p for p in ccta_root.iterdir() if p.is_dir()):
            volume = case_dir / "ccta_image.nii.gz"

            if not volume.is_file():
                continue

            truth = case_dir / "ground_truth.nii.gz"

            found["ccta"].append({
                "id": case_dir.name,
                "label": case_dir.name,
                "path": str(volume),
                "size_mb": round(volume.stat().st_size / (1024 * 1024), 1),
                "ground_truth": str(truth) if truth.is_file() else None,
                "dataset": "MedHK23/CCA",
            })

    # ---- echo: one directory per patient, several frames each ----
    echo_root = SAMPLES_DIR / "echo"

    if echo_root.is_dir():
        for patient_dir in sorted(p for p in echo_root.iterdir() if p.is_dir()):
            frames: list[dict[str, Any]] = []

            for frame in sorted(patient_dir.glob("*.nii.gz")):
                name = frame.name

                # The _gt files are the dataset's own label maps, not images.
                # They are offered beside a prediction as a reference and never
                # sent through the model.
                if "_gt." in name:
                    continue

                truth = frame.with_name(name.replace(".nii.gz", "_gt.nii.gz"))

                frames.append({
                    "id": name.replace(".nii.gz", ""),
                    "path": str(frame),
                    "ground_truth": str(truth) if truth.is_file() else None,
                })

            if not frames:
                continue

            found["echo"].append({
                "id": patient_dir.name,
                "label": patient_dir.name,
                "frames": frames,
                "info": sorted(str(p) for p in patient_dir.glob("Info_*.cfg")),
                "citation": (
                    str(patient_dir / "MANDATORY_CITATION.md")
                    if (patient_dir / "MANDATORY_CITATION.md").is_file()
                    else None
                ),
                "dataset": "CAMUS",
            })

    # ---- ECG: WFDB pairs, header plus signal file ----
    ecg_root = SAMPLES_DIR / "ecg"

    if ecg_root.is_dir():
        for header in sorted(ecg_root.glob("*.hea")):
            companions = [
                str(path)
                for suffix in (".mat", ".dat")
                for path in [header.with_suffix(suffix)]
                if path.is_file()
            ]

            if not companions:
                # A header alone is not readable, and pretending otherwise
                # would fail inside the loader instead of here.
                continue

            found["ecg"].append({
                "id": header.stem,
                "label": header.stem,
                "path": str(header),
                "companions": companions,
                "dataset": "PTB-XL",
            })

    return found


# ============================================================
# RUNNING THE SHARED PIPELINE
# ============================================================
#
# One entry point per modality. Both the upload path and the sample path go
# through these, so a sample case is analysed by exactly the same code as a file
# an operator dragged in — no precomputed results are substituted.


def run_analysis(modality: str, spinner: str, **kwargs: Any) -> Optional[dict[str, Any]]:
    """
    Load the model if needed, run the shared analysis, keep it in the session.

    No ``case_id`` is passed, so the core archives nothing: this client does not
    write to the case store. Returns ``None`` after reporting the failure, so the
    caller can simply stop.
    """
    functions = {
        "ccta": analyze_ccta,
        "echo": analyze_echo,
        "ecg": analyze_ecg,
    }

    with st.spinner(f"Loading the {modality.upper()} model…"):
        model, error = load_model(modality)

    if model is None:
        st.error(error or f"The {modality.upper()} model is not available.")
        return None

    try:
        with st.spinner(spinner):
            payload = functions[modality](**kwargs)
    except AnalysisError as error:
        st.error(str(error))
        return None
    except Exception as error:                          # pragma: no cover
        st.error(f"The analysis failed: {error}")
        return None

    record_analysis(modality, payload)

    return payload


def read_sample(path: str) -> bytes:
    """
    Read one sample file at the moment it is needed.

    Not cached: the CCTA volumes are 70 MB and 97 MB, and holding them in a
    Streamlit cache would keep both resident for the life of the process.
    """
    return Path(path).read_bytes()


def show_ground_truth(path: str, caption: str) -> None:
    """
    Display a dataset label map beside a prediction, as a reference.

    This is the dataset's own expert annotation. It is not a model output and is
    never fed to a model — it is here so a prediction can be eyeballed against
    the reference the model was scored on. Two cases is an anecdote, not a
    metric.
    """
    try:
        import nibabel as nib
        import numpy as np

        from cardiovision.rendering.echo import colorize_mask
        from cardiovision.rendering.primitives import to_png_data_url

        array = np.asarray(nib.load(path).dataobj)
    except Exception as error:
        st.caption(f"The reference annotation could not be read ({error}).")
        return

    if array.ndim != 2:
        st.caption(
            "The reference annotation for this file is a volume or a sequence, "
            "so it is not drawn here."
        )
        return

    show_data_url(
        # No transpose and no reorientation, matching the image loader exactly:
        # a fix-up here would put the reference in a different orientation from
        # the prediction it is meant to be compared against.
        to_png_data_url(colorize_mask(array.astype("int64"))),
        caption=caption,
    )


# ============================================================
# RESULT VIEWS
# ============================================================


def show_echo_result(payload: dict[str, Any]) -> None:
    structures = payload.get("structures") or []
    quantification = payload.get("quantification") or {}
    orientation = payload.get("orientation") or {}
    explainability = payload.get("explainability") or {}
    images = payload.get("images") or {}

    st.success(
        f"Segmented in {payload.get('inference_ms', 0)} ms on "
        f"{str(payload.get('device', '')).upper()}."
    )

    st.markdown("#### Structures")

    st.dataframe(
        [
            {
                "Structure": item.get("name"),
                "Identified": "yes" if item.get("present") else "no",
                "Pixels": item.get("pixels"),
                "Area (cm²)": item.get("area_cm2"),
                "Share of frame (%)": item.get("area_percent"),
                "Mean confidence": item.get("mean_confidence"),
            }
            for item in structures
        ],
        hide_index=True,
    )

    st.caption(quantification.get("presence_rule", ""))
    st.caption(
        "A structure below the threshold is reported as **not identified**, "
        "which is a size cutoff — not a clinical absence."
    )

    columns = st.columns(3)

    with columns[0]:
        if images.get("original"):
            show_data_url(images["original"], "Input, as analysed")

    with columns[1]:
        if images.get("overlay"):
            show_data_url(images["overlay"], "Segmentation overlay")

    with columns[2]:
        # Hidden entirely when the gradient was unavailable: an all-zero
        # saliency map still renders as a smooth, convincing picture.
        if explainability.get("available") and images.get("saliency_overlay"):
            show_data_url(
                images["saliency_overlay"],
                f"Input-gradient saliency · {explainability.get('target_class')}",
            )
        else:
            st.info("Saliency was not available for this run, so it is not shown.")

    st.caption(
        f"Analysed orientation: rotation {orientation.get('rotation_applied')}°, "
        f"flip {'yes' if orientation.get('flip_applied') else 'no'}. "
        f"Training orientation: {ECHO_TRAINING_ORIENTATION}."
    )

    show_notes(payload.get("notes"))


def show_ccta_result(payload: dict[str, Any]) -> None:
    coverage = payload.get("coverage") or {}
    findings = payload.get("findings") or []
    explainability = payload.get("explainability") or {}
    figures = payload.get("figures") or {}

    st.success(
        f"Segmented in {payload.get('inference_ms', 0)} ms on "
        f"{str(payload.get('device', '')).upper()} at threshold "
        f"{payload.get('threshold')}."
    )

    if not coverage.get("complete", True):
        st.warning(
            f"Only {coverage.get('analysed_percent')}% of the volume was "
            f"analysed ({coverage.get('windows_run')} of "
            f"{coverage.get('windows_total')} windows). "
            + str(coverage.get("note", ""))
        )

    st.markdown("#### Findings")

    st.dataframe(
        [
            {
                "Structure": item.get("name"),
                "Identified": "yes" if item.get("present") else "no",
                "Voxels": item.get("voxels"),
                "Volume (mL)": item.get("volume_ml"),
                "Share of analysed volume (%)": item.get("percent_of_analysed"),
                "Mean probability": item.get("mean_probability"),
                "Max probability": item.get("max_probability"),
                "Components": item.get("components"),
            }
            for item in findings
        ],
        hide_index=True,
    )

    st.caption((payload.get("quantification") or {}).get("note", ""))

    for limitation in payload.get("limitations") or []:
        st.warning(limitation)

    if figures:
        names = sorted(figures)
        chosen = st.selectbox(
            "Panel",
            names,
            key=f"ccta_panel_{payload.get('inference_ms')}",
        )
        show_data_url(figures[chosen], chosen.replace("_", " "))

        if not explainability.get("available") and chosen.startswith("gradcam"):
            st.info("Grad-CAM was not available for this run.")
    else:
        st.info("Figures were not rendered for this run.")

    if explainability.get("available"):
        st.caption(
            f"{explainability.get('method')} on "
            f"{explainability.get('target_layer')} · "
            + str(explainability.get("note", ""))
        )

    show_notes(payload.get("notes"))


def show_ecg_result(payload: dict[str, Any]) -> None:
    predictions = payload.get("predictions") or []
    positive = payload.get("positive_classes") or []
    leads = payload.get("lead_attribution") or []
    figures = payload.get("figures") or {}

    st.success(
        f"Classified in {payload.get('inference_ms', 0)} ms on "
        f"{str(payload.get('device', '')).upper()}."
    )

    st.markdown("#### Class probabilities")

    # Every class, not only the calls: the operating point is a choice, and a
    # reader cannot revisit it without seeing the probabilities under it.
    st.dataframe(
        [
            {
                "Class": item.get("name"),
                "Superclass": item.get("label"),
                "Probability": item.get("probability"),
                f"Called at p ≥ {ECG_THRESHOLD}": (
                    "yes" if item.get("positive") else "no"
                ),
                "Test precision": (item.get("operating_point") or {}).get(
                    "precision"
                ),
                "Test recall": (item.get("operating_point") or {}).get("recall"),
            }
            for item in predictions
        ],
        hide_index=True,
    )

    st.caption(payload.get("threshold_note", ""))
    st.caption(
        "Precision and recall are dataset-level figures from the held-out "
        "PTB-XL test split at this threshold. They do not describe this "
        "recording."
    )

    if not positive:
        st.info(
            "No superclass reached the threshold. That is a completed result — "
            "\"none of these five above the operating point\" — not a normal "
            "study and not an absence of disease."
        )

    for name, caveat in (payload.get("weak_class_warnings") or {}).items():
        st.warning(f"**{name}** — {caveat}")

    if payload.get("saliency_available"):
        st.caption(
            "Lead attribution is input-gradient saliency for "
            f"{payload.get('saliency_class')}. It is not Grad-CAM, and it "
            "shows which leads the output moved with, not where disease is."
        )
    else:
        st.info(
            "Lead attribution is unavailable for this recording, so it is not "
            "shown rather than shown as zero."
        )

    if leads:
        with st.expander("Per-lead attribution scores"):
            st.dataframe(
                [
                    {
                        "Lead": lead.get("name"),
                        "Relative score (%)": lead.get("percent"),
                    }
                    for lead in leads
                ],
                hide_index=True,
            )

    # render_ecg_images returns "strip" and "lead_attribution"; either can be
    # missing when rendering degraded, so both are checked.
    for key, caption in (
        ("strip", "12-lead strip"),
        ("lead_attribution", "lead attribution"),
    ):
        if figures.get(key):
            show_data_url(figures[key], caption)

    show_notes(payload.get("notes"))


# ============================================================
# SECTIONS
# ============================================================


def section_dashboard() -> None:
    st.title(APP_NAME)
    st.caption(f"Version {APP_VERSION} · research interface")
    st.warning(DISCLAIMER)

    st.markdown("### Modalities")

    # MODALITY_STATUS is the single source of truth for what exists. The React
    # client reads it through /api/health; this one reads it directly.
    rows = []
    for key, status in MODALITY_STATUS.items():
        available = bool(status.get("available"))

        # .is_loaded only — reading it must never trigger a multi-gigabyte load
        # just because someone opened the dashboard.
        singleton = _MODELS.get(key, (None,))[0]

        if not available:
            state = "no trained model in this project"
        elif singleton is None:
            state = "—"
        elif singleton.is_loaded:
            state = "loaded"
        else:
            state = "not loaded yet (loads on first use)"

        rows.append({
            "Modality": status.get("label", key.upper()),
            "Implemented": "yes" if available else "no",
            "Model": status.get("model") or "—",
            "In this process": state,
            "Note": status.get("note") or "",
        })

    st.dataframe(rows, hide_index=True)

    st.markdown("### Runtime")

    narrative_ready, narrative_reason = medgemma_state()

    left, right = st.columns(2)
    left.metric("Configured device", str(DEVICE).upper())
    right.metric("Report narrative", "available" if narrative_ready else "unavailable")

    if not narrative_ready:
        st.info(
            "The clinical language model is not available here — "
            f"{narrative_reason}. Segmentation, classification, explainability "
            "and the structured report are unaffected: the report layer is "
            "deterministic and needs no language model. See **AI Assistant** for "
            "how to enable narrative text locally."
        )

    st.caption(
        "Nothing is saved. Results live in this browser session only — patient "
        "records and the case store belong to the authenticated React client."
    )

    st.markdown("### What this interface does not do")
    st.markdown(
        "- no diagnosis, no stenosis grade, no calcium score, no CAD-RADS\n"
        "- no ejection fraction, strain or chamber volume over a cycle\n"
        "- no risk score — there is no risk model in this project\n"
        "- no per-case confidence: every published metric is dataset-level\n"
        "- no patient records, no database, no case lifecycle"
    )


def section_ccta() -> None:
    st.title("Coronary CT angiography")
    st.caption(MODALITY_STATUS["ccta"]["task"])

    for note in CCTA_WEAK_NOTES:
        st.warning(note)

    upload = st.file_uploader(
        "CT volume",
        type=upload_types(ALLOWED_CCTA_SUFFIXES),
        help=f"Accepted: {suffix_help(ALLOWED_CCTA_SUFFIXES)}",
        key="ccta_upload",
    )

    left, right = st.columns(2)

    max_windows = left.number_input(
        "Sliding-window budget",
        min_value=1,
        max_value=4000,
        value=CCTA_MAX_WINDOWS,
        step=8,
        help=(
            "Full coverage of a 1 mm whole-chest volume needs several hundred "
            "windows and minutes of CPU. A short budget analyses a centred crop "
            "and the result says so — it is not a complete pass."
        ),
    )

    include_gradcam = right.checkbox("Compute 3-D Grad-CAM", value=True)
    include_figures = right.checkbox("Render figures", value=True)

    if upload is not None and st.button("Analyse volume", type="primary"):
        mark_provided("ccta")

        payload = run_analysis(
            "ccta",
            "Resampling and running the sliding window…",
            data=upload.getvalue(),
            filename=upload.name,
            max_windows=int(max_windows),
            include_gradcam=include_gradcam,
            include_figures=include_figures,
        )

        if payload:
            show_ccta_result(payload)

    elif case_state().get("ccta"):
        st.markdown("#### Current result for this session")
        show_ccta_result(case_state()["ccta"])


def section_echo() -> None:
    st.title("Echocardiography")
    st.caption(MODALITY_STATUS["echo"]["task"])

    st.info(
        "The model was trained on images in this orientation: "
        f"{ECHO_TRAINING_ORIENTATION}. A conventional apex-up display needs a "
        "quarter turn before inference, so the rotation below is part of the "
        "analysis, not a viewing preference."
    )

    upload = st.file_uploader(
        "Echo image or DICOM",
        type=upload_types(ALLOWED_ECHO_SUFFIXES),
        help=f"Accepted: {suffix_help(ALLOWED_ECHO_SUFFIXES)}",
        key="echo_upload",
    )

    left, middle, right = st.columns(3)

    rotate = left.selectbox("Rotation before inference (°)", (0, 90, 180, 270))
    flip = middle.checkbox("Mirror horizontally")
    frame = right.number_input(
        "Frame (multi-frame DICOM)",
        min_value=0,
        value=0,
        step=1,
        help="Ignored for single-frame input.",
    )

    if upload is not None and st.button("Segment frame", type="primary"):
        mark_provided("echo")

        payload = run_analysis(
            "echo",
            "Segmenting…",
            data=upload.getvalue(),
            filename=upload.name,
            frame=int(frame) or None,
            rotate=int(rotate),
            flip=flip,
            # The mask is a 65k-element array used for client-side rendering;
            # the figures here are already rendered server-side.
            include_mask=False,
        )

        if payload:
            show_echo_result(payload)

    elif case_state().get("echo"):
        st.markdown("#### Current result for this session")
        show_echo_result(case_state()["echo"])


def section_ecg() -> None:
    st.title("Electrocardiography")
    st.caption(MODALITY_STATUS["ecg"]["task"])

    st.info(
        "WFDB splits one recording across a header and a signal file, and "
        "neither is readable alone. Upload the .hea below and add the .dat or "
        ".mat as a companion, or upload a .zip containing both."
    )

    upload = st.file_uploader(
        "Recording (or WFDB header)",
        type=upload_types(ALLOWED_ECG_SUFFIXES),
        help=f"Accepted: {suffix_help(ALLOWED_ECG_SUFFIXES)}",
        key="ecg_upload",
    )

    extras = st.file_uploader(
        "Companion files belonging to the same recording",
        type=upload_types(ALLOWED_ECG_SUFFIXES),
        accept_multiple_files=True,
        key="ecg_companions",
    )

    left, right = st.columns(2)

    frequency = left.number_input(
        "Source sampling rate (Hz)",
        min_value=0.0,
        max_value=10000.0,
        value=0.0,
        step=50.0,
        help=(
            "Only for formats that do not record it (CSV, NPY). Leave at 0 to "
            "use whatever the file declares — getting this wrong rescales the "
            "whole recording in time."
        ),
    )

    target_class = right.selectbox(
        "Explain which class",
        ("highest probability",) + tuple(ECG_CLASS_NAMES),
        help="Which class the lead attribution is computed for.",
    )

    if upload is not None and st.button("Classify recording", type="primary"):
        mark_provided("ecg")

        payload = run_analysis(
            "ecg",
            "Filtering, resampling and classifying…",
            data=upload.getvalue(),
            filename=upload.name,
            companions={
                extra.name: extra.getvalue() for extra in extras or []
            },
            sampling_frequency=float(frequency) or None,
            target_class=(
                None if target_class == "highest probability" else target_class
            ),
        )

        if payload:
            show_ecg_result(payload)

    elif case_state().get("ecg"):
        st.markdown("#### Current result for this session")
        show_ecg_result(case_state()["ecg"])


def _medgemma_notice(reason: Optional[str]) -> None:
    """
    Say plainly that the language model is absent, and what still works.

    Two things this must not do: imply the analysis is degraded (it is not — no
    number in this application comes from a language model), and hide the reason
    behind a generic failure. The weights are the one dependency the repository
    cannot ship.
    """
    st.warning(
        f"The clinical language model is not available — {reason}. "
        "No measurement depends on it: every number comes from one of the three "
        "imaging models, and the integrated report below is deterministic and "
        "complete without a narrative."
    )

    with st.expander("Enabling narrative text on a local machine"):
        st.markdown(
            "MedGemma is roughly 8.6 GB of vendor weights under Google's Health "
            "AI Developer Foundations terms, so it is not redistributed with "
            "this repository and is not present in a hosted deployment built "
            "from it. To enable it locally, accept the terms and download the "
            "weights into the path the configuration already expects:"
        )
        st.code(
            "huggingface-cli download google/medgemma-1.5-4b-it \\\n"
            f"  --local-dir {MEDGEMMA_PATH.name}",
            language="bash",
        )
        st.caption(
            f"Expected under models/{MEDGEMMA_PATH.name}/ — resolved from "
            "config.py, so no path needs editing. Restart the app afterwards."
        )


def section_assistant() -> None:
    st.title("AI Assistant")
    st.caption(
        "Answers are written by the language model from the case context below. "
        "They are explanation, not measurement — every number in the context "
        "came from one of the three imaging models."
    )

    narrative_ready, narrative_reason = medgemma_state()

    if not narrative_ready:
        _medgemma_notice(narrative_reason)

    case = case_state()
    context = build_case_context(case)

    if context:
        with st.expander("The context sent with each question"):
            st.text(context)
        st.caption(
            "Patient name and medical record number are withheld from every "
            "prompt. Age, sex, study date and the clinical form are included."
        )
    else:
        st.info(
            "No analysis is attached to this session yet, so questions are "
            "answered from general medical knowledge with no case context. "
            "Run a sample case or upload a study first."
        )

    if narrative_ready:
        history: list[dict[str, str]] = st.session_state.setdefault("chat", [])

        for message in history:
            with st.chat_message(message["role"]):
                st.markdown(message["text"])

        question = st.chat_input("Ask about this case, or about the models")

        if question:
            history.append({"role": "user", "text": question})

            with st.chat_message("user"):
                st.markdown(question)

            with st.spinner("Loading the language model…"):
                model, error = load_model("medgemma")

            with st.chat_message("assistant"):
                if model is None:
                    answer = error or "The language model is not available."
                    st.error(answer)
                else:
                    try:
                        with st.spinner("Generating…"):
                            answer = model.generate(
                                question=question, context=context
                            )
                        st.markdown(answer)
                    except Exception as failure:
                        answer = f"Generation failed: {failure}"
                        st.error(answer)

            history.append({"role": "assistant", "text": answer})

    _report_panel(case, narrative_ready)


def _report_panel(case: dict[str, Any], narrative_ready: bool) -> None:
    """
    The structured report: deterministic evidence, with an optional narrative.

    The narrative is optional in the schema for a reason — every finding comes
    from a modality model, so the report stands complete without it and records
    why it is missing. When the weights are absent the checkbox is not offered at
    all, and ``ai_summary_error`` carries the reason into the report itself.
    """
    st.divider()
    st.markdown("### Integrated report")

    if narrative_ready:
        want_summary = st.checkbox(
            "Include the language-model narrative", value=True
        )
    else:
        want_summary = False
        st.caption(
            "The structured report is built without a narrative here. Every "
            "status, finding and uncertainty in it is computed deterministically "
            "from the analyses in this session."
        )

    if not st.button("Build report"):
        if st.session_state.get("report"):
            _show_report(st.session_state["report"])
        return

    evidence = build_case_evidence(case)
    prompt = build_report_prompt(evidence)

    summary: Optional[str] = None
    ai_error: Optional[str] = None

    if want_summary:
        with st.spinner("Loading the language model…"):
            model, error = load_model("medgemma")

        if model is None:
            ai_error = error or "The language model is not available."
        else:
            try:
                with st.spinner("Writing the narrative…"):
                    summary = model.generate(
                        question=prompt["question"], context=prompt["context"]
                    )
            except Exception as failure:
                ai_error = str(failure)
    elif narrative_ready:
        ai_error = "The narrative was not requested."
    else:
        # The report records why, rather than leaving a reader to wonder whether
        # generation was skipped or failed.
        ai_error = (
            "The clinical language model is not available in this deployment "
            f"({medgemma_state()[1]}), so no narrative was generated."
        )

    report = build_report(
        evidence=evidence,
        patient=case.get("patient"),
        ai_summary=summary,
        ai_error=ai_error,
    )

    st.session_state["report"] = report
    st.session_state["report_prompt"] = prompt["context"]

    _show_report(report)


def _show_report(report: dict[str, Any]) -> None:
    st.markdown("#### Per-modality status")

    st.dataframe(
        [
            {
                "Modality": item.get("label"),
                "Status": item.get("status"),
                "Status meaning": item.get("status_meaning", ""),
            }
            for item in (report.get("modality_results") or {}).values()
        ],
        hide_index=True,
    )

    integrated = report.get("integrated_evidence") or {}

    st.caption(integrated.get("integration_method", ""))

    for item in integrated.get("cross_modal_evidence") or []:
        st.markdown(
            f"- **{', '.join(item.get('modalities') or [])}** — "
            f"{item.get('statement')} (_inference: "
            f"{item.get('inference')}_)"
        )

    if report.get("ai_summary"):
        st.markdown("#### Narrative")
        st.markdown(report["ai_summary"])
        st.caption(report.get("ai_summary_scope", ""))
    else:
        st.info(
            "The narrative is not part of this report. "
            + str(report.get("ai_summary_error") or "")
        )

    st.markdown("#### Uncertainties")
    for item in report.get("uncertainties") or []:
        st.markdown(
            f"- **{item.get('scope')} / {item.get('kind')}** — "
            f"{item.get('detail')}"
        )

    st.markdown("#### Next steps")
    for item in report.get("recommendations") or []:
        st.markdown(f"- **{item.get('action')}** — {item.get('reason')}")
    st.caption(report.get("recommendations_scope", ""))

    st.warning(report.get("disclaimer", DISCLAIMER))

    with st.expander("The prompt the narrative was written from"):
        st.text(st.session_state.get("report_prompt", ""))

    with st.expander("The report as JSON"):
        st.json(report)

    st.download_button(
        "Download report (JSON)",
        data=json.dumps(report, indent=2),
        file_name=f"{report.get('case_id') or 'session'}-report.json",
        mime="application/json",
    )
    st.caption(
        "Downloading is the only way out of this interface — nothing is written "
        "to disk on the server."
    )


def section_samples() -> None:
    st.title("Sample cases")
    st.caption(
        "Real recordings from the public datasets, tracked in samples/. Each one "
        "runs through the same pipeline as an uploaded file — nothing here is a "
        "precomputed result."
    )

    samples = discover_samples()

    if not any(samples.values()):
        st.info(
            "No sample cases are present in this checkout. They are tracked "
            "through Git LFS; `git lfs pull` fetches them."
        )
        return

    ccta_tab, echo_tab, ecg_tab = st.tabs(
        ["CCTA", "Echocardiography", "ECG"]
    )

    with ccta_tab:
        _sample_ccta(samples["ccta"])

    with echo_tab:
        _sample_echo(samples["echo"])

    with ecg_tab:
        _sample_ecg(samples["ecg"])

    st.divider()
    st.markdown("#### Attribution")
    st.markdown(
        "- **CAMUS** (echo) — University of Lyon, CREATIS. Research use; cite "
        "the dataset paper if you publish results.\n"
        "- **MedHK23/CCA** (CCTA) — check the source's terms before "
        "redistributing anything derived from it.\n"
        "- **PTB-XL** (ECG) — PhysioNet, Open Data Commons Attribution "
        "Licence v1.0. Attribution is required."
    )
    st.caption(
        "Two cases per modality is an illustration, not an evaluation set. See "
        "samples/README.md for which split each case came from."
    )


def _discuss_hint() -> None:
    st.info(
        "The result is attached to this session — open **AI Assistant** to "
        "discuss it with the language model, or to build the integrated report."
    )


def _sample_ccta(cases: list[dict[str, Any]]) -> None:
    if not cases:
        st.info("No CCTA sample case is currently available.")
        return

    labels = [f"{item['label']} ({item['size_mb']} MB)" for item in cases]
    chosen = cases[labels.index(st.radio("Case", labels, key="sample_ccta"))]

    st.caption(f"Dataset: {chosen['dataset']}")
    st.warning(
        "A whole-chest volume at 1 mm needs several hundred sliding windows. "
        "The default budget analyses a centred crop in a minute or two; raise "
        "it for full coverage and expect minutes of CPU."
    )

    budget = st.number_input(
        "Sliding-window budget",
        min_value=1,
        max_value=4000,
        value=CCTA_MAX_WINDOWS,
        step=8,
        key="sample_ccta_budget",
    )

    if st.button("Analyse this case", type="primary", key="sample_ccta_run"):
        mark_provided("ccta")

        payload = run_analysis(
            "ccta",
            "Resampling and running the sliding window…",
            data=read_sample(chosen["path"]),
            filename=Path(chosen["path"]).name,
            max_windows=int(budget),
            include_gradcam=True,
            include_figures=True,
        )

        if payload:
            show_ccta_result(payload)
            _discuss_hint()

    if chosen.get("ground_truth"):
        st.markdown("#### Reference annotation")
        st.caption(
            "The dataset's own expert label map for this case. It is a "
            "reference, not a model output, and it is never sent to the model. "
            "It is a 3-D volume, so it is not drawn as a single image here."
        )


def _sample_echo(cases: list[dict[str, Any]]) -> None:
    if not cases:
        st.info("No echocardiography sample case is currently available.")
        return

    labels = [item["label"] for item in cases]
    patient = cases[labels.index(st.radio("Patient", labels, key="sample_echo"))]

    st.caption(f"Dataset: {patient['dataset']}")

    frame_ids = [frame["id"] for frame in patient["frames"]]
    frame = patient["frames"][
        frame_ids.index(st.selectbox("View", frame_ids, key="sample_echo_frame"))
    ]

    st.caption(
        "CAMUS frames are stored in the orientation the model was trained on "
        f"({ECHO_TRAINING_ORIENTATION}), so no rotation is applied."
    )

    if st.button("Segment this frame", type="primary", key="sample_echo_run"):
        mark_provided("echo")

        payload = run_analysis(
            "echo",
            "Segmenting…",
            data=read_sample(frame["path"]),
            filename=Path(frame["path"]).name,
            rotate=0,
            flip=False,
            include_mask=False,
        )

        if payload:
            show_echo_result(payload)

            if frame.get("ground_truth"):
                st.markdown("#### Reference annotation")
                show_ground_truth(
                    frame["ground_truth"],
                    "Dataset expert annotation — reference only, not a "
                    "model output",
                )

            _discuss_hint()

    if patient.get("citation"):
        with st.expander("Mandatory dataset citation"):
            st.text(Path(patient["citation"]).read_text(encoding="utf-8"))


def _sample_ecg(cases: list[dict[str, Any]]) -> None:
    if not cases:
        st.info("No ECG sample case is currently available.")
        return

    labels = [item["label"] for item in cases]
    record = cases[labels.index(st.radio("Record", labels, key="sample_ecg"))]

    st.caption(f"Dataset: {record['dataset']}")
    st.caption(
        "WFDB: the header and the signal file are read together, exactly as an "
        "uploaded pair would be."
    )

    if st.button("Classify this record", type="primary", key="sample_ecg_run"):
        mark_provided("ecg")

        payload = run_analysis(
            "ecg",
            "Filtering, resampling and classifying…",
            data=read_sample(record["path"]),
            filename=Path(record["path"]).name,
            companions={
                Path(path).name: read_sample(path)
                for path in record["companions"]
            },
        )

        if payload:
            show_ecg_result(payload)
            _discuss_hint()


def section_about() -> None:
    st.title("About")

    st.markdown(
        f"**{APP_NAME}** {APP_VERSION} — a research prototype for "
        "cardiovascular image analysis with three independently trained models "
        "and a deterministic evidence layer over their outputs."
    )

    st.markdown("### What is implemented")
    st.markdown(
        "- **Coronary CT angiography** — 3-D U-Net lumen segmentation with "
        "3-D Grad-CAM. Lumen mask only.\n"
        "- **Echocardiography** — UNet++ / EfficientNet-B3 segmentation of LV "
        "cavity, myocardium and left atrium on a single frame.\n"
        "- **Electrocardiography** — 1-D residual CNN over the five PTB-XL "
        "diagnostic superclasses, with per-lead input-gradient attribution.\n"
        "- **Report layer** — deterministic aggregation of whatever was "
        "analysed, with an optional language-model narrative."
    )

    st.markdown("### What is not")
    st.markdown(
        "- no clinical risk model and no learned multimodal fusion — both are "
        "reported as unavailable rather than approximated\n"
        "- no regulatory clearance, no prospective evaluation, no reader study\n"
        "- no diagnosis, and no per-case confidence figure\n"
        "- no sign-in, no patient records and no case store in this interface — "
        "those belong to the authenticated React client, which is unchanged\n"
        "- no language model unless its weights were downloaded separately; the "
        "structured report does not depend on one"
    )

    st.divider()
    st.markdown("### Developer")
    st.markdown(
        "**Md. Mehedi Hasan**  \n"
        "Software Developer & AI Engineer  \n"
        "Department of Computer Science and Engineering, Bangladesh University "
        "of Business and Technology (BUBT), Dhaka, Bangladesh"
    )
    st.markdown(
        "Research and technical areas: Machine Learning, Deep Learning, "
        "Explainable AI, Medical Imaging, Medical AI, Bioinformatics, Graph "
        "Neural Networks"
    )
    st.markdown(
        "- GitHub: https://github.com/0mehedihasan\n"
        "- Project repository: https://github.com/0mehedihasan/CardioVision-AI"
    )

    st.warning(DISCLAIMER)


# ============================================================
# ENTRY POINT
# ============================================================


_SECTIONS = {
    "Dashboard": section_dashboard,
    "CCTA": section_ccta,
    "Echocardiography": section_echo,
    "ECG": section_ecg,
    "AI Assistant": section_assistant,
    "Sample Cases": section_samples,
    "About / Developer": section_about,
}


def sidebar() -> str:
    with st.sidebar:
        st.markdown(f"### {APP_NAME}")
        st.caption(f"Version {APP_VERSION}")

        chosen = st.radio("Section", SECTIONS, label_visibility="collapsed")

        st.divider()

        case = case_state()

        st.markdown("**This session**")

        for key, label in (
            ("ccta", "CCTA"),
            ("echo", "Echo"),
            ("ecg", "ECG"),
        ):
            if case.get(key):
                st.caption(f"{label}: analysed")
            elif case["modalities_provided"].get(key):
                st.caption(f"{label}: file provided, not analysed")
            else:
                st.caption(f"{label}: no input")

        st.caption("Not saved anywhere — this session only.")

        if st.button("Clear session"):
            # Only the working case and the transcript. The cached models stay
            # loaded — clearing them would mean reloading gigabytes to start a
            # second case.
            for key in ("case", "chat", "report", "report_prompt"):
                st.session_state.pop(key, None)
            st.rerun()

        st.divider()
        st.caption("Research use only. Not a medical device.")

    return str(chosen)


def main() -> None:
    _SECTIONS[sidebar()]()


main()
