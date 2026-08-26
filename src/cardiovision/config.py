"""
CardioVision AI — central configuration.

Paths, device selection, and the *real* model metadata for the three trained
models: echocardiography segmentation, 12-lead ECG classification, and coronary
CT angiography lumen segmentation.

Nothing in here is a guess. Every metric is copied from the executed output of
the notebook that produced the weights, with the notebook named alongside it,
and anything the checkpoint can tell us about itself is read from the
checkpoint instead of being written down twice.
"""

import os
from pathlib import Path

from cardiovision import __version__

# ============================================================
# IDENTITY
# ============================================================
#
# One name and one version for the whole application. The FastAPI title, the
# CLI banner and the generated_by block of every report read from here, so a
# version bump in ``cardiovision/__init__.py`` reaches all of them.

APP_NAME = "CardioVision AI"
APP_VERSION = __version__

# ============================================================
# PATHS
# ============================================================


def _find_project_root() -> Path:
    """
    Locate the repository root, which is where ``models/`` and ``data/`` live.

    Walking up from this file used to be enough, but the package now sits under
    ``src/``, and it can be pip-installed somewhere else entirely. So: honour an
    explicit override first, then look upward for the marker files that only the
    repo root has, and only fall back to arithmetic on parents if neither works.
    """
    override = os.environ.get("CARDIOVISION_HOME", "").strip()
    if override:
        return Path(override).expanduser().resolve()

    here = Path(__file__).resolve()

    for candidate in here.parents:
        if (candidate / "pyproject.toml").is_file() or (candidate / "models").is_dir():
            return candidate

    # Installed outside a checkout with no override. src/cardiovision/config.py
    # -> two levels up is the best available guess; set CARDIOVISION_HOME if it
    # is wrong, because a wrong root shows up as "checkpoint not found" rather
    # than as anything mysterious.
    return here.parents[2]


PROJECT_ROOT = _find_project_root()

MODELS_DIR = PROJECT_ROOT / "models"

MEDGEMMA_PATH = MODELS_DIR / "medgemma-1.5-4b-it"

ECHO_CHECKPOINT_PATH = (
    MODELS_DIR / "echo" / "cardiovision_echo_unetplusplus_best.pth"
)

ECG_CHECKPOINT_PATH = (
    MODELS_DIR / "ecg" / "cardioVision_ptbxl_ecg_resnet1d_full.pt"
)

CCTA_CHECKPOINT_PATH = (
    MODELS_DIR / "ccta" / "best_3d_unet_cca_v2.pth"
)

# ---- case storage ------------------------------------------------
#
# Saved cases live entirely on this machine. The database holds patient
# details and findings; the rendered PNGs and original uploads are written
# alongside it as files, because a few megabytes of base64 per case would
# bloat every row and slow down the case list.

DATA_DIR = PROJECT_ROOT / "data"

CASE_DB_PATH = DATA_DIR / "cardiovision.db"

CASE_FILES_DIR = DATA_DIR / "cases"


# ============================================================
# AUTHENTICATION
# ============================================================
#
# One fixed operator account. Override both values with the environment
# variables CARDIOVISION_USER and CARDIOVISION_PASSWORD; the defaults exist
# so the app is usable out of the box, not because they are secure.
#
# See cardiovision/services/auth.py for what this does and does not protect
# against.

AUTH_DEFAULT_USERNAME = "medexpert"
AUTH_DEFAULT_PASSWORD = "1111"

# Eight hours: long enough for a working day, short enough that a forgotten
# session does not stay open indefinitely. Expiry slides on each request.
AUTH_SESSION_TTL_SECONDS = 8 * 60 * 60


# ============================================================
# DEVICE
# ============================================================
#
# torch is imported here rather than at module scope on purpose. This module is
# the single source of truth for every constant in the project — class names,
# thresholds, published metrics, file paths — and it is read by the renderers,
# the case store, the context builder and the tests, none of which touch a
# tensor. A top-level `import torch` would make all of them depend on the ML
# stack being installed, and would take the whole API down at import time on a
# machine where torch is missing, instead of at the point where a model is
# actually needed.

TORCH_AVAILABLE: bool = False


def select_device() -> str:
    """
    The best available device, or "cpu" when torch is not installed.

    A missing torch is not reported as an error here. The inference modules
    import torch directly and raise on load with a message about the model they
    were trying to open, which is far more useful than an ImportError raised
    while reading configuration.
    """
    global TORCH_AVAILABLE

    try:
        import torch
    except ImportError:
        TORCH_AVAILABLE = False
        return "cpu"

    TORCH_AVAILABLE = True

    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


DEVICE = select_device()


# ============================================================
# MEDGEMMA
# ============================================================

MEDGEMMA_NAME = "MedGemma 1.5 4B IT"

# 32 is too short for useful clinical answers.
MAX_NEW_TOKENS = 256


# ============================================================
# ECHO SEGMENTATION MODEL
# ============================================================
#
# These must match notebooks/02_Echo_Training.ipynb exactly, otherwise
# the checkpoint will not load or the predictions will be wrong.

ECHO_ARCHITECTURE = "UnetPlusPlus"
ECHO_ENCODER = "timm-efficientnet-b3"
ECHO_IMAGE_SIZE = 256
ECHO_NUM_CLASSES = 4
ECHO_IN_CHANNELS = 1

# Class index -> anatomical structure. Index 0 is background.
ECHO_CLASS_NAMES = {
    0: "Background",
    1: "LV cavity",
    2: "Myocardium",
    3: "Left atrium",
}

# Foreground classes only (what we actually report on).
ECHO_FOREGROUND_CLASSES = (1, 2, 3)

# Display colours (R, G, B) used for mask overlays. Kept in sync with
# the frontend so server-rendered PNGs and the client-side canvas agree.
ECHO_CLASS_COLORS = {
    0: (0, 0, 0),
    1: (239, 68, 68),      # LV cavity      — red
    2: (34, 197, 94),      # Myocardium     — green
    3: (59, 130, 246),     # Left atrium    — blue
}

# The class explained by the gradient-saliency map, matching the
# TARGET_CLASS = 1 choice in the training notebook.
ECHO_SALIENCY_CLASS = 1

# A structure is only reported as present once this many mask pixels carry
# its label. Below the threshold a few scattered pixels are argmax noise,
# not anatomy. Surfaced through the API so "Not identified" is never read
# as a clinical absence when it is really a size cutoff.
ECHO_PRESENCE_THRESHOLD_PX = 50

# How the training images were oriented. The CAMUS NIfTI arrays were fed to
# the model as stored, with no reorientation, which puts the ultrasound
# sector's apex on the LEFT and opens the beam to the right. Verifiable in
# the notebook's own saved figure, models/echo/__results___0_129.png.
# Conventional echo displays are apex-up, i.e. a quarter turn away.
ECHO_TRAINING_ORIENTATION = "sector apex left, beam opening right"


# ============================================================
# HELD-OUT TEST METRICS
# ============================================================
#
# SOURCE OF TRUTH: the executed output of
# notebooks/02_Echo_Training.ipynb (single cell, execution_count 1),
# section "FINAL TEST EVALUATION" / "PER-CLASS TEST METRICS".
#
# Training ran on Kaggle (Tesla T4). Early stopping fired at epoch 38;
# the best checkpoint is epoch 30.
#
# Evaluated on a patient-level held-out test split:
#   500 patients total -> 350 train / 75 validation / 75 test
#   (2000 image-mask pairs -> 1400 / 300 / 300)
# Patient disjointness was asserted in the notebook, so these numbers
# contain no patient leakage.
#
# These are dataset-level metrics describing the model, NOT a
# confidence score for any individual prediction. The frontend must
# label them as such.
#
# val_dice / val_iou are NOT hardcoded here — they are read back out of
# the checkpoint at load time so they can never drift from the weights.

ECHO_TEST_METRICS = {
    "test_dice": 0.9044,
    "test_iou": 0.8282,
    "per_class_dice": {
        "LV cavity": 0.9379,
        "Myocardium": 0.8759,
        "Left atrium": 0.8994,
    },
    "dataset": "CAMUS",
    "test_patients": 75,
    "test_pairs": 300,
    "trained_on": "Tesla T4 (Kaggle)",
    "source": "notebooks/02_Echo_Training.ipynb",
}


# ============================================================
# ECG CLASSIFICATION MODEL
# ============================================================
#
# These must match notebooks/03_ECG.ipynb exactly, otherwise the checkpoint
# will not load or the predictions will be wrong.

ECG_ARCHITECTURE = "ECGResNet1D"
ECG_NUM_CLASSES = 5
ECG_IN_CHANNELS = 12
ECG_INPUT_LENGTH = 1000            # samples
ECG_DROPOUT = 0.30
ECG_PARAMETERS = 3_884_165

# PTB-XL diagnostic superclasses, in the checkpoint's own column order.
# Reordering this list silently reassigns every prediction, so it is read back
# out of the checkpoint at load time and cross-checked against this.
ECG_CLASS_NAMES = ("NORM", "MI", "STTC", "CD", "HYP")

ECG_CLASS_LABELS = {
    "NORM": "Normal ECG",
    "MI": "Myocardial infarction",
    "STTC": "ST/T change",
    "CD": "Conduction disturbance",
    "HYP": "Hypertrophy",
}

ECG_CLASS_DESCRIPTIONS = {
    "NORM": "No abnormality in the diagnostic superclasses below.",
    "MI": "Pattern consistent with prior or acute myocardial infarction.",
    "STTC": "ST-segment or T-wave abnormality.",
    "CD": "Conduction disturbance, e.g. bundle branch or AV block.",
    "HYP": "Ventricular or atrial hypertrophy pattern.",
}

# Standard 12-lead order. Input in any other order is not reordered for you:
# the loader reports the order it found so a mismatch is visible.
ECG_LEAD_NAMES = (
    "I", "II", "III", "aVR", "aVL", "aVF",
    "V1", "V2", "V3", "V4", "V5", "V6",
)

# ---- preprocessing (models/ecg/preprocessing_config.json) ----------
ECG_TARGET_FS = 100               # Hz, after resampling
ECG_TRAINING_SOURCE_FS = 500      # Hz, PTB-XL's high-resolution records
ECG_DURATION_SECONDS = 10
ECG_BANDPASS_LOW_HZ = 0.5
ECG_BANDPASS_HIGH_HZ = 40.0
ECG_BANDPASS_ORDER = 4
ECG_CLIP_RANGE = (-10.0, 10.0)
ECG_NORMALIZATION = "per-lead robust median/IQR"

# Multi-label sigmoid threshold. The training run reported F1, precision and
# recall at this operating point, so moving it invalidates those numbers —
# which is why the API returns every probability rather than only the calls.
ECG_THRESHOLD = 0.5


# ============================================================
# ECG HELD-OUT TEST METRICS
# ============================================================
#
# SOURCE OF TRUTH: notebooks/03_ECG.ipynb, mirrored in
# models/ecg/test_metrics.json and inside the checkpoint itself. The loader
# prefers the checkpoint copy; this dict is the fallback and the documentation.
#
# PTB-XL, split at PATIENT level:
#   21837 records discovered -> 14957 train / 3199 validation / 3232 test
#   13031 / 2793 / 2793 patients, disjointness asserted in the notebook.
# Multi-label BCEWithLogitsLoss with pos_weight; best epoch 14 by validation
# macro AUROC (0.9203).
#
# Dataset-level metrics describing the model, NOT a confidence score for any
# individual ECG. The frontend must label them as such.
#
# READ THE HYP ROW. Macro AUROC 0.913 looks reassuring and is not the whole
# story: hypertrophy detection sits at AP 0.478 with precision 0.361, so
# roughly two in three positive HYP calls are wrong. That single number is the
# reason ECG_WEAK_CLASSES exists and is surfaced in the UI and in the MedGemma
# prompt instead of being averaged away.

ECG_TEST_METRICS = {
    "macro_AUROC": 0.9125,
    "macro_AP": 0.7804,
    "macro_F1": 0.7086,
    "macro_Precision": 0.6353,
    "macro_Recall": 0.8095,
    "per_class": {
        "NORM": {"AUROC": 0.9498, "AP": 0.9211, "F1": 0.8585,
                 "Precision": 0.8009, "Recall": 0.9251},
        "MI":   {"AUROC": 0.9173, "AP": 0.8294, "F1": 0.7357,
                 "Precision": 0.6865, "Recall": 0.7925},
        "STTC": {"AUROC": 0.9303, "AP": 0.8126, "F1": 0.7305,
                 "Precision": 0.6166, "Recall": 0.8959},
        "CD":   {"AUROC": 0.9329, "AP": 0.8612, "F1": 0.7668,
                 "Precision": 0.7113, "Recall": 0.8318},
        "HYP":  {"AUROC": 0.8323, "AP": 0.4777, "F1": 0.4516,
                 "Precision": 0.3614, "Recall": 0.6020},
    },
    "test_prevalence": {
        "NORM": 0.4171, "MI": 0.2565, "STTC": 0.2438,
        "CD": 0.2373, "HYP": 0.1259,
    },
    "dataset": "PTB-XL",
    "test_records": 3232,
    "test_patients": 2793,
    "best_epoch": 14,
    "best_validation_macro_AUROC": 0.9203,
    "threshold": ECG_THRESHOLD,
    "source": "notebooks/03_ECG.ipynb",
}

# Classes whose precision is too low to act on a positive call. Anything listed
# here is flagged in the response, in the UI and in the language-model prompt.
ECG_WEAK_CLASSES = {
    "HYP": (
        "Hypertrophy is the weakest of the five classes by a wide margin: "
        "average precision 0.478 and precision 0.361 at the 0.5 threshold, "
        "against 0.83-0.92 average precision for the others. Roughly two in "
        "three positive hypertrophy calls from this model are false. Treat a "
        "positive HYP as a prompt to look at the tracing, never as a finding."
    ),
}

# Lead attribution for ONE example record, from models/ecg/lead_importance.csv.
#
# Read the notebook before reusing this: it is `saliency.mean(axis=1)` for test
# record HR00025 alone, not an average over the test set. The filename invites
# the opposite reading, which is exactly why it is named and commented this way
# here — it is provenance for the shipped example figure, and must never be
# rendered as "which leads this model relies on". The per-recording ranking the
# UI shows is computed at request time from the uploaded ECG.
ECG_EXAMPLE_LEAD_IMPORTANCE = {
    "record": "HR00025",
    "leads": {
        "I": 0.2450, "V4": 0.2387, "aVL": 0.2358, "III": 0.2266,
        "V1": 0.2115, "V6": 0.2108, "V2": 0.2106, "II": 0.2068,
        "aVF": 0.2065, "aVR": 0.2056, "V5": 0.1867, "V3": 0.1619,
    },
}


# ============================================================
# CCTA SEGMENTATION MODEL
# ============================================================
#
# These must match notebooks/01_CCTA_Training.ipynb exactly, otherwise the
# checkpoint will not load or the predictions will be wrong. Every value below
# was verified against the 50 tensors inside
# models/ccta/best_3d_unet_cca_v2.pth, not read off the prose in the notebook.

CCTA_ARCHITECTURE = "Small3DUNet"
CCTA_BASE_CHANNELS = 16
CCTA_IN_CHANNELS = 1
CCTA_OUT_CHANNELS = 1               # one logit per voxel, sigmoid, not softmax
CCTA_PARAMETERS = 1_401_265

# What the single output channel means. This is the whole vocabulary of the
# model: one foreground class, no vessel identity, no lesion semantics.
CCTA_CLASS_NAMES = {
    0: "Background",
    1: "Coronary artery lumen",
}

CCTA_CLASS_COLORS = {
    0: (0, 0, 0),
    1: (250, 204, 21),      # lumen — amber, distinct from every echo colour
}

# ---- preprocessing -------------------------------------------------
#
# The notebook resamples every volume to 1 mm isotropic before patching, then
# windows Hounsfield units and rescales to [-1, 1]. Air lands on exactly -1.0,
# which is also the pad value used by the sliding window — so padding is
# indistinguishable from air and introduces no edge artefact.

CCTA_TARGET_SPACING = (1.0, 1.0, 1.0)      # mm
CCTA_HU_MIN = -1000.0
CCTA_HU_MAX = 1000.0
CCTA_PAD_VALUE = -1.0
CCTA_NORMALIZATION = "HU clipped to [-1000, 1000], scaled to [-1, 1]"
CCTA_RESAMPLE_ORDER = 1                    # trilinear for the image volume

# ---- inference -----------------------------------------------------

CCTA_PATCH_SIZE = (96, 96, 96)
CCTA_INFERENCE_OVERLAP = 0.50
CCTA_INFERENCE_BATCH_SIZE = 2

# Validation-selected operating point, recorded in the checkpoint as
# `selected_threshold`. The loader reads it from the checkpoint; this is the
# fallback and the documentation. It was chosen on the validation split and
# never on the test split.
CCTA_THRESHOLD = 0.60

# A full study is large. A 0.5 mm 832x832x576 volume becomes 416x416x288 at
# 1 mm, which is roughly 486 sliding windows at 50% overlap — minutes on CPU.
# Rather than let a request hang, the service stops after this many windows and
# reports the fraction of the volume it actually covered. Unanalysed voxels are
# reported as unanalysed, never as background.
CCTA_MAX_WINDOWS = 600

# Grad-CAM target layer, spelled the way the notebook spells it. The last
# activation of the third encoder block: deep enough to be semantic, shallow
# enough to still have usable spatial resolution.
CCTA_GRADCAM_LAYER = "enc3.block[-1]"

# Below this many voxels the mask is reported as "nothing identified" rather
# than as a finding. At 1 mm isotropic a voxel is 1 mm3, so this is half a
# millilitre — far less than any real coronary tree, and about what a handful
# of scattered false positives looks like. Surfaced through the API so an empty
# result is never read as an anatomical absence when it is a size cutoff.
CCTA_PRESENCE_THRESHOLD_VOXELS = 500


# ============================================================
# CCTA HELD-OUT TEST METRICS
# ============================================================
#
# SOURCE OF TRUTH: models/ccta/test_metrics.csv, produced by
# notebooks/01_CCTA_Training.ipynb. Aggregates below are computed from that
# file's three rows, not copied from a printed summary.
#
# Dataset MedHK23/CCA: 20 annotated CCTA volumes, all 832x832x576 at 0.5 mm
# isotropic, split at CASE level into 14 train / 3 validation / 3 test.
# Coronary lumen occupies about 0.11% of voxels (mean foreground ratio
# 0.001110), which is why Dice rather than accuracy is the reported metric.
#
# READ THE TEST SIZE. The held-out split is THREE cases: 9, 14 and 15. Three
# observations support no confidence interval and no claim of generalisation.
# The standard deviation below is the spread of three numbers. Wherever the
# Dice is displayed, n=3 is displayed with it — the same discipline that puts
# HYP's precision next to the ECG macro AUROC.
#
# READ THE HD95 ROW. A 95th-percentile Hausdorff distance of 82-131 mm means
# the predicted surface has outlier components most of a heart-width away from
# the annotation. Overlap is moderate; geometric fidelity is not established.
# This mask locates contrast-filled lumen. It is not a verified coronary tree.

CCTA_TEST_METRICS = {
    "dice": {"mean": 0.5996, "sd": 0.1182, "min": 0.4929, "max": 0.7266},
    "iou": {"mean": 0.4351, "sd": 0.1241, "min": 0.3270, "max": 0.5707},
    "sensitivity": {"mean": 0.6157, "sd": 0.0874, "min": 0.5232, "max": 0.6969},
    "precision": {"mean": 0.5878, "sd": 0.1527, "min": 0.4658, "max": 0.7590},
    "hd95_mm": {"mean": 109.5094, "sd": 25.0774, "min": 82.1985, "max": 131.4994},
    "dataset": "MedHK23/CCA",
    "dataset_cases": 20,
    "split": "14 train / 3 validation / 3 test, split at case level",
    "test_cases": 3,
    "test_case_ids": (9, 14, 15),
    "mean_foreground_ratio": 0.001110,
    "best_epoch": 11,
    "best_validation_dice": 0.6505,
    "threshold": CCTA_THRESHOLD,
    "trained_on": "Tesla T4 (Kaggle)",
    "source": "notebooks/01_CCTA_Training.ipynb",
}

# The equivalent of ECG_WEAK_CLASSES, except the weakness is the whole model
# rather than one class of it. Surfaced in the API response, in the UI and in
# the language-model prompt.
CCTA_WEAK_NOTES = (
    "This is the weakest of the three trained models and it was evaluated on "
    "three cases. Dice 0.60 means roughly two fifths of the annotated lumen "
    "volume is either missed or over-called, and a 95th-percentile Hausdorff "
    "distance of 82-131 mm means some predicted components sit most of a heart-"
    "width away from the annotation. Read the mask as a contrast-density "
    "highlight to review, never as a verified coronary tree.",
    "The model has exactly one foreground class. It does not grade stenosis, "
    "does not compute a calcium score, does not assign a CAD-RADS category and "
    "does not label which vessel a voxel belongs to. Any such statement would "
    "have to come from the reader, not from this model.",
    "Twenty volumes from a single public dataset, all acquired at 0.5 mm "
    "isotropic. Behaviour on a different scanner, a different contrast "
    "protocol or a different slice thickness is unmeasured.",
)

# The Grad-CAM figures and NIfTI masks shipped in models/ccta/ are for test
# cases 9, 14 and 15 of the dataset above. They are provenance for the
# notebook. They are not a patient result and must never be rendered as one —
# the same trap as ECG_EXAMPLE_LEAD_IMPORTANCE.
CCTA_EXAMPLE_ARTIFACTS = {
    "cases": (9, 14, 15),
    "figures": "models/ccta/case_{n}_xai.png",
    "volumes": "models/ccta/case_{n}_gradcam_full_resampled.nii.gz",
    "note": "Dataset test cases from the training notebook, not patient output.",
}


# ============================================================
# MODALITY AVAILABILITY
# ============================================================
#
# Single source of truth for what is actually implemented. The frontend
# reads this from /api/health so the UI can never claim a capability
# the backend does not have.
#
# Flip `available` to True here as each pipeline lands.

MODALITY_STATUS = {
    "echo": {
        "available": True,
        "label": "Echocardiography",
        "model": "UNet++ / EfficientNet-B3",
        "task": "4-class cardiac structure segmentation",
        "note": "Trained on CAMUS, 500 patients.",
    },
    "ccta": {
        "available": True,
        "label": "Coronary CT angiography",
        "model": "Small3DUNet (3-D U-Net, 1.4M parameters)",
        "task": "Binary coronary artery lumen segmentation of a CT volume",
        "note": (
            "Trained on MedHK23/CCA, 20 volumes. Dice 0.60 on a held-out split "
            "of THREE cases, HD95 82-131 mm. Outputs a lumen mask only — no "
            "stenosis grading, no calcium score, no vessel labelling. The "
            "weakest of the three models; see CCTA_WEAK_NOTES."
        ),
    },
    "ecg": {
        "available": True,
        "label": "Electrocardiography",
        "model": "ECGResNet1D (1-D residual CNN)",
        "task": "5-class multi-label diagnostic screening of 12-lead ECG",
        "note": (
            "Trained on PTB-XL, 21837 records. Macro AUROC 0.913 on a "
            "patient-disjoint test split. Hypertrophy is unreliable — see the "
            "per-class metrics."
        ),
    },
    "clinical": {
        "available": False,
        "label": "Clinical risk model",
        "model": None,
        "task": None,
        "note": "Model not yet trained. No clinical-risk notebook exists. "
                "Clinical values are passed to the language model as context only.",
    },
    "fusion": {
        "available": False,
        "label": "Multimodal fusion",
        "model": None,
        "task": None,
        "note": "Model not yet trained. notebooks/04_Multimodal_Fusion.ipynb is empty.",
    },
}


# ============================================================
# UPLOAD LIMITS
# ============================================================

MAX_UPLOAD_BYTES = 200 * 1024 * 1024      # 200 MB (DICOM cine loops are large)

# CCTA is in a different size class. One volume from the training dataset is
# 832x832x576 int16, which is 797 MB raw and still substantial as .nii.gz, so
# the general limit would reject a legitimate study.
MAX_CCTA_UPLOAD_BYTES = 800 * 1024 * 1024      # 800 MB

# Refuse a volume this large before allocating anything, measured AFTER
# resampling to 1 mm. 200 million voxels is 800 MB as float32 and the pipeline
# needs several arrays of that size at once. A CT study that exceeds this is
# almost certainly a whole-body scan rather than a cardiac one.
MAX_CCTA_VOXELS = 200_000_000

ALLOWED_ECHO_SUFFIXES = {
    ".png", ".jpg", ".jpeg",
    ".nii", ".nii.gz", ".gz",
    ".dcm", ".dicom",
}

# WFDB is PTB-XL's native format and needs two files, so .hea and .dat are both
# accepted and the loader pairs them — or a .zip holding both, which is the only
# way to send a WFDB record through a single file input. .npy is what the
# training notebook cached its preprocessed arrays as, which makes it the format
# to use when checking this pipeline against known-good input.
ALLOWED_ECG_SUFFIXES = {
    ".hea", ".dat",
    ".csv", ".txt", ".tsv",
    ".npy",
    ".json",
    ".zip",
}

# CCTA needs a whole 3-D volume, so a single 2-D image is not acceptable input
# here even though the echo pipeline accepts one. NIfTI carries its own voxel
# spacing, which the resampling step requires; a DICOM series carries the same
# information across many files, so it is accepted as a .zip of one series.
ALLOWED_CCTA_SUFFIXES = {
    ".nii", ".nii.gz", ".gz",
    ".zip",
}
