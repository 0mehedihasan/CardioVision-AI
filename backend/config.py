"""
CardioVision AI — central configuration.

Paths, device selection, and the *real* model metadata for the trained
echocardiography model.
"""

from pathlib import Path

import torch

# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODELS_DIR = PROJECT_ROOT / "models"

MEDGEMMA_PATH = MODELS_DIR / "medgemma-1.5-4b-it"

ECHO_CHECKPOINT_PATH = (
    MODELS_DIR / "echo" / "cardiovision_echo_unetplusplus_best.pth"
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
# See backend/auth.py for what this does and does not protect against.

AUTH_DEFAULT_USERNAME = "medexpert"
AUTH_DEFAULT_PASSWORD = "1111"

# Eight hours: long enough for a working day, short enough that a forgotten
# session does not stay open indefinitely. Expiry slides on each request.
AUTH_SESSION_TTL_SECONDS = 8 * 60 * 60


# ============================================================
# DEVICE
# ============================================================

def select_device() -> str:
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
        "available": False,
        "label": "Coronary CT angiography",
        "model": None,
        "task": None,
        "note": "Model not yet trained. notebooks/01_CCTA_Training.ipynb is empty.",
    },
    "ecg": {
        "available": False,
        "label": "Electrocardiography",
        "model": None,
        "task": None,
        "note": "No ECG pipeline exists yet.",
    },
    "clinical": {
        "available": False,
        "label": "Clinical risk model",
        "model": None,
        "task": None,
        "note": "Model not yet trained. notebooks/03_Clinical_Model.ipynb is empty. "
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

ALLOWED_ECHO_SUFFIXES = {
    ".png", ".jpg", ".jpeg",
    ".nii", ".nii.gz", ".gz",
    ".dcm", ".dicom",
}
