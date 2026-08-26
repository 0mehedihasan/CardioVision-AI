"""
CardioVision AI — 12-lead ECG classification inference.

Rebuilds the ``ECGResNet1D`` architecture from notebooks/03_ECG.ipynb, loads the
trained checkpoint, and runs 5-class multi-label diagnostic screening with
input-gradient saliency.

Design notes
------------
* The architecture is reconstructed here rather than unpickled, so the
  checkpoint file only ever needs to contain tensors. It is verified against the
  parameter count the notebook printed (3,884,165) at load time: if the
  reconstruction were wrong in any layer, that number would move, and
  ``strict=True`` would fail anyway.
* Everything the checkpoint knows about itself — class order, lead order, input
  length, sampling rate, metrics — is read back out and cross-checked against
  :mod:`cardiovision.config`. A class-order mismatch silently reassigns every
  prediction, so it is a hard failure rather than a warning.
* This is a *multi-label* model: the five classes are independent sigmoids, not
  a softmax. A recording can be positive for none of them or for four of them,
  and the API returns every probability rather than only the calls above
  threshold, because the threshold is a choice and the caller is entitled to see
  what it did.
* Lead attribution is computed from raw gradient magnitudes, not from the
  per-lead-normalised saliency used for display. See :meth:`_rank_leads` for
  why those two are not interchangeable.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
import torch
import torch.nn as nn

from cardiovision.config import (
    DEVICE,
    ECG_ARCHITECTURE,
    ECG_CHECKPOINT_PATH,
    ECG_CLASS_DESCRIPTIONS,
    ECG_CLASS_LABELS,
    ECG_CLASS_NAMES,
    ECG_DROPOUT,
    ECG_IN_CHANNELS,
    ECG_INPUT_LENGTH,
    ECG_LEAD_NAMES,
    ECG_NUM_CLASSES,
    ECG_PARAMETERS,
    ECG_TARGET_FS,
    ECG_TEST_METRICS,
    ECG_THRESHOLD,
    ECG_WEAK_CLASSES,
)


class EcgModelUnavailable(RuntimeError):
    """Raised when the ECG model cannot be loaded or used."""


# ============================================================
# ARCHITECTURE
# ============================================================
#
# Verbatim from notebooks/03_ECG.ipynb. Do not "improve" anything here: every
# layer name is a key in the saved state dict, and every shape is baked into the
# weights.


class ConvBlock(nn.Module):
    """Conv -> BatchNorm -> GELU -> Dropout, along the time axis."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(
                in_channels,
                out_channels,
                kernel_size,
                stride=stride,
                padding=kernel_size // 2,
                bias=False,
            ),
            nn.BatchNorm1d(out_channels),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class ECGResNet1D(nn.Module):
    """
    1-D residual CNN over a 12-lead, 10-second, 100 Hz recording.

    Input ``(batch, 12, 1000)``; output ``(batch, 5)`` raw logits — the sigmoid
    is applied by the caller, because the loss was ``BCEWithLogitsLoss``.

    The residual adds are placed around each pair of ConvBlocks, with the
    channel change and the stride-2 downsample living in the 1x1 ``down``
    convolutions between stages. That is what makes ``x + residual`` shape-safe
    without a projection inside the block.
    """

    def __init__(
        self,
        n_classes: int = ECG_NUM_CLASSES,
        in_channels: int = ECG_IN_CHANNELS,
        dropout: float = ECG_DROPOUT,
    ) -> None:
        super().__init__()

        self.stem = nn.Sequential(
            nn.Conv1d(in_channels, 64, kernel_size=15, stride=2, padding=7, bias=False),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.MaxPool1d(3, stride=2, padding=1),
        )

        self.block1 = nn.Sequential(
            ConvBlock(64, 64, 7, dropout=dropout),
            ConvBlock(64, 64, 7, dropout=dropout),
        )
        self.down1 = nn.Conv1d(64, 128, kernel_size=1, stride=2, bias=False)

        self.block2 = nn.Sequential(
            ConvBlock(128, 128, 7, dropout=dropout),
            ConvBlock(128, 128, 7, dropout=dropout),
        )
        self.down2 = nn.Conv1d(128, 256, kernel_size=1, stride=2, bias=False)

        self.block3 = nn.Sequential(
            ConvBlock(256, 256, 5, dropout=dropout),
            ConvBlock(256, 256, 5, dropout=dropout),
        )
        self.down3 = nn.Conv1d(256, 512, kernel_size=1, stride=2, bias=False)

        self.block4 = nn.Sequential(
            ConvBlock(512, 512, 5, dropout=dropout),
            ConvBlock(512, 512, 5, dropout=dropout),
        )

        self.pool = nn.AdaptiveAvgPool1d(1)

        self.classifier = nn.Sequential(
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)

        residual = x
        x = self.block1(x)
        x = x + residual

        residual = self.down1(x)
        x = self.block2(residual)
        x = x + residual

        residual = self.down2(x)
        x = self.block3(residual)
        x = x + residual

        residual = self.down3(x)
        x = self.block4(residual)
        x = x + residual

        x = self.pool(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)


# ============================================================
# RESULT TYPES
# ============================================================


@dataclass
class ClassPrediction:
    """One of the five diagnostic superclasses, for one recording."""

    name: str
    label: str
    description: str
    probability: float
    positive: bool
    threshold: float
    # Dataset-level operating characteristics for this class, so the UI can put
    # a probability next to the precision it was measured at. Never a
    # per-recording confidence.
    auroc: Optional[float]
    average_precision: Optional[float]
    precision: Optional[float]
    recall: Optional[float]
    prevalence: Optional[float]
    caveat: Optional[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "label": self.label,
            "description": self.description,
            "probability": round(self.probability, 4),
            "percent": round(self.probability * 100.0, 1),
            "positive": self.positive,
            "threshold": self.threshold,
            "operating_point": {
                "auroc": self.auroc,
                "average_precision": self.average_precision,
                "precision": self.precision,
                "recall": self.recall,
                "test_prevalence": self.prevalence,
                "scope": "dataset-level, held-out PTB-XL test split",
            },
            "caveat": self.caveat,
        }


@dataclass
class LeadAttribution:
    """How much the model's decision moved with each lead."""

    name: str
    score: float           # normalised to the strongest lead, in [0, 1]
    raw_score: float       # mean |gradient| in the input's own units

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "score": round(self.score, 4),
            "percent": round(self.score * 100.0, 1),
            "raw_score": float(f"{self.raw_score:.6g}"),
        }


@dataclass
class EcgAnalysis:
    predictions: list[ClassPrediction]
    # (12, 1000) float32 in [0, 1], normalised per lead — for the display
    # overlay only. Not comparable between leads; use `leads` for that.
    saliency: np.ndarray
    saliency_class: Optional[str]
    saliency_available: bool
    leads: list[LeadAttribution]
    positive_classes: list[str]
    inference_ms: float
    compute_device: str
    notes: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "predictions": [p.to_dict() for p in self.predictions],
            "positive_classes": self.positive_classes,
            "threshold": ECG_THRESHOLD,
            "saliency_class": self.saliency_class,
            "saliency_available": self.saliency_available,
            "lead_attribution": [lead.to_dict() for lead in self.leads],
            "inference_ms": round(self.inference_ms, 1),
            "compute_device": self.compute_device,
            "notes": self.notes,
        }


# ============================================================
# CLASSIFIER
# ============================================================


class EcgClassifier:
    """Lazily-loaded singleton wrapper around the trained ECG model."""

    def __init__(self) -> None:
        self._model: Optional[ECGResNet1D] = None
        self._checkpoint_meta: dict[str, Any] = {}
        self._metrics: dict[str, Any] = {}
        self._load_error: Optional[str] = None
        self._load_notes: list[str] = []
        # Same reason as the echo model: zero_grad() and tensor.grad are shared
        # mutable state, and FastAPI runs sync endpoints in a threadpool.
        self._lock = threading.Lock()

    # ---- state ------------------------------------------------

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    @property
    def load_error(self) -> Optional[str]:
        return self._load_error

    # ---- loading ----------------------------------------------

    def load(self) -> None:
        """Build the architecture and load the trained weights."""
        if self._model is not None:
            return

        if not ECG_CHECKPOINT_PATH.exists():
            self._load_error = (
                f"ECG checkpoint not found at {ECG_CHECKPOINT_PATH}. Expected "
                "cardioVision_ptbxl_ecg_resnet1d_full.pt (~16 MB) — if you "
                "cloned this repo, the file is tracked with Git LFS, so run: "
                "git lfs pull"
            )
            raise EcgModelUnavailable(self._load_error)

        size = ECG_CHECKPOINT_PATH.stat().st_size
        if size < 1_000_000:
            self._load_error = (
                f"ECG checkpoint at {ECG_CHECKPOINT_PATH} is only {size} bytes, "
                "which looks like an unresolved Git LFS pointer rather than "
                "real weights. Run: git lfs pull"
            )
            raise EcgModelUnavailable(self._load_error)

        print("=" * 60)
        print("Loading CardioVision ECG classification model...")
        print(f"Checkpoint: {ECG_CHECKPOINT_PATH.name}")
        print(f"Device: {DEVICE}")

        notes: list[str] = []

        try:
            checkpoint = torch.load(
                ECG_CHECKPOINT_PATH, map_location="cpu", weights_only=False
            )

            if not isinstance(checkpoint, dict):
                raise EcgModelUnavailable(
                    "Unexpected checkpoint format: expected a dict saved by "
                    "torch.save with a 'model_state_dict' key."
                )

            config = checkpoint.get("model_config") or {}

            model = ECGResNet1D(
                n_classes=int(config.get("n_classes", ECG_NUM_CLASSES)),
                in_channels=int(
                    checkpoint.get("input_channels", ECG_IN_CHANNELS)
                ),
                dropout=float(config.get("dropout", ECG_DROPOUT)),
            )

            state_dict = checkpoint.get("model_state_dict", checkpoint)

            # strict=True on purpose, and it doubles as a check on the
            # architecture reconstruction above: any layer with the wrong name
            # or shape fails here rather than producing confident nonsense.
            model.load_state_dict(state_dict, strict=True)

        except EcgModelUnavailable:
            raise
        except Exception as error:
            self._load_error = f"Failed to load the ECG model: {error}"
            raise EcgModelUnavailable(self._load_error) from error

        # ---- cross-check the checkpoint against the config ----
        #
        # These are not cosmetic. Class order decides which probability is
        # called "MI"; lead order decides which column the model reads as V1.
        # A mismatch means every subsequent number is wrong in a way no
        # downstream check would catch.

        stored_classes = checkpoint.get("target_classes")
        if stored_classes is not None:
            stored = tuple(str(name) for name in stored_classes)
            if stored != tuple(ECG_CLASS_NAMES):
                self._load_error = (
                    f"The checkpoint's class order is {stored}, but this build "
                    f"expects {tuple(ECG_CLASS_NAMES)}. Loading it anyway would "
                    "attach the wrong diagnosis to every probability, so this "
                    "is refused. Update ECG_CLASS_NAMES in "
                    "cardiovision/config.py to match the checkpoint."
                )
                raise EcgModelUnavailable(self._load_error)

        stored_leads = checkpoint.get("lead_names")
        if stored_leads is not None:
            leads = tuple(str(name) for name in stored_leads)
            if leads != tuple(ECG_LEAD_NAMES):
                self._load_error = (
                    f"The checkpoint was trained on lead order {leads}, but "
                    f"this build expects {tuple(ECG_LEAD_NAMES)}. Refusing to "
                    "load: the model would read each lead as a different one."
                )
                raise EcgModelUnavailable(self._load_error)

        stored_length = checkpoint.get("input_length")
        if stored_length is not None and int(stored_length) != ECG_INPUT_LENGTH:
            self._load_error = (
                f"The checkpoint expects {int(stored_length)} samples per lead "
                f"and this build preprocesses to {ECG_INPUT_LENGTH}. Refusing "
                "to load rather than feed it the wrong window length."
            )
            raise EcgModelUnavailable(self._load_error)

        stored_fs = checkpoint.get("target_fs")
        if stored_fs is not None and float(stored_fs) != float(ECG_TARGET_FS):
            self._load_error = (
                f"The checkpoint was trained at {float(stored_fs)} Hz and this "
                f"build resamples to {ECG_TARGET_FS} Hz. Refusing to load: the "
                "same waveform at the wrong rate is a different signal to a "
                "convolutional model."
            )
            raise EcgModelUnavailable(self._load_error)

        parameters = sum(p.numel() for p in model.parameters())
        if parameters != ECG_PARAMETERS:
            # Not fatal — strict=True already proved the weights fit — but it
            # means the notebook and this file have drifted, and the recorded
            # metrics may describe a different network.
            notes.append(
                f"This build has {parameters:,} parameters; the training "
                f"notebook recorded {ECG_PARAMETERS:,}. The weights still "
                "loaded strictly, so check whether the architecture was "
                "edited after training."
            )

        model.eval()
        model.to(DEVICE)

        self._model = model

        # Prefer the checkpoint's own metrics over the copy in config.py, so
        # they always describe these exact weights.
        self._metrics = dict(checkpoint.get("test_metrics") or {})

        self._checkpoint_meta = {
            "architecture": ECG_ARCHITECTURE,
            "parameters": parameters,
            "best_epoch": checkpoint.get("best_epoch"),
            "best_validation_macro_auroc": checkpoint.get(
                "best_validation_macro_AUROC"
            ),
            "input_channels": int(
                checkpoint.get("input_channels", ECG_IN_CHANNELS)
            ),
            "input_length": int(checkpoint.get("input_length", ECG_INPUT_LENGTH)),
            "target_fs": float(checkpoint.get("target_fs", ECG_TARGET_FS)),
            "dropout": float((checkpoint.get("model_config") or {}).get(
                "dropout", ECG_DROPOUT
            )),
            "preprocessing": dict(checkpoint.get("preprocessing_config") or {}),
            # True only when the per-class rows really came out of the file, not
            # just when a test_metrics key exists. The flat "HYP_Precision"
            # naming is easy to miss, and missing it fails soft.
            "metrics_from_checkpoint": any(
                f"{name}_AUROC" in self._metrics for name in ECG_CLASS_NAMES
            ),
        }

        self._load_notes = notes
        self._load_error = None

        print(
            "ECG model loaded. "
            f"epoch={self._checkpoint_meta['best_epoch']} "
            f"val_macro_AUROC={self._checkpoint_meta['best_validation_macro_auroc']} "
            f"params={parameters:,}"
        )
        for note in notes:
            print(f"  note: {note}")
        print("=" * 60)

    # ---- metadata ---------------------------------------------

    def _class_metrics(self, name: str) -> dict[str, Any]:
        """
        Per-class test metrics, checkpoint first, config as the fallback.

        The checkpoint stores these flat — ``NORM_AUROC``, ``HYP_Precision`` and
        so on — rather than nested under a per-class key, so they are collected
        by prefix here. Getting this wrong is not loud: a lookup that misses
        just falls back to the copy in config.py, which is correct today and
        would quietly stop tracking the weights the moment anyone retrains. Hence
        ``metrics_from_checkpoint`` in the model card, which reports which of the
        two actually answered.
        """
        keys = ("AUROC", "AP", "F1", "Precision", "Recall")

        row = {
            key: self._metrics[f"{name}_{key}"]
            for key in keys
            if f"{name}_{key}" in self._metrics
        }

        # Also accept a nested form, in case a future export nests them.
        nested = (self._metrics.get("per_class") or {}).get(name)
        if isinstance(nested, dict):
            row = {**nested, **row}

        if not row:
            row = dict(ECG_TEST_METRICS["per_class"].get(name, {}))

        # numpy scalars all the way from sklearn. They survive arithmetic
        # happily and then blow up in json.dumps, several layers from here.
        return {key: _as_float(value) for key, value in row.items()}

    def describe(self) -> dict[str, Any]:
        """Model card: what these weights are and how well they scored."""
        meta = dict(self._checkpoint_meta)

        per_class = {
            name: {
                "label": ECG_CLASS_LABELS[name],
                **{
                    key: self._class_metrics(name).get(key)
                    for key in ("AUROC", "AP", "F1", "Precision", "Recall")
                },
                "test_prevalence": ECG_TEST_METRICS["test_prevalence"].get(name),
                "caveat": ECG_WEAK_CLASSES.get(name),
            }
            for name in ECG_CLASS_NAMES
        }

        def macro(key: str) -> Optional[float]:
            """Checkpoint's macro metric, falling back to the config copy."""
            if key in self._metrics:
                return _as_float(self._metrics[key])
            return _as_float(ECG_TEST_METRICS.get(key))

        return {
            "architecture": meta.get("architecture", ECG_ARCHITECTURE),
            "task": "multi-label classification (independent sigmoids)",
            "input": {
                "leads": meta.get("input_channels", ECG_IN_CHANNELS),
                "samples": meta.get("input_length", ECG_INPUT_LENGTH),
                "sampling_rate_hz": meta.get("target_fs", ECG_TARGET_FS),
                "duration_seconds": (
                    meta.get("input_length", ECG_INPUT_LENGTH)
                    / meta.get("target_fs", ECG_TARGET_FS)
                ),
                "lead_names": list(ECG_LEAD_NAMES),
            },
            "parameters": meta.get("parameters"),
            "best_epoch": meta.get("best_epoch"),
            "class_names": list(ECG_CLASS_NAMES),
            "class_labels": dict(ECG_CLASS_LABELS),
            "threshold": ECG_THRESHOLD,
            "metrics": {
                "macro_AUROC": macro("macro_AUROC"),
                "macro_AP": macro("macro_AP"),
                "macro_F1": macro("macro_F1"),
                "macro_Precision": macro("macro_Precision"),
                "macro_Recall": macro("macro_Recall"),
                "validation_macro_AUROC": _as_float(
                    meta.get("best_validation_macro_auroc")
                ),
                "per_class": per_class,
                "dataset": ECG_TEST_METRICS["dataset"],
                "test_records": ECG_TEST_METRICS["test_records"],
                "test_patients": ECG_TEST_METRICS["test_patients"],
                "source": ECG_TEST_METRICS["source"],
                "read_from_checkpoint": meta.get("metrics_from_checkpoint", False),
                # Guard rail for the UI, same wording as the echo card: these
                # describe the model on a held-out cohort, not this recording.
                "scope": "dataset-level, held-out test split",
                # The macro numbers above hide this, so it travels with them.
                "weak_classes": dict(ECG_WEAK_CLASSES),
            },
            "explainability": {
                "method": "input-gradient saliency (|d logit / d sample|)",
                "note": (
                    "Gradient magnitude shows which samples the logit was most "
                    "sensitive to. It is not a measurement of any waveform "
                    "feature, and it does not localise a diagnosis."
                ),
            },
            "device": DEVICE,
            "notes": list(self._load_notes),
        }

    # ---- inference --------------------------------------------

    def _forward_with_saliency(
        self,
        tensor: torch.Tensor,
        device: str,
        target_class: Optional[int],
    ) -> tuple[torch.Tensor, Optional[torch.Tensor], int]:
        """
        One forward pass plus the input gradient for the reported class.

        Mirrors ``ECGSaliency.generate`` in the notebook: the gradient is taken
        on the raw *logit*, not the sigmoid. Past the tails of the sigmoid the
        derivative is nearly zero, which would flatten the map for exactly the
        confident recordings someone most wants to inspect.
        """
        model = self._model
        assert model is not None

        working = tensor.to(device)
        working.requires_grad_(True)

        model.zero_grad(set_to_none=True)
        working.grad = None

        logits = model(working)

        if target_class is None:
            target_class = int(torch.argmax(logits[0]).item())

        logits[0, target_class].backward()

        gradient = working.grad
        saliency = gradient.detach().abs() if gradient is not None else None

        return logits.detach(), saliency, target_class

    def analyze(
        self,
        signal: np.ndarray,
        target_class: Optional[str] = None,
    ) -> EcgAnalysis:
        """
        Classify one preprocessed 12-lead recording.

        ``signal`` is ``(1000, 12)`` float32, already through
        :func:`cardiovision.preprocessing.ecg_io.preprocess_ecg` — this method
        does no filtering, resampling or normalisation of its own, because
        doing it in two places is how the two drift apart.
        """
        if self._model is None:
            raise EcgModelUnavailable(
                self._load_error or "The ECG model is not loaded."
            )

        if signal.shape != (ECG_INPUT_LENGTH, ECG_IN_CHANNELS):
            raise EcgModelUnavailable(
                f"Expected a preprocessed ({ECG_INPUT_LENGTH}, "
                f"{ECG_IN_CHANNELS}) signal, got {signal.shape}. Run it "
                "through preprocess_ecg first."
            )

        target_index: Optional[int] = None
        if target_class is not None:
            if target_class not in ECG_CLASS_NAMES:
                raise EcgModelUnavailable(
                    f"Unknown target class {target_class!r}. "
                    f"Expected one of {', '.join(ECG_CLASS_NAMES)}."
                )
            target_index = ECG_CLASS_NAMES.index(target_class)

        notes: list[str] = list(self._load_notes)

        # (1000, 12) -> (1, 12, 1000): the model is channels-first.
        tensor = torch.from_numpy(
            np.ascontiguousarray(signal.T)
        ).float().unsqueeze(0)

        with self._lock:
            started = time.perf_counter()
            compute_device = DEVICE

            try:
                logits, saliency, used_index = self._forward_with_saliency(
                    tensor, DEVICE, target_index
                )
            except (RuntimeError, NotImplementedError) as error:
                # Same MPS fallback as the echo path: a missing gradient kernel
                # should cost us the accelerator, not the analysis.
                notes.append(
                    f"Inference failed on {DEVICE} ({error}); retried on CPU."
                )
                try:
                    self._model.to("cpu")
                    logits, saliency, used_index = self._forward_with_saliency(
                        tensor, "cpu", target_index
                    )
                    compute_device = "cpu"
                finally:
                    self._model.to(DEVICE)

            probabilities = (
                torch.sigmoid(logits[0].float()).to("cpu").numpy()
            )

            if saliency is None:
                saliency_available = False
                saliency_class = None
                raw_gradient = None
                display_saliency = np.zeros(
                    (ECG_IN_CHANNELS, ECG_INPUT_LENGTH), dtype=np.float32
                )
                notes.append(
                    "Input gradients were unavailable, so no saliency was "
                    "produced. The probabilities themselves are unaffected."
                )
            else:
                saliency_available = True
                saliency_class = ECG_CLASS_NAMES[used_index]
                raw_gradient = saliency[0].to("cpu").numpy().astype(np.float32)
                # Per-lead normalisation, exactly as the notebook: it is what
                # makes every lead's band visible on the strip regardless of
                # how much gradient that lead carries.
                display_saliency = (
                    raw_gradient
                    / (raw_gradient.max(axis=1, keepdims=True) + 1e-8)
                ).astype(np.float32)

            inference_ms = (time.perf_counter() - started) * 1000.0

        predictions = self._build_predictions(probabilities)
        positive = [p.name for p in predictions if p.positive]
        notes.extend(self._build_notes(predictions))

        return EcgAnalysis(
            predictions=predictions,
            saliency=display_saliency,
            saliency_class=saliency_class,
            saliency_available=saliency_available,
            leads=self._rank_leads(raw_gradient),
            positive_classes=positive,
            inference_ms=inference_ms,
            compute_device=compute_device,
            notes=notes,
        )

    # ---- reporting --------------------------------------------

    def _build_predictions(
        self, probabilities: np.ndarray
    ) -> list[ClassPrediction]:
        """Attach each probability to the operating point it was measured at."""
        predictions: list[ClassPrediction] = []

        for index, name in enumerate(ECG_CLASS_NAMES):
            probability = float(probabilities[index])
            row = self._class_metrics(name)

            predictions.append(
                ClassPrediction(
                    name=name,
                    label=ECG_CLASS_LABELS[name],
                    description=ECG_CLASS_DESCRIPTIONS[name],
                    probability=probability,
                    positive=probability >= ECG_THRESHOLD,
                    threshold=ECG_THRESHOLD,
                    auroc=_as_float(row.get("AUROC")),
                    average_precision=_as_float(row.get("AP")),
                    precision=_as_float(row.get("Precision")),
                    recall=_as_float(row.get("Recall")),
                    prevalence=_as_float(
                        ECG_TEST_METRICS["test_prevalence"].get(name)
                    ),
                    caveat=ECG_WEAK_CLASSES.get(name),
                )
            )

        # Highest probability first: multi-label output has no natural order,
        # and class index order would bury a strong MI under a weak NORM.
        predictions.sort(key=lambda p: p.probability, reverse=True)
        return predictions

    @staticmethod
    def _build_notes(predictions: list[ClassPrediction]) -> list[str]:
        """
        The caveats that have to travel with these numbers.

        Separated from :meth:`analyze` so it can be tested without torch, and
        because it is policy rather than plumbing: what a reader is warned about
        is a decision, and decisions deserve their own function.
        """
        notes: list[str] = []
        positive = [p for p in predictions if p.positive]

        if not positive:
            notes.append(
                f"No class reached the {ECG_THRESHOLD} threshold, including "
                "NORM. That is not a normal ECG and it is not an abnormal one — "
                "it is a recording this model has no confident call on."
            )

        # A NORM call alongside an abnormality call is not a contradiction in a
        # multi-label model, but it reads like one, so say what it means.
        abnormal = [p for p in positive if p.name != "NORM"]
        if any(p.name == "NORM" for p in positive) and abnormal:
            names = ", ".join(p.label for p in abnormal)
            notes.append(
                f"Both NORM and {names} came out above threshold. The five "
                "classes are independent sigmoids rather than one softmax, so "
                "this is the model disagreeing with itself, not a combined "
                "finding. Read the tracing."
            )

        for prediction in positive:
            if prediction.caveat:
                notes.append(f"{prediction.label}: {prediction.caveat}")

        # A class sitting just under the line is worth mentioning: at 0.49 the
        # only thing separating it from a positive call is a threshold someone
        # chose, and a reader who never sees it cannot weigh that.
        borderline = [
            p for p in predictions
            if not p.positive and p.probability >= ECG_THRESHOLD - 0.1
        ]
        if borderline:
            listed = ", ".join(
                f"{p.label} {p.probability:.2f}" for p in borderline
            )
            notes.append(
                f"Just below the {ECG_THRESHOLD} threshold: {listed}. Reported "
                "as negative only because of where the threshold sits."
            )

        return notes

    @staticmethod
    def _rank_leads(
        raw_gradient: Optional[np.ndarray],
    ) -> list[LeadAttribution]:
        """
        Rank leads by mean absolute gradient, from the RAW map.

        This deliberately does not reuse the per-lead-normalised saliency that
        the strip is drawn from, and the difference matters. Dividing each lead
        by its own maximum makes every lead peak at 1.0, so the mean of the
        normalised trace measures how *spread out* a lead's attribution is, not
        how much attribution it has. A lead with one sharp spike would score
        near zero and a uniformly noisy lead would score near one — the
        opposite of what a reader would take from a bar labelled "importance".

        models/ecg/lead_importance.csv, shipped from the notebook, was computed
        the normalised way, so it is provenance for that one figure and is not
        comparable with the ranking here.
        """
        if raw_gradient is None:
            return []

        means = raw_gradient.mean(axis=1)
        strongest = float(means.max())

        attributions = [
            LeadAttribution(
                name=ECG_LEAD_NAMES[index],
                score=float(means[index] / strongest) if strongest > 0 else 0.0,
                raw_score=float(means[index]),
            )
            for index in range(len(ECG_LEAD_NAMES))
        ]

        attributions.sort(key=lambda lead: lead.raw_score, reverse=True)
        return attributions


def _as_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        return round(float(value), 4)
    except (TypeError, ValueError):
        return None


# Module-level singleton.
ecg_classifier = EcgClassifier()
