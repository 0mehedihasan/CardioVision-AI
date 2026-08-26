"""
CardioVision AI — echocardiography segmentation inference.

Rebuilds the UNet++ / EfficientNet-B3 architecture from
notebooks/02_Echo_Training.ipynb, loads the trained checkpoint, and runs
4-class cardiac structure segmentation with gradient-saliency
explainability.

Design notes
------------
* The encoder is built with ``encoder_weights=None``. The trained weights
  come entirely from the checkpoint, so the backend never reaches out to
  the network and stays usable fully offline.
* Inference runs in float32. The notebook used ``torch.autocast`` with
  float16, which is CUDA-only; on Apple Silicon (MPS) float32 is both
  safer and numerically closer to the saved weights.
* Validation metrics are read out of the checkpoint rather than
  hardcoded, so they can never drift away from the weights they describe.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
import torch

from cardiovision.config import (
    DEVICE,
    ECHO_ARCHITECTURE,
    ECHO_CHECKPOINT_PATH,
    ECHO_CLASS_NAMES,
    ECHO_ENCODER,
    ECHO_FOREGROUND_CLASSES,
    ECHO_IMAGE_SIZE,
    ECHO_IN_CHANNELS,
    ECHO_NUM_CLASSES,
    ECHO_PRESENCE_THRESHOLD_PX,
    ECHO_SALIENCY_CLASS,
    ECHO_TEST_METRICS,
)
from cardiovision.preprocessing.image_io import preprocess_to_tensor


class EchoModelUnavailable(RuntimeError):
    """Raised when the echo model cannot be loaded or used."""


@dataclass
class StructureFinding:
    class_index: int
    name: str
    present: bool
    pixels: int
    area_fraction: float
    mean_confidence: float
    area_cm2: Optional[float] = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "class_index": self.class_index,
            "name": self.name,
            "present": self.present,
            "pixels": self.pixels,
            "area_fraction": round(self.area_fraction, 6),
            "area_percent": round(self.area_fraction * 100.0, 3),
            "mean_confidence": round(self.mean_confidence, 4),
            "area_cm2": (
                round(self.area_cm2, 3) if self.area_cm2 is not None else None
            ),
        }


@dataclass
class EchoAnalysis:
    prediction: np.ndarray           # (256, 256) uint8 class indices
    saliency: np.ndarray             # (256, 256) float32 in [0, 1]
    normalised_input: np.ndarray     # (256, 256) float32 in [0, 1]
    structures: list[StructureFinding]
    inference_ms: float
    saliency_available: bool
    saliency_device: str
    # The device the forward pass whose output we kept actually ran on. It
    # differs from the configured DEVICE whenever the CPU retry supplied the
    # result, and the response must report the truth rather than the intent.
    compute_device: str
    notes: list[str]


# ============================================================
# SEGMENTER
# ============================================================

class EchoSegmenter:
    """Lazily-loaded singleton wrapper around the trained echo model."""

    def __init__(self) -> None:
        self._model = None
        self._checkpoint_meta: dict[str, Any] = {}
        self._load_error: Optional[str] = None
        # model.zero_grad() and tensor.grad are shared mutable state, and
        # FastAPI runs sync endpoints in a threadpool, so inference must be
        # serialised.
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

        if not ECHO_CHECKPOINT_PATH.exists():
            self._load_error = (
                f"Echo checkpoint not found at {ECHO_CHECKPOINT_PATH}. "
                "Expected cardiovision_echo_unetplusplus_best.pth "
                "(~160 MB) — if you cloned this repo, the file is tracked "
                "with Git LFS, so run: git lfs pull"
            )
            raise EchoModelUnavailable(self._load_error)

        # A Git LFS pointer file is a few hundred bytes of text; real
        # weights are ~160 MB. Catch this early with a clear message
        # instead of an opaque unpickling error.
        size = ECHO_CHECKPOINT_PATH.stat().st_size
        if size < 1_000_000:
            self._load_error = (
                f"Echo checkpoint at {ECHO_CHECKPOINT_PATH} is only "
                f"{size} bytes, which looks like an unresolved Git LFS "
                "pointer rather than real weights. Run: git lfs pull"
            )
            raise EchoModelUnavailable(self._load_error)

        try:
            import segmentation_models_pytorch as smp
        except ImportError as error:
            self._load_error = (
                "segmentation-models-pytorch is not installed, so the echo "
                "architecture cannot be rebuilt. Install it with: "
                "pip install segmentation-models-pytorch timm"
            )
            raise EchoModelUnavailable(self._load_error) from error

        print("=" * 60)
        print("Loading CardioVision echo segmentation model...")
        print(f"Checkpoint: {ECHO_CHECKPOINT_PATH.name}")
        print(f"Device: {DEVICE}")

        try:
            model = smp.UnetPlusPlus(
                encoder_name=ECHO_ENCODER,
                # None, not "imagenet": the checkpoint supplies every
                # weight, and this keeps startup fully offline.
                encoder_weights=None,
                in_channels=ECHO_IN_CHANNELS,
                classes=ECHO_NUM_CLASSES,
                activation=None,
            )

            checkpoint = torch.load(
                ECHO_CHECKPOINT_PATH,
                map_location="cpu",
                weights_only=False,
            )

            if not isinstance(checkpoint, dict):
                raise EchoModelUnavailable(
                    "Unexpected checkpoint format: expected a dict saved by "
                    "torch.save with a 'model_state_dict' key."
                )

            state_dict = checkpoint.get("model_state_dict", checkpoint)

            # strict=True on purpose. A silent partial load would produce
            # plausible-looking but meaningless segmentations, which is the
            # worst possible failure mode for a clinical tool.
            model.load_state_dict(state_dict, strict=True)

        except EchoModelUnavailable:
            raise
        except Exception as error:
            self._load_error = f"Failed to load the echo model: {error}"
            raise EchoModelUnavailable(self._load_error) from error

        model.eval()
        model.to(DEVICE)

        self._model = model

        # Read the metrics back out of the checkpoint so they always
        # describe these exact weights.
        self._checkpoint_meta = {
            "epoch": checkpoint.get("epoch"),
            "validation_dice": checkpoint.get("val_dice"),
            "validation_iou": checkpoint.get("val_iou"),
            "image_size": checkpoint.get("image_size", ECHO_IMAGE_SIZE),
            "num_classes": checkpoint.get("num_classes", ECHO_NUM_CLASSES),
            "architecture": checkpoint.get("architecture", ECHO_ARCHITECTURE),
            "encoder": checkpoint.get("encoder", ECHO_ENCODER),
            "parameters": sum(p.numel() for p in model.parameters()),
        }

        self._load_error = None

        print(
            "Echo model loaded. "
            f"epoch={self._checkpoint_meta['epoch']} "
            f"val_dice={self._checkpoint_meta['validation_dice']} "
            f"params={self._checkpoint_meta['parameters']:,}"
        )
        print("=" * 60)

    # ---- metadata ---------------------------------------------

    def describe(self) -> dict[str, Any]:
        """Model card: what these weights are and how well they scored."""
        meta = dict(self._checkpoint_meta)

        validation_dice = meta.get("validation_dice")
        validation_iou = meta.get("validation_iou")

        return {
            "architecture": meta.get("architecture", ECHO_ARCHITECTURE),
            "encoder": meta.get("encoder", ECHO_ENCODER),
            "input_size": meta.get("image_size", ECHO_IMAGE_SIZE),
            "num_classes": meta.get("num_classes", ECHO_NUM_CLASSES),
            "parameters": meta.get("parameters"),
            "best_epoch": meta.get("epoch"),
            "class_names": ECHO_CLASS_NAMES,
            "metrics": {
                # From the checkpoint itself.
                "validation_dice": (
                    round(float(validation_dice), 4)
                    if validation_dice is not None else None
                ),
                "validation_iou": (
                    round(float(validation_iou), 4)
                    if validation_iou is not None else None
                ),
                # From the notebook's held-out test evaluation.
                "test_dice": ECHO_TEST_METRICS["test_dice"],
                "test_iou": ECHO_TEST_METRICS["test_iou"],
                "per_class_test_dice": ECHO_TEST_METRICS["per_class_dice"],
                "dataset": ECHO_TEST_METRICS["dataset"],
                "test_patients": ECHO_TEST_METRICS["test_patients"],
                "test_pairs": ECHO_TEST_METRICS["test_pairs"],
                "source": ECHO_TEST_METRICS["source"],
                # Guard rail for the UI: these describe the model on a
                # held-out cohort. They are NOT a per-prediction score.
                "scope": "dataset-level, held-out test split",
            },
            "device": DEVICE,
        }

    # ---- inference --------------------------------------------

    def _forward_with_saliency(
        self,
        tensor: torch.Tensor,
        device: str,
    ) -> tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        Single forward pass that yields both the prediction and the
        input-gradient saliency, mirroring the notebook's XAI block.
        """
        model = self._model
        assert model is not None

        working = tensor.to(device)
        working.requires_grad_(True)

        # Both the parameter grads and the leaf's own .grad must start clean.
        # backward() accumulates, so a stale gradient left over from a failed
        # attempt would be added to this one and silently corrupt the map.
        model.zero_grad(set_to_none=True)
        working.grad = None

        logits = model(working)
        probabilities = torch.softmax(logits.float(), dim=1)

        # Target: mean probability of LV cavity (class 1), matching
        # TARGET_CLASS = 1 in the training notebook.
        target_score = probabilities[0, ECHO_SALIENCY_CLASS].mean()
        target_score.backward()

        gradient = working.grad
        saliency = gradient.detach().abs() if gradient is not None else None

        return probabilities.detach(), working.detach(), saliency

    def analyze(
        self,
        pixels: np.ndarray,
        original_shape: tuple[int, int],
        pixel_spacing_mm: Optional[tuple[float, float]] = None,
    ) -> EchoAnalysis:
        """Segment one echo frame and quantify the predicted structures."""
        if self._model is None:
            raise EchoModelUnavailable(
                self._load_error or "The echo model is not loaded."
            )

        notes: list[str] = []
        tensor = preprocess_to_tensor(pixels)

        with self._lock:
            # Timed inside the lock: a request that waited behind another
            # one would otherwise report its queueing delay as compute time.
            started = time.perf_counter()

            compute_device = DEVICE
            saliency_device = DEVICE
            saliency_available = True

            try:
                probabilities, used_input, saliency = (
                    self._forward_with_saliency(tensor, DEVICE)
                )
            except (RuntimeError, NotImplementedError) as error:
                # MPS occasionally rejects a backward pass for ops that
                # lack a gradient kernel. Retry on CPU rather than losing
                # the whole analysis. Only backend/kernel errors are caught;
                # anything else is a real bug and must not be swallowed.
                notes.append(
                    f"Gradient saliency failed on {DEVICE} ({error}); "
                    "retried on CPU."
                )
                try:
                    self._model.to("cpu")
                    probabilities, used_input, saliency = (
                        self._forward_with_saliency(tensor, "cpu")
                    )
                    compute_device = "cpu"
                    saliency_device = "cpu"
                finally:
                    self._model.to(DEVICE)

            if saliency is None:
                saliency_available = False
                saliency_map = np.zeros(
                    (ECHO_IMAGE_SIZE, ECHO_IMAGE_SIZE), dtype=np.float32
                )
                notes.append(
                    "Input gradients were unavailable, so no saliency map "
                    "was produced. The segmentation itself is unaffected."
                )
            else:
                saliency_map = saliency[0, 0].to("cpu").numpy()
                # Replicates the notebook's normalisation exactly, including
                # its divide-by-max (rather than max-min) form, so this
                # matches models/echo/xai_lv_cavity_saliency.npy.
                saliency_map = (
                    (saliency_map - saliency_map.min())
                    / (saliency_map.max() + 1e-8)
                ).astype(np.float32)

            prediction_tensor = torch.argmax(probabilities, dim=1)[0]
            prediction = prediction_tensor.to("cpu").numpy().astype(np.uint8)

            probability_map = probabilities[0].to("cpu").numpy()
            normalised_input = (
                used_input[0, 0].to("cpu").numpy().astype(np.float32)
            )

            inference_ms = (time.perf_counter() - started) * 1000.0

        structures = self._quantify(
            prediction=prediction,
            probability_map=probability_map,
            original_shape=original_shape,
            pixel_spacing_mm=pixel_spacing_mm,
        )

        return EchoAnalysis(
            prediction=prediction,
            saliency=saliency_map,
            normalised_input=normalised_input,
            structures=structures,
            inference_ms=inference_ms,
            saliency_available=saliency_available,
            saliency_device=saliency_device,
            compute_device=compute_device,
            notes=notes,
        )

    # ---- quantification ---------------------------------------

    @staticmethod
    def _quantify(
        prediction: np.ndarray,
        probability_map: np.ndarray,
        original_shape: tuple[int, int],
        pixel_spacing_mm: Optional[tuple[float, float]],
    ) -> list[StructureFinding]:
        """
        Turn a class mask into per-structure measurements.

        Absolute areas are only reported when the source image carried
        pixel spacing. The mask lives at 256×256 while spacing describes
        the original grid, so area is computed as a fraction of the
        original field of view rather than by scaling pixel counts.
        """
        total_pixels = int(prediction.size)

        physical_area_mm2: Optional[float] = None
        if pixel_spacing_mm is not None:
            row_mm, col_mm = pixel_spacing_mm
            rows, cols = original_shape
            physical_area_mm2 = (rows * row_mm) * (cols * col_mm)

        findings: list[StructureFinding] = []

        for class_index in ECHO_FOREGROUND_CLASSES:
            mask = prediction == class_index
            pixel_count = int(mask.sum())
            area_fraction = pixel_count / total_pixels if total_pixels else 0.0

            if pixel_count > 0:
                mean_confidence = float(
                    probability_map[class_index][mask].mean()
                )
            else:
                mean_confidence = 0.0

            area_cm2: Optional[float] = None
            if physical_area_mm2 is not None:
                area_cm2 = (area_fraction * physical_area_mm2) / 100.0

            findings.append(
                StructureFinding(
                    class_index=class_index,
                    name=ECHO_CLASS_NAMES[class_index],
                    # A handful of stray pixels is argmax noise, not a
                    # structure. The threshold is reported through the API
                    # so absence is never mistaken for a clinical finding.
                    present=pixel_count >= ECHO_PRESENCE_THRESHOLD_PX,
                    pixels=pixel_count,
                    area_fraction=area_fraction,
                    mean_confidence=mean_confidence,
                    area_cm2=area_cm2,
                )
            )

        return findings


# Module-level singleton.
echo_segmenter = EchoSegmenter()
