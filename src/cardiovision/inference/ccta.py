"""
CardioVision AI — coronary CT angiography segmentation inference.

Rebuilds the ``Small3DUNet`` from notebooks/01_CCTA_Training.ipynb, loads the
trained checkpoint, and runs sliding-window binary lumen segmentation with 3-D
Grad-CAM explainability.

What this model does
--------------------
One sigmoid output channel per voxel: contrast-filled coronary lumen, or not.
That is the entire vocabulary. It does not grade stenosis, does not compute a
calcium score, does not assign CAD-RADS and does not label which vessel a voxel
belongs to. Nothing downstream may imply otherwise.

Design notes
------------
* The architecture is asserted against the checkpoint, not merely loaded from
  it. ``strict=True`` plus a parameter-count check means a silently mismatched
  block cannot produce a plausible-looking mask from the wrong weights.
* The operating threshold is read out of the checkpoint's ``selected_threshold``
  rather than hardcoded, so it cannot drift from the validation run that chose
  it. The config value is the fallback and the documentation.
* ``LeakyReLU(inplace=False)`` is deliberate and matches the notebook: an
  in-place activation overwrites the tensor the Grad-CAM backward hook needs.
* A full study is hundreds of sliding windows. Rather than run until something
  times out, the service works within a window budget and reports the fraction
  of the volume it actually covered. Voxels outside the analysed region are
  reported as UNANALYSED, never as background — a mask that quietly says
  "no lumen here" about a region it never looked at is the exact failure this
  project exists to avoid.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from cardiovision.config import (
    CCTA_ARCHITECTURE,
    CCTA_BASE_CHANNELS,
    CCTA_CHECKPOINT_PATH,
    CCTA_CLASS_NAMES,
    CCTA_GRADCAM_LAYER,
    CCTA_IN_CHANNELS,
    CCTA_INFERENCE_BATCH_SIZE,
    CCTA_INFERENCE_OVERLAP,
    CCTA_MAX_WINDOWS,
    CCTA_OUT_CHANNELS,
    CCTA_PAD_VALUE,
    CCTA_PARAMETERS,
    CCTA_PATCH_SIZE,
    CCTA_PRESENCE_THRESHOLD_VOXELS,
    CCTA_TARGET_SPACING,
    CCTA_TEST_METRICS,
    CCTA_THRESHOLD,
    CCTA_WEAK_NOTES,
    DEVICE,
)


class CctaModelUnavailable(RuntimeError):
    """Raised when the CCTA model cannot be loaded or used."""


# ============================================================
# ARCHITECTURE
# ============================================================
#
# Must match the notebook tensor for tensor. Verified against the 50 tensors
# and 1,401,265 values in models/ccta/best_3d_unet_cca_v2.pth.


class ConvBlock3D(nn.Module):
    """Conv-Norm-Act twice. InstanceNorm because batches are 2 patches deep."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1,
                      bias=False),
            nn.InstanceNorm3d(out_channels, affine=True),
            # inplace=False is required: the Grad-CAM backward hook reads this
            # activation's output, and an in-place op destroys it.
            nn.LeakyReLU(0.01, inplace=False),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1,
                      bias=False),
            nn.InstanceNorm3d(out_channels, affine=True),
            nn.LeakyReLU(0.01, inplace=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class Small3DUNet(nn.Module):
    """Three-level 3-D U-Net, 1.4M parameters, one sigmoid output channel."""

    def __init__(
        self,
        in_channels: int = CCTA_IN_CHANNELS,
        out_channels: int = CCTA_OUT_CHANNELS,
        base: int = CCTA_BASE_CHANNELS,
    ) -> None:
        super().__init__()

        self.enc1 = ConvBlock3D(in_channels, base)
        self.pool1 = nn.MaxPool3d(2)
        self.enc2 = ConvBlock3D(base, base * 2)
        self.pool2 = nn.MaxPool3d(2)
        self.enc3 = ConvBlock3D(base * 2, base * 4)
        self.pool3 = nn.MaxPool3d(2)

        self.bottleneck = ConvBlock3D(base * 4, base * 8)

        self.up3 = nn.ConvTranspose3d(base * 8, base * 4, kernel_size=2, stride=2)
        self.dec3 = ConvBlock3D(base * 8, base * 4)
        self.up2 = nn.ConvTranspose3d(base * 4, base * 2, kernel_size=2, stride=2)
        self.dec2 = ConvBlock3D(base * 4, base * 2)
        self.up1 = nn.ConvTranspose3d(base * 2, base, kernel_size=2, stride=2)
        self.dec1 = ConvBlock3D(base * 2, base)

        self.out = nn.Conv3d(base, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))

        b = self.bottleneck(self.pool3(e3))

        d3 = self.dec3(torch.cat([self.up3(b), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))

        return self.out(d1)


# ============================================================
# RESULTS
# ============================================================


@dataclass
class LumenFinding:
    """Everything measurable about the predicted mask, and nothing more."""

    name: str
    present: bool
    voxels: int
    volume_ml: float
    fraction_of_analysed: float
    mean_probability: float
    max_probability: float
    components: Optional[int] = None
    largest_component_fraction: Optional[float] = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "present": self.present,
            "voxels": self.voxels,
            "volume_ml": round(self.volume_ml, 3),
            "fraction_of_analysed": round(self.fraction_of_analysed, 8),
            "percent_of_analysed": round(self.fraction_of_analysed * 100.0, 5),
            "mean_probability": round(self.mean_probability, 4),
            "max_probability": round(self.max_probability, 4),
            "components": self.components,
            "largest_component_fraction": (
                round(self.largest_component_fraction, 4)
                if self.largest_component_fraction is not None else None
            ),
        }


@dataclass
class CctaAnalysis:
    mask: np.ndarray                 # uint8, 1 = lumen, full volume shape
    analysed: np.ndarray             # bool, True where a window actually ran
    probability: np.ndarray          # float32 in [0, 1], full volume shape
    gradcam: Optional[np.ndarray]    # float32 in [0, 1] over the CAM patch
    gradcam_origin: Optional[tuple[int, int, int]]
    gradcam_shape: Optional[tuple[int, int, int]]
    findings: list[LumenFinding]
    threshold: float
    windows_run: int
    windows_total: int
    coverage: float                  # fraction of volume actually analysed
    inference_ms: float
    compute_device: str
    gradcam_available: bool
    gradcam_device: Optional[str]
    notes: list[str] = field(default_factory=list)


# ============================================================
# SLIDING WINDOW GEOMETRY
# ============================================================


def compute_starts(length: int, patch: int, overlap: float) -> list[int]:
    """
    Window start offsets along one axis, matching the notebook exactly.

    The trailing ``length - patch`` start is appended when the stride does not
    land on the end, so the final voxels are covered rather than silently
    dropped. That last window overlaps its neighbour by more than the nominal
    amount, which the count map handles.
    """
    if length <= patch:
        return [0]

    stride = max(1, int(patch * (1.0 - overlap)))
    starts = list(range(0, length - patch + 1, stride))

    last = length - patch
    if not starts or starts[-1] != last:
        starts.append(last)

    return starts


def _plan_windows(
    shape: tuple[int, int, int],
    patch: tuple[int, int, int],
    overlap: float,
    budget: int,
) -> tuple[tuple[slice, slice, slice], list[list[int]], int, int]:
    """
    Decide which part of the volume to analyse within a window budget.

    Returns the crop to analyse, the per-axis window starts *within that crop*,
    the number of windows that will run, and the number a full pass would have
    taken. When the full pass fits, the crop is the whole volume and the two
    counts are equal.

    Shrinking is proportional across all three axes and the crop is centred,
    because a cardiac CT has the heart near the middle of the field of view.
    This is a heuristic about framing, not about anatomy, which is why the
    analysed region is reported rather than assumed.
    """
    full_starts = [
        compute_starts(shape[axis], patch[axis], overlap) for axis in range(3)
    ]
    total = len(full_starts[0]) * len(full_starts[1]) * len(full_starts[2])

    if total <= budget:
        return (
            (slice(0, shape[0]), slice(0, shape[1]), slice(0, shape[2])),
            full_starts,
            total,
            total,
        )

    scale = (budget / total) ** (1.0 / 3.0)

    crop: list[slice] = []
    kept_starts: list[list[int]] = []

    for axis in range(3):
        stride = max(1, int(patch[axis] * (1.0 - overlap)))
        wanted = max(1, int(len(full_starts[axis]) * scale))
        extent = min(shape[axis], patch[axis] + (wanted - 1) * stride)
        origin = max(0, (shape[axis] - extent) // 2)
        crop.append(slice(origin, origin + extent))
        kept_starts.append(compute_starts(extent, patch[axis], overlap))

    run = len(kept_starts[0]) * len(kept_starts[1]) * len(kept_starts[2])

    return (crop[0], crop[1], crop[2]), kept_starts, run, total


# ============================================================
# SEGMENTER
# ============================================================


class CctaSegmenter:
    """Lazily-loaded singleton wrapper around the trained CCTA model."""

    def __init__(self) -> None:
        self._model: Optional[Small3DUNet] = None
        self._checkpoint_meta: dict[str, Any] = {}
        self._threshold: float = CCTA_THRESHOLD
        self._load_error: Optional[str] = None
        # Grad-CAM installs hooks and calls zero_grad, both of which are shared
        # mutable state, and FastAPI runs sync endpoints in a threadpool. One
        # volume at a time.
        self._lock = threading.Lock()

    # ---- state ------------------------------------------------

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    @property
    def load_error(self) -> Optional[str]:
        return self._load_error

    @property
    def threshold(self) -> float:
        return self._threshold

    # ---- loading ----------------------------------------------

    def load(self) -> None:
        """Build the architecture and load the trained weights."""
        if self._model is not None:
            return

        if not CCTA_CHECKPOINT_PATH.exists():
            self._load_error = (
                f"CCTA checkpoint not found at {CCTA_CHECKPOINT_PATH}. "
                "Expected best_3d_unet_cca_v2.pth (~17 MB) — if you cloned this "
                "repo, the file is tracked with Git LFS, so run: git lfs pull"
            )
            raise CctaModelUnavailable(self._load_error)

        size = CCTA_CHECKPOINT_PATH.stat().st_size
        if size < 1_000_000:
            self._load_error = (
                f"CCTA checkpoint at {CCTA_CHECKPOINT_PATH} is only {size} "
                "bytes, which looks like an unresolved Git LFS pointer rather "
                "than real weights. Run: git lfs pull"
            )
            raise CctaModelUnavailable(self._load_error)

        print("=" * 60)
        print("Loading CardioVision CCTA segmentation model...")
        print(f"Checkpoint: {CCTA_CHECKPOINT_PATH.name}")
        print(f"Device: {DEVICE}")

        try:
            checkpoint = torch.load(
                CCTA_CHECKPOINT_PATH, map_location="cpu", weights_only=False
            )

            if not isinstance(checkpoint, dict):
                raise CctaModelUnavailable(
                    "Unexpected checkpoint format: expected a dict saved by "
                    "torch.save with a 'model_state_dict' key."
                )

            state_dict = checkpoint.get("model_state_dict", checkpoint)

            config = checkpoint.get("config") or {}
            base = int(config.get("base_channels", CCTA_BASE_CHANNELS))

            model = Small3DUNet(
                in_channels=CCTA_IN_CHANNELS,
                out_channels=CCTA_OUT_CHANNELS,
                base=base,
            )

            # strict=True on purpose. A partial load would produce a mask that
            # looks like a coronary tree and means nothing.
            model.load_state_dict(state_dict, strict=True)

            parameters = sum(p.numel() for p in model.parameters())
            if base == CCTA_BASE_CHANNELS and parameters != CCTA_PARAMETERS:
                # strict=True already caught any missing or extra tensor, so
                # this is the belt to that braces: it catches a same-named
                # tensor of a different size, which load_state_dict reports but
                # which is easy to lose in a long error.
                raise CctaModelUnavailable(
                    f"Loaded {parameters:,} parameters but this architecture "
                    f"should have {CCTA_PARAMETERS:,}. The checkpoint does not "
                    "match the model definition in this file."
                )

        except CctaModelUnavailable:
            raise
        except Exception as error:
            self._load_error = f"Failed to load the CCTA model: {error}"
            raise CctaModelUnavailable(self._load_error) from error

        model.eval()
        model.to(DEVICE)

        self._model = model

        # The operating point that produced every metric in CCTA_TEST_METRICS.
        # Read from the weights, not written down twice.
        selected = checkpoint.get("selected_threshold")
        try:
            self._threshold = float(selected)
        except (TypeError, ValueError):
            self._threshold = CCTA_THRESHOLD

        if not (0.0 < self._threshold < 1.0):
            self._threshold = CCTA_THRESHOLD

        self._checkpoint_meta = {
            "epoch": checkpoint.get("epoch"),
            "best_validation_dice": checkpoint.get("best_val_dice"),
            "selected_threshold": self._threshold,
            "threshold_from_checkpoint": selected is not None,
            "base_channels": base,
            "patch_size": list(config.get("patch_size", CCTA_PATCH_SIZE)),
            "target_spacing_mm": list(
                config.get("target_spacing", CCTA_TARGET_SPACING)
            ),
            "parameters": parameters,
        }

        self._load_error = None

        print(
            "CCTA model loaded. "
            f"epoch={self._checkpoint_meta['epoch']} "
            f"val_dice={self._checkpoint_meta['best_validation_dice']} "
            f"threshold={self._threshold:.2f} "
            f"params={parameters:,}"
        )
        print("=" * 60)

    # ---- metadata ---------------------------------------------

    def describe(self) -> dict[str, Any]:
        """Model card: what these weights are, and how little was measured."""
        meta = dict(self._checkpoint_meta)
        dice = CCTA_TEST_METRICS["dice"]

        best_val = meta.get("best_validation_dice")

        return {
            "architecture": CCTA_ARCHITECTURE,
            "task": "Binary coronary artery lumen segmentation",
            "parameters": meta.get("parameters", CCTA_PARAMETERS),
            "in_channels": CCTA_IN_CHANNELS,
            "out_channels": CCTA_OUT_CHANNELS,
            "class_names": CCTA_CLASS_NAMES,
            "best_epoch": meta.get("epoch", CCTA_TEST_METRICS["best_epoch"]),
            "input": {
                "target_spacing_mm": meta.get(
                    "target_spacing_mm", list(CCTA_TARGET_SPACING)
                ),
                "patch_size": meta.get("patch_size", list(CCTA_PATCH_SIZE)),
                "overlap": CCTA_INFERENCE_OVERLAP,
                "hu_window": [-1000.0, 1000.0],
                "normalization": "HU clipped to [-1000, 1000], scaled to [-1, 1]",
            },
            "threshold": {
                "value": self._threshold if self.is_loaded else CCTA_THRESHOLD,
                "from_checkpoint": bool(meta.get("threshold_from_checkpoint")),
                "note": (
                    "Selected on the validation split, never on the test "
                    "split. Every metric below is measured at this threshold."
                ),
            },
            "metrics": {
                "validation_dice": (
                    round(float(best_val), 4) if best_val is not None else None
                ),
                "test_dice": dice,
                "test_iou": CCTA_TEST_METRICS["iou"],
                "test_sensitivity": CCTA_TEST_METRICS["sensitivity"],
                "test_precision": CCTA_TEST_METRICS["precision"],
                "test_hd95_mm": CCTA_TEST_METRICS["hd95_mm"],
                "dataset": CCTA_TEST_METRICS["dataset"],
                "dataset_cases": CCTA_TEST_METRICS["dataset_cases"],
                "split": CCTA_TEST_METRICS["split"],
                "test_cases": CCTA_TEST_METRICS["test_cases"],
                "source": CCTA_TEST_METRICS["source"],
                "scope": "dataset-level, held-out test split of 3 cases",
                # The single most important qualifier on this model, placed
                # inside the metrics block so it cannot be read without it.
                "caveat": (
                    f"n={CCTA_TEST_METRICS['test_cases']}. Three cases support "
                    "no confidence interval and no claim of generalisation. "
                    "The spread shown is the range of three numbers."
                ),
            },
            "explainability": {
                "method": "3-D Grad-CAM",
                "target_layer": CCTA_GRADCAM_LAYER,
                "objective": "mean sigmoid foreground probability",
                "scope": (
                    "One 96x96x96 patch — the patch containing the most "
                    "predicted lumen. A full-volume CAM would need one backward "
                    "pass per window."
                ),
            },
            "limitations": list(CCTA_WEAK_NOTES),
            "device": DEVICE,
        }

    # ---- inference --------------------------------------------

    def analyze(
        self,
        volume: np.ndarray,
        spacing_mm: tuple[float, float, float] = CCTA_TARGET_SPACING,
        max_windows: Optional[int] = None,
        include_gradcam: bool = True,
    ) -> CctaAnalysis:
        """
        Segment one preprocessed CCTA volume.

        ``volume`` must already be 1 mm isotropic and scaled to [-1, 1] by
        ``preprocessing.ccta_io``. Passing raw Hounsfield units here would not
        raise — it would just produce a meaningless mask — which is why the
        loader and this method are always used as a pair.

        ``max_windows`` bounds the sliding-window pass. It is a parameter rather
        than a constant because the trade it makes is the caller's to make: a
        whole-chest study at 1 mm needs several hundred windows, and the choice
        between waiting minutes for full coverage and getting a centred crop in
        seconds depends on why the volume is being run. Whatever is skipped is
        reported as unanalysed either way.
        """
        if self._model is None:
            raise CctaModelUnavailable(
                self._load_error or "The CCTA model is not loaded."
            )

        if volume.ndim != 3:
            raise CctaModelUnavailable(
                f"Expected a 3-D volume, got shape {volume.shape}."
            )

        budget = CCTA_MAX_WINDOWS if max_windows is None else max(1, int(max_windows))

        notes: list[str] = []
        patch = tuple(int(p) for p in CCTA_PATCH_SIZE)
        original_shape = tuple(int(d) for d in volume.shape)

        working = np.ascontiguousarray(volume, dtype=np.float32)

        # Pad any axis smaller than the patch. -1.0 is the normalised value of
        # air, so padding is indistinguishable from the space around the
        # patient and contributes no edge artefact.
        pad_widths = [
            (0, max(0, patch[axis] - original_shape[axis])) for axis in range(3)
        ]
        if any(width[1] for width in pad_widths):
            working = np.pad(
                working, pad_widths, mode="constant",
                constant_values=float(CCTA_PAD_VALUE),
            )
            notes.append(
                f"Volume was smaller than the {patch[0]}^3 patch on at least "
                f"one axis; padded with air to {'x'.join(str(d) for d in working.shape)}."
            )

        padded_shape = tuple(int(d) for d in working.shape)

        crop, starts, windows_run, windows_total = _plan_windows(
            padded_shape, patch, CCTA_INFERENCE_OVERLAP, budget
        )

        if windows_run < windows_total:
            notes.append(
                f"A full pass over this volume is {windows_total} sliding "
                f"windows, above the {budget}-window budget. "
                f"{windows_run} windows were run over a centred region of "
                f"{'x'.join(str(crop[a].stop - crop[a].start) for a in range(3))} "
                "voxels. Everything outside that region is reported as "
                "UNANALYSED, not as background — the model has not looked at it."
            )

        region = working[crop]

        with self._lock:
            started = time.perf_counter()

            compute_device = DEVICE
            try:
                probability_region = self._sliding_window(region, starts, DEVICE)
            except (RuntimeError, NotImplementedError) as error:
                notes.append(
                    f"Inference failed on {DEVICE} ({error}); retried on CPU."
                )
                try:
                    self._model.to("cpu")
                    probability_region = self._sliding_window(region, starts, "cpu")
                    compute_device = "cpu"
                finally:
                    self._model.to(DEVICE)

            inference_ms = (time.perf_counter() - started) * 1000.0

        threshold = self._threshold
        mask_region = (probability_region >= threshold).astype(np.uint8)

        # Grad-CAM on the patch with the most predicted lumen. Outside the
        # lock's inference timing because it is a separate, optional pass.
        gradcam = None
        gradcam_origin = None
        gradcam_shape = None
        gradcam_device: Optional[str] = None

        if not include_gradcam:
            notes.append(
                "Grad-CAM was not requested, so no attention map was computed "
                "for this volume. The segmentation is unaffected."
            )
        else:
            with self._lock:
                try:
                    gradcam, gradcam_origin = self._gradcam(
                        region, mask_region, probability_region, starts,
                        compute_device,
                    )
                    gradcam_device = compute_device
                except (RuntimeError, NotImplementedError) as error:
                    notes.append(
                        f"Grad-CAM failed on {compute_device} ({error}); "
                        "retried on CPU."
                    )
                    try:
                        self._model.to("cpu")
                        gradcam, gradcam_origin = self._gradcam(
                            region, mask_region, probability_region, starts,
                            "cpu",
                        )
                        gradcam_device = "cpu"
                    except (RuntimeError, NotImplementedError) as retry_error:
                        notes.append(
                            f"Grad-CAM was unavailable ({retry_error}). The "
                            "segmentation itself is unaffected."
                        )
                        gradcam = None
                        gradcam_origin = None
                    finally:
                        self._model.to(DEVICE)

        if gradcam is not None:
            gradcam_shape = tuple(int(d) for d in gradcam.shape)
            if gradcam_origin is not None:
                # Report the offset in the original volume's coordinates, which
                # is what the figures and any exported NIfTI are indexed by.
                gradcam_origin = tuple(
                    int(gradcam_origin[axis] + crop[axis].start) for axis in range(3)
                )

        # Assemble full-volume outputs. Anything outside the analysed region is
        # zero in the mask AND false in `analysed`, and the two must be read
        # together: mask==0 alone does not mean "no lumen".
        probability = np.zeros(padded_shape, dtype=np.float32)
        probability[crop] = probability_region

        mask = np.zeros(padded_shape, dtype=np.uint8)
        mask[crop] = mask_region

        analysed = np.zeros(padded_shape, dtype=bool)
        analysed[crop] = True

        # Undo the padding so every array matches the volume the caller sent.
        unpad = tuple(slice(0, original_shape[axis]) for axis in range(3))
        probability = probability[unpad]
        mask = mask[unpad]
        analysed = analysed[unpad]

        analysed_voxels = int(analysed.sum())
        coverage = analysed_voxels / float(mask.size) if mask.size else 0.0

        findings = self._quantify(
            mask=mask,
            probability=probability,
            analysed_voxels=analysed_voxels,
            spacing_mm=spacing_mm,
            notes=notes,
        )

        return CctaAnalysis(
            mask=mask,
            analysed=analysed,
            probability=probability,
            gradcam=gradcam,
            gradcam_origin=gradcam_origin,
            gradcam_shape=gradcam_shape,
            findings=findings,
            threshold=threshold,
            windows_run=windows_run,
            windows_total=windows_total,
            coverage=coverage,
            inference_ms=inference_ms,
            compute_device=compute_device,
            gradcam_available=gradcam is not None,
            gradcam_device=gradcam_device,
            notes=notes,
        )

    # ---- sliding window ---------------------------------------

    def _sliding_window(
        self,
        region: np.ndarray,
        starts: list[list[int]],
        device: str,
    ) -> np.ndarray:
        """
        Accumulate patch probabilities and divide by the overlap count.

        The count map is uint8: at 50% overlap a voxel is covered at most 2
        times per axis, 3 where the trailing window lands, so 27 is the ceiling
        and 255 is ample. Using float32 here would cost another 200 MB on a
        full study for no benefit.
        """
        model = self._model
        assert model is not None

        patch = tuple(int(p) for p in CCTA_PATCH_SIZE)
        shape = tuple(int(d) for d in region.shape)

        accumulator = np.zeros(shape, dtype=np.float32)
        counts = np.zeros(shape, dtype=np.uint8)

        offsets = [
            (z, y, x)
            for z in starts[0]
            for y in starts[1]
            for x in starts[2]
        ]

        batch_size = max(1, int(CCTA_INFERENCE_BATCH_SIZE))

        with torch.inference_mode():
            for index in range(0, len(offsets), batch_size):
                chunk = offsets[index:index + batch_size]

                patches = np.stack([
                    region[
                        z:z + patch[0],
                        y:y + patch[1],
                        x:x + patch[2],
                    ]
                    for z, y, x in chunk
                ])

                tensor = torch.from_numpy(patches).unsqueeze(1).to(device)
                logits = model(tensor)
                probabilities = torch.sigmoid(logits.float())[:, 0]
                result = probabilities.to("cpu").numpy()

                for position, (z, y, x) in enumerate(chunk):
                    window = (
                        slice(z, z + patch[0]),
                        slice(y, y + patch[1]),
                        slice(x, x + patch[2]),
                    )
                    accumulator[window] += result[position]
                    counts[window] += 1

        # Every voxel in the region is covered by construction, but dividing by
        # a clipped count keeps this safe if the geometry ever changes.
        np.divide(
            accumulator,
            np.maximum(counts, 1).astype(np.float32),
            out=accumulator,
        )

        return accumulator

    # ---- explainability ---------------------------------------

    def _gradcam(
        self,
        region: np.ndarray,
        mask_region: np.ndarray,
        probability_region: np.ndarray,
        starts: list[list[int]],
        device: str,
    ) -> tuple[Optional[np.ndarray], Optional[tuple[int, int, int]]]:
        """
        3-D Grad-CAM for the single most informative patch.

        The objective is the mean sigmoid probability rather than a sum over
        thresholded voxels: it stays differentiable and stable when the
        prediction is sparse, which for a structure occupying 0.1% of the volume
        it almost always is. This matches the notebook.
        """
        model = self._model
        assert model is not None

        patch = tuple(int(p) for p in CCTA_PATCH_SIZE)

        origin = self._best_patch(mask_region, probability_region, starts, patch)
        if origin is None:
            return None, None

        z, y, x = origin
        window = region[z:z + patch[0], y:y + patch[1], x:x + patch[2]]

        target_layer = model.enc3.block[-1]

        activations: dict[str, torch.Tensor] = {}
        gradients: dict[str, torch.Tensor] = {}

        def forward_hook(_module, _inputs, output) -> None:
            activations["value"] = output

        def backward_hook(_module, _grad_input, grad_output) -> None:
            gradients["value"] = grad_output[0]

        handles = [
            target_layer.register_forward_hook(forward_hook),
            target_layer.register_full_backward_hook(backward_hook),
        ]

        # The forward pass may have fallen back to CPU while the model was
        # moved back to DEVICE afterwards, so put it where this pass expects it
        # rather than trusting that it is already there.
        model.to(device)

        try:
            tensor = (
                torch.from_numpy(np.ascontiguousarray(window, dtype=np.float32))
                .unsqueeze(0)
                .unsqueeze(0)
                .to(device)
            )

            model.zero_grad(set_to_none=True)

            # Deliberately NOT inference_mode: this pass needs a graph.
            logits = model(tensor)
            objective = torch.sigmoid(logits.float()).mean()
            objective.backward()

            if "value" not in activations or "value" not in gradients:
                return None, None

            activation = activations["value"].detach()
            gradient = gradients["value"].detach()

            weights = gradient.mean(dim=(2, 3, 4), keepdim=True)
            cam = (weights * activation).sum(dim=1, keepdim=True)
            cam = F.relu(cam)

            cam = F.interpolate(
                cam, size=patch, mode="trilinear", align_corners=False
            )

            values = cam[0, 0].to("cpu").numpy().astype(np.float32)
            lowest = float(values.min())
            highest = float(values.max())
            values = (values - lowest) / (highest - lowest + 1e-8)

            return values, (z, y, x)
        finally:
            for handle in handles:
                handle.remove()
            model.zero_grad(set_to_none=True)

    @staticmethod
    def _best_patch(
        mask_region: np.ndarray,
        probability_region: np.ndarray,
        starts: list[list[int]],
        patch: tuple[int, int, int],
    ) -> Optional[tuple[int, int, int]]:
        """
        The patch worth explaining: the one with the most predicted lumen.

        Falls back to highest total probability when the mask is empty, so an
        all-negative volume still gets a CAM showing where the model came
        closest — which is more useful than no explanation at all.
        """
        offsets = [
            (z, y, x)
            for z in starts[0]
            for y in starts[1]
            for x in starts[2]
        ]
        if not offsets:
            return None

        best = None
        best_score = -1.0

        for z, y, x in offsets:
            window = (
                slice(z, z + patch[0]),
                slice(y, y + patch[1]),
                slice(x, x + patch[2]),
            )
            score = float(mask_region[window].sum())
            if score == 0.0:
                score = float(probability_region[window].sum()) * 1e-6

            if score > best_score:
                best_score = score
                best = (z, y, x)

        return best

    # ---- quantification ---------------------------------------

    @staticmethod
    def _quantify(
        mask: np.ndarray,
        probability: np.ndarray,
        analysed_voxels: int,
        spacing_mm: tuple[float, float, float],
        notes: list[str],
    ) -> list[LumenFinding]:
        """
        Turn the mask into measurements, and only measurements.

        Volume is exact arithmetic on voxel size. Component count is reported
        because a coronary tree should be a small number of connected
        structures: a mask fragmented into hundreds of pieces is the visible
        form of the HD95 problem in this model's metrics, and the reader should
        be able to see it rather than infer it.
        """
        selected = mask.astype(bool)
        voxels = int(selected.sum())

        voxel_ml = float(
            spacing_mm[0] * spacing_mm[1] * spacing_mm[2]
        ) / 1000.0

        if voxels:
            values = probability[selected]
            mean_probability = float(values.mean())
            max_probability = float(probability.max())
        else:
            mean_probability = 0.0
            max_probability = float(probability.max()) if probability.size else 0.0

        components: Optional[int] = None
        largest_fraction: Optional[float] = None

        if voxels:
            try:
                from scipy.ndimage import label

                labelled, count = label(selected)
                components = int(count)
                if count:
                    sizes = np.bincount(labelled.ravel())[1:]
                    largest_fraction = float(sizes.max()) / float(voxels)
            except ImportError:
                notes.append(
                    "scipy is not installed, so the mask's connected-component "
                    "count could not be computed. Fragmentation is the most "
                    "direct sign of an unreliable mask in this model, so "
                    "install scipy to see it."
                )

        return [
            LumenFinding(
                name=CCTA_CLASS_NAMES[1],
                present=voxels >= CCTA_PRESENCE_THRESHOLD_VOXELS,
                voxels=voxels,
                volume_ml=voxels * voxel_ml,
                fraction_of_analysed=(
                    voxels / float(analysed_voxels) if analysed_voxels else 0.0
                ),
                mean_probability=mean_probability,
                max_probability=max_probability,
                components=components,
                largest_component_fraction=largest_fraction,
            )
        ]


# Module-level singleton.
ccta_segmenter = CctaSegmenter()
