#!/usr/bin/env python3
"""
CCTA preprocessing, sliding-window geometry and quantification.

The CCTA path is the weakest of the three models (Dice 0.60 on three test
cases), which makes the arithmetic around it more important, not less: if the
window grid, the voxel volume or the presence threshold is wrong, the number on
screen is wrong in a way no one can see. Everything checked here is exact
arithmetic or pure geometry, so it can be verified without a forward pass.

    python3 tests/test_ccta_pipeline.py

No torch needed. tests/torch_stub.py stands in only so the inference module can
be imported for its geometry helpers and its model card.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(HERE))

import torch_stub  # noqa: E402

STUBBED_TORCH = torch_stub.install()

from cardiovision.config import (  # noqa: E402
    CCTA_ARCHITECTURE,
    CCTA_CLASS_NAMES,
    CCTA_HU_MAX,
    CCTA_HU_MIN,
    CCTA_INFERENCE_OVERLAP,
    CCTA_IN_CHANNELS,
    CCTA_OUT_CHANNELS,
    CCTA_PATCH_SIZE,
    CCTA_PRESENCE_THRESHOLD_VOXELS,
    CCTA_TARGET_SPACING,
    CCTA_TEST_METRICS,
    CCTA_THRESHOLD,
    MODALITY_STATUS,
)
from cardiovision.inference.ccta import (  # noqa: E402
    CctaModelUnavailable,
    CctaSegmenter,
    LumenFinding,
    compute_starts,
    _plan_windows,
)
from cardiovision.preprocessing.ccta_io import (  # noqa: E402
    LoadedVolume,
    UnsupportedVolumeError,
    detect_volume_format,
    resampled_shape,
    _window_and_scale,
)
from cardiovision.rendering.ccta import _best_index, slice_labels  # noqa: E402

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
print("\n=== 1. Format detection prefers content over filename ===")

NIFTI_LE = (348).to_bytes(4, "little") + b"\x00" * 400
NIFTI_BE = (348).to_bytes(4, "big") + b"\x00" * 400
ZIP = b"PK\x03\x04" + b"\x00" * 64
GZIP = b"\x1f\x8b" + b"\x00" * 64
DICOM = b"\x00" * 128 + b"DICM" + b"\x00" * 64

check("a zip is a zip whatever it is called",
      detect_volume_format("study.nii.gz", ZIP) == "zip",
      "a .zip of DICOMs renamed .nii.gz is a common bad export")
check("gzip magic reads as NIfTI",
      detect_volume_format("anything.bin", GZIP) == "nifti")
check("a little-endian 348 header reads as NIfTI",
      detect_volume_format("anything.bin", NIFTI_LE) == "nifti")
check("a big-endian 348 header reads as NIfTI too",
      detect_volume_format("anything.bin", NIFTI_BE) == "nifti")
check("DICM at byte 128 reads as DICOM",
      detect_volume_format("slice.txt", DICOM) == "dicom")
check("magic bytes beat a contradicting suffix",
      detect_volume_format("series.dcm", NIFTI_LE) == "nifti")

PLAIN = b"\x99" * 512
check("suffix is the fallback, not the first test",
      detect_volume_format("volume.nii", PLAIN) == "nifti"
      and detect_volume_format("series.zip", PLAIN) == "zip"
      and detect_volume_format("image.dcm", PLAIN) == "dicom")

try:
    detect_volume_format("mystery.bin", PLAIN)
    unknown_refused = False
    unknown_message = ""
except UnsupportedVolumeError as error:
    unknown_refused = True
    unknown_message = str(error)
check("an unrecognisable file is refused rather than guessed at",
      unknown_refused, unknown_message[:56])
check("and the refusal names the formats that would work",
      ".nii" in unknown_message and "zip" in unknown_message)


# ============================================================
print("\n=== 2. Resampling geometry ===")

check("already-isotropic volumes come back unchanged",
      resampled_shape((128, 128, 64), (1.0, 1.0, 1.0)) == (128, 128, 64))
check("a 0.5 mm in-plane scan halves in those axes",
      resampled_shape((256, 256, 60), (0.5, 0.5, 1.0)) == (128, 128, 60),
      "256 voxels of 0.5 mm is 128 mm, which is 128 voxels at 1 mm")
check("a 3 mm slice gap triples along the slice axis",
      resampled_shape((512, 512, 40), (0.7, 0.7, 3.0)) == (358, 358, 120),
      "round, not ceil — matching the training notebook")
check("rounding is round-half-to-even, as the notebook's round() is",
      resampled_shape((5, 5, 5), (1.5, 1.5, 1.5)) == (8, 8, 8))
check("no axis can collapse to zero",
      resampled_shape((1, 1, 1), (0.01, 0.01, 0.01)) == (1, 1, 1))
check("the declared target spacing is 1 mm isotropic",
      tuple(CCTA_TARGET_SPACING) == (1.0, 1.0, 1.0))


# ============================================================
print("\n=== 3. HU windowing to the trained range ===")

raw = np.array(
    [[-3000.0, CCTA_HU_MIN, 0.0, CCTA_HU_MAX, 5000.0]],
    dtype=np.float32,
)
scaled = _window_and_scale(raw)

check("the HU floor maps to -1", float(scaled[0, 1]) == -1.0)
check("the HU ceiling maps to +1", float(scaled[0, 3]) == 1.0)
check("0 HU lands mid-range", abs(float(scaled[0, 2])) < 1e-6)
check("values below the floor are clipped, not wrapped",
      float(scaled[0, 0]) == -1.0)
check("values above the ceiling are clipped too",
      float(scaled[0, 4]) == 1.0)
check("the input array is not modified in place",
      float(raw[0, 2]) == 0.0,
      "the caller may still need the HU volume for display")

nan_input = np.array([[np.nan, np.inf, -np.inf]], dtype=np.float32)
nan_scaled = _window_and_scale(nan_input)
check("NaN becomes the HU floor rather than propagating",
      float(nan_scaled[0, 0]) == -1.0,
      "one NaN voxel would otherwise poison a whole patch")
check("+inf clips to +1 and -inf to -1",
      float(nan_scaled[0, 1]) == 1.0 and float(nan_scaled[0, 2]) == -1.0)
check("nothing survives outside [-1, 1]",
      float(nan_scaled.min()) >= -1.0 and float(nan_scaled.max()) <= 1.0)
check("the output is float32, matching the weights",
      scaled.dtype == np.float32)


# ============================================================
print("\n=== 4. Sliding-window start offsets ===")

check("a volume smaller than the patch gets one window at 0",
      compute_starts(64, 96, 0.5) == [0])
check("a volume exactly the patch size gets one window",
      compute_starts(96, 96, 0.5) == [0])
starts = compute_starts(200, 96, 0.5)
check("50% overlap gives a stride of half the patch",
      starts[:3] == [0, 48, 96], str(starts))
check("the trailing window is appended so the end is covered",
      starts[-1] == 200 - 96,
      "otherwise the last voxels are silently never looked at")
check("no window runs off the end",
      all(s + 96 <= 200 for s in starts))
check("starts are strictly increasing",
      all(b > a for a, b in zip(starts, starts[1:])))
check("an exact fit does not duplicate the final start",
      compute_starts(192, 96, 0.5) == [0, 48, 96],
      str(compute_starts(192, 96, 0.5)))
check("zero overlap strides by the full patch",
      compute_starts(300, 100, 0.0) == [0, 100, 200])
check("the declared inference overlap is 50%",
      CCTA_INFERENCE_OVERLAP == 0.50)
check("the declared patch is 96 cubed",
      tuple(CCTA_PATCH_SIZE) == (96, 96, 96))


# ============================================================
print("\n=== 5. Window budget and partial coverage ===")

SHAPE = (300, 300, 300)
crop, kept, run, total = _plan_windows(SHAPE, (96, 96, 96), 0.5, 10_000)
check("a pass that fits the budget analyses the whole volume",
      crop == (slice(0, 300), slice(0, 300), slice(0, 300)))
check("and reports run == total",
      run == total and run > 0, f"{run} windows")

crop, kept, run, total = _plan_windows(SHAPE, (96, 96, 96), 0.5, 8)
check("an over-budget pass crops instead of skipping windows",
      crop != (slice(0, 300), slice(0, 300), slice(0, 300)))
check("the truncated run is honestly smaller than the full pass",
      run < total, f"{run} of {total}")
check("the run stays within the budget",
      run <= 8, f"{run} <= 8")
check("the crop is centred, because the heart is mid-field-of-view",
      all(
          abs(c.start - (SHAPE[i] - (c.stop - c.start)) // 2) <= 1
          for i, c in enumerate(crop)
      ))
check("the crop never leaves the volume",
      all(c.start >= 0 and c.stop <= SHAPE[i] for i, c in enumerate(crop)))
check("window starts are planned within the crop, not the full volume",
      all(
          kept[i][-1] + 96 <= (crop[i].stop - crop[i].start)
          for i in range(3)
      ))
check("a budget of one still produces one runnable window",
      _plan_windows(SHAPE, (96, 96, 96), 0.5, 1)[2] >= 1)


# ============================================================
print("\n=== 6. Quantification is measurement only ===")

quantify = CctaSegmenter._quantify
probability = np.zeros((20, 20, 20), dtype=np.float32)
mask = np.zeros((20, 20, 20), dtype=np.uint8)
mask[:10, :10, :10] = 1              # 1000 voxels
probability[mask.astype(bool)] = 0.9
probability[15, 15, 15] = 0.95       # a high probability outside the mask
notes: list[str] = []

finding = quantify(mask, probability, 8000, (1.0, 1.0, 1.0), notes)[0]

check("one finding is produced, the lumen class",
      len(quantify(mask, probability, 8000, (1.0, 1.0, 1.0), [])) == 1
      and finding.name == CCTA_CLASS_NAMES[1], finding.name)
check("the voxel count is exact", finding.voxels == 1000)
check("volume is voxel count times voxel volume, in mL",
      abs(finding.volume_ml - 1.0) < 1e-9,
      "1000 voxels of 1 mm^3 = 1000 mm^3 = 1 mL")
check("the fraction is of the analysed region, not the whole volume",
      abs(finding.fraction_of_analysed - 0.125) < 1e-9,
      "1000 / 8000")
check("mean probability is taken over masked voxels only",
      abs(finding.mean_probability - 0.9) < 1e-6)
check("max probability is over the whole map, including outside the mask",
      abs(finding.max_probability - 0.95) < 1e-6,
      "a near-threshold voxel outside the mask is still worth seeing")

anisotropic = quantify(mask, probability, 8000, (0.5, 0.5, 2.0), [])[0]
check("voxel volume follows the reported spacing, not an assumed 1 mm",
      abs(anisotropic.volume_ml - 0.5) < 1e-9,
      "0.5*0.5*2.0 = 0.5 mm^3 per voxel")

small = np.zeros((20, 20, 20), dtype=np.uint8)
small[:1, :10, :10] = 1              # 100 voxels, below the threshold
below = quantify(small, probability, 8000, (1.0, 1.0, 1.0), [])[0]
check("a mask below the presence threshold is reported as absent",
      below.present is False and below.voxels == 100,
      f"threshold is {CCTA_PRESENCE_THRESHOLD_VOXELS} voxels")
check("but its voxel count is still reported, not zeroed",
      below.voxels == 100,
      "absent means 'too little to call', not 'nothing there'")
check("a mask at the threshold is present",
      quantify(
          np.pad(
              np.ones(CCTA_PRESENCE_THRESHOLD_VOXELS, np.uint8),
              (0, 8000 - CCTA_PRESENCE_THRESHOLD_VOXELS),
          ).reshape(20, 20, 20),
          probability, 8000, (1.0, 1.0, 1.0), [],
      )[0].present is True)
check("the threshold is a declared constant, not a literal in the code",
      CCTA_PRESENCE_THRESHOLD_VOXELS == 500)

empty = quantify(
    np.zeros((20, 20, 20), np.uint8), probability, 8000, (1.0, 1.0, 1.0), []
)[0]
check("an empty mask gives zero volume and no crash",
      empty.voxels == 0 and empty.volume_ml == 0.0 and empty.present is False)
check("an empty mask still reports the map's max probability",
      abs(empty.max_probability - 0.95) < 1e-6,
      "'found nothing' and 'nearly found something' are different results")
check("zero analysed voxels give a zero fraction, not a division error",
      quantify(mask, probability, 0, (1.0, 1.0, 1.0), [])[0]
      .fraction_of_analysed == 0.0)

no_scipy_notes: list[str] = []
quantify(mask, probability, 8000, (1.0, 1.0, 1.0), no_scipy_notes)
if finding.components is None:
    check("without scipy the component count is None and a note says why",
          len(no_scipy_notes) == 1 and "scipy" in no_scipy_notes[0],
          no_scipy_notes[0][:52] if no_scipy_notes else "no note")
else:
    check("with scipy the single blob counts as one component",
          finding.components == 1
          and abs((finding.largest_component_fraction or 0) - 1.0) < 1e-6,
          f"components={finding.components}")


# ============================================================
print("\n=== 7. The finding dictionary the API returns ===")

payload = finding.to_dict()
EXPECTED_KEYS = {
    "name", "present", "voxels", "volume_ml", "fraction_of_analysed",
    "percent_of_analysed", "mean_probability", "max_probability",
    "components", "largest_component_fraction",
}
check("the payload has exactly the documented keys",
      set(payload) == EXPECTED_KEYS,
      str(sorted(set(payload) ^ EXPECTED_KEYS)) if set(payload) != EXPECTED_KEYS else "")
check("percent is the fraction times 100, both carried",
      abs(payload["percent_of_analysed"] - 12.5) < 1e-6
      and abs(payload["fraction_of_analysed"] - 0.125) < 1e-8)
check("presence is a real bool, not a truthy number",
      isinstance(payload["present"], bool))
check("voxels is an int the frontend can format",
      isinstance(payload["voxels"], int))
check("nothing in the payload names a file path or a patient",
      not any(
          isinstance(v, str) and ("/" in v or "\\" in v)
          for v in payload.values()
      ))
check("a None component count survives serialisation as null",
      LumenFinding("Lumen", False, 0, 0.0, 0.0, 0.0, 0.0).to_dict()["components"]
      is None)


# ============================================================
print("\n=== 8. Slice selection shows something worth looking at ===")

sparse_mask = np.zeros((30, 30, 30), dtype=np.uint8)
sparse_mask[7, 4:8, 4:8] = 1
sparse_prob = np.zeros((30, 30, 30), dtype=np.float32)
sparse_prob[sparse_mask.astype(bool)] = 0.8

check("the chosen slice is the one with the most predicted lumen",
      _best_index(sparse_mask, sparse_prob, 0) == 7,
      "not index 15, which is empty")

no_mask = np.zeros((30, 30, 30), dtype=np.uint8)
prob_only = np.zeros((30, 30, 30), dtype=np.float32)
prob_only[22] = 0.4
check("with an empty mask it falls back to peak probability",
      _best_index(no_mask, prob_only, 0) == 22,
      "a sub-threshold prediction is still the informative slice")
check("with nothing at all it falls back to the centre",
      _best_index(no_mask, np.zeros((30, 30, 30), np.float32), 0) == 15)

labels = slice_labels(sparse_mask, sparse_prob)
check("every plane gets an index reported",
      set(labels) == {"axis0", "axis1", "axis2"}, str(sorted(labels)))
check("planes are named by array axis, not by anatomy",
      not any(
          word in " ".join(labels)
          for word in ("axial", "coronal", "sagittal")
      ),
      "the loader never verifies patient orientation")
check("each index is inside its own axis",
      all(
          0 <= labels[name] < sparse_mask.shape[axis]
          for name, axis in (("axis0", 0), ("axis1", 1), ("axis2", 2))
      ))


# ============================================================
print("\n=== 9. The model card admits how little was measured ===")

card = CctaSegmenter().describe()

check("the architecture is the one that was trained",
      card["architecture"] == CCTA_ARCHITECTURE == "Small3DUNet")
check("the task is stated as lumen segmentation, nothing more",
      "lumen segmentation" in card["task"].lower(),
      card["task"])
check("one output channel — sigmoid, not softmax over classes",
      card["out_channels"] == CCTA_OUT_CHANNELS == 1
      and card["in_channels"] == CCTA_IN_CHANNELS == 1)

metrics = card["metrics"]
for name in ("test_dice", "test_iou", "test_sensitivity",
             "test_precision", "test_hd95_mm"):
    check(f"{name} is a spread, not a bare number",
          isinstance(metrics[name], dict)
          and {"mean", "sd", "min", "max"} <= set(metrics[name]),
          "a single float would read as a stable estimate")
check("validation dice is a single float, because it is one number",
      metrics["validation_dice"] is None
      or isinstance(metrics["validation_dice"], float))
check("the test split is three cases and says so",
      metrics["test_cases"] == 3, str(metrics["test_cases"]))
check("the n=3 caveat travels inside the metrics block",
      "n=3" in metrics["caveat"]
      and "no claim of generalisation" in metrics["caveat"])
check("the caveat calls the spread a range of three numbers",
      "range of three numbers" in metrics["caveat"])
check("the dataset is named", metrics["dataset"] == "MedHK23/CCA")
check("the split is described at case level",
      "case level" in metrics["split"], metrics["split"])
check("the threshold is reported with where it came from",
      card["threshold"]["value"] == CCTA_THRESHOLD
      and "validation split" in card["threshold"]["note"])
check("the threshold note rules out test-set tuning",
      "never on the test" in card["threshold"]["note"])
check("Grad-CAM's single-patch scope is stated, not implied",
      card["explainability"]["method"] == "3-D Grad-CAM"
      and "One 96x96x96 patch" in card["explainability"]["scope"])
check("the limitations list is non-empty",
      isinstance(card["limitations"], list) and len(card["limitations"]) > 0,
      f"{len(card['limitations'])} entries")
LIMITS = " ".join(card["limitations"]).lower()
check("the limitations disclaim stenosis grading",
      "stenosis" in LIMITS)
check("and disclaim calcium scoring",
      "calcium" in LIMITS)
check("and disclaim vessel labelling",
      "vessel" in LIMITS)


# ============================================================
print("\n=== 10. Declared status matches what exists ===")

check("CCTA is declared available, because a checkpoint exists",
      MODALITY_STATUS["ccta"]["available"] is True)
CCTA_NOTE = MODALITY_STATUS["ccta"]["note"]
check("its note carries the three-case Dice, not a bare 0.60",
      "0.60" in CCTA_NOTE and "THREE" in CCTA_NOTE.upper())
check("its note says the output is a lumen mask only",
      "lumen mask only" in CCTA_NOTE.lower())
check("clinical risk stays unavailable — no model was ever trained",
      MODALITY_STATUS["clinical"]["available"] is False)
check("fusion stays unavailable — the notebook is empty",
      MODALITY_STATUS["fusion"]["available"] is False)
check("the metrics constant and the model card agree on the dice mean",
      abs(CCTA_TEST_METRICS["dice"]["mean"] - metrics["test_dice"]["mean"])
      < 1e-9,
      "one source of truth, read twice")


# ============================================================
print("\n=== 11. An unloaded model refuses to answer ===")

fresh = CctaSegmenter()
check("a fresh segmenter reports itself unloaded",
      fresh.is_loaded is False)
try:
    fresh.analyze(np.zeros((32, 32, 32), np.float32))
    refused = False
    refusal = ""
except CctaModelUnavailable as error:
    refused = True
    refusal = str(error)
check("analyze() before load() raises rather than returning an empty mask",
      refused, refusal[:60])
check("the threshold falls back to the configured default when unloaded",
      fresh.threshold == CCTA_THRESHOLD)


# ============================================================
print("\n=== 12. The input summary is geometry, never identity ===")

loaded = LoadedVolume(
    volume=np.zeros((8, 8, 8), dtype=np.float32),
    affine=np.eye(4),
    source_format="nifti",
    original_shape=(16, 16, 4),
    original_spacing_mm=(0.5, 0.5, 2.0),
    spacing_mm=(1.0, 1.0, 1.0),
    resampled=True,
    hu_min_observed=-1024.0,
    hu_max_observed=1512.0,
    slices=4,
    notes=["resampled from 0.5x0.5x2.0 mm"],
)
summary = loaded.summary()

check("both the source and the analysed geometry are reported",
      summary["original_shape"] == [16, 16, 4]
      and summary["analysed_shape"] == [8, 8, 8],
      "the reader needs to know the model saw a resampled grid")
check("resampling is flagged explicitly", summary["resampled"] is True)
check("the observed HU range is reported beside the training window",
      summary["hu_range_observed"] == [-1024.0, 1512.0]
      and summary["hu_window"] == [CCTA_HU_MIN, CCTA_HU_MAX],
      "contrast above the window is visibly clipped")
check("the voxel count is derived from the analysed array",
      summary["voxels"] == 512)
check("no filesystem path, series UID or patient field is present",
      not {"path", "filename", "series_uid", "patient_name", "patient_id"}
      & set(summary),
      str(sorted(summary)))
check("no value in the summary looks like a path",
      not any(
          isinstance(v, str) and ("/" in v or "\\" in v)
          for v in summary.values()
      ))
check("notes are carried through for the UI to show",
      summary["notes"] == ["resampled from 0.5x0.5x2.0 mm"])


# ============================================================
print("\n" + "=" * 62)

if STUBBED_TORCH:
    print("NOTE: torch is not installed here, so no forward pass ran and no")
    print("      checkpoint was loaded. Everything above is real arithmetic")
    print("      and real geometry; the masks fed in were synthetic.")
    print("=" * 62)

if FAILURES:
    print(f"{len(FAILURES)} FAILURE(S) out of {CHECKS} checks:")
    for item in FAILURES:
        print(f"  - {item}")
    sys.exit(1)

print(f"ALL {CHECKS} CCTA PIPELINE CHECKS PASSED")
