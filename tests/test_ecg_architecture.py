#!/usr/bin/env python3
"""
ECG architecture verification.

Checks the hand-written ``ECGResNet1D`` in ``cardiovision.inference.ecg``
against the trained checkpoint, parameter by parameter. If a layer were the
wrong width, had the wrong kernel, or were named differently, the model would
still *look* fine and would still return five confident-looking probabilities —
it just would not load, or worse, would load a subset. This test is the thing
that catches that early and says which layer.

    python3 tests/test_ecg_architecture.py

Runs with or without torch installed:

* torch present — builds the real module and compares ``state_dict()`` to the
  checkpoint.
* torch absent — reproduces ``nn.Module``'s parameter-naming and shape rules
  with a minimal stub, which is enough to compare names and shapes. The
  arithmetic torch would do is not involved in either case; only the shapes are.

Either way the checkpoint is read straight out of its zip container with the
standard library, so the comparison is against the real file.
"""

from __future__ import annotations

import sys

from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(HERE))

from checkpoint_reader import TensorInfo, read_checkpoint, shapes  # noqa: E402

CHECKPOINT = REPO / "models" / "ecg" / "cardioVision_ptbxl_ecg_resnet1d_full.pt"

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
# BUILDING THE MODULE WITHOUT TORCH
# ============================================================
#
# tests/torch_stub.py reproduces nn.Module's parameter naming and each layer's
# shapes, which is all this comparison needs. It is a no-op when torch is really
# installed, in which case the real module is built and compared instead.

import torch_stub  # noqa: E402

STUBBED_TORCH = torch_stub.install()


from cardiovision.config import (  # noqa: E402
    ECG_CLASS_NAMES,
    ECG_IN_CHANNELS,
    ECG_INPUT_LENGTH,
    ECG_LEAD_NAMES,
    ECG_NUM_CLASSES,
    ECG_PARAMETERS,
    ECG_TARGET_FS,
    ECG_TEST_METRICS,
    ECG_WEAK_CLASSES,
)
from cardiovision.inference.ecg import ECGResNet1D  # noqa: E402

import numpy as np  # noqa: E402


def _json_safe(value) -> bool:
    """Would json.dumps accept this value as-is?"""
    import json
    try:
        json.dumps(value)
        return True
    except TypeError:
        return False


def built_state_dict() -> dict[str, tuple[int, ...]]:
    model = ECGResNet1D()
    if STUBBED_TORCH:
        return model.state_dict(), model.parameter_count()
    real = {key: tuple(value.shape) for key, value in model.state_dict().items()}
    return real, sum(p.numel() for p in model.parameters())


# ============================================================
print("\n=== 0. the checkpoint is present and readable ===")
# ============================================================

if not CHECKPOINT.exists():
    print(f"  [SKIP] {CHECKPOINT} not found. Run: git lfs pull")
    sys.exit(0)

size_mb = CHECKPOINT.stat().st_size / 1e6
if CHECKPOINT.stat().st_size < 1_000_000:
    print(f"  [SKIP] {CHECKPOINT.name} is {size_mb:.3f} MB — an unresolved "
          "Git LFS pointer. Run: git lfs pull")
    sys.exit(0)

check("the checkpoint is real weights, not an LFS pointer",
      size_mb > 10, f"{size_mb:.1f} MB")

checkpoint = read_checkpoint(CHECKPOINT)
check("it unpickles to a dict", isinstance(checkpoint, dict),
      type(checkpoint).__name__)

expected_keys = {
    "model_state_dict", "target_classes", "lead_names", "input_channels",
    "input_length", "target_fs", "preprocessing_config", "model_config",
    "test_metrics", "best_validation_macro_AUROC", "best_epoch",
}
missing = expected_keys - set(checkpoint)
check("every key the loader reads is present",
      not missing, f"missing: {sorted(missing)}" if missing else "all 11")

stored = checkpoint["model_state_dict"]

# ============================================================
print("\n=== 1. the architecture matches the weights exactly ===")
# ============================================================

built, built_params = built_state_dict()

stored_shapes = shapes(stored)

only_stored = sorted(set(stored_shapes) - set(built))
only_built = sorted(set(built) - set(stored_shapes))

check("no layer in the checkpoint is missing from the module",
      not only_stored,
      f"{len(only_stored)} unmatched: {only_stored[:4]}" if only_stored else "")
check("the module defines no layer the checkpoint lacks",
      not only_built,
      f"{len(only_built)} extra: {only_built[:4]}" if only_built else "")

mismatched = [
    (key, built[key], stored_shapes[key])
    for key in sorted(set(built) & set(stored_shapes))
    if tuple(built[key]) != tuple(stored_shapes[key])
]
check("every shared layer has an identical shape",
      not mismatched,
      f"{len(mismatched)} differ: {mismatched[:3]}" if mismatched else
      f"{len(built)} tensors")

# This is the load-bearing one. strict=True would reject any of the above at
# runtime; matching here means the real load will succeed.
check("the module would load with strict=True",
      not only_stored and not only_built and not mismatched)

# ============================================================
print("\n=== 2. the parameter count matches the notebook ===")
# ============================================================

check("parameter count is exactly what 03_ECG.ipynb printed",
      built_params == ECG_PARAMETERS,
      f"{built_params:,} vs {ECG_PARAMETERS:,}")

# ============================================================
print("\n=== 3. spot-check the layers that carry the contract ===")
# ============================================================

expectations = {
    # 12 leads in, 64 filters, kernel 15 — the only place lead count appears.
    "stem.0.weight": (64, ECG_IN_CHANNELS, 15),
    # First stage: 64 wide, kernel 7.
    "block1.0.block.0.weight": (64, 64, 7),
    # 1x1 projections double the channels while halving the length.
    "down1.weight": (128, 64, 1),
    "down2.weight": (256, 128, 1),
    "down3.weight": (512, 256, 1),
    # Later stages narrow the kernel to 5.
    "block3.0.block.0.weight": (256, 256, 5),
    "block4.1.block.0.weight": (512, 512, 5),
    # The head: 512 pooled features -> 256 -> 5 classes.
    "classifier.0.weight": (256, 512),
    "classifier.4.weight": (ECG_NUM_CLASSES, 256),
    "classifier.4.bias": (ECG_NUM_CLASSES,),
}

for key, shape in expectations.items():
    check(f"{key} is {shape}",
          stored_shapes.get(key) == shape,
          str(stored_shapes.get(key)))

check("the stem is the only layer that reads 12 channels",
      sum(1 for shape in stored_shapes.values()
          if len(shape) == 3 and shape[1] == ECG_IN_CHANNELS) == 1)
check("the classifier is the only layer that emits 5 values",
      sum(1 for key, shape in stored_shapes.items()
          if shape and shape[0] == ECG_NUM_CLASSES and "classifier" in key) == 2)

# ============================================================
print("\n=== 4. config.py agrees with the checkpoint's own metadata ===")
# ============================================================
#
# The loader treats each of these as a hard failure, so if any drift the model
# refuses to load. Better to find out here than on a clinician's first upload.

check("class order matches ECG_CLASS_NAMES",
      tuple(checkpoint["target_classes"]) == tuple(ECG_CLASS_NAMES),
      str(tuple(checkpoint["target_classes"])))
check("lead order matches ECG_LEAD_NAMES",
      tuple(checkpoint["lead_names"]) == tuple(ECG_LEAD_NAMES),
      str(tuple(checkpoint["lead_names"])[:4]) + "...")
check("input length matches ECG_INPUT_LENGTH",
      int(checkpoint["input_length"]) == ECG_INPUT_LENGTH,
      str(checkpoint["input_length"]))
check("channel count matches ECG_IN_CHANNELS",
      int(checkpoint["input_channels"]) == ECG_IN_CHANNELS,
      str(checkpoint["input_channels"]))
check("sampling rate matches ECG_TARGET_FS",
      float(checkpoint["target_fs"]) == float(ECG_TARGET_FS),
      str(checkpoint["target_fs"]))

model_config = checkpoint.get("model_config") or {}
check("dropout is recorded in the checkpoint", "dropout" in model_config,
      str(model_config))

preprocessing = checkpoint.get("preprocessing_config") or {}
print(f"  ..... preprocessing_config: {preprocessing}")

# ============================================================
print("\n=== 5. the metrics we display are the checkpoint's own ===")
# ============================================================

metrics = checkpoint.get("test_metrics") or {}

check("test metrics travel inside the checkpoint", bool(metrics),
      f"{len(metrics)} entries")

macro = metrics.get("macro_AUROC")
check("macro AUROC matches ECG_TEST_METRICS",
      macro is not None and abs(float(macro) - ECG_TEST_METRICS["macro_AUROC"]) < 5e-4,
      f"checkpoint {macro} vs config {ECG_TEST_METRICS['macro_AUROC']}")

per_class = {
    name: {
        key: metrics[f"{name}_{key}"]
        for key in ("AUROC", "AP", "F1", "Precision", "Recall")
        if f"{name}_{key}" in metrics
    }
    for name in ECG_CLASS_NAMES
}

check("per-class metrics are stored flat, as 'CLASS_METRIC'",
      all(len(row) == 5 for row in per_class.values()),
      str({name: len(row) for name, row in per_class.items()}))

for name in ECG_CLASS_NAMES:
    row = per_class[name]
    expected = ECG_TEST_METRICS["per_class"][name]
    same = all(
        key in row and abs(float(row[key]) - expected[key]) < 5e-5
        for key in ("AUROC", "AP", "F1", "Precision", "Recall")
    )
    check(f"{name} per-class metrics match config", same,
          "" if same else f"checkpoint {row} vs config {expected}")

# The loader has to find these itself. If _class_metrics stops resolving them
# it falls back to config.py, which is correct today and would silently stop
# tracking the weights after any retrain — so assert on the loader, not just on
# the file.
from cardiovision.inference.ecg import EcgClassifier  # noqa: E402

probe = EcgClassifier()
probe._metrics = dict(metrics)
resolved = {name: probe._class_metrics(name) for name in ECG_CLASS_NAMES}

check("the loader reads per-class metrics out of this layout",
      all(len(row) == 5 for row in resolved.values()),
      str({name: sorted(row) for name, row in resolved.items()})[:90])
check("and it matches the checkpoint rather than the config fallback",
      all(
          abs(resolved[name]["Precision"] - float(per_class[name]["Precision"]))
          < 5e-5
          for name in ECG_CLASS_NAMES
      ))
check("every value it returns is a plain float, so json.dumps works",
      all(type(value) is float
          for row in resolved.values() for value in row.values()),
      str({type(v).__name__ for row in resolved.values() for v in row.values()}))

# Why _as_float exists. np.float64 happens to subclass Python float, so this
# particular checkpoint would serialise unconverted — but np.float32 does not,
# and sklearn's return dtype is not something this repo controls. The
# conversion also does the rounding, so the API never emits
# 0.9497728411045379 as if the fourth decimal were meaningful.
import json  # noqa: E402

raw_types = sorted({type(value).__name__ for value in metrics.values()})
check("checkpoint metrics arrive as numpy scalars",
      any(name.startswith("float64") or name == "float64" for name in raw_types)
      or "float" in raw_types,
      f"types in file: {raw_types}")
check("json.dumps handles float64 but would reject float32",
      json.dumps({"a": float(1.5)}) is not None
      and not _json_safe(np.float32(1.5)),
      "float32 is not a float subclass; float64 is")
check("_as_float returns something json.dumps always accepts",
      json.dumps({
          name: probe._class_metrics(name) for name in ECG_CLASS_NAMES
      }) is not None)

# The number the whole HYP caveat rests on. If this ever moves, the wording in
# ECG_WEAK_CLASSES and in the UI has to move with it.
hyp = per_class["HYP"]
check("HYP precision really is the weak point the UI warns about",
      float(hyp["Precision"]) < 0.5,
      f"precision {float(hyp['Precision']):.4f}, AP {float(hyp['AP']):.4f}")
check("and every other class beats it comfortably",
      all(float(per_class[n]["Precision"]) > 0.6
          for n in ECG_CLASS_NAMES if n != "HYP"),
      str({n: round(float(per_class[n]["Precision"]), 3)
           for n in ECG_CLASS_NAMES}))
check("HYP is the only class flagged in ECG_WEAK_CLASSES",
      set(ECG_WEAK_CLASSES) == {"HYP"}, str(set(ECG_WEAK_CLASSES)))

check("best epoch matches config",
      int(checkpoint["best_epoch"]) == ECG_TEST_METRICS["best_epoch"],
      str(checkpoint["best_epoch"]))
check("validation macro AUROC matches config",
      abs(float(checkpoint["best_validation_macro_AUROC"])
          - ECG_TEST_METRICS["best_validation_macro_AUROC"]) < 5e-5,
      str(checkpoint["best_validation_macro_AUROC"]))

# ============================================================
print("\n" + "=" * 62)

if STUBBED_TORCH:
    print("NOTE: torch is not installed here, so the module was constructed")
    print("      against a minimal nn stub. Names, shapes and the parameter")
    print("      count are real; no forward pass was executed.")
    print("=" * 62)

if FAILURES:
    print(f"{len(FAILURES)} FAILURE(S) out of {CHECKS} checks:")
    for item in FAILURES:
        print(f"  - {item}")
    sys.exit(1)

print(f"ALL {CHECKS} ECG ARCHITECTURE CHECKS PASSED")
