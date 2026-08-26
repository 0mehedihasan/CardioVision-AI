"""
Read a ``.pt`` checkpoint without torch installed.

``torch.save`` writes a zip archive: ``data.pkl`` holds the pickled object graph
and each tensor's bytes live in a separate entry. Unpickling with a custom
``find_class`` recovers the tensor *shapes* and all the plain-Python metadata
while never allocating a tensor or importing torch.

Two uses:

* the architecture test compares checkpoint shapes against the module's, so a
  drifted layer is caught by name instead of as an opaque load failure;
* ``python3 tests/checkpoint_reader.py <path>`` dumps the metadata, which is how
  you find out what a checkpoint actually contains before writing code that
  reads it.

Only shapes and metadata come back. Nothing here can produce a prediction, and
it is not a substitute for loading the model properly.
"""

from __future__ import annotations

import pickle
import zipfile
from pathlib import Path
from typing import Any

_STORAGE_DTYPES = {
    "FloatStorage": "float32",
    "DoubleStorage": "float64",
    "HalfStorage": "float16",
    "BFloat16Storage": "bfloat16",
    "LongStorage": "int64",
    "IntStorage": "int32",
    "ShortStorage": "int16",
    "ByteStorage": "uint8",
    "CharStorage": "int8",
    "BoolStorage": "bool",
}


class TensorInfo:
    """Stands in for a tensor, carrying only its shape and dtype."""

    __slots__ = ("shape", "dtype")

    def __init__(self, shape, dtype: str) -> None:
        self.shape = tuple(int(dim) for dim in shape)
        self.dtype = dtype

    def numel(self) -> int:
        count = 1
        for dim in self.shape:
            count *= dim
        return count

    def __repr__(self) -> str:
        return f"Tensor{self.shape}:{self.dtype}"


class _Storage:
    __slots__ = ("dtype",)

    def __init__(self, dtype: str) -> None:
        self.dtype = dtype


def _rebuild_tensor_v2(storage, storage_offset, size, stride, *rest):
    return TensorInfo(size, getattr(storage, "dtype", "?"))


def _rebuild_parameter(data, requires_grad=True, backward_hooks=None):
    return data


class _Unknown:
    """Any class in the pickle we do not model. Never hit for weights."""

    def __init__(self, *args, **kwargs) -> None:
        self.args = args

    def __setstate__(self, state) -> None:
        self.state = state

    def __repr__(self) -> str:
        return f"<unmodelled {self.args!r}>"


class _CheckpointUnpickler(pickle.Unpickler):
    def find_class(self, module: str, name: str):
        if module.startswith("torch"):
            if name == "_rebuild_tensor_v2":
                return _rebuild_tensor_v2
            if name == "_rebuild_parameter":
                return _rebuild_parameter
            if name in _STORAGE_DTYPES:
                # Only ever referenced through a persistent id.
                return lambda *args, **kwargs: None
            if name == "Size":
                return tuple
        if module.startswith("collections") and name == "OrderedDict":
            import collections
            return collections.OrderedDict
        if module == "numpy.core.multiarray" and name == "scalar":
            return lambda *args, **kwargs: None
        try:
            return super().find_class(module, name)
        except Exception:
            return _Unknown

    def persistent_load(self, pid):
        # ('storage', <storage class>, key, location, numel)
        dtype = "?"
        if isinstance(pid, (tuple, list)) and len(pid) >= 2:
            name = getattr(pid[1], "__name__", str(pid[1])).split(".")[-1]
            dtype = _STORAGE_DTYPES.get(name, name)
        return _Storage(dtype)


def read_checkpoint(path: str | Path) -> Any:
    """Return the checkpoint's object graph, tensors replaced by TensorInfo."""
    path = Path(path)
    with zipfile.ZipFile(path) as archive:
        entry = next(
            name for name in archive.namelist() if name.endswith("data.pkl")
        )
        with archive.open(entry) as handle:
            return _CheckpointUnpickler(handle).load()


def shapes(state_dict: dict) -> dict[str, tuple[int, ...]]:
    """``{parameter name: shape}`` for every tensor in a state dict."""
    return {
        key: tuple(value.shape)
        for key, value in state_dict.items()
        if hasattr(value, "shape")
    }


def _describe(value: Any, indent: int = 0) -> str:
    pad = "  " * indent
    if isinstance(value, dict):
        lines = []
        for key, item in value.items():
            if isinstance(item, dict):
                lines.append(f"{pad}{key}:")
                lines.append(_describe(item, indent + 1))
            elif hasattr(item, "shape"):
                lines.append(f"{pad}{key}: {item!r}")
            elif isinstance(item, (list, tuple)) and len(item) > 8:
                lines.append(f"{pad}{key}: [{len(item)} items] {list(item[:4])}...")
            else:
                lines.append(f"{pad}{key}: {item!r}")
        return "\n".join(lines)
    return f"{pad}{value!r}"


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print(__doc__)
        print("usage: python3 tests/checkpoint_reader.py <checkpoint.pt>")
        raise SystemExit(2)

    data = read_checkpoint(sys.argv[1])

    if isinstance(data, dict) and "model_state_dict" in data:
        tensors = shapes(data["model_state_dict"])
        total = sum(
            info.numel()
            for info in data["model_state_dict"].values()
            if hasattr(info, "numel")
        )
        print(f"model_state_dict: {len(tensors)} tensors, {total:,} values")
        for key, shape in tensors.items():
            print(f"  {key:44} {shape}")
        rest = {k: v for k, v in data.items() if k != "model_state_dict"}
        print("\nmetadata:")
        print(_describe(rest, 1))
    else:
        print(_describe(data))
