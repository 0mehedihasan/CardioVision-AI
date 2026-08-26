"""
A minimal ``torch`` stand-in for tests that must not depend on torch.

Enough of ``torch.nn`` to *construct* the models in this repo — the 1-D ECG
ResNet, the 2-D echo encoder and the 3-D CCTA U-Net — and inspect their
parameter names and shapes. Nothing here computes anything: there is no forward
pass, no autograd, and no tensor arithmetic. That is deliberate — a stub that
pretended to compute would invite someone to read its output as a result.
``torch.nn.functional`` is present but every member of it raises, for the same
reason.

Use it for structural checks (does this architecture match the checkpoint?) and
for exercising pure-Python reporting logic that happens to live in a module
which imports torch at the top. For anything about numbers the model produces,
install torch and run the real thing.

``install()`` is a no-op when torch is genuinely available, so a real
environment is never shadowed. It returns True when the stub was used, which
callers should print — "all checks passed" means different things either way.
"""

from __future__ import annotations

import sys
import types

# Buffers, not parameters: torch's numel() sum over .parameters() excludes
# these, so a parameter count that includes them will not match the notebook's.
BUFFER_NAMES = frozenset({"running_mean", "running_var", "num_batches_tracked"})


class Module:
    """Reproduces nn.Module's child registration and dotted parameter naming."""

    def __init__(self) -> None:
        object.__setattr__(self, "_children", {})
        object.__setattr__(self, "_params", {})

    def __setattr__(self, key, value):
        if isinstance(value, Module):
            self._children[key] = value
        object.__setattr__(self, key, value)

    def register(self, name: str, shape: tuple[int, ...]) -> None:
        self._params[name] = tuple(shape)

    def state_dict(self, prefix: str = "") -> dict[str, tuple[int, ...]]:
        out: dict[str, tuple[int, ...]] = {}
        for name, shape in self._params.items():
            out[prefix + name] = shape
        for name, child in self._children.items():
            out.update(child.state_dict(f"{prefix}{name}."))
        return out

    def parameter_count(self) -> int:
        total = 0
        for name, shape in self._params.items():
            if name in BUFFER_NAMES:
                continue
            count = 1
            for dim in shape:
                count *= dim
            total += count
        for child in self._children.values():
            total += child.parameter_count()
        return total

    # The no-ops a caller might reach for after construction.
    def eval(self):
        return self

    def train(self, mode: bool = True):
        return self

    def to(self, *args, **kwargs):
        return self

    def zero_grad(self, set_to_none: bool = True) -> None:
        return None

    def __call__(self, *args, **kwargs):
        raise NotImplementedError(
            "This is the torch stub: it can build a model but not run one. "
            "Install torch to execute a forward pass."
        )


class Sequential(Module):
    def __init__(self, *layers) -> None:
        super().__init__()
        for index, layer in enumerate(layers):
            setattr(self, str(index), layer)


class Conv1d(Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, bias=True, **kwargs) -> None:
        super().__init__()
        self.register("weight", (out_channels, in_channels, kernel_size))
        if bias:
            self.register("bias", (out_channels,))


class Conv2d(Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, bias=True, **kwargs) -> None:
        super().__init__()
        size = (
            kernel_size if isinstance(kernel_size, tuple)
            else (kernel_size, kernel_size)
        )
        self.register("weight", (out_channels, in_channels, *size))
        if bias:
            self.register("bias", (out_channels,))


def _triple(value) -> tuple[int, int, int]:
    if isinstance(value, tuple):
        return value
    return (value, value, value)


class Conv3d(Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, bias=True, **kwargs) -> None:
        super().__init__()
        self.register("weight", (out_channels, in_channels, *_triple(kernel_size)))
        if bias:
            self.register("bias", (out_channels,))


class ConvTranspose3d(Module):
    """Note the transposed weight layout: (in, out, k, k, k), as torch stores it."""

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, bias=True, **kwargs) -> None:
        super().__init__()
        self.register("weight", (in_channels, out_channels, *_triple(kernel_size)))
        if bias:
            self.register("bias", (out_channels,))


class InstanceNorm3d(Module):
    """Affine instance norm has weight and bias and no running statistics."""

    def __init__(self, features, affine=False, track_running_stats=False,
                 **kwargs) -> None:
        super().__init__()
        if affine:
            self.register("weight", (features,))
            self.register("bias", (features,))
        if track_running_stats:
            self.register("running_mean", (features,))
            self.register("running_var", (features,))
            self.register("num_batches_tracked", ())


class BatchNorm1d(Module):
    def __init__(self, features, **kwargs) -> None:
        super().__init__()
        self.register("weight", (features,))
        self.register("bias", (features,))
        self.register("running_mean", (features,))
        self.register("running_var", (features,))
        self.register("num_batches_tracked", ())


class Linear(Module):
    def __init__(self, in_features, out_features, bias=True) -> None:
        super().__init__()
        self.register("weight", (out_features, in_features))
        if bias:
            self.register("bias", (out_features,))


class Passthrough(Module):
    """Activations, pooling and dropout: no parameters, no behaviour."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__()


def _unavailable(name: str):
    def raiser(*args, **kwargs):
        raise NotImplementedError(
            f"torch.{name} is not implemented by the test stub. This code path "
            "needs real torch."
        )
    return raiser


def install() -> bool:
    """
    Register the stub in ``sys.modules``. Returns True if it was needed.

    Call before importing anything from ``cardiovision.inference``.
    """
    try:
        import torch  # noqa: F401
        return False
    except ImportError:
        pass

    nn = types.ModuleType("torch.nn")
    nn.Module = Module
    nn.Sequential = Sequential
    nn.Conv1d = Conv1d
    nn.Conv2d = Conv2d
    nn.Conv3d = Conv3d
    nn.ConvTranspose3d = ConvTranspose3d
    nn.BatchNorm1d = BatchNorm1d
    nn.InstanceNorm3d = InstanceNorm3d
    nn.Linear = Linear
    nn.GELU = Passthrough
    nn.ReLU = Passthrough
    nn.LeakyReLU = Passthrough
    nn.Dropout = Passthrough
    nn.MaxPool1d = Passthrough
    nn.MaxPool3d = Passthrough
    nn.AdaptiveAvgPool1d = Passthrough
    nn.Identity = Passthrough

    # torch.nn.functional: every member computes, so every member refuses.
    functional = types.ModuleType("torch.nn.functional")
    for name in ("interpolate", "relu", "sigmoid", "softmax", "pad"):
        setattr(functional, name, _unavailable(f"nn.functional.{name}"))
    nn.functional = functional

    torch = types.ModuleType("torch")
    torch.nn = nn
    torch.cuda = types.SimpleNamespace(is_available=lambda: False)
    torch.backends = types.SimpleNamespace(
        mps=types.SimpleNamespace(is_available=lambda: False)
    )
    torch.device = lambda name: name
    torch.float32 = "float32"
    torch.no_grad = lambda: _NullContext()
    torch.inference_mode = lambda *args, **kwargs: _NullContext()

    # Anything that would produce numbers fails loudly rather than returning
    # something plausible.
    for name in ("from_numpy", "sigmoid", "softmax", "argmax", "flatten",
                 "load", "save", "tensor", "zeros", "ones", "cat", "stack"):
        setattr(torch, name, _unavailable(name))

    sys.modules["torch"] = torch
    sys.modules["torch.nn"] = nn
    sys.modules["torch.nn.functional"] = functional
    return True


class _NullContext:
    def __enter__(self):
        return None

    def __exit__(self, *args):
        return False
