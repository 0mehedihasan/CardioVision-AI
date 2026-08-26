#!/usr/bin/env python3
"""
ECG pipeline verification.

Covers the whole path from an uploaded file to the array the model is handed:
the four readers, WFDB formats 16 and 212, lead-order and sampling-frequency
handling, and every rejection that exists to stop a plausible-looking answer
being computed from the wrong input.

    python3 tests/test_ecg_pipeline.py

torch is not needed. SciPy is used if installed; if it is missing, the two
filter/resample calls are replaced with numpy stand-ins so that everything
else — the readers, validation, length handling and normalisation — is still
exercised for real. The banner says which mode ran, because "all checks
passed" means something different in each.
"""

from __future__ import annotations

import io
import json
import sys
import types
import zipfile
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))


# ---- torch stub ------------------------------------------------------
# config.py selects a device at import time. Nothing here runs a model, so
# importing torch would cost seconds for nothing — but only stub it when it is
# genuinely absent, so a real install is never shadowed.
if "torch" not in sys.modules:
    try:
        import torch  # noqa: F401
    except ImportError:
        stub = types.ModuleType("torch")
        stub.cuda = types.SimpleNamespace(is_available=lambda: False)
        stub.backends = types.SimpleNamespace(
            mps=types.SimpleNamespace(is_available=lambda: False)
        )
        stub.device = lambda name: name
        sys.modules["torch"] = stub


# ---- scipy stand-in --------------------------------------------------
def _install_scipy_stub() -> bool:
    """
    Replace scipy.signal with numpy equivalents. Returns True if it was needed.

    ``sosfiltfilt`` becomes the identity and ``resample_poly`` becomes nearest
    -neighbour decimation. Neither is a correct filter, and that is the point:
    they preserve the shape and length contracts the rest of the pipeline
    depends on without pretending to reproduce the DSP. The two functions also
    raise on the same conditions the real ones do (cutoff above Nyquist, record
    shorter than the filter padding) so the passthrough branch is reachable.
    """
    try:
        import scipy.signal  # noqa: F401
        return False
    except ImportError:
        pass

    scipy_module = types.ModuleType("scipy")
    signal_module = types.ModuleType("scipy.signal")

    def butter(order, cut, btype, fs, output):
        if max(cut) >= fs / 2:
            raise ValueError("cutoff at or above Nyquist")
        return np.zeros((order, 6))

    def sosfiltfilt(sos, x, axis=0):
        if x.shape[axis] <= 3 * (2 * sos.shape[0] + 1):
            raise ValueError("record shorter than the filter padding")
        return x

    def resample_poly(x, up, down, axis=0):
        count = int(round(x.shape[axis] * up / down))
        idx = (np.arange(count) * down / up).astype(int)
        return x[np.clip(idx, 0, x.shape[axis] - 1)]

    signal_module.butter = butter
    signal_module.sosfiltfilt = sosfiltfilt
    signal_module.resample_poly = resample_poly
    scipy_module.signal = signal_module

    sys.modules["scipy"] = scipy_module
    sys.modules["scipy.signal"] = signal_module
    return True


STUBBED_SCIPY = _install_scipy_stub()

from cardiovision.config import (  # noqa: E402
    ECG_INPUT_LENGTH,
    ECG_LEAD_NAMES,
    ECG_TARGET_FS,
)
from cardiovision.preprocessing import ecg_io as E  # noqa: E402

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


def rejects(label: str, call, expect: str = "") -> None:
    """Assert that a call raises UnsupportedEcgError, and say what it said."""
    global CHECKS
    CHECKS += 1
    try:
        call()
    except E.UnsupportedEcgError as error:
        message = str(error)
        if expect and expect not in message:
            print(f"  [FAIL] {label}  -> raised, but without {expect!r}: {message}")
            FAILURES.append(label)
        else:
            print(f"  [PASS] {label}  -> {message[:72]}")
        return
    print(f"  [FAIL] {label}  -> no error raised")
    FAILURES.append(label)


# ---- a synthetic 10 s, 500 Hz, 12-lead recording ---------------------
RNG = np.random.default_rng(7)
BEAT = np.sin(np.linspace(0, 60 * np.pi, 5000))[:, None] * 0.6
ECG = (RNG.normal(0, 0.25, (5000, 12)) + BEAT).astype(np.float32)
HEADER = ",".join(ECG_LEAD_NAMES)


def as_csv(matrix: np.ndarray, header: str = HEADER) -> bytes:
    rows = "\n".join(",".join(f"{value:.5f}" for value in row) for row in matrix)
    return ((header + "\n" if header else "") + rows).encode()


def as_npy(matrix: np.ndarray) -> bytes:
    buffer = io.BytesIO()
    np.save(buffer, np.ascontiguousarray(matrix))
    return buffer.getvalue()


def wfdb_16(matrix: np.ndarray, name: str = "HR99999") -> tuple[bytes, bytes]:
    """Header + little-endian int16 signal file, gain 1000 ADU/mV."""
    counts = np.clip((matrix * 1000).round(), -32000, 32000).astype("<i2")
    header = f"{name} 12 500 {matrix.shape[0]}\n" + "".join(
        f"{name}.dat 16 1000(0)/mV 16 0 0 0 0 {lead}\n" for lead in ECG_LEAD_NAMES
    )
    return header.encode(), counts.tobytes()


def wfdb_212(matrix: np.ndarray, name: str = "HR212") -> tuple[bytes, bytes, np.ndarray]:
    """Header + 12-bit packed signal file: three bytes carry two samples."""
    counts = np.clip((matrix * 100).round(), -2000, 2000).astype(np.int32)
    flat = counts.ravel()
    packed = bytearray()

    for index in range(0, len(flat) - 1, 2):
        low = int(flat[index]) & 0xFFF
        high = int(flat[index + 1]) & 0xFFF
        packed += bytes([
            low & 0xFF,
            ((low >> 8) & 0x0F) | ((high >> 8) << 4),
            high & 0xFF,
        ])

    header = f"{name} 12 500 {matrix.shape[0]}\n" + "".join(
        f"{name}.dat 212 100(0)/mV 12 0 0 0 0 {lead}\n" for lead in ECG_LEAD_NAMES
    )
    return header.encode(), bytes(packed), counts / 100.0


# ============================================================
print("\n=== 1. the preprocessing chain reproduces the training contract ===")
# ============================================================

processed = E.preprocess_ecg(ECG, 500)

check("5000 samples at 500 Hz -> (1000, 12)",
      processed.shape == (ECG_INPUT_LENGTH, 12), str(processed.shape))
check("float32, as the model expects", processed.dtype == np.float32)
check("clipped into the training range",
      processed.min() >= -10.0 and processed.max() <= 10.0,
      f"{processed.min():.2f} .. {processed.max():.2f}")
check("no non-finite values survive", np.isfinite(processed).all())

medians = np.median(processed, axis=0)
check("every lead is centred on its own median",
      np.abs(medians).max() < 0.05, f"max |median| = {np.abs(medians).max():.5f}")

iqr = np.percentile(processed, 75, axis=0) - np.percentile(processed, 25, axis=0)
check("every lead is scaled to unit IQR",
      np.allclose(iqr, 1.0, atol=0.02), f"{iqr.min():.4f} .. {iqr.max():.4f}")

# Per-lead, not global: one saturated lead must not rescale the others.
loud = ECG.copy()
loud[:, 5] *= 50
loud_out = E.robust_normalize(loud)
quiet_out = E.robust_normalize(ECG)
check("a saturated lead does not rescale its neighbours",
      np.allclose(loud_out[:, 0], quiet_out[:, 0], atol=1e-5))

# ============================================================
print("\n=== 2. length handling pads and truncates at the END ===")
# ============================================================

check("a longer record is cut, keeping the start",
      np.array_equal(E.fix_length(ECG[:1200], 1000), ECG[:1000]))

padded = E.fix_length(ECG[:400], 1000)
check("a shorter record keeps its samples at the front",
      np.array_equal(padded[:400], ECG[:400]))
check("and the padding is zeros at the end, not a time shift",
      not padded[400:].any())
check("an exact-length record is returned untouched",
      np.array_equal(E.fix_length(ECG[:1000], 1000), ECG[:1000]))

# ============================================================
print("\n=== 3. degenerate leads cannot blow up the scale ===")
# ============================================================

flat = ECG.copy()
flat[:, 3] = 2.5                       # a dead lead: IQR is exactly zero
flat_out = E.robust_normalize(flat)
check("a flat lead becomes zeros rather than amplified noise",
      np.isfinite(flat_out).all() and np.abs(flat_out[:, 3]).max() < 1e-5,
      f"max |value| = {np.abs(flat_out[:, 3]).max():.2e}")
check("and the other leads are unaffected",
      np.allclose(flat_out[:, 0], quiet_out[:, 0], atol=1e-5))

# ============================================================
print("\n=== 4. input the model cannot honestly be run on is refused ===")
# ============================================================

rejects("a 1-D trace is rejected",
        lambda: E.preprocess_ecg(ECG[:, 0], 500), "2-D")
rejects("an 8-lead recording is rejected, not padded",
        lambda: E.preprocess_ecg(ECG[:, :8], 500), "12 leads")

nan_ecg = ECG.copy()
nan_ecg[10, 2] = np.nan
rejects("NaN samples are rejected, not zero-filled",
        lambda: E.preprocess_ecg(nan_ecg, 500), "fabricate")

inf_ecg = ECG.copy()
inf_ecg[10, 2] = np.inf
rejects("infinite samples are rejected too",
        lambda: E.preprocess_ecg(inf_ecg, 500))

rejects("a 15-lead recording is rejected with the reason",
        lambda: E.load_ecg("x.npy", as_npy(np.zeros((1000, 15), np.float32))),
        "12")
rejects("a single sample is not a recording",
        lambda: E.load_ecg("x.npy", as_npy(np.zeros((1, 12), np.float32))))
rejects("a 12x12 array is ambiguous and says so",
        lambda: E.load_ecg("x.npy", as_npy(np.zeros((12, 12), np.float32))),
        "ambiguous")

for name in ("scan.pdf", "record.edf", "noextension"):
    rejects(f"{name} is refused with the accepted list",
            lambda name=name: E.load_ecg(name, b"junk"), "csv")

# ============================================================
print("\n=== 5. CSV ===")
# ============================================================

csv = E.load_ecg("record.csv", as_csv(ECG[:1000]))

check("a headed CSV loads", csv.signal.shape == (ECG_INPUT_LENGTH, 12))
check("1000 samples is read as 10 s at 100 Hz",
      csv.sampling_frequency == 100.0, str(csv.sampling_frequency))
check("the assumed rate is reported, not hidden",
      any("assumed 100 Hz" in note for note in csv.notes), str(csv.notes))
check("the lead order is recognised", csv.lead_names == tuple(ECG_LEAD_NAMES))
check("tensor_input is (1, 12, 1000), channels first as trained",
      csv.tensor_input.shape == (1, 12, ECG_INPUT_LENGTH),
      str(csv.tensor_input.shape))
check("to_dict() reports the real rate and the resampled one",
      csv.to_dict()["sampling_frequency_hz"] == 100.0
      and csv.to_dict()["resampled_to_hz"] == ECG_TARGET_FS)

headerless = E.load_ecg("bare.csv", as_csv(ECG[:1000], header=""))
check("a headerless CSV still loads",
      headerless.signal.shape == (ECG_INPUT_LENGTH, 12))
check("and falls back to the standard lead order",
      headerless.lead_names == tuple(ECG_LEAD_NAMES))

tabbed = E.load_ecg("record.tsv", as_csv(ECG[:1000]).replace(b",", b"\t"))
check("tab-separated works the same way",
      np.allclose(tabbed.signal, csv.signal))

ragged = b"I,II,III\n1,2,3\n4,5\n"
rejects("rows of different widths are rejected",
        lambda: E.load_ecg("r.csv", ragged), "inconsistent")
rejects("a non-numeric cell names its row",
        lambda: E.load_ecg("r.csv", (HEADER + "\n" + ",".join(["1"] * 11) + ",oops\n").encode()),
        "not a number")
rejects("a header with no data underneath is rejected",
        lambda: E.load_ecg("r.csv", (HEADER + "\n").encode()), "no samples")

# ============================================================
print("\n=== 6. a time column is dropped, at either end ===")
# ============================================================

time_axis = (np.arange(1000) / 100).astype(np.float32)

leading = E.load_ecg(
    "lead.csv",
    as_csv(np.c_[time_axis, ECG[:1000]], "time_seconds," + HEADER),
)
check("a leading time column is dropped, not read as a 13th lead",
      leading.signal.shape == (ECG_INPUT_LENGTH, 12))
check("and it is reported",
      any("leading" in note for note in leading.notes), str(leading.notes))

# The training notebook's own saliency export puts time_seconds last.
trailing = E.load_ecg(
    "trail.csv",
    as_csv(np.c_[ECG[:1000], time_axis], HEADER + ",time_seconds"),
)
check("a trailing time column is dropped too",
      trailing.signal.shape == (ECG_INPUT_LENGTH, 12))
check("and it is reported",
      any("trailing" in note for note in trailing.notes), str(trailing.notes))
check("both give the same signal as the 12-column file",
      np.allclose(leading.signal, csv.signal)
      and np.allclose(trailing.signal, csv.signal))

# A 13th column that is NOT a time axis must still be refused, and the
# message should name the offending label rather than talk about array axes.
rejects("a genuine 13th lead is rejected, naming the surplus column",
        lambda: E.load_ecg("x.csv", as_csv(np.c_[ECG[:1000], time_axis],
                                           HEADER + ",V7")),
        "V7")
rejects("an 11-lead labelled export is refused too",
        lambda: E.load_ecg("x.csv", as_csv(ECG[:1000, :11],
                                           ",".join(ECG_LEAD_NAMES[:11]))),
        "11 labelled columns")

# ============================================================
print("\n=== 7. lead order is checked, never silently permuted ===")
# ============================================================

swapped_names = list(ECG_LEAD_NAMES)
swapped_names[0], swapped_names[1] = swapped_names[1], swapped_names[0]
swapped = E.load_ecg("s.csv", as_csv(ECG[:1000], ",".join(swapped_names)))

check("a swapped lead order is reported",
      any("NOT reordered" in note for note in swapped.notes), str(swapped.notes))
check("the columns really were left alone",
      np.allclose(swapped.display_signal[:, 0], ECG[:1000, 0], atol=1e-4))
check("lead_order_matches_training is False so the UI can warn",
      swapped.to_dict()["lead_order_matches_training"] is False)

prefixed = E.load_ecg(
    "p.csv",
    as_csv(ECG[:1000], ",".join("Lead " + name for name in ECG_LEAD_NAMES)),
)
check("a cosmetic 'Lead I' prefix is not treated as a mismatch",
      prefixed.to_dict()["lead_order_matches_training"] is True,
      str(prefixed.notes))

lowercase = E.load_ecg(
    "l.csv", as_csv(ECG[:1000], ",".join(n.lower() for n in ECG_LEAD_NAMES))
)
check("lowercase lead names are matched case-insensitively",
      lowercase.to_dict()["lead_order_matches_training"] is True)

foreign = E.load_ecg(
    "f.csv", as_csv(ECG[:1000], ",".join(f"ch{i}" for i in range(12)))
)
check("unrecognised labels are reported as given, not invented",
      foreign.lead_names == tuple(f"ch{i}" for i in range(12)))
check("and the note says the training order was assumed",
      any("assumed to already be in the order" in n for n in foreign.notes))

# ============================================================
print("\n=== 8. NPY, in either orientation ===")
# ============================================================

samples_first = E.load_ecg("a.npy", as_npy(ECG[:1000]))
leads_first = E.load_ecg("b.npy", as_npy(ECG[:1000].T))

check("(samples, leads) loads", samples_first.signal.shape == (1000, 12))
check("(leads, samples) is transposed", leads_first.signal.shape == (1000, 12))
check("the transposition is reported",
      any("leads-first" in note for note in leads_first.notes))
check("both orientations produce an identical signal",
      np.allclose(samples_first.signal, leads_first.signal))

batched = E.load_ecg("c.npy", as_npy(ECG[:1000][np.newaxis, ...]))
check("a cached batch of one is unwrapped",
      batched.signal.shape == (1000, 12))

rejects("a 3-D array with a real batch dimension is rejected",
        lambda: E.load_ecg("d.npy", as_npy(np.zeros((4, 1000, 12), np.float32))),
        "2-D")
rejects("a corrupt .npy says so",
        lambda: E.load_ecg("e.npy", b"not a numpy file"), "npy")

# ============================================================
print("\n=== 9. JSON ===")
# ============================================================

payload = json.dumps({
    "sampling_frequency": 500,
    "units": "mV",
    "leads": {name: ECG[:, i].tolist() for i, name in enumerate(ECG_LEAD_NAMES)},
}).encode()

js = E.load_ecg("r.json", payload)
check("JSON with a declared rate loads",
      js.signal.shape == (1000, 12) and js.sampling_frequency == 500.0)
check("units are carried through for the waveform axis", js.units == "mV")
check("duration comes out at 10 s",
      abs(js.duration_seconds - 10.0) < 1e-6, f"{js.duration_seconds}")
check("no rate was guessed",
      not any("assumed" in note for note in js.notes), str(js.notes))

rejects("JSON without a leads mapping is rejected",
        lambda: E.load_ecg("x.json", b'{"data": [1,2,3]}'), "leads")
rejects("JSON with unequal lead lengths is rejected",
        lambda: E.load_ecg("x.json", b'{"leads": {"I": [1,2], "II": [1]}}'),
        "different lengths")
rejects("malformed JSON says so",
        lambda: E.load_ecg("x.json", b"{nope"), "JSON")

# ============================================================
print("\n=== 10. an unknown sampling rate is a hard error, never a guess ===")
# ============================================================

odd = as_npy(ECG[:3333])
rejects("3333 samples with no rate is refused",
        lambda: E.load_ecg("odd.npy", odd), "sampling_frequency")

told = E.load_ecg("odd.npy", odd, sampling_frequency=333.3)
check("the same file is accepted once the rate is supplied",
      told.signal.shape == (1000, 12))
check("and the supplied rate is what gets reported",
      told.sampling_frequency == 333.3)

five_k = E.load_ecg("f.npy", as_npy(ECG))
check("5000 samples is read as 10 s at 500 Hz",
      five_k.sampling_frequency == 500.0, str(five_k.sampling_frequency))

# ============================================================
print("\n=== 11. WFDB format 16 (PTB-XL's native layout) ===")
# ============================================================

hea, dat = wfdb_16(ECG)
wf = E.load_ecg("HR99999.hea", hea, companion={"HR99999.dat": dat})

check("format 16 decodes to (1000, 12)", wf.signal.shape == (1000, 12))
check("the sampling rate comes from the header, not a guess",
      wf.sampling_frequency == 500.0, str(wf.sampling_frequency))
check("no rate was assumed",
      not any("assumed" in note for note in wf.notes), str(wf.notes))
check("lead names come from the header",
      wf.lead_names == tuple(ECG_LEAD_NAMES))
check("the record name is picked up", wf.record_name == "HR99999")
check("units are read from the header", wf.units == "mV")
check("gain is applied, so values are millivolts not ADC counts",
      np.abs(wf.display_signal).max() < 10,
      f"max = {np.abs(wf.display_signal).max():.3f} mV")
check("the decoded samples match what was encoded",
      np.allclose(wf.display_signal[:20, :3],
                  (np.clip((ECG * 1000).round(), -32000, 32000) / 1000)[::5][:20, :3],
                  atol=2e-3))

baselined = wfdb_16(ECG)[0].replace(b"1000(0)/mV", b"1000(500)/mV")
shifted = E.load_ecg("HR99999.hea", baselined, companion={"HR99999.dat": dat})
check("an ADC baseline offset is subtracted",
      np.allclose(shifted.display_signal, wf.display_signal - 0.5, atol=1e-5),
      f"delta = {(shifted.display_signal - wf.display_signal).mean():.4f}")

rejects("a .hea with no signal file explains the pairing",
        lambda: E.load_ecg("HR99999.hea", hea), "zip")
rejects("a bare .dat explains that the .hea holds the layout",
        lambda: E.load_ecg("HR99999.dat", dat), ".hea")
rejects("an empty header is rejected",
        lambda: E.load_ecg("x.hea", b"", companion={"x.dat": dat}), "empty")
rejects("an unsupported storage format names itself",
        lambda: E.load_ecg(
            "x.hea",
            b"x 12 500 5000\n" + b"".join(
                f"x.dat 24 1000(0)/mV 24 0 0 0 0 {n}\n".encode()
                for n in ECG_LEAD_NAMES
            ),
            companion={"x.dat": dat},
        ),
        "format 24")

# ============================================================
print("\n=== 12. WFDB format 212 (12-bit packed) ===")
# ============================================================

hea212, dat212, expected212 = wfdb_212(ECG)
wf212 = E.load_ecg("HR212.hea", hea212, companion={"HR212.dat": dat212})

check("format 212 decodes to (1000, 12)", wf212.signal.shape == (1000, 12))
check("the 12-bit unpacking round-trips",
      np.allclose(wf212.display_signal[:20], expected212[::5][:20], atol=0.02),
      f"{wf212.display_signal[0, :3]} vs {expected212[0, :3]}")
check("negative samples survive sign extension",
      wf212.display_signal.min() < -0.1, f"min = {wf212.display_signal.min():.3f}")

# ============================================================
print("\n=== 13. a zip of a WFDB pair ===")
# ============================================================

buffer = io.BytesIO()
with zipfile.ZipFile(buffer, "w") as archive:
    archive.writestr("HR99999.hea", hea.decode())
    archive.writestr("HR99999.dat", dat)

zipped = E.load_ecg("record.zip", buffer.getvalue())
check("a zip pairs the .hea with its .dat",
      zipped.signal.shape == (1000, 12) and zipped.sampling_frequency == 500.0)
check("the zip result is identical to the loose pair",
      np.allclose(zipped.signal, wf.signal))

empty_zip = io.BytesIO()
with zipfile.ZipFile(empty_zip, "w") as archive:
    archive.writestr("notes.txt", "nothing here")
rejects("a zip with no .hea lists what it did find",
        lambda: E.load_ecg("x.zip", empty_zip.getvalue()), "no .hea")
rejects("a corrupt zip says so",
        lambda: E.load_ecg("x.zip", b"PK\x03\x04 truncated"), "zip")

multi = io.BytesIO()
with zipfile.ZipFile(multi, "w") as archive:
    archive.writestr("HR99999.hea", hea.decode())
    archive.writestr("HR99999.dat", dat)
    archive.writestr("HR00001.hea", hea.decode())
loaded_multi = E.load_ecg("m.zip", multi.getvalue())
check("a multi-record zip says which record it used",
      any("records" in note for note in loaded_multi.notes),
      str(loaded_multi.notes))

# ============================================================
print("\n=== 14. records that are not 10 seconds are flagged ===")
# ============================================================

short = E.load_ecg("s.npy", as_npy(ECG[:600]), sampling_frequency=100)
check("a 6 s record warns that 4 s of silence was padded in",
      any("zero-padded" in note for note in short.notes), str(short.notes))
check("the padding warning names the amount",
      any("4.00 s" in note for note in short.notes))

long_record = E.load_ecg("l.npy", as_npy(ECG[:2000]), sampling_frequency=100)
check("a 20 s record warns that only the first 10 s were read",
      any("first 10 seconds" in note for note in long_record.notes),
      str(long_record.notes))

exact = E.load_ecg("e.npy", as_npy(ECG[:1000]), sampling_frequency=100)
check("a 10 s record gets no duration warning",
      not any("zero-padded" in n or "first 10 seconds" in n for n in exact.notes),
      str(exact.notes))

# ============================================================
print("\n=== 15. unfiltered input is reported rather than passed off ===")
# ============================================================

# Two ways the bandpass genuinely cannot be applied: a rate too low for a
# 40 Hz cutoff, and a record shorter than filtfilt's padding.
check("a 60 Hz record is flagged as unfiltered",
      not E.bandpass_is_applicable(ECG[:600], 60.0))
check("a 20-sample record is flagged as unfiltered",
      not E.bandpass_is_applicable(ECG[:20], 500.0))
check("a normal record is not flagged",
      E.bandpass_is_applicable(ECG, 500.0))

low_rate = E.load_ecg("lr.npy", as_npy(ECG[:600]), sampling_frequency=60)
check("the unfiltered warning reaches the response notes",
      any("unfiltered" in note for note in low_rate.notes), str(low_rate.notes))
check("and it says the output is provisional",
      any("provisional" in note for note in low_rate.notes))

# ============================================================
print("\n=== 16. display signal shares the model's time base ===")
# ============================================================

check("display_signal is resampled to the same length",
      csv.display_signal.shape == csv.signal.shape,
      f"{csv.display_signal.shape} vs {csv.signal.shape}")
check("but is NOT normalised, so the figure can show real units",
      not np.allclose(wf.display_signal, wf.signal, atol=1e-3))
check("its scale is the source's, so mV stays mV",
      np.abs(wf.display_signal).max() < 5,
      f"max = {np.abs(wf.display_signal).max():.3f}")

# ============================================================
print("\n" + "=" * 62)

if STUBBED_SCIPY:
    print("NOTE: SciPy is not installed here, so bandpass_filter and")
    print("      resample_ecg ran against numpy stand-ins. Everything above")
    print("      is real except the filter arithmetic itself — re-run this")
    print("      with SciPy installed to cover that too.")
    print("=" * 62)

if FAILURES:
    print(f"{len(FAILURES)} FAILURE(S) out of {CHECKS} checks:")
    for item in FAILURES:
        print(f"  - {item}")
    sys.exit(1)

print(f"ALL {CHECKS} ECG PIPELINE CHECKS PASSED")
