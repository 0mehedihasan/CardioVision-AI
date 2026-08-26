"""
Reading 12-lead ECG recordings and preparing them for the model.

The preprocessing chain is a straight port of notebooks/03_ECG.ipynb — same
functions, same order, same constants. It has to be: the model learned on
arrays produced by exactly this code, so any deviation moves the input off the
training distribution in a way no error message would reveal.

    bandpass 0.5-40 Hz  ->  resample to 100 Hz  ->  fix to 1000 samples
                        ->  per-lead robust normalise  ->  scrub non-finite

Four input formats are accepted:

    WFDB (.hea + .dat)  PTB-XL's native format. Sampling frequency, lead names
                        and per-lead gain all come from the header, so this is
                        the only format that needs no assumptions.
    CSV / TSV / TXT     One column per lead, one row per sample. A header row
                        is detected and used to check the lead order.
    NPY                 A (samples, 12) or (12, samples) float array, which is
                        what the notebook cached its preprocessed data as.
    JSON               ``{"leads": {"I": [...], ...}, "sampling_frequency": 500}``

What this module refuses to do
------------------------------
Guess the sampling frequency. Everything downstream depends on it: filter
cutoffs are specified in Hz, and resampling to 100 Hz is a ratio against it. A
10-second record read as 500 Hz when it was 100 Hz becomes a 2-second record
padded with two thirds of silence, and the model will still return five
confident-looking probabilities. So an unlabelled format must either be told
the rate or be exactly 1000 or 5000 samples long, where the duration is
unambiguous for a standard PTB-XL recording.

Reorder leads. If a CSV header says the columns are in a different order, that
is reported, not silently fixed — a lead swap that the operator does not know
about is worse than a rejected upload.

Accept anything other than 12 leads. The first convolution has 12 input
channels; 8 leads cannot be run at all, and 15 would have to be truncated by
guessing which three to drop.
"""

from __future__ import annotations

import io
import json
import math
import re
import zipfile
from dataclasses import dataclass, field
from pathlib import PurePosixPath
from typing import Any, Optional, Sequence

import numpy as np

from cardiovision.config import (
    ECG_BANDPASS_HIGH_HZ,
    ECG_BANDPASS_LOW_HZ,
    ECG_BANDPASS_ORDER,
    ECG_CLIP_RANGE,
    ECG_IN_CHANNELS,
    ECG_INPUT_LENGTH,
    ECG_LEAD_NAMES,
    ECG_TARGET_FS,
)


class UnsupportedEcgError(ValueError):
    """The upload is not an ECG this pipeline can read, and we can say why."""


# ============================================================
# THE PREPROCESSING CHAIN (ported verbatim)
# ============================================================
#
# scipy is imported lazily inside bandpass_filter and resample_ecg so that the
# rest of this module — the readers, the validation, the length and
# normalisation steps — stays importable and testable without it.


def _require_scipy_signal():
    """
    Import scipy.signal, and be explicit when it is missing.

    Deliberately not folded into the try/except inside
    :func:`bandpass_filter`. That fallback exists for records the filter cannot
    physically be applied to, and it returns the signal unfiltered — which is a
    reasonable answer for a short record and a terrible one for "the dependency
    is not installed", because every ECG would then be analysed off-distribution
    with only a note to show for it.
    """
    try:
        import scipy.signal as scipy_signal
    except ImportError as error:                          # pragma: no cover
        raise UnsupportedEcgError(
            "SciPy is required to filter and resample ECG signals, and it is "
            "not installed. Running the model on unfiltered input would put it "
            "off its training distribution while still returning five "
            "plausible probabilities, so this is a hard failure. Install it "
            "with: pip install scipy"
        ) from error

    return scipy_signal


def bandpass_filter(
    signal: np.ndarray,
    fs: float,
    lowcut: float = ECG_BANDPASS_LOW_HZ,
    highcut: float = ECG_BANDPASS_HIGH_HZ,
    order: int = ECG_BANDPASS_ORDER,
) -> np.ndarray:
    """
    Zero-phase Butterworth bandpass, applied down the time axis.

    Returns the input unfiltered if the filter cannot be built or applied,
    which is what the training code did. That fallback matters at the edges:
    ``sosfiltfilt`` needs the record to be longer than its padding length, and
    Nyquist puts a hard ceiling on ``highcut``, so a short or low-rate record
    reaches the model unfiltered rather than not at all. The caller is told,
    because unfiltered input is a real deviation from training.
    """
    scipy_signal = _require_scipy_signal()

    try:
        sos = scipy_signal.butter(
            order,
            [lowcut, highcut],
            btype="bandpass",
            fs=fs,
            output="sos",
        )
        filtered = scipy_signal.sosfiltfilt(sos, signal, axis=0)
        return np.asarray(filtered, dtype=np.float32)
    except Exception:
        return np.asarray(signal, dtype=np.float32)


def bandpass_is_applicable(signal: np.ndarray, fs: float) -> bool:
    """
    Whether :func:`bandpass_filter` will actually filter rather than pass through.

    Reimplementing the two conditions is worth it so the API can report
    "unfiltered" as a note instead of the operator having to infer it: the
    upper cutoff must sit below Nyquist, and filtfilt's padding needs
    ``3 * (2 * order + 1)`` samples of signal to work with.
    """
    if fs <= 0:
        return False
    if ECG_BANDPASS_HIGH_HZ >= fs / 2:
        return False
    return signal.shape[0] > 3 * (2 * ECG_BANDPASS_ORDER + 1)


def resample_ecg(
    signal: np.ndarray,
    original_fs: float,
    target_fs: int = ECG_TARGET_FS,
) -> np.ndarray:
    """Polyphase resample down the time axis, with the ratio in lowest terms."""
    if int(original_fs) == int(target_fs):
        return np.asarray(signal, dtype=np.float32)

    scipy_signal = _require_scipy_signal()

    gcd = math.gcd(int(original_fs), int(target_fs))
    up = int(target_fs) // gcd
    down = int(original_fs) // gcd

    resampled = scipy_signal.resample_poly(signal, up, down, axis=0)
    return np.asarray(resampled, dtype=np.float32)


def fix_length(
    signal: np.ndarray,
    target_length: int = ECG_INPUT_LENGTH,
) -> np.ndarray:
    """
    Truncate or zero-pad at the END, never the start.

    Padding at the front would shift every beat later in time, and the model's
    receptive field is large enough for that to matter.
    """
    n_samples = signal.shape[0]

    if n_samples == target_length:
        return np.asarray(signal, dtype=np.float32)

    if n_samples > target_length:
        return np.asarray(signal[:target_length], dtype=np.float32)

    pad_length = target_length - n_samples

    return np.pad(
        signal,
        ((0, pad_length), (0, 0)),
        mode="constant",
    ).astype(np.float32)


def robust_normalize(signal: np.ndarray) -> np.ndarray:
    """
    Per-lead median/IQR standardisation, then clip.

    Median and IQR rather than mean and standard deviation because a single
    pacing spike or a saturated lead would otherwise rescale the whole trace.
    A flat lead has IQR 0, so the scale is floored at 1.0 rather than dividing
    by something near zero — a dead lead comes out as a constant offset, not as
    amplified noise.
    """
    signal = np.asarray(signal, dtype=np.float32)

    median = np.median(signal, axis=0, keepdims=True)
    q25 = np.percentile(signal, 25, axis=0, keepdims=True)
    q75 = np.percentile(signal, 75, axis=0, keepdims=True)

    scale = q75 - q25
    scale[scale < 1e-6] = 1.0

    signal = (signal - median) / scale
    signal = np.clip(signal, ECG_CLIP_RANGE[0], ECG_CLIP_RANGE[1])

    return signal.astype(np.float32)


def preprocess_ecg(signal: np.ndarray, fs: float) -> np.ndarray:
    """
    Full chain: ``(n_samples, 12)`` at ``fs`` Hz -> ``(1000, 12)`` at 100 Hz.

    Order is load-bearing. Filtering before resampling means the 40 Hz cutoff
    is applied while the high-frequency content is still there to remove;
    normalising last means the statistics are computed on the array the model
    actually sees.
    """
    signal = np.asarray(signal, dtype=np.float32)

    if signal.ndim != 2:
        raise UnsupportedEcgError(
            f"Expected a 2-D ECG of shape (samples, leads), got {signal.shape}."
        )

    if signal.shape[1] != ECG_IN_CHANNELS:
        raise UnsupportedEcgError(
            f"Expected {ECG_IN_CHANNELS} leads, got {signal.shape[1]}."
        )

    if not np.isfinite(signal).all():
        raise UnsupportedEcgError(
            "The ECG contains NaN or infinite samples. Zero-filling them here "
            "would fabricate signal, so the recording is rejected instead."
        )

    signal = bandpass_filter(signal, fs)
    signal = resample_ecg(signal, int(fs), ECG_TARGET_FS)
    signal = fix_length(signal, ECG_INPUT_LENGTH)
    signal = robust_normalize(signal)
    signal = np.nan_to_num(signal, nan=0.0, posinf=0.0, neginf=0.0)

    return signal.astype(np.float32)


# ============================================================
# LOADED RECORD
# ============================================================


@dataclass
class LoadedEcg:
    """One ECG, decoded and preprocessed, with everything the API must report."""

    # (1000, 12) float32, ready for the model after a transpose.
    signal: np.ndarray

    # The raw signal before preprocessing, resampled to 100 Hz for display only.
    # Held separately so the waveform figure shows millivolts where the units
    # are known, rather than the normalised values the model consumes.
    display_signal: np.ndarray

    source_format: str
    sampling_frequency: float
    n_samples_original: int
    duration_seconds: float
    lead_names: tuple[str, ...]
    units: Optional[str] = None
    record_name: Optional[str] = None
    notes: list[str] = field(default_factory=list)

    @property
    def tensor_input(self) -> np.ndarray:
        """``(1, 12, 1000)`` — batch of one, channels first, as trained."""
        return self.signal.T[np.newaxis, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "format": self.source_format,
            "record_name": self.record_name,
            "sampling_frequency_hz": self.sampling_frequency,
            "resampled_to_hz": ECG_TARGET_FS,
            "samples_original": self.n_samples_original,
            "samples_analysed": int(self.signal.shape[0]),
            "duration_seconds": round(self.duration_seconds, 3),
            "lead_names": list(self.lead_names),
            "lead_order_matches_training": (
                tuple(self.lead_names) == tuple(ECG_LEAD_NAMES)
            ),
            "units": self.units,
        }


# ============================================================
# WFDB HEADER + SIGNAL
# ============================================================
#
# A minimal WFDB reader rather than a dependency on the `wfdb` package. Formats
# 16 and 212 cover PTB-XL, the header grammar used here is a few lines, and the
# alternative is asking the operator to install a package to open the very files
# the model was trained on.


_WFDB_RECORD_LINE = re.compile(
    r"^(?P<name>[\w./-]+)\s+(?P<nsig>\d+)"
    r"(?:\s+(?P<fs>[\d.]+)(?:/[\d.]+)?)?"
    r"(?:\s+(?P<nsamp>\d+))?"
)


@dataclass
class _WfdbHeader:
    record_name: str
    n_signals: int
    sampling_frequency: float
    n_samples: int
    file_names: list[str]
    formats: list[int]
    gains: list[float]
    baselines: list[int]
    adc_zeros: list[int]
    units: list[str]
    lead_names: list[str]


def _parse_wfdb_header(text: str) -> _WfdbHeader:
    lines = [
        line.strip()
        for line in text.splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]

    if not lines:
        raise UnsupportedEcgError("The .hea file is empty.")

    match = _WFDB_RECORD_LINE.match(lines[0])
    if not match:
        raise UnsupportedEcgError(
            f"Could not parse the WFDB record line: {lines[0]!r}"
        )

    n_signals = int(match.group("nsig"))
    fs = float(match.group("fs") or 0) or 0.0
    n_samples = int(match.group("nsamp") or 0)

    file_names: list[str] = []
    formats: list[int] = []
    gains: list[float] = []
    baselines: list[int] = []
    adc_zeros: list[int] = []
    units: list[str] = []
    lead_names: list[str] = []

    for line in lines[1 : 1 + n_signals]:
        fields = line.split()
        if len(fields) < 2:
            raise UnsupportedEcgError(
                f"Malformed WFDB signal line: {line!r}"
            )

        file_names.append(fields[0])

        # format[xN:offset] — the parts after the format number describe
        # interleaving and byte offsets, neither of which PTB-XL uses.
        fmt_field = fields[1]
        formats.append(int(re.match(r"^(\d+)", fmt_field).group(1)))

        # gain(baseline)/units
        gain_field = fields[2] if len(fields) > 2 else "200"
        gain_match = re.match(
            r"^(?P<gain>[-\d.eE+]+)"
            r"(?:\((?P<baseline>-?\d+)\))?"
            r"(?:/(?P<units>\S+))?",
            gain_field,
        )
        gain = float(gain_match.group("gain") or 200.0) if gain_match else 200.0
        gains.append(gain if gain else 200.0)
        baselines.append(
            int(gain_match.group("baseline") or 0) if gain_match else 0
        )
        units.append(
            (gain_match.group("units") if gain_match else None) or "mV"
        )

        adc_zeros.append(int(fields[4]) if len(fields) > 4 else 0)

        # Everything past the numeric fields is the free-text lead description.
        lead_names.append(fields[8] if len(fields) > 8 else "")

    return _WfdbHeader(
        record_name=match.group("name"),
        n_signals=n_signals,
        sampling_frequency=fs,
        n_samples=n_samples,
        file_names=file_names,
        formats=formats,
        gains=gains,
        baselines=baselines,
        adc_zeros=adc_zeros,
        units=units,
        lead_names=lead_names,
    )


def _read_format_16(data: bytes, n_signals: int, n_samples: int) -> np.ndarray:
    """Little-endian int16, samples interleaved across signals."""
    available = len(data) // (2 * n_signals)
    count = min(n_samples, available) if n_samples else available

    if count <= 0:
        raise UnsupportedEcgError(
            "The .dat file holds no complete samples for this many signals."
        )

    flat = np.frombuffer(data, dtype="<i2", count=count * n_signals)
    return flat.reshape(count, n_signals).astype(np.float32)


def _read_format_212(data: bytes, n_signals: int, n_samples: int) -> np.ndarray:
    """
    12-bit packed: every three bytes carry two samples.

    Low byte, then a byte whose low nibble is the first sample's high bits and
    whose high nibble is the second sample's high bits. Both are signed 12-bit.
    """
    n_pairs = len(data) // 3
    raw = np.frombuffer(data, dtype=np.uint8, count=n_pairs * 3)
    raw = raw.reshape(n_pairs, 3).astype(np.int32)

    first = raw[:, 0] | ((raw[:, 1] & 0x0F) << 8)
    second = raw[:, 2] | ((raw[:, 1] >> 4) << 8)

    first[first > 2047] -= 4096
    second[second > 2047] -= 4096

    flat = np.empty(n_pairs * 2, dtype=np.float32)
    flat[0::2] = first
    flat[1::2] = second

    usable = (len(flat) // n_signals) * n_signals
    samples = flat[:usable].reshape(-1, n_signals)

    if n_samples:
        samples = samples[:n_samples]

    return samples


def _decode_wfdb(header: _WfdbHeader, signal_bytes: bytes) -> tuple[np.ndarray, str]:
    fmt = header.formats[0]

    if len(set(header.formats)) != 1:
        raise UnsupportedEcgError(
            "This WFDB record mixes storage formats across leads, which this "
            "reader does not handle. Convert it to CSV first."
        )

    if fmt == 16:
        raw = _read_format_16(signal_bytes, header.n_signals, header.n_samples)
    elif fmt == 212:
        raw = _read_format_212(signal_bytes, header.n_signals, header.n_samples)
    else:
        raise UnsupportedEcgError(
            f"WFDB storage format {fmt} is not supported (16 and 212 are). "
            "Export the record as CSV and upload that instead."
        )

    # ADC counts -> physical units. Baseline first, then gain, per lead.
    gains = np.asarray(header.gains, dtype=np.float32)
    baselines = np.asarray(header.baselines, dtype=np.float32)

    physical = (raw - baselines[np.newaxis, :]) / gains[np.newaxis, :]

    return physical.astype(np.float32), header.units[0] if header.units else "mV"


# ============================================================
# TEXT AND ARRAY READERS
# ============================================================


def _looks_like_header(row: Sequence[str]) -> bool:
    """A row is a header if not one cell parses as a number."""
    for cell in row:
        try:
            float(cell)
        except ValueError:
            return True
    return False


def _split_row(line: str) -> list[str]:
    if "\t" in line:
        return [cell.strip() for cell in line.split("\t")]
    if "," in line:
        return [cell.strip() for cell in line.split(",")]
    return line.split()


def _read_delimited(data: bytes) -> tuple[np.ndarray, Optional[list[str]]]:
    text = data.decode("utf-8", errors="replace")
    lines = [line for line in text.splitlines() if line.strip()]

    if not lines:
        raise UnsupportedEcgError("The file contains no data rows.")

    header: Optional[list[str]] = None
    first = _split_row(lines[0])

    if _looks_like_header(first):
        header = first
        lines = lines[1:]

    if not lines:
        raise UnsupportedEcgError(
            "The file has a header row but no samples underneath it."
        )

    rows: list[list[float]] = []
    for number, line in enumerate(lines, start=2 if header else 1):
        cells = _split_row(line)
        try:
            rows.append([float(cell) for cell in cells])
        except ValueError as error:
            raise UnsupportedEcgError(
                f"Row {number} contains a value that is not a number: {error}"
            ) from error

    widths = {len(row) for row in rows}
    if len(widths) != 1:
        raise UnsupportedEcgError(
            f"Rows have inconsistent lengths ({sorted(widths)}). Every row must "
            "hold one sample for each of the 12 leads."
        )

    return np.asarray(rows, dtype=np.float32), header


def _read_npy(data: bytes) -> np.ndarray:
    try:
        array = np.load(io.BytesIO(data), allow_pickle=False)
    except Exception as error:
        raise UnsupportedEcgError(
            f"Could not read the .npy file: {error}"
        ) from error

    if array.ndim == 3 and array.shape[0] == 1:
        # A cached batch of one, which is how the notebook stored samples.
        array = array[0]

    if array.ndim != 2:
        raise UnsupportedEcgError(
            f"Expected a 2-D array, got shape {array.shape}."
        )

    return np.asarray(array, dtype=np.float32)


def _read_json(data: bytes) -> tuple[np.ndarray, list[str], Optional[float], Optional[str]]:
    try:
        payload = json.loads(data.decode("utf-8", errors="replace"))
    except Exception as error:
        raise UnsupportedEcgError(f"Could not parse the JSON: {error}") from error

    if not isinstance(payload, dict):
        raise UnsupportedEcgError(
            "The JSON must be an object with a 'leads' mapping."
        )

    leads = payload.get("leads") or payload.get("signals")

    if not isinstance(leads, dict) or not leads:
        raise UnsupportedEcgError(
            'The JSON must contain "leads": {"I": [...], "II": [...], ...}.'
        )

    names = list(leads.keys())
    lengths = {len(leads[name]) for name in names}

    if len(lengths) != 1:
        raise UnsupportedEcgError(
            f"Lead traces have different lengths ({sorted(lengths)})."
        )

    matrix = np.asarray(
        [leads[name] for name in names], dtype=np.float32
    ).T

    fs = payload.get("sampling_frequency") or payload.get("fs")

    return (
        matrix,
        names,
        float(fs) if fs else None,
        payload.get("units"),
    )


# ============================================================
# ORIENTATION AND FREQUENCY
# ============================================================


def _orient_leads_last(array: np.ndarray) -> tuple[np.ndarray, bool]:
    """
    Return ``(samples, leads)``, transposing if the array arrived the other way.

    Unambiguous whenever exactly one axis has length 12, which is every real
    12-lead recording — 12 samples would be 0.12 seconds.
    """
    if array.shape[1] == ECG_IN_CHANNELS and array.shape[0] != ECG_IN_CHANNELS:
        return array, False

    if array.shape[0] == ECG_IN_CHANNELS and array.shape[1] != ECG_IN_CHANNELS:
        return array.T, True

    if array.shape == (ECG_IN_CHANNELS, ECG_IN_CHANNELS):
        raise UnsupportedEcgError(
            "A 12x12 array is ambiguous — there is no way to tell samples from "
            "leads. Supply a longer recording."
        )

    raise UnsupportedEcgError(
        f"Neither axis of this {array.shape[0]}x{array.shape[1]} array has "
        f"length {ECG_IN_CHANNELS}. This model reads standard 12-lead ECGs "
        "only, and cannot be run on a different lead set."
    )


def _infer_sampling_frequency(n_samples: int) -> tuple[float, str]:
    """
    Recover the rate from the sample count, for formats that carry no metadata.

    Only the two PTB-XL record lengths are treated as knowable. Anything else
    is a hard error, because a wrong rate does not fail — it silently produces
    a differently-shaped signal and five plausible probabilities.
    """
    known = {
        1000: (100.0, "1000 samples matches a 10-second record at 100 Hz."),
        5000: (500.0, "5000 samples matches a 10-second record at 500 Hz."),
    }

    if n_samples in known:
        return known[n_samples]

    raise UnsupportedEcgError(
        f"This format carries no sampling frequency, and {n_samples} samples "
        "does not match a standard 10-second PTB-XL record (1000 at 100 Hz or "
        "5000 at 500 Hz). Pass the rate explicitly with the "
        "sampling_frequency parameter — guessing it would rescale the whole "
        "recording in time and the result would look entirely normal."
    )


# ============================================================
# ENTRY POINT
# ============================================================


def _suffix_of(filename: str) -> str:
    return PurePosixPath(filename.lower()).suffix


def load_ecg(
    filename: str,
    data: bytes,
    sampling_frequency: Optional[float] = None,
    companion: Optional[dict[str, bytes]] = None,
) -> LoadedEcg:
    """
    Decode an uploaded ECG and run the training preprocessing chain over it.

    ``companion`` carries the other half of a WFDB pair — upload the ``.hea``
    and pass ``{"HR00025.dat": <bytes>}``, or upload a ``.zip`` holding both.
    """
    notes: list[str] = []
    suffix = _suffix_of(filename)
    companion = companion or {}

    lead_names: Optional[list[str]] = None
    units: Optional[str] = None
    record_name: Optional[str] = None
    declared_fs: Optional[float] = sampling_frequency
    source_format = suffix.lstrip(".") or "unknown"

    # ---- WFDB (or a zip containing one) -------------------------
    if suffix == ".zip":
        matrix, lead_names, declared_fs, units, record_name, extra = _load_zip(
            data, sampling_frequency
        )
        notes.extend(extra)
        source_format = "wfdb (zip)"

    elif suffix == ".hea":
        header = _parse_wfdb_header(data.decode("utf-8", errors="replace"))
        signal_bytes = _find_companion(header, companion)

        matrix, units = _decode_wfdb(header, signal_bytes)
        lead_names = header.lead_names
        record_name = header.record_name
        declared_fs = sampling_frequency or header.sampling_frequency or None
        source_format = f"wfdb (format {header.formats[0]})"

    elif suffix == ".dat":
        raise UnsupportedEcgError(
            "A .dat file is raw samples with no description of how they are "
            "laid out — number of leads, storage format, gain and sampling "
            "rate all live in the matching .hea file. Upload the .hea (with "
            "the .dat alongside it, or both in a .zip)."
        )

    elif suffix == ".npy":
        matrix = _read_npy(data)
        source_format = "npy"

    elif suffix == ".json":
        matrix, lead_names, json_fs, units = _read_json(data)
        declared_fs = sampling_frequency or json_fs
        source_format = "json"

    elif suffix in (".csv", ".tsv", ".txt"):
        matrix, header_row = _read_delimited(data)
        if header_row:
            lead_names = header_row
        source_format = suffix.lstrip(".")

    else:
        raise UnsupportedEcgError(
            f"Unsupported ECG file type {suffix or '(none)'}. Accepted: WFDB "
            "(.hea with its .dat, or a .zip of both), .csv, .tsv, .txt, .npy "
            "and .json."
        )

    # ---- drop a time axis, then orient, then validate -----------
    #
    # This has to happen BEFORE the orientation check, not after: a 13-column
    # export has no axis of length 12, so the orienter would reject it outright
    # with a message about lead counts that sends the operator looking for the
    # wrong problem. Only header-bearing formats reach this, which is exactly
    # the set where a time column is identifiable.
    if lead_names and len(lead_names) == matrix.shape[1] == ECG_IN_CHANNELS + 1:
        matrix, lead_names, dropped = _drop_time_column(matrix, lead_names)
        if dropped:
            notes.append(
                f"Dropped the {dropped} column, which is a time axis rather "
                "than a lead."
            )

    # When the columns are labelled and there are still the wrong number of
    # them, say which labels are the problem. _orient_leads_last would also
    # reject this, but only in terms of array axes, and "neither axis has
    # length 12" is a poor way to tell someone their export has a V7 column in
    # it. The labels are right here; use them.
    if lead_names and len(lead_names) == matrix.shape[1] != ECG_IN_CHANNELS:
        recognised = {name.upper() for name in ECG_LEAD_NAMES}
        surplus = [name for name in lead_names if name.strip().upper() not in recognised]
        detail = (
            f" Unexpected column(s): {', '.join(surplus)}."
            if surplus
            else ""
        )
        raise UnsupportedEcgError(
            f"The file has {len(lead_names)} labelled columns; this model reads "
            f"exactly {ECG_IN_CHANNELS} leads "
            f"({', '.join(ECG_LEAD_NAMES)}).{detail} Dropping or synthesising "
            "columns to reach 12 would change what the model is looking at, so "
            "export the standard 12 and upload again."
        )

    matrix, transposed = _orient_leads_last(matrix)

    if transposed:
        notes.append(
            "The array was stored leads-first and has been transposed to "
            "(samples, leads)."
        )
        if lead_names and len(lead_names) != ECG_IN_CHANNELS:
            # A header row belongs to the other axis, so it describes samples.
            lead_names = None

    if matrix.shape[1] != ECG_IN_CHANNELS:
        raise UnsupportedEcgError(
            f"This recording has {matrix.shape[1]} leads. The model's first "
            f"convolution takes exactly {ECG_IN_CHANNELS} channels in the "
            "standard order, so a different lead set cannot be run — not even "
            "by padding, which would feed it flat traces as if they were "
            "recorded."
        )

    if matrix.shape[0] < 2:
        raise UnsupportedEcgError(
            f"Only {matrix.shape[0]} sample(s) per lead. That is not a "
            "recording."
        )

    # ---- sampling frequency -------------------------------------
    if declared_fs and declared_fs > 0:
        fs = float(declared_fs)
    else:
        fs, reason = _infer_sampling_frequency(matrix.shape[0])
        notes.append(
            f"No sampling frequency was supplied; assumed {fs:g} Hz. {reason}"
        )

    duration = matrix.shape[0] / fs

    # ---- lead names ---------------------------------------------
    resolved_leads = _resolve_lead_names(lead_names, notes)

    # ---- warn about the things that change the answer -----------
    if not bandpass_is_applicable(matrix, fs):
        notes.append(
            f"The {ECG_BANDPASS_LOW_HZ:g}-{ECG_BANDPASS_HIGH_HZ:g} Hz bandpass "
            "could not be applied to this recording (too short, or its "
            "sampling rate is too low for a 40 Hz cutoff), so it reaches the "
            "model unfiltered. Every training example was filtered, so treat "
            "the output as provisional."
        )

    if abs(duration - 10.0) > 0.05:
        if duration > 10.0:
            notes.append(
                f"The recording is {duration:.2f} s long and the model reads "
                "10 s, so only the first 10 seconds were analysed."
            )
        else:
            notes.append(
                f"The recording is only {duration:.2f} s long. The remaining "
                f"{10.0 - duration:.2f} s were zero-padded to reach the "
                "model's fixed 10-second input — that padding is silence the "
                "patient never produced, and it dilutes the evidence the "
                "model has to work with."
            )

    # ---- preprocess ---------------------------------------------
    processed = preprocess_ecg(matrix, fs)

    # A display copy: same time base as `processed` so the saliency overlay
    # lines up, but in the original units rather than IQR-standardised.
    display = fix_length(resample_ecg(matrix, int(fs), ECG_TARGET_FS))

    return LoadedEcg(
        signal=processed,
        display_signal=display,
        source_format=source_format,
        sampling_frequency=fs,
        n_samples_original=int(matrix.shape[0]),
        duration_seconds=duration,
        lead_names=resolved_leads,
        units=units,
        record_name=record_name,
        notes=notes,
    )


_TIME_COLUMN_NAMES = frozenset({
    "time", "times", "time_s", "time_sec", "time_seconds", "t",
    "second", "seconds", "sample", "samples", "index", "n", "",
})


def _drop_time_column(
    matrix: np.ndarray,
    lead_names: Sequence[str],
) -> tuple[np.ndarray, list[str], Optional[str]]:
    """
    Remove a leading or trailing time/index column, if that is what it is.

    Only ever driven by the header label. Guessing from the values would be
    tempting — a time axis is monotonic — but lead III during a long
    ST-elevation can be monotonic too across a short window, and dropping a
    real lead would leave eleven traces silently padded to twelve.
    """
    names = [str(name).strip() for name in lead_names]

    if names[0].lower() in _TIME_COLUMN_NAMES:
        return matrix[:, 1:], names[1:], f"leading '{names[0]}'"

    if names[-1].lower() in _TIME_COLUMN_NAMES:
        return matrix[:, :-1], names[:-1], f"trailing '{names[-1]}'"

    return matrix, names, None


def _resolve_lead_names(
    lead_names: Optional[Sequence[str]],
    notes: list[str],
) -> tuple[str, ...]:
    """
    Check supplied lead names against the training order without reordering.

    A mismatch is reported and the file is still analysed, because the most
    common cause is cosmetic (``Lead I`` versus ``I``, or lowercase ``v1``).
    A genuine reordering is a different problem, and one the operator has to
    resolve — silently permuting columns to match would produce a clean-looking
    result from leads the model was never given.
    """
    if not lead_names:
        return tuple(ECG_LEAD_NAMES)

    cleaned = [
        re.sub(r"^lead[\s_-]*", "", str(name).strip(), flags=re.IGNORECASE)
        for name in lead_names
    ]

    if len(cleaned) != ECG_IN_CHANNELS:
        return tuple(ECG_LEAD_NAMES)

    canonical = {name.lower(): name for name in ECG_LEAD_NAMES}
    resolved = [canonical.get(name.lower(), name) for name in cleaned]

    if tuple(resolved) != tuple(ECG_LEAD_NAMES):
        if sorted(n.lower() for n in resolved) == sorted(
            n.lower() for n in ECG_LEAD_NAMES
        ):
            notes.append(
                "The file's lead order is "
                + ", ".join(resolved)
                + " — the model expects "
                + ", ".join(ECG_LEAD_NAMES)
                + ". The columns were NOT reordered, because a silent "
                "permutation would give you a confident result computed from "
                "the wrong leads. Reorder them and upload again."
            )
        else:
            notes.append(
                "The file's lead labels ("
                + ", ".join(resolved)
                + ") do not match the standard 12-lead set, so they are "
                "reported as given and the columns are assumed to already be "
                "in the order "
                + ", ".join(ECG_LEAD_NAMES)
                + "."
            )

    return tuple(resolved)


def _find_companion(
    header: _WfdbHeader,
    companion: dict[str, bytes],
) -> bytes:
    """Locate the .dat that goes with a .hea, by name then by extension."""
    wanted = {name.lower() for name in header.file_names}

    for name, payload in companion.items():
        if PurePosixPath(name.lower()).name in wanted:
            return payload

    for name, payload in companion.items():
        if name.lower().endswith(".dat"):
            return payload

    raise UnsupportedEcgError(
        "The header names "
        + ", ".join(header.file_names)
        + " but no matching signal file was uploaded. WFDB records are two "
        "files: send the .dat alongside the .hea, or zip both together."
    )


def _load_zip(
    data: bytes,
    sampling_frequency: Optional[float],
) -> tuple[np.ndarray, list[str], Optional[float], Optional[str], Optional[str], list[str]]:
    notes: list[str] = []

    try:
        archive = zipfile.ZipFile(io.BytesIO(data))
    except Exception as error:
        raise UnsupportedEcgError(f"Could not open the zip: {error}") from error

    members = [
        name for name in archive.namelist()
        if not name.endswith("/") and "__MACOSX" not in name
    ]

    headers = [name for name in members if name.lower().endswith(".hea")]

    if not headers:
        raise UnsupportedEcgError(
            "The zip contains no .hea file, so there is no WFDB record in it. "
            "Found: " + (", ".join(members[:10]) or "nothing")
        )

    if len(headers) > 1:
        notes.append(
            f"The zip holds {len(headers)} records; the first, "
            f"{headers[0]}, was analysed."
        )

    header = _parse_wfdb_header(
        archive.read(headers[0]).decode("utf-8", errors="replace")
    )

    companion = {
        name: archive.read(name)
        for name in members
        if name.lower().endswith(".dat")
    }

    matrix, units = _decode_wfdb(header, _find_companion(header, companion))

    return (
        matrix,
        header.lead_names,
        sampling_frequency or header.sampling_frequency or None,
        units,
        header.record_name,
        notes,
    )
