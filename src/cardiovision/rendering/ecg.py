"""
CardioVision AI — server-side rendering of the 12-lead ECG.

Two figures, both emitted as SVG text and offered to the API as base64 data
URLs so the frontend can drop them into an ``<img>`` exactly as it does the echo
PNGs:

``render_ecg_strip``        the 12-lead waveform with a per-lead saliency band
``render_lead_attribution`` the lead-importance ranking

Four decisions worth knowing before changing anything here.

**SVG rather than PNG.** An ECG is read by zooming into millimetre-scale detail
— ST elevation, Q-wave width, P-wave morphology. Raster output blurs at exactly
the magnification where the reading happens. Vector output stays sharp, the text
stays selectable, and it keeps matplotlib a notebook dependency rather than a
runtime one. The cost is payload: 12000 plotted points is roughly 150 KB of
markup, which is acceptable for a local deployment and would want thinning for
anything served over a network.

**Twelve full-length rows, not the 4x3 clinical printout.** A printed ECG shows
each lead for 2.5 s in a 4x3 grid. The model consumed all 1000 samples of all 12
leads, and the saliency is defined over that whole extent, so a 4x3 layout would
display a quarter of each lead's attribution while looking complete. The strip is
laid out the way the model saw it.

**One amplitude scale shared by all 12 leads.** Per-lead autoscaling makes every
row look well-formed and destroys amplitude comparison between leads. Amplitude
is diagnostic: hypertrophy criteria are amplitude criteria, and HYP is both a
target class and the weakest one this model has. Autoscaling would remove the
reader's ability to check by eye precisely the class most in need of checking.
Traces are clipped to their row when they exceed it, and the clipping is stated
rather than left to look like flat-topped morphology.

**Millivolts only when the source said millivolts.** These figures plot
``LoadedEcg.display_signal``, which is resampled but neither filtered nor
normalised, so its units are whatever the file declared. The array the model
consumes is robust median/IQR normalised into dimensionless IQR multiples; if it
were ever plotted here, the mV grid would be a fabrication. When the source
declares no units the grid is labelled in raw sample values and no calibration
is claimed.
"""

from __future__ import annotations

import warnings
from typing import Optional, Sequence

import numpy as np

from cardiovision.config import (
    ECG_CLASS_LABELS,
    ECG_IN_CHANNELS,
    ECG_LEAD_NAMES,
    ECG_TARGET_FS,
)
from cardiovision.rendering.primitives import (
    apply_jet,
    jet_hex,
    to_png_data_url,
    to_svg_data_url,
    xml_text,
)

# ============================================================
# GEOMETRY
# ============================================================
#
# One large grid square is 0.2 s wide, as on ECG paper, and the small squares
# divide it into five 0.04 s columns. Everything horizontal derives from
# PLOT_WIDTH; everything vertical derives from LARGE_SQUARE so the grid stays
# square and one large square means a fixed amplitude step.

PLOT_WIDTH = 860.0                 # px for the full record
GUTTER = 56.0                      # lead labels
MARGIN_RIGHT = 20.0
HEADER_HEIGHT = 64.0
FOOTER_HEIGHT = 92.0

LARGE_SQUARE = PLOT_WIDTH * 0.02   # 0.2 s of 10 s => 17.2 px
SMALL_SQUARE = LARGE_SQUARE / 5.0

# Three large squares above and below the baseline, i.e. six per row, which is
# 30 mm of paper — squarely within the range real ECG paper allots to a lead.
#
# The count is what decides the amplitude step, so it decides the gain. At the
# standard 10 mm/mV a large square is 0.5 mV, so three of them hold a 1.5 mV
# deflection: taller than an ordinary R wave, which is the case that has to be
# displayed at the gain a reader's eye is calibrated to. Two squares were tried
# first and are not enough — they cap the row at 1.0 mV, so a perfectly normal
# 1.2 mV R wave forced the step to 1.0 mV and quietly halved the gain to
# 5 mm/mV. Genuinely high voltages still step down, and the footer says so.
TRACE_HALF = LARGE_SQUARE * 3.0
TRACE_HEIGHT = TRACE_HALF * 2.0
BAND_HEIGHT = 6.0                  # saliency heat band
ROW_GAP = 4.0
ROW_HEIGHT = TRACE_HEIGHT + BAND_HEIGHT + ROW_GAP

# ECG paper is printed on a red-orange grid. Keeping that convention makes the
# figure legible as an ECG at a glance instead of as a generic line chart.
GRID_MINOR = "#f3c9c2"
GRID_MAJOR = "#e39a8f"
PAPER = "#fffafa"
TRACE_COLOR = "#141821"
INK = "#1f2937"
MUTED = "#6b7280"
RULE = "#d8dee9"
BADGE = "#b91c1c"

# Units the display signal may declare, and the factor that takes them to mV.
_VOLTAGE_UNITS = {
    "mv": 1.0, "millivolt": 1.0, "millivolts": 1.0,
    "uv": 1e-3, "µv": 1e-3, "μv": 1e-3,
    "microvolt": 1e-3, "microvolts": 1e-3,
    "v": 1e3, "volt": 1e3, "volts": 1e3,
}

# Amplitude per large square, in mV. 0.5 is the familiar clinical setting
# (10 mm/mV); the coarser rungs exist for recordings that exceed it and the finer
# ones so a badly scaled export renders as a waveform rather than a flat line.
_MV_STEPS = (
    0.01, 0.02, 0.025, 0.05, 0.1, 0.2, 0.25, 0.5, 1.0, 2.0, 2.5, 5.0, 10.0,
)


# ============================================================
# SCALING
# ============================================================


def _calibration(units: Optional[str]) -> tuple[Optional[float], str]:
    """
    Resolve declared units to (factor into mV, axis label).

    A factor of None means the source declared nothing recognisable, in which
    case the figure plots raw sample values and says so.
    """
    if not units:
        return None, "raw units"

    factor = _VOLTAGE_UNITS.get(str(units).strip().lower())
    if factor is None:
        return None, f"{units} (uncalibrated)"

    return factor, "mV"


def _nice_step(value: float, calibrated: bool) -> float:
    """
    Amplitude represented by one large grid square.

    Picked so three steps cover the signal, matching the three large squares of
    row height either side of the baseline. For a normal ECG peaking anywhere up
    to 1.5 mV that lands on 0.5 mV per square, i.e. the standard 10 mm/mV gain.

    The divisor is tied to ``TRACE_HALF`` and must move with it: it is the same
    number of squares, expressed once in pixels and once in millivolts.
    """
    target = max(float(value), 1e-9) / (TRACE_HALF / LARGE_SQUARE)

    if calibrated:
        for step in _MV_STEPS:
            if step >= target:
                return step
        return _MV_STEPS[-1]

    exponent = np.floor(np.log10(target))
    for multiple in (1.0, 2.0, 2.5, 5.0, 10.0):
        step = multiple * float(10.0 ** exponent)
        if step >= target:
            return step
    return float(10.0 ** (exponent + 1))


def _robust_extent(signal: np.ndarray) -> float:
    """
    Amplitude the scale has to cover: robust to artefacts, not to baseline.

    The plain maximum would let one pacing spike or one lead-off artefact
    flatten all twelve traces into straight lines. A percentile taken over
    every sample of every lead has the opposite failure and is the worse of
    the two: most of an ECG is baseline, so even the 99.5th percentile of the
    pooled samples sits below the R waves and amputates them on an entirely
    normal recording.

    So the estimate is built per lead and then pooled:

    * Within a lead, the peak is the 99.8th percentile of |x|. At 1000 samples
      that is about the second-largest sample, which rejects a single-sample
      artefact while a genuine QRS — 4 to 8 samples wide at 100 Hz — survives
      it comfortably.
    * Across leads, the scale is the largest lead peak, capped at three times
      the median lead peak. One saturated or disconnected lead therefore
      cannot decide the scale for the other eleven.

    Anything past the cap is clipped, and the footer says so; that is the
    trade this function exists to make, and it is stated rather than hidden.
    """
    array = np.asarray(signal, dtype=np.float64)
    if array.ndim == 1:
        array = array[:, np.newaxis]

    magnitude = np.abs(array)
    magnitude[~np.isfinite(magnitude)] = np.nan

    if not np.any(np.isfinite(magnitude)):
        return 1.0

    # One peak per lead. Leads that are entirely non-finite drop out rather
    # than poisoning the median with a nan. numpy warns about the all-NaN
    # slice such a lead produces; that case is handled two lines below, so the
    # warning is suppressed rather than left to surface as if it mattered.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        peaks = np.nanpercentile(magnitude, 99.8, axis=0)

    peaks = peaks[np.isfinite(peaks)]
    peaks = peaks[peaks > 1e-9]

    if peaks.size == 0:
        # Flatline, or amplitudes below the numerical floor. 1.0 keeps the
        # grid on a sane clinical step instead of dividing by ~zero.
        return 1.0

    cap = 3.0 * float(np.median(peaks))
    extent = float(min(peaks.max(), cap))

    return extent if extent > 1e-9 else 1.0


def _format_step(step: float) -> str:
    """Grid step as the shortest string that does not lose a digit."""
    for places in (0, 1, 2, 3):
        text = f"{step:.{places}f}"
        if abs(float(text) - step) < 1e-9:
            return text
    return f"{step:g}"


# ============================================================
# THE 12-LEAD STRIP
# ============================================================


def render_ecg_strip(
    display_signal: np.ndarray,
    saliency: Optional[np.ndarray] = None,
    *,
    lead_names: Sequence[str] = ECG_LEAD_NAMES,
    units: Optional[str] = None,
    sampling_frequency: float = ECG_TARGET_FS,
    record_name: Optional[str] = None,
    saliency_class: Optional[str] = None,
    ranked_leads: Sequence[str] = (),
    saliency_available: bool = True,
) -> str:
    """
    The 12-lead strip, as SVG markup.

    ``display_signal`` is ``(n_samples, 12)`` in whatever units the source
    declared — pass ``LoadedEcg.display_signal``, not the model's normalised
    input. ``saliency`` is ``(12, n_samples)`` in [0, 1], already normalised
    per lead by the classifier, or None when no gradient was available.

    ``ranked_leads`` is the lead-attribution order; the top three get a rank
    badge so the strip and the ranking figure agree with each other.
    """
    signal = np.asarray(display_signal, dtype=np.float32)

    if signal.ndim != 2 or signal.shape[1] != ECG_IN_CHANNELS:
        raise ValueError(
            f"Expected a (samples, {ECG_IN_CHANNELS}) display signal, "
            f"got {signal.shape}."
        )

    n_samples = int(signal.shape[0])
    if n_samples < 2:
        raise ValueError(f"Need at least two samples to draw a trace, got {n_samples}.")

    if len(lead_names) != ECG_IN_CHANNELS:
        raise ValueError(
            f"Expected {ECG_IN_CHANNELS} lead names, got {len(lead_names)}."
        )

    heat = None
    if saliency is not None and saliency_available:
        heat = np.asarray(saliency, dtype=np.float32)
        if heat.shape != (ECG_IN_CHANNELS, n_samples):
            raise ValueError(
                f"Saliency must be ({ECG_IN_CHANNELS}, {n_samples}) to line up "
                f"with the signal, got {heat.shape}."
            )

    fs = float(sampling_frequency) or float(ECG_TARGET_FS)
    duration = n_samples / fs

    factor, unit_label = _calibration(units)
    plotted = signal * factor if factor is not None else signal

    step = _nice_step(_robust_extent(plotted), calibrated=factor is not None)

    # Half the row, in signal units. Derived from the geometry rather than
    # written down, so the grid, the trace and the caption cannot disagree.
    half_range = step * (TRACE_HALF / LARGE_SQUARE)

    width = GUTTER + PLOT_WIDTH + MARGIN_RIGHT
    plot_height = ROW_HEIGHT * ECG_IN_CHANNELS
    height = HEADER_HEIGHT + plot_height + FOOTER_HEIGHT

    rank = {name: index for index, name in enumerate(ranked_leads)}

    parts: list[str] = [
        f'<svg xmlns="http://www.w3.org/2000/svg" '
        f'xmlns:xlink="http://www.w3.org/1999/xlink" '
        f'width="{width:.0f}" height="{height:.0f}" '
        f'viewBox="0 0 {width:.0f} {height:.0f}" '
        f'role="img" aria-label="12-lead ECG with per-lead saliency">',
        _strip_defs(),
        f'<rect width="{width:.0f}" height="{height:.0f}" fill="#ffffff"/>',
    ]

    parts.append(
        _strip_header(
            width=width,
            record_name=record_name,
            duration=duration,
            fs=fs,
            n_samples=n_samples,
            saliency_class=saliency_class,
            saliency_available=heat is not None,
        )
    )

    # Paper and grid, drawn once behind every row.
    parts.append(
        f'<g><rect x="{GUTTER:.1f}" y="{HEADER_HEIGHT:.1f}" '
        f'width="{PLOT_WIDTH:.1f}" height="{plot_height:.1f}" fill="{PAPER}"/>'
        f'<rect x="{GUTTER:.1f}" y="{HEADER_HEIGHT:.1f}" '
        f'width="{PLOT_WIDTH:.1f}" height="{plot_height:.1f}" '
        f'fill="url(#ecgGrid)"/></g>'
    )

    clipped: list[str] = []

    for index in range(ECG_IN_CHANNELS):
        row_top = HEADER_HEIGHT + index * ROW_HEIGHT
        baseline = row_top + TRACE_HALF
        name = str(lead_names[index])

        lead_values = plotted[:, index]
        was_clipped = bool(np.any(np.abs(lead_values[np.isfinite(lead_values)])
                                 > half_range))
        if was_clipped:
            clipped.append(name)

        parts.append(
            _lead_row(
                name=name,
                values=lead_values,
                heat=None if heat is None else heat[index],
                row_top=row_top,
                baseline=baseline,
                half_range=half_range,
                step=step,
                n_samples=n_samples,
                fs=fs,
                duration=duration,
                rank=rank.get(name),
                clipped=was_clipped,
            )
        )

    parts.append(_time_axis(HEADER_HEIGHT + plot_height, duration))
    parts.append(
        _strip_footer(
            top=HEADER_HEIGHT + plot_height + 26.0,
            width=width,
            step=step,
            unit_label=unit_label,
            calibrated=factor is not None,
            declared_units=units,
            saliency_available=heat is not None,
            clipped=clipped,
        )
    )

    parts.append("</svg>")
    return "\n".join(parts)


def _strip_defs() -> str:
    """The grid pattern and the saliency legend gradient."""
    stops = "".join(
        f'<stop offset="{position:.2f}" stop-color="{jet_hex(position)}"/>'
        for position in np.linspace(0.0, 1.0, 9)
    )

    return (
        "<defs>"
        f'<pattern id="ecgGrid" width="{LARGE_SQUARE:.4f}" '
        f'height="{LARGE_SQUARE:.4f}" patternUnits="userSpaceOnUse">'
        f'<path d="M {LARGE_SQUARE:.4f} 0 L 0 0 0 {LARGE_SQUARE:.4f}" '
        f'fill="none" stroke="{GRID_MAJOR}" stroke-width="0.7"/>'
        f'<path d="M {SMALL_SQUARE:.4f} 0 V {LARGE_SQUARE:.4f} '
        f'M {SMALL_SQUARE * 2:.4f} 0 V {LARGE_SQUARE:.4f} '
        f'M {SMALL_SQUARE * 3:.4f} 0 V {LARGE_SQUARE:.4f} '
        f'M {SMALL_SQUARE * 4:.4f} 0 V {LARGE_SQUARE:.4f} '
        f'M 0 {SMALL_SQUARE:.4f} H {LARGE_SQUARE:.4f} '
        f'M 0 {SMALL_SQUARE * 2:.4f} H {LARGE_SQUARE:.4f} '
        f'M 0 {SMALL_SQUARE * 3:.4f} H {LARGE_SQUARE:.4f} '
        f'M 0 {SMALL_SQUARE * 4:.4f} H {LARGE_SQUARE:.4f}" '
        f'fill="none" stroke="{GRID_MINOR}" stroke-width="0.4"/>'
        "</pattern>"
        f'<linearGradient id="salRamp" x1="0" y1="0" x2="1" y2="0">{stops}</linearGradient>'
        "</defs>"
    )


def _strip_header(
    *,
    width: float,
    record_name: Optional[str],
    duration: float,
    fs: float,
    n_samples: int,
    saliency_class: Optional[str],
    saliency_available: bool,
) -> str:
    title = xml_text(record_name) if record_name else "12-lead ECG"

    detail = (
        f"{duration:.2f} s &#183; {n_samples} samples at {fs:g} Hz "
        f"&#183; all 12 leads, full length"
    )

    if saliency_available and saliency_class:
        label = ECG_CLASS_LABELS.get(saliency_class, saliency_class)
        attribution = (
            f"Saliency explains: {xml_text(label)} "
            f"({xml_text(saliency_class)})"
        )
    elif saliency_available:
        attribution = "Saliency: highest-probability class"
    else:
        attribution = "Saliency unavailable for this recording"

    return (
        f'<g font-family="Inter, Helvetica, Arial, sans-serif">'
        f'<text x="{GUTTER:.1f}" y="26" font-size="15" font-weight="600" '
        f'fill="{INK}">{title}</text>'
        f'<text x="{GUTTER:.1f}" y="45" font-size="11" fill="{MUTED}">{detail}</text>'
        f'<text x="{width - MARGIN_RIGHT:.1f}" y="26" font-size="11" '
        f'text-anchor="end" fill="{MUTED}">{attribution}</text>'
        f'<line x1="{GUTTER:.1f}" y1="{HEADER_HEIGHT - 8:.1f}" '
        f'x2="{width - MARGIN_RIGHT:.1f}" y2="{HEADER_HEIGHT - 8:.1f}" '
        f'stroke="{RULE}" stroke-width="1"/>'
        f"</g>"
    )


def _lead_row(
    *,
    name: str,
    values: np.ndarray,
    heat: Optional[np.ndarray],
    row_top: float,
    baseline: float,
    half_range: float,
    step: float,
    n_samples: int,
    fs: float,
    duration: float,
    rank: Optional[int],
    clipped: bool,
) -> str:
    parts: list[str] = ["<g>"]

    # The zero reference for this lead.
    parts.append(
        f'<line x1="{GUTTER:.1f}" y1="{baseline:.2f}" '
        f'x2="{GUTTER + PLOT_WIDTH:.1f}" y2="{baseline:.2f}" '
        f'stroke="{GRID_MAJOR}" stroke-width="0.9"/>'
    )

    # The trace. Clipped to the row so a spike cannot bleed into the lead above.
    finite = np.nan_to_num(values, nan=0.0, posinf=half_range, neginf=-half_range)
    bounded = np.clip(finite, -half_range, half_range)

    x = GUTTER + (np.arange(n_samples, dtype=np.float64) / fs) / duration * PLOT_WIDTH
    y = baseline - (bounded / half_range) * TRACE_HALF

    points = " ".join(
        f"{px:.1f},{py:.1f}" for px, py in zip(x.tolist(), y.tolist())
    )
    parts.append(
        f'<polyline points="{points}" fill="none" stroke="{TRACE_COLOR}" '
        f'stroke-width="1.1" stroke-linejoin="round" stroke-linecap="round"/>'
    )

    # Saliency band, hard against the bottom of the row.
    band_top = row_top + TRACE_HEIGHT + 1.0
    if heat is not None:
        parts.append(
            f'<image x="{GUTTER:.1f}" y="{band_top:.1f}" '
            f'width="{PLOT_WIDTH:.1f}" height="{BAND_HEIGHT:.1f}" '
            f'preserveAspectRatio="none" xlink:href="{_heat_strip(heat)}"/>'
        )
    else:
        parts.append(
            f'<rect x="{GUTTER:.1f}" y="{band_top:.1f}" '
            f'width="{PLOT_WIDTH:.1f}" height="{BAND_HEIGHT:.1f}" '
            f'fill="#f1f2f4" stroke="{RULE}" stroke-width="0.5"/>'
        )

    # Lead label, plus a rank badge for the three leads that mattered most.
    parts.append(
        f'<text x="{GUTTER - 10:.1f}" y="{baseline + 4:.1f}" '
        f'font-family="Inter, Helvetica, Arial, sans-serif" font-size="12" '
        f'font-weight="600" text-anchor="end" fill="{INK}">{xml_text(name)}</text>'
    )

    if rank is not None and rank < 3:
        parts.append(
            f'<text x="{GUTTER - 10:.1f}" y="{baseline + 17:.1f}" '
            f'font-family="Inter, Helvetica, Arial, sans-serif" font-size="8.5" '
            f'text-anchor="end" fill="{BADGE}">#{rank + 1}</text>'
        )

    if clipped:
        parts.append(
            f'<text x="{GUTTER + PLOT_WIDTH - 4:.1f}" y="{row_top + 11:.1f}" '
            f'font-family="Inter, Helvetica, Arial, sans-serif" font-size="8.5" '
            f'text-anchor="end" fill="{BADGE}">clipped</text>'
        )

    parts.append("</g>")
    return "".join(parts)


def _heat_strip(heat: np.ndarray) -> str:
    """One lead's saliency as a 1-pixel-tall PNG data URL."""
    row = np.clip(np.nan_to_num(heat, nan=0.0), 0.0, 1.0).reshape(1, -1)
    return to_png_data_url(apply_jet(row), scale=1)


def _time_axis(top: float, duration: float) -> str:
    parts = [
        f'<g font-family="Inter, Helvetica, Arial, sans-serif" font-size="9.5" '
        f'fill="{MUTED}">',
        f'<line x1="{GUTTER:.1f}" y1="{top + 4:.1f}" '
        f'x2="{GUTTER + PLOT_WIDTH:.1f}" y2="{top + 4:.1f}" '
        f'stroke="{RULE}" stroke-width="1"/>',
    ]

    for second in range(int(duration) + 1):
        if second > duration:
            break
        x = GUTTER + (second / duration) * PLOT_WIDTH
        parts.append(
            f'<line x1="{x:.1f}" y1="{top + 4:.1f}" x2="{x:.1f}" '
            f'y2="{top + 9:.1f}" stroke="{RULE}" stroke-width="1"/>'
        )
        parts.append(
            f'<text x="{x:.1f}" y="{top + 20:.1f}" text-anchor="middle">'
            f"{second}s</text>"
        )

    parts.append("</g>")
    return "".join(parts)


def _strip_footer(
    *,
    top: float,
    width: float,
    step: float,
    unit_label: str,
    calibrated: bool,
    declared_units: Optional[str],
    saliency_available: bool,
    clipped: Sequence[str],
) -> str:
    lines: list[str] = []

    scale = (
        f"Grid: one large square = 0.2 s &#215; {_format_step(step)} "
        f"{xml_text(unit_label)}. One amplitude scale for all 12 leads, so "
        f"heights are comparable between them."
    )
    lines.append(scale)

    if calibrated:
        # A large square is square in pixels and spans 0.2 s by `step` mV, which
        # is 5 mm by 5 mm of ECG paper. That fixes the sweep at 25 mm/s and the
        # gain at 5/step mm/mV, so the familiar 0.5 mV square is 10 mm/mV.
        lines.append(
            f"Aspect ratio matches ECG paper: equivalent to 25 mm/s and "
            f"{5.0 / step:g} mm/mV"
            + ("," if abs(step - 0.5) < 1e-9 else ", not")
            + " the standard clinical gain."
        )
    else:
        lines.append(
            "The source declared no amplitude units"
            + (f' (found "{xml_text(declared_units)}")' if declared_units else "")
            + ", so the vertical scale is raw sample values and no millivolt "
            "calibration is claimed."
        )

    if saliency_available:
        lines.append(
            "Saliency band: each lead is normalised to its own peak, so colour "
            "compares moments within a lead and not one lead against another. "
            "Use the lead ranking for that."
        )
    else:
        lines.append(
            "No gradient was available for this recording, so the bands are "
            "blank rather than showing a flat map that would look like a result."
        )

    if clipped:
        lines.append(
            f"Clipped to fit the row: {xml_text(', '.join(clipped))}. The flat "
            "tops there are the figure's limit, not the recording's."
        )

    parts = [
        f'<g font-family="Inter, Helvetica, Arial, sans-serif" font-size="9.5" '
        f'fill="{MUTED}">'
    ]

    y = top
    if saliency_available:
        parts.append(
            f'<rect x="{GUTTER:.1f}" y="{y - 8:.1f}" width="86" height="8" '
            f'fill="url(#salRamp)" stroke="{RULE}" stroke-width="0.5"/>'
            f'<text x="{GUTTER + 92:.1f}" y="{y - 1:.1f}">low &#8594; high '
            f"attribution</text>"
        )
        y += 15.0

    for line in lines:
        parts.append(f'<text x="{GUTTER:.1f}" y="{y:.1f}">{line}</text>')
        y += 13.0

    parts.append("</g>")
    return "".join(parts)


# ============================================================
# LEAD ATTRIBUTION
# ============================================================


def render_lead_attribution(
    leads: Sequence[object],
    *,
    saliency_class: Optional[str] = None,
) -> str:
    """
    The lead-importance ranking, as SVG markup.

    ``leads`` is the ``LeadAttribution`` list from ``EcgAnalysis`` — already
    sorted, already normalised to the strongest lead. Accepts either the
    dataclass or its ``to_dict()`` form so a stored analysis can be re-rendered
    without rebuilding objects.
    """
    rows = [_attribution_row(item) for item in leads]

    bar_left = 64.0
    bar_width = 420.0
    row_height = 22.0
    width = bar_left + bar_width + 76.0
    header = 58.0
    footer = 46.0
    height = header + row_height * max(len(rows), 1) + footer

    parts: list[str] = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width:.0f}" '
        f'height="{height:.0f}" viewBox="0 0 {width:.0f} {height:.0f}" '
        f'role="img" aria-label="ECG lead attribution ranking">',
        f'<rect width="{width:.0f}" height="{height:.0f}" fill="#ffffff"/>',
        '<g font-family="Inter, Helvetica, Arial, sans-serif">',
        f'<text x="20" y="26" font-size="14" font-weight="600" fill="{INK}">'
        f"Lead attribution</text>",
    ]
    subtitle = (
        f"Mean absolute gradient per lead"
        + (
            f", for {xml_text(ECG_CLASS_LABELS.get(saliency_class, saliency_class))}"
            if saliency_class else ""
        )
        + ", relative to the strongest lead"
    )
    parts.append(
        f'<text x="20" y="44" font-size="10" fill="{MUTED}">{subtitle}</text>'
    )

    if not rows:
        parts.append(
            f'<text x="20" y="{header + 18:.1f}" font-size="11" fill="{MUTED}">'
            "No gradient was available for this recording, so there is no "
            "ranking to show.</text>"
        )
        parts.append("</g></svg>")
        return "\n".join(parts)

    for index, (name, score, raw) in enumerate(rows):
        y = header + index * row_height
        centre = y + row_height / 2.0
        length = max(float(score) * bar_width, 1.0)

        parts.append(
            f'<text x="{bar_left - 10:.1f}" y="{centre + 4:.1f}" font-size="11" '
            f'font-weight="600" text-anchor="end" fill="{INK}">'
            f"{xml_text(name)}</text>"
        )
        parts.append(
            f'<rect x="{bar_left:.1f}" y="{y + 4:.1f}" width="{bar_width:.1f}" '
            f'height="{row_height - 9:.1f}" fill="#f3f4f6"/>'
        )
        parts.append(
            f'<rect x="{bar_left:.1f}" y="{y + 4:.1f}" width="{length:.1f}" '
            f'height="{row_height - 9:.1f}" fill="{jet_hex(float(score))}"/>'
        )
        parts.append(
            f'<text x="{bar_left + bar_width + 10:.1f}" y="{centre + 4:.1f}" '
            f'font-size="10" fill="{MUTED}">{float(score):.2f}'
            + (f'  <tspan fill="#9ca3af">({raw})</tspan>' if raw else "")
            + "</text>"
        )

    parts.append(
        f'<text x="20" y="{height - 24:.1f}" font-size="9.5" fill="{MUTED}">'
        "Computed from raw gradient magnitude, not from the display band above: "
        "per-lead normalisation would rank a</text>"
    )
    parts.append(
        f'<text x="20" y="{height - 12:.1f}" font-size="9.5" fill="{MUTED}">'
        "quiet lead level with a loud one. Attribution shows where the model "
        "looked, which is not the same as where the pathology is.</text>"
    )

    parts.append("</g></svg>")
    return "\n".join(parts)


def _attribution_row(item: object) -> tuple[str, float, str]:
    """Accept a LeadAttribution or its dict form; return (name, score, raw)."""
    if isinstance(item, dict):
        name = item.get("name", "?")
        score = item.get("score", 0.0)
        raw = item.get("raw_score")
    else:
        name = getattr(item, "name", "?")
        score = getattr(item, "score", 0.0)
        raw = getattr(item, "raw_score", None)

    formatted = ""
    if raw is not None:
        try:
            formatted = f"{float(raw):.3g}"
        except (TypeError, ValueError):
            formatted = ""

    return str(name), float(np.clip(float(score), 0.0, 1.0)), formatted


# ============================================================
# THE API PAYLOAD
# ============================================================


def render_ecg_images(
    display_signal: np.ndarray,
    analysis: object,
    *,
    lead_names: Sequence[str] = ECG_LEAD_NAMES,
    units: Optional[str] = None,
    sampling_frequency: float = ECG_TARGET_FS,
    record_name: Optional[str] = None,
) -> dict[str, str]:
    """
    Both figures as data URLs, mirroring ``render_analysis_images`` for echo.

    ``analysis`` is an ``EcgAnalysis``. Taken as a loose object rather than an
    import so this module never pulls torch in behind it — importing a renderer
    should not require the inference stack.
    """
    saliency = getattr(analysis, "saliency", None)
    available = bool(getattr(analysis, "saliency_available", saliency is not None))
    leads = getattr(analysis, "leads", ()) or ()
    saliency_class = getattr(analysis, "saliency_class", None)

    strip = render_ecg_strip(
        display_signal,
        saliency,
        lead_names=lead_names,
        units=units,
        sampling_frequency=sampling_frequency,
        record_name=record_name,
        saliency_class=saliency_class,
        ranked_leads=[
            getattr(lead, "name", lead.get("name") if isinstance(lead, dict) else "?")
            for lead in leads
        ],
        saliency_available=available,
    )

    attribution = render_lead_attribution(
        leads if available else (),
        saliency_class=saliency_class,
    )

    return {
        "strip": to_svg_data_url(strip),
        "lead_attribution": to_svg_data_url(attribution),
    }
