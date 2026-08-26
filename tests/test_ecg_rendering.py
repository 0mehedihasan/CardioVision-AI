#!/usr/bin/env python3
"""
ECG figure rendering.

Checks the SVG the API hands the browser: that it parses, that the geometry puts
each lead where it claims to, that the time and amplitude mappings are the ones
the captions describe, and that every honesty caveat fires when its condition is
met and stays quiet when it is not.

    python3 tests/test_ecg_rendering.py

This is the one part of the ECG path that runs for real here — no torch, no
scipy, no HTTP. The figures below were actually generated and parsed.
"""

from __future__ import annotations

import base64
import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(REPO / "src"))

from cardiovision.config import (  # noqa: E402
    ECG_IN_CHANNELS,
    ECG_INPUT_LENGTH,
    ECG_LEAD_NAMES,
    ECG_TARGET_FS,
)
from cardiovision.rendering.ecg import (  # noqa: E402
    BAND_HEIGHT,
    GUTTER,
    HEADER_HEIGHT,
    PLOT_WIDTH,
    ROW_HEIGHT,
    TRACE_HALF,
    TRACE_HEIGHT,
    _calibration,
    _nice_step,
    _robust_extent,
    render_ecg_images,
    render_ecg_strip,
    render_lead_attribution,
)

SVG_NS = "{http://www.w3.org/2000/svg}"

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
# SYNTHETIC INPUT
# ============================================================


def synthetic_ecg(
    n_samples: int = ECG_INPUT_LENGTH,
    fs: float = ECG_TARGET_FS,
    amplitude: float = 1.2,
) -> np.ndarray:
    """
    A (n_samples, 12) waveform with a recognisable QRS at 1 Hz.

    Not physiological — it exists so the geometry checks have something with a
    known peak location to measure.
    """
    t = np.arange(n_samples) / fs
    beat = np.zeros(n_samples, dtype=np.float32)

    for onset in np.arange(0.5, t[-1], 1.0):
        beat += 0.15 * np.exp(-((t - onset + 0.18) ** 2) / (2 * 0.02 ** 2))   # P
        beat += 1.00 * np.exp(-((t - onset) ** 2) / (2 * 0.008 ** 2))         # R
        beat -= 0.25 * np.exp(-((t - onset - 0.03) ** 2) / (2 * 0.010 ** 2))  # S
        beat += 0.30 * np.exp(-((t - onset - 0.28) ** 2) / (2 * 0.035 ** 2))  # T

    signal = np.zeros((n_samples, ECG_IN_CHANNELS), dtype=np.float32)
    for lead in range(ECG_IN_CHANNELS):
        # Lead-dependent gain so the shared-scale checks have something to see.
        signal[:, lead] = beat * amplitude * (0.4 + 0.6 * (lead + 1) / 12.0)

    return signal


def synthetic_saliency(n_samples: int = ECG_INPUT_LENGTH) -> np.ndarray:
    """(12, n) in [0, 1], each lead peaking at a different time."""
    t = np.linspace(0.0, 1.0, n_samples)
    heat = np.zeros((ECG_IN_CHANNELS, n_samples), dtype=np.float32)
    for lead in range(ECG_IN_CHANNELS):
        centre = (lead + 0.5) / ECG_IN_CHANNELS
        heat[lead] = np.exp(-((t - centre) ** 2) / (2 * 0.04 ** 2))
    return heat


def polylines(svg: str) -> list[np.ndarray]:
    """Every polyline in the document, as (n, 2) float arrays."""
    root = ET.fromstring(svg)
    out = []
    for node in root.iter(f"{SVG_NS}polyline"):
        pairs = [
            tuple(float(value) for value in point.split(","))
            for point in node.get("points", "").split()
        ]
        out.append(np.array(pairs, dtype=np.float64))
    return out


def all_text(svg: str) -> str:
    """Concatenated text content, entities already resolved by the parser."""
    root = ET.fromstring(svg)
    return " ".join(
        "".join(node.itertext()) for node in root.iter(f"{SVG_NS}text")
    )


SIGNAL = synthetic_ecg()
HEAT = synthetic_saliency()
STRIP = render_ecg_strip(
    SIGNAL, HEAT, units="mV", record_name="HR00025",
    saliency_class="MI", ranked_leads=["V1", "II", "aVF", "V3"],
)

# ============================================================
print("\n=== 1. the figures are well-formed SVG ===")
# ============================================================

try:
    root = ET.fromstring(STRIP)
    parsed = True
    parse_error = ""
except ET.ParseError as error:
    root = None
    parsed = False
    parse_error = str(error)

check("the strip parses as XML", parsed, parse_error)
check("the root element is <svg>",
      parsed and root.tag == f"{SVG_NS}svg", root.tag if parsed else "")
check("it declares an explicit viewBox, so it scales without clipping",
      parsed and root.get("viewBox") is not None, root.get("viewBox") or "")
check("it carries an accessible label rather than being an opaque image",
      parsed and "ECG" in (root.get("aria-label") or ""),
      root.get("aria-label") or "")

attribution_svg = render_lead_attribution(
    [{"name": name, "score": 1.0 - index * 0.08, "raw_score": 0.9 - index * 0.07}
     for index, name in enumerate(ECG_LEAD_NAMES)],
    saliency_class="MI",
)
try:
    ET.fromstring(attribution_svg)
    attribution_parsed = True
except ET.ParseError as error:                      # pragma: no cover
    attribution_parsed = False
    print(f"        {error}")
check("the attribution figure parses too", attribution_parsed)

# ============================================================
print("\n=== 2. all twelve leads are drawn, in order ===")
# ============================================================

traces = polylines(STRIP)
check("there is exactly one trace per lead",
      len(traces) == ECG_IN_CHANNELS, str(len(traces)))
check("each trace keeps every sample — no thinning that would drop a spike",
      all(len(trace) == ECG_INPUT_LENGTH for trace in traces),
      str(sorted({len(trace) for trace in traces})))

text = all_text(STRIP)
positions = [text.find(name) for name in ECG_LEAD_NAMES]
check("every lead is labelled",
      all(position >= 0 for position in positions),
      str([name for name, position in zip(ECG_LEAD_NAMES, positions)
           if position < 0]))
check("labels appear in the trained lead order, I first and V6 last",
      positions == sorted(positions),
      f"I at {positions[0]}, V6 at {positions[-1]}")

# ============================================================
print("\n=== 3. each trace stays inside its own row ===")
# ============================================================
#
# The failure this catches: an unclipped spike drawing across the lead above,
# which would look like a finding in a lead that never had one.

overflow = []
for index, trace in enumerate(traces):
    row_top = HEADER_HEIGHT + index * ROW_HEIGHT
    if trace[:, 1].min() < row_top - 0.01 or trace[:, 1].max() > row_top + TRACE_HEIGHT + 0.01:
        overflow.append(ECG_LEAD_NAMES[index])

check("no trace escapes its row", not overflow, str(overflow))
check("rows are drawn top to bottom in lead order",
      all(traces[i][:, 1].mean() < traces[i + 1][:, 1].mean()
          for i in range(len(traces) - 1)))

# The last sample of a 10 s record at 100 Hz sits at t=9.99 s, so the trace ends
# one sample short of the right edge. That is correct, not a rounding slip, so
# the tolerance here is one sample width rather than a fudged constant.
sample_width = PLOT_WIDTH / ECG_INPUT_LENGTH
check("traces start at the left edge and run to within one sample of the right",
      abs(traces[0][0, 0] - GUTTER) < 0.05
      and abs(traces[0][-1, 0] - (GUTTER + PLOT_WIDTH)) <= sample_width + 0.05,
      f"x from {traces[0][0, 0]:.2f} to {traces[0][-1, 0]:.2f}, "
      f"edge at {GUTTER + PLOT_WIDTH:.2f}")

# ============================================================
print("\n=== 4. the time axis maps where it says it does ===")
# ============================================================

spike = np.zeros((ECG_INPUT_LENGTH, ECG_IN_CHANNELS), dtype=np.float32)
spike[500, 0] = 5.0                      # t = 5.00 s exactly
spike_trace = polylines(render_ecg_strip(spike, units="mV"))[0]

peak_x = float(spike_trace[np.argmin(spike_trace[:, 1]), 0])
expected_x = GUTTER + PLOT_WIDTH * 0.5

check("a spike at t=5.00 s lands at the midpoint of the plot",
      abs(peak_x - expected_x) < 1.5, f"{peak_x:.1f} vs {expected_x:.1f}")
check("a positive deflection goes UP, not down",
      spike_trace[500, 1] < spike_trace[0, 1],
      f"y {spike_trace[500, 1]:.1f} vs baseline {spike_trace[0, 1]:.1f}")

second_ticks = re.findall(r">(\d+)s</text>", STRIP)
check("the axis is ticked once per second, 0 through 10",
      second_ticks == [str(n) for n in range(11)], str(second_ticks))

# ============================================================
print("\n=== 5. one amplitude scale, shared by all twelve leads ===")
# ============================================================
#
# Per-lead autoscaling would make these deflections identical. Hypertrophy is an
# amplitude criterion and HYP is this model's weakest class, so flattening
# amplitude differences would disable exactly the check a reader most needs.

paired = np.zeros((ECG_INPUT_LENGTH, ECG_IN_CHANNELS), dtype=np.float32)
paired[400, 0] = 1.0                     # lead I,  full amplitude
paired[400, 1] = 0.5                     # lead II, half
paired[400, 2] = 1.0                     # lead III, full again

pair_traces = polylines(render_ecg_strip(paired, units="mV"))


def deflection(trace: np.ndarray, index: int) -> float:
    baseline = HEADER_HEIGHT + index * ROW_HEIGHT + TRACE_HALF
    return abs(baseline - float(trace[400, 1]))


first, second, third = (deflection(pair_traces[i], i) for i in range(3))

check("equal amplitudes give equal deflections",
      abs(first - third) < 0.2, f"{first:.2f} vs {third:.2f}")
check("half the amplitude gives half the deflection",
      abs(second - first / 2.0) < 0.6, f"{second:.2f} vs {first / 2.0:.2f}")

# Touching the row edge is fine; escaping it is not. A 1.0 mV peak lands exactly
# on a grid-ladder boundary, so the tallest deflection legitimately reaches the
# edge without being clipped — and rows are separated by BAND_HEIGHT + ROW_GAP,
# so a trace at its limit still cannot collide with its neighbour. What matters
# is that nothing is truncated, which is asserted directly instead.
check("the tallest deflection fits inside its row",
      first <= TRACE_HALF + 0.01, f"{first:.2f} of {TRACE_HALF:.2f}")
check("and fitting means no clipping note is emitted",
      "clipped" not in render_ecg_strip(paired, units="mV").lower())

# ============================================================
print("\n=== 6. millivolts are claimed only when the source declares them ===")
# ============================================================

check("mV is recognised", _calibration("mV") == (1.0, "mV"))
check("microvolts are converted rather than rejected",
      _calibration("uV")[0] == 1e-3, str(_calibration("uV")))
check("volts too", _calibration("V")[0] == 1e3, str(_calibration("V")))
check("an absent unit is not silently treated as mV",
      _calibration(None) == (None, "raw units"), str(_calibration(None)))
check("an unrecognised unit is labelled uncalibrated, not guessed",
      _calibration("counts")[0] is None
      and "uncalibrated" in _calibration("counts")[1],
      str(_calibration("counts")))

calibrated_text = all_text(STRIP)
check("a calibrated figure states the mV grid step",
      "mV" in calibrated_text and "0.2 s" in calibrated_text,
      re.search(r"one large square[^.]*\.", calibrated_text).group(0)
      if re.search(r"one large square[^.]*\.", calibrated_text) else "")

uncal_text = all_text(render_ecg_strip(SIGNAL, HEAT, units=None))
check("an uncalibrated figure says no millivolt calibration is claimed",
      "no millivolt" in uncal_text and "raw sample values" in uncal_text)
check("and the calibrated one does not carry that disclaimer",
      "no millivolt" not in calibrated_text)

# A 1000 uV signal and a 1 mV signal are the same recording; they must render
# to the same geometry.
micro = polylines(render_ecg_strip(paired * 1000.0, units="uV"))
check("1000 uV renders identically to 1 mV",
      abs(deflection(micro[0], 0) - first) < 0.2,
      f"{deflection(micro[0], 0):.2f} vs {first:.2f}")

# ============================================================
print("\n=== 7. clipping is announced, never silent ===")
# ============================================================

huge = np.zeros((ECG_INPUT_LENGTH, ECG_IN_CHANNELS), dtype=np.float32)
huge[:, 0] = 0.8
huge[300:320, 0] = 400.0                 # far past any sane scale
huge_svg = render_ecg_strip(huge, units="mV")

check("an out-of-range excursion is flagged in the figure",
      "clipped" in huge_svg.lower(), "")
check("the caption names the lead and says the flat top is the figure's limit",
      "Clipped to fit the row" in all_text(huge_svg)
      and "not the recording's" in all_text(huge_svg))
check("a normal recording carries no clipping note",
      "Clipped to fit" not in calibrated_text)

# ============================================================
print("\n=== 8. missing saliency is shown as missing ===")
# ============================================================

blank = render_ecg_strip(SIGNAL, None, units="mV")
blank_text = all_text(blank)

check("no heat images are emitted when there is no gradient",
      blank.count("<image") == 0, str(blank.count("<image")))
check("the header says saliency is unavailable",
      "Saliency unavailable" in blank_text)
check("the caption explains why the bands are blank rather than flat",
      "would look like a result" in blank_text)
check("the legend ramp is omitted, since there is nothing to key",
      "low" not in blank_text.lower() or "attribution" not in blank_text)

check("with a gradient, one heat band per lead is emitted",
      STRIP.count("<image") == ECG_IN_CHANNELS, str(STRIP.count("<image")))
check("the bands are real PNG payloads, not placeholders",
      STRIP.count("data:image/png;base64,") == ECG_IN_CHANNELS)
check("the band is stretched without preserving aspect, so it spans the row",
      STRIP.count('preserveAspectRatio="none"') == ECG_IN_CHANNELS)

band_heights = {float(value) for value in re.findall(
    r'<image[^>]*height="([\d.]+)"', STRIP)}
check("every band has the documented height",
      band_heights == {BAND_HEIGHT}, str(band_heights))

empty_attribution = all_text(render_lead_attribution([]))
check("the attribution figure says so too when there is no ranking",
      "no ranking to show" in empty_attribution, empty_attribution[:70])

# ============================================================
print("\n=== 9. per-lead normalisation of the band is disclosed ===")
# ============================================================
#
# Each band is normalised to its own lead's peak, so comparing colour between
# rows is invalid. If the figure does not say that, someone will do it.

check("the caption warns against comparing bands between leads",
      "not one lead against another" in calibrated_text, "")
check("and points to the ranking as the thing that does compare leads",
      "lead ranking" in calibrated_text)
check("the ranking figure states it uses raw gradient, not the display band",
      "not from the display band" in all_text(attribution_svg))
check("and separates 'where the model looked' from 'where the pathology is'",
      "not the same as where the pathology is" in all_text(attribution_svg))

# ============================================================
print("\n=== 10. the ranking figure is proportional and ordered ===")
# ============================================================

root = ET.fromstring(attribution_svg)
bars = [
    node for node in root.iter(f"{SVG_NS}rect")
    if node.get("fill", "").startswith("#") and float(node.get("width", 0)) > 0
]
# Track rects share one grey fill; the value bars each get a jet colour.
value_bars = [node for node in bars if node.get("fill") != "#f3f4f6"]
value_bars = [node for node in value_bars if node.get("height") is not None
              and float(node.get("width")) <= 420.0]

widths = [float(node.get("width")) for node in value_bars if node.get("y")]
check("there is a bar per lead",
      len(widths) == ECG_IN_CHANNELS, str(len(widths)))
check("bar lengths decrease with rank",
      all(widths[i] >= widths[i + 1] - 0.01 for i in range(len(widths) - 1)),
      str([round(w, 1) for w in widths[:4]]))
check("the strongest lead fills the track",
      abs(widths[0] - 420.0) < 1.0, f"{widths[0]:.1f}")

# Proportional across the board, not just at the ends: a bar is its score times
# the track. Anything else would misstate the ratio between two leads.
scores = [1.0 - index * 0.08 for index in range(ECG_IN_CHANNELS)]
worst = max(abs(width - score * 420.0)
            for width, score in zip(widths, scores))
check("every bar is exactly its score times the track width",
      worst < 0.6, f"largest discrepancy {worst:.2f} px")

ranking_text = all_text(attribution_svg)
check("the raw gradient magnitude is shown alongside the relative score",
      "0.9" in ranking_text and "0.83" in ranking_text)
check("the class the attribution explains is named",
      "Myocardial infarction" in ranking_text)

# A dict and a dataclass-like object must render the same.
class _Lead:
    def __init__(self, name, score, raw_score):
        self.name, self.score, self.raw_score = name, score, raw_score


object_form = render_lead_attribution(
    [_Lead(name, 1.0 - index * 0.08, 0.9 - index * 0.07)
     for index, name in enumerate(ECG_LEAD_NAMES)],
    saliency_class="MI",
)
check("dataclass and dict inputs render identically",
      object_form == attribution_svg)

# ============================================================
print("\n=== 11. the top leads are marked on the strip itself ===")
# ============================================================
#
# Badges sit on the badged lead's own row, so they appear in row order (I..V6),
# not in rank order. Checking the set alone would pass even if #1 landed on the
# wrong lead, so map each badge back to the row it was drawn on.

badge_rows = [
    (float(y), number)
    for y, number in re.findall(
        r'<text x="[\d.]+" y="([\d.]+)"[^>]*fill="#b91c1c">#(\d)</text>', STRIP)
]
badged = {
    number: ECG_LEAD_NAMES[int((y - HEADER_HEIGHT) // ROW_HEIGHT)]
    for y, number in badge_rows
}

check("exactly three badges are drawn",
      sorted(badged) == ["1", "2", "3"], str(sorted(badged)))
check("each badge is on the row of the lead it ranks",
      badged == {"1": "V1", "2": "II", "3": "aVF"}, str(badged))
check("the fourth-ranked lead is not badged, so the badges stay meaningful",
      "#4</text>" not in STRIP)
check("no badges appear when no ranking was supplied",
      "#1</text>" not in render_ecg_strip(SIGNAL, HEAT, units="mV"))

# ============================================================
print("\n=== 12. uploaded text cannot break or inject into the markup ===")
# ============================================================
#
# Record names come out of uploaded files, so they are untrusted input that ends
# up inside markup the browser will parse.

hostile = render_ecg_strip(
    SIGNAL, HEAT, units="mV",
    record_name='<script>alert(1)</script> & "quoted" <tspan/>',
)
try:
    hostile_root = ET.fromstring(hostile)
    hostile_ok = True
except ET.ParseError as error:
    hostile_root = None
    hostile_ok = False
    print(f"        {error}")

check("a hostile record name still yields parseable SVG", hostile_ok)
check("no script element is created",
      hostile_ok and not list(hostile_root.iter("script"))
      and not list(hostile_root.iter(f"{SVG_NS}script")))
check("the raw tag is escaped, not embedded",
      "<script>" not in hostile and "&lt;script&gt;" in hostile)
check("but the name is still readable as text",
      "alert(1)" in all_text(hostile))
check("the ampersand survives as an ampersand",
      "&" in all_text(hostile) and "&amp;amp;" not in hostile)

# ============================================================
print("\n=== 13. bad input is refused rather than half-drawn ===")
# ============================================================


def refuses(label: str, call, expect: str) -> None:
    try:
        call()
    except ValueError as error:
        check(label, expect.lower() in str(error).lower(), str(error)[:78])
        return
    except Exception as error:                       # pragma: no cover
        check(label, False, f"raised {type(error).__name__}: {error}")
        return
    check(label, False, "no error raised")


refuses("a signal with the wrong lead count is refused",
        lambda: render_ecg_strip(np.zeros((1000, 8), np.float32)), "12")
refuses("a transposed signal is refused rather than plotted sideways",
        lambda: render_ecg_strip(np.zeros((12, 1000), np.float32)), "12")
refuses("a one-sample signal is refused",
        lambda: render_ecg_strip(np.zeros((1, 12), np.float32)), "two samples")
refuses("saliency that does not line up with the signal is refused",
        lambda: render_ecg_strip(np.zeros((1000, 12), np.float32),
                                 np.zeros((12, 500), np.float32)),
        "line up")
refuses("the wrong number of lead names is refused",
        lambda: render_ecg_strip(np.zeros((1000, 12), np.float32),
                                 lead_names=("I", "II")),
        "lead names")

# ============================================================
print("\n=== 14. a shorter record is drawn honestly, not padded to look full ===")
# ============================================================

short = synthetic_ecg(n_samples=500)
short_svg = render_ecg_strip(short, units="mV", sampling_frequency=ECG_TARGET_FS)
short_text = all_text(short_svg)

check("the header reports the real duration",
      "5.00 s" in short_text, short_text[:60])
check("and the real sample count",
      "500 samples" in short_text)
check("the axis stops at 5 s rather than running to 10",
      re.findall(r">(\d+)s</text>", short_svg) == ["0", "1", "2", "3", "4", "5"],
      str(re.findall(r">(\d+)s</text>", short_svg)))
check("the trace still uses the full plot width, at a wider sample spacing",
      abs(polylines(short_svg)[0][-1, 0] - (GUTTER + PLOT_WIDTH))
      <= PLOT_WIDTH / 500 + 0.05,
      f"ends at {polylines(short_svg)[0][-1, 0]:.2f}, "
      f"edge at {GUTTER + PLOT_WIDTH:.2f}")

# ============================================================
print("\n=== 15. the grid step adapts without lying about itself ===")
# ============================================================

check("a 1 mV signal gets the clinical 0.5 mV large square",
      _nice_step(1.0, calibrated=True) == 0.5, str(_nice_step(1.0, True)))
check("an ordinary 1.5 mV R wave still gets the clinical square",
      _nice_step(1.5, calibrated=True) == 0.5, str(_nice_step(1.5, True)))
check("a tiny signal gets a finer step rather than a flat line",
      _nice_step(0.05, calibrated=True) <= 0.05, str(_nice_step(0.05, True)))
check("a huge signal gets a coarser one rather than clipping everything",
      _nice_step(9.0, calibrated=True) >= 3.0, str(_nice_step(9.0, True)))

# The uncalibrated ladder is a property, not a lookup table: whatever decade the
# signal lands in, the mantissa has to be one of 1 / 2 / 2.5 / 5. Asserting a
# specific return value here would really be asserting the row height, and would
# break the next time the geometry moves without anything being wrong.
uncalibrated_steps = [_nice_step(v, calibrated=False)
                      for v in (0.003, 0.4, 7.0, 300.0, 91000.0)]
mantissas = [round(s / 10.0 ** np.floor(np.log10(s)), 3)
             for s in uncalibrated_steps]
check("uncalibrated steps land on 1/2/2.5/5 decades",
      all(m in (1.0, 2.0, 2.5, 5.0) for m in mantissas), str(mantissas))
check("and each one actually covers the signal it was chosen for",
      all(s * 3.0 >= v - 1e-9
          for s, v in zip(uncalibrated_steps, (0.003, 0.4, 7.0, 300.0, 91000.0))),
      str(uncalibrated_steps))


def caption_step(svg: str) -> str:
    """The grid step the figure claims, straight out of its own caption."""
    found = re.search(r"one large square = 0\.2 s . ([\d.]+)", all_text(svg))
    return found.group(1) if found else ""


# The caption has to report the step actually used. A hardcoded "0.5 mV" would
# read as a calibration statement while being false for any rescaled export.
check("a normal recording is captioned at the standard 0.5 mV per square",
      caption_step(STRIP) == "0.5", caption_step(STRIP))

tiny = render_ecg_strip(SIGNAL * 0.01, units="mV")
check("a rescaled recording is captioned at its own finer step",
      caption_step(tiny) not in ("", "0.5")
      and float(caption_step(tiny)) < 0.5,
      f'"{caption_step(tiny)}" vs "{caption_step(STRIP)}"')
check("and the standard clinical gain is only claimed when it applies",
      "10 mm/mV," in all_text(STRIP) and "10 mm/mV," not in all_text(tiny),
      re.search(r"equivalent to [^.]*\.", all_text(tiny)).group(0)
      if re.search(r"equivalent to [^.]*\.", all_text(tiny)) else "")

# A flat trace must not divide by zero or vanish.
flat_svg = render_ecg_strip(np.zeros((1000, 12), np.float32), units="mV")
flat_traces = polylines(flat_svg)
check("an all-zero recording renders as flat lines, not an exception",
      len(flat_traces) == ECG_IN_CHANNELS)
check("and each sits on its own baseline",
      all(abs(flat_traces[i][:, 1].std()) < 1e-6
          for i in range(ECG_IN_CHANNELS)))

# ============================================================
print("\n=== 16. the API payload is a pair of usable data URLs ===")
# ============================================================


class _Analysis:
    saliency = HEAT
    saliency_available = True
    saliency_class = "MI"
    leads = [
        _Lead(name, 1.0 - index * 0.08, 0.9 - index * 0.07)
        for index, name in enumerate(("V1", "II", "aVF", "V3", "I", "III",
                                      "aVR", "aVL", "V2", "V4", "V5", "V6"))
    ]


payload = render_ecg_images(
    SIGNAL, _Analysis(), units="mV", record_name="HR00025",
)

check("both figures are returned",
      set(payload) == {"strip", "lead_attribution"}, str(sorted(payload)))
check("each is an SVG data URL",
      all(value.startswith("data:image/svg+xml;base64,")
          for value in payload.values()))

decoded = base64.b64decode(payload["strip"].split(",", 1)[1]).decode("utf-8")
check("the strip round-trips through base64 intact",
      decoded.startswith("<svg") and decoded.rstrip().endswith("</svg>"))
check("and parses after decoding", ET.fromstring(decoded) is not None)
check("the ranking from the analysis drives the badges on the strip",
      re.search(r'>V1</text>', decoded) is not None
      and '#1</text>' in decoded)

sizes = {name: len(value) // 1024 for name, value in payload.items()}
check("the payload is within the size the module documents",
      sizes["strip"] < 400, f"strip {sizes['strip']} KB, "
      f"ranking {sizes['lead_attribution']} KB")


class _NoGradient:
    saliency = None
    saliency_available = False
    saliency_class = None
    leads = []


absent = render_ecg_images(SIGNAL, _NoGradient(), units="mV")
absent_strip = base64.b64decode(
    absent["strip"].split(",", 1)[1]).decode("utf-8")
absent_rank = base64.b64decode(
    absent["lead_attribution"].split(",", 1)[1]).decode("utf-8")

check("with no gradient the strip still renders the waveform",
      len(polylines(absent_strip)) == ECG_IN_CHANNELS)
check("but emits no heat bands", "<image" not in absent_strip)
check("and the ranking figure reports the absence",
      "no ranking to show" in all_text(absent_rank))

# ============================================================
print("\n=== 17. the amplitude estimate survives both failure modes ===")
# ============================================================
#
# _robust_extent has to be robust in one direction without becoming wrong in the
# other, and the first version was only robust in one of them. Both directions
# are pinned here because the failure is silent either way: an underestimate
# amputates R waves into flat-topped morphology, an overestimate compresses all
# twelve traces towards their baselines. Neither looks like a bug on screen.

normal = synthetic_ecg()
lead_peaks = np.abs(normal).max(axis=0)

check("the estimate covers the true peak, so a normal ECG is not amputated",
      _robust_extent(normal) >= lead_peaks.max() - 1e-6,
      f"estimate {_robust_extent(normal):.3f} vs peak {lead_peaks.max():.3f}")
check("a normal recording therefore carries no clipping note",
      "clipped" not in render_ecg_strip(normal, units="mV").lower())

# The regression itself: pooling every sample of every lead puts the percentile
# down among the baseline, because that is where most of an ECG's samples are.
check("and it is not the pooled percentile that let the R waves through",
      float(np.percentile(np.abs(normal), 99.5)) < lead_peaks.max(),
      f"pooled 99.5th {np.percentile(np.abs(normal), 99.5):.3f} would sit "
      f"below the {lead_peaks.max():.3f} it has to cover")

spiked = normal.copy()
spiked[500, 3] = 40.0                    # one-sample artefact on aVR
check("a single-sample artefact does not set the scale for twelve leads",
      _robust_extent(spiked) < 2.0 * _robust_extent(normal),
      f"{_robust_extent(spiked):.3f} vs {_robust_extent(normal):.3f} clean")

blown = normal.copy()
blown[:, 3] *= 60.0                      # aVR saturated for its whole length

# The cap is three times the median of *this* signal's lead peaks, so it has to
# be computed from the blown signal, not the clean one — moving one lead to the
# top of the order shifts the median by half a position on its own.
blown_cap = 3.0 * float(np.median(np.abs(blown).max(axis=0)))
check("nor does one saturated lead, which the median cap holds back",
      _robust_extent(blown) <= blown_cap + 1e-6,
      f"{_robust_extent(blown):.3f}, cap {blown_cap:.3f}")
check("the cap sits far below the blown lead, which is the point of it",
      _robust_extent(blown) < 0.1 * float(np.abs(blown[:, 3]).max()),
      f"{_robust_extent(blown):.3f} vs the lead's "
      f"{np.abs(blown[:, 3]).max():.3f}")
check("and the eleven good leads keep a readable deflection",
      abs(HEADER_HEIGHT + TRACE_HALF
          - float(polylines(render_ecg_strip(blown, units="mV"))[0][:, 1].min()))
      > TRACE_HALF * 0.15,
      "lead I still deflects more than 15% of its row")
check("while the lead that blew the scale is announced as clipped",
      "clipped" in render_ecg_strip(blown, units="mV").lower())

check("a flatline does not divide by zero",
      _robust_extent(np.zeros((ECG_INPUT_LENGTH, ECG_IN_CHANNELS),
                              dtype=np.float32)) == 1.0)
check("an all-NaN signal falls back rather than raising",
      _robust_extent(np.full((ECG_INPUT_LENGTH, ECG_IN_CHANNELS), np.nan,
                             dtype=np.float32)) == 1.0)

half_nan = normal.copy()
half_nan[:, 5] = np.nan
check("one dead lead does not poison the estimate for the rest",
      abs(_robust_extent(half_nan) - _robust_extent(normal)) < 1e-6,
      f"{_robust_extent(half_nan):.3f} vs {_robust_extent(normal):.3f}")

# ============================================================
print("\n" + "=" * 62)

if FAILURES:
    print(f"{len(FAILURES)} FAILURE(S) out of {CHECKS} checks:")
    for item in FAILURES:
        print(f"  - {item}")
    sys.exit(1)

print(f"ALL {CHECKS} ECG RENDERING CHECKS PASSED")
print("Both figures were generated and parsed for real — this path needs")
print("neither torch nor scipy, so nothing here is stubbed.")
