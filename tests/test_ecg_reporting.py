#!/usr/bin/env python3
"""
ECG reporting logic.

Everything between "the model produced five numbers" and "a clinician reads a
screen": how probabilities are ordered, which are called positive, which
caveats travel with them, and how leads are ranked. This is the part that
decides what someone actually sees, so it is worth testing on its own rather
than only as a side effect of a full inference run.

    python3 tests/test_ecg_reporting.py

No torch needed — none of this touches a tensor. tests/torch_stub.py stands in
only so the module can be imported.
"""

from __future__ import annotations

import json
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
    ECG_CLASS_LABELS,
    ECG_CLASS_NAMES,
    ECG_IN_CHANNELS,
    ECG_INPUT_LENGTH,
    ECG_LEAD_NAMES,
    ECG_THRESHOLD,
    ECG_WEAK_CLASSES,
)
from cardiovision.inference.ecg import EcgClassifier, EcgModelUnavailable  # noqa: E402

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


CLASSIFIER = EcgClassifier()


def predict(**probabilities: float):
    """Build predictions from a {class: probability} mapping."""
    vector = np.array(
        [probabilities.get(name, 0.0) for name in ECG_CLASS_NAMES],
        dtype=np.float32,
    )
    return CLASSIFIER._build_predictions(vector)


def notes_for(**probabilities: float) -> list[str]:
    return CLASSIFIER._build_notes(predict(**probabilities))


# ============================================================
print("\n=== 1. probabilities map to the right classes ===")
# ============================================================

# The single most consequential thing in this module: index 1 must be MI. If the
# vector were read in a different order every screen would be wrong and nothing
# would look broken.
predictions = predict(NORM=0.10, MI=0.91, STTC=0.20, CD=0.30, HYP=0.05)
by_name = {p.name: p for p in predictions}

check("all five classes are reported, not just the positives",
      len(predictions) == 5, str(len(predictions)))
check("each probability lands on its own class",
      (abs(by_name["MI"].probability - 0.91) < 1e-6
       and abs(by_name["NORM"].probability - 0.10) < 1e-6
       and abs(by_name["HYP"].probability - 0.05) < 1e-6),
      f"MI={by_name['MI'].probability}, NORM={by_name['NORM'].probability}")
check("labels are human-readable, not codes",
      by_name["MI"].label == ECG_CLASS_LABELS["MI"], by_name["MI"].label)
check("every class carries a plain-language description",
      all(len(p.description) > 20 for p in predictions))

# ============================================================
print("\n=== 2. ordering puts the finding first ===")
# ============================================================

check("highest probability comes first",
      predictions[0].name == "MI", predictions[0].name)
check("the whole list is sorted descending",
      all(predictions[i].probability >= predictions[i + 1].probability
          for i in range(len(predictions) - 1)),
      str([round(p.probability, 2) for p in predictions]))

# Class-index order would bury a strong MI beneath a weak NORM, since NORM is
# index 0.
index_order = [p.name for p in predict(NORM=0.05, MI=0.95)]
check("a strong MI is not buried under a weak NORM",
      index_order[0] == "MI", str(index_order))

# ============================================================
print("\n=== 3. the threshold is applied and disclosed ===")
# ============================================================

edge = {p.name: p for p in predict(NORM=0.5, MI=0.4999, STTC=0.5001)}

check("exactly at the threshold counts as positive",
      edge["NORM"].positive, f"{edge['NORM'].probability} -> {edge['NORM'].positive}")
check("just under does not",
      not edge["MI"].positive)
check("just over does",
      edge["STTC"].positive)
check("every prediction reports the threshold it was judged at",
      all(p.threshold == ECG_THRESHOLD for p in predictions))
check("the threshold is the documented 0.5",
      ECG_THRESHOLD == 0.5, str(ECG_THRESHOLD))

# ============================================================
print("\n=== 4. each probability arrives with its operating point ===")
# ============================================================
#
# A bare "MI 91%" invites the reading that the model is 91% likely to be right.
# The precision it was measured at is the correction for that, so it has to be
# attached to the same object rather than living in a separate model card.

CLASSIFIER._metrics = {}          # force the config fallback
fallback = {p.name: p for p in predict(MI=0.9)}

check("AUROC is attached per class",
      fallback["MI"].auroc is not None, str(fallback["MI"].auroc))
check("so is precision, which is the number that bounds a positive call",
      fallback["MI"].precision is not None, str(fallback["MI"].precision))
check("and recall",
      fallback["MI"].recall is not None, str(fallback["MI"].recall))
check("and test prevalence, without which precision cannot be interpreted",
      fallback["MI"].prevalence is not None, str(fallback["MI"].prevalence))

payload = fallback["MI"].to_dict()
check("the serialised form scopes the metrics as dataset-level",
      "dataset-level" in payload["operating_point"]["scope"],
      payload["operating_point"]["scope"])
check("the operating point is nested, not mixed in with the probability",
      "probability" in payload and "auroc" in payload["operating_point"])
check("it serialises to JSON",
      json.loads(json.dumps(payload))["name"] == "MI")

# ============================================================
print("\n=== 5. the HYP caveat rides along with a positive HYP ===")
# ============================================================

check("HYP carries a caveat at every probability",
      predict(HYP=0.9)[0].caveat is not None)
# The caveat is a property of the class, not of this recording, so it is
# attached regardless of probability — and only to HYP.
check("HYP is the only class carrying one",
      {p.name for p in predict(MI=0.9, STTC=0.9, CD=0.9, NORM=0.9, HYP=0.9)
       if p.caveat} == {"HYP"},
      str({p.name: bool(p.caveat)
           for p in predict(MI=0.9, STTC=0.9, CD=0.9, NORM=0.9, HYP=0.9)}))
check("and it matches ECG_WEAK_CLASSES",
      set(ECG_WEAK_CLASSES) == {"HYP"}, str(set(ECG_WEAK_CLASSES)))

hyp_notes = notes_for(HYP=0.9)
check("a positive HYP produces a note",
      any("Hypertrophy" in note for note in hyp_notes), str(len(hyp_notes)))
check("the note quantifies the weakness rather than hedging vaguely",
      any("0.361" in note or "two in three" in note for note in hyp_notes),
      next((n for n in hyp_notes if "Hypertrophy" in n), "")[:80])

check("a NEGATIVE HYP does not raise the caveat as a note",
      not any("Hypertrophy is the weakest" in note
              for note in notes_for(HYP=0.2, MI=0.8)),
      str(notes_for(HYP=0.2, MI=0.8)))
check("but the caveat still travels on the class object for the UI",
      predict(HYP=0.2)[-1].caveat is not None or
      any(p.name == "HYP" and p.caveat for p in predict(HYP=0.2)))

# ============================================================
print("\n=== 6. an all-negative result is not reported as normal ===")
# ============================================================
#
# The failure mode worth guarding: five sub-threshold probabilities are not a
# normal ECG. NORM is a class the model can also miss.

quiet = notes_for(NORM=0.3, MI=0.2, STTC=0.1, CD=0.1, HYP=0.1)
check("nothing above threshold produces a note",
      any("No class reached" in note for note in quiet), str(quiet))
check("the note refuses both readings explicitly",
      any("not a normal ECG" in note and "not an abnormal one" in note
          for note in quiet))
check("a positive NORM produces no such note",
      not any("No class reached" in note for note in notes_for(NORM=0.9)))

# ============================================================
print("\n=== 7. contradictory multi-label output is explained ===")
# ============================================================

conflict = notes_for(NORM=0.8, MI=0.7)
check("NORM plus an abnormality is flagged",
      any("disagreeing with itself" in note for note in conflict), str(conflict))
check("the note explains why that is possible at all",
      any("independent sigmoids" in note for note in conflict))
check("NORM alone is not flagged",
      not any("disagreeing" in note for note in notes_for(NORM=0.9)))
check("two abnormalities together are not flagged — that is a real combination",
      not any("disagreeing" in note for note in notes_for(MI=0.8, STTC=0.7)),
      str(notes_for(MI=0.8, STTC=0.7)))

# ============================================================
print("\n=== 8. near-misses are surfaced, not silently dropped ===")
# ============================================================

near = notes_for(NORM=0.9, MI=0.47)
check("a class at 0.47 is mentioned",
      any("Just below" in note for note in near), str(near))
check("the note names the class and its probability",
      any("Myocardial infarction 0.47" in note for note in near))
check("and says the call turned on the threshold",
      any("where the threshold sits" in note for note in near))
check("a class at 0.10 is not mentioned — that is not a near miss",
      not any("Just below" in note for note in notes_for(NORM=0.9, MI=0.10)),
      str(notes_for(NORM=0.9, MI=0.10)))

# ============================================================
print("\n=== 9. lead attribution comes from the raw gradient ===")
# ============================================================
# The methodological trap this guards: per-lead normalisation makes every lead
# peak at 1.0, so a mean over the normalised trace measures how spread out a
# lead's attribution is, not how much of it there is. To show that, every lead
# needs some content — the contrast is uniform-vs-spiky, not loud-vs-silent.

gradient = np.full((ECG_IN_CHANNELS, ECG_INPUT_LENGTH), 0.01, dtype=np.float32)
gradient[0] = 0.05                        # lead I, uniform, mildly above floor
gradient[6] = 0.90                        # lead V1, uniform, clearly dominant
gradient[7, 500] = 5.00                   # lead V2, one huge spike on the floor

leads = CLASSIFIER._rank_leads(gradient)
ranked = [lead.name for lead in leads]
scores = {lead.name: lead.score for lead in leads}

check("every lead is ranked", len(leads) == 12, str(len(leads)))
check("the dominant lead comes first",
      ranked[0] == "V1", str(ranked[:4]))
check("a uniformly larger lead outranks a uniformly smaller one",
      ranked.index("I") < ranked.index("V6"), str(ranked))
check("the spike lead ranks above the flat baseline leads",
      ranked.index("V2") < ranked.index("V6"),
      f"V2 at {ranked.index('V2')}, V6 at {ranked.index('V6')}")
check("scores are normalised to the strongest lead",
      abs(max(scores.values()) - 1.0) < 1e-6, f"max = {max(scores.values())}")
check("the raw magnitude is reported too, so the scale is recoverable",
      abs(leads[0].raw_score - 0.90) < 1e-5, str(leads[0].raw_score))

# The same gradient, ranked the way models/ecg/lead_importance.csv was computed.
per_lead_normalised = gradient / (gradient.max(axis=1, keepdims=True) + 1e-8)
normalised_means = per_lead_normalised.mean(axis=1)
normalised_ranking = [
    str(name) for name in np.array(ECG_LEAD_NAMES)[np.argsort(-normalised_means)]
]

check("under normalisation the spike lead sinks to LAST, "
      "the opposite of its raw rank",
      normalised_ranking[-1] == "V2" and ranked.index("V2") < 11,
      f"normalised last = {normalised_ranking[-1]}, "
      f"raw rank = {ranked.index('V2')} of 12")
check("and normalisation flattens the other eleven into a meaningless tie",
      float(np.ptp(np.delete(normalised_means, 7))) < 1e-6,
      f"spread across the other 11 = "
      f"{float(np.ptp(np.delete(normalised_means, 7))):.2e}")
check("whereas the raw scores separate them",
      float(np.ptp([lead.raw_score for lead in leads])) > 0.5,
      f"spread = {float(np.ptp([lead.raw_score for lead in leads])):.3f}")

# Silent leads, checked on their own so they cannot interfere with the above.
sparse = np.zeros((ECG_IN_CHANNELS, ECG_INPUT_LENGTH), dtype=np.float32)
sparse[3] = 0.4
sparse_scores = {lead.name: lead.score for lead in CLASSIFIER._rank_leads(sparse)}
check("a lead with no gradient scores exactly zero",
      sparse_scores["V6"] == 0.0, str(sparse_scores["V6"]))
check("and the one lead with gradient scores one",
      abs(sparse_scores["aVR"] - 1.0) < 1e-6, str(sparse_scores["aVR"]))

check("no gradient means no attribution, rather than a flat fake ranking",
      CLASSIFIER._rank_leads(None) == [])
check("an all-zero gradient does not divide by zero",
      all(lead.score == 0.0 for lead in CLASSIFIER._rank_leads(
          np.zeros((ECG_IN_CHANNELS, ECG_INPUT_LENGTH), dtype=np.float32))))
check("attribution serialises to JSON",
      json.loads(json.dumps([lead.to_dict() for lead in leads]))[0]["name"]
      in ECG_LEAD_NAMES)

# ============================================================
print("\n=== 10. the model refuses to be used before it is loaded ===")
# ============================================================

fresh = EcgClassifier()
check("is_loaded is False before load()", not fresh.is_loaded)

try:
    fresh.analyze(np.zeros((ECG_INPUT_LENGTH, ECG_IN_CHANNELS), np.float32))
    unloaded_refused = False
    message = ""
except EcgModelUnavailable as error:
    unloaded_refused = True
    message = str(error)
check("analyze() on an unloaded model raises rather than returning zeros",
      unloaded_refused, message[:60])

# ============================================================
print("\n" + "=" * 62)

if STUBBED_TORCH:
    print("NOTE: torch is not installed here. Everything above is real —")
    print("      none of it touches a tensor — but no forward pass ran, so")
    print("      the probabilities fed in were synthetic.")
    print("=" * 62)

if FAILURES:
    print(f"{len(FAILURES)} FAILURE(S) out of {CHECKS} checks:")
    for item in FAILURES:
        print(f"  - {item}")
    sys.exit(1)

print(f"ALL {CHECKS} ECG REPORTING CHECKS PASSED")
