#!/usr/bin/env python3
"""
Case lifecycle verification.

Exercises the store the way the frontend actually drives it: the exact payload
shapes App.jsx builds, in the order it builds them — mint a case on analysis,
attach the echo result and images, append to the transcript, list, reopen,
delete. Then a set of static assertions on App.jsx for the wiring that a
round-trip test cannot see.

Everything runs against a real SQLite file in a temporary directory, so it
touches no patient data and needs no server:

    python3 tests/test_case_lifecycle.py

No torch and no model weights required.
"""

from __future__ import annotations

import base64
import re
import shutil
import sqlite3
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
FRONTEND = REPO / "frontend" / "src"

WORK = Path(tempfile.mkdtemp(prefix="cv-lifecycle-"))

# Run from a checkout without installing first. `pip install -e .` makes this
# line redundant rather than wrong.
sys.path.insert(0, str(REPO / "src"))

# config.select_device() imports torch lazily and falls back to "cpu" when it is
# absent, so nothing here needs a stub: the store, the context builder and the
# schema never touch a tensor.
from cardiovision.config import ECG_LEAD_NAMES, ECG_NORMALIZATION  # noqa: E402
from cardiovision.services.database import (  # noqa: E402
    CaseStore,
    derive_age,
    media_type_for,
)

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


def png_data_url() -> str:
    """A real 1x1 PNG, so the stored bytes can be checked for magic numbers."""
    raw = base64.b64decode(
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8"
        "z8DwHwAFAAH/q842iQAAAABJRU5ErkJggg=="
    )
    return "data:image/png;base64," + base64.b64encode(raw).decode()


def svg_data_url() -> str:
    """
    A real SVG data URL, encoded the way rendering.primitives does it.

    Base64 rather than percent-encoded, because that is what the renderer
    emits and what the store's decoder looks for — a percent-encoded URL would
    be silently skipped rather than stored.
    """
    markup = '<svg xmlns="http://www.w3.org/2000/svg" width="4" height="4"></svg>'
    return "data:image/svg+xml;base64," + base64.b64encode(
        markup.encode("utf-8")
    ).decode("ascii")


store = CaseStore(
    db_path=WORK / "cardiovision.db",
    files_dir=WORK / "cases",
)

# ============================================================
print("\n=== 1. connect_error reports WHY storage is down ===")
# ============================================================

check(
    "not ready before connect",
    not store.is_ready,
)
check(
    "connect_error explains an unopened database",
    isinstance(store.connect_error, str) and len(store.connect_error) > 10,
    store.connect_error,
)

broken = CaseStore(
    # A path whose parent is a file, not a directory: mkdir must fail.
    db_path=WORK / "not-a-dir" / "x.db",
    files_dir=WORK / "not-a-dir" / "cases",
)
(WORK / "not-a-dir").write_text("i am a file", encoding="utf-8")

raised = False
try:
    broken.connect()
except Exception:
    raised = True

check("connect() re-raises a real failure", raised)
check(
    "connect_error names the exception type",
    broken.connect_error is not None and ":" in broken.connect_error,
    broken.connect_error,
)

store.connect()
check("ready after connect", store.is_ready)
check("connect_error is None once ready", store.connect_error is None)

# ============================================================
print("\n=== 2. mint on analysis (frontend step 1) ===")
# ============================================================

# App.jsx persists BEFORE analysing, so the backend has a case to file the
# source image under. That first save carries patient + clinical only.
patient = {
    "name": "Rahman, Ayesha",
    "mrn": "MRN-48213",
    "dateOfBirth": "1967-04-18",
    "sex": "Female",
    "studyDate": "2026-08-25",
    "referringClinician": "Dr T. Iqbal, Cardiology",
    "notes": "Exertional dyspnoea, 3 months. Prior MI 2019.",
}
clinical = {
    "age": "59",
    "sex": "Female",
    "symptoms": "Dyspnoea on exertion",
    "bloodPressure": "148/92",
    "heartRate": "88",
    "diabetes": True,
    "hypertension": True,
    "smoking": False,
}

first = store.save({"patient": patient, "clinical": clinical})
case_id = first["case_id"]

check("a case_id was minted", bool(case_id), case_id)
check(
    "minted id matches CV-YYYYMMDD-XXXXXX",
    len(case_id) == 18 and case_id.startswith("CV-"),
    case_id,
)
check("echo_analyzed is 0 with no echo", first["echo_analyzed"] is False)
check("name round-trips", first["patient"]["name"] == patient["name"])
check("mrn round-trips", first["patient"]["mrn"] == "MRN-48213")
check("referring clinician round-trips",
      first["patient"]["referringClinician"] == patient["referringClinician"])
check("notes round-trip", first["patient"]["notes"] == patient["notes"])
check(
    "age is derived from DOB, not from the typed clinical age",
    first["patient"]["age"] == derive_age("1967-04-18"),
    f"derived={first['patient']['age']} typed={clinical['age']}",
)
check("clinical booleans survive JSON", first["clinical"]["diabetes"] is True)
check("clinical false survives as false", first["clinical"]["smoking"] is False)

# ============================================================
print("\n=== 3. attach echo + images (frontend step 2) ===")
# ============================================================

# This is exactly what persist() sends: the full result minus `images`,
# with `analyzed: true` forced on, and the data URLs in a sibling field.
echo_result = {
    "analyzed": True,
    "model": {
        "architecture": "UNet++",
        "encoder": "EfficientNet-B3",
        "num_classes": 4,
        "metrics": {"test_dice": 0.9044, "test_patients": 75},
    },
    "structures": [
        {"class_index": 1, "name": "LV cavity", "present": True,
         "area_percent": 8.1, "mean_confidence": 0.97},
        {"class_index": 2, "name": "Myocardium", "present": True,
         "area_percent": 11.4, "mean_confidence": 0.91},
        {"class_index": 3, "name": "Left atrium", "present": False,
         "area_percent": 0.0},
    ],
    "input": {"format": "png", "filename": "a4c_frame.png",
              "has_spatial_calibration": False},
    "orientation": {"display_oriented_format": True, "reoriented": True,
                    "rotation_applied": 90, "flip_applied": False},
    "quantification": {"presence_threshold_pixels": 50},
    "mask": {"width": 2, "height": 2, "num_classes": 4,
             "class_colors": {"1": [255, 0, 0]}, "data": [0, 1, 1, 0]},
    "explainability": {"available": True, "method": "Input-gradient saliency"},
    "notes": [],
    "inference_ms": 412,
    "device": "mps",
}

images = {
    key: png_data_url()
    for key in ("original", "mask", "overlay", "saliency_overlay", "combined")
}

second = store.save({
    "case_id": case_id,
    "patient": patient,
    "clinical": clinical,
    "echo": echo_result,
    "images": images,
})

check("same case_id, not a second row", second["case_id"] == case_id)
check("store still holds exactly 1 case", store.count() == 1)
check("created_at preserved across update",
      second["created_at"] == first["created_at"])
check("updated_at advanced",
      second["updated_at"] > first["updated_at"],
      f"{first['updated_at']} -> {second['updated_at']}")
check("echo_analyzed now true", second["echo_analyzed"] is True)
check("structures_found counts only present=True",
      second["structures_found"] == 2, str(second["structures_found"]))
check("echo_filename captured", second["echo"]["input"]["filename"] == "a4c_frame.png")
check("mask survived the round trip",
      second["echo"]["mask"]["data"] == [0, 1, 1, 0])
check("inference timing survived", second["echo"]["inference_ms"] == 412)

check(
    "images come back as authenticated endpoints, not base64",
    all(
        url == f"/api/cases/{case_id}/images/{key}"
        for key, url in second["images"].items()
    ),
    ", ".join(sorted(second["images"])),
)

for key in images:
    payload = store.read_image(case_id, key)
    check(
        f"{key}.png written with a real PNG header",
        payload is not None and payload[:8] == b"\x89PNG\r\n\x1a\n",
    )

check(
    "saliency.png absent because it was never supplied",
    store.read_image(case_id, "saliency") is None,
)

# ============================================================
print("\n=== 4. metadata-only re-save must not wipe the echo ===")
# ============================================================

# The Save button sends no `echo` when nothing was analysed this session.
edited = dict(patient, notes="Added after review: consider stress echo.")
third = store.save({"case_id": case_id, "patient": edited, "clinical": clinical})

check("echo survives a metadata-only save", third["echo"] is not None)
check("echo_analyzed stays true", third["echo_analyzed"] is True)
check("images survive", len(third["images"]) == 5, str(len(third["images"])))
check("edited note took effect", third["patient"]["notes"].startswith("Added after"))
check("mask still intact", third["echo"]["mask"]["data"] == [0, 1, 1, 0])

# The denormalised columns feed the sidebar tag directly. If they are not
# protected alongside echo_json, renaming a patient makes an analysed case
# advertise "Echo · 0/3" while its segmentation sits intact one click away.
check("structures_found survives a metadata-only save",
      third["structures_found"] == 2, str(third["structures_found"]))

summary = store.list(search=case_id)[0]
check("the sidebar still shows 2 structures",
      summary["structures_found"] == 2, str(summary["structures_found"]))
check("the sidebar still shows the echo flag", summary["echo_analyzed"] is True)
check("the sidebar still shows the filename",
      summary["echo_filename"] == "a4c_frame.png", repr(summary["echo_filename"]))

# ============================================================
print("\n=== 5. conversation is replaced, never duplicated ===")
# ============================================================

conversation = [
    {"role": "user", "text": "What do the segmented structures show?"},
    {"role": "assistant", "text": "Two structures were segmented...",
     "model": "MedGemma-4B", "device": "mps"},
]

store.save({"case_id": case_id, "patient": edited, "clinical": clinical,
            "conversation": conversation})
after_one = store.get(case_id)

check("two messages stored", len(after_one["conversation"]) == 2)

# Saving again with the same list must not append a second copy.
store.save({"case_id": case_id, "patient": edited, "clinical": clinical,
            "conversation": conversation})
after_two = store.get(case_id)

check("re-saving does not duplicate the transcript",
      len(after_two["conversation"]) == 2,
      str(len(after_two["conversation"])))

grown = conversation + [{"role": "user", "text": "And the left atrium?"}]
store.save({"case_id": case_id, "patient": edited, "clinical": clinical,
            "conversation": grown})
after_three = store.get(case_id)

check("appending a message grows the transcript",
      len(after_three["conversation"]) == 3)
check("message order preserved",
      [m["role"] for m in after_three["conversation"]]
      == ["user", "assistant", "user"])
check("model/device kept on the assistant turn",
      after_three["conversation"][1]["model"] == "MedGemma-4B")
check("restored turns cannot claim a context preview",
      all(m["contextUsed"] is False for m in after_three["conversation"]))

# ============================================================
print("\n=== 6. the sidebar list (explicit columns, no echo blob) ===")
# ============================================================

# A second and third case so ordering and search have something to do.
store.save({"patient": {"name": "Chowdhury, Imran", "mrn": "MRN-90011"},
            "clinical": {}})
store.save({"patient": {"name": "", "mrn": ""}, "clinical": {"age": "44"}})

listing = store.list()

check("all three cases listed", len(listing) == 3, str(len(listing)))
check("newest first",
      listing[0]["updated_at"] >= listing[1]["updated_at"] >= listing[2]["updated_at"])
check("every row has a display_name",
      all(row.get("display_name") for row in listing),
      " | ".join(row["display_name"] for row in listing))
check("a nameless case is labelled, not blank",
      any(row["display_name"] == "Unnamed patient" for row in listing))
check(
    "the 65k-element mask is NOT in the list payload",
    all("echo" not in row and "mask" not in row for row in listing),
    "summary keys: " + ", ".join(sorted(listing[0])),
)
check("echo flag exposed for the sidebar tag",
      any(row["echo_analyzed"] for row in listing))
check("structures_found exposed for the sidebar tag",
      any(row["structures_found"] == 2 for row in listing))

by_name = store.list(search="ayesha")
check("search by name is case-insensitive",
      len(by_name) == 1 and by_name[0]["case_id"] == case_id,
      str(len(by_name)))

by_mrn = store.list(search="MRN-900")
check("search by partial MRN",
      len(by_mrn) == 1 and by_mrn[0]["patient_name"] == "Chowdhury, Imran")

by_id = store.list(search=case_id[-6:])
check("search by case-id fragment", len(by_id) == 1)

by_note = store.list(search="stress echo")
check("search reaches the notes field", len(by_note) == 1)

check("no match returns empty, not everything",
      store.list(search="zzzzzz-nobody") == [])
check("limit is honoured", len(store.list(limit=2)) == 2)

# ============================================================
print("\n=== 7. reopening a case gives the frontend what it needs ===")
# ============================================================

reopened = store.get(case_id)

required = ("case_id", "patient", "clinical", "echo", "images",
            "source_file", "conversation", "echo_analyzed",
            "structures_found", "created_at", "updated_at")
missing = [key for key in required if key not in reopened]
check("every field App.jsx seeds state from is present", not missing,
      f"missing: {missing}" if missing else "all present")

check("rotation is recoverable for the orientation control",
      reopened["echo"]["orientation"]["rotation_applied"] == 90)
check("flip is recoverable",
      reopened["echo"]["orientation"]["flip_applied"] is False)
check("image keys drive fetchCaseImages()",
      sorted(reopened["images"]) == sorted(images))
check("unknown case returns None, not a blank record",
      store.get("CV-19700101-ZZZZZZ") is None)

# ============================================================
print("\n=== 8. deleting a case takes its files and messages ===")
# ============================================================

case_dir = store.case_dir(case_id)
check("case directory exists before delete", case_dir.is_dir())

check("delete reports success", store.delete(case_id) is True)
check("row is gone", store.get(case_id) is None)
check("directory removed", not case_dir.exists())
check("count dropped to 2", store.count() == 2)
check("deleting twice is False, not an error", store.delete(case_id) is False)

orphans = store._require().execute(
    "SELECT COUNT(*) AS n FROM case_messages WHERE case_id = ?", (case_id,)
).fetchone()["n"]
check("messages cascaded away", orphans == 0, f"{orphans} orphan rows")

# ============================================================
print("\n=== 9. case_context withholds the name, keeps the medicine ===")
# ============================================================

from cardiovision.services import case_context  # noqa: E402

text = case_context.build_case_context({
    "case_id": "CV-20260825-ABC123",
    "patient": patient,
    "clinical": clinical,
    "echo": echo_result,
    "modalities_provided": {"echo": True, "ccta": False, "ecg": False},
})

check("context was built", isinstance(text, str) and len(text) > 100)
check("the patient's NAME is absent from the prompt",
      "Ayesha" not in text and "Rahman" not in text)
check("the MRN is absent from the prompt", "48213" not in text)
check("the derived age IS present",
      f"Age: {derive_age('1967-04-18')} years" in text)
check("the clinician's note IS present", "Exertional dyspnoea" in text)
check("study date IS present", "2026-08-25" in text)
check("the typed age is suppressed in favour of the DOB",
      text.count("Age:") == 1, f"{text.count('Age:')} age lines")
check("unavailable modalities are still named",
      "CCTA" in text or "Coronary" in text)
check("smoking is UNKNOWN, not denied",
      "Not recorded either way" in text and "Smoking" in text)

# ============================================================
print("\n=== 10. App.jsx wiring (static, but specific) ===")
# ============================================================

app = (FRONTEND / "App.jsx").read_text()
case_list = (FRONTEND / "components" / "CaseList.jsx").read_text()
api = (FRONTEND / "api.js").read_text()

check(
    "the New case button calls startNewCase, not scrollToSection",
    'onClick={startNewCase}' in app
    and 'onClick={() => scrollToSection("case")}>\n            New case' not in app,
)
check(
    "startNewCase confirms before discarding unsaved work",
    'confirmDiscard("Start a new case")' in app,
)
check(
    "opening another case confirms too",
    'confirmDiscard("Open another case")' in app,
)
check(
    "the reset is a remount, so no field can be missed",
    "key={caseKey}" in app and "setCaseKey((previous) => previous + 1)" in app,
)
check(
    "the hardcoded CV-<year>-001 case id is gone",
    "CV-${new Date().getFullYear()}-001" not in app and "patientId" not in app,
)
check(
    "analysis passes the case id so the upload is archived",
    "caseId: targetCase || undefined" in app,
)
check(
    "a case row is minted before analysis when none exists",
    "const ensureCase = useCallback" in app
    and "if (caseId || !storageReady) return caseId;" in app
    and "const targetCase = await ensureCase();" in app,
)
check(
    "the analysis result is persisted, not just shown",
    "echo: result," in app,
)
check(
    "blob URLs are revoked on unmount",
    "releaseImages(blobUrls.current)" in app,
)
check(
    "the orientation re-run is hidden when there is no File to re-run",
    "files.echo ? reanalyzeWithOrientation : undefined" in app,
)
check(
    "PatientForm is mounted in section 01",
    "<PatientForm" in app and 'id="case"' in app,
)
check(
    "every patient field is wired",
    all(
        field in app or field in (FRONTEND / "components" / "PatientForm.jsx").read_text()
        for field in ("name", "mrn", "dateOfBirth", "sex", "studyDate",
                      "referringClinician", "notes")
    ),
)
check(
    "the login gate wraps the app, not just a banner",
    "if (!session) {" in app and "<Login" in app,
)
check(
    "a 401 anywhere returns to the login screen",
    "setUnauthorizedHandler((message)" in app and "setSession(null)" in app,
)
check(
    "search is debounced rather than firing per keystroke",
    "setTimeout(() => onSearch(term), 250)" in case_list,
)
check(
    "the token is never put in a URL",
    "token=" not in api and "Bearer ${token}" in api,
)
check(
    # api.js mentions localStorage in a comment explaining why it is not
    # used, so the check has to be about the call, not the word.
    "the token lives in sessionStorage, not localStorage",
    "window.sessionStorage" in api and "window.localStorage" not in api,
)

# ============================================================
print("\n=== 11. ECG attaches independently of echo ===")
# ============================================================

# A second case, so the ECG path is exercised without the echo case's history
# confusing which column protected what.
ecg_case = store.save({"patient": {"name": "Barua, Nipa", "mrn": "MRN-77120"},
                       "clinical": {}})["case_id"]

check("a fresh case reports no ECG", store.get(ecg_case)["ecg_analyzed"] is False)
check("and no positives", store.get(ecg_case)["ecg_positive_count"] == 0)
check("and ecg is None, not an empty dict",
      store.get(ecg_case)["ecg"] is None)

# Shaped exactly like the /api/analyze/ecg response: predictions for all five
# classes, whether or not they were called.
ecg_result = {
    "predictions": [
        {"name": "NORM", "label": "Normal ECG", "probability": 0.07,
         "positive": False},
        {"name": "MI", "label": "Myocardial infarction", "probability": 0.81,
         "positive": True},
        {"name": "STTC", "label": "ST/T change", "probability": 0.64,
         "positive": True},
        {"name": "CD", "label": "Conduction disturbance", "probability": 0.11,
         "positive": False},
        {"name": "HYP", "label": "Hypertrophy", "probability": 0.55,
         "positive": True},
    ],
    "positive_classes": ["MI", "STTC", "HYP"],
    "threshold": 0.5,
    "saliency_class": "MI",
    "saliency_available": True,
    "lead_attribution": [{"name": "V2", "importance": 1.0}],
    "inference_ms": 38.4,
    "compute_device": "mps",
    "notes": [],
    "input": {"format": "wfdb", "filename": "HR00025.hea"},
    # Sent nested, which is what the endpoint returns verbatim. The store must
    # lift these out rather than committing 200 KB of base64 to a JSON column.
    "figures": {
        "strip": svg_data_url(),
        "lead_attribution": svg_data_url(),
    },
}

with_ecg = store.save({"case_id": ecg_case, "ecg": ecg_result})

check("ecg_analyzed flipped true", with_ecg["ecg_analyzed"] is True)
check("positives counted", with_ecg["ecg_positive_count"] == 3,
      str(with_ecg["ecg_positive_count"]))
check("all five predictions round-trip",
      len(with_ecg["ecg"]["predictions"]) == 5)
check("a negative class is kept, not dropped",
      any(p["name"] == "CD" and p["positive"] is False
          for p in with_ecg["ecg"]["predictions"]))
check("the weak class survives as a positive call",
      "HYP" in with_ecg["ecg"]["positive_classes"])
check("saliency class round-trips", with_ecg["ecg"]["saliency_class"] == "MI")

check(
    "the SVG figures were lifted OUT of the stored JSON",
    "figures" not in with_ecg["ecg"],
    "ecg keys: " + ", ".join(sorted(with_ecg["ecg"])),
)
check(
    "and came back as authenticated endpoints",
    with_ecg["ecg_figures"] == {
        "strip": f"/api/cases/{ecg_case}/images/ecg_strip",
        "lead_attribution": f"/api/cases/{ecg_case}/images/ecg_lead_attribution",
    },
    ", ".join(sorted(with_ecg["ecg_figures"])),
)
check(
    "the ECG figures are NOT mixed into the echo image set",
    with_ecg["images"] == {},
    str(with_ecg["images"]),
)

for name in ("ecg_strip", "ecg_lead_attribution"):
    blob = store.read_image(ecg_case, name)
    check(f"{name}.svg written as real SVG markup",
          blob is not None and blob.lstrip().startswith(b"<svg"))
    check(f"{name} is served as SVG, not PNG",
          media_type_for(name) == "image/svg+xml", media_type_for(name))

check("echo renders still declare image/png",
      media_type_for("overlay") == "image/png")
check("an unknown figure name is refused, not guessed",
      store.read_image(ecg_case, "../../etc/passwd") is None
      and media_type_for("nonsense") == "application/octet-stream")

# ============================================================
print("\n=== 12. zero positives is a finding, not a missing result ===")
# ============================================================

quiet_case = store.save({"patient": {"name": "Das, Shuvo"}, "clinical": {}})["case_id"]

quiet_ecg = {
    "predictions": [
        {"name": name, "probability": 0.2, "positive": False}
        for name in ("NORM", "MI", "STTC", "CD", "HYP")
    ],
    # Every class below threshold. This is a completed analysis whose answer
    # happens to be "none of these" — the flag must say analysed even though
    # the count is zero.
    "positive_classes": [],
    "threshold": 0.5,
    "input": {"filename": "quiet.npy"},
}

quiet = store.save({"case_id": quiet_case, "ecg": quiet_ecg})

check("analysed with nothing called is still analysed",
      quiet["ecg_analyzed"] is True)
check("count is zero", quiet["ecg_positive_count"] == 0)
check("the predictions are still there to show",
      len(quiet["ecg"]["predictions"]) == 5)

quiet_summary = store.list(search=quiet_case)[0]
check("the sidebar can tell 'analysed, none called' from 'not analysed'",
      quiet_summary["ecg_analyzed"] is True
      and quiet_summary["ecg_positive_count"] == 0)

# A metadata-only re-save must not turn a real zero into a wiped zero, and
# must not disturb the other modality either.
renamed = store.save({"case_id": quiet_case, "patient": {"name": "Das, S."}})
check("zero positives survives a metadata-only save",
      renamed["ecg_analyzed"] is True and renamed["ecg"] is not None)
check("ecg_filename survives too",
      store.list(search=quiet_case)[0]["ecg_filename"] == "quiet.npy",
      repr(store.list(search=quiet_case)[0]["ecg_filename"]))

# ============================================================
print("\n=== 13. the two modalities do not overwrite each other ===")
# ============================================================

both = store.save({"case_id": ecg_case, "echo": echo_result, "images": images})

check("adding echo keeps the ECG", both["ecg_analyzed"] is True)
check("and keeps its positive count", both["ecg_positive_count"] == 3)
check("and keeps its analysis payload",
      both["ecg"]["positive_classes"] == ["MI", "STTC", "HYP"])
check("while the echo landed", both["echo_analyzed"] is True)
check("with its own structure count", both["structures_found"] == 2)

# Now the reverse: an ECG-only save on a case that already has echo.
ecg_again = dict(ecg_result, positive_classes=["MI"],
                 input={"filename": "second.hea"})
ecg_again.pop("figures")
after = store.save({"case_id": ecg_case, "ecg": ecg_again})

check("re-analysing the ECG does not disturb the echo",
      after["echo_analyzed"] is True and after["structures_found"] == 2)
check("the echo filename is untouched",
      store.list(search=ecg_case)[0]["echo_filename"] == "a4c_frame.png")
check("the new ECG count replaced the old one",
      after["ecg_positive_count"] == 1, str(after["ecg_positive_count"]))
check("the previously stored figures are still on disk",
      store.read_image(ecg_case, "ecg_strip") is not None)

summary = store.list(search=ecg_case)[0]
check("the sidebar row carries both modality flags",
      summary["echo_analyzed"] is True and summary["ecg_analyzed"] is True)

# ============================================================
print("\n=== 14. an older database is migrated, not rejected ===")
# ============================================================

# The pre-ECG schema, verbatim. CREATE TABLE IF NOT EXISTS is a no-op against
# a file that already has a `cases` table, so without the ALTER TABLE pass this
# database opens cleanly and then fails on the first ECG save.
LEGACY_DDL = """
CREATE TABLE cases (
    case_id             TEXT PRIMARY KEY,
    patient_name        TEXT NOT NULL DEFAULT '',
    patient_mrn         TEXT NOT NULL DEFAULT '',
    date_of_birth       TEXT NOT NULL DEFAULT '',
    sex                 TEXT NOT NULL DEFAULT '',
    study_date          TEXT NOT NULL DEFAULT '',
    referring_clinician TEXT NOT NULL DEFAULT '',
    notes               TEXT NOT NULL DEFAULT '',
    clinical_json       TEXT NOT NULL DEFAULT '{}',
    echo_json           TEXT,
    files_json          TEXT NOT NULL DEFAULT '{}',
    echo_analyzed       INTEGER NOT NULL DEFAULT 0,
    structures_found    INTEGER NOT NULL DEFAULT 0,
    echo_filename       TEXT NOT NULL DEFAULT '',
    created_at          TEXT NOT NULL,
    updated_at          TEXT NOT NULL
);
CREATE TABLE case_messages (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    case_id     TEXT NOT NULL,
    role        TEXT NOT NULL,
    text        TEXT NOT NULL,
    model       TEXT NOT NULL DEFAULT '',
    device      TEXT NOT NULL DEFAULT '',
    created_at  TEXT NOT NULL,
    FOREIGN KEY (case_id) REFERENCES cases (case_id) ON DELETE CASCADE
);
"""

legacy_dir = WORK / "legacy"
legacy_dir.mkdir(parents=True, exist_ok=True)
legacy_db = legacy_dir / "cardiovision.db"

seed = sqlite3.connect(legacy_db)
seed.executescript(LEGACY_DDL)
seed.execute(
    """
    INSERT INTO cases (case_id, patient_name, patient_mrn, clinical_json,
                       echo_json, files_json, echo_analyzed, structures_found,
                       echo_filename, created_at, updated_at)
    VALUES (?,?,?,?,?,?,?,?,?,?,?)
    """,
    ("CV-20250101-OLD001", "Legacy, Patient", "MRN-OLD", '{"age": "71"}',
     '{"analyzed": true, "structures": []}', "{}", 1, 3, "old_frame.png",
     "2025-01-01T00:00:00+00:00", "2025-01-01T00:00:00+00:00"),
)
seed.commit()

legacy_columns = {
    row[1] for row in seed.execute("PRAGMA table_info(cases)").fetchall()
}
seed.close()

check(
    "the legacy fixture really is missing the ECG columns",
    not (legacy_columns & {"ecg_json", "ecg_analyzed",
                           "ecg_positive_count", "ecg_filename"}),
    f"{len(legacy_columns)} columns",
)

upgraded = CaseStore(db_path=legacy_db, files_dir=legacy_dir / "cases")
upgraded.connect()

migrated_columns = {
    row["name"]
    for row in upgraded._require().execute("PRAGMA table_info(cases)").fetchall()
}

for column in ("ecg_json", "ecg_analyzed", "ecg_positive_count", "ecg_filename"):
    check(f"{column} was added by the migration", column in migrated_columns)

old = upgraded.get("CV-20250101-OLD001")
check("the existing row survived", old is not None)
check("its echo findings are intact",
      old["echo_analyzed"] is True and old["structures_found"] == 3)
check("its patient details are intact", old["patient"]["mrn"] == "MRN-OLD")
check("it reports no ECG rather than crashing",
      old["ecg"] is None and old["ecg_analyzed"] is False)
check("the created_at from 2025 was not rewritten",
      old["created_at"].startswith("2025-01-01"))

# The whole point: an ECG can now be attached to a case that predates ECG.
attached = upgraded.save({"case_id": "CV-20250101-OLD001", "ecg": quiet_ecg})
check("an ECG saves against a migrated row", attached["ecg_analyzed"] is True)
check("and the legacy echo is still there", attached["structures_found"] == 3)

# Reconnecting must not try to add the columns a second time.
upgraded.close()
again = CaseStore(db_path=legacy_db, files_dir=legacy_dir / "cases")
reconnect_error = None
try:
    again.connect()
except Exception as error:                                  # pragma: no cover
    reconnect_error = f"{type(error).__name__}: {error}"

check("re-opening a migrated database is a no-op", reconnect_error is None,
      reconnect_error or "clean")
check("and the ECG saved a moment ago is still readable",
      again.get("CV-20250101-OLD001")["ecg_analyzed"] is True)
again.close()

# ============================================================
print("\n=== 15. ECG context carries every number's qualifier ===")
# ============================================================


def ecg_prediction(name, label, probability, positive, precision, caveat=None):
    return {
        "name": name, "label": label, "probability": probability,
        "positive": positive, "threshold": 0.5, "caveat": caveat,
        "operating_point": {"precision": precision},
    }


HYP_CAVEAT = (
    "Hypertrophy is the weakest of the five classes by a wide margin: average "
    "precision 0.478 and precision 0.361 at the 0.5 threshold. Roughly two in "
    "three positive hypertrophy calls from this model are false."
)

context_ecg = {
    "model": {"architecture": "ECGResNet1D",
              "metrics": {"macro_AUROC": 0.9125, "test_records": 3232,
                          "test_patients": 2793}},
    "input": {"format": "wfdb", "record_name": "HR00025",
              "sampling_frequency_hz": 500, "resampled_to_hz": 100,
              "lead_names": list(ECG_LEAD_NAMES),
              "lead_order_matches_training": True, "units": "mV"},
    "preprocessing": {"normalization": ECG_NORMALIZATION},
    "threshold": 0.5,
    "predictions": [
        ecg_prediction("NORM", "Normal ECG", 0.07, False, 0.8009),
        ecg_prediction("MI", "Myocardial infarction", 0.81, True, 0.6865),
        # Deliberately just under the line: the whole reason every probability
        # is listed is so this one is not reported as a clean negative.
        ecg_prediction("STTC", "ST/T change", 0.46, False, 0.6166),
        ecg_prediction("CD", "Conduction disturbance", 0.11, False, 0.7113),
        ecg_prediction("HYP", "Hypertrophy", 0.55, True, 0.3614, HYP_CAVEAT),
    ],
    "saliency_available": True,
    "saliency_class": "MI",
    "lead_attribution": [{"name": "V2", "score": 1.0},
                         {"name": "V3", "score": 0.88}],
}

ecg_text = case_context.build_case_context({
    "case_id": "CV-20260826-ECG001",
    "patient": patient,
    "clinical": clinical,
    "ecg": context_ecg,
    "modalities_provided": {"echo": False, "ccta": False, "ecg": True},
})

check("an ECG-only case produces a context", bool(ecg_text))
check("the ECG section is labelled as AI output",
      "ELECTROCARDIOGRAPHY — AI SCREENING RESULT" in ecg_text)
check("the patient's name is still withheld",
      "Ayesha" not in ecg_text and "Rahman" not in ecg_text)

check("all five classes are listed, not just the calls",
      all(f"({name})" in ecg_text for name in ECG_LEAD_NAMES[:0] or
          ("NORM", "MI", "STTC", "CD", "HYP")))
check("the sub-threshold near-miss is present with its number",
      "p = 0.46" in ecg_text and "not called" in ecg_text)
check("positives are ordered above the classes below threshold",
      ecg_text.index("p = 0.81") < ecg_text.index("p = 0.46")
      < ecg_text.index("p = 0.07"))

check(
    "a positive call carries the precision it was measured at",
    "precision 0.69" in ecg_text and "31% of positive calls" in ecg_text,
)
check(
    "the HYP false-positive rate is stated as a number, not implied",
    "precision 0.36" in ecg_text and "64% of positive calls" in ecg_text,
)
check(
    "and the HYP caveat is raised as a warning on top of that",
    "WARNING — HYP was called positive" in ecg_text
    and "two in three" in ecg_text,
)
check(
    "the operating point is NOT attached to classes that stayed quiet",
    "precision 0.62" not in ecg_text and "precision 0.71" not in ecg_text,
    "STTC 0.6166 / CD 0.7113 must not appear",
)
check(
    "independent sigmoids are spelled out, so probabilities are not read as shares",
    "INDEPENDENT sigmoids" in ecg_text and "do not sum to 1" in ecg_text,
)
check(
    "the macro AUROC is labelled as describing the model, not the recording",
    "NOT the certainty of this particular reading" in ecg_text,
)
check(
    "saliency is described as attention, not as localisation",
    "where the model looked, NOT where an abnormality is" in ecg_text,
)
check(
    "the per-lead normalisation is tied to the hypertrophy weakness",
    "removes absolute voltage" in ecg_text and "hypertrophy" in ecg_text.lower(),
)
check(
    "the model's scope limits are stated",
    all(phrase in ecg_text for phrase in
        ("does not measure heart rate", "does not localise an infarct",
         "atrial fibrillation", "does not output a diagnosis")),
)
check(
    "ECG is no longer listed as an unavailable modality",
    "Electrocardiography: not available" not in ecg_text,
)
check(
    "but the modalities that never got a model still are",
    "Multimodal fusion: not available" in ecg_text
    and "Clinical risk" in ecg_text,
)
# CCTA acquired a trained model, so "not available" would now be a lie about
# the system. What must survive is the distinction between "no model" and
# "model exists, nothing analysed" — the second still forbids CT findings.
check(
    "CCTA is no longer listed as a modality without a model",
    "Coronary CT angiography: not available" not in ecg_text,
)
check(
    "and is instead described as available but unanalysed in this case",
    "a CT lumen segmentation model is " in ecg_text
    and "no CT volume has been analysed in this case" in ecg_text,
)
check(
    "which still forbids inventing coronary findings",
    "Do not infer coronary anatomy" in ecg_text,
)
check(
    "and the missing echo is still called out",
    "no echo image has been analysed" in ecg_text,
)

# ---- nothing called ------------------------------------------------
quiet_text = case_context.build_case_context({
    "patient": patient,
    "clinical": clinical,
    "ecg": dict(
        context_ecg,
        predictions=[
            ecg_prediction(name, name, 0.2, False, 0.5)
            for name in ("NORM", "MI", "STTC", "CD", "HYP")
        ],
    ),
})

check(
    "an ECG with nothing called is not reported as a normal ECG",
    "no superclass reached the threshold" in quiet_text.lower()
    and "do not upgrade it" in quiet_text,
)
check(
    "and it explains that NORM itself was not called",
    "NORM is itself one of the five classes and it was not called" in quiet_text,
)

# ---- NORM positive alongside an abnormal call ----------------------
contradictory_text = case_context.build_case_context({
    "patient": patient,
    "ecg": dict(
        context_ecg,
        predictions=[
            ecg_prediction("NORM", "Normal ECG", 0.72, True, 0.8009),
            ecg_prediction("MI", "Myocardial infarction", 0.68, True, 0.6865),
        ],
    ),
})

check("NORM plus MI is flagged as a contradiction",
      "CONTRADICTION" in contradictory_text)
check("and the model is told to report the uncertainty, not resolve it",
      "genuinely uncertain" in contradictory_text
      and "whichever answer reads better" in contradictory_text)

# ---- wrong lead order ---------------------------------------------
scrambled_text = case_context.build_case_context({
    "patient": patient,
    "ecg": dict(
        context_ecg,
        input=dict(context_ecg["input"], lead_order_matches_training=False,
                   lead_names=["II", "I", "III", "aVR", "aVL", "aVF",
                               "V1", "V2", "V3", "V4", "V5", "V6"]),
    ),
})

check("a lead-order mismatch invalidates the whole reading in the prompt",
      "did not arrive in the order the model was trained on" in scrambled_text
      and "every probability above may be wrong" in scrambled_text)

# ---- no ECG at all -------------------------------------------------
echo_only_text = case_context.build_case_context({
    "patient": patient,
    "clinical": clinical,
    "echo": echo_result,
    "modalities_provided": {"echo": True, "ccta": False, "ecg": False},
})

check(
    "a case with no ECG says so without claiming the model is missing",
    "no ECG has been analysed in this case" in echo_only_text
    and "No trained model exists" not in echo_only_text.split(
        "Electrocardiography:")[1].split("\n")[0],
)
check(
    "and forbids inferring a rhythm from the other data",
    "Do not infer a rhythm" in echo_only_text,
)
check(
    "the ECG section itself is absent when there is no ECG",
    "ELECTROCARDIOGRAPHY — AI SCREENING RESULT" not in echo_only_text,
)

# ---- saliency unavailable ------------------------------------------
no_saliency_text = case_context.build_case_context({
    "patient": patient,
    "ecg": dict(context_ecg, saliency_available=False, lead_attribution=[]),
})

check("a missing saliency map is stated, not silently omitted",
      "No saliency was computed" in no_saliency_text
      and "Do not speculate about lead involvement" in no_saliency_text)
check("and no lead ranking is invented",
      "V2 (1.00)" not in no_saliency_text)

# ============================================================
print("\n=== 16. ECG frontend wiring (static, but specific) ===")
# ============================================================

# The same treatment section 10 gives the echo wiring. None of this can be
# executed here — vite and oxlint are macOS-native binaries — so the checks
# pin the handful of lines where a plausible-looking mistake produces a
# confident wrong answer on screen rather than an error.

ecg_view = (FRONTEND / "components" / "EcgResult.jsx").read_text()
css = (FRONTEND / "App.css").read_text()

# JSX wraps prose across lines, so a sentence in the source is not a
# contiguous string. Prose assertions run against the collapsed copy;
# code assertions stay on the original.
ecg_prose = " ".join(ecg_view.split())

check(
    "the ECG modality accepts a set of files, because WFDB is a pair",
    "multiple: true" in app and 'multiple={Boolean(modality.multiple)}' in app,
)
check(
    "the primary file is the one that is not a .dat",
    'endsWith(".dat")' in app and "splitEcgSelection" in app,
)
check(
    "companions are sent under their own field, not merged into file",
    'formData.append("companions", companion)' in api,
)
check(
    "the sampling rate is only sent when the operator supplied one",
    "Number.isFinite(samplingFrequency) && samplingFrequency > 0" in api,
)
check(
    "echo and ECG have separate analyse actions",
    "runEcgAnalysis" in app and "runAnalysis" in app
    and "disabled={!canAnalyzeEcg}" in app,
)
check(
    "and separate error banners, so one failure does not mask the other",
    "ecgError" in app and "analysisError" in app
    and "ECG analysis failed" in app and "Echo analysis failed" in app,
)
check(
    "the ECG result is persisted under its own key",
    "payload.ecg = { ...rest, analyzed: true }" in app
    and "payload.ecg_figures" in app,
)
check(
    "the results panel opens off the results themselves, not a stored flag",
    "Boolean(echoResult) || Boolean(ecgResult) || Boolean(cctaResult)" in app
    and "setAnalysisComplete" not in app,
)
check(
    "restored ECG figures follow the paths the backend sent",
    "initialCase?.ecg_figures" in app and "fetchCaseFigures(stored)" in app,
)
check(
    "the ECG blob URLs are released on unmount too",
    "releaseImages(figureUrls.current)" in app,
)
check(
    "the ECG reaches the language model with its qualifiers",
    all(key in app for key in ("weak_class_warnings:", "lead_attribution:",
                               "saliency_available:", "positive_classes:")),
)
check(
    # The figures are data URLs; sending them would be tens of kilobytes of
    # base64 into a prompt that cannot read an image.
    "but the figures are not sent to the language model",
    "figures: ecgResult.figures" not in app,
)
check(
    "the overview no longer claims the ECG has no model",
    '<Metric label="ECG" value="No model"' not in app
    and 'label="ECG classification"' in app,
)
check(
    "the overview stops saying 'not analysed' about a classified recording",
    'ecgAnalysed ? "" : " (not analysed)"' in app,
)
check(
    "a WFDB pair with no .dat selected is visible as such",
    "cv-modality-companions" in app and "cv-modality-companions" in css,
)

# ---- the honesty rules, in the component that renders them ----------

check(
    "zero positives is stated as a result and not as a normal ECG",
    "not the same as a normal ECG" in ecg_prose
    and "none of these five" in ecg_prose,
)
check(
    "all five probabilities are rendered, not only the calls",
    "ranked.map((prediction)" in ecg_view
    and "Below threshold" in ecg_view,
)
check(
    "the threshold is drawn on the same axis as the probability",
    "left: `${threshold * 100}%`" in ecg_view
    and ".cv-ecg-prediction-bar b" in css,
)
check(
    "the operating point is attached only to a positive call",
    "{positive && (" in ecg_view and "cv-ecg-operating-point" in ecg_view,
)
check(
    "precision is also stated as a false-call rate",
    "falseCallRate" in ecg_view
    and "of positive calls for this class were wrong" in ecg_prose,
)
check(
    "the per-class table is shown in full, so the weak class is visible",
    "cv-ecg-metric-table" in ecg_view and ".cv-ecg-metric-table tr.weak" in css,
)
check(
    "a NORM call alongside an abnormal one is surfaced as a contradiction",
    'positives.includes("NORM") && positives.length > 1' in ecg_view,
)
check(
    "saliency is framed as where the model looked",
    "where the model looked" in ecg_prose,
)
check(
    "and an unavailable attribution renders nothing rather than a zero map",
    "cv-xai-unavailable" in ecg_view
    and "saliencyAvailable ? (" in ecg_view,
)
check(
    "the per-lead normalisation explains why hypertrophy is weak",
    "millimetre voltage criteria" in ecg_prose,
)
check(
    "a lead-order mismatch invalidates the reading on screen",
    "lead_order_matches_training === false" in ecg_view
    and "treated as unreliable" in ecg_prose,
)
check(
    "the validation split is kept apart from the test figures",
    "cv-metrics-note secondary" in ecg_view
    and "not an independent" in ecg_prose,
)
check(
    "the device fallback is reported when it happens",
    "result.device !== result.configured_device" in ecg_view,
)
check(
    "the scope disclaimer names what the model does not do",
    all(phrase in ecg_prose for phrase in
        ("does not measure heart rate", "atrial fibrillation",
         "does not localise an infarct", "does not produce a diagnosis")),
)

# Every cv-ecg-* class the component asks for has to exist, or the panel
# renders as unstyled stacked text and nobody notices until it ships.
used_classes = sorted({
    token
    for match in re.findall(r'className=\{?[`"]([^`"]+)[`"]', ecg_view)
    for token in match.split()
    if token.startswith("cv-")
})
unstyled = [name for name in used_classes if f".{name}" not in css]

check("every class the ECG view uses is styled", not unstyled,
      ", ".join(unstyled) or f"{len(used_classes)} classes")

# ---- cleanup ---------------------------------------------------------
store.close()
shutil.rmtree(WORK, ignore_errors=True)

print("\n" + "=" * 62)
if FAILURES:
    print(f"{len(FAILURES)} FAILURE(S) out of {CHECKS} checks:")
    for item in FAILURES:
        print(f"  - {item}")
    sys.exit(1)

print(f"ALL {CHECKS} CASE-LIFECYCLE CHECKS PASSED")
