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

    cd backend && python3 test_case_lifecycle.py

No torch and no model weights required.
"""

from __future__ import annotations

import base64
import shutil
import sys
import tempfile
import types
from pathlib import Path

BACKEND = Path(__file__).resolve().parent
REPO = BACKEND.parent
FRONTEND = REPO / "frontend" / "src"

# config.py picks a device at import time. Importing torch just for that costs
# seconds and is pointless here, so stub it — but only if it is genuinely
# absent, so a real install is never shadowed.
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

WORK = Path(tempfile.mkdtemp(prefix="cv-lifecycle-"))

sys.path.insert(0, str(BACKEND))

from database import CaseStore, derive_age  # noqa: E402

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

import case_context  # noqa: E402

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
    "if (!targetCase && storageReady)" in app,
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
