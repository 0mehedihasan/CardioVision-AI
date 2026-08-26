"""
CardioVision AI — local case store.

Persists patient cases to a SQLite file next to the project. Nothing is
uploaded anywhere; the database and its image files never leave this machine.

LAYOUT
------
    data/cardiovision.db          patient records, findings, Q&A transcripts
    data/cases/<case_id>/         rendered PNGs + the original upload

Images are files on disk rather than blobs in the database. One echo analysis
produces six PNGs, one CCTA analysis up to sixteen, and the ECG two SVGs — a few
megabytes per case; putting that in a row would make the case list slow to query
and the database awkward to back up.

A NOTE ON WHAT IS STORED
------------------------
These records contain patient identifiers — name, MRN, date of birth. The
SQLite file is NOT encrypted. Anyone with filesystem access can read it.
That is an acceptable trade-off for a local research tool on a single
machine, but it means the data directory should not be synced to cloud
storage or committed to git. See the .gitignore rule this module's setup
writes.
"""

from __future__ import annotations

import base64
import json
import shutil
import sqlite3
import threading
import uuid
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Optional

from cardiovision.config import CASE_DB_PATH, CASE_FILES_DIR, DATA_DIR


# ============================================================
# SCHEMA
# ============================================================

_SCHEMA = """
CREATE TABLE IF NOT EXISTS cases (
    case_id             TEXT PRIMARY KEY,

    -- patient identity
    patient_name        TEXT NOT NULL DEFAULT '',
    patient_mrn         TEXT NOT NULL DEFAULT '',
    date_of_birth       TEXT NOT NULL DEFAULT '',
    sex                 TEXT NOT NULL DEFAULT '',

    -- study metadata
    study_date          TEXT NOT NULL DEFAULT '',
    referring_clinician TEXT NOT NULL DEFAULT '',
    notes               TEXT NOT NULL DEFAULT '',

    -- payloads, stored as JSON text
    clinical_json       TEXT NOT NULL DEFAULT '{}',
    ccta_json           TEXT,
    echo_json           TEXT,
    ecg_json            TEXT,
    report_json         TEXT,
    files_json          TEXT NOT NULL DEFAULT '{}',

    -- denormalised for the case list, so listing never parses JSON
    ccta_analyzed       INTEGER NOT NULL DEFAULT 0,
    ccta_lumen_voxels   INTEGER NOT NULL DEFAULT 0,
    ccta_filename       TEXT NOT NULL DEFAULT '',

    echo_analyzed       INTEGER NOT NULL DEFAULT 0,
    structures_found    INTEGER NOT NULL DEFAULT 0,
    echo_filename       TEXT NOT NULL DEFAULT '',

    ecg_analyzed        INTEGER NOT NULL DEFAULT 0,
    ecg_positive_count  INTEGER NOT NULL DEFAULT 0,
    ecg_filename        TEXT NOT NULL DEFAULT '',

    created_at          TEXT NOT NULL,
    updated_at          TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_cases_updated
    ON cases (updated_at DESC);

CREATE INDEX IF NOT EXISTS idx_cases_patient
    ON cases (patient_name, patient_mrn);

CREATE TABLE IF NOT EXISTS case_messages (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    case_id     TEXT NOT NULL,
    role        TEXT NOT NULL,
    text        TEXT NOT NULL,
    model       TEXT NOT NULL DEFAULT '',
    device      TEXT NOT NULL DEFAULT '',
    created_at  TEXT NOT NULL,

    FOREIGN KEY (case_id) REFERENCES cases (case_id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_messages_case
    ON case_messages (case_id, id);
"""


# ============================================================
# MIGRATIONS
# ============================================================
#
# CREATE TABLE IF NOT EXISTS above only builds a *new* database. A database
# created before the ECG pipeline existed already has a `cases` table, so the
# IF NOT EXISTS makes the statement a no-op and the four ECG columns never
# appear — the first ECG save then fails on "no such column" against a file
# that opened without complaint.
#
# Each entry is (column, DDL). Applied only when PRAGMA table_info says the
# column is absent, so running this on an up-to-date database does nothing and
# running it twice is harmless. Kept as a list rather than a version counter
# because the check is per-column: a database that was half-migrated by an
# interrupted upgrade is repaired rather than skipped.
#
# ADDITIVE ONLY. Never rename or drop a column here — this file holds the
# operator's patient records, and there is no second copy to restore from.

_MIGRATIONS: tuple[tuple[str, str], ...] = (
    ("ecg_json", "ALTER TABLE cases ADD COLUMN ecg_json TEXT"),
    (
        "ecg_analyzed",
        "ALTER TABLE cases ADD COLUMN ecg_analyzed INTEGER NOT NULL DEFAULT 0",
    ),
    (
        "ecg_positive_count",
        "ALTER TABLE cases ADD COLUMN ecg_positive_count INTEGER NOT NULL DEFAULT 0",
    ),
    (
        "ecg_filename",
        "ALTER TABLE cases ADD COLUMN ecg_filename TEXT NOT NULL DEFAULT ''",
    ),
    ("ccta_json", "ALTER TABLE cases ADD COLUMN ccta_json TEXT"),
    (
        "ccta_analyzed",
        "ALTER TABLE cases ADD COLUMN ccta_analyzed INTEGER NOT NULL DEFAULT 0",
    ),
    (
        "ccta_lumen_voxels",
        "ALTER TABLE cases ADD COLUMN ccta_lumen_voxels INTEGER NOT NULL DEFAULT 0",
    ),
    (
        "ccta_filename",
        "ALTER TABLE cases ADD COLUMN ccta_filename TEXT NOT NULL DEFAULT ''",
    ),
    ("report_json", "ALTER TABLE cases ADD COLUMN report_json TEXT"),
)


# ============================================================
# STORED ASSETS
# ============================================================
#
# Rendered figures are files on disk, keyed by name. The suffix is part of the
# table rather than assumed, because the echo renders are PNG raster and the
# ECG renders are SVG: the strip is a 10-second grid of hairlines and text,
# which stays sharp at any zoom as vector and turns to mush as a raster of any
# reasonable size.
#
# This map is also the allow-list. read_image() rejects any name that is not a
# key here, which is what stops a crafted `name` from walking out of the case
# directory — so a new figure must be added here to be storable at all.

_ECHO_IMAGE_KEYS = (
    "original",
    "mask",
    "overlay",
    "saliency",
    "saliency_overlay",
    "combined",
)

_ECG_FIGURE_KEYS = (
    "ecg_strip",
    "ecg_lead_attribution",
)

# The CCTA renderer produces three views per array axis, a maximum-intensity
# projection per axis, and four Grad-CAM panels. Generated rather than typed out
# because the renderer generates them the same way, and a hand-maintained copy
# of a generated list is a list that will disagree with it.
_CCTA_PLANE_VIEWS = ("ct", "overlay", "probability", "mip")
_CCTA_PLANES = ("axis0", "axis1", "axis2")
_CCTA_GRADCAM_VIEWS = (
    "gradcam",
    "gradcam_overlay",
    "gradcam_ct",
    "gradcam_mask_overlay",
)

_ASSET_SUFFIX: dict[str, str] = {
    **{key: ".png" for key in _ECHO_IMAGE_KEYS},
    **{key: ".svg" for key in _ECG_FIGURE_KEYS},
}

_ASSET_MEDIA_TYPE = {
    ".png": "image/png",
    ".svg": "image/svg+xml",
}

# The ECG renderer returns short keys ("strip", "lead_attribution"). They are
# stored under the prefixed names above so that adding, say, a CCTA "strip"
# later cannot collide with the ECG one inside a single case directory.
_ECG_FIGURE_ALIASES = {
    "strip": "ecg_strip",
    "lead_attribution": "ecg_lead_attribution",
}

# Same treatment for CCTA, and here the collision is not hypothetical: the CCTA
# renderer emits "gradcam" and "overlay"-suffixed keys, and the echo renderer
# already owns "overlay". Prefixing on the way in keeps one case directory
# unambiguous while both renderers keep their own natural names.
_CCTA_FIGURE_ALIASES = {
    **{
        f"{plane}_{view}": f"ccta_{plane}_{view}"
        for plane in _CCTA_PLANES
        for view in _CCTA_PLANE_VIEWS
    },
    **{view: f"ccta_{view}" for view in _CCTA_GRADCAM_VIEWS},
}

_CCTA_FIGURE_KEYS = tuple(_CCTA_FIGURE_ALIASES.values())

_ASSET_SUFFIX.update({key: ".png" for key in _CCTA_FIGURE_KEYS})

# One lookup for the store to translate an incoming figure name. Merged rather
# than tried in sequence so that a future collision between two alias maps
# fails loudly in review instead of resolving by argument order at runtime.
_FIGURE_ALIASES: dict[str, str] = {
    **_ECG_FIGURE_ALIASES,
    **_CCTA_FIGURE_ALIASES,
}


def media_type_for(name: str) -> str:
    """
    The Content-Type for a stored figure.

    Serving an SVG as image/png makes the browser refuse to draw it, which
    looks exactly like a missing file, so the endpoint asks rather than
    assuming.
    """
    return _ASSET_MEDIA_TYPE.get(_ASSET_SUFFIX.get(name, ""), "application/octet-stream")



class CaseStoreError(RuntimeError):
    """Raised when a case cannot be read or written."""


@dataclass
class CaseSummary:
    """The subset of a case shown in the sidebar list."""

    case_id: str
    patient_name: str
    patient_mrn: str
    sex: str
    study_date: str
    ccta_analyzed: bool
    ccta_lumen_voxels: int
    ccta_filename: str
    echo_analyzed: bool
    structures_found: int
    echo_filename: str
    ecg_analyzed: bool
    ecg_positive_count: int
    ecg_filename: str
    has_report: bool
    created_at: str
    updated_at: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "patient_name": self.patient_name,
            "patient_mrn": self.patient_mrn,
            "sex": self.sex,
            "study_date": self.study_date,
            "ccta_analyzed": self.ccta_analyzed,
            "ccta_lumen_voxels": self.ccta_lumen_voxels,
            "ccta_filename": self.ccta_filename,
            "echo_analyzed": self.echo_analyzed,
            "structures_found": self.structures_found,
            "echo_filename": self.echo_filename,
            "ecg_analyzed": self.ecg_analyzed,
            "ecg_positive_count": self.ecg_positive_count,
            "ecg_filename": self.ecg_filename,
            "has_report": self.has_report,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            # A blank name is normal — a case can be saved before the
            # demographics are filled in — so the UI needs something to show.
            "display_name": self.patient_name or "Unnamed patient",
        }


# ============================================================
# HELPERS
# ============================================================

def _now() -> str:
    """
    An ISO timestamp with microsecond precision.

    Precision matters here: updated_at is the sort key for the case list, and
    truncating to whole seconds made two cases saved in the same second sort
    arbitrarily against each other. Microseconds plus the rowid tiebreak in
    list() give a total order.
    """
    return datetime.now(timezone.utc).isoformat()


def _text(value: Any) -> str:
    """Coerce to a stripped string. None becomes empty, never 'None'."""
    if value is None:
        return ""
    return str(value).strip()


def derive_age(date_of_birth: str, reference: Optional[str] = None) -> Optional[int]:
    """
    Age in whole years from an ISO date of birth.

    Derived rather than stored: an age typed in once is wrong a year later,
    whereas a date of birth stays correct. Returns None on anything
    unparseable so a malformed date degrades to "not recorded" instead of
    raising mid-request.
    """
    dob = _text(date_of_birth)
    if not dob:
        return None

    try:
        born = date.fromisoformat(dob[:10])
    except ValueError:
        return None

    if reference:
        try:
            today = date.fromisoformat(_text(reference)[:10])
        except ValueError:
            today = date.today()
    else:
        today = date.today()

    if born > today:
        return None

    years = today.year - born.year

    # Subtract one if the birthday has not happened yet this year.
    if (today.month, today.day) < (born.month, born.day):
        years -= 1

    return max(0, years)


def new_case_id() -> str:
    """
    A readable, sortable, collision-free case identifier.

    CV-<YYYYMMDD>-<6 hex>. The date prefix makes a directory listing
    chronological; the random suffix means two cases created in the same
    second cannot collide, which a plain counter would risk.
    """
    stamp = datetime.now().strftime("%Y%m%d")
    return f"CV-{stamp}-{uuid.uuid4().hex[:6].upper()}"


def _decode_data_url(value: str) -> Optional[bytes]:
    """Extract the bytes from a data: URL, or None if it is not one."""
    if not isinstance(value, str) or "base64," not in value:
        return None

    try:
        return base64.b64decode(value.split("base64,", 1)[1])
    except (ValueError, TypeError):
        return None


# ============================================================
# STORE
# ============================================================

class CaseStore:
    """
    SQLite-backed case persistence.

    One connection guarded by a lock. FastAPI runs sync endpoints in a
    threadpool and SQLite connections are not safe to share across threads,
    so either every call opens its own connection or a single connection is
    serialised. Serialising is simpler and a local single-operator tool has
    no concurrency to speak of.
    """

    def __init__(
        self,
        db_path: Path = CASE_DB_PATH,
        files_dir: Path = CASE_FILES_DIR,
    ) -> None:
        self._db_path = Path(db_path)
        self._files_dir = Path(files_dir)
        self._lock = threading.Lock()
        self._connection: Optional[sqlite3.Connection] = None
        # Kept so /api/health can say *why* saving is unavailable. "Case
        # storage unavailable" with no reason is the kind of message that
        # sends someone hunting through server logs.
        self._connect_error: Optional[str] = None

    # ---- lifecycle ------------------------------------------------

    def connect(self) -> None:
        """Create the data directory, open the database, apply the schema."""
        try:
            self._open()
        except Exception as error:
            self._connect_error = f"{type(error).__name__}: {error}"
            raise

        self._connect_error = None

    def _open(self) -> None:
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._files_dir.mkdir(parents=True, exist_ok=True)

        connection = sqlite3.connect(
            self._db_path,
            # Guarded by our own lock, so SQLite's thread check would only
            # reject legitimate threadpool access.
            check_same_thread=False,
        )
        connection.row_factory = sqlite3.Row

        # ON DELETE CASCADE is off by default in SQLite, which would orphan
        # every message when its case is deleted.
        connection.execute("PRAGMA foreign_keys = ON")

        # Survives an abrupt shutdown without corrupting the file.
        connection.execute("PRAGMA journal_mode = WAL")

        connection.executescript(_SCHEMA)
        self._migrate(connection)
        connection.commit()

        self._connection = connection
        self._write_gitignore()

    @staticmethod
    def _migrate(connection: sqlite3.Connection) -> None:
        """
        Bring an older database up to the current column set.

        Additive only, and driven by what the file actually contains rather
        than by a stored version number — see the note on _MIGRATIONS.
        """
        present = {
            row["name"]
            for row in connection.execute("PRAGMA table_info(cases)").fetchall()
        }

        # An empty set means the table does not exist yet, which cannot happen
        # after executescript succeeded. Guarding anyway: ALTER TABLE on a
        # missing table would turn a fresh install into a startup failure.
        if not present:
            return

        for column, statement in _MIGRATIONS:
            if column not in present:
                connection.execute(statement)

    def close(self) -> None:
        with self._lock:
            if self._connection is not None:
                self._connection.close()
                self._connection = None

    def _write_gitignore(self) -> None:
        """
        Keep patient data out of git.

        Written programmatically because the directory is created at runtime;
        a rule in the root .gitignore would not exist on a fresh clone until
        someone remembered to add it.
        """
        marker = DATA_DIR / ".gitignore"

        if marker.exists():
            return

        try:
            DATA_DIR.mkdir(parents=True, exist_ok=True)
            marker.write_text(
                "# Patient data. Never commit this, and do not sync it to\n"
                "# cloud storage — the database is not encrypted.\n"
                "cardiovision.db\n"
                "cardiovision.db-wal\n"
                "cardiovision.db-shm\n"
                "cases/\n",
                encoding="utf-8",
            )
        except OSError:
            # A missing .gitignore must not stop the service from starting.
            pass

    @property
    def is_ready(self) -> bool:
        return self._connection is not None

    @property
    def connect_error(self) -> Optional[str]:
        """Why the database is not open, or None when it is."""
        return None if self.is_ready else (
            self._connect_error or "The case database was never opened."
        )

    def _require(self) -> sqlite3.Connection:
        if self._connection is None:
            raise CaseStoreError(
                "The case database is not open. This is a startup failure — "
                "check the backend log for a permissions error on the data/ "
                "directory."
            )
        return self._connection

    # ---- images ---------------------------------------------------

    def case_dir(self, case_id: str) -> Path:
        return self._files_dir / case_id

    def _store_images(
        self,
        case_id: str,
        images: Optional[dict[str, str]],
    ) -> dict[str, str]:
        """
        Write rendered data-URL figures out as files.

        Returns a name -> filename map for files_json. Anything that is not a
        decodable data URL is skipped rather than written as garbage, and
        anything whose key is not in the asset table is ignored — a figure the
        store does not know how to serve is not worth writing.
        """
        if not images:
            return {}

        directory = self.case_dir(case_id)
        directory.mkdir(parents=True, exist_ok=True)

        stored: dict[str, str] = {}

        for key, value in images.items():
            name = _FIGURE_ALIASES.get(key, key)
            suffix = _ASSET_SUFFIX.get(name)

            if suffix is None:
                continue

            payload = _decode_data_url(value)

            if payload is None:
                continue

            filename = f"{name}{suffix}"
            (directory / filename).write_bytes(payload)
            stored[name] = filename

        return stored

    def store_source_file(
        self,
        case_id: str,
        filename: str,
        data: bytes,
    ) -> str:
        """
        Keep the original upload alongside the case.

        Without it a saved case could never be re-analysed at a different
        rotation, which is exactly what an operator wants when a result
        looks anatomically wrong.
        """
        directory = self.case_dir(case_id)
        directory.mkdir(parents=True, exist_ok=True)

        # Only the basename is used, so a crafted filename cannot escape the
        # case directory.
        safe = Path(_text(filename) or "source").name
        (directory / f"source_{safe}").write_bytes(data)

        return f"source_{safe}"

    def read_image(self, case_id: str, name: str) -> Optional[bytes]:
        """
        Read one stored figure back. Returns None when it does not exist.

        ``name`` is checked against the asset table, not sanitised, so no
        traversal sequence can reach a path outside the case directory. Pair
        with :func:`media_type_for` to serve it.
        """
        suffix = _ASSET_SUFFIX.get(name)

        if suffix is None:
            return None

        path = self.case_dir(case_id) / f"{name}{suffix}"

        if not path.is_file():
            return None

        return path.read_bytes()

    # ---- write ----------------------------------------------------

    def save(self, payload: dict[str, Any]) -> dict[str, Any]:
        """
        Create or update a case. Returns the stored record.

        An existing case_id updates in place and keeps its created_at, so
        re-saving after editing demographics does not look like a new case.
        """
        connection = self._require()

        case_id = _text(payload.get("case_id")) or new_case_id()

        patient = payload.get("patient") or {}
        clinical = payload.get("clinical") or {}
        ccta = payload.get("ccta")
        echo = payload.get("echo")
        ecg = payload.get("ecg")
        images = dict(payload.get("images") or {})

        # An echo payload that explicitly says analyzed=false is a case where
        # the clinician attached an image but never ran the model.
        echo_analyzed = bool(echo) and echo.get("analyzed") is not False
        structures = (echo or {}).get("structures") or []
        structures_found = sum(
            1 for structure in structures if structure.get("present")
        )

        echo_filename = _text(
            ((echo or {}).get("input") or {}).get("filename")
        )

        # ---- ECG ---------------------------------------------------
        #
        # The two ECG figures are ~200 KB of SVG between them. They are lifted
        # out of the payload and written as files here, for the same reason the
        # echo renders are: ecg_json is read back in full every time a case is
        # opened, and a base64 SVG in a JSON column would be carried on every
        # one of those reads to display a picture the browser can fetch and
        # cache separately.
        #
        # Lifted rather than required-absent on purpose. The caller may send the
        # analysis exactly as the endpoint returned it, figures included; the
        # store then does the right thing instead of silently swallowing 200 KB
        # into a column because the frontend forgot to split it.
        if ecg:
            ecg = dict(ecg)
            nested = ecg.pop("figures", None)
            if isinstance(nested, dict):
                images.update(nested)

        if isinstance(payload.get("ecg_figures"), dict):
            images.update(payload["ecg_figures"])

        ecg_analyzed = bool(ecg) and ecg.get("analyzed") is not False

        # Zero is a real result here, not a missing one. An ECG where all five
        # superclasses fall below the threshold is a completed analysis whose
        # answer is "none of these" — which is why ecg_analyzed is a separate
        # column and the sidebar must read the flag, not the count.
        ecg_positive_count = len((ecg or {}).get("positive_classes") or [])

        ecg_filename = _text(
            ((ecg or {}).get("input") or {}).get("filename")
        )

        # ---- CCTA --------------------------------------------------
        #
        # Same lift as the ECG figures, for a bigger payload: up to sixteen PNG
        # panels. The CCTA analysis dict is also the one place where an array
        # could accidentally be persisted, so `mask` and `probability` never
        # reach this method — the router returns measurements, not volumes.
        if ccta:
            ccta = dict(ccta)
            nested = ccta.pop("figures", None)
            if isinstance(nested, dict):
                images.update(nested)

        if isinstance(payload.get("ccta_figures"), dict):
            images.update(payload["ccta_figures"])

        ccta_analyzed = bool(ccta) and ccta.get("analyzed") is not False

        # The single number worth denormalising. CCTA has one predicted class,
        # so a present/absent count would only ever be 0 or 1; the voxel total
        # is what distinguishes "a plausible coronary tree" from "eleven stray
        # voxels above threshold", and the sidebar can show it directly.
        ccta_lumen_voxels = sum(
            int(finding.get("voxels") or 0)
            for finding in ((ccta or {}).get("findings") or [])
        )

        ccta_filename = _text(
            ((ccta or {}).get("input") or {}).get("filename")
        )

        timestamp = _now()

        with self._lock:
            existing = connection.execute(
                "SELECT created_at, files_json FROM cases WHERE case_id = ?",
                (case_id,),
            ).fetchone()

            created_at = existing["created_at"] if existing else timestamp

            # Preserve any previously stored files (notably the source
            # upload) so a metadata-only re-save cannot wipe the images.
            files: dict[str, str] = {}
            if existing:
                try:
                    files = json.loads(existing["files_json"]) or {}
                except (ValueError, TypeError):
                    files = {}

            if images:
                files.update(self._store_images(case_id, images))

            if payload.get("source_filename"):
                files["source"] = _text(payload["source_filename"])

            row = (
                case_id,
                _text(patient.get("name")),
                _text(patient.get("mrn")),
                _text(patient.get("dateOfBirth")),
                _text(patient.get("sex")) or _text(clinical.get("sex")),
                _text(patient.get("studyDate")),
                _text(patient.get("referringClinician")),
                _text(patient.get("notes")),
                json.dumps(clinical),
                json.dumps(ccta) if ccta else None,
                json.dumps(echo) if echo else None,
                json.dumps(ecg) if ecg else None,
                json.dumps(files),
                1 if ccta_analyzed else 0,
                ccta_lumen_voxels,
                ccta_filename,
                1 if echo_analyzed else 0,
                structures_found,
                echo_filename,
                1 if ecg_analyzed else 0,
                ecg_positive_count,
                ecg_filename,
                created_at,
                timestamp,
            )

            connection.execute(
                """
                INSERT INTO cases (
                    case_id, patient_name, patient_mrn, date_of_birth, sex,
                    study_date, referring_clinician, notes,
                    clinical_json, ccta_json, echo_json, ecg_json, files_json,
                    ccta_analyzed, ccta_lumen_voxels, ccta_filename,
                    echo_analyzed, structures_found, echo_filename,
                    ecg_analyzed, ecg_positive_count, ecg_filename,
                    created_at, updated_at
                ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                ON CONFLICT (case_id) DO UPDATE SET
                    patient_name        = excluded.patient_name,
                    patient_mrn         = excluded.patient_mrn,
                    date_of_birth       = excluded.date_of_birth,
                    sex                 = excluded.sex,
                    study_date          = excluded.study_date,
                    referring_clinician = excluded.referring_clinician,
                    notes               = excluded.notes,
                    clinical_json       = excluded.clinical_json,
                    files_json          = excluded.files_json,
                    updated_at          = excluded.updated_at,

                    -- The four echo columns move together or not at all.
                    -- excluded.echo_json is NULL exactly when this save
                    -- carried no echo payload, which is what a Save after
                    -- editing demographics looks like. Updating the
                    -- denormalised columns anyway would leave the sidebar
                    -- reporting "Echo · 0/3" and a blank filename on a case
                    -- whose segmentation is sitting intact in echo_json.
                    echo_json           = COALESCE(
                        excluded.echo_json, cases.echo_json
                    ),
                    echo_analyzed       = MAX(
                        excluded.echo_analyzed, cases.echo_analyzed
                    ),
                    structures_found    = CASE
                        WHEN excluded.echo_json IS NULL
                        THEN cases.structures_found
                        ELSE excluded.structures_found
                    END,
                    echo_filename       = CASE
                        WHEN excluded.echo_json IS NULL
                        THEN cases.echo_filename
                        ELSE excluded.echo_filename
                    END,

                    -- The ECG columns get the same treatment, keyed off
                    -- excluded.ecg_json independently: the two modalities are
                    -- saved at different times, so a save that carries a fresh
                    -- ECG and no echo must update one group and leave the
                    -- other alone. ecg_positive_count is guarded by the JSON
                    -- being NULL rather than by the count being zero, because
                    -- zero positives is a real finding that must survive a
                    -- later metadata-only save.
                    ecg_json            = COALESCE(
                        excluded.ecg_json, cases.ecg_json
                    ),
                    ecg_analyzed        = MAX(
                        excluded.ecg_analyzed, cases.ecg_analyzed
                    ),
                    ecg_positive_count  = CASE
                        WHEN excluded.ecg_json IS NULL
                        THEN cases.ecg_positive_count
                        ELSE excluded.ecg_positive_count
                    END,
                    ecg_filename        = CASE
                        WHEN excluded.ecg_json IS NULL
                        THEN cases.ecg_filename
                        ELSE excluded.ecg_filename
                    END,

                    -- And CCTA, keyed off its own JSON. Zero lumen voxels is a
                    -- real result here for the same reason zero positive ECG
                    -- classes is: the model ran and found nothing above
                    -- threshold, which the guard must not confuse with "this
                    -- save did not carry a CT".
                    ccta_json           = COALESCE(
                        excluded.ccta_json, cases.ccta_json
                    ),
                    ccta_analyzed       = MAX(
                        excluded.ccta_analyzed, cases.ccta_analyzed
                    ),
                    ccta_lumen_voxels   = CASE
                        WHEN excluded.ccta_json IS NULL
                        THEN cases.ccta_lumen_voxels
                        ELSE excluded.ccta_lumen_voxels
                    END,
                    ccta_filename       = CASE
                        WHEN excluded.ccta_json IS NULL
                        THEN cases.ccta_filename
                        ELSE excluded.ccta_filename
                    END
                """,
                row,
            )

            self._replace_messages_locked(
                connection, case_id, payload.get("conversation") or []
            )

            connection.commit()

        stored = self.get(case_id)

        if stored is None:                                # pragma: no cover
            raise CaseStoreError(f"Case {case_id} vanished immediately after save.")

        return stored

    def _replace_messages_locked(
        self,
        connection: sqlite3.Connection,
        case_id: str,
        conversation: list[dict[str, Any]],
    ) -> None:
        """
        Rewrite the transcript wholesale.

        The frontend owns the conversation and sends the full list, so
        replacing is correct and avoids duplicating every message on each
        save. Caller must hold the lock.
        """
        connection.execute(
            "DELETE FROM case_messages WHERE case_id = ?", (case_id,)
        )

        if not conversation:
            return

        timestamp = _now()

        connection.executemany(
            """
            INSERT INTO case_messages
                (case_id, role, text, model, device, created_at)
            VALUES (?,?,?,?,?,?)
            """,
            [
                (
                    case_id,
                    _text(message.get("role")) or "user",
                    _text(message.get("text")),
                    _text(message.get("model")),
                    _text(message.get("device")),
                    timestamp,
                )
                for message in conversation
                if _text(message.get("text"))
            ],
        )

    def save_report(self, case_id: str, report: dict[str, Any]) -> None:
        """
        Store the generated report against an existing case.

        Separate from :meth:`save` on purpose. A report is assembled from model
        results the server already has, not typed in the browser, so letting a
        routine case save carry a ``report`` field would give the frontend a way
        to overwrite the generated document with whatever it happened to be
        holding. Keeping the writer separate also means ``report_json`` is absent
        from the ON CONFLICT list in :meth:`save`, so a demographics edit cannot
        erase a report.

        Only the last report is kept. Reports are derived — regenerating one from
        the stored analyses reproduces every structured field, and the narrative
        is the only part that varies between runs — so a history would grow the
        database without recording anything that cannot be recomputed.

        Raises :class:`CaseStoreError` when the case does not exist, rather than
        creating a bare row: a report about a case that was never saved has no
        patient attached to it.
        """
        connection = self._require()
        identifier = _text(case_id)

        payload = json.dumps(report)
        timestamp = _now()

        with self._lock:
            cursor = connection.execute(
                """
                UPDATE cases
                SET report_json = ?, updated_at = ?
                WHERE case_id = ?
                """,
                (payload, timestamp, identifier),
            )
            connection.commit()

        if cursor.rowcount == 0:
            raise CaseStoreError(
                f"No case {identifier!r} is saved, so its report could not be "
                "stored. Save the case first."
            )

    # ---- read -----------------------------------------------------

    def get(self, case_id: str) -> Optional[dict[str, Any]]:
        connection = self._require()

        with self._lock:
            row = connection.execute(
                "SELECT * FROM cases WHERE case_id = ?", (_text(case_id),)
            ).fetchone()

            if row is None:
                return None

            messages = connection.execute(
                """
                SELECT role, text, model, device
                FROM case_messages
                WHERE case_id = ?
                ORDER BY id
                """,
                (_text(case_id),),
            ).fetchall()

        return self._hydrate(row, messages)

    def list(
        self,
        search: str = "",
        limit: int = 200,
    ) -> list[dict[str, Any]]:
        """Case summaries, most recently updated first."""
        connection = self._require()

        # Explicit columns, not SELECT *: echo_json holds the full segmentation
        # payload including a 65k-element mask array, ecg_json holds five
        # probabilities plus twelve lead attributions, and report_json holds a
        # whole generated document. Pulling those off disk for every row of a
        # list view nobody scrolls would cost megabytes per keystroke once the
        # search box is wired up. report_json is reduced to a boolean in SQL for
        # the same reason — the sidebar needs to know a report exists, not read
        # it.
        query = (
            "SELECT case_id, patient_name, patient_mrn, sex, study_date,"
            " ccta_analyzed, ccta_lumen_voxels, ccta_filename,"
            " echo_analyzed, structures_found, echo_filename,"
            " ecg_analyzed, ecg_positive_count, ecg_filename,"
            " (report_json IS NOT NULL) AS has_report,"
            " created_at, updated_at, notes, rowid"
            " FROM cases"
        )
        params: list[Any] = []

        term = _text(search)
        if term:
            # Case ID is searchable too: it is what the operator sees in the
            # header chip and the most likely thing to be copied around.
            query += (
                " WHERE patient_name LIKE ? COLLATE NOCASE"
                " OR patient_mrn LIKE ? COLLATE NOCASE"
                " OR case_id LIKE ? COLLATE NOCASE"
                " OR notes LIKE ? COLLATE NOCASE"
            )
            like = f"%{term}%"
            params.extend([like, like, like, like])

        # rowid breaks ties deterministically. Two cases can still share a
        # timestamp if the clock is coarse or is stepped backwards by NTP,
        # and an unstable sort would shuffle the sidebar between refreshes.
        query += " ORDER BY updated_at DESC, rowid DESC LIMIT ?"
        params.append(max(1, min(int(limit), 1000)))

        with self._lock:
            rows = connection.execute(query, params).fetchall()

        return [
            CaseSummary(
                case_id=row["case_id"],
                patient_name=row["patient_name"],
                patient_mrn=row["patient_mrn"],
                sex=row["sex"],
                study_date=row["study_date"],
                ccta_analyzed=bool(row["ccta_analyzed"]),
                ccta_lumen_voxels=row["ccta_lumen_voxels"],
                ccta_filename=row["ccta_filename"],
                echo_analyzed=bool(row["echo_analyzed"]),
                structures_found=row["structures_found"],
                echo_filename=row["echo_filename"],
                ecg_analyzed=bool(row["ecg_analyzed"]),
                ecg_positive_count=row["ecg_positive_count"],
                ecg_filename=row["ecg_filename"],
                has_report=bool(row["has_report"]),
                created_at=row["created_at"],
                updated_at=row["updated_at"],
            ).to_dict()
            for row in rows
        ]

    def count(self) -> int:
        connection = self._require()

        with self._lock:
            return connection.execute(
                "SELECT COUNT(*) AS total FROM cases"
            ).fetchone()["total"]

    def delete(self, case_id: str) -> bool:
        """Remove a case and its stored images."""
        connection = self._require()
        identifier = _text(case_id)

        with self._lock:
            cursor = connection.execute(
                "DELETE FROM cases WHERE case_id = ?", (identifier,)
            )
            connection.commit()
            removed = cursor.rowcount > 0

        if removed:
            directory = self.case_dir(identifier)

            # Confined to the configured files directory, so a hostile
            # case_id cannot delete anything outside it.
            try:
                resolved = directory.resolve()
                if (
                    resolved.is_dir()
                    and self._files_dir.resolve() in resolved.parents
                ):
                    shutil.rmtree(resolved)
            except OSError:
                # The row is already gone; leftover files are not worth
                # failing the request over.
                pass

        return removed

    # ---- hydration ------------------------------------------------

    def _hydrate(
        self,
        row: sqlite3.Row,
        messages: list[sqlite3.Row],
    ) -> dict[str, Any]:
        """Turn database rows back into the shape the frontend sent."""

        def parse(value: Optional[str], fallback: Any) -> Any:
            if not value:
                return fallback
            try:
                return json.loads(value)
            except (ValueError, TypeError):
                return fallback

        files = parse(row["files_json"], {})
        ccta = parse(row["ccta_json"], None)
        echo = parse(row["echo_json"], None)
        ecg = parse(row["ecg_json"], None)
        report = parse(row["report_json"], None)

        # Rebuild the figure URLs as backend endpoints rather than inlining
        # megabytes of base64 into the case payload. The browser then caches
        # them like any other image.
        #
        # Split by modality on the way out even though they share one directory
        # on disk: the echo viewer, the ECG viewer and the CCTA viewer each want
        # their own set, and handing them a mixed dict to filter is how a stale
        # ECG strip ends up rendered above an echo overlay.
        images = {
            key: f"/api/cases/{row['case_id']}/images/{key}"
            for key in _ECHO_IMAGE_KEYS
            if key in files
        }

        ecg_figures = {
            short: f"/api/cases/{row['case_id']}/images/{full}"
            for short, full in _ECG_FIGURE_ALIASES.items()
            if full in files
        }

        ccta_figures = {
            short: f"/api/cases/{row['case_id']}/images/{full}"
            for short, full in _CCTA_FIGURE_ALIASES.items()
            if full in files
        }

        date_of_birth = row["date_of_birth"]

        return {
            "case_id": row["case_id"],
            "patient": {
                "name": row["patient_name"],
                "mrn": row["patient_mrn"],
                "dateOfBirth": date_of_birth,
                "sex": row["sex"],
                "studyDate": row["study_date"],
                "referringClinician": row["referring_clinician"],
                "notes": row["notes"],
                # Derived on read so it is never stale.
                "age": derive_age(date_of_birth),
            },
            "clinical": parse(row["clinical_json"], {}),
            "ccta": ccta,
            "echo": echo,
            "ecg": ecg,
            "report": report,
            "images": images,
            "ecg_figures": ecg_figures,
            "ccta_figures": ccta_figures,
            "source_file": files.get("source", ""),
            "conversation": [
                {
                    "role": message["role"],
                    "text": message["text"],
                    "model": message["model"] or None,
                    "device": message["device"] or None,
                    # Restored transcripts are historical, so the "show case
                    # context" toggle has nothing to reveal.
                    "contextUsed": False,
                }
                for message in messages
            ],
            "echo_analyzed": bool(row["echo_analyzed"]),
            "structures_found": row["structures_found"],
            "ecg_analyzed": bool(row["ecg_analyzed"]),
            "ecg_positive_count": row["ecg_positive_count"],
            "ccta_analyzed": bool(row["ccta_analyzed"]),
            "ccta_lumen_voxels": row["ccta_lumen_voxels"],
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
        }


store = CaseStore()
