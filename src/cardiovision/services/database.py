"""
CardioVision AI — local case store.

Persists patient cases to a SQLite file next to the project. Nothing is
uploaded anywhere; the database and its image files never leave this machine.

LAYOUT
------
    data/cardiovision.db          patient records, findings, Q&A transcripts
    data/cases/<case_id>/         rendered PNGs + the original upload

Images are files on disk rather than blobs in the database. A single analysis
produces six PNGs plus the source frame, which is a few megabytes; putting
that in a row would make the case list slow to query and the database
awkward to back up.

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

from config import CASE_DB_PATH, CASE_FILES_DIR, DATA_DIR


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
    echo_json           TEXT,
    files_json          TEXT NOT NULL DEFAULT '{}',

    -- denormalised for the case list, so listing never parses JSON
    echo_analyzed       INTEGER NOT NULL DEFAULT 0,
    structures_found    INTEGER NOT NULL DEFAULT 0,
    echo_filename       TEXT NOT NULL DEFAULT '',

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


# The keys of the images dict returned by rendering.render_analysis_images,
# each written as <name>.png inside the case directory.
_IMAGE_KEYS = (
    "original",
    "mask",
    "overlay",
    "saliency",
    "saliency_overlay",
    "combined",
)


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
    echo_analyzed: bool
    structures_found: int
    echo_filename: str
    created_at: str
    updated_at: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "patient_name": self.patient_name,
            "patient_mrn": self.patient_mrn,
            "sex": self.sex,
            "study_date": self.study_date,
            "echo_analyzed": self.echo_analyzed,
            "structures_found": self.structures_found,
            "echo_filename": self.echo_filename,
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
        connection.commit()

        self._connection = connection
        self._write_gitignore()

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
        Write the rendered data-URL PNGs out as files.

        Returns a name -> filename map for files_json. Anything that is not
        a decodable data URL is skipped rather than written as garbage.
        """
        if not images:
            return {}

        directory = self.case_dir(case_id)
        directory.mkdir(parents=True, exist_ok=True)

        stored: dict[str, str] = {}

        for key in _IMAGE_KEYS:
            payload = _decode_data_url(images.get(key, ""))

            if payload is None:
                continue

            filename = f"{key}.png"
            (directory / filename).write_bytes(payload)
            stored[key] = filename

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
        """Read one stored PNG back. Returns None when it does not exist."""
        if name not in _IMAGE_KEYS:
            return None

        path = self.case_dir(case_id) / f"{name}.png"

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
        echo = payload.get("echo")
        images = payload.get("images")

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
                json.dumps(echo) if echo else None,
                json.dumps(files),
                1 if echo_analyzed else 0,
                structures_found,
                echo_filename,
                created_at,
                timestamp,
            )

            connection.execute(
                """
                INSERT INTO cases (
                    case_id, patient_name, patient_mrn, date_of_birth, sex,
                    study_date, referring_clinician, notes,
                    clinical_json, echo_json, files_json,
                    echo_analyzed, structures_found, echo_filename,
                    created_at, updated_at
                ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
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
        # payload including a 65k-element mask array, and pulling that off disk
        # for every row of a list view nobody scrolls would cost megabytes per
        # keystroke once the search box is wired up.
        query = (
            "SELECT case_id, patient_name, patient_mrn, sex, study_date,"
            " echo_analyzed, structures_found, echo_filename,"
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
                echo_analyzed=bool(row["echo_analyzed"]),
                structures_found=row["structures_found"],
                echo_filename=row["echo_filename"],
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
        echo = parse(row["echo_json"], None)

        # Rebuild the image URLs as backend endpoints rather than inlining
        # megabytes of base64 into the case payload. The browser then caches
        # them like any other image.
        images = {
            key: f"/api/cases/{row['case_id']}/images/{key}"
            for key in _IMAGE_KEYS
            if key in files
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
            "echo": echo,
            "images": images,
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
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
        }


store = CaseStore()
