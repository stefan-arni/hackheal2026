"""Where speech data goes: recorded sections and per-visit results.

Everything goes through two small interfaces, so the destination can change
later without touching the analysis:

  RecordingSink.save(wav_bytes, meta) -> where it was stored
  VisitStore.previous(patient_id) / VisitStore.save(patient_id, summary)

Implementations:
  Supabase (now):  sections -> Storage bucket + a row in `speech_recordings`;
                   visits -> rows in `speech_visits`. Schema: supabase_schema.sql
  Local (fallback / offline): files under ./speech_data/

make_storage() picks Supabase when SUPABASE_URL and SUPABASE_SERVICE_KEY are set
(in the environment or a .env file, loaded by speech_server.py via env.py;
server-side key, never ship it in the iPhone app), otherwise local files.
"""

from __future__ import annotations

import json
import os
import re
import time
import uuid
from pathlib import Path
from typing import Protocol

DATA_DIR = Path(__file__).parent / "speech_data"


class RecordingSink(Protocol):
    def save(self, wav: bytes, meta: dict) -> dict: ...


class VisitStore(Protocol):
    def history(self, patient_id: str, before_visit_id: str | None = None,
                limit: int = 200) -> list[dict]: ...      # prior visits, newest first
    def previous(self, patient_id: str, before_visit_id: str | None = None) -> dict | None: ...
    def save(self, patient_id: str, visit_id: str, summary: dict) -> dict: ...


def _safe(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", s or "unknown")[:80]


def _recording_path(meta: dict) -> str:
    stamp = time.strftime("%Y%m%d-%H%M%S", time.gmtime(meta.get("started_at", time.time())))
    return (f"{_safe(meta.get('patient_id'))}/{_safe(meta.get('visit_id'))}/"
            f"{stamp}_{_safe(meta.get('label', 'section'))}_{uuid.uuid4().hex[:6]}.wav")


# --------------------------------------------------------------------------- #
# Local files
# --------------------------------------------------------------------------- #

class LocalRecordingSink:
    def __init__(self, root: Path = DATA_DIR / "recordings"):
        self.root = Path(root)

    def save(self, wav: bytes, meta: dict) -> dict:
        path = self.root / _recording_path(meta)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(wav)
        path.with_suffix(".json").write_text(json.dumps(meta, indent=2, default=str))
        return {"backend": "local", "path": str(path)}


class LocalVisitStore:
    def __init__(self, path: Path = DATA_DIR / "visits.json"):
        self.path = Path(path)

    def _load(self) -> list[dict]:
        if self.path.exists():
            return json.loads(self.path.read_text())
        return []

    def history(self, patient_id, before_visit_id=None, limit=200):
        rows = [r for r in self._load() if r["patient_id"] == patient_id and r["visit_id"] != before_visit_id]
        return sorted(rows, key=lambda r: r["created_at"], reverse=True)[:limit]

    def previous(self, patient_id, before_visit_id=None):
        rows = self.history(patient_id, before_visit_id, limit=1)
        return rows[0] if rows else None

    def save(self, patient_id, visit_id, summary):
        rows = [r for r in self._load() if r["visit_id"] != visit_id]
        row = {"patient_id": patient_id, "visit_id": visit_id, "created_at": time.time(),
               "summary": summary}
        rows.append(row)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(rows, indent=1, default=str))
        return {"backend": "local", "path": str(self.path)}


# --------------------------------------------------------------------------- #
# Supabase
# --------------------------------------------------------------------------- #

class SupabaseRecordingSink:
    """Uploads the WAV to a Storage bucket and records it in `speech_recordings`."""

    def __init__(self, client, bucket: str = "speech-recordings", table: str = "speech_recordings"):
        self.client, self.bucket, self.table = client, bucket, table

    def save(self, wav: bytes, meta: dict) -> dict:
        path = _recording_path(meta)
        self.client.storage.from_(self.bucket).upload(
            path, wav, {"content-type": "audio/wav", "upsert": "false"})
        row = {
            "patient_id": meta.get("patient_id"),
            "visit_id": meta.get("visit_id"),
            "label": meta.get("label"),
            "storage_path": f"{self.bucket}/{path}",
            "duration_s": meta.get("duration_s"),
            "started_offset_s": meta.get("started_offset_s"),
            "metrics": meta.get("metrics"),
        }
        res = self.client.table(self.table).insert(row).execute()
        rid = (res.data or [{}])[0].get("id") if getattr(res, "data", None) else None
        return {"backend": "supabase", "bucket": self.bucket, "path": path, "row_id": rid}


class SupabaseVisitStore:
    def __init__(self, client, table: str = "speech_visits"):
        self.client, self.table = client, table

    def history(self, patient_id, before_visit_id=None, limit=200):
        q = (self.client.table(self.table).select("*").eq("patient_id", patient_id)
             .order("created_at", desc=True).limit(limit + 1))
        rows = q.execute().data or []
        return [r for r in rows if r.get("visit_id") != before_visit_id][:limit]

    def previous(self, patient_id, before_visit_id=None):
        rows = self.history(patient_id, before_visit_id, limit=1)
        return rows[0] if rows else None

    def save(self, patient_id, visit_id, summary):
        row = {"patient_id": patient_id, "visit_id": visit_id, "summary": summary}
        res = self.client.table(self.table).upsert(row, on_conflict="visit_id").execute()
        rid = (res.data or [{}])[0].get("id") if getattr(res, "data", None) else None
        return {"backend": "supabase", "table": self.table, "row_id": rid}


def make_storage(prefer: str = "auto") -> tuple[RecordingSink, VisitStore, str]:
    """('auto' | 'supabase' | 'local') -> (recording sink, visit store, description)."""
    url, key = os.environ.get("SUPABASE_URL"), os.environ.get("SUPABASE_SERVICE_KEY")
    if prefer == "supabase" or (prefer == "auto" and url and key):
        if not (url and key):
            raise SystemExit("Set SUPABASE_URL and SUPABASE_SERVICE_KEY to use Supabase.")
        from supabase import create_client
        client = create_client(url, key)
        bucket = os.environ.get("SUPABASE_SPEECH_BUCKET", "speech-recordings")
        return (SupabaseRecordingSink(client, bucket), SupabaseVisitStore(client),
                f"supabase ({url}, bucket '{bucket}')")
    return LocalRecordingSink(), LocalVisitStore(), f"local files ({DATA_DIR})"
