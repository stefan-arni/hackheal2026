"""HTTP-level tests for the AT voice-extension endpoints (Phase 3).

Sprint: Voice symptom entry — AT extension.
Plan: docs/superpowers/plans/2026-05-05-sprint-voice-at-extension.md

Three endpoints exercised here:
    POST   /api/teams/{team_id}/voice/observation/extract
    POST   /api/symptoms/voice/extract  (extended with optional
                                         target_user_id + team_id)
    POST   /api/symptoms/scribe-log

Phase 3 locked decisions (re-stated here so the tests document the
contract):

  - Scribe gate = active ``at_symptom_scribe`` consent on the SPECIFIC
    team_id passed in the payload (not "any team"). Multi-team-coverage
    AT must pick the team they're scribing under.
  - Observation gate = ``team_share_consents`` is_team_share_active
    (any scope grants visibility — observations are gated by consent
    *existence*, not by a specific scope).
  - Scribe role = AT, head AT, OR coach (any team_memberships role).
    Widens R7 risk vs the original "AT only" plan; mitigated by the
    affirmative consent + review-then-submit flow.
  - Audio retention = match athlete /extract: discard blob on
    extraction success; preserve on extraction error.
  - Scribe-log dedupe = none (mirrors athlete /api/symptoms/ POST).
  - Scribe-log change-detection hook = match log_symptom (safe_record +
    note_service.mark_stale_for_athlete). The plan's earlier reference
    to check_z_drop_for_metric + mark_today_stale was a planning slip;
    those are connector-ingest hooks, not symptom-ingest hooks.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timedelta
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from api.ai import inference as inference_module
from api.ai.inference import MockInferenceClient
from api.auth import create_jwt
from api.database import get_connection, run_migrations


# ---------------------------------------------------------------------------
# Fixtures + helpers (modeled on test_voice_symptoms_router.py)
# ---------------------------------------------------------------------------


@pytest.fixture
def client(tmp_db, monkeypatch, tmp_path):
    monkeypatch.setenv("DB_PATH", tmp_db)
    monkeypatch.setenv("VOICE_AUDIO_TMP_DIR", str(tmp_path / "voice"))
    from api.main import create_app

    app = create_app()
    with TestClient(app) as c:
        yield c


def _h(uid: str, role: str) -> dict:
    return {"Authorization": f"Bearer {create_jwt(uid, f'{uid}@e.com', role)}"}


def _future(days: int = 365) -> str:
    return (datetime.utcnow() + timedelta(days=days)).strftime("%Y-%m-%d %H:%M:%S")


def _seed(client) -> dict:
    """Common seed: 2 athletes, 1 doc, 2 ATs, 1 head AT, 1 coach, 1 org."""
    with get_connection() as db:
        db.executescript(
            """
            INSERT INTO users (id, email, role, display_name) VALUES
              ('ath1', 'ath1@e.com', 'athlete', 'Alice'),
              ('ath2', 'ath2@e.com', 'athlete', 'Bob'),
              ('doc1', 'doc1@e.com', 'clinician', 'Doctor'),
              ('at1',  'at1@e.com',  'athletic_trainer', 'AT One'),
              ('at2',  'at2@e.com',  'athletic_trainer', 'AT Two'),
              ('hat1', 'hat1@e.com', 'athletic_trainer', 'Head AT'),
              ('coach1', 'coach1@e.com', 'athletic_trainer', 'Coach One');

            INSERT OR IGNORE INTO organization_memberships
              (organization_id, user_id, role) VALUES
              ('org_doc1', 'at1', 'athletic_trainer'),
              ('org_doc1', 'at2', 'athletic_trainer'),
              ('org_doc1', 'hat1', 'athletic_trainer'),
              ('org_doc1', 'coach1', 'athletic_trainer');

            INSERT INTO clinical_links
              (id, athlete_id, clinician_id, status) VALUES
              ('cl_ath1_doc1', 'ath1', 'doc1', 'active'),
              ('cl_ath2_doc1', 'ath2', 'doc1', 'active');
            """
        )
        db.commit()
    return {
        "ath1": "ath1", "ath2": "ath2",
        "doc1": "doc1",
        "at1": "at1", "at2": "at2",
        "hat1": "hat1",
        "coach1": "coach1",
        "org": "org_doc1",
    }


def _make_team(client, s, *, members: list[tuple[str, str]] | None = None,
               athletes: list[str] | None = None) -> dict:
    """Create a team and (optionally) seed AT/coach memberships and athlete roster.

    members: list of (user_id, role) where role ∈
             {athletic_trainer, head_athletic_trainer, coach}.
    """
    h_doc = _h(s["doc1"], "clinician")
    t = client.post("/api/teams",
                    json={"org_id": s["org"], "name": "Lacrosse"},
                    headers=h_doc).json()
    for uid, role in (members or []):
        client.post(
            f"/api/teams/{t['id']}/members",
            json={"user_id": uid, "role": role},
            headers=h_doc,
        )
    # Use the first AT-tier member to add athletes (any team-membership role
    # can call add_athlete in the existing service).
    member_ids = [m[0] for m in (members or [])]
    if athletes and member_ids:
        h_member = _h(member_ids[0], "athletic_trainer")
        for ath in athletes:
            client.post(
                f"/api/teams/{t['id']}/athletes",
                json={"athlete_user_id": ath},
                headers=h_member,
            )
    return t


def _grant_consent(client, *, athlete: str, team_id: int, scopes: list[str]) -> dict:
    """Athlete grants team-share consent on (team, scopes)."""
    h_ath = _h(athlete, "athlete")
    r = client.post(
        f"/api/teams/{team_id}/share-consent",
        json={"expires_at": _future(), "scopes": scopes},
        headers=h_ath,
    )
    assert r.status_code == 200, r.text
    return r.json()


def _stub_inference(monkeypatch, *, canned_response: str):
    """Pin every callsite that fetches an inference client to a Mock with
    the canned generate() response."""
    mock = MockInferenceClient(canned_response=canned_response)
    monkeypatch.setattr(inference_module, "get_inference_client", lambda: mock)
    from api.ai import voice_extraction as ve
    monkeypatch.setattr(ve, "get_inference_client", lambda: mock)
    from api.routers import voice_symptoms as vs
    monkeypatch.setattr(vs, "get_inference_client", lambda: mock)
    return mock


def _make_audio() -> bytes:
    return b"\x1aE\xdf\xa3" + (b"\x00" * 4092)


def _do_upload(client, user_id: str, role: str = "athletic_trainer") -> int:
    """Upload a clip as a given user; return upload_id.

    Note: uploads are JWT-scoped by user_id; the AT uploads under their
    own id, even when the eventual scribe-log targets an athlete.
    """
    resp = client.post(
        "/api/symptoms/voice/upload",
        files={"file": ("clip.webm", _make_audio(), "audio/webm")},
        data={"mime_type": "audio/webm"},
        headers=_h(user_id, role),
    )
    assert resp.status_code == 201, resp.text
    return resp.json()["upload_id"]


# Canonical canned responses for the inference Mock
_OBS_HAPPY = json.dumps({
    "observation_text": "the athlete looked wobbly after a hit and reported sensitivity to gym lights.",
    "observed_signs": ["wobbly", "light-sensitivity"],
})
_SYMPTOM_HAPPY = json.dumps({
    "symptoms": [
        {"symptom": "headache", "severity": 7, "notes": "since this morning"},
        {"symptom": "light_sensitivity", "severity": 5, "notes": None},
    ],
})


# ===========================================================================
# /api/teams/{team_id}/voice/observation/extract
# ===========================================================================


class TestObservationExtract:
    def _setup(self, client):
        s = _seed(client)
        t = _make_team(
            client, s,
            members=[(s["at1"], "athletic_trainer")],
            athletes=[s["ath1"]],
        )
        _grant_consent(client, athlete=s["ath1"], team_id=t["id"],
                       scopes=["readiness_signal"])
        return s, t

    def test_requires_auth(self, client):
        s, t = self._setup(client)
        r = client.post(
            f"/api/teams/{t['id']}/voice/observation/extract",
            json={"transcript": "x", "athlete_user_id": s["ath1"]},
        )
        assert r.status_code == 401

    def test_athlete_role_denied(self, client, monkeypatch):
        _stub_inference(monkeypatch, canned_response=_OBS_HAPPY)
        s, t = self._setup(client)
        r = client.post(
            f"/api/teams/{t['id']}/voice/observation/extract",
            json={"transcript": "x", "athlete_user_id": s["ath1"]},
            headers=_h(s["ath1"], "athlete"),
        )
        assert r.status_code == 403

    def test_clinician_role_denied(self, client, monkeypatch):
        _stub_inference(monkeypatch, canned_response=_OBS_HAPPY)
        s, t = self._setup(client)
        r = client.post(
            f"/api/teams/{t['id']}/voice/observation/extract",
            json={"transcript": "x", "athlete_user_id": s["ath1"]},
            headers=_h(s["doc1"], "clinician"),
        )
        assert r.status_code == 403

    def test_at_not_on_team_denied(self, client, monkeypatch):
        _stub_inference(monkeypatch, canned_response=_OBS_HAPPY)
        s, t = self._setup(client)
        # at2 is in the org but NOT on the team
        r = client.post(
            f"/api/teams/{t['id']}/voice/observation/extract",
            json={"transcript": "x", "athlete_user_id": s["ath1"]},
            headers=_h(s["at2"], "athletic_trainer"),
        )
        assert r.status_code == 403

    def test_athlete_not_on_active_roster_denied(self, client, monkeypatch):
        _stub_inference(monkeypatch, canned_response=_OBS_HAPPY)
        s, t = self._setup(client)
        # ath2 exists but isn't on the team's roster
        r = client.post(
            f"/api/teams/{t['id']}/voice/observation/extract",
            json={"transcript": "x", "athlete_user_id": s["ath2"]},
            headers=_h(s["at1"], "athletic_trainer"),
        )
        assert r.status_code == 403

    def test_no_team_share_consent_denied(self, client, monkeypatch):
        _stub_inference(monkeypatch, canned_response=_OBS_HAPPY)
        s = _seed(client)
        t = _make_team(client, s,
                       members=[(s["at1"], "athletic_trainer")],
                       athletes=[s["ath1"]])
        # Intentionally skip _grant_consent
        r = client.post(
            f"/api/teams/{t['id']}/voice/observation/extract",
            json={"transcript": "x", "athlete_user_id": s["ath1"]},
            headers=_h(s["at1"], "athletic_trainer"),
        )
        assert r.status_code == 403

    def test_coach_can_extract_observation(self, client, monkeypatch):
        """Coach role is permitted on observation extract (matches the
        existing add_observation gate which accepts any
        VALID_MEMBERSHIP_ROLES)."""
        _stub_inference(monkeypatch, canned_response=_OBS_HAPPY)
        s = _seed(client)
        t = _make_team(client, s,
                       members=[(s["coach1"], "coach"),
                                (s["at1"], "athletic_trainer")],
                       athletes=[s["ath1"]])
        _grant_consent(client, athlete=s["ath1"], team_id=t["id"],
                       scopes=["readiness_signal"])
        r = client.post(
            f"/api/teams/{t['id']}/voice/observation/extract",
            json={"transcript": "athlete looked wobbly",
                  "athlete_user_id": s["ath1"]},
            headers=_h(s["coach1"], "athletic_trainer"),
        )
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["observation_text"]
        assert "wobbly" in body["observed_signs"]

    def test_happy_path_returns_observation_and_signs(self, client, monkeypatch):
        _stub_inference(monkeypatch, canned_response=_OBS_HAPPY)
        s, t = self._setup(client)
        r = client.post(
            f"/api/teams/{t['id']}/voice/observation/extract",
            json={"transcript": "athlete looked wobbly after a hit",
                  "athlete_user_id": s["ath1"]},
            headers=_h(s["at1"], "athletic_trainer"),
        )
        assert r.status_code == 200, r.text
        body = r.json()
        assert "observation_text" in body
        assert "observed_signs" in body
        assert body["error"] is None
        assert isinstance(body["observed_signs"], list)
        assert "wobbly" in body["observed_signs"]

    def test_extra_field_forbidden(self, client, monkeypatch):
        _stub_inference(monkeypatch, canned_response=_OBS_HAPPY)
        s, t = self._setup(client)
        r = client.post(
            f"/api/teams/{t['id']}/voice/observation/extract",
            json={"transcript": "x", "athlete_user_id": s["ath1"], "evil": True},
            headers=_h(s["at1"], "athletic_trainer"),
        )
        assert r.status_code == 422

    def test_route_order_observation_extract_not_caught_by_team_id_route(
        self, client, monkeypatch
    ):
        """Route declared before /{team_id} catchall. Pinning here so a
        future refactor that reorders the router doesn't silently break
        with `team_id="voice"` 4xx errors (the trap that bit the
        symptom-feed and patient-invite sprints)."""
        _stub_inference(monkeypatch, canned_response=_OBS_HAPPY)
        s, t = self._setup(client)
        r = client.post(
            f"/api/teams/{t['id']}/voice/observation/extract",
            json={"transcript": "x", "athlete_user_id": s["ath1"]},
            headers=_h(s["at1"], "athletic_trainer"),
        )
        # If the route isn't matched, FastAPI returns 404 / 405 — anything
        # other than 200 here would mean the route is being shadowed.
        assert r.status_code == 200, r.text

    def test_success_deletes_blob_and_sets_deleted_at(self, client, monkeypatch):
        _stub_inference(monkeypatch, canned_response=_OBS_HAPPY)
        s, t = self._setup(client)
        upload_id = _do_upload(client, s["at1"], role="athletic_trainer")
        blob_path = Path(os.environ["VOICE_AUDIO_TMP_DIR"]) / f"{upload_id}.bin"
        assert blob_path.exists()  # sanity

        r = client.post(
            f"/api/teams/{t['id']}/voice/observation/extract",
            json={"transcript": "athlete looked wobbly",
                  "athlete_user_id": s["ath1"],
                  "upload_id": upload_id},
            headers=_h(s["at1"], "athletic_trainer"),
        )
        assert r.status_code == 200, r.text
        assert r.json()["error"] is None

        # Audio retention: blob deleted on success
        assert not blob_path.exists()
        with get_connection() as db:
            row = db.execute(
                "SELECT deleted_at, extraction_error FROM audio_uploads WHERE id=?",
                (upload_id,),
            ).fetchone()
        assert row["deleted_at"] is not None
        assert row["extraction_error"] is None

    def test_extraction_error_preserves_blob(self, client, monkeypatch):
        # Force a schema error
        _stub_inference(monkeypatch, canned_response="not json at all")
        s, t = self._setup(client)
        upload_id = _do_upload(client, s["at1"], role="athletic_trainer")
        blob_path = Path(os.environ["VOICE_AUDIO_TMP_DIR"]) / f"{upload_id}.bin"

        r = client.post(
            f"/api/teams/{t['id']}/voice/observation/extract",
            json={"transcript": "anything",
                  "athlete_user_id": s["ath1"],
                  "upload_id": upload_id},
            headers=_h(s["at1"], "athletic_trainer"),
        )
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["error"] == "schema_invalid_json"
        assert body["observation_text"] == ""
        assert body["observed_signs"] == []

        # Blob preserved per locked retention decision
        assert blob_path.exists()
        with get_connection() as db:
            row = db.execute(
                "SELECT deleted_at, extraction_error FROM audio_uploads WHERE id=?",
                (upload_id,),
            ).fetchone()
        assert row["deleted_at"] is None
        assert row["extraction_error"] == "schema_invalid_json"


# ===========================================================================
# /api/symptoms/voice/extract — extended with target_user_id + team_id
# ===========================================================================


class TestScribeExtract:
    def _setup_with_scribe_consent(self, client, *, scribe_role: str = "athletic_trainer"):
        """Seed a team with the AT and an athlete who's granted
        at_symptom_scribe consent on that team."""
        s = _seed(client)
        # Pick which user plays the team-member role
        scribe_user = (
            s["coach1"] if scribe_role == "coach"
            else s["hat1"] if scribe_role == "head_athletic_trainer"
            else s["at1"]
        )
        membership_role = (
            "coach" if scribe_role == "coach"
            else "head_athletic_trainer" if scribe_role == "head_athletic_trainer"
            else "athletic_trainer"
        )
        t = _make_team(
            client, s,
            members=[(scribe_user, membership_role)],
            athletes=[s["ath1"]],
        )
        _grant_consent(client, athlete=s["ath1"], team_id=t["id"],
                       scopes=["readiness_signal", "at_symptom_scribe"])
        return s, t, scribe_user

    def test_legacy_path_unchanged_when_no_target(self, client, monkeypatch):
        """Existing athlete-self path: no target_user_id → behaves as
        today regardless of team membership / consent."""
        _stub_inference(monkeypatch, canned_response=_SYMPTOM_HAPPY)
        s = _seed(client)
        r = client.post(
            "/api/symptoms/voice/extract",
            json={"transcript": "I have a headache"},
            headers=_h(s["ath1"], "athlete"),
        )
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["error"] is None
        assert len(body["symptoms"]) == 2

    def test_athlete_role_denied_in_scribe_mode(self, client, monkeypatch):
        _stub_inference(monkeypatch, canned_response=_SYMPTOM_HAPPY)
        s, t, _ = self._setup_with_scribe_consent(client)
        r = client.post(
            "/api/symptoms/voice/extract",
            json={"transcript": "x",
                  "target_user_id": s["ath1"],
                  "team_id": t["id"]},
            headers=_h(s["ath1"], "athlete"),
        )
        assert r.status_code == 403

    def test_clinician_role_denied_in_scribe_mode(self, client, monkeypatch):
        _stub_inference(monkeypatch, canned_response=_SYMPTOM_HAPPY)
        s, t, _ = self._setup_with_scribe_consent(client)
        r = client.post(
            "/api/symptoms/voice/extract",
            json={"transcript": "x",
                  "target_user_id": s["ath1"],
                  "team_id": t["id"]},
            headers=_h(s["doc1"], "clinician"),
        )
        assert r.status_code == 403

    def test_at_without_scribe_consent_denied(self, client, monkeypatch):
        _stub_inference(monkeypatch, canned_response=_SYMPTOM_HAPPY)
        s = _seed(client)
        t = _make_team(client, s,
                       members=[(s["at1"], "athletic_trainer")],
                       athletes=[s["ath1"]])
        # Grant only readiness_signal — no at_symptom_scribe
        _grant_consent(client, athlete=s["ath1"], team_id=t["id"],
                       scopes=["readiness_signal"])
        r = client.post(
            "/api/symptoms/voice/extract",
            json={"transcript": "x",
                  "target_user_id": s["ath1"],
                  "team_id": t["id"]},
            headers=_h(s["at1"], "athletic_trainer"),
        )
        assert r.status_code == 403

    def test_at_not_on_team_denied(self, client, monkeypatch):
        _stub_inference(monkeypatch, canned_response=_SYMPTOM_HAPPY)
        s, t, _ = self._setup_with_scribe_consent(client)
        # at2 is NOT a team member
        r = client.post(
            "/api/symptoms/voice/extract",
            json={"transcript": "x",
                  "target_user_id": s["ath1"],
                  "team_id": t["id"]},
            headers=_h(s["at2"], "athletic_trainer"),
        )
        assert r.status_code == 403

    def test_athlete_not_on_active_roster_denied(self, client, monkeypatch):
        _stub_inference(monkeypatch, canned_response=_SYMPTOM_HAPPY)
        s, t, _ = self._setup_with_scribe_consent(client)
        # ath2 is not on the roster (only ath1 is)
        r = client.post(
            "/api/symptoms/voice/extract",
            json={"transcript": "x",
                  "target_user_id": s["ath2"],
                  "team_id": t["id"]},
            headers=_h(s["at1"], "athletic_trainer"),
        )
        assert r.status_code == 403

    def test_team_id_required_when_target_user_id_set(self, client, monkeypatch):
        _stub_inference(monkeypatch, canned_response=_SYMPTOM_HAPPY)
        s, t, _ = self._setup_with_scribe_consent(client)
        r = client.post(
            "/api/symptoms/voice/extract",
            json={"transcript": "x",
                  "target_user_id": s["ath1"]},  # missing team_id
            headers=_h(s["at1"], "athletic_trainer"),
        )
        # 400 (validation) — team_id is required to ground the consent gate
        assert r.status_code == 400

    def test_happy_path_at(self, client, monkeypatch):
        _stub_inference(monkeypatch, canned_response=_SYMPTOM_HAPPY)
        s, t, _ = self._setup_with_scribe_consent(client)
        r = client.post(
            "/api/symptoms/voice/extract",
            json={"transcript": "she says her head hurts",
                  "target_user_id": s["ath1"],
                  "team_id": t["id"]},
            headers=_h(s["at1"], "athletic_trainer"),
        )
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["error"] is None
        assert len(body["symptoms"]) == 2
        # target_user_id + team_id pass through to response so frontend
        # can route Submit straight to /scribe-log
        assert body["target_user_id"] == s["ath1"]
        assert body["team_id"] == t["id"]

    def test_happy_path_coach(self, client, monkeypatch):
        """Coach role accepted (decision: AT/coach/head AT all scribe)."""
        _stub_inference(monkeypatch, canned_response=_SYMPTOM_HAPPY)
        s, t, scribe_uid = self._setup_with_scribe_consent(
            client, scribe_role="coach"
        )
        r = client.post(
            "/api/symptoms/voice/extract",
            json={"transcript": "she says her head hurts",
                  "target_user_id": s["ath1"],
                  "team_id": t["id"]},
            headers=_h(scribe_uid, "athletic_trainer"),
        )
        assert r.status_code == 200, r.text

    def test_cross_team_consent_does_not_unlock(self, client, monkeypatch):
        """Athlete granted scribe consent on team A but caller passes
        team B — must 403 even when consent exists somewhere else."""
        _stub_inference(monkeypatch, canned_response=_SYMPTOM_HAPPY)
        s = _seed(client)
        team_a = _make_team(client, s,
                            members=[(s["at1"], "athletic_trainer")],
                            athletes=[s["ath1"]])
        team_b = _make_team(client, s,
                            members=[(s["at1"], "athletic_trainer")],
                            athletes=[s["ath1"]])
        # Consent only on team A
        _grant_consent(client, athlete=s["ath1"], team_id=team_a["id"],
                       scopes=["at_symptom_scribe"])
        # Caller invokes scribe mode with team_b
        r = client.post(
            "/api/symptoms/voice/extract",
            json={"transcript": "x",
                  "target_user_id": s["ath1"],
                  "team_id": team_b["id"]},
            headers=_h(s["at1"], "athletic_trainer"),
        )
        assert r.status_code == 403


# ===========================================================================
# /api/symptoms/scribe-log
# ===========================================================================


class TestScribeLog:
    def _setup_with_scribe_consent(self, client):
        s = _seed(client)
        t = _make_team(
            client, s,
            members=[(s["at1"], "athletic_trainer")],
            athletes=[s["ath1"]],
        )
        _grant_consent(client, athlete=s["ath1"], team_id=t["id"],
                       scopes=["at_symptom_scribe"])
        return s, t

    def test_requires_auth(self, client):
        s, t = self._setup_with_scribe_consent(client)
        r = client.post(
            "/api/symptoms/scribe-log",
            json={"target_user_id": s["ath1"],
                  "team_id": t["id"],
                  "symptoms": [{"symptom": "headache", "severity": 6}]},
        )
        assert r.status_code == 401

    def test_athlete_role_denied(self, client):
        s, t = self._setup_with_scribe_consent(client)
        r = client.post(
            "/api/symptoms/scribe-log",
            json={"target_user_id": s["ath1"],
                  "team_id": t["id"],
                  "symptoms": [{"symptom": "headache", "severity": 6}]},
            headers=_h(s["ath1"], "athlete"),
        )
        assert r.status_code == 403

    def test_clinician_role_denied(self, client):
        s, t = self._setup_with_scribe_consent(client)
        r = client.post(
            "/api/symptoms/scribe-log",
            json={"target_user_id": s["ath1"],
                  "team_id": t["id"],
                  "symptoms": [{"symptom": "headache", "severity": 6}]},
            headers=_h(s["doc1"], "clinician"),
        )
        assert r.status_code == 403

    def test_no_consent_403(self, client):
        s = _seed(client)
        t = _make_team(client, s,
                       members=[(s["at1"], "athletic_trainer")],
                       athletes=[s["ath1"]])
        _grant_consent(client, athlete=s["ath1"], team_id=t["id"],
                       scopes=["readiness_signal"])  # no at_symptom_scribe
        r = client.post(
            "/api/symptoms/scribe-log",
            json={"target_user_id": s["ath1"],
                  "team_id": t["id"],
                  "symptoms": [{"symptom": "headache", "severity": 6}]},
            headers=_h(s["at1"], "athletic_trainer"),
        )
        assert r.status_code == 403

    def test_at_not_on_team_403(self, client):
        s, t = self._setup_with_scribe_consent(client)
        r = client.post(
            "/api/symptoms/scribe-log",
            json={"target_user_id": s["ath1"],
                  "team_id": t["id"],
                  "symptoms": [{"symptom": "headache", "severity": 6}]},
            headers=_h(s["at2"], "athletic_trainer"),  # at2 NOT on team
        )
        assert r.status_code == 403

    def test_athlete_not_on_active_roster_403(self, client):
        s, t = self._setup_with_scribe_consent(client)
        # ath2 isn't on the roster
        r = client.post(
            "/api/symptoms/scribe-log",
            json={"target_user_id": s["ath2"],
                  "team_id": t["id"],
                  "symptoms": [{"symptom": "headache", "severity": 6}]},
            headers=_h(s["at1"], "athletic_trainer"),
        )
        assert r.status_code == 403

    def test_happy_path_inserts_symptom_logs_with_logged_by(self, client):
        s, t = self._setup_with_scribe_consent(client)
        r = client.post(
            "/api/symptoms/scribe-log",
            json={"target_user_id": s["ath1"],
                  "team_id": t["id"],
                  "symptoms": [
                      {"symptom": "headache", "severity": 7, "notes": "since AM"},
                      {"symptom": "light_sensitivity", "severity": 5},
                  ]},
            headers=_h(s["at1"], "athletic_trainer"),
        )
        assert r.status_code == 201, r.text
        body = r.json()
        assert body["status"] == "logged"
        assert body["count"] == 2

        # Rows landed scoped to athlete with logged_by = AT
        with get_connection() as db:
            rows = db.execute(
                "SELECT symptom, severity, notes, user_id, logged_by_user_id "
                "FROM symptom_logs WHERE user_id = ? "
                "ORDER BY id ASC",
                (s["ath1"],),
            ).fetchall()
        assert len(rows) == 2
        assert rows[0]["symptom"] == "headache"
        assert rows[0]["severity"] == 7
        assert rows[0]["notes"] == "since AM"
        assert rows[0]["user_id"] == s["ath1"]
        assert rows[0]["logged_by_user_id"] == s["at1"]
        assert rows[1]["symptom"] == "light_sensitivity"
        assert rows[1]["logged_by_user_id"] == s["at1"]

    def test_invalid_symptom_400(self, client):
        s, t = self._setup_with_scribe_consent(client)
        r = client.post(
            "/api/symptoms/scribe-log",
            json={"target_user_id": s["ath1"],
                  "team_id": t["id"],
                  "symptoms": [{"symptom": "not_a_real_symptom", "severity": 6}]},
            headers=_h(s["at1"], "athletic_trainer"),
        )
        assert r.status_code == 400

    def test_severity_out_of_range_422(self, client):
        s, t = self._setup_with_scribe_consent(client)
        r = client.post(
            "/api/symptoms/scribe-log",
            json={"target_user_id": s["ath1"],
                  "team_id": t["id"],
                  "symptoms": [{"symptom": "headache", "severity": 99}]},
            headers=_h(s["at1"], "athletic_trainer"),
        )
        assert r.status_code == 422  # Pydantic field constraint

    def test_extra_field_rejected(self, client):
        s, t = self._setup_with_scribe_consent(client)
        r = client.post(
            "/api/symptoms/scribe-log",
            json={"target_user_id": s["ath1"],
                  "team_id": t["id"],
                  "symptoms": [{"symptom": "headache", "severity": 6}],
                  "evil": True},
            headers=_h(s["at1"], "athletic_trainer"),
        )
        assert r.status_code == 422

    def test_consent_revoked_mid_flight_403(self, client):
        """Athlete revokes scribe consent between extract and scribe-log
        → scribe-log must 403 (consent gate re-checks on every call)."""
        s, t = self._setup_with_scribe_consent(client)
        # First call succeeds
        r = client.post(
            "/api/symptoms/scribe-log",
            json={"target_user_id": s["ath1"],
                  "team_id": t["id"],
                  "symptoms": [{"symptom": "headache", "severity": 5}]},
            headers=_h(s["at1"], "athletic_trainer"),
        )
        assert r.status_code == 201

        # Athlete revokes by deleting the consent row directly (mirrors
        # the /share-consent/{cid} DELETE path; we don't need to exercise
        # the route here, just the post-revoke state)
        with get_connection() as db:
            db.execute(
                "UPDATE team_share_consents SET revoked_at = datetime('now') "
                "WHERE athlete_user_id=? AND team_id=?",
                (s["ath1"], t["id"]),
            )
            db.commit()

        # Second call must 403
        r = client.post(
            "/api/symptoms/scribe-log",
            json={"target_user_id": s["ath1"],
                  "team_id": t["id"],
                  "symptoms": [{"symptom": "nausea", "severity": 4}]},
            headers=_h(s["at1"], "athletic_trainer"),
        )
        assert r.status_code == 403

    def test_audit_trail_records_scribe(self, client):
        """audit_log must record actor=AT, target context = athlete + team."""
        s, t = self._setup_with_scribe_consent(client)
        client.post(
            "/api/symptoms/scribe-log",
            json={"target_user_id": s["ath1"],
                  "team_id": t["id"],
                  "symptoms": [{"symptom": "headache", "severity": 7}]},
            headers=_h(s["at1"], "athletic_trainer"),
        )
        with get_connection() as db:
            rows = db.execute(
                "SELECT actor_id, event_type, resource_type, detail "
                "FROM audit_log "
                "WHERE event_type = 'symptom_scribed' "
                "ORDER BY created_at DESC",
            ).fetchall()
        assert len(rows) >= 1
        row = rows[0]
        assert row["actor_id"] == s["at1"]
        # Detail should record the athlete (target) and team_id for forensics
        detail = json.loads(row["detail"]) if row["detail"] else {}
        assert detail.get("target_user_id") == s["ath1"]
        assert detail.get("team_id") == t["id"]
