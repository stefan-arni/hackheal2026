"""Session mode: scheduler priorities, staged publishing (error -> full/deadline -> update), live feed.

The service runs against the offline fal mock built from a SYNTHETIC run (nothing billed).
"""

import asyncio
import importlib
import json
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from service.scheduler import FrameScheduler, Job, coarse_level

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from make_synthetic_run import generate  # noqa: E402


# --- scheduler ---------------------------------------------------------------------


def test_coarse_level():
    assert [coarse_level(k) for k in range(8)] == [0, 2, 1, 2, 0, 2, 1, 2]


def _trial(created, deadline=None):
    return SimpleNamespace(created_at=created, deadline_at=deadline)


def test_job_order_burst_then_deadline_then_coarse():
    now = 1000.0
    a, b = _trial(900, deadline=1040), _trial(990)  # a ended (deadline in 40 s), b still recording (guess 1065)
    jobs = [Job(b, "b-u0", "uniform", 0, 1, seq=0), Job(a, "a-u2", "uniform", 2, 2, seq=1),
            Job(a, "a-u0", "uniform", 0, 3, seq=2), Job(b, "b-burst", "burst", 0, 4, seq=3),
            Job(a, "a-burst", "burst", 0, 5, seq=4)]
    order = [j.stem for j in sorted(jobs, key=lambda j: j.key(now, 45))]
    assert order == ["a-burst", "b-burst", "a-u0", "a-u2", "b-u0"]


def test_anchor_frames_run_with_the_bursts():
    tr = _trial(0, deadline=100)
    jobs = [Job(tr, "u-fine", "uniform", 2, 1, seq=0), Job(tr, "u-coarse", "uniform", 0, 2, seq=1),
            Job(tr, "anchor", "uniform", 0, 3, seq=2, anchor=True), Job(tr, "burst", "burst", 0, 4, seq=3)]
    assert [j.stem for j in sorted(jobs, key=lambda j: j.key(10, 45))] == ["anchor", "burst", "u-coarse", "u-fine"]


def test_non_foot_errors_keep_stance_frames():
    import numpy as np
    from service.pipeline import stance_mask
    t = np.arange(0, 5, 0.5)
    ev = [{"t": 1000 * x, "kind": "hands_off_hips"} for x in np.arange(0, 5, 0.25)] + [{"t": 2000, "kind": "foot_lift"}]
    m = stance_mask(t, ev)
    assert m.sum() == 5 and not m[(t >= 1.5) & (t <= 2.5)].any()  # only the foot lift (±1 s) is excluded


def test_past_deadline_goes_last():
    late, fresh = _trial(0, deadline=10), _trial(50, deadline=200)
    jobs = [Job(late, "late-burst", "burst", 0, 1, seq=0), Job(fresh, "fresh-u", "uniform", 2, 2, seq=1)]
    assert [j.stem for j in sorted(jobs, key=lambda j: j.key(100, 45))] == ["fresh-u", "late-burst"]


def test_scheduler_runs_in_priority_with_limited_workers():
    ran = []

    async def run(job):
        ran.append(job.stem)
        await asyncio.sleep(0.01)

    async def main():
        s = FrameScheduler(run, workers=1, deadline_s=45)
        tr = _trial(time.time())
        gate = asyncio.Event()
        orig = s.run

        async def first(job):  # hold the only worker until everything is queued
            await gate.wait()
            await orig(job)

        s.run = first
        await s.submit(Job(tr, "u-first", "uniform", 0, 0))
        await asyncio.sleep(0.01)
        for k, (stem, kind) in enumerate([("u2", "uniform"), ("u1", "uniform"), ("burst", "burst")]):
            await s.submit(Job(tr, stem, kind, {"u2": 2, "u1": 1, "burst": 0}[stem], k + 1))
        assert s.pending(tr) == 3 and s.in_flight(tr) == 1
        assert s.eta_s(tr) is not None
        gate.set()
        while s.pending() or s.in_flight():
            await asyncio.sleep(0.01)

    asyncio.run(main())
    assert ran == ["u-first", "burst", "u1", "u2"]


# --- service (mock fal) ------------------------------------------------------------


@pytest.fixture(scope="module")
def service(tmp_path_factory):
    syn = tmp_path_factory.mktemp("syn")
    gt = generate(syn, seed=3, duration_s=8.0, fps=3.0, render_images=True)
    env = {"REPLAY_MOCK_FAL": str(syn), "REPLAY_TRIALS_DIR": str(tmp_path_factory.mktemp("trials")),
           "REPLAY_MOCK_LATENCY": "0.05", "REPLAY_DEADLINE_S": "20", "REPLAY_FAL_WORKERS": "3"}
    old = {k: os.environ.get(k) for k in env}
    os.environ.update(env)
    sys.modules.pop("service.app", None)
    app_mod = importlib.import_module("service.app")
    from fastapi.testclient import TestClient

    with TestClient(app_mod.app) as client:
        yield SimpleNamespace(client=client, app=app_mod, syn=syn, gt=gt)
    sys.modules.pop("service.app", None)
    for k, v in old.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v


def _post_frames(c, tid, sid, syn, burst_every=0):
    summ = json.loads((syn / "summary.json").read_text())
    frames = [f for f in summ["frames"] if f.get("usable")]
    for k, f in enumerate(frames):
        jpeg = (syn / "input" / f"{f['stem']}.jpg").read_bytes()
        kind = "burst" if burst_every and k % burst_every == 0 else "uniform"
        r = c.post(f"/replay/{tid}/frame", files={"jpeg": ("f.jpg", jpeg, "image/jpeg")},
                   data={"t": str(f["t_ms"]), "crop": json.dumps([0, 0, 10, 10]), "frame_size": "[10, 10]",
                         "kind": kind, "session": sid})
        assert r.status_code == 200, r.text
    return len(frames)


def _wait(c, tid, pred, timeout=60):
    end = time.time() + timeout
    while time.time() < end:
        st = c.get(f"/replay/{tid}/status").json()
        if pred(st):
            return st
        time.sleep(0.2)
    raise AssertionError(f"timeout: {st}")


def test_session_flow(service):
    c, syn, gt = service.client, service.syn, service.gt
    sid = c.post("/session", json={"name": "test"}).json()["sessionId"]
    assert c.get(f"/dashboard/{sid}").status_code == 200

    # posecam publisher: a summary and an event for the first trial
    tid = "bess-double-1000"
    r = c.post(f"/live/{sid}/push", json={"items": [
        {"type": "summary", "t": 1, "trial": tid, "session": sid, "landmarks": [], "bess": {"phase": "running"}},
        {"type": "event", "kind": "bess_started", "stance": "double", "trial": tid, "session": sid}]})
    assert r.json() == {"ok": True}

    n = _post_frames(c, tid, sid, syn, burst_every=4)
    r = c.post(f"/replay/{tid}/end", json={"events": gt["events"], "patient_height_cm": gt["patient_height_m"] * 100,
                                           "session_id": sid, "bess": {"result": "bess_done", "stance": "double", "errors": 1}})
    assert r.status_code == 202
    st = _wait(c, tid, lambda s: s["timeline"].get("final_kind"))
    assert st["timeline"]["final_kind"] == "full" and st["done"] == n and st["fal_mode"] == "mock"
    meta = c.get(f"/replay/{tid}/meta.json").json()
    assert meta["coverage"]["done"] == n and meta["session"] == sid and meta["stance"] == "double"
    assert meta["quality"]["MOCK_FAL"] is True

    info = c.get(f"/session/{sid}").json()
    assert info["trials"][0]["trialId"] == tid
    page = c.get(f"/session/{sid}/report").text
    assert "Shortened 10 s demo protocol" in page and tid in page and "not a diagnostic" in page
    s = service.app.HUB.get(sid)
    kinds = [i.get("type") for i in s.feed]
    assert {"event", "trial", "progress", "trial_end", "replay"} <= set(kinds)
    assert s.summary["bess"]["phase"] == "running"


def test_deadline_then_straggler_update(service, monkeypatch):
    app, c, syn, gt = service.app, service.client, service.syn, service.gt
    sid = c.post("/session").json()["sessionId"]
    tid = "bess-single-2000"
    monkeypatch.setattr(app, "DEADLINE_S", 1.0)
    mock = app.BACKEND
    monkeypatch.setattr(mock, "straggler", (mock.calls + 5, 4.0))  # 5th call of this trial takes 4 s
    n = _post_frames(c, tid, sid, syn)
    c.post(f"/replay/{tid}/end", json={"events": gt["events"], "session_id": sid})
    st = _wait(c, tid, lambda s: s["timeline"].get("final_kind"))
    assert st["timeline"]["final_kind"] == "deadline"
    done_at_deadline = int(st["timeline"]["final_frames"].split("/")[0])
    assert done_at_deadline < n
    st = _wait(c, tid, lambda s: s["timeline"].get("updates_s") and s["done"] == n and s["stage"] == "update")
    assert st["stragglers"] >= 1
    meta = c.get(f"/replay/{tid}/meta.json").json()
    assert meta["coverage"]["stage"] == "update" and meta["coverage"]["done"] == n
