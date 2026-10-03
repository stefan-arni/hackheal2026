"""FastAPI service: sessions, live dashboard, frame ingest, fal queue, staged replay publishing.

    uv run uvicorn service.app:app --port 8017                                          # cache only (default)
    REPLAY_LIVE=1 REPLAY_YES=1 uv run uvicorn service.app:app --port 8017               # billed fal calls allowed
    REPLAY_MOCK_FAL=data/fal_out/demo uv run uvicorn service.app:app --port 8017        # offline mock (rehearsals)

fal budget: frames are looked up in the shared cache (data/fal_cache) first. Without
REPLAY_LIVE=1 nothing is sent and cache misses are listed in /status. Live mode also needs
REPLAY_YES=1; caps are 500 calls total / 80 per trial (REPLAY_BUDGET_OVERRIDE=1 to exceed).

Ports: 8017 (8000 is often taken; posecam uses 8765 pose / 8766 eyes).

Session + live dashboard:
    POST /session                    {"name"?} -> {"sessionId"}
    GET  /session/{id}               trials and their replay status
    GET  /session/{id}/report        one-page BESS-style session summary
    POST /live/{id}/push             posecam's publisher: {"items": [summary | event, ...]}
    GET  /live/{id}/stream           server-sent events for the dashboard
    GET  /dashboard/{id}             the doctor's live page
Trials (from posecam's forwarder):
    POST /replay/{trialId}/frame     multipart: jpeg, t, crop, frame_size, kind, session?, mask?, gravity?
    POST /replay/{trialId}/end       JSON: events, patient_height_cm?, session_id?, bess?, clock?
    GET  /replay/{trialId}/status    stage, frames done/received, ETA, coverage, timeline
    GET  /replay/{trialId}/report    trial report;  /replay/{trialId}/<bundle files>
    POST /replay/warmup?force=1      one fal call to avoid a cold start (force: skip the cache)

Queue (service/scheduler.py): one priority order across the session's trials, with
REPLAY_FAL_WORKERS (default 3 = fal's measured parallelism) frames in flight: error bursts
first, then coarse-to-fine uniform frames by trial deadline. Publishing per trial: an "error"
replay as soon as its burst frames are done, then "full" (all frames) or "deadline"
(REPLAY_DEADLINE_S after the trial ends, default 30 s, with whatever is done; the viewer
interpolates the gaps); late frames are logged and trigger an "update" republish.
"""

from __future__ import annotations

import asyncio
import io
import json
import logging
import os
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image
from pydantic import BaseModel

from service import analytics, bundle, geometry, pipeline, report, session_report
from service.capture_defaults import LIVE_MAX_HEIGHT_PX
from service.fal_budget import Budget, BudgetError
from service.fal_client_wrap import REPLAY_ROOT, CacheMiss, SamBodyClient
from service.fal_mock import MockFal
from service.images import downscale_jpeg
from service.live import LiveHub, Session
from service.runs import load_records, write_frame, write_summary
from service.scheduler import ANCHOR_FRAMES, FrameScheduler, Job, coarse_level

log = logging.getLogger("replay")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

TRIALS_DIR = Path(os.environ.get("REPLAY_TRIALS_DIR", REPLAY_ROOT / "data/trials"))
DEADLINE_S = float(os.environ.get("REPLAY_DEADLINE_S", 30))
WORKERS = int(os.environ.get("REPLAY_FAL_WORKERS", 3))
UPDATE_EVERY_S = float(os.environ.get("REPLAY_UPDATE_EVERY_S", 10))  # late frames: republish at most this often
MIN_FRAMES = 6  # fewer reconstructed frames than this: no bundle yet
ERROR_WAIT_S = 15.0  # a straggling burst frame holds the error replay at most this long after the trial ends
MOCK_RUN = os.environ.get("REPLAY_MOCK_FAL")
CONVENTIONS: dict | None = None  # None -> pipeline looks for data/conventions.json
LIVE = os.environ.get("REPLAY_LIVE") == "1"
# Phone gravity (per frame, `gravity` form field): "check" = report its angle to the feet's floor;
# "override" = use it for the floor tilt. On IMG_9691, SAM's whole mesh sat ~4.9° off true gravity
# while its feet stayed flat on its own floor, so overriding moved the COM ~9 cm the wrong way.
GRAVITY_MODE = os.environ.get("REPLAY_GRAVITY_MODE", "check")
MAX_USD = float(os.environ["REPLAY_MAX_USD"]) if os.environ.get("REPLAY_MAX_USD") else None  # hard stop per service run
BUDGET = Budget(override=os.environ.get("REPLAY_BUDGET_OVERRIDE") == "1")
if MAX_USD is not None:
    BUDGET.max_session_calls = int(MAX_USD / BUDGET.price()[0] + 1e-9)
    log.warning("spend cap: at most %d fal calls ($%.2f) while this service runs", BUDGET.max_session_calls, MAX_USD)
BACKEND = None
if MOCK_RUN:
    dirs = [Path(d) if Path(d).is_absolute() else REPLAY_ROOT / d for d in MOCK_RUN.split(",")]
    BACKEND = MockFal(dirs)
    log.warning("fal is MOCKED from %s (%d frames; %s); nothing billed or cached",
                ", ".join(str(d) for d in dirs), len(BACKEND.records), BACKEND.describe())
    conv_file = dirs[0] / "conventions.json"
    CONVENTIONS = json.loads(conv_file.read_text()) if conv_file.exists() else None
elif LIVE:
    if os.environ.get("REPLAY_YES") != "1":
        raise SystemExit("REPLAY_LIVE=1 also needs REPLAY_YES=1: the service can't ask before each batch. "
                         f"Budget now: {BUDGET.status_line()}")
    log.warning("fal LIVE: billed calls allowed. %s", BUDGET.status_line())
else:
    log.warning("fal CACHE-ONLY: frames not in data/fal_cache will be reported missing (REPLAY_LIVE=1 to pay)")

app = FastAPI(title="Instant Replay")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
HUB = LiveHub(TRIALS_DIR / "_sessions")

_sam: SamBodyClient | None = None


def sam() -> SamBodyClient:
    global _sam
    if _sam is None:  # one client: the fal concurrency limit is per account, shared across trials
        _sam = SamBodyClient(live=LIVE, run="service", budget=BUDGET, backend=BACKEND, concurrency=WORKERS)
    return _sam


@dataclass
class Trial:
    id: str
    dir: Path
    session: str | None = None
    created_at: float = field(default_factory=time.time)
    state: str = "receiving"  # receiving -> processing -> ready | failed
    stage: str = "receiving"  # receiving | processing | error | full | deadline | update
    crop: list[int] | None = None
    received: int = 0
    done: int = 0
    failed: int = 0
    bursts: int = 0
    bursts_done: int = 0
    anchors: int = 0
    anchors_done: int = 0
    uniform_k: int = 0
    missing: list[str] = field(default_factory=list)
    t_end: float | None = None
    deadline_at: float | None = None
    payload: Any = None
    version: int = 0
    final_published: bool = False
    error_published: bool = False
    stragglers: int = 0
    republish: asyncio.Task | None = None
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    timeline: dict = field(default_factory=dict)
    aligned: bool | None = None
    error: str | None = None

    @property
    def frames_dir(self) -> Path:
        return self.dir / "frames"

    @property
    def stance(self) -> str | None:
        parts = self.id.split("-")
        return parts[1] if len(parts) >= 3 and parts[0] == "bess" else None


trials: dict[str, Trial] = {}


def get_trial(trial_id: str, create: bool = False, session: str | None = None) -> Trial:
    if trial_id not in trials:
        if not create:
            raise HTTPException(404, f"unknown trial {trial_id}")
        if "/" in trial_id or trial_id.startswith((".", "_")):
            raise HTTPException(400, "bad trial id")
        trials[trial_id] = Trial(trial_id, TRIALS_DIR / trial_id)
        trials[trial_id].frames_dir.mkdir(parents=True, exist_ok=True)
    tr = trials[trial_id]
    if session and not tr.session:
        tr.session = session
        HUB.add_trial(HUB.get(session, create=True), trial_id)
    return tr


def live(tr: Trial, item: dict) -> None:
    if tr.session and (s := HUB.get(tr.session)):
        HUB.publish(s, {"trial": tr.id, "t_wall": time.time(), **item})


# ------------------------------------------------------------------ frames + queue

SRC_HEIGHT = 640


def save_source_frame(path: Path, jpeg: bytes) -> None:
    """Keep a small copy of the image sent to fal, for the viewer's video overlay and the report."""
    with Image.open(io.BytesIO(jpeg)) as im:
        k = SRC_HEIGHT / im.height
        im.convert("RGB").resize((round(im.width * k), SRC_HEIGHT), Image.LANCZOS).save(path, "JPEG", quality=82)


def downscale_for_live(jpeg: bytes, mask: bytes | None) -> tuple[bytes, bytes | None, list[int], float]:
    """Live trials: shrink frames taller than LIVE_MAX_HEIGHT_PX (faster fal inference). The
    trial's fixed crop means every frame gets the same scale. Returns (jpeg, mask, size, scale)."""
    jpeg, size, s = downscale_jpeg(jpeg, LIVE_MAX_HEIGHT_PX)
    if s != 1.0 and mask is not None:
        with Image.open(io.BytesIO(mask)) as m:
            mb = io.BytesIO()
            m.resize(tuple(size), Image.NEAREST).save(mb, "PNG")
            mask = mb.getvalue()
    return jpeg, mask, size, s


async def run_job(job: Job) -> None:
    tr, p = job.trial, job.payload
    ok = False
    try:
        jpeg, mask, size, scale = downscale_for_live(p["jpeg"], p["mask"])
        await asyncio.to_thread(save_source_frame, tr.frames_dir / f"{job.stem}_src.jpg", jpeg)
        extra = {**p["extra"], "downscale": round(scale, 5)}
        t_call = time.time()
        res = await sam().reconstruct(jpeg, mask, run=tr.id, priority=0 if job.kind == "burst" else 1)
        extra.update(call_start_wall=round(t_call, 3), call_end_wall=round(time.time(), 3),  # parallelism report
                     from_cache=res.from_cache)
        write_frame(tr.frames_dir, job.stem, t_ms=job.t_ms, image_size=size, response=res.response,
                    latency_s=res.latency_s, ply=res.ply, visualization=res.visualization, extra=extra)
        tr.done += 1
        ok = True
    except CacheMiss as e:
        tr.failed += 1
        tr.missing.append(job.stem)
        log.warning("trial %s frame %s MISSING from fal cache (sha1 %s); not sent (cache-only)", tr.id, job.stem, e.key[:12])
    except BudgetError as e:
        tr.failed += 1
        log.error("trial %s frame %s not sent: %s", tr.id, job.stem, e)
    except Exception:
        tr.failed += 1
        log.exception("trial %s frame %s failed", tr.id, job.stem)
    if job.kind == "burst":
        tr.bursts_done += 1
    if job.anchor:
        tr.anchors_done += 1
    live(tr, {"type": "progress", "done": tr.done, "received": tr.received, "failed": tr.failed,
              "eta_s": SCHED.eta_s(tr), "stage": tr.stage})
    if ok and tr.final_published:  # a straggler: log it, republish shortly (debounced)
        tr.stragglers += 1
        log.warning("trial %s: late frame %s done %.1fs after the %s replay; republish queued",
                    tr.id, job.stem, time.time() - (tr.timeline.get("final_wall") or time.time()),
                    tr.timeline.get("final_kind"))
        if tr.republish is None or tr.republish.done():
            tr.republish = asyncio.create_task(republish_later(tr))


SCHED = FrameScheduler(run_job, workers=WORKERS, deadline_s=DEADLINE_S)


@app.post("/replay/warmup")
async def warmup(jpeg: UploadFile = File(...), force: bool = False) -> dict:
    """Cached frames cost nothing; force=1 really calls fal (one billed call in live mode)."""
    try:
        return {"latency_s": round(await sam().warmup(await jpeg.read(), force=force), 2),
                "budget": BUDGET.status_line()}
    except CacheMiss as e:
        raise HTTPException(409, f"warm-up needs live fal (REPLAY_LIVE=1) or the mock: {e.key[:12]}")


@app.post("/replay/{trial_id}/frame")
async def post_frame(
    trial_id: str,
    jpeg: UploadFile = File(...),
    t: float = Form(...),
    crop: str = Form(...),
    frame_size: str = Form(...),
    kind: str = Form("uniform"),
    session: str | None = Form(None),
    mask: UploadFile | None = File(None),
    gravity: str | None = Form(None),
) -> dict:
    trial = get_trial(trial_id, create=True, session=session)
    if trial.state != "receiving":
        raise HTTPException(409, f"trial {trial_id} already ended")
    crop_box = json.loads(crop)
    if trial.crop is None:
        trial.crop = crop_box
    elif crop_box != trial.crop:  # per-frame crops create fake translation
        raise HTTPException(400, f"crop changed mid-trial: {crop_box} != {trial.crop}")
    if kind not in ("uniform", "burst"):
        raise HTTPException(400, f"bad kind {kind}")
    stem = f"t{int(round(t)):013d}"
    extra = {"kind": kind, "crop": crop_box, "frame_size": json.loads(frame_size),
             "gravity": json.loads(gravity) if gravity else None}
    payload = {"jpeg": await jpeg.read(), "mask": await mask.read() if mask is not None else None, "extra": extra}
    level, anchor = 0, False
    if kind == "uniform":
        level = coarse_level(trial.uniform_k)
        anchor = level == 0 and trial.uniform_k < 4 * ANCHOR_FRAMES
        trial.uniform_k += 1
    else:
        trial.bursts += 1
    trial.anchors += anchor
    trial.received += 1
    await SCHED.submit(Job(trial, stem, kind, level, t, payload, anchor=anchor))
    return {"accepted": stem, "received": trial.received, "queued": SCHED.pending(trial)}


class EndPayload(BaseModel):
    events: list[dict[str, Any]] = []
    patient_height_cm: float | None = None  # None: keep SAM's own metric scale
    landmarks_2d: Any | None = None
    session_id: str | None = None
    bess: dict[str, Any] | None = None
    clock: str | None = None


@app.post("/replay/{trial_id}/end", status_code=202)
async def post_end(trial_id: str, payload: EndPayload) -> dict:
    trial = get_trial(trial_id, session=payload.session_id)
    if trial.state != "receiving":
        raise HTTPException(409, f"trial {trial_id} already ended")
    trial.state, trial.stage, trial.payload = "processing", "processing", payload
    trial.t_end = time.time()
    trial.deadline_at = trial.t_end + DEADLINE_S
    trial.timeline["end_wall"] = trial.t_end
    (trial.dir / "end.json").write_text(payload.model_dump_json())
    await SCHED.poke()
    live(trial, {"type": "trial_end", "received": trial.received, "bursts": trial.bursts, "deadline_s": DEADLINE_S,
                 "bess": payload.bess})
    asyncio.create_task(finalize(trial))
    return {"status": "processing", "received": trial.received, "deadline_s": DEADLINE_S}


# ------------------------------------------------------------------ staged publishing

async def finalize(tr: Trial) -> None:
    """Error replay as soon as the burst frames are in; then full (all frames) or deadline."""
    tried_at = -1  # frames done at the last error-replay attempt (retry only with new frames)
    try:
        while True:
            settled = tr.done + tr.failed >= tr.received
            prio, prio_done = tr.bursts + tr.anchors, tr.bursts_done + tr.anchors_done
            prio_ready = prio_done >= prio or (time.time() - tr.t_end > ERROR_WAIT_S and prio_done >= 0.75 * prio)
            if (not tr.error_published and tr.bursts and prio_ready and tr.done >= MIN_FRAMES and not settled
                    and tr.done != tried_at):
                tried_at = tr.done
                tr.error_published = await publish(tr, "error")
            if settled:
                await publish(tr, "full", final=True)
                return
            if time.time() >= tr.deadline_at:
                await publish(tr, "deadline", final=True)
                return
            await asyncio.sleep(0.5)
    except Exception as e:
        tr.state, tr.error = "failed", repr(e)
        log.exception("trial %s failed", tr.id)
        live(tr, {"type": "replay", "stage": "failed", "error": repr(e)})


async def republish_later(tr: Trial) -> None:
    last = tr.timeline.get("last_publish_wall") or 0
    await asyncio.sleep(max(3.0, last + UPDATE_EVERY_S - time.time()))  # collect late frames per republish
    await publish(tr, "update", final=True)


async def publish(tr: Trial, stage: str, final: bool = False) -> bool:
    async with tr.lock:
        if tr.done < (MIN_FRAMES if stage == "error" else 1):
            if final and not tr.final_published:
                tr.state, tr.error = "failed", f"no frames reconstructed ({tr.failed} failed, {len(tr.missing)} missing)"
                live(tr, {"type": "replay", "stage": "failed", "error": tr.error})
            return False
        t0 = time.time()
        try:
            meta = await asyncio.to_thread(build_bundle, tr, stage)
        except ValueError as e:  # e.g. too few stance frames for an early error replay
            log.info("trial %s: %s replay not possible yet (%s)", tr.id, stage, e)
            return False
        tr.version += 1
        tr.stage = stage
        now = time.time()
        tr.timeline["last_publish_wall"] = now
        if stage == "error":
            tr.timeline["error_ready_s"] = round(now - tr.t_end, 2)
        elif stage in ("full", "deadline"):
            tr.timeline["final_ready_s"] = round(now - tr.t_end, 2)
            tr.timeline["final_kind"] = stage
            tr.timeline["final_frames"] = f"{tr.done}/{tr.received}"
            tr.timeline["final_wall"] = now
        elif stage == "update":
            tr.timeline.setdefault("updates_s", []).append(round(now - tr.t_end, 2))
        if final:
            tr.final_published, tr.state = True, "ready"
        tr.aligned = meta["quality"].get("aligned")
        log.info("trial %s: %s replay v%d published %.1fs after trial end (%d/%d frames, built in %.1fs)",
                 tr.id, stage, tr.version, now - tr.t_end, tr.done, tr.received, now - t0)
        live(tr, {"type": "replay", "stage": stage, "version": tr.version, "url": f"/replay/{tr.id}/",
                  "report": f"/replay/{tr.id}/report", "coverage": meta.get("coverage"),
                  "after_end_s": round(now - tr.t_end, 2), "analytics": meta.get("analytics")})
        if final and stage != "update":
            await notify_ready(tr)
        return True


def build_bundle(tr: Trial, stage: str) -> dict:
    """Pipeline over the frames done so far -> bundle + analytics + report, swapped in atomically."""
    write_summary(tr.frames_dir, {"trialId": tr.id})
    run = pipeline.load_run(tr.frames_dir, CONVENTIONS)
    ev = tr.payload.events if tr.payload else []
    up = phone_up_in_sam_frame(tr, run)
    up_kw = {}
    if up is not None:
        up_kw = {"up_cam": up, "up_source": "phone_gravity"} if GRAVITY_MODE == "override" else {"up_check": up}
    height = tr.payload.patient_height_cm / 100 if tr.payload and tr.payload.patient_height_cm else None
    try:
        result = pipeline.process(run, ev, height, **up_kw)
    except NotImplementedError:
        result = pipeline.raw_display(run)
    except ValueError:
        if stage == "error":
            raise
        log.warning("trial %s: pipeline failed on %d frames; publishing raw camera-frame bundle", tr.id, len(run.t_s))
        result = pipeline.raw_display(run)
    recs = load_records(tr.frames_dir)
    if any(r["metadata"].get("SYNTHETIC") for r in recs):
        result["quality"]["SYNTHETIC"] = True
    if BACKEND is not None:
        result["quality"]["MOCK_FAL"] = True
    sizes = {Path(r["frame"]).stem: r["image_size"][1] for r in recs}
    frames = []
    for t, v, stem in zip(result["t_s"], result.get("vis_files", run.vis_files), result.get("stems", run.stems)):
        fr = {"t": round(t * 1000, 1), "fal_vis_url": f"frames/{v}" if v else None}
        src = tr.frames_dir / f"{stem}_src.jpg"
        if src.exists():
            fr["src_url"], fr["src_scale"] = f"frames/{stem}_src.jpg", round(SRC_HEIGHT / sizes.get(stem, SRC_HEIGHT), 5)
        frames.append(fr)
    stats = analytics.compute(result, ev) if result["quality"].get("aligned") else None
    coverage = {"done": tr.done, "received": tr.received, "ratio": round(tr.done / max(tr.received, 1), 3),
                "stage": stage, "version": tr.version + 1, "failed": tr.failed}
    staging = tr.dir / "_staging"
    shutil.rmtree(staging, ignore_errors=True)
    staging.mkdir()
    meta = bundle.write_bundle(staging, tr.id, result, events=ev, frames=frames, analytics=stats,
                               extra_meta={"coverage": coverage, "session": tr.session, "stance": tr.stance,
                                           "bess": tr.payload.bess if tr.payload else None})
    if stats is not None:
        report.write_report(tr.dir, meta, stats, result, title=f"{(tr.stance or 'trial').capitalize()} stance · {tr.id}")
    for name in ("faces.bin", "verts.bin", "verts_raw.bin", "analytics.json", "meta.json"):  # meta last
        if (staging / name).exists():
            os.replace(staging / name, tr.dir / name)
    shutil.rmtree(staging, ignore_errors=True)
    return meta


def phone_up_in_sam_frame(trial: Trial, run: pipeline.Run) -> np.ndarray | None:
    """Median phone gravity -> 'up' in SAM's camera frame, or None if the phone sent none.

    Contract (capture side): `gravity` = direction of gravity (pointing DOWN) in the camera's
    OpenCV frame for the full, uncropped image: x right, y down, z forward; any magnitude.
    iOS rear camera in portrait: (gx, -gy, -gz) from CMDeviceMotion.gravity.
    """
    recs = [r for r in load_records(trial.frames_dir) if r.get("gravity") and r.get("usable")]
    if not recs:
        return None
    gv = np.array([r["gravity"] for r in recs], float)
    gv /= np.linalg.norm(gv, axis=1, keepdims=True)
    up_true = -np.median(gv, axis=0)
    crop, size = np.array(recs[0]["crop"], float), recs[0]["frame_size"]
    R = geometry.true_to_sam_rotation((crop[:2] + crop[2:]) / 2, tuple(size), float(np.median(run.focal)))
    return R @ (up_true / np.linalg.norm(up_true))


async def notify_ready(trial: Trial) -> None:
    msg = {"type": "replay_ready", "trialId": trial.id, "url": f"/replay/{trial.id}/"}
    relay = os.environ.get("RELAY_WS_URL")
    if not relay:
        log.info("replay_ready (no RELAY_WS_URL set): %s", msg)
        return
    try:
        import websockets

        async with websockets.connect(relay) as ws:
            await ws.send(json.dumps(msg))
    except Exception:
        log.exception("relay notify failed")


def trial_status(tr: Trial) -> dict:
    return {
        "trialId": tr.id, "session": tr.session, "stance": tr.stance, "state": tr.state, "stage": tr.stage,
        "version": tr.version, "received": tr.received, "done": tr.done, "failed": tr.failed,
        "bursts": tr.bursts, "bursts_done": tr.bursts_done, "queued": SCHED.pending(tr), "in_flight": SCHED.in_flight(tr),
        "eta_s": SCHED.eta_s(tr), "stragglers": tr.stragglers, "aligned": tr.aligned, "error": tr.error,
        "missing_from_cache": tr.missing, "fal_mode": "mock" if BACKEND else ("live" if LIVE else "cache-only"),
        "deadline_in_s": round(tr.deadline_at - time.time(), 1) if tr.deadline_at and not tr.final_published else None,
        "timeline": {k: v for k, v in tr.timeline.items() if not k.endswith("_wall")},
        "url": f"/replay/{tr.id}/" if tr.version else None,
        "report": f"/replay/{tr.id}/report" if tr.version and (tr.dir / "report.html").exists() else None,
    }


@app.get("/replay/{trial_id}/status")
async def status(trial_id: str) -> dict:
    return trial_status(get_trial(trial_id))


@app.get("/replay/{trial_id}/report")
async def report_page(trial_id: str) -> FileResponse:
    path = (TRIALS_DIR / trial_id / "report.html").resolve()
    if not path.is_relative_to(TRIALS_DIR.resolve()) or not path.is_file():
        raise HTTPException(404, "no report for this trial yet")
    return FileResponse(path, headers={"Cache-Control": "no-store"})


@app.api_route("/replay/{trial_id}/{path:path}", methods=["GET", "HEAD"])
async def bundle_file(trial_id: str, path: str) -> FileResponse:
    base = (TRIALS_DIR / trial_id).resolve()
    target = (base / path).resolve()
    if not target.is_relative_to(base) or not target.is_file():
        raise HTTPException(404)
    return FileResponse(target, headers={"Cache-Control": "no-store"})  # bundles are republished in place


# ------------------------------------------------------------------ sessions, live feed, pages

class NewSession(BaseModel):
    name: str = ""


@app.post("/session")
async def new_session(body: NewSession | None = None) -> dict:
    s = HUB.create(body.name if body else "")
    return {"sessionId": s.id, "dashboard": f"/dashboard/{s.id}", "report": f"/session/{s.id}/report"}


def get_session(sid: str) -> Session:
    s = HUB.get(sid)
    if s is None:
        raise HTTPException(404, f"unknown session {sid}")
    return s


@app.get("/session/{sid}")
async def session_info(sid: str) -> dict:
    s = get_session(sid)
    return {**s.to_json(), "trials": [trial_status(trials[t]) if t in trials else {"trialId": t, "state": "unknown"}
                                      for t in s.trials]}


@app.get("/session/{sid}/report", response_class=HTMLResponse)
async def session_report_page(sid: str) -> HTMLResponse:
    s = get_session(sid)
    rows = []
    for tid in s.trials:
        d = TRIALS_DIR / tid
        a = json.loads((d / "analytics.json").read_text()) if (d / "analytics.json").exists() else None
        m = json.loads((d / "meta.json").read_text()) if (d / "meta.json").exists() else {}
        end = json.loads((d / "end.json").read_text()) if (d / "end.json").exists() else {}
        rows.append({"trial": tid, "stance": tid.split("-")[1] if tid.startswith("bess-") else "", "analytics": a,
                     "meta": m, "end": end, "status": trial_status(trials[tid]) if tid in trials else None})
    return HTMLResponse(session_report.render(s.to_json(), rows), headers={"Cache-Control": "no-store"})


@app.post("/live/{sid}/push")
async def live_push(sid: str, body: dict) -> dict:
    s = HUB.get(sid, create=True)
    for item in body.get("items", []):
        if item.get("trial"):
            get_trial(item["trial"], create=True, session=sid)
        HUB.publish(s, item)
    return {"ok": True}


@app.get("/live/{sid}/stream")
async def live_stream(sid: str) -> StreamingResponse:
    s = HUB.get(sid, create=True)
    return StreamingResponse(HUB.stream(s), media_type="text/event-stream",
                             headers={"Cache-Control": "no-store", "X-Accel-Buffering": "no"})


@app.get("/dashboard/{sid}", response_class=HTMLResponse)
async def dashboard(sid: str) -> HTMLResponse:
    HUB.get(sid, create=True)
    return HTMLResponse((REPLAY_ROOT / "dashboard" / "index.html").read_text(), headers={"Cache-Control": "no-store"})


(REPLAY_ROOT / "cache").mkdir(exist_ok=True)
app.mount("/cache", StaticFiles(directory=REPLAY_ROOT / "cache"), name="cache")
app.mount("/viewer", StaticFiles(directory=REPLAY_ROOT / "viewer", html=True), name="viewer")
