"""FastAPI service: ingest frames, call fal, post-process, serve bundles.

    uv run uvicorn service.app:app --port 8017                                          # cache only (default)
    REPLAY_LIVE=1 REPLAY_YES=1 uv run uvicorn service.app:app --port 8017               # billed fal calls allowed
    REPLAY_MOCK_FAL=data/fal_out/synthetic uv run uvicorn service.app:app --port 8017   # offline mock

fal budget: frames are looked up in the shared cache (data/fal_cache) first. Without
REPLAY_LIVE=1 nothing is sent and cache misses are listed in /status. Live mode also needs
REPLAY_YES=1 (the server can't prompt per batch); caps are 500 calls total / 80 per trial
(REPLAY_BUDGET_OVERRIDE=1 to exceed) and only 3 calls until the price is confirmed.

Ports: 8017 (8000 is often taken; posecam uses 8765 pose / 8766 eyes).

Routes (REPLAY_SPEC.md "Interfaces"):
    POST /replay/{trialId}/frame     multipart: jpeg, mask?, t, crop, frame_size, kind, gravity?
    POST /replay/{trialId}/end       JSON: events, patient_height_cm, landmarks_2d?
    GET  /replay/{trialId}/status    progress / readiness
    GET  /replay/{trialId}/meta.json | faces.bin | verts.bin | frames/<vis>
    POST /replay/warmup              multipart: jpeg — one request to avoid a cold start
    GET  /viewer/                    the three.js viewer (?src=/replay/<trialId>/)

Each frame is submitted to fal the moment it arrives; /end awaits stragglers, runs
the geometry pipeline and publishes the bundle. If geometry.py isn't implemented yet,
the bundle falls back to raw camera-frame meshes (quality.aligned = false).
"""

from __future__ import annotations

import asyncio
import io
import json
import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image
from pydantic import BaseModel

import numpy as np

from service import analytics, bundle, geometry, pipeline, report
from service.fal_budget import Budget, BudgetError
from service.fal_client_wrap import REPLAY_ROOT, CacheMiss, SamBodyClient
from service.fal_mock import MockFal
from service.capture_defaults import LIVE_MAX_HEIGHT_PX
from service.images import downscale_jpeg
from service.runs import DEFAULT_CONVENTIONS, load_records, write_frame, write_summary

log = logging.getLogger("replay")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

TRIALS_DIR = Path(os.environ.get("REPLAY_TRIALS_DIR", REPLAY_ROOT / "data/trials"))
MOCK_RUN = os.environ.get("REPLAY_MOCK_FAL")
CONVENTIONS: dict | None = None  # None -> pipeline looks for data/conventions.json

LIVE = os.environ.get("REPLAY_LIVE") == "1"
# Phone gravity (per frame, `gravity` form field): "check" = report its angle to the feet's floor;
# "override" = use it for the floor tilt. On IMG_9691, SAM's whole mesh sat ~4.9° off true gravity
# while its feet stayed flat on its own floor, so overriding moved the COM ~9 cm the wrong way.
GRAVITY_MODE = os.environ.get("REPLAY_GRAVITY_MODE", "check")
BUDGET = Budget(override=os.environ.get("REPLAY_BUDGET_OVERRIDE") == "1")
BACKEND = None
if MOCK_RUN:
    mock_dir = Path(MOCK_RUN) if Path(MOCK_RUN).is_absolute() else REPLAY_ROOT / MOCK_RUN
    BACKEND = MockFal(mock_dir)
    log.warning("fal is MOCKED from %s (%d frames); nothing billed or cached", mock_dir, len(BACKEND.index))
    conv_file = mock_dir / "conventions.json"
    CONVENTIONS = json.loads(conv_file.read_text()) if conv_file.exists() else DEFAULT_CONVENTIONS
elif LIVE:
    if os.environ.get("REPLAY_YES") != "1":
        raise SystemExit("REPLAY_LIVE=1 also needs REPLAY_YES=1: the service can't ask before each batch. "
                         f"Budget now: {BUDGET.status_line()}")
    log.warning("fal LIVE: billed calls allowed. %s", BUDGET.status_line())
else:
    log.warning("fal CACHE-ONLY: frames not in data/fal_cache will be reported missing (REPLAY_LIVE=1 to pay)")

app = FastAPI(title="Instant Replay")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

_sam: SamBodyClient | None = None


def sam() -> SamBodyClient:
    global _sam
    if _sam is None:  # one client: the fal concurrency limit is per account, shared across trials
        _sam = SamBodyClient(live=LIVE, run="service", budget=BUDGET, backend=BACKEND)
    return _sam


@dataclass
class Trial:
    id: str
    dir: Path
    state: str = "receiving"  # receiving -> processing -> ready | failed
    crop: list[int] | None = None
    tasks: list[asyncio.Task] = field(default_factory=list)
    received: int = 0
    done: int = 0
    failed: int = 0
    missing: list[str] = field(default_factory=list)  # frames not in the cache (cache-only mode)
    t_end: float | None = None
    t_ready: float | None = None
    aligned: bool | None = None
    error: str | None = None

    @property
    def frames_dir(self) -> Path:
        return self.dir / "frames"


trials: dict[str, Trial] = {}


def get_trial(trial_id: str, create: bool = False) -> Trial:
    if trial_id not in trials:
        if not create:
            raise HTTPException(404, f"unknown trial {trial_id}")
        if "/" in trial_id or trial_id.startswith("."):
            raise HTTPException(400, "bad trial id")
        trials[trial_id] = Trial(trial_id, TRIALS_DIR / trial_id)
        trials[trial_id].frames_dir.mkdir(parents=True, exist_ok=True)
    return trials[trial_id]


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


async def run_frame(trial: Trial, stem: str, jpeg: bytes, mask: bytes | None, t_ms: float, extra: dict) -> None:
    try:
        jpeg, mask, size, scale = downscale_for_live(jpeg, mask)
        await asyncio.to_thread(save_source_frame, trial.frames_dir / f"{stem}_src.jpg", jpeg)
        extra = {**extra, "downscale": round(scale, 5)}
        res = await sam().reconstruct(jpeg, mask, run=trial.id, priority=0 if extra["kind"] == "burst" else 1)
        write_frame(
            trial.frames_dir, stem, t_ms=t_ms, image_size=size, response=res.response,
            latency_s=res.latency_s, ply=res.ply, visualization=res.visualization, extra=extra,
        )
        trial.done += 1
    except CacheMiss as e:
        trial.failed += 1
        trial.missing.append(stem)
        log.warning("trial %s frame %s MISSING from fal cache (sha1 %s); not sent (cache-only)", trial.id, stem, e.key[:12])
    except BudgetError as e:
        trial.failed += 1
        log.error("trial %s frame %s not sent: %s", trial.id, stem, e)
    except Exception:
        trial.failed += 1
        log.exception("trial %s frame %s failed", trial.id, stem)


@app.post("/replay/warmup")
async def warmup(jpeg: UploadFile = File(...)) -> dict:
    """Cached frames cost nothing; a new frame is one billed call (live mode only)."""
    try:
        return {"latency_s": await sam().warmup(await jpeg.read())}
    except CacheMiss as e:
        raise HTTPException(409, f"warmup frame not cached and fal is not live: {e.key[:12]}")


@app.post("/replay/{trial_id}/frame")
async def post_frame(
    trial_id: str,
    jpeg: UploadFile = File(...),
    t: float = Form(...),
    crop: str = Form(...),
    frame_size: str = Form(...),
    kind: str = Form("uniform"),
    mask: UploadFile | None = File(None),
    gravity: str | None = Form(None),
) -> dict:
    trial = get_trial(trial_id, create=True)
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
    jpeg_bytes = await jpeg.read()
    mask_bytes = await mask.read() if mask is not None else None
    # Pipelining: submit now, don't wait for the result.
    trial.tasks.append(asyncio.create_task(run_frame(trial, stem, jpeg_bytes, mask_bytes, t, extra)))
    trial.received += 1
    return {"accepted": stem, "received": trial.received, "inflight": trial.received - trial.done - trial.failed}


class EndPayload(BaseModel):
    events: list[dict[str, Any]] = []
    patient_height_cm: float | None = None  # None: keep SAM's own metric scale
    landmarks_2d: Any | None = None


async def finalize(trial: Trial, payload: EndPayload) -> None:
    try:
        await asyncio.gather(*trial.tasks)  # stragglers
        if trial.missing:
            log.warning("trial %s: %d/%d frames missing from fal cache: %s", trial.id, len(trial.missing),
                        trial.received, ", ".join(trial.missing))
        if trial.done == 0:
            raise RuntimeError(f"no frames reconstructed ({len(trial.missing)} missing from fal cache, "
                               f"{trial.failed} failed); see /status")
        write_summary(trial.frames_dir, {"trialId": trial.id})
        run = await asyncio.to_thread(pipeline.load_run, trial.frames_dir, CONVENTIONS)
        up = phone_up_in_sam_frame(trial, run)
        up_kw = {}
        if up is not None:
            up_kw = {"up_cam": up, "up_source": "phone_gravity"} if GRAVITY_MODE == "override" else {"up_check": up}
        try:
            height_m = payload.patient_height_cm / 100 if payload.patient_height_cm else None
            result = await asyncio.to_thread(lambda: pipeline.process(run, payload.events, height_m, **up_kw))
        except NotImplementedError as e:
            log.warning("geometry.py not implemented (%s); publishing raw camera-frame bundle", e)
            result = pipeline.raw_display(run)
        if any(r["metadata"].get("SYNTHETIC") for r in load_records(trial.frames_dir)):
            result["quality"]["SYNTHETIC"] = True
        frames = []
        for t, v, stem in zip(result["t_s"], result.get("vis_files", run.vis_files), result.get("stems", run.stems)):
            fr = {"t": round(t * 1000, 1), "fal_vis_url": f"frames/{v}" if v else None}
            src = trial.frames_dir / f"{stem}_src.jpg"
            if src.exists():
                with Image.open(src) as im:
                    sent_h = next((r["image_size"][1] for r in load_records(trial.frames_dir)
                                   if Path(r["frame"]).stem == stem), im.height)
                fr["src_url"], fr["src_scale"] = f"frames/{stem}_src.jpg", round(im.height / sent_h, 5)
            frames.append(fr)
        stats = analytics.compute(result, payload.events) if result["quality"].get("aligned") else None
        meta = await asyncio.to_thread(lambda: bundle.write_bundle(trial.dir, trial.id, result, events=payload.events,
                                                                   frames=frames, analytics=stats))
        if stats is not None:
            await asyncio.to_thread(report.write_report, trial.dir, meta, stats, result)
        trial.aligned = result["quality"]["aligned"]
        trial.state, trial.t_ready = "ready", time.time()
        log.info("trial %s ready in %.2fs after /end (%d frames)", trial.id, trial.t_ready - trial.t_end, len(run.t_s))
        await notify_ready(trial)
    except Exception as e:
        trial.state, trial.error = "failed", repr(e)
        log.exception("trial %s failed", trial.id)


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


@app.post("/replay/{trial_id}/end", status_code=202)
async def post_end(trial_id: str, payload: EndPayload) -> dict:
    trial = get_trial(trial_id)
    if trial.state != "receiving":
        raise HTTPException(409, f"trial {trial_id} already ended")
    trial.state, trial.t_end = "processing", time.time()
    (trial.dir / "end.json").write_text(payload.model_dump_json())
    asyncio.create_task(finalize(trial, payload))
    return {"status": "processing", "received": trial.received, "inflight": trial.received - trial.done - trial.failed,
            "missing_from_cache": trial.missing}


@app.get("/replay/{trial_id}/report")
async def report_page(trial_id: str) -> FileResponse:
    path = (TRIALS_DIR / trial_id / "report.html").resolve()
    if not path.is_relative_to(TRIALS_DIR.resolve()) or not path.is_file():
        raise HTTPException(404, "no report for this trial yet")
    return FileResponse(path)


@app.get("/replay/{trial_id}/status")
async def status(trial_id: str) -> dict:
    tr = get_trial(trial_id)
    return {
        "trialId": tr.id, "state": tr.state, "received": tr.received, "done": tr.done, "failed": tr.failed,
        "aligned": tr.aligned, "error": tr.error, "missing_from_cache": tr.missing,
        "fal_mode": "mock" if BACKEND else ("live" if LIVE else "cache-only"),
        "ready_after_end_s": round(tr.t_ready - tr.t_end, 2) if tr.t_ready and tr.t_end else None,
        "url": f"/replay/{tr.id}/" if tr.state == "ready" else None,
    }


@app.api_route("/replay/{trial_id}/{path:path}", methods=["GET", "HEAD"])
async def bundle_file(trial_id: str, path: str) -> FileResponse:
    base = (TRIALS_DIR / trial_id).resolve()
    target = (base / path).resolve()
    if not target.is_relative_to(base) or not target.is_file():
        raise HTTPException(404)
    return FileResponse(target)


app.mount("/viewer", StaticFiles(directory=REPLAY_ROOT / "viewer", html=True), name="viewer")
