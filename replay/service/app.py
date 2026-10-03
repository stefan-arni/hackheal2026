"""FastAPI service: ingest frames, call fal, post-process, serve bundles.

    uv run uvicorn service.app:app --port 8000                                          # real fal
    REPLAY_MOCK_FAL=data/fal_out/synthetic uv run uvicorn service.app:app --port 8000   # offline

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

from service import bundle, pipeline
from service.fal_client_wrap import REPLAY_ROOT, SamBodyClient
from service.runs import DEFAULT_CONVENTIONS, load_records, write_frame, write_summary

log = logging.getLogger("replay")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

TRIALS_DIR = Path(os.environ.get("REPLAY_TRIALS_DIR", REPLAY_ROOT / "data/trials"))
MOCK_RUN = os.environ.get("REPLAY_MOCK_FAL")
CONVENTIONS: dict | None = None  # None -> pipeline looks for data/conventions.json

if MOCK_RUN:
    from service import fal_mock

    mock_dir = Path(MOCK_RUN) if Path(MOCK_RUN).is_absolute() else REPLAY_ROOT / MOCK_RUN
    log.warning("fal is MOCKED from %s (%d frames)", mock_dir, fal_mock.install(mock_dir))
    conv_file = mock_dir / "conventions.json"
    CONVENTIONS = json.loads(conv_file.read_text()) if conv_file.exists() else DEFAULT_CONVENTIONS

app = FastAPI(title="Instant Replay")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

_sam: SamBodyClient | None = None


def sam() -> SamBodyClient:
    global _sam
    if _sam is None:
        _sam = SamBodyClient(concurrency=int(os.environ.get("REPLAY_FAL_CONCURRENCY", 8)))
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


async def run_frame(trial: Trial, stem: str, jpeg: bytes, mask: bytes | None, t_ms: float, extra: dict) -> None:
    try:
        with Image.open(io.BytesIO(jpeg)) as im:
            size = list(im.size)
        res = await sam().reconstruct(jpeg, mask)
        write_frame(
            trial.frames_dir, stem, t_ms=t_ms, image_size=size, response=res.response,
            latency_s=res.latency_s, ply=res.ply, visualization=res.visualization, extra=extra,
        )
        trial.done += 1
    except Exception:
        trial.failed += 1
        log.exception("trial %s frame %s failed", trial.id, stem)


@app.post("/replay/warmup")
async def warmup(jpeg: UploadFile = File(...)) -> dict:
    return {"latency_s": await sam().warmup(await jpeg.read())}


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
    patient_height_cm: float
    landmarks_2d: Any | None = None


async def finalize(trial: Trial, payload: EndPayload) -> None:
    try:
        await asyncio.gather(*trial.tasks)  # stragglers
        write_summary(trial.frames_dir, {"trialId": trial.id})
        run = await asyncio.to_thread(pipeline.load_run, trial.frames_dir, CONVENTIONS)
        try:
            result = await asyncio.to_thread(pipeline.process, run, payload.events, payload.patient_height_cm / 100)
        except NotImplementedError as e:
            log.warning("geometry.py not implemented (%s); publishing raw camera-frame bundle", e)
            result = pipeline.raw_display(run)
        if any(r["metadata"].get("SYNTHETIC") for r in load_records(trial.frames_dir)):
            result["quality"]["SYNTHETIC"] = True
        frames = [{"t": round(t * 1000, 1), "fal_vis_url": f"frames/{v}" if v else None}
                  for t, v in zip(run.t_s, run.vis_files)]
        await asyncio.to_thread(bundle.write_bundle, trial.dir, trial.id, result, events=payload.events, frames=frames)
        trial.aligned = result["quality"]["aligned"]
        trial.state, trial.t_ready = "ready", time.time()
        log.info("trial %s ready in %.2fs after /end (%d frames)", trial.id, trial.t_ready - trial.t_end, len(run.t_s))
        await notify_ready(trial)
    except Exception as e:
        trial.state, trial.error = "failed", repr(e)
        log.exception("trial %s failed", trial.id)


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
    return {"status": "processing", "received": trial.received, "inflight": trial.received - trial.done - trial.failed}


@app.get("/replay/{trial_id}/status")
async def status(trial_id: str) -> dict:
    tr = get_trial(trial_id)
    return {
        "trialId": tr.id, "state": tr.state, "received": tr.received, "done": tr.done, "failed": tr.failed,
        "aligned": tr.aligned, "error": tr.error,
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
