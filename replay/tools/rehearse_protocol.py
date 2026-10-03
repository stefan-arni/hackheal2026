# /// script
# requires-python = ">=3.10"
# dependencies = ["websockets>=12", "numpy>=1.26", "httpx>=0.27"]
# ///
"""Rehearse the demo protocol: stream a video into posecam in real time, like the phone does.

    ./run_demo.sh --mock            # (other terminal) replay service + posecam
    uv run tools/rehearse_protocol.py --session <sessionId> [--shots data/reports/rehearsal]

Protocol: 3 back-to-back BESS trials (double, tandem, single leg). Per trial: bess_start, the
5 s countdown and the 10 s trial (posecam must run with --bess-duration 10), then a ~15 s gap.
Frames go to ws://localhost:8765 as JSON {"type":"frame","image":<base64 JPEG>,"timestamp_ms"}
at the clip's frame rate, timestamp_ms = this laptop's clock (stands in for the phone clock).

Video: data/protocol.mov if it exists (a recording of the whole protocol; bess_start at
--starts). Otherwise three 10 s segments of data/IMG_9691.mov that best match each stance,
picked from the scout landmarks (data/scout/IMG_9691/timeline.npz): ankle spacing / hip width
and ankle height difference per frame; the single-leg window must contain the clip's final
step-down. Between trials the clip's longest planted stretch loops (ping-pong) so the live
skeleton keeps moving.

Live camera (Plan B, no iOS code): --camera streams a macOS camera instead of the clip, e.g. an
iPhone as Continuity Camera. --list-cameras shows the devices; --camera auto prefers an iPhone.
--rotate 90 / 270 for a phone mounted in portrait; each trial starts on Enter (--trigger key) or
on a timer (--trigger timer: first trial after --first-delay, then --gap after each trial ends).

    uv run tools/rehearse_protocol.py --list-cameras
    uv run tools/rehearse_protocol.py --session <id> --camera auto --rotate 90 --trigger key

Afterwards it waits for every trial's final replay (full or deadline, plus straggler updates up
to --settle s) and prints the measured timeline from /session/{id}. With --shots it saves
headless-Chrome screenshots of the dashboard at each stage (trial running, error replay,
final replay, session report).
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import json
import re
import subprocess
import sys
import time
from pathlib import Path

import httpx
import numpy as np
import websockets

REPLAY_ROOT = Path(__file__).resolve().parents[1]
STANCES = ("double", "tandem", "single")


# ------------------------------------------------------------------ segment choice

def stance_masks(timeline: Path) -> tuple[np.ndarray, dict[str, np.ndarray], float | None]:
    z = np.load(timeline)
    lm, t = z["lm"], z["t_ms"] / 1000.0  # lm: full-res px (x, y, z, vis)
    r = np.abs(lm[:, 27, 0] - lm[:, 28, 0]) / np.maximum(np.abs(lm[:, 23, 0] - lm[:, 24, 0]), 1)
    ady = np.abs(lm[:, 27, 1] - lm[:, 28, 1])  # ankle height difference, px
    masks = {"double": (r > 0.3) & (r < 1.2) & (ady < 25),     # feet side by side, both down
             "tandem": (r < 0.35) & (ady > 20) & (ady < 90),   # one foot in front of the other
             "single": ady > 90}                               # one foot clearly lifted
    downs = [t[i] for i in range(len(t)) if ady[i] < 30 and (ady[(t > t[i] - 1) & (t < t[i])] > 90).any()]
    return t, masks, (max(downs) if downs else None)


def pick_segments(timeline: Path, countdown: float, duration: float) -> dict[str, float]:
    """Trial start (clip s) per stance: the window with most frames matching the stance."""
    t, masks, step_down = stance_masks(timeline)
    picks = {}
    for st in STANCES:
        best = None
        for t0 in np.arange(countdown, t[-1] - duration, 0.25):
            if st == "single" and step_down is not None and not (t0 + 1 < step_down < t0 + duration):
                continue
            s = masks[st][(t >= t0) & (t < t0 + duration)].mean()
            if best is None or s > best[0]:
                best = (s, float(t0))
        picks[st] = best[1]
        print(f"  {st:6s}: trial {best[1]:.2f}-{best[1] + duration:.2f} s of the clip "
              f"({100 * best[0]:.0f}% of frames match the stance; countdown from {best[1] - countdown:.2f} s)")
    if step_down is not None:
        print(f"  (single leg includes the step-down at {step_down:.2f} s)")
    return picks


APP_VF = "scale=-2:640,transpose=2"  # --as-app: like the iPhone app, 640 px long side, sensor landscape


def decode(clip: Path, start: float, length: float, width: int, vf: str | None = None) -> list[bytes]:
    """JPEG frames of clip[start, start+length] at the clip's rate, rotation applied, `width` px wide."""
    out = subprocess.run(["ffmpeg", "-v", "error", "-ss", f"{start:.3f}", "-t", f"{length:.3f}", "-i", str(clip),
                          "-vf", vf or f"scale={width}:-2", "-c:v", "mjpeg", "-q:v", "4", "-f", "image2pipe", "-"],
                         capture_output=True, check=True).stdout
    frames, i = [], 0
    while True:
        a = out.find(b"\xff\xd8", i)
        b = out.find(b"\xff\xd9", a)
        if a < 0 or b < 0:
            return frames
        frames.append(out[a:b + 2])
        i = b + 2


# ------------------------------------------------------------------ live camera (macOS)

def list_cameras() -> list[tuple[int, str]]:
    """AVFoundation video devices as (index, name), via ffmpeg."""
    out = subprocess.run(["ffmpeg", "-hide_banner", "-f", "avfoundation", "-list_devices", "true", "-i", ""],
                         capture_output=True, text=True).stderr
    devs, video = [], False
    for line in out.splitlines():
        if "AVFoundation video devices" in line:
            video = True
        elif "AVFoundation audio devices" in line:
            video = False
        elif video and (m := re.search(r"\[(\d+)\] (.+)$", line)):
            devs.append((int(m.group(1)), m.group(2).strip()))
    return devs


def pick_camera(spec: str) -> tuple[int, str]:
    devs = [d for d in list_cameras() if not d[1].startswith("Capture screen")]
    if not devs:
        raise SystemExit("no camera found (System Settings > Privacy & Security > Camera: allow your terminal)")
    if spec == "auto":  # an iPhone (Continuity Camera) if present, never its Desk View
        phones = [d for d in devs if "iphone" in d[1].lower() and "desk view" not in d[1].lower()]
        return (phones or devs)[0]
    if spec.isdigit():
        return next((d for d in devs if d[0] == int(spec)), None) or sys.exit(f"no camera with index {spec}")
    return next((d for d in devs if spec.lower() in d[1].lower()), None) or sys.exit(f"no camera matching {spec!r}")


ROTATE = {0: [], 90: ["transpose=1"], 180: ["hflip", "vflip"], 270: ["transpose=2"]}


class Camera:
    """ffmpeg avfoundation -> MJPEG on a pipe; .latest() is the newest frame (older ones are dropped)."""

    def __init__(self, index: int, rotate: int, width: int, fps: int = 30):
        vf = ",".join(ROTATE[rotate] + [f"scale={width}:-2"])
        self.cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-f", "avfoundation", "-framerate", str(fps), "-pixel_format", "nv12",
                    "-i", f"{index}:none", "-vf", vf, "-c:v", "mjpeg", "-q:v", "4", "-f", "image2pipe", "-"]
        self.frame: bytes | None = None
        self.seq = 0
        self.proc = None

    async def run(self) -> None:
        self.proc = await asyncio.create_subprocess_exec(*self.cmd, stdout=asyncio.subprocess.PIPE)
        buf = b""
        while chunk := await self.proc.stdout.read(1 << 16):
            buf += chunk
            while (a := buf.find(b"\xff\xd8")) >= 0 and (b := buf.find(b"\xff\xd9", a)) >= 0:
                self.frame, self.seq = buf[a:b + 2], self.seq + 1
                buf = buf[b + 2:]
        raise RuntimeError("camera stream ended (camera permission? device busy?)")

    def stop(self) -> None:
        if self.proc and self.proc.returncode is None:
            self.proc.kill()


async def stream_camera(ws, cam: Camera, counter: dict) -> None:
    """Send every new camera frame to posecam with the laptop's clock as timestamp_ms."""
    last = 0
    while True:
        if cam.seq != last and cam.frame is not None:
            last = cam.seq
            counter["sent"] += 1
            await ws.send(json.dumps({"type": "frame", "image": base64.b64encode(cam.frame).decode(),
                                      "frame_id": counter["sent"], "timestamp_ms": round(time.time() * 1000, 1)}))
        await asyncio.sleep(0.005)


async def run_camera_protocol(ws, a, t0: float, trial_end: asyncio.Queue) -> int:
    idx, name = pick_camera(a.camera)
    print(f"camera [{idx}] {name}, rotate {a.rotate}°, {a.width} px wide")
    cam = Camera(idx, a.rotate, a.width)
    counter = {"sent": 0}
    tasks = [asyncio.create_task(cam.run()), asyncio.create_task(stream_camera(ws, cam, counter))]
    loop = asyncio.get_running_loop()
    try:
        for k, st in enumerate(STANCES):
            if a.trigger == "key":
                await loop.run_in_executor(None, input, f"\n>>> {st}: patient ready? press Enter to start ")
            else:
                wait = a.first_delay if k == 0 else a.gap
                print(f"  next: {st} in {wait:.0f} s")
                await asyncio.sleep(wait)
            if any(t.done() for t in tasks):
                for t in tasks:
                    if t.done():
                        t.result()  # raises the camera error
            print(f"[{time.time() - t0:6.1f}s] bess_start {st} ({counter['sent']} frames so far)")
            await ws.send(json.dumps({"type": "bess_start", "stance": st, "nondominant": a.nondominant}))
            try:  # bess_done / failed / cancelled from posecam
                await asyncio.wait_for(trial_end.get(), timeout=a.countdown + a.duration + 20)
            except asyncio.TimeoutError:
                print(f"  {st}: no result from posecam (whole body in frame?)")
        await asyncio.sleep(2.0)
    finally:
        for t in tasks:
            t.cancel()
        cam.stop()
    return counter["sent"]


# ------------------------------------------------------------------ streaming

class Streamer:
    def __init__(self, ws, fps: float, as_app: bool = False):
        self.ws, self.period, self.sent, self.fid = ws, 1.0 / fps, 0, 0
        self.next_t = time.monotonic()
        self.as_app = as_app  # sideways frames + "rotate": 90 + device-uptime timestamps, like the iPhone app

    async def frames(self, jpegs: list[bytes]) -> None:
        for j in jpegs:
            self.next_t += self.period
            await asyncio.sleep(max(0.0, self.next_t - time.monotonic()))
            self.fid += 1
            msg = {"type": "frame", "image": base64.b64encode(j).decode(), "frame_id": self.fid,
                   "timestamp_ms": round((time.monotonic() if self.as_app else time.time()) * 1000, 1)}
            if self.as_app:
                msg.update(rotate=90, camera="lidar")
            await self.ws.send(json.dumps(msg))
            self.sent += 1

    async def idle(self, loop: list[bytes], seconds: float) -> None:
        pingpong = loop + loop[::-1]
        n = int(seconds / self.period)
        await self.frames([pingpong[k % len(pingpong)] for k in range(n)])


async def listen(ws, log: list[dict], t0: float, trial_end: asyncio.Queue | None = None) -> None:
    async for raw in ws:
        if isinstance(raw, bytes):
            continue
        m = json.loads(raw)
        if m.get("type") in ("event", "status", "result", "ack", "error") and m.get("kind") != "duck":
            m["_wall"] = time.time()
            log.append(m)
            k = m.get("kind") or m.get("command") or m.get("error")
            extra = {key: m[key] for key in ("stance", "errors", "foot", "label", "reason") if key in m}
            print(f"  [{time.time() - t0:6.1f}s] posecam {m['type']}: {k} {extra if extra else ''}")
            if trial_end is not None and m.get("kind") in ("bess_done", "bess_failed", "bess_cancelled"):
                trial_end.put_nowait(m)


class Shots:
    def __init__(self, out: Path | None, base: str, sid: str):
        self.out, self.base, self.sid, self.n = out, base, sid, 0
        self.procs: list[subprocess.Popen] = []
        if out:
            out.mkdir(parents=True, exist_ok=True)

    def take(self, name: str, path: str | None = None, wait_ms: int = 6000) -> None:
        if not self.out:
            return
        self.n += 1
        f = self.out / f"{self.n:02d}_{name}.png"
        url = self.base + (path or f"/dashboard/{self.sid}")
        self.procs.append(subprocess.Popen(  # DevTools screenshot: a live page never finishes loading
            ["node", str(REPLAY_ROOT / "tools/screenshot.mjs"), url, str(f), str(wait_ms), "1600x1000"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL))
        print(f"  screenshot → {f}")

    def wait(self) -> None:
        for p in self.procs:
            try:
                p.wait(timeout=60)
            except subprocess.TimeoutExpired:
                p.kill()


async def watch_stages(http: httpx.AsyncClient, sid: str, shots: Shots, stop: asyncio.Event) -> None:
    """Screenshot the dashboard the first time each trial reaches the error / final stage."""
    seen = set()
    while not stop.is_set():
        try:
            info = (await http.get(f"/session/{sid}")).json()
            for tr in info.get("trials", []):
                tl = tr.get("timeline") or {}
                for key, label in (("error_ready_s", "error"), ("final_kind", "final")):
                    if tl.get(key) is not None and (tr["trialId"], label) not in seen:
                        seen.add((tr["trialId"], label))
                        shots.take(f"{tr.get('stance') or tr['trialId']}_{label}_{tl.get('final_kind', '')}".rstrip("_"))
        except (httpx.HTTPError, ValueError):
            pass
        await asyncio.sleep(1.0)


async def main_async(a) -> None:
    if a.list_cameras:
        for i, n in list_cameras():
            print(f"  [{i}] {n}")
        return
    if a.camera:
        return await session_run(a, None, None, None, None, None)
    clip = a.clip or (REPLAY_ROOT / "data/protocol.mov" if (REPLAY_ROOT / "data/protocol.mov").exists()
                      else REPLAY_ROOT / "data/IMG_9691.mov")
    trial_len = a.countdown + a.duration
    print(f"clip: {clip}")
    if clip.name == "protocol.mov":
        starts = {st: float(x) for st, x in (s.split(":") for s in a.starts.split(","))}
        segments = None
    else:
        segments = pick_segments(REPLAY_ROOT / f"data/scout/{clip.stem}/timeline.npz", a.countdown, a.duration)
        rep = json.loads((REPLAY_ROOT / f"data/scout/{clip.stem}/report.json").read_text())
        lp = rep["longest_planted"]
    print("decoding…")
    if segments:
        vf = APP_VF if a.as_app else None
        seg_frames = {st: decode(clip, t0 - a.countdown, trial_len + 0.5, a.width, vf) for st, t0 in segments.items()}
        idle = decode(clip, lp["start_s"], lp["duration_s"], a.width, vf)
        fps = max(len(f) for f in seg_frames.values()) / (trial_len + 0.5)
    else:
        whole = decode(clip, 0, 1e6, a.width)
        fps = a.fps
    print(f"  {fps:.1f} fps, {a.width} px wide")
    if a.dry_run:
        return
    await session_run(a, clip, segments, seg_frames if segments else None, idle if segments else whole, fps,
                      starts=None if segments else starts)


async def session_run(a, clip, segments, seg_frames, idle, fps, starts=None) -> None:
    trial_len = a.countdown + a.duration
    whole = idle
    http = httpx.AsyncClient(base_url=a.replay, timeout=10)
    sid = a.session or (await http.post("/session", json={"name": "rehearsal"})).json()["sessionId"]
    print(f"session {sid}\n  dashboard {a.replay}/dashboard/{sid}\n  report    {a.replay}/session/{sid}/report")
    shots = Shots(a.shots, a.replay, sid)
    stop = asyncio.Event()
    watcher = asyncio.create_task(watch_stages(http, sid, shots, stop)) if a.shots else None

    log: list[dict] = []
    t0 = time.time()
    trial_end: asyncio.Queue = asyncio.Queue()
    async with websockets.connect(a.ws, max_size=None) as ws:
        lt = asyncio.create_task(listen(ws, log, t0, trial_end))
        s = Streamer(ws, fps or 30.0, as_app=a.as_app)
        if a.camera:
            s.sent = await run_camera_protocol(ws, a, t0, trial_end)
        elif segments:
            await s.idle(idle, 3.0)  # posecam sees a person before the first start
            for k, st in enumerate(STANCES):
                print(f"[{time.time() - t0:6.1f}s] bess_start {st}")
                await ws.send(json.dumps({"type": "bess_start", "stance": st, "nondominant": a.nondominant}))
                if k == 0:
                    asyncio.get_running_loop().call_later(a.countdown + 4, shots.take, "double_running", None, 4000)
                await s.frames(seg_frames[st])
                if k < len(STANCES) - 1:
                    await s.idle(idle, a.gap)
            await s.idle(idle, 4.0)  # let the last bess_done go out
        else:
            sched = sorted(starts.items(), key=lambda kv: kv[1])
            pos = 0
            for st, t_start in sched:
                upto = int(t_start * fps)
                await s.frames(whole[pos:upto])
                pos = upto
                print(f"[{time.time() - t0:6.1f}s] bess_start {st}")
                await ws.send(json.dumps({"type": "bess_start", "stance": st, "nondominant": a.nondominant}))
            await s.frames(whole[pos:])
        lt.cancel()
    t_stream = time.time()
    print(f"streamed {s.sent} frames in {t_stream - t0:.1f} s")

    # wait for the replays
    deadline = time.time() + a.settle
    while time.time() < deadline:
        info = (await http.get(f"/session/{sid}")).json()
        trs = info.get("trials", [])
        if trs and all((tr.get("timeline") or {}).get("final_kind") for tr in trs) \
                and all(tr.get("queued", 0) == 0 and tr.get("in_flight", 0) == 0 for tr in trs):
            break
        await asyncio.sleep(2)
    shots.take("dashboard_end", wait_ms=8000)
    shots.take("session_report", f"/session/{sid}/report", wait_ms=3000)
    stop.set()
    if watcher:
        await watcher
    shots.wait()

    info = (await http.get(f"/session/{sid}")).json()
    rows = []
    for tr in info["trials"]:
        tl = tr.get("timeline") or {}
        rows.append({"trial": tr["trialId"], "stance": tr.get("stance"), "received": tr.get("received"),
                     "bursts": tr.get("bursts"), "done": tr.get("done"), "error_ready_s": tl.get("error_ready_s"),
                     "final_kind": tl.get("final_kind"), "final_ready_s": tl.get("final_ready_s"),
                     "final_frames": tl.get("final_frames"), "updates_s": tl.get("updates_s"),
                     "stragglers": tr.get("stragglers"), "fal_mode": tr.get("fal_mode")})
    print("\nmeasured timeline (seconds after each trial's end):")
    print(f"{'stance':7s} {'frames':>6s} {'burst':>5s} {'error':>7s} {'final':>16s} {'frames':>8s}  updates")
    for r in rows:
        print(f"{r['stance'] or '?':7s} {r['received'] or 0:6d} {r['bursts'] or 0:5d} "
              f"{'-' if r['error_ready_s'] is None else r['error_ready_s']:>7} "
              f"{(r['final_kind'] or '-') + ' ' + str(r['final_ready_s'] or '-'):>16s} {r['final_frames'] or '-':>8s}  "
              f"{r['updates_s'] or ''}")
    total = sum(r["received"] or 0 for r in rows)
    print(f"total frames sent to the replay: {total}  →  live fal ≈ ${total * a.price:.2f} at ${a.price}/call")
    stats = fal_stats([r["trial"] for r in rows])
    if stats:
        print(f"fal: {stats['calls']} calls, latency p50 {stats['latency_p50_s']} s / p95 {stats['latency_p95_s']} s "
              f"(min {stats['latency_min_s']} s); in flight {stats['in_flight_mean']} on average (max "
              f"{stats['in_flight_max']}); {stats['throughput_fps']} frames/s → fal ran ≈ {stats['fal_parallel_est']} at once")
    out = {"session": sid, "clip": str(clip), "camera": a.camera, "segments": segments, "fal": stats, "trials": rows, "total_frames": total,
           "price_per_call": a.price, "posecam_messages": [{k: v for k, v in m.items()} for m in log]}
    if a.shots:
        (a.shots / "timeline.json").write_text(json.dumps(out, indent=1, default=str))
        print(f"saved {a.shots / 'timeline.json'}")
    await http.aclose()


def fal_stats(trial_ids: list[str]) -> dict | None:
    """Latency and parallelism of the fal calls behind these trials (frame records of the service)."""
    calls = []
    for tid in trial_ids:
        for p in (REPLAY_ROOT / "data/trials" / tid / "frames").glob("t*.json"):
            r = json.loads(p.read_text())
            if r.get("call_start_wall") and not r.get("from_cache"):
                calls.append((r["call_start_wall"], r["call_end_wall"], r["latency_s"]))
    if len(calls) < 3:
        return None
    lat = np.array([c[2] for c in calls])
    starts, ends = np.array([c[0] for c in calls]), np.array([c[1] for c in calls])
    grid = np.arange(starts.min(), ends.max(), 0.1)
    inflight = ((grid[:, None] >= starts[None]) & (grid[:, None] < ends[None])).sum(axis=1)
    busy = inflight > 0
    thr = len(calls) / (busy.sum() * 0.1)  # frames per busy second
    return {"calls": len(calls), "latency_p50_s": round(float(np.percentile(lat, 50)), 2),
            "latency_p95_s": round(float(np.percentile(lat, 95)), 2), "latency_min_s": round(float(lat.min()), 2),
            "in_flight_mean": round(float(inflight[busy].mean()), 2), "in_flight_max": int(inflight.max()),
            "throughput_fps": round(float(thr), 3),
            # Little's law with the fastest call as fal's own processing time: how many it really ran at once
            "fal_parallel_est": round(float(thr * lat.min()), 2)}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--clip", type=Path)
    p.add_argument("--ws", default="ws://localhost:8765")
    p.add_argument("--replay", default="http://localhost:8017")
    p.add_argument("--session", help="existing session ID (posecam's --session-id); default: create one")
    p.add_argument("--countdown", type=float, default=5.0, help="posecam's BESS countdown (BessConfig.countdown_s)")
    p.add_argument("--duration", type=float, default=10.0, help="posecam's --bess-duration")
    p.add_argument("--gap", type=float, default=15.0, help="seconds between a trial's end and the next bess_start")
    p.add_argument("--nondominant", default="left")
    p.add_argument("--starts", default="double:0,tandem:30,single:60", help="protocol.mov only: bess_start times (s)")
    p.add_argument("--fps", type=float, default=30.0, help="protocol.mov only")
    p.add_argument("--width", type=int, default=720, help="frame width sent (iPhone-like 720x1280)")
    p.add_argument("--settle", type=float, default=240.0, help="max wait for final replays + stragglers")
    p.add_argument("--price", type=float, default=0.018)
    p.add_argument("--shots", type=Path, help="save dashboard screenshots + timeline.json here")
    p.add_argument("--dry-run", action="store_true", help="only print the chosen segments")
    p.add_argument("--as-app", action="store_true",
                   help="impersonate the iPhone app: 640x480-style sideways JPEGs with rotate 90, uptime timestamps")
    p.add_argument("--list-cameras", action="store_true", help="list macOS camera devices and exit")
    p.add_argument("--camera", help="live camera instead of the clip: auto (iPhone if present), an index, or a name")
    p.add_argument("--rotate", type=int, choices=[0, 90, 180, 270], default=0, help="camera: rotate clockwise")
    p.add_argument("--trigger", choices=["key", "timer"], default="key", help="camera: start trials on Enter or a timer")
    p.add_argument("--first-delay", type=float, default=5.0, help="camera + timer: seconds before the first trial")
    asyncio.run(main_async(p.parse_args()))


if __name__ == "__main__":
    main()
