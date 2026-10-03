"""Pose server: receives video frames from the iPhone app, runs MediaPipe
full-body pose detection, and sends back landmarks, spine angles,
single-leg balance status, duck events and BESS balance-test scores as JSON.

Run:
    python server.py                    # ws://0.0.0.0:8765
    python server.py --model lite       # faster; "heavy" is more accurate
    python server.py --no-balance       # skip single-leg balance detection
    python server.py --no-duck          # skip duck detection

Frame/ping protocol: see ws_server.py. Also:
  {"type": "recalibrate"}  ->  reset balance stats and the duck count
  {"type": "bess_start", "stance": "double" | "tandem" | "single", "nondominant": "left"}
  {"type": "bess_cancel"} / {"type": "bess_reset"} / {"type": "bess_status"}
  {"type": "bess_mark", "error": "hands_off_hips"}  (manual error during a test)
  {"type": "sway_start", "test": "quiet" | "tandem" | "romberg_eo" | "romberg_ec", "duration": 30}
  {"type": "sway_cancel"} / {"type": "sway_reset"} / {"type": "sway_status"}
  {"type": "record_start", "name": "optional"} / {"type": "record_stop"}  (save the session)
  Frames may carry a LiDAR / TrueDepth depth map (see depth.py) for sway in real cm.

Reply, one per processed frame:
  {"type": "pose", "frame_id": 42, "client_timestamp_ms": ..., "detected": true,
   "landmarks": [33 x {name, x, y, z, visibility, presence}],
   "world_landmarks": [33 x {name, x, y, z, visibility}],
   "spine": {trunk_angle_deg, inclination_deg, flexion_deg, lateral_deg, ...},
   "balance": {"state": "balancing", "lifted_foot": "right", "standing_foot": "left",
               "balance_time_s": 4.1, "foot_heights": {"left": 0.0, "right": 0.12}, ...},
   "duck": {"ducking": false, "count": 2, "drop": 0.03, "threshold": 0.2, "mode": "world"},
   "inference_ms": 18.2, "dropped_frames": 0}

Plus, when something changes, an extra message right after that frame:
  {"type": "status", "kind": "balance_started", "lifted_foot": "right", "standing_foot": "left"}
  {"type": "alert", "kind": "foot_touchdown", "foot": "right", "held_s": 6.3,
   "touch_count": 1, "best_hold_s": 6.3}
  {"type": "event", "kind": "duck", "count": 3, "drop": 0.27, "mode": "world"}   (-> quack!)
  {"type": "event", "kind": "bess_error", "error": "hands_off_hips", "t": 7.2, "counted": true, ...}
  {"type": "result", "kind": "bess_done", "stance": "tandem", "errors": 3, "by_type": {...},
   "session": {"scores": {"double": 1, "tandem": 3, "single": null}, "total": 4, "complete": false}}

Eye tracking is a separate program: eye_server.py (port 8766).
"""

from __future__ import annotations

import argparse
import asyncio
import logging

from ws_server import decode_message, make_handler, serve  # noqa: F401  (re-exported)

log = logging.getLogger("pose-server")


def main():
    p = argparse.ArgumentParser(description="MediaPipe pose WebSocket server")
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=8765)
    p.add_argument("--model", choices=["lite", "full", "heavy"], default="full")
    p.add_argument("--min-visibility", type=float, default=0.5)
    p.add_argument("--no-balance", action="store_true", help="turn off single-leg balance detection")
    p.add_argument("--balance-mode", choices=["world", "image"], default="world",
                   help="measure foot height in 3D meters (world) or 2D leg-lengths (image)")
    p.add_argument("--lift-threshold", type=float,
                   help="foot counts as raised above this (default 0.08 m / 0.10 leg-lengths)")
    p.add_argument("--touch-threshold", type=float,
                   help="raised foot counts as touching down below this (default 0.03 m / 0.04)")
    p.add_argument("--no-floor-calibration", action="store_true",
                   help="balance: measure the raised foot against the other foot instead of a learned floor")
    p.add_argument("--no-duck", action="store_true", help="turn off duck detection")
    p.add_argument("--duck-drop", type=float, default=0.20,
                   help="full body in frame: head drop that counts as a duck, as a fraction "
                        "of standing height (default 0.20)")
    p.add_argument("--duck-drop-image", type=float, default=0.8,
                   help="feet out of frame: nose drop that counts as a duck, in shoulder-widths "
                        "(default 0.8)")
    p.add_argument("--no-bess", action="store_true", help="turn off the BESS balance test")
    p.add_argument("--bess-eyes", choices=["off", "auto", "manual"], default="manual",
                   help="BESS eyes-open errors: manual (default) = the examiner marks them in the "
                        "app, auto = face check (unreliable at full-body distance), off = not scored")
    p.add_argument("--bess-countdown", type=float, default=1.0,
                   help="seconds before scoring starts; the last second records the start "
                        "position (default 1, i.e. no get-ready countdown)")
    p.add_argument("--no-sway", action="store_true",
                   help="turn off the sway tests (quiet stance, tandem, Romberg)")
    p.add_argument("--sway-duration", type=float, default=30.0,
                   help="default length of a sway test in seconds (the app can override)")
    p.add_argument("--record", action="store_true",
                   help="save every session to --record-dir (the app can also start/stop it)")
    p.add_argument("--record-dir", default="recordings", help="where recordings go")
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    from balance import BalanceConfig
    from bess import BessConfig
    from duck import DuckConfig
    from sway import SwayConfig
    balance_cfg = duck_cfg = bess_cfg = sway_cfg = None
    if not args.no_balance:
        try:
            balance_cfg = BalanceConfig(mode=args.balance_mode, lift_threshold=args.lift_threshold,
                                        touch_threshold=args.touch_threshold,
                                        calibrate=not args.no_floor_calibration)
        except ValueError as e:
            p.error(str(e))
        log.info("balance detection on (%s mode, lift %.3f, touch %.3f, floor calibration %s)",
                 balance_cfg.mode, balance_cfg.lift_threshold, balance_cfg.touch_threshold,
                 "on" if balance_cfg.calibrate else "off")
    if not args.no_duck:
        duck_cfg = DuckConfig(world_drop=args.duck_drop, image_drop=args.duck_drop_image)
        log.info("duck detection on (drop %.0f%% of height, or %.1f shoulder-widths)",
                 duck_cfg.world_drop * 100, duck_cfg.image_drop)

    if not args.no_bess:
        bess_cfg = BessConfig(countdown_s=args.bess_countdown)
        log.info("BESS on (eyes: %s)", args.bess_eyes)
    if not args.no_sway:
        sway_cfg = SwayConfig(duration_s=args.sway_duration)
        log.info("sway tests on (%.0f s; quiet, tandem, romberg_eo, romberg_ec)", sway_cfg.duration_s)

    from pose_analyzer import PoseAnalyzer, ensure_model
    from pose_pipeline import PosePipeline, pose_events
    ensure_model(args.model)  # download once up front, not per connection
    if bess_cfg and args.bess_eyes == "auto":
        from mp_models import ensure_face_model
        ensure_face_model()

    def factory():
        pose = PoseAnalyzer(model=args.model, min_visibility=args.min_visibility)
        if balance_cfg is None and duck_cfg is None and bess_cfg is None and sway_cfg is None:
            return pose
        # fresh config objects per connection, so each client has its own state
        return PosePipeline(pose,
                            BalanceConfig(**vars(balance_cfg)) if balance_cfg else None,
                            DuckConfig(**vars(duck_cfg)) if duck_cfg else None,
                            BessConfig(**vars(bess_cfg)) if bess_cfg else None,
                            bess_eyes=args.bess_eyes,
                            sway=SwayConfig(**vars(sway_cfg)) if sway_cfg else None)

    try:
        asyncio.run(serve(args.host, args.port, factory, result_type="pose",
                          events_fn=pose_events, name="pose server",
                          record_dir=args.record_dir, record_all=args.record))
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
