"""Per-trial analytics from a pipeline result (written to the bundle as analytics.json).

Floor frame: y up, x = patient's left (image right for a patient facing the camera),
z = toward the camera (forward). Forward/back (z) numbers carry "ap_estimated": true when the
noise-floor gate failed (depth jitter >= 1 cm). Timestamps may be non-uniform (bursts):
durations use each frame's share of time (half-way to its neighbours).
"""

from __future__ import annotations

from typing import Any

import numpy as np

LIFT_MIN_M = 0.03  # stance check: a foot is "up" above max(3 cm, LIFT_NOISE_K × vertical noise), down below half
LIFT_NOISE_K = 4.0  # (SAM floor-frame heights have no perspective confound, unlike MediaPipe world heights)
QUALITY = {  # (good, fair) thresholds
    "reprojection_median_px": (20.0, 35.0),
    "noise_x_cm": (1.0, 2.0),
    "dropped_frac": (0.05, 0.15),
    "ground_anchor_median_cm": (1.5, 3.0),
}


def _frame_dt(t: np.ndarray) -> np.ndarray:
    """Each frame's share of the timeline (s): half the gap to each neighbour."""
    if len(t) < 2:
        return np.ones(len(t))
    edges = np.concatenate([[t[0]], (t[1:] + t[:-1]) / 2, [t[-1]]])
    return np.diff(edges)


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    m = np.concatenate([[False], mask, [False]])
    d = np.flatnonzero(np.diff(m.astype(int)))
    return list(zip(d[::2], d[1::2]))


def _hysteresis(x: np.ndarray, up: float, down: float) -> np.ndarray:
    out, on = np.zeros(len(x), bool), False
    for i, v in enumerate(x):
        if not np.isnan(v):
            on = v > (down if on else up)
        out[i] = on
    return out


def _kp(names: list[str], *wanted: str) -> list[int]:
    return [i for i, n in enumerate(names) if any(w in n for w in wanted)]


UNCERTAIN_K = 3.0  # excursions smaller than this × the side-to-side noise floor are "uncertain"
EVENT_LABELS = {"toe_touch": "Toe touch", "step": "Step", "foot_down": "Foot down", "foot_lift": "Foot lift",
                "foot_touchdown": "Foot down", "touchdown": "Foot down"}


def compute(result: dict[str, Any], events: list[dict] | None = None,
            expected_stance: str | None = None) -> dict[str, Any]:
    """expected_stance: optional "double", "tandem", "single_left" or "single_right" to check."""
    t = np.asarray(result["t_s"], float)
    nf_x = float((result.get("noise_floor") or {}).get("x", 0.0))
    small = UNCERTAIN_K * nf_x  # meters
    dt = _frame_dt(t)
    q = result.get("quality", {})
    ap_est = not q.get("ap_real", False)
    out: dict[str, Any] = {"frames": int(len(t)), "duration_s": round(float(t[-1] - t[0]), 2),
                           "t_start_ms": round(float(t[0]) * 1000, 1), "ap_estimated": ap_est}

    # --- margin of stability (COM floor projection vs base of support)
    m = np.array([np.nan if x is None else x for x in result["margin"]], float)
    ok = ~np.isnan(m)
    if ok.any():
        i_min = int(np.nanargmin(m))
        outside = ok & (m < 0)
        episodes = [{"start_ms": round(float(t[a]) * 1000, 1), "end_ms": round(float(t[b - 1]) * 1000, 1),
                     "duration_s": round(float(dt[a:b].sum()), 2), "min_margin_cm": round(float(m[a:b].min()) * 100, 2),
                     "uncertain": bool(abs(float(m[a:b].min())) < small)}
                    for a, b in _runs(outside)]
        out["margin"] = {"min_cm": round(float(m[i_min]) * 100, 2), "min_at_ms": round(float(t[i_min]) * 1000, 1),
                         "median_cm": round(float(np.nanmedian(m)) * 100, 2),
                         "time_outside_bos_s": round(float(dt[outside].sum()), 2),
                         "fraction_outside": round(float(dt[outside].sum() / dt[ok].sum()), 3),
                         "outside_episodes": episodes, "frames_without_bos": int((~ok).sum()),
                         "uncertain_below_cm": round(small * 100, 2)}

    # --- sway of the COM (x side-to-side; z forward/back, estimated if the gate failed)
    com = np.asarray(result["com"], float)
    rate = lambda d: float(np.sum(np.abs(np.diff(d)))) / max(float(t[-1] - t[0]), 1e-9)  # noqa: E731
    for axis, name in ((0, "ml_sway"), (2, "ap_sway")):
        x = com[:, axis]
        xc = x - np.average(x, weights=dt)
        out[name] = {"rms_cm": round(float(np.sqrt(np.average(xc**2, weights=dt))) * 100, 2),
                     "range_cm": round(float(np.ptp(x)) * 100, 2),
                     "path_length_cm": round(float(np.sum(np.abs(np.diff(x)))) * 100, 1),
                     "mean_speed_cm_s": round(rate(x) * 100, 2)}
        if axis == 2:
            out[name]["estimated"] = ap_est
    out["ml_sway"]["note"] = "COM x after 0.3 s Gaussian smoothing; path length depends on frame rate"

    names = result.get("kp_names") or []
    K = result.get("keypoints")
    if K is not None and len(names):
        K = np.asarray(K, float)
        hips, neck = _kp(names, "left_hip", "right_hip"), _kp(names, "neck")
        top = neck or _kp(names, "left_shoulder", "right_shoulder")
        if hips and top:  # --- trunk lean: pelvis -> neck vs vertical
            v = K[:, top].mean(1) - K[:, hips].mean(1)
            ml = np.degrees(np.arctan2(v[:, 0], v[:, 1]))
            ap = np.degrees(np.arctan2(v[:, 2], v[:, 1]))
            p5 = lambda x: round(float(np.percentile(x, 5)), 1)  # noqa: E731
            p95 = lambda x: round(float(np.percentile(x, 95)), 1)  # noqa: E731
            out["trunk_lean_deg"] = {
                "ml_p5": p5(ml), "ml_p95": p95(ml), "ap_p5": p5(ap), "ap_p95": p95(ap),
                "ml_min": round(float(ml.min()), 1), "ml_max": round(float(ml.max()), 1),
                "ml_range": round(float(np.ptp(ml)), 1),
                "ap_min": round(float(ap.min()), 1), "ap_max": round(float(ap.max()), 1),
                "ap_range": round(float(np.ptp(ap)), 1), "ap_estimated": ap_est,
                "ml_max_at_ms": round(float(t[int(np.argmax(np.abs(ml)))]) * 1000, 1),
                "ap_max_at_ms": round(float(t[int(np.argmax(np.abs(ap)))]) * 1000, 1),
                "convention": "+ml toward patient's left (+x), +ap forward (toward camera)"}
            series_lean = (ml, ap)
        else:
            series_lean = None
        up = max(LIFT_MIN_M, LIFT_NOISE_K * float((result.get("noise_floor") or {}).get("y", 0.0)))
        out["stance"], stance_label = _stance(t, dt, K, names, expected_stance, up)
        if stance_label is not None:
            out["touchdowns"] = _touchdowns(t, m, K, names, stance_label, out.get("margin", {}).get("outside_episodes", []),
                                            events or [], small)
        out["series"] = {
            "t_ms": [round(float(x) * 1000, 1) for x in t],
            "trunk_ml_deg": None if series_lean is None else [round(float(x), 1) for x in series_lean[0]],
            "trunk_ap_deg": None if series_lean is None else [round(float(x), 1) for x in series_lean[1]],
            "stance": None if stance_label is None else [str(x) for x in stance_label],
            "outside_cum_s": [round(float(x), 2) for x in np.cumsum(np.where(ok & (m < 0), dt, 0.0))],
        }

    nf = result.get("noise_floor") or {}
    out["noise_floor_cm"] = {a: round(float(v) * 100, 2) for a, v in nf.items()}
    out["events"] = events or []
    out["quality"] = _quality(q, nf)
    return out


def _stance(t, dt, K, names, expected, up):
    feet = {s: _kp(names, f"{s}_heel", f"{s}_big_toe", f"{s}_small_toe") for s in ("left", "right")}
    if not all(feet.values()):
        return {"available": False}, None
    low = {s: K[:, idx, 1].min(axis=1) for s, idx in feet.items()}  # lowest foot keypoint height
    rel = low["left"] - low["right"]  # > 0: left foot higher
    left_up, right_up = _hysteresis(rel, up, up / 2), _hysteresis(-rel, up, up / 2)
    label = np.where(left_up, "single_right", np.where(right_up, "single_left", "double"))
    centre = {s: K[:, idx][:, :, [0, 2]].mean(axis=1) for s, idx in feet.items()}
    d = centre["left"] - centre["right"]  # (x, z) separation
    placement = np.where(np.abs(d[:, 1]) > 0.12, np.where(np.abs(d[:, 0]) < 0.10, "tandem", "staggered"), "side_by_side")
    label = np.where(label == "double", np.char.add("double_", placement.astype(str)), label)
    seconds: dict[str, float] = {}
    for lab in np.unique(label):
        seconds[str(lab)] = round(float(dt[label == lab].sum()), 2)
    segs = []
    for lab in np.unique(label):
        for a, b in _runs(label == lab):
            segs.append({"stance": str(lab), "start_ms": round(float(t[a]) * 1000, 1),
                         "end_ms": round(float(t[b - 1]) * 1000, 1), "duration_s": round(float(dt[a:b].sum()), 2)})
    segs.sort(key=lambda s: s["start_ms"])
    res = {"available": True, "seconds": seconds, "segments": segs,
           "method": f"lowest heel/toe keypoint of each foot; up > {up * 100:.1f} cm above the other foot, down < {up * 50:.1f} cm"}
    if expected:
        match = np.array([lab == expected or (expected == "double" and lab.startswith("double"))
                          or (expected == "tandem" and lab == "double_tandem") for lab in label])
        frac = float(dt[match].sum() / dt.sum())
        res["expected"] = {"stance": expected, "fraction_matching": round(frac, 3), "pass": frac >= 0.9}
    return res, label


def _foot_xz(K, names, side, i):
    heel = [j for j, n in enumerate(names) if n.startswith(f"{side}_heel")]
    toes = [j for j, n in enumerate(names) if n.startswith(f"{side}_big_toe") or n.startswith(f"{side}_small_toe")]
    if not heel or not toes:
        return None
    h, to = K[i, heel, :][:, [0, 2]].mean(0), K[i, toes, :][:, [0, 2]].mean(0)
    return {"heel": [round(float(h[0]), 4), round(float(h[1]), 4)], "toe": [round(float(to[0]), 4), round(float(to[1]), 4)]}


def _touchdowns(t, m, K, names, label, episodes, events, small):
    """Foot landings: SAM stance transitions single -> double (the lifted foot lands), plus external
    events (scout / metrics) placed at that foot's floor position. Each gets the margin story: the
    outside-BOS episode that started up to 1.5 s before landing and the COM-exit -> landing lead."""
    out = []
    for i in range(1, len(label)):
        if label[i - 1].startswith("single") and label[i].startswith("double"):
            side = "right" if label[i - 1] == "single_left" else "left"
            t_land = (t[i - 1] + t[i]) / 2  # landed between these two frames
            out.append({"kind": "foot_down", "side": side, "t_ms": round(float(t_land) * 1000, 1),
                        "resolution_s": round(float(t[i] - t[i - 1]) / 2, 2), "source": "SAM stance (both feet down)",
                        "frame": i, "foot": _foot_xz(K, names, side, i), "uncertain": False})
    for e in events:
        te = e["t"] / 1000
        i = int(np.argmin(np.abs(t - te)))
        if abs(t[i] - te) > 1.0:
            continue
        side = e.get("side") or "right"
        kind = e.get("kind", "event")
        out.append({"kind": kind, "side": side, "t_ms": round(float(te) * 1000, 1), "resolution_s": None,
                    "source": e.get("source", "event list"), "frame": i, "foot": _foot_xz(K, names, side, i),
                    "uncertain": kind in ("toe_touch", "possible_touchdown")})
    for d in out:
        tl = d["t_ms"] / 1000
        ep = [e for e in episodes if tl - 1.5 <= e["start_ms"] / 1000 <= tl + 0.2]
        win = (t >= tl - 1.5) & (t <= tl + 0.2) & ~np.isnan(m)
        d["min_margin_cm"] = round(float(m[win].min()) * 100, 2) if win.any() else None
        if d["min_margin_cm"] is not None and abs(d["min_margin_cm"] / 100) < small:
            d["uncertain"] = True
        if ep:
            d["com_exit_ms"] = ep[0]["start_ms"]
            d["lead_s"] = round(tl - ep[0]["start_ms"] / 1000, 2)
        side_txt = d["side"].capitalize()
        base = EVENT_LABELS.get(d["kind"], d["kind"].replace("_", " ").capitalize())
        d["label"] = f"{side_txt} foot down" if d["kind"] == "foot_down" else (f"{base} ({side_txt.lower()})")
        if d["uncertain"]:
            d["label"] += " · uncertain"
    out.sort(key=lambda d: d["t_ms"])
    for k, d in enumerate(out, 1):
        d["n"] = k
    return out


def _quality(q: dict, nf: dict) -> dict:
    rep = q.get("reprojection") or {}
    frames = q.get("frames", 0)
    dropped = len(rep.get("dropped", []))
    metrics = {
        "reprojection_median_px": rep.get("median_px"),
        "noise_x_cm": round(nf["x"] * 100, 2) if "x" in nf else None,
        "dropped_frac": round(dropped / max(frames + dropped, 1), 3) if rep else None,
        "ground_anchor_median_cm": (q.get("ground_anchor_cm") or {}).get("median_abs"),
    }
    grade, reasons = "good", []
    for k, v in metrics.items():
        if v is None:
            continue
        good, fair = QUALITY[k]
        if v > fair:
            grade = "poor"
            reasons.append(f"{k} {v} > {fair}")
        elif v > good:
            grade = "fair" if grade == "good" else grade
            reasons.append(f"{k} {v} > {good}")
    if frames < 10:
        grade = "poor"
        reasons.append(f"only {frames} frames")
    return {"label": grade, "reasons": reasons, "metrics": metrics,
            "forward_back": "estimated" if not q.get("ap_real", False) else "measured",
            "floor_source": q.get("floor_source"), "contact_band_cm": q.get("contact_band_cm"),
            "gravity_vs_feet_deg": q.get("gravity_vs_feet_deg")}
