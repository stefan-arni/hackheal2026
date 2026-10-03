"""Cross-validate the SAM replay against posecam (teammates' MediaPipe pipeline) and the scout.

    uv run --with matplotlib python tools/cross_validate.py \\
        --run data/fal_out/demo --events data/events/IMG_9691.json \\
        --landmarks data/scout/IMG_9691/landmarks_2d.json \\
        --posecam data/reports/IMG_9691/posecam_run.json --out data/reports/IMG_9691

Side-to-side (x) series are compared in cm: MediaPipe hip pixels are converted with SAM's
own depth, focal and height scale (cm/px = s·Z̄/f̄), never fitted, then demeaned and smoothed
with the replay's 0.3 s kernel. Writes cross_validation.json and cross_validation.png.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPLAY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPLAY_ROOT))

from service import analytics, geometry as g, pipeline  # noqa: E402

L_HIP, R_HIP, L_SH, R_SH = 23, 24, 11, 12


def smooth_to(t_src, x_src, t_dst, sigma=pipeline.SMOOTH_SIGMA_S):
    """Gaussian-weighted average of a (dense) series at the replay's timestamps."""
    ok = ~np.isnan(x_src)
    t_src, x_src = t_src[ok], x_src[ok]
    w = np.exp(-((t_dst[:, None] - t_src[None]) ** 2) / (2 * sigma**2))
    return (w * x_src).sum(1) / w.sum(1)


def compare(a, b):
    a, b = a - a.mean(), b - b.mean()
    return {"r": round(float(np.corrcoef(a, b)[0, 1]), 3), "rms_diff_cm": round(float(np.sqrt(np.mean((a - b) ** 2))), 2)}


def main(a: argparse.Namespace) -> None:
    ev = json.loads(a.events.read_text())
    base = ev["clip_start_epoch_ms"] / 1000
    run = pipeline.load_run(a.run)
    lm = json.loads(a.landmarks.read_text())
    stance = [(base + t0, base + t1) for t0, t1 in ev["stance_intervals_clip_s"]]
    res = pipeline.process(run, ev["events"], a.height_cm / 100, landmarks=lm, stance_intervals_s=stance)
    stats = analytics.compute(res, ev["events"])
    t = res["t_s"] - base  # clip seconds
    names = res["kp_names"]
    K = res["keypoints"]
    hip_i = [names.index("left_hip"), names.index("right_hip")]
    neck_i = names.index("neck")
    sam_com_x = res["com"][:, 0] * 100
    sam_hip_x = K[:, hip_i, 0].mean(1) * 100
    v = K[:, neck_i] - K[:, hip_i].mean(1)
    sam_lean = np.degrees(np.arctan2(v[:, 0], v[:, 1]))

    # px -> cm in SAM's metric frame at the body's depth (no fitting)
    f_bar = float(np.median(run.focal))
    z_bar = float(np.median(g.normalize_focal(run.cam_t, run.focal)[:, 2]))
    cm_per_px = res["scale"] * z_bar / f_bar * 100
    sx = (run.crop[0, 2] - run.crop[0, 0]) / run.image_size[0, 0] if run.crop is not None else 1.0
    cm_per_px *= sx

    pc = json.loads(a.posecam.read_text())
    pf = [f for f in pc["frames"] if f.get("xyv")]
    pt = np.array([f["t"] for f in pf])
    p_xy = np.array([f["xyv"] for f in pf], float)
    pc_hip_x = smooth_to(pt, p_xy[:, [L_HIP, R_HIP], 0].mean(1) * cm_per_px, t)
    pc_lean = smooth_to(pt, np.array([(f.get("spine") or {}).get("trunk_angle_deg", np.nan) for f in pf], float), t)
    s_t = np.asarray(lm["t_ms"]) / 1000
    s_xy = np.asarray(lm["xyv"], float)
    sc_hip_x = smooth_to(s_t, s_xy[:, [L_HIP, R_HIP], 0].mean(1) * cm_per_px, t)

    sway = {"cm_per_px": round(cm_per_px, 4),
            "sam_hip_vs_posecam_hip": compare(sam_hip_x, pc_hip_x),
            "sam_hip_vs_scout_hip": compare(sam_hip_x, sc_hip_x),
            "posecam_hip_vs_scout_hip": compare(pc_hip_x, sc_hip_x),
            "sam_com_vs_posecam_hip": compare(sam_com_x, pc_hip_x),
            "sam_com_vs_sam_hip": compare(sam_com_x, sam_hip_x)}
    lean = {"sam_trunk_ml_vs_posecam_trunk_angle_2d": {
        "r": round(float(np.corrcoef(sam_lean, pc_lean)[0, 1]), 3),
        "rms_diff_deg": round(float(np.sqrt(np.mean((sam_lean - pc_lean) ** 2))), 1)}}

    # --- events on one clock (clip seconds)
    rep = json.loads((a.landmarks.parent / "report.json").read_text())
    scout = [e for e in rep["candidate_events"] if e["t_end_s"] >= t[0] - 2]
    msgs = pc["messages"]
    pc_lifts, open_lift = [], None
    for m in msgs:
        if m.get("kind") == "balance_started":
            open_lift = (m["t_clip_s"], m["lifted_foot"])
        elif m.get("kind") == "foot_touchdown" and open_lift:
            pc_lifts.append((open_lift[0], m["t_clip_s"], open_lift[1]))
            open_lift = None
    bess = [(m["t_clip_s"], m["label"]) for m in msgs if m.get("kind") == "bess_error" and m.get("counted")]
    sam_out = [((e["start_ms"] / 1000 - base), (e["end_ms"] / 1000 - base), e["min_margin_cm"])
               for e in stats["margin"]["outside_episodes"]]
    sam_single = [((s["start_ms"] / 1000 - base), (s["end_ms"] / 1000 - base), s["stance"])
                  for s in stats["stance"]["segments"] if s["stance"].startswith("single")]
    pick = lambda xs, lo, hi: [x for x in xs if lo <= x <= hi]  # noqa: E731
    right_down = {"scout_lift_end": next((e["t_end_s"] for e in scout if e["kind"] == "foot_lift" and e["side"] == "right"
                                          and e["t_end_s"] > 54), None),
                  "posecam_touchdown": next((b for _, b, s in pc_lifts if s == "right"), None),
                  "sam_single_left_end": next((b for _, b, s in sam_single if s == "single_left"), None)}
    toe = {"scout_possible_touchdown": next(([e["t_start_s"], e["t_end_s"]] for e in scout if e["kind"] == "possible_touchdown"), None),
           "posecam_touchdowns_in_window": pick([b for _, b, _ in pc_lifts], 48.0, 50.5),
           "sam_outside_bos": next(([round(s, 2), round(e, 2), m] for s, e, m in sam_out if 48 <= s <= 50.5), None),
           "posecam_bess_errors_47_51": [(round(x, 2), l) for x, l in bess if 47 <= x <= 51]}
    step = {"scout_step": next((e["t_start_s"] for e in scout if e["kind"] == "step" and 57 <= e["t_start_s"] <= 58), None),
            "posecam_bess_step": next((x for x, l in bess if "Step" in l and 56 <= x <= 58.5), None),
            "sam_double_stance_from": next((s["start_ms"] / 1000 - base for s in stats["stance"]["segments"]
                                            if s["stance"].startswith("double") and s["start_ms"] / 1000 - base > 56), None)}
    summary = {"window_clip_s": [round(float(t[0]), 2), round(float(t[-1]), 2)], "frames": int(len(t)),
               "side_to_side_sway": sway, "trunk_lean": lean,
               "events": {"right_foot_down": right_down, "toe_touch": toe, "step": step,
                          "posecam_lifts": pc_lifts, "posecam_bess_counted": bess,
                          "sam_outside_bos": sam_out, "sam_single_stance": sam_single}}
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "cross_validation.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps({k: v for k, v in summary.items() if k != "events"}, indent=1))
    print(json.dumps({k: summary["events"][k] for k in ("right_foot_down", "toe_touch", "step")}, indent=1))
    plot(a.out / "cross_validation.png", t, sam_com_x, sam_hip_x, pc_hip_x, sc_hip_x, sam_lean, pc_lean,
         res["margin"] * 100, sway, lean, scout, pc_lifts, bess, sam_single, sam_out)


def plot(path, t, com, hip, pch, sch, lean, pcl, margin, sway, lean_s, scout, pc_lifts, bess, sam_single, sam_out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    c = {"sam": "#2563eb", "samc": "#7c3aed", "pc": "#ea580c", "sc": "#16a34a"}
    fig = plt.figure(figsize=(16, 9), dpi=110)
    gs = fig.add_gridspec(3, 1, height_ratios=[2.2, 1.4, 1.5], hspace=0.12)
    ax0 = fig.add_subplot(gs[0])
    ax1 = fig.add_subplot(gs[1], sharex=ax0)
    ax2 = fig.add_subplot(gs[2], sharex=ax0)
    dm = lambda x: x - x.mean()  # noqa: E731
    s1, s2 = sway["sam_hip_vs_posecam_hip"], sway["sam_hip_vs_scout_hip"]
    ax0.plot(t, dm(hip), color=c["sam"], lw=3, label="SAM 3D Body: hip centre (replay)")
    ax0.plot(t, dm(com), color=c["samc"], lw=2, ls="--", label="SAM 3D Body: whole-body COM")
    ax0.plot(t, dm(pch), color=c["pc"], lw=2.2, label=f"posecam (MediaPipe): hip centre   r={s1['r']:.2f}, RMS diff {s1['rms_diff_cm']:.1f} cm")
    ax0.plot(t, dm(sch), color=c["sc"], lw=1.6, ls=":", label=f"scout (MediaPipe): hip centre   r={s2['r']:.2f}, RMS diff {s2['rms_diff_cm']:.1f} cm")
    ax0.set_ylabel("side-to-side (cm)", fontsize=13)
    lo, hi = ax0.get_ylim()
    ax0.set_ylim(lo, hi + 0.55 * (hi - lo))  # headroom for the legend
    ax0.legend(loc="upper left", fontsize=11, frameon=False, ncol=2)
    ax0.set_title("IMG_9691 (44.8–64.1 s): SAM replay vs posecam vs scout — side-to-side sway, trunk lean, events",
                  fontsize=15, weight="bold", loc="left")
    l1 = lean_s["sam_trunk_ml_vs_posecam_trunk_angle_2d"]
    ax1.plot(t, lean, color=c["sam"], lw=3, label="SAM: trunk lean (pelvis→neck, floor frame)")
    ax1.plot(t, pcl, color=c["pc"], lw=2.2, label=f"posecam: trunk_angle_deg (2D)   r={l1['r']:.2f}, RMS diff {l1['rms_diff_deg']:.0f}°")
    ax1.axhline(0, color="#9ca3af", lw=1)
    ax1.set_ylabel("side lean (°)", fontsize=13)
    lo, hi = ax1.get_ylim()
    ax1.set_ylim(lo, hi + 0.45 * (hi - lo))
    ax1.legend(loc="upper left", fontsize=11, frameon=False, ncol=2)
    lanes = {"scout": 3, "posecam": 2, "SAM": 1}
    for e in scout:
        col = "#16a34a" if e["kind"] == "foot_lift" else ("#dc2626" if e["kind"] == "possible_touchdown" else "#111827")
        ax2.plot([e["t_start_s"], max(e["t_end_s"], e["t_start_s"] + 0.15)], [3, 3], color=col, lw=10, solid_capstyle="butt")
    for s, e, side in pc_lifts:
        ax2.plot([s, e], [2, 2], color="#16a34a", lw=10, solid_capstyle="butt")
        ax2.plot([e], [2], marker="v", color="#dc2626", ms=13)
    for x, lab in bess:
        ax2.plot([x], [2.28], marker="|", color="#111827", ms=16, mew=3)
    for s, e, lab in sam_single:
        ax2.plot([s, e], [1, 1], color="#16a34a", lw=10, solid_capstyle="butt")
    for s, e, m in sam_out:
        ax2.plot([s, max(e, s + 0.15)], [0.72, 0.72], color="#dc2626", lw=8, solid_capstyle="butt")
    ax2.set_yticks([1, 2, 3], ["SAM", "posecam", "scout"], fontsize=12)
    ax2.set_ylim(0.4, 3.6)
    ax2.set_xlabel("clip time (s)", fontsize=13)
    ax2.text(1.0, -0.42, "green = foot lifted / single-leg stance   red bar = SAM COM outside BOS / scout possible touchdown   "
             "▼ posecam foot touchdown   | posecam BESS error   black = scout step", transform=ax2.transAxes,
             ha="right", fontsize=10, color="#374151")
    for ax in (ax0, ax1, ax2):
        ax.grid(alpha=0.25)
        ax.set_xlim(t[0], t[-1])
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    plt.setp(ax0.get_xticklabels(), visible=False)
    plt.setp(ax1.get_xticklabels(), visible=False)
    fig.savefig(path, bbox_inches="tight")
    print("wrote", path)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--events", type=Path, required=True)
    p.add_argument("--landmarks", type=Path, required=True)
    p.add_argument("--posecam", type=Path, required=True)
    p.add_argument("--height-cm", type=float, default=185)
    p.add_argument("--out", type=Path, required=True)
    main(p.parse_args())
