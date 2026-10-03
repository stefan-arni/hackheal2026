"""One-page trial report (report.html + report_assets/) written next to a replay bundle.

Self-contained: inline SVG charts, PNG renders from service/render.py (no browser needed),
printable to PDF (one page). Times are trial-relative (0 = first frame). Wording is about
agreement and estimates, never validation.
"""

from __future__ import annotations

import html
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

from service import render

GREEN, AMBER, RED, BLUE, INK, MUTED = "#16a34a", "#d97706", "#dc2626", "#0284c7", "#0f172a", "#64748b"


def _margin_color(m_cm: float | None) -> str:
    if m_cm is None or np.isnan(m_cm):
        return MUTED
    return RED if m_cm < 0 else (AMBER if m_cm < 1.5 else GREEN)


def _fmt(x, nd=1, unit=""):
    return "–" if x is None else f"{x:+.{nd}f}{unit}" if unit == " cm" else f"{x:.{nd}f}{unit}"


# --------------------------------------------------------------------------- images

def hero_image(path: Path, V: np.ndarray, faces: np.ndarray, com: np.ndarray, hull, margin_cm, touchdowns,
               size=(820, 1000)) -> None:
    W, H = size
    target = np.array([com[0], 0.9, com[2]])
    az, el, dist = np.radians(32), np.radians(16), 3.6
    eye = target + dist * np.array([np.sin(az) * np.cos(el), np.sin(el), np.cos(az) * np.cos(el)])
    R, t = render.look_at(eye, target)
    f, cx, cy = 1.55 * H, W / 2, H * 0.45
    img = Image.new("RGBA", (W, H), (248, 250, 252, 255))
    d = ImageDraw.Draw(img, "RGBA")
    for k in range(-6, 7):  # 10 cm floor grid, ±60 cm around the feet
        v = k / 10
        render.floor_polyline(img, [[-0.6, v], [0.6, v]], R, t, f, cx, cy, (203, 213, 225, 255), width=2, closed=False)
        render.floor_polyline(img, [[v, -0.6], [v, 0.6]], R, t, f, cx, cy, (203, 213, 225, 255), width=2, closed=False)
    if hull is not None and len(hull) >= 3:
        render.floor_polyline(img, hull, R, t, f, cx, cy, (2, 132, 199, 255), width=5, fill=(56, 189, 248, 90))
    for td in touchdowns:
        if td.get("foot"):
            render.floor_polyline(img, [td["foot"]["heel"], td["foot"]["toe"]], R, t, f, cx, cy,
                                  (217, 119, 6, 200), width=14, closed=False)
    img = render.draw_mesh(img, V @ R.T + t, faces, f, cx, cy, color=(214, 222, 232), alpha=0.82, rim=1.1)
    d = ImageDraw.Draw(img, "RGBA")
    col = (220, 38, 38) if margin_cm is not None and margin_cm < 0 else (22, 163, 74)
    top = render.project(np.array([com]) @ R.T + t, f, cx, cy)[0]
    foot = render.project(np.array([[com[0], 0, com[2]]]) @ R.T + t, f, cx, cy)[0]
    d.line([tuple(top), tuple(foot)], fill=col + (255,), width=6)
    for r, a in ((26, 70), (17, 255)):
        d.ellipse([top[0] - r, top[1] - r, top[0] + r, top[1] + r], fill=col + (a,))
    d.ellipse([foot[0] - 12, foot[1] - 7, foot[0] + 12, foot[1] + 7], fill=col + (255,), outline=(255, 255, 255, 255), width=2)
    img.convert("RGB").save(path, "PNG", optimize=True)


def overlay_image(path: Path, src: Path, V: np.ndarray, faces: np.ndarray, M, t, focal: float,
                  image_size, src_scale: float, label: str, crop_to_body=True) -> None:
    im = Image.open(src).convert("RGB")
    Vc = V @ np.asarray(M).T + np.asarray(t)
    f = focal * src_scale
    cx, cy = image_size[0] * src_scale / 2, image_size[1] * src_scale / 2
    out = render.draw_mesh(im, Vc, faces, f, cx, cy, color=(56, 189, 248), alpha=0.42, rim=0.8, rim_color=(255, 255, 255))
    if crop_to_body:
        uv = render.project(Vc, f, cx, cy)
        x0, y0 = np.nanmin(uv, 0)
        x1, y1 = np.nanmax(uv, 0)
        h = (y1 - y0) * 1.08
        w = h * 0.62
        xc, yc = (x0 + x1) / 2, (y0 + y1) / 2
        out = out.crop((int(xc - w / 2), int(yc - h / 2), int(xc + w / 2), int(yc + h / 2)))
    d = ImageDraw.Draw(out, "RGBA")
    d.rectangle([0, 0, out.width, 26], fill=(15, 23, 42, 210))
    d.text((6, 4), label, fill=(255, 255, 255, 255), font_size=17)
    out.convert("RGB").save(path, "JPEG", quality=86)


# --------------------------------------------------------------------------- SVG charts

def margin_chart(t: np.ndarray, m_cm: np.ndarray, episodes, touchdowns, W=760, H=230) -> str:
    pad_l, pad_r, pad_t, pad_b = 48, 12, 34, 30
    T = float(t[-1]) if len(t) else 1.0
    ok = ~np.isnan(m_cm)
    lo, hi = min(-2.0, float(np.nanmin(m_cm)) - 1), max(4.0, float(np.nanmax(m_cm)) + 1)
    X = lambda x: pad_l + (W - pad_l - pad_r) * x / T  # noqa: E731
    Y = lambda y: pad_t + (H - pad_t - pad_b) * (hi - y) / (hi - lo)  # noqa: E731
    s = [f'<svg viewBox="0 0 {W} {H}" class="chart" role="img" aria-label="Margin of stability over time">']
    for e in episodes:
        x0, x1 = X(e["t0"]), X(max(e["t1"], e["t0"] + 0.1))
        s.append(f'<rect x="{x0:.1f}" y="{pad_t}" width="{x1 - x0:.1f}" height="{H - pad_t - pad_b}" '
                 f'fill="{RED}" opacity="{0.10 if e["uncertain"] else 0.18}"/>')
    for v in range(int(np.floor(lo / 5) * 5), int(hi) + 1, 5):
        s.append(f'<line x1="{pad_l}" x2="{W - pad_r}" y1="{Y(v):.1f}" y2="{Y(v):.1f}" stroke="#e2e8f0"/>'
                 f'<text x="{pad_l - 6}" y="{Y(v) + 4:.1f}" text-anchor="end" class="ax">{v}</text>')
    s.append(f'<line x1="{pad_l}" x2="{W - pad_r}" y1="{Y(0):.1f}" y2="{Y(0):.1f}" stroke="{INK}" stroke-width="1.5"/>')
    for k in range(len(t) - 1):
        if ok[k] and ok[k + 1]:
            col = _margin_color((m_cm[k] + m_cm[k + 1]) / 2)
            s.append(f'<line x1="{X(t[k]):.1f}" y1="{Y(m_cm[k]):.1f}" x2="{X(t[k + 1]):.1f}" y2="{Y(m_cm[k + 1]):.1f}" '
                     f'stroke="{col}" stroke-width="4" stroke-linecap="round"/>')
    last_x = -1e9
    for td in touchdowns:
        x = X(td["tt"])
        bx = max(x, last_x + 22)  # keep number badges from overlapping
        last_x = bx
        s.append(f'<line x1="{x:.1f}" x2="{x:.1f}" y1="{pad_t - 4}" y2="{H - pad_b}" stroke="{INK}" '
                 f'stroke-dasharray="{"3 3" if td["uncertain"] else "0"}" stroke-width="1.5"/>'
                 f'<line x1="{x:.1f}" x2="{bx:.1f}" y1="{pad_t - 4}" y2="{pad_t - 8}" stroke="{INK}"/>'
                 f'<circle cx="{bx:.1f}" cy="{pad_t - 16}" r="10" fill="{AMBER}"/>'
                 f'<text x="{bx:.1f}" y="{pad_t - 12}" text-anchor="middle" class="num">{td["n"]}</text>')
    for v in range(0, int(T) + 1, 2):
        s.append(f'<text x="{X(v):.1f}" y="{H - 10}" text-anchor="middle" class="ax">{v}</text>')
    s.append(f'<text x="{W - pad_r}" y="{H - 10}" text-anchor="end" class="ax">s</text>'
             f'<text x="12" y="{pad_t + 8}" class="ax" transform="rotate(-90 12 {pad_t + 8})" text-anchor="end">margin cm</text></svg>')
    return "".join(s)


def stabilogram(com_cm: np.ndarray, m_cm: np.ndarray, hulls, key_hull, touchdowns, W=360, H=360) -> str:
    pts = [com_cm[:, [0, 2]]] + [np.asarray(h) * 100 for h in hulls if h is not None and len(h)]
    pts += [np.array([td["foot"]["heel"], td["foot"]["toe"]]) * 100 for td in touchdowns if td.get("foot")]
    allp = np.vstack(pts)
    c = (allp.min(0) + allp.max(0)) / 2
    half = max(15.0, float((allp.max(0) - allp.min(0)).max()) / 2 + 6)
    S = (W - 40) / (2 * half)
    X = lambda x: 20 + (x - (c[0] - half)) * S  # noqa: E731
    Z = lambda z: 20 + (z - (c[1] - half)) * S  # noqa: E731
    s = [f'<svg viewBox="0 0 {W} {H}" class="chart" role="img" aria-label="Top-down stabilogram">']
    g0 = int(np.floor((c[0] - half) / 10) * 10)
    for v in range(g0, int(c[0] + half) + 11, 10):
        s.append(f'<line x1="{X(v):.1f}" x2="{X(v):.1f}" y1="20" y2="{H - 20}" stroke="#e2e8f0"/>')
    g1 = int(np.floor((c[1] - half) / 10) * 10)
    for v in range(g1, int(c[1] + half) + 11, 10):
        s.append(f'<line y1="{Z(v):.1f}" y2="{Z(v):.1f}" x1="20" x2="{W - 20}" stroke="#e2e8f0"/>')
    for h in hulls[:: max(1, len(hulls) // 25)]:
        if h is not None and len(h) >= 3:
            p = " ".join(f"{X(x * 100):.1f},{Z(z * 100):.1f}" for x, z in h)
            s.append(f'<polygon points="{p}" fill="{BLUE}" fill-opacity="0.04" stroke="{BLUE}" stroke-opacity="0.15"/>')
    if key_hull is not None and len(key_hull) >= 3:
        p = " ".join(f"{X(x * 100):.1f},{Z(z * 100):.1f}" for x, z in key_hull)
        s.append(f'<polygon points="{p}" fill="{BLUE}" fill-opacity="0.18" stroke="{BLUE}" stroke-width="3"/>')
    for k in range(len(com_cm) - 1):
        col = _margin_color(m_cm[k])
        s.append(f'<line x1="{X(com_cm[k, 0]):.1f}" y1="{Z(com_cm[k, 2]):.1f}" x2="{X(com_cm[k + 1, 0]):.1f}" '
                 f'y2="{Z(com_cm[k + 1, 2]):.1f}" stroke="{col}" stroke-width="3" stroke-linecap="round"/>')
    for td in touchdowns:
        if not td.get("foot"):
            continue
        (hx, hz), (tx, tz) = np.array(td["foot"]["heel"]) * 100, np.array(td["foot"]["toe"]) * 100
        s.append(f'<line x1="{X(hx):.1f}" y1="{Z(hz):.1f}" x2="{X(tx):.1f}" y2="{Z(tz):.1f}" stroke="{AMBER}" '
                 f'stroke-width="{9 * S / 3:.1f}" stroke-linecap="round" opacity="{0.45 if td["uncertain"] else 0.8}"/>')
        mx, mz = X((hx + tx) / 2), Z((hz + tz) / 2)
        s.append(f'<circle cx="{mx:.1f}" cy="{mz:.1f}" r="10" fill="{INK}"/>'
                 f'<text x="{mx:.1f}" y="{mz + 4:.1f}" text-anchor="middle" class="num">{td["n"]}</text>')
    s.append(f'<line x1="{W - 20 - 10 * S:.1f}" x2="{W - 20}" y1="{H - 10}" y2="{H - 10}" stroke="{INK}" stroke-width="3"/>'
             f'<text x="{W - 20 - 5 * S:.1f}" y="{H - 14}" text-anchor="middle" class="ax">10 cm</text>'
             f'<text x="22" y="{H - 6}" class="ax">camera ↓ · patient’s left →</text></svg>')
    return "".join(s)


# --------------------------------------------------------------------------- page

def write_report(out: Path, meta: dict, stats: dict, result: dict, *, title: str | None = None,
                 cross_validation: dict | None = None) -> Path:
    assets = out / "report_assets"
    assets.mkdir(exist_ok=True)
    t_abs = np.asarray(result["t_s"], float)
    t = t_abs - t_abs[0]
    V = np.asarray(result["verts"], float)
    V_raw = np.asarray(result.get("verts_raw", result["verts"]), float)  # overlays: unsmoothed (no lag)
    faces = np.asarray(result["faces"])
    com = np.asarray(result["com"], float)
    m_cm = np.array([np.nan if x is None else x for x in result["margin"]], float) * 100
    hulls = result["bos"]
    cam, frames = meta.get("camera") or {}, meta.get("frames") or []
    tds = [dict(d, tt=d["t_ms"] / 1000 - t_abs[0]) for d in stats.get("touchdowns", [])]
    episodes = [dict(e, t0=e["start_ms"] / 1000 - t_abs[0], t1=e["end_ms"] / 1000 - t_abs[0])
                for e in stats.get("margin", {}).get("outside_episodes", [])]
    key = int(np.nanargmin(m_cm)) if np.isfinite(m_cm).any() else 0
    hero_image(assets / "hero.png", V[key], faces, com[key], hulls[key], m_cm[key], tds)

    cards = []
    for d in tds:
        strip = []
        for k, dtv in enumerate((-0.5, 0.0, 0.5)):
            i = int(np.argmin(np.abs(t - (d["tt"] + dtv))))
            fr = frames[i] if i < len(frames) else {}
            if fr.get("src_url") and cam.get("t"):
                name = f"film_{d['n']}_{k}.jpg"
                overlay_image(assets / name, out / fr["src_url"], V_raw[i], faces, cam["M"], cam["t"][i], cam["focal"][i],
                              cam["image_size"][i], fr.get("src_scale", 1.0), f"{t[i]:.2f} s ({dtv:+.1f})")
                strip.append(f'<img src="report_assets/{name}" alt="frame at {t[i]:.2f} s with the replay mesh overlaid">')
        lead = (f'COM left the base of support <b>{d["lead_s"]:.1f} s</b> before' if d.get("lead_s") is not None
                else "COM stayed inside the base of support beforehand")
        unc = '<span class="pill warn">uncertain</span>' if d["uncertain"] else ""
        res_txt = f' ± {d["resolution_s"]:.2f} s' if d.get("resolution_s") else ""
        cards.append(f'''<div class="event"><div class="ev-h"><span class="badge">{d["n"]}</span>
          <b>{html.escape(d["label"].replace(" · uncertain", ""))}</b> {unc}<span class="t">{d["tt"]:.2f} s{res_txt}</span></div>
          <div class="ev-m">min margin <b style="color:{_margin_color(d.get("min_margin_cm"))}">{_fmt(d.get("min_margin_cm"), 1, " cm")}</b>
          · {lead} · source: {html.escape(d["source"])}</div><div class="strip">{"".join(strip)}</div></div>''')

    mg, ml, tl = stats.get("margin", {}), stats.get("ml_sway", {}), stats.get("trunk_lean_deg", {})
    st, q, nf = stats.get("stance", {}), stats.get("quality", {}), stats.get("noise_floor_cm", {})
    stance_txt = ", ".join(f"{k.replace('_', ' ')} {v:.1f} s" for k, v in sorted(st.get("seconds", {}).items(), key=lambda kv: -kv[1]))
    min_unc = mg.get("min_cm") is not None and abs(mg["min_cm"]) < mg.get("uncertain_below_cm", 0)
    date = datetime.fromtimestamp(t_abs[0]).strftime("%Y-%m-%d %H:%M") if t_abs[0] > 1e9 else "–"
    height = meta.get("quality", {}).get("patient_height_cm")
    scale_txt = f"scaled to {height:.0f} cm" if height else "SAM's own scale (no height given)"
    qcol = {"good": GREEN, "fair": AMBER, "poor": RED}.get(q.get("label"), MUTED)
    xv = ""
    if cross_validation:
        sw = cross_validation["side_to_side_sway"]["sam_hip_vs_posecam_hip"]
        ln = cross_validation["trunk_lean"]["sam_trunk_ml_vs_posecam_trunk_angle_2d"]
        xv = (f"<b>Agreement check:</b> two models on the same video agree — side-to-side hip sway SAM 3D Body vs "
              f"MediaPipe (posecam) r = {sw['r']:.2f}, RMS difference {sw['rms_diff_cm']:.1f} cm; trunk side lean "
              f"r = {ln['r']:.2f}, {ln['rms_diff_deg']:.1f}°. This is agreement between two models, not ground truth; "
              f"the cm conversion of the MediaPipe pixels uses SAM's scale.")
    page = f'''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Trial report</title><style>
:root{{--ink:{INK};--muted:{MUTED};--line:#e2e8f0;--bg:#ffffff;--card:#f8fafc}}
*{{box-sizing:border-box}} body{{margin:0;background:#e2e8f0;color:var(--ink);font:14px/1.35 system-ui,-apple-system,"Segoe UI",sans-serif}}
.page{{width:210mm;min-height:297mm;margin:16px auto;background:var(--bg);padding:9mm 10mm;box-shadow:0 2px 12px #0002}}
header{{display:flex;align-items:flex-start;gap:12px;border-bottom:3px solid var(--ink);padding-bottom:6px}}
h1{{font-size:22px;margin:0}} .sub{{color:var(--muted);font-weight:600}} .right{{margin-left:auto;text-align:right}}
.pill{{display:inline-block;padding:2px 9px;border-radius:999px;font-weight:800;font-size:12px;color:#fff;background:var(--muted)}}
.pill.warn{{background:{AMBER}}} .grid{{display:grid;grid-template-columns:62mm 1fr;gap:5mm;margin-top:4mm}}
.hero{{width:100%;border-radius:8px;border:1px solid var(--line)}} .cards{{display:grid;grid-template-columns:1fr 1fr;gap:3mm}}
.card{{background:var(--card);border:1px solid var(--line);border-radius:8px;padding:6px 9px}} .card .k{{color:var(--muted);font-weight:700;font-size:11px;text-transform:uppercase;letter-spacing:.04em}}
.card .v{{font-size:21px;font-weight:800}} .card .n{{color:var(--muted);font-size:11.5px}}
.charts{{display:grid;grid-template-columns:1fr 58mm;gap:4mm;margin-top:3mm}} .chart{{width:100%;height:auto}} .ax{{font-size:11px;fill:{MUTED}}} .num{{font-size:12px;font-weight:800;fill:#fff}}
h2{{font-size:13px;margin:3mm 0 1mm;text-transform:uppercase;letter-spacing:.05em;color:var(--muted)}}
.events{{display:grid;grid-template-columns:repeat({max(1, len(cards))},1fr);gap:3mm}} .event{{border:1px solid var(--line);border-radius:8px;padding:6px}}
.ev-h{{display:flex;gap:6px;align-items:center}} .ev-h .t{{margin-left:auto;font-weight:800}} .badge{{background:{AMBER};color:#fff;border-radius:999px;width:20px;height:20px;display:inline-grid;place-items:center;font-weight:800;font-size:12px}}
.ev-m{{color:var(--muted);font-size:11.5px;margin:3px 0}} .strip{{display:grid;grid-template-columns:repeat(3,1fr);gap:2px}} .strip img{{width:100%;border-radius:4px}}
footer{{margin-top:3mm;border-top:1px solid var(--line);padding-top:2mm;color:var(--muted);font-size:10.5px}} footer b{{color:var(--ink)}}
a.back{{color:{BLUE}}} @media print{{body{{background:#fff}} .page{{margin:0;box-shadow:none;width:auto;min-height:0}} .noprint{{display:none}} @page{{size:A4;margin:0}}}}
</style></head><body><div class="page">
<header><div><h1>{html.escape(title or (meta.get("trialId", "Trial") + " · balance replay"))}</h1>
<div class="sub">{html.escape(stance_txt)} · {date} · {len(t)} frames over {t[-1]:.1f} s · {scale_txt}</div></div>
<div class="right"><span class="pill" style="background:{qcol}">quality: {html.escape(q.get("label", "–"))}</span>
<span class="pill warn">forward/back {html.escape(q.get("forward_back", "estimated"))}</span>
<div class="noprint" style="margin-top:6px"><a class="back" id="open3d" href="#">open 3D replay</a></div></div></header>
<script>/* the bundle folder this report lives in, for the viewer link (service: /replay/<id>/report) */
const base = location.pathname.replace(/(report(\.html)?)$/, '');
document.getElementById('open3d').href = '/viewer/?src=' + encodeURIComponent(base);</script>
<div class="grid"><img class="hero" src="report_assets/hero.png" alt="3D replay at the lowest margin, {t[key]:.2f} s">
<div><div class="cards">
<div class="card"><div class="k">Min margin</div><div class="v" style="color:{_margin_color(mg.get("min_cm"))}">{_fmt(mg.get("min_cm"), 1, " cm")}</div>
<div class="n">at {(mg.get("min_at_ms", 0) / 1000 - t_abs[0]):.2f} s{" · uncertain" if min_unc else ""} · median {_fmt(mg.get("median_cm"), 1, " cm")}</div></div>
<div class="card"><div class="k">Time outside BOS</div><div class="v">{mg.get("time_outside_bos_s", 0):.1f} s</div>
<div class="n">{100 * mg.get("fraction_outside", 0):.0f}% of the trial · {len(episodes)} episodes ({sum(e["uncertain"] for e in episodes)} uncertain)</div></div>
<div class="card"><div class="k">Side-to-side sway</div><div class="v">{ml.get("rms_cm", 0):.1f} cm <span style="font-size:13px">RMS</span></div>
<div class="n">path {ml.get("path_length_cm", 0):.0f} cm · range {ml.get("range_cm", 0):.1f} cm</div></div>
<div class="card"><div class="k">Trunk lean p5–p95</div><div class="v">{tl.get("ml_p5", 0):.0f}° … {tl.get("ml_p95", 0):.0f}°</div>
<div class="n">side; forward {tl.get("ap_p5", 0):.0f}° … {tl.get("ap_p95", 0):.0f}° (estimated)</div></div>
<div class="card" style="grid-column:1/3"><div class="k">Stance check</div><div class="v" style="font-size:16px">{html.escape(stance_txt) or "–"}</div>
<div class="n">{html.escape(st.get("method", ""))}{" · expected " + st["expected"]["stance"] + (": pass" if st["expected"]["pass"] else ": not met") if st.get("expected") else ""}</div></div>
</div></div></div>
<div class="charts"><div><h2>Margin of stability over time (red = COM outside the base of support)</h2>{margin_chart(t, m_cm, episodes, tds)}</div>
<div><h2>Top-down stabilogram</h2>{stabilogram(com * 100, m_cm, hulls, hulls[key], tds)}</div></div>
<h2>Touchdowns and steps (mesh = our 3D replay drawn on the video frame)</h2><div class="events">{"".join(cards) or "<p>No touchdowns detected.</p>"}</div>
<footer><b>Method:</b> SAM 3D Body (fal) on {len(t)} frames ({json.dumps(stats.get("frames"))} kept); floor from the foot soles; whole-body COM from the mesh volume;
base of support = foot vertices within {q.get("contact_band_cm") or "–"} cm of the floor; 0.3 s smoothing; trunk lean = pelvis→neck. Height: {scale_txt}.
<b>Noise floor</b> (still feet): side-to-side {nf.get("x", 0):.2f} cm, vertical {nf.get("y", 0):.2f} cm, depth {nf.get("z", 0):.2f} cm — forward/back numbers are estimates;
excursions under {mg.get("uncertain_below_cm", 0):.1f} cm (3× side-to-side noise) are marked uncertain. {xv}
Research demo, not a diagnostic device.</footer></div></body></html>'''
    path = out / "report.html"
    path.write_text(page)
    return path
