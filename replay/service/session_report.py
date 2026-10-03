"""One-page BESS-style session summary (/session/{id}/report). Printable; honest wording."""

from __future__ import annotations

import html
from datetime import datetime

STANCE_ORDER = {"double": 0, "tandem": 1, "single": 2}
STANCE_NAME = {"double": "Double leg (feet together)", "tandem": "Tandem", "single": "Single leg"}
GREEN, AMBER, RED, MUTED, INK = "#16a34a", "#d97706", "#dc2626", "#64748b", "#0f172a"


def _num(x, fmt="{:.1f}", unit=""):
    return "–" if x is None else fmt.format(x) + unit


def _margin_col(m):
    return MUTED if m is None else (RED if m < 0 else (AMBER if m < 1.5 else GREEN))


def render(session: dict, rows: list[dict]) -> str:
    rows = sorted(rows, key=lambda r: (STANCE_ORDER.get(r["stance"], 9), r["trial"]))
    body, total_err, any_est, mock, warnings, scored = [], 0, False, False, [], False
    for r in rows:
        a, m, end, st = r["analytics"] or {}, r["meta"] or {}, r["end"] or {}, r["status"] or {}
        bess = (end.get("bess") or {})
        errs = bess.get("errors")
        if errs is None and end.get("events") is not None:  # count counted BESS errors from the event list
            errs = sum(1 for e in end["events"] if e.get("counted") and e.get("kind") not in ("foot_down",))
        total_err += errs or 0
        scored |= errs is not None
        mg, ml, q = a.get("margin", {}), a.get("ml_sway", {}), a.get("quality", {})
        any_est |= q.get("forward_back") == "estimated"
        mock |= bool((m.get("quality") or {}).get("MOCK_FAL") or (m.get("quality") or {}).get("SYNTHETIC"))
        if m.get("warning") and m["warning"] not in warnings:
            warnings.append(m["warning"])
        cov = m.get("coverage") or {}
        cov_txt = f'{cov.get("done", "–")}/{cov.get("received", "–")} frames ({cov.get("stage", st.get("stage", "–"))})' if cov else st.get("stage", "–")
        unc = mg.get("min_cm") is not None and abs(mg["min_cm"]) < mg.get("uncertain_below_cm", 0)
        qlab = q.get("label", "–")
        qcol = {"good": GREEN, "fair": AMBER, "poor": RED}.get(qlab, MUTED)
        tid = html.escape(r["trial"])
        links = (f'<a href="/viewer/?src=/replay/{tid}/&jump=step">3D replay</a> · <a href="/replay/{tid}/report">report</a>'
                 if m else "processing…")
        body.append(f"""<tr><td><b>{html.escape(STANCE_NAME.get(r["stance"], r["stance"] or "trial"))}</b><div class="tid">{tid}</div></td>
<td class="num">{"–" if errs is None else errs}</td>
<td class="num">{_num(ml.get("rms_cm"), unit=" cm")}<div class="sub">path {_num(ml.get("path_length_cm"), "{:.0f}", " cm")}</div></td>
<td class="num" style="color:{_margin_col(mg.get("min_cm"))}">{_num(mg.get("min_cm"), "{:+.1f}", " cm")}{"<div class=sub>uncertain</div>" if unc else ""}</td>
<td class="num">{_num(mg.get("time_outside_bos_s"), unit=" s")}</td>
<td><span class="pill" style="background:{qcol}">{html.escape(qlab)}</span><div class="sub">{html.escape(cov_txt)}</div></td>
<td>{links}</td></tr>""")
    created = datetime.fromtimestamp(session.get("created", 0)).strftime("%Y-%m-%d %H:%M") if session.get("created") else "–"
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Session summary</title><style>
*{{box-sizing:border-box}} body{{margin:0;background:#e2e8f0;color:{INK};font:14px/1.4 system-ui,-apple-system,"Segoe UI",sans-serif}}
.page{{width:210mm;min-height:297mm;margin:16px auto;background:#fff;padding:12mm;box-shadow:0 2px 12px #0002}}
h1{{font-size:24px;margin:0}} .sub{{color:{MUTED};font-size:12px;font-weight:600}} header{{border-bottom:3px solid {INK};padding-bottom:6px;margin-bottom:6mm}}
.note{{background:#fef3c7;border:1px solid #f59e0b;border-radius:8px;padding:6px 10px;font-weight:700;margin:4mm 0}}
table{{width:100%;border-collapse:collapse}} th{{text-align:left;font-size:11px;text-transform:uppercase;letter-spacing:.05em;color:{MUTED};border-bottom:2px solid #e2e8f0;padding:6px}}
td{{border-bottom:1px solid #e2e8f0;padding:9px 6px;vertical-align:top}} td.num{{font-size:20px;font-weight:800;white-space:nowrap}}
.mock{{background:#facc15;color:#111;border-radius:8px;padding:8px 10px;font-weight:900;margin:4mm 0}} .tid{{color:{MUTED};font-size:11px}}
.pill{{display:inline-block;color:#fff;border-radius:999px;padding:2px 9px;font-weight:800;font-size:12px}}
.total{{font-size:28px;font-weight:900}} footer{{margin-top:6mm;color:{MUTED};font-size:11px;border-top:1px solid #e2e8f0;padding-top:3mm}}
a{{color:#0284c7;font-weight:700}} @media print{{body{{background:#fff}} .page{{margin:0;box-shadow:none;width:auto;min-height:0}} @page{{size:A4;margin:0}}}}
</style></head><body><div class="page">
<header><h1>Balance session summary</h1><div class="sub">session {html.escape(session.get("id", ""))} · {created} · {len(rows)} trial(s)</div></header>
{'<div class="mock">fal MOCK / SYNTHETIC data: 3D poses are stand-ins from a stored run, so sway, margin and time outside BOS are not meaningful. Rehearsal only.</div>' if mock else ''}
{''.join(f'<div class="note">{html.escape(w)}</div>' for w in warnings)}
<div class="note">Shortened 10 s demo protocol (standard BESS uses 20 s per stance). Scores are not comparable to published BESS norms.</div>
<p>Total BESS errors (posecam): <span class="total">{total_err if scored else "–"}</span>{"" if scored else ' <span class="sub">(not scored: offline recording, posecam did not run)</span>'}</p>
<table><thead><tr><th>Stance</th><th>BESS errors</th><th>Side-to-side sway (RMS)</th><th>Min margin</th><th>Time outside BOS</th><th>Quality · frames</th><th>Replay</th></tr></thead>
<tbody>{"".join(body) or '<tr><td colspan="7">No trials yet.</td></tr>'}</tbody></table>
<footer><b>How to read this.</b> BESS errors are counted by posecam (MediaPipe 2D landmarks, live). Sway, margin of stability
and time outside the base of support come from the 3D replay (SAM 3D Body, a subset of frames), are estimates, and
{"forward/back numbers are estimates because depth noise exceeded 1 cm; " if any_est else ""}excursions smaller than 3× the
side-to-side noise floor are marked uncertain. Where frames were missing at the deadline the replay interpolates between
reconstructed frames (see each trial's frame coverage). Two models on the same video were compared for agreement, not
validated against a reference system. Research demo, not a diagnostic device.</footer></div></body></html>"""
