"""BESS side panel for the laptop test client: buttons for the three tests,
live countdown / timer / errors, and the score table.

Click the buttons, or use keys: 1 feet together, 2 tandem, 3 single leg,
n switch non-dominant leg, x cancel, 0 reset scores.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from bess import ERROR_LABELS, STANCES

PANEL_W = 320
BG = (32, 32, 36)
FG = (235, 235, 235)
DIM = (150, 150, 150)
ACCENT = (255, 190, 60)
GOOD = (90, 210, 90)
BAD = (70, 70, 235)


@dataclass
class Button:
    key: str
    label: str
    x0: int
    y0: int
    x1: int
    y1: int

    def hit(self, x, y):
        return self.x0 <= x <= self.x1 and self.y0 <= y <= self.y1


class BessPanel:
    def __init__(self, nondominant: str = "left"):
        self.nondominant = nondominant
        self.buttons: list[Button] = []
        self.clicked: list[str] = []
        self.session = {"scores": {s: None for s in STANCES}, "total": 0, "complete": False}
        self.last_result: dict | None = None
        self.message: str | None = None

    # ---- input ----

    def on_mouse(self, event, x, y, video_w: int):
        if event != cv2.EVENT_LBUTTONDOWN or x < video_w:
            return
        for b in self.buttons:
            if b.hit(x - video_w, y):
                self.clicked.append(b.key)

    def key_action(self, key: int) -> str | None:
        return {ord("1"): "double", ord("2"): "tandem", ord("3"): "single", ord("n"): "toggle_nd",
                ord("x"): "cancel", ord("0"): "reset"}.get(key)

    def take_actions(self) -> list[str]:
        a, self.clicked = self.clicked, []
        return a

    def command_for(self, action: str) -> dict | None:
        """Turn a button/key action into a server command (or handle it locally)."""
        if action in STANCES:
            self.last_result = None
            return {"type": "bess_start", "stance": action, "nondominant": self.nondominant}
        if action == "toggle_nd":
            self.nondominant = "right" if self.nondominant == "left" else "left"
            return None
        if action == "cancel":
            return {"type": "bess_cancel"}
        if action == "reset":
            self.last_result = None
            return {"type": "bess_reset"}
        return None

    # ---- state from server ----

    def update(self, b: dict | None):
        if not b:
            return
        self.session = b.get("session", self.session)
        for ev in b.get("events", []):
            if ev["kind"] == "bess_done":
                self.last_result = ev
                self.session = ev["session"]
                self.message = None
            elif ev["kind"] == "bess_failed":
                self.message = ev["reason"]
            elif ev["kind"] == "bess_started":
                self.message = None

    # ---- drawing ----

    def render(self, frame: np.ndarray, b: dict | None) -> np.ndarray:
        h = frame.shape[0]
        p = np.full((h, PANEL_W, 3), BG, np.uint8)
        y = 28
        self._text(p, "BESS balance test", 14, y, 0.65, ACCENT, 2)
        y += 26
        self._text(p, f"Non-dominant leg: {self.nondominant} (n)", 14, y, 0.45, DIM)
        y += 14

        self.buttons = []
        phase = (b or {}).get("phase", "idle")
        active = (b or {}).get("stance")
        for i, (key, label) in enumerate(STANCES.items()):
            y0 = y + 8 + i * 46
            self._button(p, key, f"{i + 1}  {label}", 14, y0, PANEL_W - 14, y0 + 38,
                         highlight=(key == active and phase != "idle"))
        y = y + 8 + 3 * 46 + 4
        half = (PANEL_W - 14 * 3) // 2
        self._button(p, "cancel", "x Cancel", 14, y, 14 + half, y + 30, small=True)
        self._button(p, "reset", "0 Reset scores", 28 + half, y, PANEL_W - 14, y + 30, small=True)
        y += 52
        limit = h - 132          # score table is pinned below this

        if phase == "countdown":
            self._text(p, f"{STANCES[active]}: get in position", 14, y, 0.55, FG)
            self._text(p, "hands on hips", 14, y + 22, 0.5, DIM)
            if b.get("waiting_for_view"):
                self._text(p, "Waiting to see the whole body...", 14, y + 52, 0.5, BAD)
            else:
                big = min(1.8, max(0.9, (limit - y - 30) / 40))
                self._text(p, f"{b['countdown_left']:.0f}", PANEL_W // 2 - 20,
                           min(y + 30 + int(36 * big), limit - 8), big, ACCENT, 4)
        elif phase == "running":
            self._text(p, f"{STANCES[active]}", 14, y, 0.6, FG, 2)
            self._text(p, f"{b['time_left']:.1f}s", 200, y, 0.75, ACCENT, 2)
            frac = 1 - b["time_left"] / 20.0
            cv2.rectangle(p, (14, y + 10), (PANEL_W - 14, y + 18), (70, 70, 70), -1)
            cv2.rectangle(p, (14, y + 10), (14 + int((PANEL_W - 28) * frac), y + 18), ACCENT, -1)
            y += 46
            self._text(p, f"Errors: {b['errors']}", 14, y, 0.8, BAD if b["errors"] else GOOD, 2)
            hips = b.get("hip_angles") or {}
            if hips:
                txt = "hip " + "  ".join(f"{s[0].upper()} {hips[s]:.0f}" for s in ("left", "right")
                                         if s in hips) + " deg"
                over = max(hips.values()) > 30
                self._text(p, txt, 160, y - 4, 0.45, BAD if over else DIM)
            y += 26
            lines = [(f"now: {ERROR_LABELS[k]}", 0.48, BAD) for k in b.get("active", [])]
            lines += [(f"{e['t']:5.1f}s {'+1' if e['counted'] else ' 0'} {e['label']}", 0.45,
                       FG if e["counted"] else DIM) for e in reversed(b.get("log", [])[-4:])]
            lines += [("! " + w, 0.42, ACCENT) for w in b.get("warnings", [])[:2]]
            for text, scale, color in lines:
                if y > limit - 6:
                    break
                y = self._wrapped(p, text, 14, y, scale, color, max_y=limit - 6)
        elif self.last_result:
            r = self.last_result
            self._text(p, f"{r['label']}: {r['errors']} error{'s' * (r['errors'] != 1)}",
                       14, y, 0.6, FG, 2)
            y += 24
            for k, n in r["by_type"].items():
                if n and y < limit - 6:
                    self._text(p, f"  {ERROR_LABELS[k]}: {n}", 14, y, 0.45, FG)
                    y += 19
            y += 6
        elif self.message:
            y = self._wrapped(p, self.message, 14, y, 0.45, BAD)
        else:
            self._text(p, "Press a test to start.", 14, y, 0.5, DIM)
            y += 24

        # score table, pinned to the bottom
        ty = limit + 22
        cv2.rectangle(p, (0, limit), (PANEL_W, h), BG, -1)
        cv2.line(p, (14, limit + 4), (PANEL_W - 14, limit + 4), (80, 80, 80), 1)
        self._text(p, "Score (errors)", 14, ty, 0.55, ACCENT, 2)
        for i, (key, label) in enumerate(STANCES.items()):
            v = self.session["scores"].get(key)
            self._text(p, label, 24, ty + 22 + i * 21, 0.5, FG)
            self._text(p, "-" if v is None else str(v), PANEL_W - 60, ty + 22 + i * 21, 0.55,
                       DIM if v is None else FG, 2)
        tot = self.session["total"]
        self._text(p, "Total" + ("" if self.session["complete"] else " (so far)"),
                   24, ty + 22 + 3 * 21 + 6, 0.55, ACCENT, 2)
        self._text(p, str(tot), PANEL_W - 60, ty + 22 + 3 * 21 + 6, 0.65, ACCENT, 2)
        return np.hstack([frame, p])

    def _button(self, p, key, label, x0, y0, x1, y1, highlight=False, small=False):
        cv2.rectangle(p, (x0, y0), (x1, y1), ACCENT if highlight else (70, 70, 78), -1)
        cv2.rectangle(p, (x0, y0), (x1, y1), (110, 110, 120), 1)
        scale = 0.45 if small else 0.58
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)
        cv2.putText(p, label, (x0 + 10, (y0 + y1 + th) // 2), cv2.FONT_HERSHEY_SIMPLEX, scale,
                    (20, 20, 20) if highlight else FG, 1 + (not small), cv2.LINE_AA)
        self.buttons.append(Button(key, label, x0, y0, x1, y1))

    @staticmethod
    def _text(p, s, x, y, scale, color, thick=1):
        cv2.putText(p, s, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)

    def _wrapped(self, p, s, x, y, scale, color, width=36, max_y=10_000):
        words, line = s.split(), ""
        for w in words:
            if len(line) + len(w) + 1 > width:
                if y <= max_y:
                    self._text(p, line, x, y, scale, color)
                y, line = y + 18, w
            else:
                line = f"{line} {w}".strip()
        if line and y <= max_y:
            self._text(p, line, x, y, scale, color)
        return y + 18
