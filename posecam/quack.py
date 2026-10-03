"""The duck quack, played without blocking the video loop.

Plays sounds/quack.wav (a real duck quack, trimmed to ~0.2 s so it starts
instantly). If that file is missing, it falls back to a synthesized quack.
To use a different sound, replace sounds/quack.wav with any short WAV file.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
import wave
from pathlib import Path

import numpy as np

QUACK_PATH = Path(__file__).parent / "sounds" / "quack.wav"
SYNTH_PATH = Path(__file__).parent / "sounds" / "quack_synth.wav"
RATE = 22050


def _resonate(x: np.ndarray, freq: float, q: float) -> np.ndarray:
    """Simple 2-pole resonator: emphasizes one frequency band (a vocal 'formant')."""
    w = 2 * np.pi * freq / RATE
    r = np.exp(-w / (2 * q))
    a1, a2 = -2 * r * np.cos(w), r * r
    y = np.zeros_like(x)
    for n in range(len(x)):
        y[n] = x[n] - a1 * (y[n - 1] if n > 0 else 0) - a2 * (y[n - 2] if n > 1 else 0)
    return y


def synth_quack(duration: float = 0.28) -> np.ndarray:
    t = np.arange(int(duration * RATE)) / RATE
    # pitch glides down like a quack: ~520 Hz -> ~300 Hz
    f0 = 300 + 220 * np.exp(-t / 0.08)
    phase = 2 * np.pi * np.cumsum(f0) / RATE
    saw = 2 * ((phase / (2 * np.pi)) % 1.0) - 1          # buzzy, harmonic-rich
    rasp = 1 + 0.35 * np.sin(2 * np.pi * 38 * t)          # throaty flutter
    src = saw * rasp
    # nasal "aa" formants
    voiced = 0.6 * _resonate(src, 1000, 6) + 0.4 * _resonate(src, 1900, 8) + 0.15 * src
    # envelope: fast attack, short hold, decay
    env = np.minimum(1, t / 0.012) * np.exp(-np.maximum(0, t - 0.06) / 0.09)
    y = voiced * env
    return (y / (np.abs(y).max() + 1e-9) * 0.8).astype(np.float32)


def ensure_quack_wav(path: Path = SYNTH_PATH) -> Path:
    """Write the synthesized fallback quack (used only if sounds/quack.wav is missing)."""
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        pcm = (synth_quack() * 32767).astype(np.int16)
        with wave.open(str(path), "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(RATE)
            wf.writeframes(pcm.tobytes())
    return path


def play_quack():
    """Play the quack asynchronously. Never raises."""
    try:
        path = QUACK_PATH if QUACK_PATH.exists() else ensure_quack_wav()
        if sys.platform == "win32":
            import winsound
            winsound.PlaySound(str(path), winsound.SND_FILENAME | winsound.SND_ASYNC)
            return
        for player in (["afplay"], ["paplay"], ["aplay", "-q"]):
            if shutil.which(player[0]):
                subprocess.Popen(player + [str(path)], stdout=subprocess.DEVNULL,
                                 stderr=subprocess.DEVNULL)
                return
        print("\a", end="", flush=True)   # no audio player found: terminal bell
    except Exception:
        pass


if __name__ == "__main__":
    import time
    print(f"quack -> {QUACK_PATH if QUACK_PATH.exists() else ensure_quack_wav()}")
    play_quack()
    time.sleep(0.6)
