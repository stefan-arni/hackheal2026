"""Offline stand-in for fal, backed by run folders of real (or SYNTHETIC) results.

    REPLAY_MOCK_FAL=data/fal_out/demo uv run uvicorn service.app:app --port 8017
    REPLAY_MOCK_FAL=data/fal_out/synthetic ...      (SYNTHETIC run, exact matches only)

Plugs into SamBodyClient as its `backend`: never billed, never written to the real fal cache
or spend log. Matching an uploaded image to a stored result:
  1. exact: sha256 of the stored source frame (or of its live-downscaled version)
  2. otherwise the stored result whose source frame looks most similar (normalised 18x32
     thumbnail, nearest in L2) — for rehearsals that stream a video at arbitrary times. Poses
     are then only approximately right for frames far from the stored ones.
2D metadata (focal, keypoints_2d, bbox) is rescaled to the uploaded image's size.

Latency model (env, for timing rehearsals):
  REPLAY_MOCK_LATENCY=4.4       mean seconds per frame (±15 % jitter); default 1.5–3.5 s uniform
  REPLAY_MOCK_PARALLEL=3        frames processed at once (fal's measured parallelism); default unlimited
  REPLAY_MOCK_STRAGGLER=12:150  the 12th call takes 150 s
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import io
import json
import os
import random
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from service.capture_defaults import LIVE_MAX_HEIGHT_PX
from service.images import downscale_jpeg
from service.runs import RESERVED_JSON

REPLAY_ROOT = Path(__file__).resolve().parents[1]


def thumb(img: Image.Image) -> np.ndarray:
    a = np.asarray(img.convert("L").resize((18, 32), Image.BILINEAR), float).ravel()
    return (a - a.mean()) / (a.std() + 1e-6)


class MockFal:
    def __init__(self, run_dirs, latency_s: tuple[float, float] = (1.5, 3.5)):
        if isinstance(run_dirs, (str, Path)):
            run_dirs = [run_dirs]
        self.records: list[dict] = []  # {dir, stem, rec, size}
        self.index: dict[str, int] = {}
        thumbs = []
        for d in (Path(x).resolve() for x in run_dirs):
            for p in sorted(d.glob("*.json")):
                if p.name in RESERVED_JSON:
                    continue
                rec = json.loads(p.read_text())
                if not rec.get("usable"):
                    continue
                src = self._source(d, rec)
                if src is None:
                    continue
                b = src.read_bytes()
                i = len(self.records)
                self.records.append({"dir": d, "stem": p.stem, "rec": rec})
                self.index[hashlib.sha256(b).hexdigest()] = i
                self.index[hashlib.sha256(downscale_jpeg(b, LIVE_MAX_HEIGHT_PX)[0]).hexdigest()] = i
                with Image.open(io.BytesIO(b)) as im:
                    thumbs.append(thumb(im))
        if not self.records:
            raise FileNotFoundError(f"no usable results with source frames in {run_dirs}")
        self.thumbs = np.stack(thumbs)
        self.latency_s = latency_s
        mean = os.environ.get("REPLAY_MOCK_LATENCY")
        self.mean = float(mean) if mean else None
        par = os.environ.get("REPLAY_MOCK_PARALLEL")
        self.parallel = asyncio.Semaphore(int(par)) if par else None
        st = os.environ.get("REPLAY_MOCK_STRAGGLER")
        self.straggler = tuple(float(x) for x in st.split(":")) if st else None
        self.calls = 0
        self.exact = self.nearest = 0

    @staticmethod
    def _source(d: Path, rec: dict) -> Path | None:
        """The image that was sent to fal for this record (synthetic input/, or data/frames/<run>/)."""
        for cand in (d / "input" / rec["frame"], REPLAY_ROOT / "data/frames" / d.name / rec["frame"]):
            if cand.exists():
                return cand
        return None

    def describe(self) -> str:
        lat = f"{self.mean:.1f} s ±15%" if self.mean else f"{self.latency_s[0]}–{self.latency_s[1]} s"
        par = getattr(self.parallel, "_value", None)
        s = f"latency {lat}, parallel {par or 'unlimited'}"
        if self.straggler:
            s += f", call #{int(self.straggler[0])} takes {self.straggler[1]:.0f} s"
        return s

    async def call(self, arguments: dict[str, Any]) -> dict[str, Any]:
        self.calls += 1
        n = self.calls
        jpeg = base64.b64decode(arguments["image_url"].split(",", 1)[1])
        if self.straggler and n == int(self.straggler[0]):
            delay = self.straggler[1]
        elif self.mean:
            delay = self.mean * random.uniform(0.85, 1.15)
        else:
            delay = random.uniform(*self.latency_s)
        if self.parallel is not None:
            async with self.parallel:
                await asyncio.sleep(delay)
        else:
            await asyncio.sleep(delay)
        i = self.index.get(hashlib.sha256(jpeg).hexdigest())
        with Image.open(io.BytesIO(jpeg)) as im:
            W, H = im.size
            if i is None:
                i = int(np.argmin(((self.thumbs - thumb(im)) ** 2).sum(axis=1)))
                self.nearest += 1
            else:
                self.exact += 1
        r = self.records[i]
        rec = r["rec"]
        meta = json.loads(json.dumps(rec["metadata"]))
        s = W / rec["image_size"][0]  # rescale 2D quantities to the uploaded image
        if abs(s - 1) > 1e-6:
            for p in meta.get("people", []):
                p["focal_length"] = p["focal_length"] * s
                p["keypoints_2d"] = [[c * s for c in kp] for kp in p["keypoints_2d"]]
                p["bbox"] = [c * s for c in p["bbox"]]
        vis = rec.get("vis_file")
        return {
            "meshes": [{"url": f"mock://{i}/{r['stem']}.ply"}],
            "visualization": {"url": f"mock://{i}/{vis}", "content_type": "image/jpeg"} if vis else None,
            "metadata": meta,
        }

    async def fetch(self, url: str) -> bytes | None:
        i, name = url.removeprefix("mock://").split("/", 1)
        return (self.records[int(i)]["dir"] / name).read_bytes()
