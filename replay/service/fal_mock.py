"""Offline stand-in for fal, backed by a run folder (e.g. the SYNTHETIC run).

    REPLAY_MOCK_FAL=data/fal_out/synthetic uv run uvicorn service.app:app --port 8017

Plugs into SamBodyClient as its `backend`: never billed, never written to the real
fal cache or spend log. An uploaded image is matched to a frame by the sha256 of the
run's input/<stem>.jpg; unknown images come back with 0 people.
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import random
from pathlib import Path
from typing import Any

from service.capture_defaults import LIVE_MAX_HEIGHT_PX
from service.images import downscale_jpeg


class MockFal:
    def __init__(self, run_dir: Path, latency_s: tuple[float, float] = (1.5, 3.5)):
        self.run_dir = Path(run_dir).resolve()
        self.latency_s = latency_s
        # match both the original frame and the service's live-downscaled version of it
        self.index = {}
        for p in (self.run_dir / "input").glob("*.jpg"):
            b = p.read_bytes()
            self.index[hashlib.sha256(b).hexdigest()] = p.stem
            self.index[hashlib.sha256(downscale_jpeg(b, LIVE_MAX_HEIGHT_PX)[0]).hexdigest()] = p.stem
        if not self.index:
            raise FileNotFoundError(f"no input/*.jpg in {self.run_dir}")
        self.calls = 0

    async def call(self, arguments: dict[str, Any]) -> dict[str, Any]:
        self.calls += 1
        jpeg = base64.b64decode(arguments["image_url"].split(",", 1)[1])
        await asyncio.sleep(random.uniform(*self.latency_s))
        stem = self.index.get(hashlib.sha256(jpeg).hexdigest())
        if stem is None:
            return {"meshes": [], "visualization": None, "metadata": {"num_people": 0, "people": []}}
        rec = json.loads((self.run_dir / f"{stem}.json").read_text())
        vis = rec.get("vis_file")
        return {
            "meshes": [{"url": f"mock://{stem}.ply"}],
            "visualization": {"url": f"mock://{vis}", "content_type": "image/png"} if vis else None,
            "metadata": rec["metadata"],
        }

    async def fetch(self, url: str) -> bytes | None:
        return (self.run_dir / url.removeprefix("mock://")).read_bytes()
