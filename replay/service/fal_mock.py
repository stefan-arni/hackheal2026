"""Offline stand-in for fal, backed by a run folder (e.g. the SYNTHETIC run).

    REPLAY_MOCK_FAL=data/fal_out/synthetic uv run uvicorn service.app:app

Patches fal_client.AsyncClient.subscribe, so SamBodyClient's real code path runs
(semaphore, data-URI encoding, retries). An uploaded image is matched to a frame by
the sha256 of the run's input/<stem>.jpg; unknown images come back with 0 people.
Downloads of mock:// URLs are served from the run folder.
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import os
import random
from pathlib import Path
from typing import Any

import fal_client

from service import fal_client_wrap


def install(run_dir: Path, latency_s: tuple[float, float] = (1.5, 3.5)) -> int:
    """Patch fal for this process. Returns the number of known frames."""
    run_dir = Path(run_dir).resolve()
    index = {hashlib.sha256(p.read_bytes()).hexdigest(): p.stem for p in (run_dir / "input").glob("*.jpg")}
    if not index:
        raise FileNotFoundError(f"no input/*.jpg in {run_dir}")

    async def subscribe(self, application: str, arguments: dict[str, Any], **kw) -> dict[str, Any]:
        assert application == fal_client_wrap.ENDPOINT
        jpeg = base64.b64decode(arguments["image_url"].split(",", 1)[1])
        await asyncio.sleep(random.uniform(*latency_s))
        stem = index.get(hashlib.sha256(jpeg).hexdigest())
        if stem is None:
            return {"meshes": [], "visualization": None, "metadata": {"num_people": 0, "people": []}}
        rec = json.loads((run_dir / f"{stem}.json").read_text())
        return {
            "meshes": [{"url": f"mock://{stem}.ply"}],
            "visualization": {"url": f"mock://{rec['vis_file']}", "content_type": "image/png"} if rec.get("vis_file") else None,
            "metadata": rec["metadata"],
        }

    real_get = fal_client_wrap.SamBodyClient._get

    async def get(self, url: str | None) -> bytes | None:
        if url and url.startswith("mock://"):
            return (run_dir / url.removeprefix("mock://")).read_bytes()
        return await real_get(self, url)

    fal_client.AsyncClient.subscribe = subscribe
    fal_client_wrap.SamBodyClient._get = get
    os.environ.setdefault("FAL_KEY", "mock")
    return len(index)
