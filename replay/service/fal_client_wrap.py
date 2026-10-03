"""Async wrapper around fal's SAM 3D Body endpoint.

One SamBodyClient per process. Concurrency against fal is capped by a semaphore
(check the account's concurrency limit); result downloads happen outside it so
they never hold a fal slot.

    async with SamBodyClient() as sam:
        res = await sam.reconstruct(jpeg_bytes)
        res.num_people, res.person["pred_cam_t"], res.ply
"""

from __future__ import annotations

import asyncio
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import fal_client
import httpx
from dotenv import load_dotenv

ENDPOINT = "fal-ai/sam-3/3d-body"

REPLAY_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(REPLAY_ROOT / ".env")


@dataclass
class SamResult:
    response: dict[str, Any]  # raw fal response: file URLs + metadata
    latency_s: float  # fal request time (queue + inference), excludes downloads
    ply: bytes | None = None  # first person's mesh
    visualization: bytes | None = None  # original + keypoints + mesh + side view

    @property
    def metadata(self) -> dict[str, Any]:
        return self.response.get("metadata") or {}

    @property
    def num_people(self) -> int:
        return int(self.metadata.get("num_people", 0))

    @property
    def person(self) -> dict[str, Any] | None:
        people = self.metadata.get("people") or []
        return people[0] if people else None

    @property
    def mesh_url(self) -> str | None:
        meshes = self.response.get("meshes") or []
        return meshes[0]["url"] if meshes else None

    @property
    def visualization_url(self) -> str | None:
        vis = self.response.get("visualization")
        return vis["url"] if vis else None


class SamBodyClient:
    def __init__(
        self,
        concurrency: int = 8,
        key: str | None = None,
        timeout_s: float = 120.0,
        retries: int = 2,
    ):
        key = key or os.environ.get("FAL_KEY")
        if not key:
            raise RuntimeError(f"FAL_KEY not set; put it in {REPLAY_ROOT / '.env'}")
        self._fal = fal_client.AsyncClient(key=key, default_timeout=timeout_s)
        self._sem = asyncio.Semaphore(concurrency)
        self._http = httpx.AsyncClient(timeout=timeout_s, follow_redirects=True)
        self._retries = retries

    async def __aenter__(self) -> SamBodyClient:
        return self

    async def __aexit__(self, *exc) -> None:
        await self.aclose()

    async def aclose(self) -> None:
        await self._http.aclose()

    async def reconstruct(
        self,
        jpeg: bytes,
        mask_png: bytes | None = None,
        *,
        download: bool = True,
    ) -> SamResult:
        """Run one frame through SAM 3D Body.

        `mask_png` (white = person) makes SAM skip its own detection. With
        `download`, the first person's .ply and the visualization image are
        fetched concurrently.
        """
        args: dict[str, Any] = {
            "image_url": fal_client.encode(jpeg, "image/jpeg"),
            "export_meshes": True,
            "include_3d_keypoints": False,  # True bakes marker spheres into the mesh
            "include_mhr_params": False,  # lean metadata: keypoints + camera only
        }
        if mask_png is not None:
            args["mask_url"] = fal_client.encode(mask_png, "image/png")

        async with self._sem:
            t0 = time.perf_counter()
            response = await self._subscribe_with_retry(args)
            latency = time.perf_counter() - t0

        res = SamResult(response=response, latency_s=latency)
        if download:
            res.ply, res.visualization = await asyncio.gather(
                self._get(res.mesh_url), self._get(res.visualization_url)
            )
        return res

    async def warmup(self, jpeg: bytes) -> float:
        """Fire one request to avoid a cold start right before the demo. Returns latency."""
        return (await self.reconstruct(jpeg, download=False)).latency_s

    async def _subscribe_with_retry(self, args: dict[str, Any]) -> dict[str, Any]:
        for attempt in range(self._retries + 1):
            try:
                return await self._fal.subscribe(ENDPOINT, args)
            except (fal_client.FalClientTimeoutError, httpx.TransportError):
                if attempt == self._retries:
                    raise
            except fal_client.FalClientHTTPError as e:
                if attempt == self._retries or e.status_code < 500:
                    raise
            await asyncio.sleep(0.5 * 2**attempt)
        raise AssertionError("unreachable")

    async def _get(self, url: str | None) -> bytes | None:
        if url is None:
            return None
        r = await self._http.get(url)
        r.raise_for_status()
        return r.content
