"""One SAM 3D Body result (live from fal, from the cache, or from the mock)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class SamResult:
    response: dict[str, Any]  # raw fal response: file URLs + metadata
    latency_s: float  # fal request time (queue + inference); 0 for cache hits
    ply: bytes | None = None  # first person's mesh
    visualization: bytes | None = None  # original + keypoints + mesh + side view
    key: str | None = None  # content hash (fal_cache.image_key)
    from_cache: bool = False
    request_id: str | None = None

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
