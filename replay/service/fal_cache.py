"""Content-addressed cache of fal results, shared by every run, tool and the service.

    data/fal_cache/<key>/response.json   fal response + request params + latency + request_id
    data/fal_cache/<key>/mesh.ply
    data/fal_cache/<key>/vis.<ext>

key = sha1(JPEG bytes); with a mask, sha1(JPEG) + "_m" + sha1(mask)[:12], since the
mask changes the result. An identical frame is never sent to fal twice.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import time
from pathlib import Path
from typing import Any

from service.sam_result import SamResult

REPLAY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CACHE_DIR = REPLAY_ROOT / "data/fal_cache"


def image_key(jpeg: bytes, mask: bytes | None = None) -> str:
    key = hashlib.sha1(jpeg).hexdigest()
    if mask is not None:
        key += "_m" + hashlib.sha1(mask).hexdigest()[:12]
    return key


class FalCache:
    def __init__(self, root: Path = DEFAULT_CACHE_DIR):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def path(self, key: str) -> Path:
        return self.root / key

    def has(self, key: str) -> bool:
        return (self.path(key) / "response.json").exists()

    def get(self, key: str) -> SamResult | None:
        d = self.path(key)
        if not (d / "response.json").exists():
            return None
        entry = json.loads((d / "response.json").read_text())
        vis = next(d.glob("vis.*"), None)
        return SamResult(
            response=entry["response"],
            latency_s=0.0,
            ply=(d / "mesh.ply").read_bytes() if (d / "mesh.ply").exists() else None,
            visualization=vis.read_bytes() if vis else None,
            key=key,
            from_cache=True,
            request_id=entry.get("request_id"),
        )

    def put(self, key: str, res: SamResult, *, params: dict[str, Any], vis_ext: str = ".png") -> None:
        """Atomic: files go into a temp dir that is renamed into place."""
        if self.has(key):
            return
        tmp = Path(tempfile.mkdtemp(prefix=f".{key}.", dir=self.root))
        if res.ply is not None:
            (tmp / "mesh.ply").write_bytes(res.ply)
        if res.visualization is not None:
            (tmp / f"vis{vis_ext}").write_bytes(res.visualization)
        (tmp / "response.json").write_text(json.dumps({
            "key": key,
            "request_id": res.request_id,
            "params": params,
            "latency_s": res.latency_s,
            "cached_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "response": res.response,
        }, indent=1))
        try:
            os.rename(tmp, self.path(key))
        except OSError:  # another writer won the race; keep theirs
            for p in tmp.iterdir():
                p.unlink()
            tmp.rmdir()

    def keys(self) -> list[str]:
        return sorted(p.name for p in self.root.iterdir() if p.is_dir() and not p.name.startswith("."))
