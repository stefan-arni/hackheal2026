"""Image helpers shared by the service and the fal mock (deterministic, so hashes are stable)."""

from __future__ import annotations

import io

from PIL import Image


def downscale_jpeg(jpeg: bytes, max_h: int, quality: int = 92) -> tuple[bytes, list[int], float]:
    """JPEG no taller than max_h. Returns (jpeg, [w, h], scale); unchanged bytes if already small."""
    with Image.open(io.BytesIO(jpeg)) as im:
        W, H = im.size
        if H <= max_h:
            return jpeg, [W, H], 1.0
        s = max_h / H
        size = (round(W * s), max_h)
        buf = io.BytesIO()
        im.convert("RGB").resize(size, Image.LANCZOS).save(buf, "JPEG", quality=quality)
    return buf.getvalue(), list(size), s
