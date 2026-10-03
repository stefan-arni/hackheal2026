"""Budget-guarded async client for fal's SAM 3D Body endpoint.

Order of lookups for every frame (key = sha1 of the JPEG, see fal_cache):
  1. shared cache data/fal_cache/<key>/             -> free
  2. a request_id already submitted for this key    -> fetch its result (free, never resubmit)
  3. live=False                                     -> CacheMiss (nothing is sent)
  4. live=True -> Budget.reserve (caps, price gate) -> ONE billed submit, logged to fal_spend.jsonl

Retries on the billed submit: max 2 attempts, only for 5xx and network errors, never
4xx. We POST the submit ourselves because fal_client.submit retries internally up to
10 times (including 408/409/429). Polling and downloads are free and use fal_client's
own retries. fal's backup-domain failover only fires on connect errors (request never
delivered), so it can't double-bill.

    async with SamBodyClient(live=True, run="probe") as sam:
        res = await sam.reconstruct(jpeg_bytes)
"""

from __future__ import annotations

import asyncio
import os
import time
from pathlib import Path
from typing import Any, Protocol

import fal_client
import httpx
from dotenv import load_dotenv

from service.fal_budget import Budget
from service.fal_cache import FalCache, image_key
from service.runs import vis_extension
from service.sam_result import SamResult

ENDPOINT = "fal-ai/sam-3/3d-body"
QUEUE_URL = "https://queue.fal.run/" + ENDPOINT
PARAMS = {
    "export_meshes": True,
    "include_3d_keypoints": False,  # True bakes marker spheres into the mesh
    "include_mhr_params": False,  # lean metadata: keypoints + camera only
}
MAX_SUBMIT_ATTEMPTS = 2
DEFAULT_CONCURRENCY = 2  # set REPLAY_FAL_CONCURRENCY to your account's fal limit

REPLAY_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(REPLAY_ROOT / ".env")

__all__ = ["SamBodyClient", "SamResult", "CacheMiss", "FalSubmitError", "ENDPOINT", "PARAMS", "REPLAY_ROOT"]


class CacheMiss(LookupError):
    def __init__(self, key: str):
        super().__init__(f"not in fal cache (live calls disabled): {key}")
        self.key = key


class FalSubmitError(RuntimeError):
    pass


class Backend(Protocol):
    """Offline stand-in for fal (fal_mock). Not billed, not cached."""

    async def call(self, arguments: dict[str, Any]) -> dict[str, Any]: ...
    async def fetch(self, url: str) -> bytes | None: ...


class SamBodyClient:
    def __init__(
        self,
        *,
        live: bool = False,
        run: str = "adhoc",
        cache: FalCache | None = None,
        budget: Budget | None = None,
        concurrency: int | None = None,
        key: str | None = None,
        timeout_s: float = 120.0,
        backend: Backend | None = None,
    ):
        self.live, self.run, self.backend = live, run, backend
        self.cache = cache if cache is not None else (None if backend else FalCache())
        self.budget = budget if budget is not None else Budget()
        n = concurrency or int(os.environ.get("REPLAY_FAL_CONCURRENCY", DEFAULT_CONCURRENCY))
        self._sem = asyncio.Semaphore(n)
        self._timeout = timeout_s
        self._http = httpx.AsyncClient(timeout=timeout_s, follow_redirects=True)
        self._inflight: dict[str, asyncio.Future] = {}
        self._fal: fal_client.AsyncClient | None = None
        if live and backend is None:
            key = key or os.environ.get("FAL_KEY")
            if not key:
                raise RuntimeError(f"live fal calls need FAL_KEY in {REPLAY_ROOT / '.env'}")
            self._fal = fal_client.AsyncClient(key=key, default_timeout=timeout_s)

    async def __aenter__(self) -> SamBodyClient:
        return self

    async def __aexit__(self, *exc) -> None:
        await self.aclose()

    async def aclose(self) -> None:
        await self._http.aclose()

    # --- public ----------------------------------------------------------------

    async def reconstruct(self, jpeg: bytes, mask_png: bytes | None = None, *, run: str | None = None) -> SamResult:
        """`run` (default self.run) is what the per-run budget cap counts against."""
        key = image_key(jpeg, mask_png)
        if self.cache is not None and (hit := self.cache.get(key)) is not None:
            return hit
        if key in self._inflight:  # same frame requested twice concurrently: share one call
            return await asyncio.shield(self._inflight[key])
        fut = asyncio.get_running_loop().create_future()
        self._inflight[key] = fut
        try:
            res = await self._reconstruct_uncached(key, jpeg, mask_png, run or self.run)
            fut.set_result(res)
            return res
        except BaseException as e:
            fut.set_exception(e)
            fut.exception()  # mark retrieved
            raise
        finally:
            del self._inflight[key]

    async def warmup(self, jpeg: bytes) -> float:
        """Run one frame (cached if seen before). Returns fal latency, 0 on a cache hit."""
        return (await self.reconstruct(jpeg)).latency_s

    # --- internals -------------------------------------------------------------

    def _arguments(self, jpeg: bytes, mask_png: bytes | None) -> dict[str, Any]:
        args: dict[str, Any] = {"image_url": fal_client.encode(jpeg, "image/jpeg"), **PARAMS}
        if mask_png is not None:
            args["mask_url"] = fal_client.encode(mask_png, "image/png")
        return args

    async def _reconstruct_uncached(self, key: str, jpeg: bytes, mask_png: bytes | None, run: str) -> SamResult:
        t0 = time.perf_counter()
        request_id = None
        if self.backend is not None:
            async with self._sem:
                response = await self.backend.call(self._arguments(jpeg, mask_png))
        elif (pending := self.budget.pending_request_id(key)) is not None:
            # Already paid for: fetch that result instead of submitting again.
            if self._fal is None:
                raise CacheMiss(key)
            try:
                async with self._sem:
                    handle = await self._fal.get_handle(ENDPOINT, pending)
                    response, request_id = await handle.get(), pending
            except Exception as e:
                raise FalSubmitError(
                    f"{key}: could not fetch already-billed request {pending} ({e!r}). To pay for it again: "
                    f"uv run python tools/fal_budget.py forget {key}"
                ) from e
            self.budget.log("recovered", key=key, run=run, request_id=pending, billed=False)
        elif not self.live:
            raise CacheMiss(key)
        else:
            async with self._sem:  # bounds requests in flight at fal (submit + processing)
                async with self.budget.reserve(run):
                    handle = await self._submit(key, self._arguments(jpeg, mask_png), run)
                request_id = handle.request_id
                try:
                    response = await handle.get()  # polling: free, fal_client retries it
                except Exception as e:
                    self.budget.log("failed_after_submit", key=key, run=run, request_id=request_id,
                                    billed=False, error=repr(e))
                    raise
        latency = time.perf_counter() - t0
        res = SamResult(response=response, latency_s=latency, key=key, request_id=request_id)
        res.ply, res.visualization = await asyncio.gather(self._get(res.mesh_url), self._get(res.visualization_url))
        if self.backend is None:
            self.budget.log("done", key=key, run=run, request_id=request_id, billed=False,
                            latency_s=round(latency, 3), num_people=res.num_people)
        if self.cache is not None:
            self.cache.put(key, res, params=PARAMS, vis_ext=vis_extension(response))
        return res

    async def _submit(self, key: str, arguments: dict[str, Any], run: str) -> fal_client.AsyncRequestHandle:
        """The only billed step. Max 2 attempts; retry only on 5xx / network errors."""
        assert self._fal is not None
        client = await self._fal._client  # fal's authenticated httpx client (fal-client 1.0.x)
        usd, _ = self.budget.price()
        for attempt in range(1, MAX_SUBMIT_ATTEMPTS + 1):
            last = attempt == MAX_SUBMIT_ATTEMPTS
            base = {"key": key, "run": run, "attempt": attempt}
            if attempt > 1:
                self.budget.check(0, run)  # a "maybe billed" failure may have used the slot
            try:
                r = await client.post(QUEUE_URL, json=arguments, timeout=self._timeout)
            except (httpx.ConnectError, httpx.ConnectTimeout) as e:  # never reached fal
                self.budget.log("submit_failed", **base, billed=False, error=repr(e))
                if last:
                    raise FalSubmitError(f"submit failed after {attempt} attempts: {e!r}") from e
            except httpx.TransportError as e:  # sent, response lost: may have been billed
                self.budget.log("submit_failed", **base, billed="maybe", est_usd=usd, error=repr(e))
                if last:
                    raise FalSubmitError(f"submit failed after {attempt} attempts: {e!r}") from e
            else:
                if r.status_code < 300:
                    data = r.json()
                    self.budget.log("submitted", **base, billed=True, est_usd=usd, request_id=data["request_id"])
                    return fal_client.AsyncRequestHandle(
                        request_id=data["request_id"], response_url=data["response_url"],
                        status_url=data["status_url"], cancel_url=data["cancel_url"], client=client,
                    )
                self.budget.log("submit_failed", **base, billed=False, status=r.status_code, error=r.text[:300])
                if r.status_code < 500 or last:  # never retry 4xx
                    raise FalSubmitError(f"submit HTTP {r.status_code}: {r.text[:300]}")
            await asyncio.sleep(1.0)
        raise AssertionError("unreachable")

    async def _get(self, url: str | None) -> bytes | None:
        if url is None:
            return None
        if self.backend is not None:
            return await self.backend.fetch(url)
        for attempt in range(3):  # downloads are free
            try:
                r = await self._http.get(url)
                r.raise_for_status()
                return r.content
            except httpx.TransportError:
                if attempt == 2:
                    raise
                await asyncio.sleep(0.5 * 2**attempt)
        raise AssertionError("unreachable")
