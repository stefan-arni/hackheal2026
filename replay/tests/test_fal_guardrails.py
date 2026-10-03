"""fal budget guardrails, offline: a mock of fal's queue API behind httpx.MockTransport.

Nothing here touches the network, the real data/fal_cache or data/fal_spend.jsonl.
"""

import asyncio
import json
import re

import fal_client
import httpx
import pytest

from service.fal_budget import Budget, BudgetError, PriceNotConfirmed
from service.fal_cache import FalCache, image_key
from service.fal_client_wrap import CacheMiss, FalSubmitError, SamBodyClient

RESULT = {
    "meshes": [{"url": "https://cdn.test/mesh.ply"}],
    "visualization": {"url": "https://cdn.test/vis.png", "content_type": "image/png"},
    "metadata": {"num_people": 1, "people": [{"focal_length": 1000.0, "pred_cam_t": [0, 0, 2.5]}]},
}


class FakeQueue:
    """fal queue API. `submit_plan` is consumed one item per POST: a status code or an exception."""

    def __init__(self, submit_plan=()):
        self.submit_plan = list(submit_plan)
        self.posts = 0
        self.polls = 0
        self.n = 0

    def handler(self, request: httpx.Request) -> httpx.Response:
        url = f"{request.url.scheme}://{request.url.host}{request.url.path}"  # without the query string
        if request.method == "POST" and url.startswith("https://queue.fal.run/"):
            self.posts += 1
            step = self.submit_plan.pop(0) if self.submit_plan else 200
            if isinstance(step, Exception):
                raise step
            if step != 200:
                return httpx.Response(step, text=f"error {step}")
            self.n += 1
            base = f"https://queue.fal.run/fal-ai/sam-3/requests/req-{self.n}"
            return httpx.Response(200, json={"request_id": f"req-{self.n}", "response_url": base,
                                             "status_url": base + "/status", "cancel_url": base + "/cancel"})
        if re.search(r"/requests/[^/]+/status$", url):
            self.polls += 1
            return httpx.Response(200, json={"status": "COMPLETED", "logs": []})
        if re.search(r"/requests/[^/]+$", url):
            return httpx.Response(200, json=RESULT)
        if url == "https://cdn.test/mesh.ply":
            return httpx.Response(200, content=b"ply-bytes")
        if url == "https://cdn.test/vis.png":
            return httpx.Response(200, content=b"png-bytes")
        return httpx.Response(404)


class FakeFal:
    """Stands in for fal_client.AsyncClient: its authenticated client is our mock transport."""

    def __init__(self, client):
        self.client = client

    @property
    def _client(self):
        async def get():
            return self.client
        return get()

    async def get_handle(self, app, request_id):
        return fal_client.AsyncRequestHandle.from_request_id(self.client, app, request_id)


@pytest.fixture
def env(tmp_path, monkeypatch):
    monkeypatch.setattr("service.fal_client_wrap.asyncio.sleep", _no_sleep)
    cache = FalCache(tmp_path / "cache")

    def make(queue=None, *, live=True, confirmed=True, run="r1", **budget_kw):
        queue = queue or FakeQueue()
        budget = Budget(tmp_path / "spend.jsonl", tmp_path / "price.json", quiet=True, **budget_kw)
        if confirmed and not budget.price_path.exists():
            budget.confirm_price(0.015)
        sam = SamBodyClient(live=live, run=run, cache=cache, budget=budget, key="test-key", concurrency=4)
        client = httpx.AsyncClient(transport=httpx.MockTransport(queue.handler))
        sam._fal = FakeFal(client) if live else None
        sam._http = client
        return sam, budget, queue

    return make, cache


async def _no_sleep(*_):
    return None


def run(coro):
    return asyncio.run(coro)


# --- cache ----------------------------------------------------------------------


def test_identical_frame_is_never_sent_twice(env):
    make, cache = env
    sam, budget, q = make()
    a = run(sam.reconstruct(b"frame-1"))
    b = run(sam.reconstruct(b"frame-1"))
    assert q.posts == 1 and budget.billed() == 1
    assert not a.from_cache and b.from_cache and b.ply == b"ply-bytes" and b.num_people == 1
    assert cache.has(image_key(b"frame-1"))


def test_cache_is_shared_across_runs_and_clients(env):
    make, _ = env
    sam1, _, q1 = make(run="run-a")
    run(sam1.reconstruct(b"frame-1"))
    sam2, budget, q2 = make(run="run-b")
    assert run(sam2.reconstruct(b"frame-1")).from_cache
    assert q2.posts == 0 and budget.billed("run-b") == 0


def test_concurrent_duplicates_share_one_call(env):
    make, _ = env
    sam, budget, q = make()

    async def go():
        return await asyncio.gather(*(sam.reconstruct(b"same") for _ in range(5)))

    assert len(run(go())) == 5
    assert q.posts == 1 and budget.billed() == 1


def test_mask_is_part_of_the_key():
    assert image_key(b"img") != image_key(b"img", b"mask")
    assert image_key(b"img", b"mask") == image_key(b"img", b"mask")


# --- opt-in live ---------------------------------------------------------------------


def test_cache_only_by_default_sends_nothing(env):
    make, _ = env
    sam, budget, q = make(live=False)
    with pytest.raises(CacheMiss):
        run(sam.reconstruct(b"new-frame"))
    assert q.posts == 0 and budget.billed() == 0 and budget.entries() == []


def test_cache_only_still_serves_hits(env):
    make, _ = env
    run(make()[0].reconstruct(b"f"))
    sam, _, q = make(live=False)
    assert run(sam.reconstruct(b"f")).from_cache and q.posts == 0


# --- budget ---------------------------------------------------------------------------


def test_every_billed_call_is_logged(env):
    make, _ = env
    sam, budget, _ = make()
    run(sam.reconstruct(b"f1"))
    submitted = [e for e in budget.entries() if e["event"] == "submitted"]
    assert len(submitted) == 1
    e = submitted[0]
    assert e["billed"] is True and e["key"] == image_key(b"f1") and e["run"] == "r1"
    assert e["est_usd"] == 0.015 and e["request_id"] == "req-1" and "ts" in e


def test_run_cap(env):
    make, _ = env
    sam, budget, q = make(run_cap=2)
    run(sam.reconstruct(b"a"))
    run(sam.reconstruct(b"b"))
    with pytest.raises(BudgetError, match="run cap"):
        run(sam.reconstruct(b"c"))
    assert q.posts == 2
    other, _, _ = make(run="r2", run_cap=2)
    run(other.reconstruct(b"c"))  # a different run has its own allowance


def test_total_cap_and_override(env):
    make, _ = env
    sam, _, q = make(total_cap=1)
    run(sam.reconstruct(b"a"))
    with pytest.raises(BudgetError, match="total cap"):
        run(sam.reconstruct(b"b"))
    sam2, _, q2 = make(total_cap=1, override=True)
    run(sam2.reconstruct(b"b"))
    assert q2.posts == 1


def test_session_cap_is_a_hard_stop(env):
    make, _ = env
    sam, budget, q = make()
    run(sam.reconstruct(b"earlier"))  # spend before this session does not count
    budget2_sam, budget2, q2 = make(max_session_calls=2, override=True)

    async def go():
        return await asyncio.gather(*(budget2_sam.reconstruct(f"s{i}".encode()) for i in range(6)), return_exceptions=True)

    out = run(go())
    assert q2.posts == 2 and sum(isinstance(r, BudgetError) for r in out) == 4  # even with override


def test_caps_hold_under_concurrency(env):
    make, _ = env
    sam, budget, q = make(run_cap=3)

    async def go():
        return await asyncio.gather(*(sam.reconstruct(f"f{i}".encode()) for i in range(10)), return_exceptions=True)

    out = run(go())
    assert q.posts == 3 and budget.billed() == 3
    assert sum(isinstance(r, BudgetError) for r in out) == 7


def test_price_gate_allows_only_the_probe(env):
    make, _ = env
    sam, budget, q = make(confirmed=False)
    for i in range(3):
        run(sam.reconstruct(f"probe{i}".encode()))
    with pytest.raises(PriceNotConfirmed, match="confirm-price"):
        run(sam.reconstruct(b"fourth"))
    assert q.posts == 3
    budget.confirm_price(0.015)
    run(sam.reconstruct(b"fourth"))
    assert q.posts == 4


def test_confirm_batch_checks_caps_and_needs_yes(env, capsys):
    make, _ = env
    _, budget, _ = make(run_cap=5)
    budget.confirm_batch(3, "r1", yes=True)
    assert "about to send 3 new frames ≈ $0.045, total so far $0.000" in capsys.readouterr().out
    with pytest.raises(BudgetError, match="run cap"):
        budget.confirm_batch(6, "r1", yes=True)
    with pytest.raises(BudgetError, match="--yes"):  # pytest's stdin is not a terminal
        budget.confirm_batch(1, "r1", yes=False)


# --- retries ------------------------------------------------------------------------------


def test_retry_once_on_5xx(env):
    make, _ = env
    sam, budget, q = make(FakeQueue([503, 200]))
    run(sam.reconstruct(b"f"))
    assert q.posts == 2 and budget.billed() == 1  # the 503 was not billed


def test_max_two_attempts(env):
    make, _ = env
    sam, budget, q = make(FakeQueue([500, 502, 200]))
    with pytest.raises(FalSubmitError, match="HTTP 502"):
        run(sam.reconstruct(b"f"))
    assert q.posts == 2 and budget.billed() == 0


@pytest.mark.parametrize("code", [400, 401, 403, 404, 408, 409, 422, 429])
def test_never_retry_4xx(env, code):
    make, _ = env
    sam, budget, q = make(FakeQueue([code, 200]))
    with pytest.raises(FalSubmitError, match=f"HTTP {code}"):
        run(sam.reconstruct(b"f"))
    assert q.posts == 1 and budget.billed() == 0


def test_connect_error_retried_and_not_billed(env):
    make, _ = env
    sam, budget, q = make(FakeQueue([httpx.ConnectError("down"), 200]))
    run(sam.reconstruct(b"f"))
    assert q.posts == 2 and budget.billed() == 1


def test_lost_response_counts_as_maybe_billed(env):
    make, budget_cache = env
    sam, budget, q = make(FakeQueue([httpx.ReadTimeout("lost"), httpx.ReadTimeout("lost")]))
    with pytest.raises(FalSubmitError):
        run(sam.reconstruct(b"f"))
    assert q.posts == 2 and budget.billed() == 2  # conservative: both may have been billed


def test_billed_but_unfetched_result_is_recovered_not_resubmitted(env):
    make, cache = env
    sam, budget, q = make()
    # a previous process submitted and was billed, then died before caching the result
    budget.log("submitted", key=image_key(b"f"), run="r1", billed=True, est_usd=0.015, request_id="req-old")
    res = run(sam.reconstruct(b"f"))
    assert q.posts == 0 and budget.billed() == 1 and res.request_id == "req-old"
    assert [e["event"] for e in budget.entries()][-2:] == ["recovered", "done"]
    assert cache.has(image_key(b"f"))


def test_cache_write_is_complete(env):
    make, cache = env
    sam, _, _ = make()
    run(sam.reconstruct(b"f"))
    d = cache.path(image_key(b"f"))
    entry = json.loads((d / "response.json").read_text())
    assert (d / "mesh.ply").read_bytes() == b"ply-bytes" and (d / "vis.png").read_bytes() == b"png-bytes"
    assert entry["request_id"] == "req-1" and entry["params"]["include_3d_keypoints"] is False


# --- priority (live trials: error bursts before uniform frames) ------------------------------

from service.fal_client_wrap import PrioritySemaphore  # noqa: E402


def test_priority_semaphore_order_and_limit():
    async def go():
        sem, order, active, peak = PrioritySemaphore(2), [], [0], [0]

        async def job(name, prio, hold=0.01):
            async with sem(prio):
                active[0] += 1
                peak[0] = max(peak[0], active[0])
                order.append(name)
                await asyncio.sleep(hold)
                active[0] -= 1

        tasks = [asyncio.create_task(job(f"u{i}", 1)) for i in range(6)]  # uniform frames queue up
        await asyncio.sleep(0.001)
        tasks += [asyncio.create_task(job(f"b{i}", 0)) for i in range(3)]  # then a burst arrives
        await asyncio.gather(*tasks)
        return order, peak[0]

    order, peak = asyncio.run(go())
    assert peak == 2
    assert order[:2] == ["u0", "u1"]  # already running when the burst arrived
    assert order[2:5] == ["b0", "b1", "b2"]  # burst jumps the queued uniform frames
    assert order[5:] == ["u2", "u3", "u4", "u5"]  # FIFO within a priority


def test_priority_semaphore_cancelled_waiter_does_not_leak():
    async def go():
        sem = PrioritySemaphore(1)
        await sem.acquire()
        waiter = asyncio.create_task(sem.acquire(0))
        await asyncio.sleep(0)
        waiter.cancel()
        await asyncio.gather(waiter, return_exceptions=True)
        sem.release()
        await asyncio.wait_for(sem.acquire(1), 0.1)  # slot is free again
        return sem._value

    assert asyncio.run(go()) == 0
