"""비동기 분석 작업 API (scratchpad/jobs_contract.md).

- 시작 → 폴링 → done (결과는 동기 응답과 같은 모양)
- 같은 요청 합치기 (transform_photo 1회)
- 실패 시 status error + 안전한 메시지
- 없는 작업 404, 다른 uid 404
- 대기열 가득 → 503 busy
- TTL 만료 → 404
- 워커 예외가 풀을 죽이지 않음
"""

import base64
import io
import threading
import time

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

import auth
import jobs
import server

SECRET = "s" * 48


@pytest.fixture(autouse=True)
def _env(monkeypatch):
    monkeypatch.setenv("GAMDO_SESSION_SECRET", SECRET)
    monkeypatch.delenv("GAMDO_AUTH_REQUIRED", raising=False)
    monkeypatch.setattr(server, "APP_TOKEN", "")
    monkeypatch.setattr(server, "get_reference_image_paths", lambda _uid: [])
    store = jobs.JobStore(max_workers=2, max_queued=20)
    monkeypatch.setattr(server, "_job_store", store)
    yield store
    store.shutdown(wait=True)


@pytest.fixture
def client():
    return TestClient(server.app)


def _photo_b64(size=(160, 120), seed=0) -> str:
    rng = np.random.default_rng(seed)
    arr = np.full((size[1], size[0], 3), 140, np.float32) + rng.normal(0, 8, (size[1], size[0], 3))
    buf = io.BytesIO()
    Image.fromarray(arr.clip(0, 255).astype(np.uint8)).save(buf, format="JPEG")
    return base64.b64encode(buf.getvalue()).decode()


def _bearer(uid: str) -> dict:
    return {"Authorization": f"Bearer {auth.issue_session(uid)[0]}"}


def _poll(client, job_id, headers=None, timeout=30.0) -> dict:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        r = client.get(f"/api/jobs/{job_id}", headers=headers or {})
        assert r.status_code == 200, r.text
        body = r.json()
        if body["status"] in ("done", "error"):
            return body
        time.sleep(0.05)
    raise AssertionError("job did not finish")


def _assert_error(resp, status: int, code: str):
    assert resp.status_code == status, resp.text
    body = resp.json()
    assert body["success"] is False
    assert body["error_code"] == code


def test_start_poll_done(client, monkeypatch):
    calls = []

    def fake_transform_photo(**kw):
        calls.append(kw)
        return {"subjectType": "풍경"}

    monkeypatch.setattr(server, "transform_photo", fake_transform_photo)
    body = {"image_base64": _photo_b64(), "style_profile": {}, "user_id": "111"}

    r = client.post("/api/jobs/analyze-and-transform", json=body, headers=_bearer("111"))
    assert r.status_code == 202, r.text
    start = r.json()
    assert start["success"] is True
    assert start["status"] in ("queued", "running", "done")
    assert len(start["job_id"]) >= 32

    done = _poll(client, start["job_id"], headers=_bearer("111"))
    assert done["success"] is True
    assert done["status"] == "done" and done["stage"] == "done"
    assert done["error"] is None
    assert isinstance(done["elapsed_sec"], float)
    result = done["result"]
    # 동기 응답과 같은 필드
    assert set(result) == set(server.AnalyzeAndTransformResponse.model_fields)
    assert result["success"] is True
    assert result["image_base64"]
    assert isinstance(result["params"], dict)
    assert len(calls) == 1


def test_stage_callback_order(monkeypatch):
    monkeypatch.setattr(server, "transform_photo", lambda **kw: {"subjectType": "풍경"})
    stages = []
    req = server.AnalyzeAndTransformRequest(image_base64=_photo_b64())
    resp = server._run_analyze_and_transform(req, on_stage=stages.append)
    assert resp.success is True
    assert stages == ["analyzing", "rendering"]


def test_dedupe_returns_same_job(client, monkeypatch):
    gate = threading.Event()
    calls = []

    def slow_transform_photo(**kw):
        calls.append(1)
        gate.wait(10)
        return {"subjectType": "풍경"}

    monkeypatch.setattr(server, "transform_photo", slow_transform_photo)
    body = {"image_base64": _photo_b64(), "style_profile": {"a": 1}, "user_id": "111"}
    h = _bearer("111")

    r1 = client.post("/api/jobs/analyze-and-transform", json=body, headers=h)
    r2 = client.post("/api/jobs/analyze-and-transform", json=body, headers=h)
    assert r1.json()["job_id"] == r2.json()["job_id"]

    # 다른 요청(reshape_enabled 다름)은 별개 작업
    r3 = client.post(
        "/api/jobs/analyze-and-transform", json={**body, "reshape_enabled": True}, headers=h
    )
    assert r3.json()["job_id"] != r1.json()["job_id"]

    gate.set()
    _poll(client, r1.json()["job_id"], headers=h)
    _poll(client, r3.json()["job_id"], headers=h)

    # 완료 후 보관 중에도 합친다
    r4 = client.post("/api/jobs/analyze-and-transform", json=body, headers=h)
    assert r4.json()["job_id"] == r1.json()["job_id"]
    assert r4.json()["status"] == "done"
    assert len(calls) == 2  # body 1회 + reshape 요청 1회


def test_error_status_with_safe_message(client, monkeypatch):
    def boom(**kw):
        raise RuntimeError("claude CLI failed: access_token=SECRET123 /Users/x/.claude")

    monkeypatch.setattr(server, "transform_photo", boom)
    r = client.post(
        "/api/jobs/analyze-and-transform", json={"image_base64": _photo_b64()}
    )
    assert r.status_code == 202
    body = _poll(client, r.json()["job_id"])
    assert body["status"] == "error" and body["stage"] == "error"
    assert body["result"] is None
    assert body["error"] == jobs.DEFAULT_ERROR_MESSAGE
    assert "SECRET123" not in r.text + str(body)


def test_error_job_is_not_deduped(client, monkeypatch):
    state = {"fail": True}

    def flaky(**kw):
        if state["fail"]:
            raise RuntimeError("x")
        return {"subjectType": "풍경"}

    monkeypatch.setattr(server, "transform_photo", flaky)
    body = {"image_base64": _photo_b64()}
    j1 = client.post("/api/jobs/analyze-and-transform", json=body).json()["job_id"]
    assert _poll(client, j1)["status"] == "error"
    state["fail"] = False
    j2 = client.post("/api/jobs/analyze-and-transform", json=body).json()["job_id"]
    assert j2 != j1
    assert _poll(client, j2)["status"] == "done"


def test_unknown_job_404(client):
    _assert_error(client.get("/api/jobs/" + "0" * 32), 404, "job_not_found")


def test_owner_mismatch_404(client, monkeypatch):
    monkeypatch.setattr(server, "transform_photo", lambda **kw: {"subjectType": "풍경"})
    r = client.post(
        "/api/jobs/analyze-and-transform",
        json={"image_base64": _photo_b64(), "user_id": "111"},
        headers=_bearer("111"),
    )
    job_id = r.json()["job_id"]
    _poll(client, job_id, headers=_bearer("111"))

    _assert_error(client.get(f"/api/jobs/{job_id}", headers=_bearer("222")), 404, "job_not_found")
    # 세션 없는 조회(과도기)도 남의 작업은 못 본다
    _assert_error(client.get(f"/api/jobs/{job_id}"), 404, "job_not_found")


def test_start_forbidden_user_403(client):
    r = client.post(
        "/api/jobs/analyze-and-transform",
        json={"image_base64": _photo_b64(), "user_id": "999"},
        headers=_bearer("111"),
    )
    _assert_error(r, 403, "forbidden_user")


def test_start_requires_session_when_auth_required(client, monkeypatch):
    monkeypatch.setenv("GAMDO_AUTH_REQUIRED", "1")
    r = client.post("/api/jobs/analyze-and-transform", json={"image_base64": _photo_b64()})
    _assert_error(r, 401, "session_invalid")


def test_busy_503_when_queue_full(client, monkeypatch):
    store = jobs.JobStore(max_workers=1, max_queued=1)
    monkeypatch.setattr(server, "_job_store", store)
    gate = threading.Event()
    started = threading.Event()

    def blocking(**kw):
        started.set()
        gate.wait(10)
        return {"subjectType": "풍경"}

    monkeypatch.setattr(server, "transform_photo", blocking)
    try:
        # 1번: 실행 중 (워커 1개 점유)
        r1 = client.post("/api/jobs/analyze-and-transform", json={"image_base64": _photo_b64(seed=1)})
        assert r1.status_code == 202
        assert started.wait(5)
        # 2번: 대기열 1칸 차지
        r2 = client.post("/api/jobs/analyze-and-transform", json={"image_base64": _photo_b64(seed=2)})
        assert r2.status_code == 202
        assert r2.json()["status"] == "queued"
        # 3번: 가득 → 503
        r3 = client.post("/api/jobs/analyze-and-transform", json={"image_base64": _photo_b64(seed=3)})
        _assert_error(r3, 503, "busy")
        # 이미 있는 요청은 가득 차도 합쳐서 돌려준다
        r2b = client.post("/api/jobs/analyze-and-transform", json={"image_base64": _photo_b64(seed=2)})
        assert r2b.status_code == 202 and r2b.json()["job_id"] == r2.json()["job_id"]
    finally:
        gate.set()
        store.shutdown(wait=True)


def test_ttl_expiry(client, monkeypatch):
    monkeypatch.setattr(server, "transform_photo", lambda **kw: {"subjectType": "풍경"})
    clock = {"t": 1000.0}
    monkeypatch.setattr(jobs, "_now", lambda: clock["t"])

    job_id = client.post(
        "/api/jobs/analyze-and-transform", json={"image_base64": _photo_b64()}
    ).json()["job_id"]
    body = _poll(client, job_id)
    assert body["status"] == "done"

    clock["t"] += jobs.JOB_TTL_SECONDS - 1
    assert client.get(f"/api/jobs/{job_id}").status_code == 200
    clock["t"] += 2
    _assert_error(client.get(f"/api/jobs/{job_id}"), 404, "job_not_found")


def test_total_cap_evicts_oldest_finished(monkeypatch):
    clock = {"t": 0.0}
    monkeypatch.setattr(jobs, "_now", lambda: clock["t"])
    store = jobs.JobStore(max_workers=1, max_queued=10, max_total=3)
    try:
        ids = []
        for i in range(5):
            clock["t"] += 1
            snap = store.submit(f"k{i}", None, lambda on_stage, i=i: {"i": i})
            ids.append(snap.job_id)
            deadline = time.monotonic() + 5
            while store.get(snap.job_id).status != jobs.DONE:
                assert time.monotonic() < deadline
                time.sleep(0.01)
        # 다음 등록/조회 때 정리된다
        assert store.get(ids[0]) is None and store.get(ids[1]) is None
        assert all(store.get(j) is not None for j in ids[2:])
    finally:
        store.shutdown(wait=True)


def test_worker_exception_does_not_kill_pool():
    store = jobs.JobStore(max_workers=1, max_queued=10)
    try:
        def bad(on_stage):
            raise ValueError("boom")

        s1 = store.submit("a", None, bad)
        s2 = store.submit("b", None, lambda on_stage: {"ok": True})
        deadline = time.monotonic() + 5
        while store.get(s2.job_id).status != jobs.DONE:
            assert time.monotonic() < deadline
            time.sleep(0.01)
        e = store.get(s1.job_id)
        assert e.status == jobs.ERROR and e.error == jobs.DEFAULT_ERROR_MESSAGE
    finally:
        store.shutdown(wait=True)


def test_sync_endpoint_unchanged(client, monkeypatch):
    monkeypatch.setattr(server, "transform_photo", lambda **kw: {"subjectType": "풍경"})
    r = client.post("/api/analyze-and-transform", json={"image_base64": _photo_b64()})
    assert r.status_code == 200
    body = r.json()
    assert body["success"] is True and body["image_base64"]

    def boom(**kw):
        raise RuntimeError("raw reason")

    monkeypatch.setattr(server, "transform_photo", boom)
    r = client.post("/api/analyze-and-transform", json={"image_base64": _photo_b64()})
    # 동기 엔드포인트는 예전처럼 str(e)를 그대로 싣는다
    assert r.json() == {
        "success": False, "analysis": None, "image_base64": None,
        "params": None, "params_comment": None, "error": "raw reason",
    }
