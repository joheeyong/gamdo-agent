"""server.py 엔드포인트 수준의 가드들.

- 본문 크기 상한(413), OAuth 콜백 URL 인코딩
- 로그·에러 응답의 비밀값 가리기
- feedCompatibilityBefore는 손대기 전 사진으로 잰다
- subjectType 공백 정규화 (param_engine과 같은 기준)
- apply-transform의 톤 커브 제어점·HSL 정리
"""

import base64
import io
import logging
from urllib.parse import parse_qs, urlsplit

import numpy as np
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from PIL import Image

import server


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(server, "APP_TOKEN", "")
    return TestClient(server.app)


def _photo_b64(size=(200, 100)) -> str:
    rng = np.random.default_rng(0)
    arr = np.full((size[1], size[0], 3), 140, np.float32) + rng.normal(0, 8, (size[1], size[0], 3))
    buf = io.BytesIO()
    Image.fromarray(arr.clip(0, 255).astype(np.uint8)).save(buf, format="JPEG")
    return base64.b64encode(buf.getvalue()).decode()


# ── 본문 크기 상한 ──


def _limited_app(max_bytes: int) -> TestClient:
    app = FastAPI()

    @app.post("/echo")
    def echo(payload: dict):
        return {"n": len(payload.get("data", ""))}

    app.add_middleware(server._BodySizeLimitMiddleware, max_bytes=max_bytes)
    return TestClient(app)


def test_body_limit_rejects_by_content_length():
    c = _limited_app(100)
    r = c.post("/echo", json={"data": "x" * 500})
    assert r.status_code == 413


def test_body_limit_rejects_streamed_body_without_length():
    c = _limited_app(100)

    def chunks():
        yield b'{"data": "'
        for _ in range(10):
            yield b"x" * 50
        yield b'"}'

    r = c.post("/echo", content=chunks(), headers={"content-type": "application/json"})
    assert r.status_code == 413


def test_body_limit_allows_normal_request():
    c = _limited_app(10_000)
    r = c.post("/echo", json={"data": "x" * 500})
    assert r.status_code == 200
    assert r.json() == {"n": 500}


def test_server_default_body_limit_is_generous():
    # 앱의 최대 업로드(5MB JPEG → base64 약 6.7MB)보다 충분히 커야 한다
    assert server._MAX_BODY_BYTES >= 20 * 1024 * 1024


# ── OAuth 콜백 ──


def test_instagram_callback_url_encodes_params(client):
    r = client.get(
        "/api/instagram/callback",
        params={"code": "AQ&evil=1#_", "state": "a b"},
        follow_redirects=False,
    )
    assert r.status_code in (302, 307)
    loc = r.headers["location"]
    assert loc.startswith("gamdo://oauth/instagram?")
    qs = parse_qs(urlsplit(loc).query)
    assert qs == {"code": ["AQ&evil=1#_"], "state": ["a b"]}


def test_instagram_callback_without_state(client):
    r = client.get("/api/instagram/callback", params={"code": "abc"}, follow_redirects=False)
    assert r.headers["location"] == "gamdo://oauth/instagram?code=abc"


# ── 비밀값 가리기 ──


def test_httpx_info_logging_silenced():
    # 루트 레벨에 기대지 않고 로거 자체에 명시돼 있어야 한다 (pytest는 루트를 따로 잡는다)
    assert logging.getLogger("httpx").level >= logging.WARNING
    assert logging.getLogger("httpcore").level >= logging.WARNING


def test_redact_masks_tokens_but_keeps_context():
    msg = ("Client error '400 Bad Request' for url 'https://graph.instagram.com/access_token"
           "?grant_type=ig_exchange_token&client_secret=S3CRET&access_token=IGQV-tok'")
    out = server._redact(msg)
    assert "S3CRET" not in out and "IGQV-tok" not in out
    assert "400 Bad Request" in out and "grant_type=ig_exchange_token" in out

    body = str({"error_type": "OAuthException", "code": 400, "access_token": "tok"})
    out = server._redact(body)
    assert "tok'" not in out
    assert "'code': 400" in out and "OAuthException" in out


# ── analyze-and-transform ──


def _stub_analysis(monkeypatch, analysis: dict, reference=None):
    monkeypatch.setattr(server, "transform_photo", lambda **_: analysis)
    monkeypatch.setattr(server, "get_reference_image_paths", lambda _uid: [])
    monkeypatch.setattr(server, "measure_reference_target", lambda _paths: reference)
    monkeypatch.setattr(server, "detect_tilt_angle", lambda _img: None)


def test_feed_compatibility_before_uses_original_image(client, monkeypatch, tmp_path):
    ref_path = tmp_path / "ref.jpg"
    ref_path.write_bytes(base64.b64decode(_photo_b64((120, 120))))
    reference = server.measure_reference_target([str(ref_path)])
    _stub_analysis(
        monkeypatch,
        {"subjectType": "풍경", "autoEdits": {"instagram_ratio": "1:1"}},
        reference=reference,
    )
    measured_sizes = []
    real_measure = server.measure_image_stats

    def spy(img):
        measured_sizes.append(img.size)
        return real_measure(img)

    monkeypatch.setattr(server, "measure_image_stats", spy)
    monkeypatch.setattr(server, "feed_compatibility", lambda stats, ref: 50)

    r = client.post("/api/analyze-and-transform",
                    json={"image_base64": _photo_b64((200, 100)), "user_id": "u"})
    body = r.json()
    assert body["success"], body.get("error")
    # 첫 측정(before)은 원본 200x100, 나중 측정(after)은 크롭된 결과.
    # 모델의 1:1은 제안으로만 남고, 2:1은 인스타 상한을 넘어 1.91:1로 맞춰진다.
    assert measured_sizes[0] == (200, 100)
    assert measured_sizes[-1] == (191, 100)
    assert body["analysis"]["feedCompatibilityBefore"] == 50


def test_portrait_subject_is_whitespace_normalized(client, monkeypatch):
    _stub_analysis(monkeypatch, {"subjectType": " 인물 ", "autoEdits": {"instagram_ratio": "4:5"}})
    calls = []
    monkeypatch.setattr(server, "estimate_keystone", lambda img: calls.append(1) or 0.0)

    r = client.post("/api/analyze-and-transform", json={"image_base64": _photo_b64()})
    body = r.json()
    assert body["success"], body.get("error")
    assert calls == []  # 인물이면 키스톤을 재지도 않는다
    assert body["analysis"]["autoEdits"]["allow_vertical_crop"] is False


def test_non_finite_model_tilt_is_dropped(client, monkeypatch):
    # claude_client의 JSON 파서를 우회해 NaN이 들어와도 8°로 둔갑하지 않아야 한다
    _stub_analysis(monkeypatch, {"subjectType": "풍경", "autoEdits": {"straighten": float("nan")}})
    r = client.post("/api/analyze-and-transform", json={"image_base64": _photo_b64()})
    body = r.json()
    assert body["success"], body.get("error")
    assert body["analysis"]["autoEdits"]["straighten"] is None


# ── apply-transform 입력 정리 ──


def test_apply_transform_sanitizes_curve_and_hsl(client):
    r = client.post("/api/apply-transform", json={
        "image_base64": _photo_b64(),
        "preview": True,
        "tone_curve_strength": 0.8,
        "tone_curve_points": [[1, 1], [0, 0.2], [0.5, 0.4]],
        "hsl_adjust": {"red": {"hue": "abc", "saturation": 0.3}, "bogus": 1},
    })
    body = r.json()
    assert body["success"], body.get("error")
    applied = body["params_applied"]
    assert applied["tone_curve_points"] == [[0, 0.2], [0.5, 0.4], [1, 1]]
    assert applied["hsl_adjust"] == {"red": {"hue": 0.0, "saturation": 0.3, "lightness": 0.0}}


def test_apply_transform_garbage_curve_is_ignored(client):
    r = client.post("/api/apply-transform", json={
        "image_base64": _photo_b64(),
        "preview": True,
        "tone_curve_strength": 0.8,
        "tone_curve_points": [["x", "y"], [1, 1]],
        "hsl_adjust": {"red": "loud"},
    })
    body = r.json()
    assert body["success"], body.get("error")
    assert body["params_applied"]["tone_curve_points"] is None
    assert body["params_applied"]["hsl_adjust"] is None


# ── API 문서 노출 ──


def _docs_app() -> TestClient:
    return TestClient(FastAPI(**server._docs_kwargs()))


@pytest.mark.parametrize("value", ["", "0", "false", "no"])
def test_docs_disabled_by_default(monkeypatch, value):
    monkeypatch.setenv("GAMDO_ENABLE_DOCS", value)
    c = _docs_app()
    for path in ("/docs", "/redoc", "/openapi.json"):
        assert c.get(path).status_code == 404


@pytest.mark.parametrize("value", ["1", "true", "YES"])
def test_docs_enabled_by_env(monkeypatch, value):
    monkeypatch.setenv("GAMDO_ENABLE_DOCS", value)
    c = _docs_app()
    for path in ("/docs", "/redoc", "/openapi.json"):
        assert c.get(path).status_code == 200
