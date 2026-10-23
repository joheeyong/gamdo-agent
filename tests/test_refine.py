"""결과 다듬기: 사용자 요청으로 보정 값을 고친다 (POST /api/refine-transform).

- 모델이 낸 값은 범위로 자르고, 모르는 키·설정에서 끈 보정은 버린다
- hsl_adjust는 색·항목 단위로 합친다
- 엔드포인트는 값만 돌려주고, 바뀐 것이 없으면 comment를 싣지 않는다
"""

import base64
import io
import threading

import pytest
from fastapi.testclient import TestClient
from PIL import Image

import auth
import claude_client
import server
from param_engine import apply_refine_changes

SECRET = "s" * 48


@pytest.fixture(autouse=True)
def _env(monkeypatch):
    monkeypatch.setenv("GAMDO_SESSION_SECRET", SECRET)
    monkeypatch.delenv("GAMDO_AUTH_REQUIRED", raising=False)
    monkeypatch.setattr(server, "APP_TOKEN", "")
    monkeypatch.setattr(server, "_refine_semaphore", threading.BoundedSemaphore(2))


@pytest.fixture
def client():
    return TestClient(server.app)


def _photo_b64() -> str:
    buf = io.BytesIO()
    Image.new("RGB", (120, 90), (150, 120, 100)).save(buf, "JPEG")
    return base64.b64encode(buf.getvalue()).decode()


def _bearer(uid: str) -> dict:
    return {"Authorization": f"Bearer {auth.issue_session(uid)[0]}"}


def _body(**kw):
    base = {"user_id": "111", "image_base64": _photo_b64(),
            "params": {"brightness": 0.1, "temperature": 0.2, "face_slim": 0.1},
            "request": "좀 더 밝게"}
    base.update(kw)
    return base


# ── apply_refine_changes ──


def test_범위를_넘는_값은_자르고_모르는_키와_숫자가_아닌_값은_버린다():
    out, changed = apply_refine_changes(
        {"brightness": 0.1},
        {"brightness": 5, "grain": -0.3, "foo": 0.5, "contrast": "high", "saturation": True},
    )
    assert out["brightness"] == 1.0 and out["grain"] == 0.0
    assert "foo" not in out and "contrast" not in out and "saturation" not in out
    assert changed == ["brightness"]  # grain 0 → 0은 바뀐 게 아니다


def test_피부·체형은_자동_보정과_같은_상한():
    out, _ = apply_refine_changes({}, {"skin_smoothing": 0.9, "face_slim": 0.9, "eye_enlarge": 0.9})
    assert (out["skin_smoothing"], out["face_slim"], out["eye_enlarge"]) == (0.5, 0.5, 0.3)


def test_설정에서_끈_보정은_바꾸지_않는다():
    cur = {"skin_smoothing": 0.2, "face_slim": 0.1}
    out, changed = apply_refine_changes(
        cur, {"skin_smoothing": 0.4, "face_slim": 0.3},
        reshape_enabled=False, skin_retouch_enabled=False,
    )
    assert out == cur and changed == []


def test_색상각은_0에서_360으로_감는다():
    out, _ = apply_refine_changes({}, {"split_shadow_hue": -30, "split_highlight_hue": 400})
    assert (out["split_shadow_hue"], out["split_highlight_hue"]) == (330.0, 40.0)


def test_hsl은_색과_항목_단위로_합치고_hue는_035까지():
    cur = {"hsl_adjust": {"blue": {"hue": 0.0, "saturation": 0.1, "lightness": -0.1}}}
    out, changed = apply_refine_changes(cur, {"hsl_adjust": {"blue": {"saturation": 0.3}, "green": {"hue": 0.9}}})
    assert out["hsl_adjust"]["blue"] == {"hue": 0.0, "saturation": 0.3, "lightness": -0.1}
    assert out["hsl_adjust"]["green"]["hue"] == 0.35
    assert changed == ["hsl_adjust"]


def test_건드리지_않은_값은_그대로_남는다():
    cur = {"brightness": 0.1, "tone_curve_preset": "film", "tone_curve_points": [[0, 0], [1, 1]]}
    out, _ = apply_refine_changes(cur, {"contrast": 0.2})
    assert out["tone_curve_preset"] == "film" and out["tone_curve_points"] == [[0, 0], [1, 1]]
    assert out["brightness"] == 0.1


# ── 프롬프트 ──


def test_프롬프트는_고칠_대상이_아닌_값을_숨기고_끈_보정을_알린다():
    prompt = claude_client._refine_prompt(
        "더 따뜻하게", ["밝게"], {"brightness": 0.1, "auto_wb": 0.5, "tone_curve_preset": "film", "contrast": 0},
        reshape_enabled=False, skin_retouch_enabled=True,
    )
    assert '"brightness":0.1' in prompt
    assert "auto_wb" not in prompt and "tone_curve_preset" not in prompt and '"contrast"' not in prompt
    assert "skin_smoothing" in prompt and "face_slim" not in prompt
    assert "얼굴/체형 보정이 꺼져" in prompt
    assert "- 밝게" in prompt


# ── 엔드포인트 ──


def test_고친_값과_바뀐_키를_돌려준다(monkeypatch, client):
    seen = {}

    def fake(image_b64, params, request, history, **kw):
        seen.update(request=request, history=history, **kw)
        return {"changes": {"brightness": 0.2}, "comment": "밝게 했어요", "unsupported": None}
    monkeypatch.setattr(server, "refine_transform", fake)
    r = client.post("/api/refine-transform", json=_body(history=["따뜻하게"], reshape_enabled=False),
                    headers=_bearer("111"))
    body = r.json()
    assert body["success"] and body["changed"] == ["brightness"]
    assert body["params"] == {"brightness": 0.2, "temperature": 0.2, "face_slim": 0.1}
    assert body["comment"] == "밝게 했어요"
    assert seen["history"] == ["따뜻하게"] and seen["reshape_enabled"] is False


def test_바뀐_것이_없으면_comment_없이_unsupported만(monkeypatch, client):
    monkeypatch.setattr(server, "refine_transform", lambda *a, **k: {
        "changes": {}, "comment": "그대로 뒀어요", "unsupported": "자르기는 할 수 없어요"})
    body = client.post("/api/refine-transform", json=_body(request="잘라줘"), headers=_bearer("111")).json()
    assert body["success"] and body["changed"] == []
    assert body["comment"] is None and body["unsupported"] == "자르기는 할 수 없어요"


def test_빈_요청은_모델을_부르지_않는다(monkeypatch, client):
    monkeypatch.setattr(server, "refine_transform", lambda *a, **k: pytest.fail("호출되면 안 됨"))
    body = client.post("/api/refine-transform", json=_body(request="   "), headers=_bearer("111")).json()
    assert body["success"] is False


def test_깨진_이미지는_모델을_부르지_않는다(monkeypatch, client):
    monkeypatch.setattr(server, "refine_transform", lambda *a, **k: pytest.fail("호출되면 안 됨"))
    body = client.post("/api/refine-transform", json=_body(image_base64="bm90IGFuIGltYWdl"),
                       headers=_bearer("111")).json()
    assert body["success"] is False and body["error"]


def test_모델_오류는_안전한_문구로(monkeypatch, client):
    def boom(*a, **k):
        raise RuntimeError("claude CLI inline failed (code=1): secret path /Users/x")
    monkeypatch.setattr(server, "refine_transform", boom)
    body = client.post("/api/refine-transform", json=_body(), headers=_bearer("111")).json()
    assert body["success"] is False and "/Users" not in body["error"]


def test_다른_사용자의_user_id는_거절(monkeypatch, client):
    monkeypatch.setattr(server, "refine_transform", lambda *a, **k: pytest.fail("호출되면 안 됨"))
    r = client.post("/api/refine-transform", json=_body(user_id="222"), headers=_bearer("111"))
    assert r.status_code == 403


def test_자리가_없으면_503(monkeypatch, client):
    monkeypatch.setattr(server, "_refine_semaphore", threading.BoundedSemaphore(1))
    server._refine_semaphore.acquire()
    monkeypatch.setattr(server, "refine_transform", lambda *a, **k: pytest.fail("호출되면 안 됨"))
    r = client.post("/api/refine-transform", json=_body(), headers=_bearer("111"))
    assert r.status_code == 503


def test_요청이_너무_길면_422(client):
    r = client.post("/api/refine-transform", json=_body(request="가" * 201), headers=_bearer("111"))
    assert r.status_code == 422
