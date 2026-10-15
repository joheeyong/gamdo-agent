"""사용자 데이터 삭제 — 앱의 연결 해제(DELETE /api/me/data)와 Meta 콜백(signed_request)."""

import base64
import hashlib
import hmac
import json

import pytest
from fastapi.testclient import TestClient

import auth
import claude_client as cc
import server

APP_SECRET = "meta-app-secret"
SESSION_SECRET = "s" * 48


def _b64(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).rstrip(b"=").decode()


def _signed(payload: dict, secret: str = APP_SECRET) -> str:
    body = _b64(json.dumps(payload).encode())
    sig = _b64(hmac.new(secret.encode(), body.encode(), hashlib.sha256).digest())
    return f"{sig}.{body}"


@pytest.fixture
def env(monkeypatch, tmp_path):
    monkeypatch.setattr(cc, "_REF_DIR", str(tmp_path))
    monkeypatch.setattr(server, "INSTAGRAM_CLIENT_SECRET", APP_SECRET)
    monkeypatch.setattr(server, "APP_TOKEN", "")
    monkeypatch.setenv("GAMDO_SESSION_SECRET", SESSION_SECRET)
    for uid in ("1784", "9999"):
        d = tmp_path / uid
        d.mkdir()
        (d / "ref_0.jpg").write_bytes(b"x")
    return tmp_path


# ── signed_request ──


def test_올바른_서명이면_payload를_돌려준다():
    p = {"algorithm": "HMAC-SHA256", "user_id": "1784", "issued_at": 1}
    assert auth.parse_signed_request(_signed(p), APP_SECRET)["user_id"] == "1784"


@pytest.mark.parametrize("bad", [
    None, "", "garbage", "a.b", "a" * 5000,
])
def test_망가진_signed_request는_거부(bad):
    assert auth.parse_signed_request(bad, APP_SECRET) is None


def test_다른_비밀키로_서명하면_거부():
    p = {"algorithm": "HMAC-SHA256", "user_id": "1784"}
    assert auth.parse_signed_request(_signed(p, secret="attacker"), APP_SECRET) is None


def test_알고리즘이_다르면_거부():
    p = {"algorithm": "none", "user_id": "1784"}
    assert auth.parse_signed_request(_signed(p), APP_SECRET) is None


def test_비밀키가_설정되지_않으면_거부():
    p = {"algorithm": "HMAC-SHA256", "user_id": "1784"}
    assert auth.parse_signed_request(_signed(p), "") is None


# ── 삭제 ──


def test_삭제는_그_사용자_폴더만_지운다(env):
    assert cc.delete_user_data("1784") is True
    assert not (env / "1784").exists()
    assert (env / "9999").exists()
    assert cc.delete_user_data("1784") is False  # 두 번째는 지울 것이 없다


@pytest.mark.parametrize("uid", ["..", "../9999", "/etc", ""])
def test_위험한_user_id는_아무것도_지우지_않는다(env, uid):
    assert cc.delete_user_data(uid) is False
    assert (env / "9999").exists()


def test_앱_연결_해제는_세션_사용자_데이터만_지운다(env):
    token, _ = auth.issue_session("1784")
    r = TestClient(server.app).delete("/api/me/data", headers={"Authorization": f"Bearer {token}"})
    assert r.status_code == 200 and r.json()["deleted"] is True
    assert not (env / "1784").exists()
    assert (env / "9999").exists()


def test_세션이_없으면_삭제할_수_없다(env):
    r = TestClient(server.app).delete("/api/me/data")
    assert r.status_code == 401
    assert (env / "1784").exists()


def test_Meta_연결_해제_콜백(env):
    body = {"signed_request": _signed({"algorithm": "HMAC-SHA256", "user_id": "1784"})}
    r = TestClient(server.app).post("/api/instagram/deauthorize", data=body)
    assert r.status_code == 200
    assert not (env / "1784").exists()


def test_Meta_데이터_삭제_콜백은_확인코드와_상태URL을_준다(env):
    body = {"signed_request": _signed({"algorithm": "HMAC-SHA256", "user_id": "1784"})}
    r = TestClient(server.app).post("/api/instagram/data-deletion", data=body)
    data = r.json()
    assert r.status_code == 200
    assert data["confirmation_code"] and data["url"].endswith("code=" + data["confirmation_code"])
    assert not (env / "1784").exists()


def test_위조된_콜백은_아무것도_지우지_않는다(env):
    forged = {"signed_request": _signed({"algorithm": "HMAC-SHA256", "user_id": "1784"}, secret="x")}
    client = TestClient(server.app)
    assert client.post("/api/instagram/deauthorize", data=forged).status_code == 400
    assert client.post("/api/instagram/data-deletion", data=forged).status_code == 400
    assert (env / "1784").exists()
