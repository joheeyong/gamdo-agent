"""세션 인증 계약 (scratchpad/auth_contract.md).

- 토큰 서명·검증 (위조·만료·형식 오류)
- 인가 규칙: 면제 경로, 과도기, GAMDO_AUTH_REQUIRED, 서비스 토큰, uid 불일치 403
- /api/session (Instagram /me 검증 — httpx는 가짜로 바꾼다)
- /api/firebase-token (미설정 503, 설정 시 커스텀 토큰)
- exchange-token 응답의 세션 필드
- 에러 본문 형식 {"success": false, "error", "error_code"}
"""

import base64
import json
import logging
import sys
import types

import httpx
import pytest
from fastapi.testclient import TestClient

import auth
import server

SECRET = "s" * 48
OTHER_SECRET = "o" * 48


@pytest.fixture(autouse=True)
def _env(monkeypatch):
    monkeypatch.setenv("GAMDO_SESSION_SECRET", SECRET)
    monkeypatch.delenv("GAMDO_AUTH_REQUIRED", raising=False)
    monkeypatch.delenv("FIREBASE_SERVICE_ACCOUNT", raising=False)
    monkeypatch.setattr(server, "APP_TOKEN", "")
    # 무거운 처리 대신 가짜로
    monkeypatch.setattr(server, "analyze_user", lambda **kw: {"ok": True, "user_id": kw["user_id"]})
    monkeypatch.setattr(server, "transform_photo", lambda **kw: {"ok": True})
    monkeypatch.setattr(server, "get_reference_image_paths", lambda _uid: [])
    monkeypatch.setattr(auth, "_firebase_app", None)
    monkeypatch.setattr(auth, "_firebase_app_key", None)


@pytest.fixture
def client():
    return TestClient(server.app)


def _bearer(token: str) -> dict:
    return {"Authorization": f"Bearer {token}"}


def _session(uid: str = "111") -> str:
    return auth.issue_session(uid)[0]


def _assert_error(resp, status: int, code: str):
    assert resp.status_code == status, resp.text
    body = resp.json()
    assert body["success"] is False
    assert body["error_code"] == code
    assert isinstance(body["error"], str) and body["error"]
    assert "detail" not in body


# ── 토큰 서명·검증 ──


def test_token_format_matches_contract():
    token, exp = auth.issue_session("12345", now=1_000_000)
    ver, payload_b64, sig_b64 = token.split(".")
    assert ver == "v1"
    payload = json.loads(base64.urlsafe_b64decode(payload_b64 + "=" * (-len(payload_b64) % 4)))
    assert payload == {"uid": "12345", "iat": 1_000_000, "exp": 1_000_000 + 30 * 86400}
    assert exp == payload["exp"]
    import hashlib
    import hmac

    expected = hmac.new(SECRET.encode(), f"v1.{payload_b64}".encode(), hashlib.sha256).digest()
    assert base64.urlsafe_b64decode(sig_b64 + "=" * (-len(sig_b64) % 4)) == expected


def test_roundtrip_and_expiry():
    token, exp = auth.issue_session("42", now=1000)
    assert auth.verify_session(token, now=1001) == "42"
    assert auth.verify_session(token, now=exp - 1) == "42"
    assert auth.verify_session(token, now=exp) is None
    assert auth.verify_session(token, now=exp + 10) is None


def test_padded_signature_is_accepted():
    token = _session("7")
    ver, p, s = token.split(".")
    padded = f"{ver}.{p}.{s}{'=' * (-len(s) % 4)}"
    # 패딩이 붙은 서명 문자열은 바이트가 달라 거부되지만 예외는 없어야 한다
    assert auth.verify_session(padded) in (None, "7")


def test_wrong_secret_rejected(monkeypatch):
    token = _session("1")
    monkeypatch.setenv("GAMDO_SESSION_SECRET", OTHER_SECRET)
    assert auth.verify_session(token) is None


def test_tampered_payload_rejected():
    token = _session("1")
    ver, _, sig = token.split(".")
    forged = base64.urlsafe_b64encode(
        json.dumps({"uid": "2", "iat": 0, "exp": 9_999_999_999}).encode()
    ).rstrip(b"=").decode()
    assert auth.verify_session(f"{ver}.{forged}.{sig}") is None


@pytest.mark.parametrize("bad", [
    "", "garbage", "v1.a", "v1.a.b.c", "v2.abc.def", "v1..", "v1.%%%.***",
    "v1.한글.서명", "x" * 5000,
])
def test_malformed_tokens_never_raise(bad):
    assert auth.verify_session(bad) is None


def _signed(payload: dict | list | str) -> str:
    p = base64.urlsafe_b64encode(json.dumps(payload).encode()).rstrip(b"=").decode()
    return f"v1.{p}.{auth._sign(SECRET.encode(), p)}"


@pytest.mark.parametrize("payload", [
    {"uid": "", "iat": 0, "exp": 9_999_999_999},
    {"uid": 123, "iat": 0, "exp": 9_999_999_999},
    {"uid": "1", "iat": 0},
    {"uid": "1", "iat": 0, "exp": "9999999999"},
    {"uid": "1", "iat": 0, "exp": True},
    {"uid": "1", "iat": 0, "exp": 1.0e12},
    ["uid", "1"],
    "just a string",
])
def test_signed_but_invalid_payload_rejected(payload):
    assert auth.verify_session(_signed(payload)) is None


def test_signed_valid_payload_accepted():
    assert auth.verify_session(_signed({"uid": "9", "iat": 0, "exp": 9_999_999_999})) == "9"


def test_no_secret_disables_sessions(monkeypatch):
    token = _session("1")
    monkeypatch.delenv("GAMDO_SESSION_SECRET")
    assert not auth.session_configured()
    assert auth.verify_session(token) is None
    with pytest.raises(auth.SessionNotConfigured):
        auth.issue_session("1")


def test_short_secret_is_treated_as_unconfigured(monkeypatch):
    monkeypatch.setenv("GAMDO_SESSION_SECRET", "short")
    assert not auth.session_configured()


@pytest.mark.parametrize("value,expected", [
    ("1", True), ("true", True), ("YES", True), ("on", True),
    ("0", False), ("false", False), ("", False), ("no", False),
])
def test_auth_required_flag(monkeypatch, value, expected):
    monkeypatch.setenv("GAMDO_AUTH_REQUIRED", value)
    assert auth.auth_required() is expected


def test_parse_bearer():
    assert auth.parse_bearer("Bearer abc") == "abc"
    assert auth.parse_bearer("bearer  abc ") == "abc"
    assert auth.parse_bearer("abc") == "abc"
    assert auth.parse_bearer(None) == ""


# ── 인가 규칙: 과도기 (GAMDO_AUTH_REQUIRED 없음) ──


def test_transition_no_header_passes_without_app_token(client, caplog):
    # 헤더를 보내지 않는 기존 앱 빌드가 계속 동작해야 한다
    with caplog.at_level(logging.WARNING, logger="gamdo-agent"):
        r = client.post("/api/analyze-user", json={"user_id": "123"})
    assert r.status_code == 200 and r.json()["success"] is True
    assert any("과도기" in rec.getMessage() for rec in caplog.records)


def test_transition_app_token_required_when_set(client, monkeypatch):
    monkeypatch.setattr(server, "APP_TOKEN", "apptok")
    _assert_error(client.post("/api/analyze-user", json={}), 401, "session_invalid")
    _assert_error(
        client.post("/api/analyze-user", json={}, headers=_bearer("wrong")), 401, "session_invalid"
    )
    r = client.post("/api/analyze-user", json={}, headers=_bearer("apptok"))
    assert r.status_code == 200


def test_transition_invalid_session_with_no_app_token_still_passes(client):
    # 과도기에는 헤더가 이상해도 예전처럼 통과 (APP_TOKEN 미설정 서버)
    r = client.post("/api/analyze-user", json={}, headers=_bearer("v1.bad.token"))
    assert r.status_code == 200


def test_non_ascii_bearer_does_not_crash(client, monkeypatch):
    monkeypatch.setattr(server, "APP_TOKEN", "apptok")
    r = client.post(
        "/api/analyze-user", json={}, headers={"Authorization": "Bearer caf\xe9".encode("latin-1")}
    )
    _assert_error(r, 401, "session_invalid")


# ── 인가 규칙: GAMDO_AUTH_REQUIRED=1 ──


PROTECTED = [
    ("post", "/api/analyze-user", {}),
    ("post", "/api/transform-photo", {"style_profile": {}, "image_base64": "x"}),
    ("post", "/api/analyze-and-transform", {"image_base64": "x"}),
    ("post", "/api/auto-transform", {"image_base64": "x", "analysis": {}}),
    ("post", "/api/apply-transform", {"image_base64": "x"}),
    ("get", "/api/reference-images/123", None),
]


@pytest.mark.parametrize("method,path,body", PROTECTED)
def test_required_mode_rejects_missing_session(client, monkeypatch, method, path, body):
    monkeypatch.setenv("GAMDO_AUTH_REQUIRED", "1")
    kwargs = {"json": body} if body is not None else {}
    _assert_error(getattr(client, method)(path, **kwargs), 401, "session_invalid")


def test_required_mode_rejects_expired_session(client, monkeypatch):
    monkeypatch.setenv("GAMDO_AUTH_REQUIRED", "1")
    expired, _ = auth.issue_session("123", now=0)
    _assert_error(
        client.post("/api/analyze-user", json={}, headers=_bearer(expired)), 401, "session_invalid"
    )


def test_required_mode_accepts_valid_session(client, monkeypatch):
    monkeypatch.setenv("GAMDO_AUTH_REQUIRED", "1")
    r = client.post("/api/analyze-user", json={"user_id": "123"}, headers=_bearer(_session("123")))
    assert r.status_code == 200 and r.json()["data"]["user_id"] == "123"


def test_required_mode_service_token_always_passes(client, monkeypatch):
    monkeypatch.setenv("GAMDO_AUTH_REQUIRED", "1")
    monkeypatch.setattr(server, "APP_TOKEN", "apptok")
    # 서비스 토큰은 uid 제한도 없다
    r = client.post("/api/analyze-user", json={"user_id": "anyone"}, headers=_bearer("apptok"))
    assert r.status_code == 200
    r = client.get("/api/reference-images/anyone", headers=_bearer("apptok"))
    assert r.status_code == 200


def test_required_mode_session_passes_even_with_app_token_set(client, monkeypatch):
    monkeypatch.setenv("GAMDO_AUTH_REQUIRED", "1")
    monkeypatch.setattr(server, "APP_TOKEN", "apptok")
    r = client.post("/api/analyze-user", json={}, headers=_bearer(_session("5")))
    assert r.status_code == 200


# ── uid 불일치 403 ──


@pytest.mark.parametrize("required", [False, True])
def test_forbidden_user_on_body_user_id(client, monkeypatch, required):
    if required:
        monkeypatch.setenv("GAMDO_AUTH_REQUIRED", "1")
    h = _bearer(_session("111"))
    _assert_error(
        client.post("/api/analyze-user", json={"user_id": "222"}, headers=h), 403, "forbidden_user"
    )
    _assert_error(
        client.post("/api/analyze-and-transform", json={"image_base64": "x", "user_id": "222"},
                    headers=h),
        403, "forbidden_user",
    )


@pytest.mark.parametrize("required", [False, True])
def test_forbidden_user_on_reference_images_path(client, monkeypatch, required):
    if required:
        monkeypatch.setenv("GAMDO_AUTH_REQUIRED", "1")
    h = _bearer(_session("111"))
    _assert_error(client.get("/api/reference-images/222", headers=h), 403, "forbidden_user")
    r = client.get("/api/reference-images/111", headers=h)
    assert r.status_code == 200 and r.json() == {"success": True, "images": [], "error": None}


def test_empty_user_id_with_session_is_allowed(client):
    r = client.post("/api/analyze-user", json={"user_id": ""}, headers=_bearer(_session("111")))
    assert r.status_code == 200


def test_forbidden_user_does_not_call_model(client, monkeypatch):
    calls = []
    monkeypatch.setattr(server, "analyze_user", lambda **kw: calls.append(kw) or {})
    client.post("/api/analyze-user", json={"user_id": "222"}, headers=_bearer(_session("111")))
    assert calls == []


def test_reference_images_without_session_in_transition(client):
    # 과도기에는 세션 없이도 조회 가능 (기존 앱 빌드)
    assert client.get("/api/reference-images/222").status_code == 200


# ── 면제 경로 ──


@pytest.mark.parametrize("required", [False, True])
def test_exempt_paths(client, monkeypatch, required):
    if required:
        monkeypatch.setenv("GAMDO_AUTH_REQUIRED", "1")
    monkeypatch.setattr(server, "APP_TOKEN", "apptok")
    assert client.get("/health").status_code == 200
    r = client.get("/api/instagram/callback", params={"code": "c"}, follow_redirects=False)
    assert r.status_code in (302, 307)
    # 자격증명 미설정 서버 → success:false 이지만 401은 아니다
    monkeypatch.setattr(server, "INSTAGRAM_CLIENT_ID", "")
    r = client.post("/api/instagram/exchange-token", json={"code": "c", "redirect_uri": "u"})
    assert r.status_code == 200 and r.json()["success"] is False


class _FakeHttpClient:
    """httpx.Client 대역. handler(method, url, kwargs) -> httpx.Response."""

    calls: list = []
    handler = None

    def __init__(self, *a, **kw):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def _do(self, method, url, **kw):
        _FakeHttpClient.calls.append((method, url, kw))
        resp = _FakeHttpClient.handler(method, url, kw)
        resp.request = httpx.Request(method, url)
        return resp

    def get(self, url, **kw):
        return self._do("GET", url, **kw)

    def post(self, url, **kw):
        return self._do("POST", url, **kw)


@pytest.fixture
def fake_http(monkeypatch):
    _FakeHttpClient.calls = []
    _FakeHttpClient.handler = None
    monkeypatch.setattr(server.httpx, "Client", _FakeHttpClient)
    return _FakeHttpClient


@pytest.mark.parametrize("path", ["/api/instagram/media", "/api/instagram/stories"])
def test_instagram_proxy_exempt_in_required_mode(client, monkeypatch, fake_http, path):
    monkeypatch.setenv("GAMDO_AUTH_REQUIRED", "1")
    fake_http.handler = lambda m, u, kw: httpx.Response(200, json={"data": [{"id": "1"}]})
    r = client.post(path, json={"access_token": "IGQV"})
    assert r.status_code == 200 and r.json()["success"] is True


# ── exchange-token 세션 필드 ──


def _exchange_handler(method, url, kw):
    if url == server.INSTAGRAM_TOKEN_URL:
        return httpx.Response(200, json={"access_token": "short", "user_id": 777})
    return httpx.Response(200, json={"access_token": "long"})


def test_exchange_token_adds_session(client, monkeypatch, fake_http):
    monkeypatch.setattr(server, "INSTAGRAM_CLIENT_ID", "cid")
    monkeypatch.setattr(server, "INSTAGRAM_CLIENT_SECRET", "csec")
    fake_http.handler = _exchange_handler
    r = client.post("/api/instagram/exchange-token", json={"code": "c", "redirect_uri": "u"})
    data = r.json()["data"]
    assert data["access_token"] == "long" and data["user_id"] == "777"
    assert auth.verify_session(data["session_token"]) == "777"
    assert isinstance(data["session_expires_at"], int)


def test_exchange_token_without_secret_has_null_session(client, monkeypatch, fake_http):
    monkeypatch.delenv("GAMDO_SESSION_SECRET")
    monkeypatch.setattr(server, "INSTAGRAM_CLIENT_ID", "cid")
    monkeypatch.setattr(server, "INSTAGRAM_CLIENT_SECRET", "csec")
    fake_http.handler = _exchange_handler
    r = client.post("/api/instagram/exchange-token", json={"code": "c", "redirect_uri": "u"})
    body = r.json()
    assert body["success"] is True
    assert body["data"]["session_token"] is None
    assert body["data"]["session_expires_at"] is None


# ── /api/session ──


def _me_handler(m, u, kw):
    return httpx.Response(200, json={"user_id": "17841400000", "id": "2600000", "username": "g"})


@pytest.mark.parametrize("claimed", ["17841400000", "2600000"])
def test_session_honors_claimed_user_id_matching_token(client, fake_http, claimed):
    # exchange-token의 user_id가 /me의 id 쪽일 수 있다 — 저장된 값으로 발급해야
    # 세션 uid와 RTDB 경로가 로그인 때와 같아진다
    fake_http.handler = _me_handler
    r = client.post("/api/session", json={"access_token": "IGQV", "user_id": claimed})
    assert r.status_code == 200
    data = r.json()["data"]
    assert data["user_id"] == claimed
    assert auth.verify_session(data["session_token"]) == claimed


def test_session_rejects_claimed_user_id_of_someone_else(client, fake_http):
    fake_http.handler = _me_handler
    r = client.post("/api/session", json={"access_token": "IGQV", "user_id": "999"})
    assert r.status_code == 403
    assert r.json()["error_code"] == "forbidden_user"


def test_session_issues_token_from_instagram_me(client, fake_http):
    fake_http.handler = lambda m, u, kw: httpx.Response(
        200, json={"user_id": "17841400000", "username": "gamdo", "id": "other"}
    )
    r = client.post("/api/session", json={"access_token": "IGQV-long"})
    assert r.status_code == 200
    body = r.json()
    assert body["success"] is True
    data = body["data"]
    assert data["user_id"] == "17841400000"
    assert auth.verify_session(data["session_token"]) == "17841400000"
    assert isinstance(data["session_expires_at"], int)

    method, url, kw = fake_http.calls[0]
    assert method == "GET" and url == "https://graph.instagram.com/me"
    assert kw["params"] == {"fields": "id,user_id,username"}
    # 토큰은 쿼리가 아니라 헤더로
    assert kw["headers"]["Authorization"] == "Bearer IGQV-long"
    assert "access_token" not in kw["params"]


def test_session_falls_back_to_id(client, fake_http):
    fake_http.handler = lambda m, u, kw: httpx.Response(200, json={"id": "999"})
    r = client.post("/api/session", json={"access_token": "t"})
    assert r.json()["data"]["user_id"] == "999"


@pytest.mark.parametrize("status", [400, 401, 403])
def test_session_invalid_instagram_token(client, fake_http, status):
    fake_http.handler = lambda m, u, kw: httpx.Response(
        status, json={"error": {"type": "OAuthException", "code": 190}}
    )
    _assert_error(client.post("/api/session", json={"access_token": "bad"}), 401,
                  "instagram_token_invalid")


def test_session_empty_token_skips_instagram(client, fake_http):
    _assert_error(client.post("/api/session", json={"access_token": "  "}), 401,
                  "instagram_token_invalid")
    assert fake_http.calls == []


def test_session_instagram_down_is_not_token_invalid(client, fake_http):
    # 앱은 instagram_token_invalid에서만 로그아웃한다 — 장애로 로그아웃시키면 안 된다
    fake_http.handler = lambda m, u, kw: httpx.Response(500, text="oops")
    _assert_error(client.post("/api/session", json={"access_token": "t"}), 502,
                  "instagram_unavailable")

    def boom(m, u, kw):
        raise httpx.ConnectError("down access_token=SECRET")

    fake_http.handler = boom
    _assert_error(client.post("/api/session", json={"access_token": "t"}), 502,
                  "instagram_unavailable")


def test_session_missing_user_id(client, fake_http):
    fake_http.handler = lambda m, u, kw: httpx.Response(200, json={"username": "x"})
    _assert_error(client.post("/api/session", json={"access_token": "t"}), 502,
                  "instagram_unavailable")


def test_session_not_configured(client, monkeypatch, fake_http):
    monkeypatch.delenv("GAMDO_SESSION_SECRET")
    _assert_error(client.post("/api/session", json={"access_token": "t"}), 503,
                  "session_not_configured")
    assert fake_http.calls == []


def test_session_exempt_in_required_mode(client, monkeypatch, fake_http):
    monkeypatch.setenv("GAMDO_AUTH_REQUIRED", "1")
    monkeypatch.setattr(server, "APP_TOKEN", "apptok")
    fake_http.handler = lambda m, u, kw: httpx.Response(200, json={"user_id": "1"})
    assert client.post("/api/session", json={"access_token": "t"}).status_code == 200


def test_session_does_not_log_tokens(client, fake_http, caplog):
    fake_http.handler = lambda m, u, kw: httpx.Response(200, json={"user_id": "1"})
    with caplog.at_level(logging.DEBUG):
        r = client.post("/api/session", json={"access_token": "IGQV-VERY-SECRET"})
    token = r.json()["data"]["session_token"]
    text = "\n".join(rec.getMessage() for rec in caplog.records)
    assert "IGQV-VERY-SECRET" not in text
    assert token not in text


def test_session_token_then_protected_call(client, monkeypatch, fake_http):
    """발급받은 세션으로 보호 엔드포인트를 부르는 전체 흐름."""
    monkeypatch.setenv("GAMDO_AUTH_REQUIRED", "1")
    fake_http.handler = lambda m, u, kw: httpx.Response(200, json={"user_id": "55"})
    token = client.post("/api/session", json={"access_token": "t"}).json()["data"]["session_token"]
    assert client.post("/api/analyze-user", json={"user_id": "55"},
                       headers=_bearer(token)).status_code == 200
    _assert_error(client.post("/api/analyze-user", json={"user_id": "56"}, headers=_bearer(token)),
                  403, "forbidden_user")


# ── /api/firebase-token ──


def test_firebase_token_requires_session_even_in_transition(client, monkeypatch):
    _assert_error(client.post("/api/firebase-token"), 401, "session_invalid")
    monkeypatch.setattr(server, "APP_TOKEN", "apptok")
    # 서비스 토큰은 uid가 없어 커스텀 토큰을 만들 수 없다
    _assert_error(client.post("/api/firebase-token", headers=_bearer("apptok")), 401,
                  "session_invalid")


def test_firebase_token_not_configured(client):
    _assert_error(client.post("/api/firebase-token", headers=_bearer(_session("1"))), 503,
                  "firebase_not_configured")


def test_firebase_token_missing_file(client, monkeypatch, tmp_path):
    monkeypatch.setenv("FIREBASE_SERVICE_ACCOUNT", str(tmp_path / "nope.json"))
    _assert_error(client.post("/api/firebase-token", headers=_bearer(_session("1"))), 503,
                  "firebase_not_configured")


def test_firebase_token_without_package(client, monkeypatch, tmp_path):
    sa = tmp_path / "sa.json"
    sa.write_text("{}")
    monkeypatch.setenv("FIREBASE_SERVICE_ACCOUNT", str(sa))
    # firebase_admin import가 실패하는 환경
    monkeypatch.setitem(sys.modules, "firebase_admin", None)
    _assert_error(client.post("/api/firebase-token", headers=_bearer(_session("1"))), 503,
                  "firebase_not_configured")


@pytest.fixture
def fake_firebase(monkeypatch):
    """firebase_admin 대역 — 실제 패키지·서비스 계정 없이 발급 경로를 검사한다."""
    minted = []
    apps = {}

    fa = types.ModuleType("firebase_admin")
    creds = types.ModuleType("firebase_admin.credentials")
    fb_auth = types.ModuleType("firebase_admin.auth")

    creds.Certificate = lambda path: ("cert", path)

    def get_app(name):
        if name not in apps:
            raise ValueError("no app")
        return apps[name]

    def initialize_app(cred, name):
        apps[name] = ("app", name, cred)
        return apps[name]

    def create_custom_token(uid, app=None):
        minted.append((uid, app))
        return f"custom-{uid}".encode()

    fa.get_app = get_app
    fa.initialize_app = initialize_app
    fa.credentials = creds
    fa.auth = fb_auth
    fb_auth.create_custom_token = create_custom_token
    monkeypatch.setitem(sys.modules, "firebase_admin", fa)
    monkeypatch.setitem(sys.modules, "firebase_admin.credentials", creds)
    monkeypatch.setitem(sys.modules, "firebase_admin.auth", fb_auth)
    return minted


def test_firebase_token_issued_with_raw_uid(client, monkeypatch, tmp_path, fake_firebase):
    sa = tmp_path / "sa.json"
    sa.write_text("{}")
    monkeypatch.setenv("FIREBASE_SERVICE_ACCOUNT", str(sa))
    r = client.post("/api/firebase-token", headers=_bearer(_session("17841400000")))
    assert r.status_code == 200
    assert r.json() == {
        "success": True, "data": {"firebase_token": "custom-17841400000"},
        "error": None, "error_code": None,
    }
    # uid는 "ig:" 접두어 없이 user_id 그대로 (RTDB users/{userId})
    assert fake_firebase[0][0] == "17841400000"
    # 두 번째 호출은 초기화된 앱을 재사용한다
    client.post("/api/firebase-token", headers=_bearer(_session("17841400000")))
    assert fake_firebase[0][1] is fake_firebase[1][1]


def test_firebase_mint_failure_is_503(client, monkeypatch, tmp_path, fake_firebase):
    sa = tmp_path / "sa.json"
    sa.write_text("{}")
    monkeypatch.setenv("FIREBASE_SERVICE_ACCOUNT", str(sa))

    def fail(uid, app=None):
        raise RuntimeError("private_key=SECRET")

    monkeypatch.setattr(sys.modules["firebase_admin.auth"], "create_custom_token", fail)
    _assert_error(client.post("/api/firebase-token", headers=_bearer(_session("1"))), 503,
                  "firebase_not_configured")


# ── 로그 가리기 ──


def test_redact_hides_session_and_firebase_tokens():
    out = server._redact('{"session_token": "v1.abc.def", "firebase_token": "eyJ"}')
    assert "v1.abc.def" not in out and "eyJ" not in out
