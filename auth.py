"""세션 토큰 발급·검증 + Firebase 커스텀 토큰.

토큰 형식 (앱과의 계약 — 바꾸면 앱도 같이 바꿔야 한다):
    v1.<base64url(json payload)>.<base64url(HMAC-SHA256(secret, "v1." + payload_b64))>
    payload = {"uid": "<instagram user_id>", "iat": <unix초>, "exp": <unix초>}

서명 키는 env GAMDO_SESSION_SECRET. 테스트·운영 중 env를 바꿔도 바로 반영되도록
모듈 로드 시점이 아니라 호출 시점에 읽는다.
"""

import base64
import binascii
import hashlib
import hmac
import json
import logging
import os
import threading
import time

log = logging.getLogger("gamdo-agent.auth")

SESSION_VERSION = "v1"
SESSION_TTL_SECONDS = 60 * 60 * 24 * 30  # 30일
# 짧은 비밀키는 무차별 대입으로 풀릴 수 있다. 계약상 32바이트 이상.
_MIN_SECRET_LEN = 32
# 헤더로 들어오는 값이라 길이 상한을 둔다. 정상 토큰은 200자 안팎이다.
_MAX_TOKEN_LEN = 2048

_TRUTHY = {"1", "true", "yes", "on"}


class SessionNotConfigured(Exception):
    """GAMDO_SESSION_SECRET이 없거나 너무 짧아 세션을 발급할 수 없다."""


def _b64e(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).rstrip(b"=").decode("ascii")


def _b64d(text: str) -> bytes:
    # 패딩을 떼고 보내는 쪽·붙이고 보내는 쪽 모두 받아 준다
    padded = text + "=" * (-len(text) % 4)
    return base64.b64decode(padded.encode("ascii"), altchars=b"-_", validate=True)


def session_secret() -> bytes | None:
    """서명 키. 없거나 짧으면 None (세션 기능 꺼짐)."""
    secret = os.getenv("GAMDO_SESSION_SECRET", "").strip()
    if not secret:
        return None
    if len(secret) < _MIN_SECRET_LEN:
        log.warning("GAMDO_SESSION_SECRET이 %d자 미만이라 세션을 끈다", _MIN_SECRET_LEN)
        return None
    return secret.encode("utf-8")


def session_configured() -> bool:
    return session_secret() is not None


def auth_required() -> bool:
    """GAMDO_AUTH_REQUIRED가 켜져 있으면 세션 없는 요청을 401로 막는다."""
    return os.getenv("GAMDO_AUTH_REQUIRED", "").strip().lower() in _TRUTHY


def _sign(secret: bytes, payload_b64: str) -> str:
    msg = f"{SESSION_VERSION}.{payload_b64}".encode("ascii")
    return _b64e(hmac.new(secret, msg, hashlib.sha256).digest())


def issue_session(uid: str, now: int | None = None) -> tuple[str, int]:
    """uid로 세션 토큰을 만든다. (token, exp) 반환."""
    secret = session_secret()
    if secret is None:
        raise SessionNotConfigured("GAMDO_SESSION_SECRET not configured")
    uid = str(uid or "").strip()
    if not uid:
        raise ValueError("uid is empty")
    iat = int(time.time() if now is None else now)
    exp = iat + SESSION_TTL_SECONDS
    payload = json.dumps(
        {"uid": uid, "iat": iat, "exp": exp}, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    payload_b64 = _b64e(payload)
    return f"{SESSION_VERSION}.{payload_b64}.{_sign(secret, payload_b64)}", exp


def verify_session(token: str | None, now: int | None = None) -> str | None:
    """유효한 세션이면 uid, 아니면 None. 어떤 입력에도 예외를 올리지 않는다."""
    if not token or len(token) > _MAX_TOKEN_LEN:
        return None
    secret = session_secret()
    if secret is None:
        return None
    parts = token.split(".")
    if len(parts) != 3 or parts[0] != SESSION_VERSION:
        return None
    _, payload_b64, sig_b64 = parts
    try:
        expected = _sign(secret, payload_b64)
    except UnicodeEncodeError:
        return None
    # 서명 비교는 상수 시간으로 — 응답 시간으로 서명을 한 글자씩 맞춰 가는 것을 막는다
    if not hmac.compare_digest(expected.encode("ascii"), sig_b64.encode("utf-8")):
        return None
    try:
        payload = json.loads(_b64d(payload_b64))
    except (binascii.Error, ValueError, UnicodeError):
        return None
    if not isinstance(payload, dict):
        return None
    uid = payload.get("uid")
    exp = payload.get("exp")
    if not isinstance(uid, str) or not uid:
        return None
    # bool은 int의 하위 클래스라 따로 거른다
    if not isinstance(exp, int) or isinstance(exp, bool):
        return None
    current = int(time.time() if now is None else now)
    if exp <= current:
        return None
    return uid


def parse_bearer(authorization: str | None) -> str:
    """'Bearer <token>'에서 토큰만. 스킴이 없으면 값 전체를 토큰으로 본다(기존 동작)."""
    if not authorization:
        return ""
    value = authorization.strip()
    if value[:7].lower() == "bearer ":
        value = value[7:]
    return value.strip()


def tokens_equal(a: str, b: str) -> bool:
    """상수 시간 비교. str끼리 compare_digest는 비ASCII에서 TypeError가 나므로 bytes로."""
    return hmac.compare_digest(a.encode("utf-8"), b.encode("utf-8"))


# ── Firebase 커스텀 토큰 ──
#
# firebase-admin은 선택 의존성이다 (pip install -e '.[firebase]').
# 설치돼 있지 않거나 서비스 계정이 없으면 None — 서버는 503으로 답한다.

_firebase_lock = threading.Lock()
_firebase_app = None
_firebase_app_key: str | None = None


def _firebase_app_or_none():
    global _firebase_app, _firebase_app_key
    path = os.getenv("FIREBASE_SERVICE_ACCOUNT", "").strip()
    if not path or not os.path.isfile(path):
        return None
    with _firebase_lock:
        if _firebase_app is not None and _firebase_app_key == path:
            return _firebase_app
        try:
            import firebase_admin
            from firebase_admin import credentials
        except ImportError:
            log.warning("FIREBASE_SERVICE_ACCOUNT가 있지만 firebase-admin이 설치돼 있지 않다")
            return None
        try:
            cred = credentials.Certificate(path)
            name = f"gamdo-{hashlib.sha256(path.encode()).hexdigest()[:8]}"
            try:
                app = firebase_admin.get_app(name)
            except ValueError:
                app = firebase_admin.initialize_app(cred, name=name)
        except Exception as exc:
            # 예외 메시지에 키 내용이 섞일 수 있어 종류만 남긴다
            log.error("Firebase 초기화 실패: %s", type(exc).__name__)
            return None
        _firebase_app, _firebase_app_key = app, path
        return app


def firebase_configured() -> bool:
    return _firebase_app_or_none() is not None


def create_firebase_token(uid: str) -> str | None:
    """Firebase 커스텀 토큰. 설정이 안 돼 있으면 None.

    uid는 Instagram user_id 그대로 — RTDB 규칙 users/{userId}의 auth.uid와 맞춘다.
    """
    app = _firebase_app_or_none()
    if app is None:
        return None
    from firebase_admin import auth as fb_auth

    token = fb_auth.create_custom_token(uid, app=app)
    return token.decode("utf-8") if isinstance(token, bytes) else str(token)
