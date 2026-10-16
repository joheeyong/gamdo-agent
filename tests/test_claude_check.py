"""Claude CLI 정기 점검 (ops/claude_check.py)과 GET /health/claude."""

import datetime as dt
import importlib.util
import json
import subprocess
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

_SPEC = importlib.util.spec_from_file_location(
    "claude_check", Path(__file__).resolve().parent.parent / "ops" / "claude_check.py"
)
cc = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(cc)


def _cli_json(**kw):
    base = {"type": "result", "is_error": False, "result": "OK",
            "usage": {"input_tokens": 3, "output_tokens": 1}}
    base.update(kw)
    return json.dumps(base)


def test_정상_응답은_성공():
    assert cc.classify(0, _cli_json(), "") == (True, "ok", "응답 정상")


def test_401은_인증_실패():
    out = _cli_json(is_error=True, api_error_status=401,
                    result="Failed to authenticate. API Error: 401 OAuth access token is invalid.")
    ok, kind, detail = cc.classify(1, out, "")
    assert (ok, kind) == (False, "auth")
    assert "401" in detail


def test_로그인_안됨_문구는_인증_실패():
    ok, kind, _ = cc.classify(1, "Not logged in · Please run /login", "")
    assert (ok, kind) == (False, "auth")


def test_JSON_키_이름의_token은_인증_실패로_보지_않는다():
    """usage.output_tokens 같은 키 때문에 일반 오류가 인증 실패로 잘못 알려지면 안 된다."""
    out = _cli_json(is_error=True, api_error_status=529, result="Overloaded")
    ok, kind, detail = cc.classify(1, out, "")
    assert (ok, kind, detail) == (False, "error", "Overloaded")


def test_응답이_OK가_아니면_실패():
    ok, kind, _ = cc.classify(0, _cli_json(result="Hello"), "")
    assert (ok, kind) == (False, "error")


@pytest.mark.parametrize("issued,today,left", [
    ("2026-10-12", dt.date(2026, 10, 12), 365),
    ("2026-10-12", dt.date(2027, 9, 12), 30),
    ("2026-10-12", dt.date(2027, 10, 20), -8),
])
def test_토큰_남은_날(monkeypatch, issued, today, left):
    monkeypatch.setenv("CLAUDE_CODE_OAUTH_TOKEN_ISSUED", issued)
    assert cc.token_days_left(today) == left


def test_발급일이_없으면_남은_날도_없다(monkeypatch):
    monkeypatch.delenv("CLAUDE_CODE_OAUTH_TOKEN_ISSUED", raising=False)
    assert cc.token_days_left() is None


def _status(ok=True, kind="ok", days=200, detail="응답 정상"):
    return {"ok": ok, "kind": kind, "detail": detail, "token_days_left": days}


def test_정상이고_만료가_멀면_알림_없음():
    assert cc.alerts_for(_status()) == []


def test_인증_실패와_만료_임박은_각각_알린다():
    titles = [t for t, _ in cc.alerts_for(_status(ok=False, kind="auth", days=10))]
    assert any("인증 실패" in t for t in titles)
    assert any("토큰 교체" in t for t in titles)


def test_일반_오류는_인증_실패와_다른_알림():
    titles = [t for t, _ in cc.alerts_for(_status(ok=False, kind="error", detail="timeout"))]
    assert titles == ["감도 서버: Claude 응답 없음"]


def test_점검은_결과를_기록하고_실패면_종료코드_1(monkeypatch, tmp_path):
    out = _cli_json(is_error=True, api_error_status=401, result="Failed to authenticate.")
    monkeypatch.setattr(cc.subprocess, "run",
                        lambda *a, **k: subprocess.CompletedProcess(a, 1, out, ""))
    path = tmp_path / "s.json"
    monkeypatch.setattr(cc, "STATUS_PATH", path)
    assert cc.main(["--no-notify"]) == 1
    saved = json.loads(path.read_text())
    assert saved["ok"] is False and saved["kind"] == "auth"


def test_CLI가_없으면_실패로_기록(monkeypatch, tmp_path):
    def missing(*a, **k):
        raise FileNotFoundError
    monkeypatch.setattr(cc.subprocess, "run", missing)
    monkeypatch.setattr(cc, "STATUS_PATH", tmp_path / "s.json")
    assert cc.main(["--no-notify"]) == 1


def test_이미지_점검은_색을_맞혀야_통과(monkeypatch):
    seen = {}

    def fake_inline(prompt, system, images, timeout):
        seen["images"] = images
        return "Blue."
    monkeypatch.setattr(cc, "_call_claude_inline", fake_inline)
    assert cc.run_vision_check("blue") == (True, "이미지 전달 정상")
    assert len(seen["images"]) == 1


def test_이미지를_못_받으면_이미지_점검_실패(monkeypatch):
    monkeypatch.setattr(cc, "_call_claude_inline", lambda *a, **k: "I don't see any image.")
    ok, detail = cc.run_vision_check("red")
    assert not ok and "red" in detail


def test_이미지_점검_중_CLI_오류도_실패(monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("claude CLI inline: result 이벤트가 없음")
    monkeypatch.setattr(cc, "_call_claude_inline", boom)
    ok, detail = cc.run_vision_check("green")
    assert not ok and "result" in detail


def test_인증이_통과해야_이미지_점검을_하고_실패는_vision으로_기록(monkeypatch, tmp_path):
    monkeypatch.setattr(cc.subprocess, "run",
                        lambda *a, **k: subprocess.CompletedProcess(a, 0, _cli_json(), ""))
    monkeypatch.setattr(cc, "run_vision_check", lambda: (False, "이미지 점검: red 이미지를 'blue'(으)로 답함"))
    path = tmp_path / "s.json"
    monkeypatch.setattr(cc, "STATUS_PATH", path)
    assert cc.main(["--no-notify"]) == 1
    saved = json.loads(path.read_text())
    assert (saved["ok"], saved["kind"]) == (False, "vision")
    titles = [t for t, _ in cc.alerts_for(saved)]
    assert titles == ["감도 서버: 사진 전달 실패"]


def test_인증_실패면_이미지_점검을_건너뛴다(monkeypatch, tmp_path):
    out = _cli_json(is_error=True, api_error_status=401, result="Failed to authenticate.")
    monkeypatch.setattr(cc.subprocess, "run",
                        lambda *a, **k: subprocess.CompletedProcess(a, 1, out, ""))

    def must_not_run():
        raise AssertionError("인증 실패인데 이미지 점검을 했다")
    monkeypatch.setattr(cc, "run_vision_check", must_not_run)
    monkeypatch.setattr(cc, "STATUS_PATH", tmp_path / "s.json")
    assert cc.main(["--no-notify"]) == 1


# ── GET /health/claude ──


def test_점검_결과를_비밀값_없이_보여_준다(monkeypatch, tmp_path):
    import server

    path = tmp_path / "claude_check.json"
    path.write_text(json.dumps({"ok": False, "kind": "auth", "checked_at": "2026-10-12T09:00:00+09:00",
                                "token_days_left": 12, "detail": "401 OAuth access token is invalid"}))
    monkeypatch.setattr(server, "_CLAUDE_STATUS_PATH", str(path))
    body = TestClient(server.app).get("/health/claude").json()
    assert body == {"ok": False, "kind": "auth", "checked_at": "2026-10-12T09:00:00+09:00",
                    "token_days_left": 12}


def test_점검_기록이_없으면_ok는_null(monkeypatch, tmp_path):
    import server

    monkeypatch.setattr(server, "_CLAUDE_STATUS_PATH", str(tmp_path / "none.json"))
    assert TestClient(server.app).get("/health/claude").json()["ok"] is None
