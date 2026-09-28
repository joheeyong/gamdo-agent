"""claude CLI 자식 프로세스에 넘기는 환경변수 검증.

과거 버그: 중첩 세션 변수를 확실히 지우려고 "CLAUDE_CODE_ 접두어를 전부
제거"로 바꿨다가 CLAUDE_CODE_OAUTH_TOKEN까지 날렸다. 그 변수가 곧 로그인
자격증명이라 서버의 모든 분석 요청이 "Not logged in"으로 실패했다.
"""

from claude_client import child_env


def test_oauth_token_survives():
    """자격증명은 반드시 남아야 한다 — 이걸 지우면 서버 전체가 죽는다."""
    env = child_env({"CLAUDE_CODE_OAUTH_TOKEN": "tok", "PATH": "/usr/bin"})
    assert env.get("CLAUDE_CODE_OAUTH_TOKEN") == "tok"


def test_nesting_markers_removed():
    """부모 Claude Code 세션 흔적은 지운다."""
    src = {
        "CLAUDECODE": "1",
        "CLAUDE_CODE_ENTRYPOINT": "cli",
        "CLAUDE_CODE_SESSION_ID": "s",
        "CLAUDE_CODE_CHILD_SESSION": "1",
        "CLAUDE_CODE_MESSAGING_SOCKET": "/tmp/x",
        "CLAUDE_CODE_MESSAGING_TOKEN": "t",
        "CLAUDE_CODE_EXECPATH": "/x",
        "CLAUDE_PID": "9",
        "CLAUDE_EFFORT": "high",
    }
    assert child_env(src) == {}


def test_expired_api_key_removed():
    """.env에 남은 옛 API 키가 OAuth 경로를 가로채지 않게 한다."""
    assert "ANTHROPIC_API_KEY" not in child_env({"ANTHROPIC_API_KEY": "sk-old"})


def test_other_auth_paths_preserved():
    """API 키 외의 정상 인증 수단까지 지우면 안 된다."""
    env = child_env({"ANTHROPIC_AUTH_TOKEN": "t"})
    assert env.get("ANTHROPIC_AUTH_TOKEN") == "t"


def test_ordinary_variables_untouched():
    """PATH, HOME 등은 그대로 넘어가야 CLI가 실행된다."""
    src = {"PATH": "/usr/bin", "HOME": "/Users/x", "LANG": "ko_KR.UTF-8"}
    assert child_env(src) == src


def test_realistic_nested_session():
    """실제 중첩 세션 환경 한 벌을 통째로 넣어 본다."""
    src = {
        "CLAUDE_CODE_OAUTH_TOKEN": "tok",
        "CLAUDECODE": "1",
        "CLAUDE_CODE_ENTRYPOINT": "cli",
        "CLAUDE_PID": "1",
        "ANTHROPIC_API_KEY": "sk-expired",
        "PATH": "/usr/bin",
        "HOME": "/Users/x",
    }
    env = child_env(src)
    assert set(env) == {"CLAUDE_CODE_OAUTH_TOKEN", "PATH", "HOME"}
