"""Claude CLI 인증 정기 점검.

서버는 이 Mac에 로그인된 claude CLI(또는 .env의 CLAUDE_CODE_OAUTH_TOKEN)로 사진을
분석한다. 토큰이 만료되거나 로그인이 풀려도 /health는 정상으로 나오므로, 사용자가
분석에 실패하고 나서야 알게 된다. 이 스크립트는

  1. 서버와 같은 환경(child_env)으로 claude CLI에 아주 짧은 요청을 보내 응답을 확인하고,
  2. 실패하거나 토큰 만료가 가까우면 macOS 알림을 띄우고,
  3. 결과를 상태 파일(JSON)에 남긴다 — 서버의 GET /health/claude가 이 파일을 보여 준다.

launchd(com.gamdo.claudecheck)가 로그인할 때와 매일 오전 9시에 실행한다.

  .venv/bin/python ops/claude_check.py            # 점검 + 실패 시 알림
  .venv/bin/python ops/claude_check.py --no-notify

토큰 만료 경고: .env에 CLAUDE_CODE_OAUTH_TOKEN_ISSUED=YYYY-MM-DD(발급일)를 두면
1년 만료까지 남은 날을 계산해 30일 안이면 미리 알린다.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(ROOT / ".env")

from claude_client import child_env  # noqa: E402

STATUS_PATH = Path(os.getenv(
    "GAMDO_CLAUDE_STATUS_PATH",
    str(Path.home() / "Library" / "Logs" / "gamdo" / "claude_check.json"),
))
TOKEN_VALID_DAYS = 365   # claude setup-token이 만드는 토큰의 유효기간
WARN_DAYS = 30           # 만료까지 이 날수 이하면 미리 알린다
TIMEOUT_S = 120
# 인증 문제로 보는 오류 문구. CLI JSON의 키 이름(output_tokens 등)이 아니라
# 오류 메시지(result)와 stderr에만 대어 본다.
_AUTH_MARKERS = ("not logged in", "/login", "failed to authenticate", "oauth access token",
                 "invalid api key", "token has expired", "token is expired")
_AUTH_STATUS = (401, 403)

CMD = [
    "claude", "-p", "Reply with exactly: OK",
    "--output-format", "json",
    "--model", "haiku",          # 인증만 보면 되므로 가장 가벼운 모델
    "--strict-mcp-config",
    "--tools", "",
]


def classify(returncode: int, stdout: str, stderr: str) -> tuple[bool, str, str]:
    """(성공, 종류, 설명). 종류: ok | auth | error.

    --output-format json 응답은 실패해도 JSON이다: is_error, result(오류 메시지),
    api_error_status(HTTP 상태). 인증 판정은 상태 코드와 오류 메시지로만 한다.
    """
    try:
        data = json.loads(stdout)
    except ValueError:
        data = None
    if not isinstance(data, dict):
        data = {}
    message = str(data.get("result") or "").strip()
    if returncode == 0 and not data.get("is_error") and "OK" in message:
        return True, "ok", "응답 정상"

    status = data.get("api_error_status")
    text = f"{message}\n{stderr}".lower() if data else f"{stdout}\n{stderr}".lower()
    auth = status in _AUTH_STATUS or any(m in text for m in _AUTH_MARKERS)
    detail = message or next(
        (ln.strip() for ln in f"{stderr}\n{stdout}".splitlines() if ln.strip()),
        f"exit {returncode}",
    )
    return False, ("auth" if auth else "error"), detail[:200]


def token_days_left(today: dt.date | None = None) -> int | None:
    issued = os.getenv("CLAUDE_CODE_OAUTH_TOKEN_ISSUED", "").strip()
    if not issued:
        return None
    try:
        start = dt.date.fromisoformat(issued)
    except ValueError:
        return None
    today = today or dt.date.today()
    return (start + dt.timedelta(days=TOKEN_VALID_DAYS) - today).days


def run_check() -> dict:
    started = time.monotonic()
    try:
        proc = subprocess.run(CMD, env=child_env(os.environ), capture_output=True,
                              text=True, timeout=TIMEOUT_S, cwd=str(ROOT))
        ok, kind, detail = classify(proc.returncode, proc.stdout, proc.stderr)
    except FileNotFoundError:
        ok, kind, detail = False, "error", "claude CLI를 찾지 못함 (PATH 확인)"
    except subprocess.TimeoutExpired:
        ok, kind, detail = False, "error", f"{TIMEOUT_S}초 안에 응답 없음"
    return {
        "ok": ok,
        "kind": kind,
        "detail": detail,
        "checked_at": dt.datetime.now().astimezone().isoformat(timespec="seconds"),
        "duration_s": round(time.monotonic() - started, 1),
        "token_days_left": token_days_left(),
    }


def write_status(status: dict, path: Path | None = None) -> None:
    path = path or STATUS_PATH  # 호출 시점에 읽는다 (테스트가 STATUS_PATH를 바꿀 수 있게)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(status, ensure_ascii=False, indent=1))
    tmp.replace(path)


def notify(title: str, message: str) -> None:
    """macOS 알림. 알림을 띄울 수 없는 환경이면 조용히 넘어간다."""
    def q(s: str) -> str:
        return s.replace("\\", "\\\\").replace('"', '\\"')
    script = f'display notification "{q(message)}" with title "{q(title)}" sound name "Basso"'
    try:
        subprocess.run(["osascript", "-e", script], capture_output=True, timeout=10)
    except Exception:
        pass


def alerts_for(status: dict) -> list[tuple[str, str]]:
    out = []
    if not status["ok"]:
        if status["kind"] == "auth":
            out.append(("감도 서버: Claude 인증 실패",
                        "사진 분석이 멈췄습니다. claude setup-token으로 새 토큰을 만들어 .env를 바꾸세요."))
        else:
            out.append(("감도 서버: Claude 응답 없음", status["detail"]))
    days = status.get("token_days_left")
    if days is not None and days <= WARN_DAYS:
        msg = (f"토큰 만료까지 {days}일 남았습니다." if days >= 0 else "토큰이 만료됐습니다.")
        out.append(("감도 서버: Claude 토큰 교체 필요", msg + " claude setup-token으로 새로 만드세요."))
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--no-notify", action="store_true", help="알림 없이 결과만 기록")
    args = parser.parse_args(argv)

    status = run_check()
    write_status(status)
    print(json.dumps(status, ensure_ascii=False))
    if not args.no_notify:
        for title, msg in alerts_for(status):
            notify(title, msg)
    return 0 if status["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
