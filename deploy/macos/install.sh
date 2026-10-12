#!/bin/sh
# 감도 서버를 macOS launchd에 등록한다 — 서버·ngrok 터널·잠자기 방지·Claude 정기 점검.
#
# 셋 다 로그인하면 자동으로 켜지고, 꺼지면 launchd가 다시 띄운다.
#
#   NGROK_URL=https://<고정 도메인>.ngrok-free.dev deploy/macos/install.sh
#   deploy/macos/install.sh --dry-run   # 등록하지 않고 만들어질 파일만 출력 폴더에 쓴다
#
# 전제: gamdo-agent/.venv 설치, .env에 CLAUDE_CODE_OAUTH_TOKEN, ngrok 로그인(authtoken).
set -eu

HERE=$(cd "$(dirname "$0")" && pwd)
AGENT_DIR=$(cd "$HERE/../.." && pwd)
LOG_DIR="$HOME/Library/Logs/gamdo"
TARGET="$HOME/Library/LaunchAgents"
DRY_RUN=0
[ "${1:-}" = "--dry-run" ] && DRY_RUN=1

NGROK_URL=${NGROK_URL:-https://cinderella-vitiable-noncapitalistically.ngrok-free.dev}
NGROK_BIN=$(command -v ngrok || true)
CLAUDE_BIN=$(command -v claude || true)
if [ -z "$NGROK_BIN" ]; then echo "ngrok이 없습니다 (brew install ngrok)" >&2; exit 1; fi
if [ -z "$CLAUDE_BIN" ]; then echo "claude CLI가 없습니다" >&2; exit 1; fi
if [ ! -x "$AGENT_DIR/.venv/bin/python" ]; then echo "$AGENT_DIR/.venv가 없습니다 (uv sync)" >&2; exit 1; fi

# launchd는 로그인 셸의 PATH를 모른다. 서버가 claude CLI를 찾을 수 있게 넘긴다.
PATH_VALUE="$(dirname "$CLAUDE_BIN"):$(dirname "$NGROK_BIN"):/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin"

if [ $DRY_RUN -eq 1 ]; then
  TARGET=$(mktemp -d)
  echo "dry-run: $TARGET 에 씁니다 (등록하지 않음)"
fi
mkdir -p "$TARGET" "$LOG_DIR"

for name in caffeinate server ngrok claudecheck; do
  src="$HERE/com.gamdo.$name.plist.template"
  dst="$TARGET/com.gamdo.$name.plist"
  sed -e "s#__AGENT_DIR__#$AGENT_DIR#g" \
      -e "s#__LOG_DIR__#$LOG_DIR#g" \
      -e "s#__NGROK_URL__#$NGROK_URL#g" \
      -e "s#__NGROK_BIN__#$NGROK_BIN#g" \
      -e "s#__PATH__#$PATH_VALUE#g" \
      "$src" > "$dst"
  plutil -lint "$dst" >/dev/null
  if [ $DRY_RUN -eq 0 ]; then
    # 이미 등록돼 있으면 내렸다가 새 설정으로 다시 올린다
    launchctl bootout "gui/$(id -u)/com.gamdo.$name" 2>/dev/null || true
    launchctl bootstrap "gui/$(id -u)" "$dst"
  fi
  echo "com.gamdo.$name → $dst"
done

[ $DRY_RUN -eq 1 ] && exit 0
echo
echo "확인:  curl -s http://localhost:8000/health"
echo "로그:  $AGENT_DIR/server.log, $LOG_DIR/ngrok.log"
