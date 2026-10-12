#!/bin/sh
# install.sh로 등록한 서버·ngrok 터널·잠자기 방지를 끄고 등록을 지운다.
set -u
for name in ngrok server caffeinate; do
  launchctl bootout "gui/$(id -u)/com.gamdo.$name" 2>/dev/null && echo "stopped com.gamdo.$name"
  rm -f "$HOME/Library/LaunchAgents/com.gamdo.$name.plist"
done
