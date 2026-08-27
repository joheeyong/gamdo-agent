"""Claude Code CLI(claude -p)를 사용한 분석 클라이언트. API 키 불필요."""

import json
import os
import subprocess
import logging
from prompts import (
    ANALYZE_USER_SYSTEM,
    ANALYZE_USER_PROMPT,
    TRANSFORM_PHOTO_SYSTEM,
    TRANSFORM_PHOTO_PROMPT,
)

log = logging.getLogger("gamdo-agent")


def _parse_json_response(text: str) -> dict:
    """응답 텍스트에서 JSON을 추출한다."""
    text = text.strip()
    if text.startswith("```json"):
        text = text[7:]
    elif text.startswith("```"):
        text = text[3:]
    if text.endswith("```"):
        text = text[:-3]
    return json.loads(text.strip())


def _format_items(items: list[dict]) -> str:
    """PostItem 목록을 텍스트로 변환."""
    if not items:
        return "(없음)"
    parts = []
    for i, item in enumerate(items, 1):
        line = f"[{i}]"
        if item.get("text"):
            line += f" {item['text']}"
        if item.get("image_url"):
            line += f" (이미지: {item['image_url']})"
        if item.get("image_base64"):
            line += f" (이미지 첨부됨, {len(item['image_base64'])}자)"
        if item.get("timestamp"):
            line += f" — {item['timestamp']}"
        parts.append(line)
    return "\n".join(parts)


def _call_claude(prompt: str, system_prompt: str, timeout: int = 120) -> str:
    """claude -p 를 subprocess로 호출한다. CLI 인증을 그대로 사용."""
    cmd = [
        "claude", "-p",
        "--output-format", "text",
        "--model", "claude-opus-4-6",
        "--max-turns", "1",
    ]

    if system_prompt:
        cmd.extend(["--system-prompt", system_prompt])

    log.info("Calling claude CLI (prompt length: %d)", len(prompt))

    # CLAUDECODE 환경변수를 제거해야 중첩 세션 에러가 안 남
    env = {k: v for k, v in os.environ.items() if k != "CLAUDECODE"}

    result = subprocess.run(
        cmd,
        input=prompt,
        capture_output=True,
        text=True,
        timeout=timeout,
        env=env,
    )

    if result.returncode != 0:
        log.error("claude CLI error: %s", result.stderr)
        raise RuntimeError(f"claude CLI failed: {result.stderr}")

    log.info("claude CLI response length: %d", len(result.stdout))
    return result.stdout


def analyze_user(
    posts: list[dict],
    feeds: list[dict],
    stories: list[dict],
    **_kwargs,
) -> dict:
    """사용자의 게시글/피드/스토리를 분석하여 스타일 프로필을 반환한다."""
    prompt_text = ANALYZE_USER_PROMPT.format(
        posts=_format_items(posts),
        feeds=_format_items(feeds),
        stories=_format_items(stories),
    )

    result = _call_claude(prompt_text, ANALYZE_USER_SYSTEM)
    return _parse_json_response(result)


def transform_photo(
    style_profile: dict,
    image_base64: str,
    media_type: str = "image/jpeg",
    **_kwargs,
) -> dict:
    """사용자 스타일 프로필에 맞춰 사진 보정 가이드를 반환한다."""
    prompt_text = TRANSFORM_PHOTO_PROMPT.format(
        style_profile=json.dumps(style_profile, ensure_ascii=False, indent=2),
    )

    # 이미지 정보를 프롬프트에 포함
    full_prompt = (
        f"{prompt_text}\n\n"
        f"[참고: 사진이 첨부되어 있습니다. media_type={media_type}, "
        f"base64 길이={len(image_base64)}자]\n"
        f"사진의 base64 데이터 앞부분: {image_base64[:200]}..."
    )

    result = _call_claude(full_prompt, TRANSFORM_PHOTO_SYSTEM)
    return _parse_json_response(result)
