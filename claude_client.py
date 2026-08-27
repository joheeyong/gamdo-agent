"""Anthropic Claude API 호출 래퍼."""

import json
import anthropic
from prompts import (
    ANALYZE_USER_SYSTEM,
    ANALYZE_USER_PROMPT,
    TRANSFORM_PHOTO_SYSTEM,
    TRANSFORM_PHOTO_PROMPT,
)


MODEL = "claude-sonnet-4-20250514"


def _parse_json_response(text: str) -> dict:
    """Claude 응답에서 JSON을 추출한다."""
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
            line += " (이미지 첨부됨)"
        if item.get("timestamp"):
            line += f" — {item['timestamp']}"
        parts.append(line)
    return "\n".join(parts)


def _build_image_content_blocks(items: list[dict]) -> list[dict]:
    """이미지가 있는 항목들에서 Claude Vision용 content block을 만든다."""
    blocks: list[dict] = []
    for item in items:
        if item.get("image_base64"):
            blocks.append({
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": item.get("media_type", "image/jpeg"),
                    "data": item["image_base64"],
                },
            })
    return blocks


def analyze_user(
    api_key: str,
    posts: list[dict],
    feeds: list[dict],
    stories: list[dict],
) -> dict:
    """사용자의 게시글/피드/스토리를 분석하여 스타일 프로필을 반환한다."""
    client = anthropic.Anthropic(api_key=api_key)

    prompt_text = ANALYZE_USER_PROMPT.format(
        posts=_format_items(posts),
        feeds=_format_items(feeds),
        stories=_format_items(stories),
    )

    # 이미지가 포함된 항목이 있으면 Vision API로 전달
    all_items = posts + feeds + stories
    image_blocks = _build_image_content_blocks(all_items)

    content: list[dict] = []
    content.extend(image_blocks)
    content.append({"type": "text", "text": prompt_text})

    response = client.messages.create(
        model=MODEL,
        max_tokens=4096,
        system=ANALYZE_USER_SYSTEM,
        messages=[{"role": "user", "content": content}],
    )

    text = response.content[0].text
    return _parse_json_response(text)


def transform_photo(
    api_key: str,
    style_profile: dict,
    image_base64: str,
    media_type: str = "image/jpeg",
) -> dict:
    """사용자 스타일 프로필에 맞춰 사진 보정 가이드를 반환한다."""
    client = anthropic.Anthropic(api_key=api_key)

    prompt_text = TRANSFORM_PHOTO_PROMPT.format(
        style_profile=json.dumps(style_profile, ensure_ascii=False, indent=2),
    )

    content: list[dict] = [
        {
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": media_type,
                "data": image_base64,
            },
        },
        {"type": "text", "text": prompt_text},
    ]

    response = client.messages.create(
        model=MODEL,
        max_tokens=4096,
        system=TRANSFORM_PHOTO_SYSTEM,
        messages=[{"role": "user", "content": content}],
    )

    text = response.content[0].text
    return _parse_json_response(text)
