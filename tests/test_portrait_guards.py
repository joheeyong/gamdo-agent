"""인물 사진에서 몸이 변형되지 않도록 지키는 규칙들.

과거 증상: "기존보다 다리가 짧고 옆으로 퍼져서 뚱뚱하게 나온다."
원인이 셋이었다 — 체형 보정 토글이 서버에 전달되지 않았고, 건축용 원근
보정이 인물에도 걸렸고, 세로 크롭이 전신 사진의 머리와 발을 잘랐다.
"""

import numpy as np
import pytest
from PIL import Image

from image_processor import apply_auto_edits, apply_instagram_ratio
from param_engine import build_params_with_comment

PORTRAIT_PROFILE = {
    "trendCategory": "clean_minimal",
    "colorPreference": {
        "preferredTones": "neutral", "saturationTendency": "medium",
        "brightnessTendency": "medium", "contrast": "medium",
    },
    "editingStyle": {
        "filterTendency": "moderate", "grainPreference": "none",
        "vignettePreference": "none", "skinRetouchLevel": "moderate",
    },
}

RESHAPE = {
    "face_slim": 0.3, "jaw_sharpen": 0.2, "eye_enlarge": 0.1,
    "leg_stretch": 0.3, "shoulder_width": 0.2, "waist_slim": 0.3,
}

_BODY_KEYS = ("face_slim", "jaw_sharpen", "eye_enlarge",
              "leg_stretch", "shoulder_width", "waist_slim")


@pytest.fixture
def photo():
    rng = np.random.default_rng(0)
    a = np.full((1200, 800, 3), 150, np.float32) + rng.normal(0, 6, (1200, 800, 3))
    return Image.fromarray(a.clip(0, 255).astype(np.uint8))


def test_reshape_off_by_default(photo):
    """토글을 켜지 않으면 체형·얼굴 보정값이 나가지 않는다."""
    params, _ = build_params_with_comment(
        photo, PORTRAIT_PROFILE,
        {"subjectType": "인물", "reshapeParams": RESHAPE},
    )
    assert "reshapeParams" not in params


def test_reshape_applied_when_enabled(photo):
    """사용자가 켜면 그대로 전달된다."""
    params, _ = build_params_with_comment(
        photo, PORTRAIT_PROFILE,
        {"subjectType": "인물", "reshapeParams": RESHAPE},
        reshape_enabled=True,
    )
    assert params.get("reshapeParams") == RESHAPE


def test_reshape_never_on_non_portrait(photo):
    """인물이 아니면 토글이 켜져 있어도 몸을 건드리지 않는다."""
    params, _ = build_params_with_comment(
        photo, PORTRAIT_PROFILE,
        {"subjectType": "풍경", "reshapeParams": RESHAPE},
        reshape_enabled=True,
    )
    assert "reshapeParams" not in params


def test_vertical_crop_blocked_keeps_head_and_feet():
    """전신 비율 사진의 위아래를 자르지 않는다.

    가운데 기준으로 자르면 머리와 발이 함께 날아가 다리가 짧아 보인다.
    """
    tall = Image.fromarray(np.full((1400, 900, 3), 128, np.uint8))
    assert apply_instagram_ratio(tall, "4:5", allow_vertical_crop=False).size == (900, 1400)
    assert apply_instagram_ratio(tall, "4:5", allow_vertical_crop=True).size[1] < 1400


def test_horizontal_crop_still_allowed():
    """좌우 크롭은 인물이어도 막지 않는다 (다리 길이와 무관)."""
    wide = Image.fromarray(np.full((800, 1600, 3), 128, np.uint8))
    out = apply_instagram_ratio(wide, "4:5", allow_vertical_crop=False)
    assert out.size == (640, 800)


def test_auto_edits_passes_vertical_crop_flag():
    """apply_auto_edits가 플래그를 비율 크롭까지 전달한다."""
    tall = Image.fromarray(np.full((1400, 900, 3), 128, np.uint8))
    edits = {"instagram_ratio": "4:5"}
    assert apply_auto_edits(tall, edits, allow_vertical_crop=False).size == (900, 1400)
    assert apply_auto_edits(tall, edits, allow_vertical_crop=True).size[1] < 1400
