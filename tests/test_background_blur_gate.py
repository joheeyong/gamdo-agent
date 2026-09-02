"""배경 흐림이 걸리는 조건 검증.

과거 증상: "배경 흐림이 너무 심하다. 풍경일 때는 하지 말아야 한다."
원인은 세기가 아니라 게이트였다. 최소 인물 비중이 0.5%여서, 풍경 속
작은 사람 하나로 사진의 거의 전부가 흐려졌다.
"""

import numpy as np
import pytest
from PIL import Image

from image_processor import (
    _BLUR_FULL_SUBJECT,
    _BLUR_MIN_SUBJECT,
    apply_background_blur,
    background_blur_scale,
)


def test_landscape_with_tiny_person_is_untouched():
    """풍경 속 작은 사람에게는 걸지 않는다."""
    for coverage in (0.005, 0.01, 0.02, 0.05):
        assert background_blur_scale(coverage) == 0.0, f"인물 {coverage:.1%}"


def test_closeup_gets_full_strength():
    """근접샷·상반신은 온전한 세기."""
    for coverage in (0.20, 0.35, 0.60):
        assert background_blur_scale(coverage) == 1.0, f"인물 {coverage:.1%}"


def test_scale_is_monotonic_and_bounded():
    """세기가 갑자기 튀지 않고 0~1 안에 머문다."""
    prev = -1.0
    for i in range(101):
        v = background_blur_scale(i / 100)
        assert 0.0 <= v <= 1.0
        assert v >= prev, "비중이 커졌는데 세기가 줄었다"
        prev = v


def test_taper_band_is_partial():
    """경계 구간은 0도 1도 아니어야 한다 — 갑작스러운 전환 방지."""
    mid = (_BLUR_MIN_SUBJECT + _BLUR_FULL_SUBJECT) / 2
    assert 0.0 < background_blur_scale(mid) < 1.0


def test_gate_threshold_is_meaningful():
    """게이트가 다시 유명무실해지는 것을 막는다."""
    assert _BLUR_MIN_SUBJECT >= 0.05, "풍경 속 사람을 걸러낼 수 없는 값이다"
    assert _BLUR_FULL_SUBJECT > _BLUR_MIN_SUBJECT


def test_no_person_returns_original():
    """사람이 없는 사진은 원본 그대로 (분할 실패 포함)."""
    rng = np.random.default_rng(0)
    arr = (rng.normal(128, 30, (400, 600, 3))).clip(0, 255).astype(np.uint8)
    img = Image.fromarray(arr)
    out = apply_background_blur(img, 0.5)
    assert np.array_equal(np.asarray(out), arr)


def test_zero_strength_is_noop():
    img = Image.fromarray(np.full((100, 100, 3), 120, np.uint8))
    assert apply_background_blur(img, 0.0) is img
