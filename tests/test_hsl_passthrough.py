"""색계열별(HSL) 조정이 모델 → 엔진 → 픽셀까지 이어지는지 검증한다.

apply_hsl_adjust는 오래전부터 구현돼 있었지만, 수치 계산을 서버로 옮길 때
param_engine이 hslAdjust를 내보내지 않게 되면서 통로가 끊겨 있었다.
색계열 단위 판단은 히스토그램 평균으로 나오지 않으므로 모델이 정한다.
"""

import numpy as np
import pytest
from PIL import Image

from image_processor import analysis_to_transform_params, apply_hsl_adjust
from param_engine import _HSL_LIMITS, build_params_with_comment

PROFILE = {
    "primaryStyle": "내추럴",
    "colorPreference": {"preferredTones": "neutral", "saturationTendency": "medium",
                        "brightnessTendency": "medium", "contrast": "medium"},
    "editingStyle": {"filterTendency": "strong"},   # gain 1.0 — 클램프만 보게
}


def _img() -> Image.Image:
    """초록·파랑·주황이 각각 3분의 1씩 있는 이미지."""
    arr = np.zeros((90, 120, 3), np.uint8)
    arr[:30] = (60, 150, 70)
    arr[30:60] = (60, 110, 200)
    arr[60:] = (220, 140, 60)
    return Image.fromarray(arr)


def _build(hsl):
    params, _ = build_params_with_comment(_img(), PROFILE, {"hslAdjust": hsl})
    return params


def test_model_hsl_reaches_transform_params():
    params = _build({"green": {"saturation": 0.3, "lightness": -0.2}})
    assert params["hslAdjust"] == {"green": {"saturation": 0.3, "lightness": -0.2}}
    resolved = analysis_to_transform_params({"recommendedParams": params})
    # 빠진 키는 0.0으로 채워진다 (apply_hsl_adjust가 0에 가까우면 건너뛴다)
    assert resolved["hsl_adjust"] == {
        "green": {"hue": 0.0, "saturation": 0.3, "lightness": -0.2}}


def test_values_are_clamped_per_key():
    """1.0은 채도 ±80(255 기준)이라 그대로 태우면 색이 튄다."""
    params = _build({"blue": {"hue": 1.0, "saturation": -1.0, "lightness": 1.0}})
    assert params["hslAdjust"]["blue"] == {
        "hue": _HSL_LIMITS["hue"],
        "saturation": -_HSL_LIMITS["saturation"],
        "lightness": _HSL_LIMITS["lightness"],
    }


def test_filter_tendency_scales_the_strength():
    """보정 강도 성향이 낮은 사용자는 색계열 조정도 약하게 받는다."""
    weak_profile = {**PROFILE, "editingStyle": {"filterTendency": "minimal"}}
    weak, _ = build_params_with_comment(
        _img(), weak_profile, {"hslAdjust": {"green": {"saturation": 0.4}}})
    strong = _build({"green": {"saturation": 0.4}})
    assert weak["hslAdjust"]["green"]["saturation"] < strong["hslAdjust"]["green"]["saturation"]


@pytest.mark.parametrize("hsl", [
    None, {}, "green", {"chartreuse": {"saturation": 0.3}},
    {"green": {"saturation": "조금"}}, {"green": {"saturation": 0.001}},
])
def test_unusable_values_leave_the_key_out(hsl):
    """알 수 없는 색 이름·숫자가 아닌 값·0에 가까운 값은 버린다."""
    assert "hslAdjust" not in _build(hsl)


def test_green_channel_only_moves_green_pixels():
    img = _img()
    out = np.array(apply_hsl_adjust(img, {"green": {"saturation": 0.5}}), np.int16)
    before = np.array(img, np.int16)
    delta = np.abs(out - before).sum(axis=2)
    assert delta[:30].mean() > 10, "초록이 안 바뀌었다"
    assert delta[30:].mean() < 1, "초록만 지정했는데 다른 색이 바뀌었다"
