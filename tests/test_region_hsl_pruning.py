"""영역 보정·HSL이 거의 모든 사진에 겹겹이 걸리던 문제의 회귀 테스트.

실측(7장): HSL 7/7, face 영역 4/4, background가 인물의 몸까지 덮어 전역 보정과
같은 방향으로 한 번 더 걸림(astro ΔE 3.6). 효과 없는 값·중복 값은 버리고
실제로 필요한 국소 보정(날아간 창문 등)은 남아야 한다.
"""

import copy

import numpy as np
from PIL import Image

from param_engine import _prune_hsl, _prune_region_params, build_params_with_comment

PROFILE = {
    "primaryStyle": "내추럴",
    "colorPreference": {"preferredTones": "neutral", "saturationTendency": "medium",
                        "brightnessTendency": "medium", "contrast": "medium"},
    "editingStyle": {"filterTendency": "strong"},
}
GLOBAL = {"brightness": 0.15, "contrast": 0.06, "saturation": -0.07, "temperature": 0.2,
          "highlights": 0.1, "shadows": 0.0}


def _mixed() -> Image.Image:
    """파랑 60%, 초록 39%, 빨강 1%."""
    arr = np.zeros((100, 100, 3), np.uint8)
    arr[:60] = (60, 110, 200)
    arr[60:99] = (60, 150, 70)
    arr[99:] = (210, 40, 40)
    return Image.fromarray(arr)


def _prune(regions, portrait=True, params=GLOBAL, gain=0.8):
    analysis = {"regionParams": copy.deepcopy(regions)}
    _prune_region_params(analysis, dict(params), portrait, gain)
    return analysis["regionParams"]


# ── HSL ──

def test_hsl_for_absent_color_is_dropped():
    out = _prune_hsl({"red": {"saturation": 0.2}, "green": {"saturation": 0.2}}, _mixed())
    assert "red" not in out and out["green"] == {"saturation": 0.2}


def test_hsl_only_absent_colors_becomes_none():
    assert _prune_hsl({"purple": {"saturation": -0.12}}, _mixed()) is None


def test_hsl_color_covering_most_of_frame_is_halved():
    """카페 사진의 orange(81%)처럼 사진 대부분을 덮는 색은 사실상 전역 채도다."""
    arr = np.full((100, 100, 3), (50, 100, 220), np.uint8)
    arr[80:] = (60, 150, 70)
    out = _prune_hsl({"blue": {"saturation": -0.2, "lightness": 0.1}}, Image.fromarray(arr))
    assert out["blue"] == {"saturation": -0.1, "lightness": 0.05}


def test_hsl_channel_count_is_capped():
    img = Image.fromarray(np.concatenate([
        np.full((30, 90, 3), c, np.uint8) for c in
        [(60, 110, 200), (60, 150, 70), (210, 40, 40)]]))
    out = _prune_hsl({"blue": {"saturation": 0.1}, "green": {"saturation": 0.3},
                      "red": {"saturation": 0.05}}, img)
    assert list(out) == ["green", "blue"]


def test_hsl_prune_runs_in_param_engine():
    params, _ = build_params_with_comment(
        _mixed(), PROFILE, {"hslAdjust": {"magenta": {"saturation": 0.3}}})
    assert "hslAdjust" not in params


# ── 영역 ──

def test_tiny_region_values_are_dropped():
    out = _prune({"sky": {"brightness": 0.03, "saturation": 0.02}}, portrait=False)
    assert out is None


def test_face_brightness_duplicating_global_is_dropped():
    out = _prune({"face": {"brightness": 0.1, "blemish_removal": 0.0, "skin_smoothing": 0.0}})
    assert out is None


def test_face_brightness_against_global_is_kept():
    """전역이 밝히는데 얼굴만 누르는 건 의도된 판단이다 (얼굴이 날아갈 때)."""
    out = _prune({"face": {"brightness": -0.1}})
    assert out == {"face": {"brightness": -0.1}}


def test_portrait_background_is_dropped():
    """background는 얼굴 피부만 뺀 전부라 인물의 몸·옷까지 바랜다."""
    out = _prune({"background": {"contrast": -0.1, "saturation": -0.1}})
    assert out is None


def test_landscape_background_same_direction_is_damped():
    out = _prune({"background": {"saturation": -0.12, "contrast": -0.1}}, portrait=False)
    # 채도는 전역과 같은 방향이라 절반, 대비는 반대 방향이라 그대로
    assert out == {"background": {"saturation": -0.06, "contrast": -0.1}}


def test_useful_local_edit_is_kept():
    """날아간 창문 하이라이트 복구는 전역으로 대신할 수 없다."""
    window = {"area": {"x": 0.55, "y": 0.1, "width": 0.2, "height": 0.35}, "shape": "rect",
              "feather": 0.3, "reason": "창문 날아감", "highlights": -0.45, "brightness": -0.15}
    assert _prune({"local_0": window}) == {"local_0": window}


def test_overlapping_opposite_locals_keep_only_first():
    a = {"area": {"x": 0.36, "y": 0.15, "width": 0.38, "height": 0.3},
         "brightness": -0.1, "highlights": -0.3}
    b = {"area": {"x": 0.42, "y": 0.19, "width": 0.3, "height": 0.35},
         "brightness": 0.15, "shadows": 0.2}
    assert list(_prune({"local_0": a, "local_1": b})) == ["local_0"]


def test_local_count_is_capped():
    regions = {f"local_{i}": {"area": {"x": 0.2 * i, "y": 0.0, "width": 0.15, "height": 0.2},
                              "highlights": -0.3} for i in range(4)}
    assert list(_prune(regions)) == ["local_0", "local_1"]


def test_no_correction_drops_all_regions():
    assert _prune({"local_0": {"area": {"x": 0.1, "y": 0.1, "width": 0.2, "height": 0.2},
                               "highlights": -0.4}}, gain=0.0) is None


def test_request_dict_is_not_mutated():
    regions = {"background": {"saturation": -0.1}, "face": {"brightness": 0.1}}
    snapshot = copy.deepcopy(regions)
    analysis = {"regionParams": regions}
    _prune_region_params(analysis, dict(GLOBAL), True, 0.8)
    assert regions == snapshot
