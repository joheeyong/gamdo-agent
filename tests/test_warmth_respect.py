"""웜톤이 모든 사진에 덧칠되지 않는지 — 장면의 원래 빛 존중 + 웜 보정 겹침 방지.

실측(기본 레시피 7장): temperature +0.18~0.22가 대부분 사진에서 가장 큰 보정이었다.
라벤더빛 새벽 호수·파란 수트 인물처럼 "차가운 색이 장면 자체"인 사진까지 덥혔고,
이미 따뜻한 라테에 음식 레시피 웜 상수가 그대로 얹혔다.
"""

import numpy as np
from PIL import Image

from param_engine import build_params_with_comment, measure_image_stats


def _manual(trend):
    return {"trendCategory": trend, "styleSource": "manual"}


def _cool_scene() -> Image.Image:
    """위는 밝은 라벤더 하늘, 아래는 어두운 숲 — 캐스트가 고르지 않은 쿨 장면."""
    rng = np.random.default_rng(1)
    arr = np.zeros((200, 240, 3), np.float32)
    arr[:100] = (160, 165, 198)
    arr[100:] = (42, 48, 38)
    arr += rng.normal(0, 4, arr.shape)
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))


def _neutral_scene() -> Image.Image:
    rng = np.random.default_rng(2)
    y = np.linspace(0.15, 0.9, 200)[:, None, None]
    arr = np.broadcast_to(y, (200, 240, 3)) * 255
    arr = arr + rng.normal(0, 4, arr.shape)
    arr[60:140, 60:180] = (200, 140, 110)   # 피부 비슷한 블록
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))


def _cast(img: Image.Image, rgb) -> Image.Image:
    arr = np.asarray(img, np.float32) * np.asarray(rgb, np.float32)
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))


def _params(img, profile=None, analysis=None):
    return build_params_with_comment(img, profile, analysis or {"subjectType": "풍경"})[0]


def test_fixture_is_a_non_uniform_cool_scene():
    s = measure_image_stats(_cool_scene())
    assert s["warmth"] < -0.08 and s["cast_uniformity"] < 0.2


def test_cool_scene_is_not_warmed_by_default():
    p = _params(_cool_scene())
    assert p["temperature"] <= 0.03


def test_warm_style_still_warms_a_cool_scene():
    for trend in ("golden_hour", "warm_film"):
        assert _params(_cool_scene(), _manual(trend))["temperature"] >= 0.12, trend
    warm_pref = {"colorPreference": {"preferredTones": "warm"}}
    assert _params(_cool_scene(), warm_pref)["temperature"] >= 0.12


def test_uniform_cool_cast_is_still_corrected():
    """그늘·형광등처럼 화면 전체가 고르게 푸른 건 조명 탓 — 계속 교정한다."""
    img = _cast(_neutral_scene(), (0.85, 0.95, 1.1))
    s = measure_image_stats(img)
    assert s["warmth"] < -0.06 and s["cast_uniformity"] > 0.3   # 곱셈형 캐스트
    assert _params(img)["temperature"] >= 0.08


def test_already_warm_food_gets_less_recipe_warmth():
    food = {"subjectType": "음식"}
    neutral = _params(_neutral_scene(), analysis=food)["temperature"]
    warm = _params(_cast(_neutral_scene(), (1.15, 1.0, 0.72)), analysis=food)["temperature"]
    assert neutral > 0.05
    assert warm < 0.05


def _stack_analysis():
    return {
        "subjectType": "인물",
        "hslAdjust": {"orange": {"saturation": 0.3, "lightness": 0.1}},
        "regionParams": {"face": {"temperature": 0.15, "brightness": 0.05}},
    }


def test_no_double_warm_stacking_on_skin():
    strong = {"editingStyle": {"filterTendency": "strong"}}
    analysis = _stack_analysis()
    p, _ = build_params_with_comment(_neutral_scene(), {**strong, **_manual("golden_hour")}, analysis)
    assert p["temperature"] >= 0.15
    orange = p["hslAdjust"]["orange"]
    assert orange["saturation"] < 0.3 * 0.7      # 피부 채도 부스트를 덜어냄
    assert orange["lightness"] == 0.1             # 밝기는 겹침이 아니므로 그대로
    face = analysis["regionParams"]["face"]
    assert face["temperature"] <= max(0.0, 0.15 - p["temperature"]) + 1e-6
    assert face["brightness"] == 0.05


def test_cool_style_keeps_skin_boosts():
    """전역이 식히는 쪽이면 겹칠 웜톤이 없다 — 모델 값을 그대로 둔다."""
    strong = {"editingStyle": {"filterTendency": "strong"}}
    analysis = _stack_analysis()
    p, _ = build_params_with_comment(_neutral_scene(), {**strong, **_manual("flash_digicam")}, analysis)
    assert p["temperature"] < 0.05
    assert p["hslAdjust"]["orange"]["saturation"] == 0.3
    assert analysis["regionParams"]["face"]["temperature"] == 0.15
