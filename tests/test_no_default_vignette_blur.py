"""비네팅·배경 흐림은 스타일이 원할 때만.

예전에는 피사체 레시피마다 비네팅 0.03~0.18이 들어 있어 모든 사진의 모서리가
어두워졌고(실측 7장 전부 0.09~0.14, ΔE 2~3), 인물 사진에는 배경 흐림 0.25×게인이
무조건 걸려 스튜디오 배경·건물까지 가짜 보케로 뭉개졌다(인물 4장 전부 0.2).
"""

import numpy as np
import pytest
from PIL import Image

from param_engine import _TREND_RECIPES, build_params_with_comment


@pytest.fixture
def photo():
    rng = np.random.default_rng(3)
    h, w = 320, 240
    y = np.linspace(0, 1, h)[:, None, None]
    a = 60 + 140 * y + rng.normal(0, 8, (h, w, 3))
    a[80:200, 60:180] = (190, 140, 120)
    return Image.fromarray(a.clip(0, 255).astype(np.uint8))


def _params(img, profile, subject):
    p, _ = build_params_with_comment(img, profile, {"subjectType": subject})
    return p


@pytest.mark.parametrize("subject", ["인물", "음식", "카페/일상", "사물", "동물", "풍경", "혼합"])
def test_default_recipe_has_no_vignette(photo, subject):
    assert _params(photo, None, subject)["vignette"] == 0.0


def test_default_portrait_has_no_background_blur(photo):
    p = _params(photo, None, "인물")
    assert p["background_blur"] == 0.0
    assert p["vignette"] == 0.0


@pytest.mark.parametrize("trend", sorted(_TREND_RECIPES))
def test_no_trend_blurs_portrait_background_by_default(photo, trend):
    profile = {"trendCategory": trend, "styleSource": "manual",
               "editingStyle": {"filterTendency": "very_strong"}}
    assert _params(photo, profile, "인물")["background_blur"] == 0.0


@pytest.mark.parametrize("trend", ["warm_film", "korean_gamsung", "golden_hour"])
def test_film_trends_without_vignette_stay_clean(photo, trend):
    profile = {"trendCategory": trend, "styleSource": "manual"}
    assert _params(photo, profile, "음식")["vignette"] == 0.0


@pytest.mark.parametrize("trend", ["cinematic_moody", "flash_digicam", "bw_grain"])
@pytest.mark.parametrize("subject", ["인물", "음식", "풍경"])
def test_style_trends_still_vignette(photo, trend, subject):
    profile = {"trendCategory": trend, "styleSource": "manual"}
    assert _params(photo, profile, subject)["vignette"] >= 0.09


def test_cinematic_portrait_vignette_scales_with_gain(photo):
    base = {"trendCategory": "cinematic_moody", "styleSource": "manual"}
    lo = _params(photo, {**base, "editingStyle": {"filterTendency": "minimal"}}, "인물")
    hi = _params(photo, {**base, "editingStyle": {"filterTendency": "strong"}}, "인물")
    assert 0 < lo["vignette"] < hi["vignette"]


@pytest.mark.parametrize("pref,expected", [("subtle", 0.10), ("moderate", 0.20), ("strong", 0.30)])
def test_explicit_vignette_preference_is_respected(photo, pref, expected):
    profile = {"editingStyle": {"filterTendency": "moderate", "vignettePreference": pref}}
    for subject in ("인물", "음식"):
        v = _params(photo, profile, subject)["vignette"]
        assert v > 0 and v == pytest.approx(expected, rel=0.3)


def test_explicit_none_turns_off_trend_vignette(photo):
    profile = {"trendCategory": "cinematic_moody", "styleSource": "manual",
               "editingStyle": {"vignettePreference": "none"}}
    assert _params(photo, profile, "인물")["vignette"] == 0.0


def test_recipe_background_blur_hook_applies_only_to_portraits(photo, monkeypatch):
    monkeypatch.setitem(_TREND_RECIPES, "clean_minimal",
                        {**_TREND_RECIPES["clean_minimal"], "background_blur": 0.2})
    profile = {"trendCategory": "clean_minimal", "styleSource": "manual"}
    assert _params(photo, profile, "인물")["background_blur"] > 0
    assert _params(photo, profile, "풍경")["background_blur"] == 0.0
