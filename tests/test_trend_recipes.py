"""2026 트렌드 레시피 — 기본 레시피(소프트 필름 내추럴)와 신규 트렌드 3종, 3:4 비율.

기본 레시피는 스타일 프로필이 없는 사용자 전원이 탄다. 예전에는 S커브 + 떠 있는
검정 누르기가 기본이라 모든 사진이 쨍하고 무거워졌다. 이제는 바랜 검정·순한 대비·
고운 그레인이 기본이고, 검정은 사진이 실제로 뿌옇고 평평할 때만 누른다.
"""

import re

import cv2
import numpy as np
import pytest
from PIL import Image

import prompts
from image_processor import (
    analysis_to_transform_params,
    apply_all_transforms,
    apply_auto_edits,
    apply_instagram_ratio,
)
from param_engine import (
    _DEFAULT_RECIPE,
    _TREND_LABELS,
    _TREND_RECIPES,
    build_params_with_comment,
    haze_flatness,
    measure_image_stats,
    normalize_style_profile,
)


def _scene(seed: int = 0, size=(240, 320)) -> Image.Image:
    """부드러운 그라데이션 + 색 블록 + 약간의 텍스처. 대비는 보통 (p5≈0.15, p5~p95≈0.7)."""
    rng = np.random.default_rng(seed)
    h, w = size
    y = np.linspace(0, 1, h)[:, None]
    x = np.linspace(0, 1, w)[None, :]
    base = 0.12 + 0.95 * (0.6 * y + 0.4 * x)
    arr = np.stack([base * 0.95 + 0.05, base * 0.85 + 0.04, base * 0.75 + 0.03], -1)
    arr[40:120, 30:130] = (0.78, 0.55, 0.45)    # 피부 비슷한 블록
    arr[140:220, 180:300] = (0.25, 0.45, 0.70)  # 파랑 블록
    arr[150:200, 20:90] = (0.16, 0.15, 0.15)    # 어두운 블록
    arr += rng.normal(0, 0.02, arr.shape)
    return Image.fromarray((np.clip(arr, 0, 1) * 255).astype(np.uint8))


def _hazy(img: Image.Image) -> Image.Image:
    """회색 베일을 씌운 뿌연 사진 — 바닥이 뜨고 대비가 눌린다."""
    arr = np.asarray(img, np.float32)
    return Image.fromarray(np.clip(arr * 0.5 + 0.42 * 255, 0, 255).astype(np.uint8))


def _render(img, profile, subject="혼합"):
    params, comment = build_params_with_comment(img, profile, {"subjectType": subject})
    out = apply_all_transforms(img, **analysis_to_transform_params({"recommendedParams": params}))
    return params, comment, out


def _manual(trend):
    return {"trendCategory": trend, "styleSource": "manual"}


# ── 기본 레시피: 소프트 필름 내추럴 ──


def test_default_recipe_is_soft_film():
    assert _DEFAULT_RECIPE["tone_curve"][0] == "soft_film"
    assert _DEFAULT_RECIPE["crush_blacks"] == "hazy"
    params, comment, _ = _render(_scene(), None)
    assert params["toneCurve"]["preset"] == "soft_film"
    # 유효 그레인 0.10~0.15 (게인 0.8 기준)
    assert 0.10 <= params["grain"] <= 0.15
    assert params["clarity"] <= 0.05 and params["sharpness"] <= 0.12
    assert "S커브" not in comment and "떠 있는 검정" not in comment


def test_default_does_not_crush_blacks_on_normal_contrast_image():
    img = _scene()
    stats = measure_image_stats(img)
    assert stats["shadow_p05"] > _DEFAULT_RECIPE["shadow_floor"]  # 바닥이 목표보다 떠 있다
    assert haze_flatness(stats) < 0.2
    params, _, _ = _render(img, None)
    assert params["shadows"] > -0.05


def test_default_crushes_blacks_on_hazy_image():
    img = _hazy(_scene())
    assert haze_flatness(measure_image_stats(img)) > 0.5
    params, comment, _ = _render(img, None)
    assert params["shadows"] <= -0.10
    assert "뿌옇게" in comment


def test_food_stays_a_bit_crisper_but_not_hdr():
    food, _, _ = _render(_scene(), None, "음식")
    person, _, _ = _render(_scene(), None, "카페/일상")
    assert food["clarity"] > person["clarity"]
    assert food["clarity"] <= 0.12 and food["sharpness"] <= 0.15   # 예전: 0.20 / 0.18


def test_portrait_default_uses_trend_curve_not_s_curve():
    params, _, _ = _render(_scene(), None, "인물")
    assert params["toneCurve"]["preset"] == "soft_film"


# ── 신규 트렌드 ──


def test_new_trends_are_registered_with_labels():
    for trend in ("flash_digicam", "soft_pastel", "bw_grain"):
        assert trend in _TREND_RECIPES and trend in _TREND_LABELS


def test_bw_grain_is_true_monochrome():
    params, comment, out = _render(_scene(), _manual("bw_grain"))
    assert params["saturation"] == -1.0
    hsv = cv2.cvtColor(np.asarray(out), cv2.COLOR_RGB2HSV)
    # 그레인은 휘도 노이즈라 색을 만들지 않는다 — 채도가 사실상 0
    assert hsv[..., 1].mean() < 1.5
    arr = np.asarray(out).astype(int)
    assert np.abs(arr[..., 0] - arr[..., 2]).max() <= 3
    assert params["grain"] >= 0.2
    assert "흑백" in comment


def test_bw_grain_keeps_skin_brighter_than_plain_gray():
    """따뜻한 색(피부)은 단순 L 추출보다 밝게 옮긴다."""
    img = _scene()
    _, _, out = _render(img, _manual("bw_grain"))
    lab = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2LAB).astype(float)
    skin_l_src = lab[40:120, 30:130, 0].mean()
    blue_l_src = lab[140:220, 180:300, 0].mean()
    out_l = cv2.cvtColor(np.asarray(out), cv2.COLOR_RGB2LAB).astype(float)[..., 0]
    # 파랑 대비 피부의 상대 밝기가 원본보다 커진다
    assert (out_l[40:120, 30:130].mean() - out_l[140:220, 180:300].mean()) > (skin_l_src - blue_l_src)


def test_soft_pastel_shifts_toward_pink():
    img = _scene()
    _, _, base = _render(img, None)
    params, _, pastel = _render(img, _manual("soft_pastel"))
    assert params["splitToning"]["shadow"]["strength"] > 0.2
    assert params["splitToning"]["highlight"]["strength"] > 0.2

    def a_mean(im, mask_fn):
        lab = cv2.cvtColor(np.asarray(im), cv2.COLOR_RGB2LAB).astype(float)
        m = mask_fn(lab[..., 0])
        return lab[..., 1][m].mean()

    lab0 = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2LAB)[..., 0]
    hi = lambda _l: lab0 > 170          # noqa: E731 — 원본 기준 같은 화소를 비교
    lo = lambda _l: lab0 < 90           # noqa: E731
    assert a_mean(pastel, hi) > a_mean(base, hi) + 1.5   # 하이라이트가 핑크(+a)로
    assert a_mean(pastel, lo) > a_mean(base, lo) + 1.5   # 쉐도우도 핑크로
    # 낮은 대비
    assert params["contrast"] < 0


def test_flash_digicam_is_cooler_and_punchier():
    img = _scene()
    base_p, _, base = _render(img, None)
    params, _, flash = _render(img, _manual("flash_digicam"))
    assert params["temperature"] < base_p["temperature"] and params["temperature"] < 0
    assert params["clarity"] > base_p["clarity"]
    assert params["vignette"] > 0.1
    b_base = cv2.cvtColor(np.asarray(base), cv2.COLOR_RGB2LAB)[..., 2].astype(float).mean()
    b_flash = cv2.cvtColor(np.asarray(flash), cv2.COLOR_RGB2LAB)[..., 2].astype(float).mean()
    assert b_flash < b_base   # 더 차가운(파란) 화이트밸런스

    def local_contrast(im):
        g = np.asarray(im.convert("L"), np.float32)
        return float(np.abs(g - cv2.GaussianBlur(g, (0, 0), 6)).mean())

    assert local_contrast(flash) > local_contrast(base)


# ── 수동 스타일 프로필 ──


@pytest.mark.parametrize("trend", sorted(_TREND_RECIPES))
def test_manual_only_profile_works_for_every_trend(trend):
    profile = _manual(trend)
    norm = normalize_style_profile(profile)
    assert norm["trendCategory"] == trend and norm["styleSource"] == "manual"
    params, comment, out = _render(_scene(), profile, "인물")
    assert out.size == (320, 240)
    assert comment.startswith(("흑백", _TREND_LABELS[trend])) or _TREND_LABELS[trend] in comment


def test_unknown_trend_falls_back_to_default():
    a, _, _ = _render(_scene(), _manual("y3k_neon"))
    b, _, _ = _render(_scene(), None)
    assert a == b


def test_manual_trend_ignores_feed_reference():
    img = _scene()
    ref = {"brightness": 0.7, "contrast": 0.9, "saturation": 0.7, "warmth": -0.3,
           "luma_percentiles": [0.0, 0.0, 0.01, 0.2, 0.5, 0.8, 0.95, 1.0, 1.0]}
    manual, _ = build_params_with_comment(img, _manual("soft_pastel"), {"subjectType": "혼합"}, reference=ref)
    plain, _ = build_params_with_comment(img, _manual("soft_pastel"), {"subjectType": "혼합"})
    assert manual == plain


# ── 3:4 비율 ──


def test_three_four_crop_dimensions():
    wide = Image.new("RGB", (1600, 1200))
    assert apply_instagram_ratio(wide, "3:4").size == (900, 1200)
    tall = Image.new("RGB", (1080, 1920))
    assert apply_instagram_ratio(tall, "3:4").size == (1080, 1440)
    assert apply_instagram_ratio(Image.new("RGB", (1080, 1440)), "3:4").size == (1080, 1440)
    # 기존 값도 그대로
    assert apply_instagram_ratio(wide, "4:5").size == (960, 1200)
    assert apply_instagram_ratio(wide, "1:1").size == (1200, 1200)


def test_three_four_respects_portrait_vertical_guard():
    tall = Image.new("RGB", (1080, 1920))
    assert apply_instagram_ratio(tall, "3:4", allow_vertical_crop=False).size == (1080, 1920)
    out = apply_auto_edits(tall, {"instagram_ratio": "3:4", "allow_vertical_crop": False})
    assert out.size == (1080, 1920)
    out = apply_auto_edits(tall, {"instagram_ratio": "3:4", "allow_vertical_crop": True})
    assert out.size == (1080, 1440)


def test_transform_prompt_offers_three_four():
    assert '"3:4" | "4:5" | "1:1" | null' in prompts.TRANSFORM_PHOTO_PROMPT


# ── 프롬프트 ──


def test_prompts_list_new_trend_categories():
    for trend in ("flash_digicam", "soft_pastel", "bw_grain"):
        assert trend in prompts.ANALYZE_USER_PROMPT
        assert trend in prompts.ANALYZE_USER_PROMPT.split("trendCategory 설명:")[1]


def test_prompts_have_no_unsourced_engagement_statistics():
    for name in ("ANALYZE_USER_SYSTEM", "ANALYZE_USER_PROMPT", "TRANSFORM_PHOTO_SYSTEM",
                 "TRANSFORM_PHOTO_PROMPT"):
        text = getattr(prompts, name)
        assert not re.search(r"\d+\s*%\s*[↑↓]", text), name
        for stat in ("저장률 34", "공유 27", "좋아요 24", "댓글 46", "조회수 21", "댓글 45"):
            assert stat not in text, (name, stat)
