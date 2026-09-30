"""노이즈 추정·노이즈 제거·그레인의 상호작용.

실측: 스튜디오 조명의 깨끗한 케이크 사진에서 코코아 가루 질감이 노이즈 12.2로
읽혀 denoise 0.83이 걸렸고(질감 뭉개짐), 그 위에 레시피 그레인 0.13이 다시
뿌려졌다. 질감은 노이즈가 아니고, 지운 만큼 그레인은 덜어야 한다.
"""

import numpy as np
from PIL import Image

from image_processor import estimate_noise_sigma
from param_engine import _DEFAULT_RECIPE, build_params_with_comment


def _textured_clean(size=512, texture_frac=0.6, seed=0) -> Image.Image:
    """가운데가 고운 질감(가루·스펀지 비슷한 고주파), 가장자리는 매끈한 밝은 접시. 노이즈 없음."""
    rng = np.random.default_rng(seed)
    y = np.linspace(0, 1, size)[:, None]
    arr = np.repeat((200 + 25 * y) * np.ones((1, size)), 1, axis=1)
    side = int(size * np.sqrt(texture_frac))
    o = (size - side) // 2
    tex = 90 + 70 * (rng.random((side, side)) > 0.5)   # 화소 단위 알갱이
    arr[o:o + side, o:o + side] = tex
    arr = np.clip(arr, 0, 255).astype(np.uint8)
    return Image.fromarray(np.dstack([arr, (arr * 0.85).astype(np.uint8), (arr * 0.7).astype(np.uint8)]))


def _flat_noisy(level: float, sigma: float, size=512, seed=0) -> Image.Image:
    rng = np.random.default_rng(seed)
    y = np.linspace(-8, 8, size)[:, None] * np.ones((1, size))
    base = level + y
    arr = base[:, :, None] + rng.normal(0, sigma, (size, size, 1))
    return Image.fromarray(np.clip(np.repeat(arr, 3, axis=2), 0, 255).astype(np.uint8))


def test_texture_is_not_mistaken_for_noise():
    assert estimate_noise_sigma(_textured_clean()) < 1.0


def test_real_noise_is_still_measured():
    for sigma in (3.0, 6.0, 10.0):
        est = estimate_noise_sigma(_flat_noisy(128, sigma))
        assert abs(est - sigma) / sigma < 0.15, (sigma, est)


def test_clean_textured_photo_gets_no_denoise_and_full_grain():
    params, _ = build_params_with_comment(_textured_clean(), None, {"subjectType": "음식"})
    assert params["denoise"] == 0.0
    assert params["grain"] > 0.0


def test_bright_noisy_photo_denoise_is_capped():
    params, _ = build_params_with_comment(_flat_noisy(180, 12.0), None, {"subjectType": "혼합"})
    assert 0.0 < params["denoise"] <= 0.5


def test_heavy_denoise_reduces_recipe_grain():
    clean, _ = build_params_with_comment(_flat_noisy(180, 0.0), None, {"subjectType": "혼합"})
    noisy_dark, _ = build_params_with_comment(_flat_noisy(50, 8.0), None, {"subjectType": "혼합"})
    assert clean["denoise"] == 0.0
    assert noisy_dark["denoise"] >= 0.6          # 어두운 사진은 상한 없이 잡는다
    assert noisy_dark["grain"] < clean["grain"] * 0.7
    assert _DEFAULT_RECIPE["grain"] > 0.0


def test_explicit_grain_preference_is_not_traded_away():
    profile = {"editingStyle": {"grainPreference": "moderate"}}
    clean, _ = build_params_with_comment(_flat_noisy(180, 0.0), profile, {"subjectType": "혼합"})
    noisy, _ = build_params_with_comment(_flat_noisy(50, 8.0), profile, {"subjectType": "혼합"})
    assert noisy["denoise"] > 0.3
    assert noisy["grain"] == clean["grain"]
