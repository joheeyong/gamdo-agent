"""그레인은 인스타그램 피드 크기(가로 1080px)에서 곱게 보여야 한다.

실측(리뷰): 예전 그레인은 800px 기준 알갱이를 늘려 붙여, 2560px 저장본을
1080px로 줄여도 평탄 배경 고주파 편차가 그대로였다 (기본 4.3, cinematic 6.7,
bw 7.1 레벨 / 원본 0.4). 맑은 하늘과 피부가 지저분해 보였다.
"""

import cv2
import numpy as np
from PIL import Image

from image_processor import apply_grain
from param_engine import _DEFAULT_RECIPE, _TREND_RECIPES

_DEFAULT_EFF = _DEFAULT_RECIPE["grain"] * 0.8     # 기본 게인 0.8에서 실제로 걸리는 값
_BW_EFF = _TREND_RECIPES["bw_grain"]["grain"] * 0.8


def _solid(rgb, size=(2560, 2048)) -> Image.Image:
    w, h = size
    return Image.fromarray(np.full((h, w, 3), rgb, np.uint8))


def _feed(img: Image.Image) -> Image.Image:
    img = img.copy()
    img.thumbnail((1080, 1440), Image.LANCZOS)
    return img


def _hf(img: Image.Image, box=(0.1, 0.1, 0.9, 0.9)) -> float:
    """밝기 고주파 편차 (가우시안 σ2 하이패스)."""
    g = np.asarray(img.convert("L"), np.float32)
    h, w = g.shape
    r = g[int(box[1] * h):int(box[3] * h), int(box[0] * w):int(box[2] * w)]
    return float((r - cv2.GaussianBlur(r, (0, 0), 2)).std())


def test_default_grain_is_subtle_at_feed_size():
    out = apply_grain(_solid((128, 128, 128)), _DEFAULT_EFF)
    feed = _hf(_feed(out))
    # 보이되 곱게 — 예전 값은 약 5 레벨이었다
    assert 1.2 <= feed <= 2.6, feed
    # 100%로 확대하면 알갱이가 더 또렷하다 (피드에서 평균되는 고운 알갱이)
    assert _hf(out) > feed * 1.2


def test_preview_and_saved_render_match_at_display_size():
    saved = _hf(_feed(apply_grain(_solid((128, 128, 128)), _DEFAULT_EFF)))
    preview = _hf(apply_grain(_solid((128, 128, 128), (800, 640)), _DEFAULT_EFF))
    assert abs(preview - saved) / saved < 0.3, (preview, saved)


def test_grain_strongest_in_midtones():
    mid = _hf(_feed(apply_grain(_solid((128, 128, 128)), 0.3)))
    hi = _hf(_feed(apply_grain(_solid((235, 235, 235)), 0.3)))
    lo = _hf(_feed(apply_grain(_solid((20, 20, 20)), 0.3)))
    assert mid > 1.8 * hi and mid > 1.8 * lo, (mid, hi, lo)


def test_skin_gets_less_grain_than_neutral():
    skin = _solid((212, 164, 140))     # 밝은 피부
    gray_level = int(np.asarray(skin.convert("L"))[0, 0])
    gray = _solid((gray_level,) * 3)
    s = _hf(_feed(apply_grain(skin, 0.2)))
    g = _hf(_feed(apply_grain(gray, 0.2)))
    assert s < 0.65 * g, (s, g)


def test_smooth_sky_gets_less_grain_than_neutral():
    w, h = 2560, 2048
    t = np.linspace(0, 1, h, dtype=np.float32)[:, None, None]
    top, bottom = np.array([95, 140, 205], np.float32), np.array([150, 185, 230], np.float32)
    sky = Image.fromarray((top + (bottom - top) * t).repeat(w, axis=1).astype(np.uint8))
    gray = Image.fromarray(np.repeat(np.asarray(sky.convert("L"))[:, :, None], 3, axis=2))
    box = (0.1, 0.05, 0.9, 0.35)      # 프레임 위쪽
    s = _hf(_feed(apply_grain(sky, 0.2)), box)
    g = _hf(_feed(apply_grain(gray, 0.2)), box)
    assert s < 0.7 * g, (s, g)


def test_bw_grain_stays_clearly_grainy():
    gray = _solid((128, 128, 128))
    bw = _hf(_feed(apply_grain(gray, _BW_EFF)))
    default = _hf(_feed(apply_grain(gray, _DEFAULT_EFF)))
    assert bw >= 3.5 and bw >= 2.0 * default, (bw, default)
    # 흑백 사진은 피부·하늘 감쇠를 받지 않는다 (색으로 가를 수 없고 그레인이 스타일)
    mono_skin = _solid((170, 170, 170))
    assert _hf(_feed(apply_grain(mono_skin, _BW_EFF))) > 2.5


def test_style_grain_order_keeps_character():
    g = {k: v.get("grain", 0.0) for k, v in _TREND_RECIPES.items()}
    assert g["bw_grain"] > g["cinematic_moody"] > g["warm_film"] > _DEFAULT_RECIPE["grain"]
    assert g["flash_digicam"] > _DEFAULT_RECIPE["grain"]


def test_grain_is_deterministic():
    img = _solid((120, 110, 100), (1200, 900))
    a = np.asarray(apply_grain(img, 0.2))
    b = np.asarray(apply_grain(img, 0.2))
    assert np.array_equal(a, b)
