"""파란 하늘이 회색으로 빠지던 문제의 회귀 테스트.

하늘이 넓은 야외 사진은 밝은 곳(하늘)과 그늘이 함께 파래서 '고른 파란 캐스트'로
잡혔고, 화이트밸런스 0.62 + 웜톤 0.144가 걸려 하늘 채도가 −43% 빠졌다(사용자 보고).
하늘색 화소는 캐스트 판단에서 빼고, 화이트밸런스·웜톤·채도 감소를 약하게만 건다.
"""

import cv2
import numpy as np
from PIL import Image

import image_processor as ip
from param_engine import _prune_region_params, measure_image_stats


def _sky_scene(h=240, w=320) -> Image.Image:
    """위 절반은 파란 하늘, 아래 절반은 중립에 가까운 땅."""
    rng = np.random.default_rng(0)
    arr = np.zeros((h, w, 3), np.float32)
    y = np.linspace(0, 1, h // 2)[:, None]
    arr[: h // 2] = np.stack([90 + 60 * y, 140 + 50 * y, 225 + 20 * y], -1)   # 하늘
    arr[h // 2:] = (124, 122, 118)                                           # 땅
    arr += rng.normal(0, 3, arr.shape)
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))


def _sky_chroma(img: Image.Image) -> float:
    lab = cv2.cvtColor(np.asarray(img.convert("RGB")).astype(np.float32) / 255.0, cv2.COLOR_RGB2LAB)
    top = lab[: lab.shape[0] // 2]
    return float(np.hypot(top[..., 1], top[..., 2]).mean())


def test_sky_is_detected_but_neutral_gray_is_not():
    w = ip.vivid_blue_weight(np.asarray(_sky_scene(), np.float32))
    assert w[:100].mean() > 0.6        # 하늘
    assert w[140:].mean() < 0.05       # 회색 땅
    cast = np.full((10, 10, 3), (108, 118, 138), np.float32)   # 형광등 같은 옅은 파란 캐스트
    assert ip.vivid_blue_weight(cast).max() < 0.2


def test_illuminant_estimate_is_not_pulled_by_the_sky():
    """하늘을 조명 색으로 보면 파랑을 지우려고 R을 올리고 B를 내린다(= 사진을 데운다).
    하늘을 빼면 추정은 땅만 보고, 땅이 이미 살짝 따뜻하니 오히려 B를 올린다."""
    r, _, b = ip.estimate_illuminant(_sky_scene())
    assert b >= r


def test_sky_does_not_turn_a_neutral_scene_into_a_blue_cast():
    s = measure_image_stats(_sky_scene())
    # 캐스트 방향은 땅 기준이라 파랑(음수) 캐스트로 읽히지 않는다
    assert not (s["cast_uniformity"] > 0.3 and s["warmth"] < -0.06)


def test_white_balance_keeps_the_sky_blue():
    img = _sky_scene()
    before = _sky_chroma(img)
    after = _sky_chroma(ip.apply_all_transforms(img, auto_wb=0.62, temperature=0.144))
    assert after > before * 0.8        # 예전: 약 −43%


def test_warming_and_desaturation_are_gentler_on_sky():
    img = _sky_scene()
    before = _sky_chroma(img)
    assert _sky_chroma(ip.apply_all_transforms(img, temperature=0.19)) > before * 0.9
    assert _sky_chroma(ip.apply_all_transforms(img, saturation=-0.1)) > before * 0.94


def test_model_sky_desaturation_is_dropped_but_brightness_kept():
    analysis = {"regionParams": {"sky": {"brightness": -0.08, "saturation": -0.1}}}
    _prune_region_params(analysis, {"brightness": 0.0, "saturation": 0.0}, False, 0.8)
    sky = analysis["regionParams"]["sky"]
    assert "saturation" not in sky and sky["brightness"] == -0.08
