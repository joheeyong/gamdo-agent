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


# ── 하늘 보호는 '실제 하늘'에만 (색 + 위치) ──
# 색만 보면 파란 셔츠·간판도 하늘로 잡혀, 채도를 −0.95로 내리면 파란 옷만 색이 남는
# 부분 컬러가 됐다(파랑 C 66→22, 빨강·초록 3.6). 강한 탈색에서는 하늘 보호도 0으로.

def _sky_and_patches() -> Image.Image:
    """위쪽 띠는 하늘, 가운데에 선명한 파란 옷 조각과 빨간 조각."""
    a = np.full((200, 300, 3), 120, np.uint8)
    a[:50] = (110, 160, 235)
    a[120:170, 40:110] = (40, 110, 230)
    a[120:170, 190:260] = (220, 40, 40)
    return Image.fromarray(a)


def _chroma(img: Image.Image) -> np.ndarray:
    lab = cv2.cvtColor(np.asarray(img.convert("RGB")).astype(np.float32) / 255.0, cv2.COLOR_RGB2LAB)
    return np.hypot(lab[..., 1], lab[..., 2])


def _patch_chroma(img):
    c = _chroma(img)
    return c[:45].mean(), c[125:165, 45:105].mean(), c[125:165, 195:255].mean()


def test_protect_weight_covers_sky_but_not_a_blue_garment():
    lab = cv2.cvtColor(np.asarray(_sky_and_patches()), cv2.COLOR_RGB2LAB).astype(np.float32)
    w = ip.sky_protect_weight(lab[..., 0], lab[..., 2])
    assert w[5:40].mean() > 0.4                  # 하늘
    assert w[125:165, 45:105].max() < 0.05       # 가운데 파란 옷
    # 추정 쪽 가중치(색만)는 파란 옷도 '장면의 색'으로 본다 — 일부러 그대로 둔다
    assert ip.vivid_blue_weight(np.asarray(_sky_and_patches(), np.float32))[125:165, 45:105].mean() > 0.5


def test_mild_desaturation_protects_sky_only():
    src = _sky_and_patches()
    s0, b0, r0 = _patch_chroma(src)
    s1, b1, r1 = _patch_chroma(ip.apply_all_transforms(src, saturation=-0.3))
    assert s1 / s0 > 0.8                         # 하늘은 덜 뺀다
    assert abs(b1 / b0 - r1 / r0) < 0.03         # 파란 옷은 빨강과 똑같이 빠진다


def test_strong_desaturation_has_no_partial_color_or_cliff():
    src = _sky_and_patches()
    s0, b0, r0 = _patch_chroma(src)
    for sat in (-0.95, -0.98):
        s, b, r = _patch_chroma(ip.apply_all_transforms(src, saturation=sat))
        assert s / s0 < 0.12 and b / b0 < 0.12 and r / r0 < 0.12, sat
    # −0.98 → 흑백 사이에 남은 색이 거의 없다 (예전: 하늘·파랑만 C 26~38이 남았다가 0)
    assert max(_patch_chroma(ip.apply_all_transforms(src, saturation=-0.98))) < 2.5
    assert _chroma(ip.apply_all_transforms(src, saturation=-1.0)).max() < 1.0


def test_white_balance_does_not_protect_a_blue_garment():
    """파란 옷만 있는 사진에서 WB 게인은 옷에도 그대로 걸린다(하늘 보호 없음)."""
    a = np.full((200, 300, 3), (150, 130, 110), np.uint8)   # 따뜻한 캐스트
    a[80:180, 100:200] = (40, 110, 230)
    img = Image.fromarray(a)
    gains = ip.estimate_illuminant(img)
    out = np.asarray(ip.apply_auto_white_balance(img, 1.0), np.float32)
    g = [max(0.75, min(1.25, x)) for x in gains]
    expect = np.clip(np.rint(np.array([40, 110, 230], np.float32) * g), 0, 255)
    assert np.abs(out[130, 150] - expect).max() <= 1.0
