"""옷·소품의 원색이 화이트밸런스·색온도를 끌고 가 피부가 식던 문제의 회귀 테스트.

실측: 코랄 재킷이 화면 1/3인 인물(원본 피부 a* 13.8 b* 15.4)이 warmth 0.59·
cast_uniformity 0.85로 읽혀, flash_digicam에서 auto_wb 0.77 + temperature −0.33이
걸리고 피부가 a* −3.5(청록), clean_minimal·bright_airy는 a* 0.4 b* 5(잿빛)가 됐다.
"""

import cv2
import numpy as np
from PIL import Image

import image_processor as ip
from param_engine import _guard_skin_cooling, build_params_with_comment, measure_image_stats

PORTRAIT = {"subjectType": "인물"}
TRENDS = ("warm_film", "korean_gamsung", "cinematic_moody", "golden_hour",
          "clean_minimal", "bright_airy", "flash_digicam", "soft_pastel")


def _manual(trend):
    return {"trendCategory": trend, "styleSource": "manual"}


def _coral_portrait() -> Image.Image:
    """회색 배경 + 화면 아래 절반을 채운 코랄 재킷 + 가운데 얼굴(피부)."""
    rng = np.random.default_rng(3)
    h, w = 240, 200
    y = np.linspace(0.35, 0.8, h)[:, None, None]
    arr = np.broadcast_to(y, (h, w, 3)) * 255.0
    arr = arr + rng.normal(0, 3, arr.shape)
    arr[130:] = (238, 112, 88)              # 코랄 재킷
    arr[40:130, 60:140] = (214, 160, 134)   # 얼굴
    arr[40:60, 60:140] = (150, 146, 140)    # 회색 머리
    arr += rng.normal(0, 3, arr.shape)
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))


def _skin_ab(img: Image.Image) -> tuple[float, float]:
    lab = cv2.cvtColor(np.asarray(img.convert("RGB")), cv2.COLOR_RGB2LAB).astype(np.float32)
    face = lab[70:125, 70:130]
    return float(face[..., 1].mean() - 128), float(face[..., 2].mean() - 128)


def _render(img, p):
    return ip.apply_all_transforms(img, temperature=p["temperature"], auto_wb=p["auto_wb"])


def test_cast_weight_skips_clothing_and_skin_but_keeps_casted_gray():
    px = np.array([[[238, 112, 88], [214, 160, 134], [128, 128, 128],
                    [147, 128, 92], [109, 122, 141]]] * 40, np.float32)   # 코랄·피부·회색·웜/쿨 캐스트 회색
    w = ip.cast_pixel_weight(px)[0]
    assert w[0] < 0.1 and w[1] < 0.1
    assert w[2] > 0.95 and w[3] > 0.95 and w[4] > 0.95


def test_clothing_does_not_read_as_a_warm_cast():
    s = measure_image_stats(_coral_portrait())
    assert abs(s["warmth"]) < 0.08
    r, _, b = ip.estimate_illuminant(_coral_portrait())
    assert abs(r - b) < 0.06           # 예전: R 0.81 / B 1.15 — 강하게 식혔다


def test_clothing_dominated_portrait_keeps_skin_warm_in_every_style():
    img = _coral_portrait()
    a0, b0 = _skin_ab(img)
    for trend in (None, *TRENDS):
        p = build_params_with_comment(img, _manual(trend) if trend else None, dict(PORTRAIT))[0]
        a1, b1 = _skin_ab(_render(img, p))
        assert a1 >= 0.7 * a0 and b1 >= 0.55 * b0, (trend, p["temperature"], p["auto_wb"], a1, b1)


def test_flash_is_still_cooler_than_default():
    for img in (_coral_portrait(), Image.fromarray(np.full((120, 160, 3), 128, np.uint8))):
        default = build_params_with_comment(img, None, dict(PORTRAIT))[0]["temperature"]
        flash = build_params_with_comment(img, _manual("flash_digicam"), dict(PORTRAIT))[0]["temperature"]
        assert flash < default - 0.1
        assert flash < 0


def test_skin_guard_scales_back_heavy_cooling():
    """실제로 따뜻한 조명이라도 WB + 식히기가 피부를 청록으로 만들면 덜어낸다."""
    img = Image.fromarray(np.clip(np.asarray(_coral_portrait(), np.float32)
                                  * (1.12, 1.0, 0.78), 0, 255).astype(np.uint8))
    a0, b0 = _skin_ab(img)
    heavy = {"auto_wb": 1.0, "temperature": -0.6}
    assert _skin_ab(_render(img, heavy))[0] < 0.7 * a0     # 가드가 없으면 피부 붉음이 크게 빠진다
    p = dict(heavy)
    _guard_skin_cooling(p, img)
    assert 0.0 <= p["auto_wb"] < 1.0 and -0.6 < p["temperature"] <= 0.0
    a1, b1 = _skin_ab(_render(img, p))
    assert a1 >= 0.65 * a0 and b1 >= 0.5 * b0


def test_skin_guard_leaves_gentle_or_warming_params_alone():
    img = _coral_portrait()
    for params in ({"auto_wb": 0.2, "temperature": -0.05}, {"auto_wb": 0.0, "temperature": 0.3}):
        p = dict(params)
        _guard_skin_cooling(p, img)
        assert p == params
