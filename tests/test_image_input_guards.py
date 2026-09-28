"""클라이언트가 보낸 값·이미지로부터 파이프라인을 지키는 가드들.

- 해상도: 압축 폭탄 거절, 앱 계약(짧은 변 2560)을 넘는 입력 축소
- 스마트 크롭 하한: x,y를 안쪽으로 당기지 않으면 하한이 무력해졌다
- NaN/Infinity: min/max가 NaN을 통과시켜 최대 보정이 됐다
- 톤 커브 제어점·HSL 정리: apply-transform과 분석 경로가 같은 정리를 쓴다
"""

import base64
import io

import numpy as np
import pytest
from PIL import Image

import image_processor as ip
from image_processor import (
    analysis_to_transform_params,
    apply_regional_transforms,
    apply_smart_crop,
    build_local_regions,
    decode_base64_image,
    sanitize_hsl_adjust,
    sanitize_tone_curve_points,
)

NAN = float("nan")
INF = float("inf")


def _b64(img: Image.Image, fmt: str = "PNG") -> str:
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    return base64.b64encode(buf.getvalue()).decode()


# ── 해상도 가드 ──


def test_decode_rejects_decompression_bomb():
    # 1비트 PNG는 수 KB로 64MP까지 풀린다
    bomb = Image.new("1", (8000, 8000))
    with pytest.raises(ValueError):
        decode_base64_image(_b64(bomb))


def test_decode_keeps_app_sized_photo():
    # 앱이 보내는 4:3 사진 (짧은 변 2560) — 손대지 않는다
    img = decode_base64_image(_b64(Image.new("RGB", (3413, 2560), (10, 20, 30)), "JPEG"))
    assert img.size == (3413, 2560)
    assert img.mode == "RGB"


def test_decode_downscales_short_edge_over_contract():
    img = decode_base64_image(_b64(Image.new("RGB", (3000, 4000)), "JPEG"))
    assert min(img.size) == 2560
    assert img.size == (2560, 3413)


def test_decode_downscales_huge_panorama(monkeypatch):
    monkeypatch.setattr(ip, "_MAX_PROCESS_PIXELS", 1_000_000)
    img = decode_base64_image(_b64(Image.new("RGB", (4000, 1000))))
    w, h = img.size
    assert w * h <= 1_000_000
    assert abs(w / h - 4.0) < 0.01


# ── 스마트 크롭 하한 ──


def _gradient(size=1000):
    xs = np.linspace(0, 255, size, dtype=np.float32)
    arr = np.stack([np.tile(xs, (size, 1)), np.tile(xs[:, None], (1, size)),
                    np.full((size, size), 128, np.float32)], axis=2)
    return Image.fromarray(arr.astype(np.uint8))


def test_smart_crop_min_side_not_bypassed_at_edge():
    img = _gradient()
    out = apply_smart_crop(img, {"x": 0.9, "y": 0.9, "width": 0.1, "height": 0.1})
    assert out.size == (300, 300)
    # 오른쪽 아래 끝에 붙은 300x300이어야 한다
    assert np.array_equal(np.array(out), np.array(img)[700:1000, 700:1000])


def test_smart_crop_inside_box_unchanged():
    img = _gradient()
    out = apply_smart_crop(img, {"x": 0.1, "y": 0.2, "width": 0.5, "height": 0.4})
    assert np.array_equal(np.array(out), np.array(img)[200:600, 100:600])


# ── NaN/Infinity ──


def test_analysis_params_treat_non_finite_as_missing():
    analysis = {"recommendedParams": {
        "brightness": NAN, "contrast": INF, "saturation": -INF, "denoise": NAN,
        "toneCurve": {"preset": "film", "strength": NAN, "points": [[0, 0], [NAN, 1]]},
        "splitToning": {"shadow": {"hue": NAN, "strength": INF},
                        "highlight": {"hue": INF, "strength": NAN}},
        "hslAdjust": {"green": {"hue": NAN, "saturation": INF, "lightness": 0.2}},
        "reshapeParams": {"face_slim": NAN, "shoulder_width": INF},
    }}
    p = analysis_to_transform_params(analysis)
    for key in ("brightness", "contrast", "saturation", "denoise",
                "tone_curve_strength", "split_shadow_hue", "split_shadow_strength",
                "split_highlight_hue", "split_highlight_strength",
                "face_slim", "shoulder_width"):
        assert p[key] == 0.0, key
    assert p["tone_curve_points"] is None
    assert p["hsl_adjust"] == {"green": {"hue": 0.0, "saturation": 0.0, "lightness": 0.2}}


def test_analysis_params_normal_values_unchanged():
    analysis = {"recommendedParams": {
        "brightness": 0.2, "contrast": 3, "denoise": -1,
        "toneCurve": {"preset": "film", "strength": 0.5, "points": [[1, 1], [0, 0.1]]},
        "splitToning": {"shadow": {"hue": 400, "strength": 0.3}},
        "hslAdjust": {"blue": {"saturation": 0.4}, "nope": {"hue": 1}},
    }}
    p = analysis_to_transform_params(analysis)
    assert p["brightness"] == 0.2
    assert p["contrast"] == 1.0
    assert p["denoise"] == 0.0
    assert p["tone_curve_points"] == [(0.0, 0.1), (1.0, 1.0)]
    assert p["split_shadow_hue"] == 40.0
    assert p["hsl_adjust"] == {"blue": {"hue": 0.0, "saturation": 0.4, "lightness": 0.0}}


def test_regional_nan_value_is_skipped():
    img = Image.new("RGB", (64, 64), (120, 120, 120))
    mask = np.full((64, 64), 255, np.uint8)
    out = apply_regional_transforms(img, {"sky": mask}, {"sky": {"brightness": NAN}})
    assert out is img


def test_local_region_nan_area_is_dropped():
    regions = build_local_regions((100, 100), {
        "local_0": {"area": {"x": NAN, "y": 0.1, "width": 0.3, "height": 0.3},
                    "brightness": 0.2},
    })
    assert regions == {}


# ── 톤 커브·HSL 정리 (apply-transform 경로) ──


def test_sanitize_tone_curve_points():
    assert sanitize_tone_curve_points([[1, 1], [0, 0], [0.5, 0.6]]) == [
        (0.0, 0.0), (0.5, 0.6), (1.0, 1.0)]
    assert sanitize_tone_curve_points([["a", 0], [1, 1]]) is None
    assert sanitize_tone_curve_points([[0, 0]]) is None
    assert sanitize_tone_curve_points([[0, 0], [1, INF]]) is None
    assert sanitize_tone_curve_points("0,0,1,1") is None
    assert sanitize_tone_curve_points(None) is None


def test_sanitize_hsl_adjust():
    assert sanitize_hsl_adjust({"red": {"hue": "x", "saturation": 2}}) == {
        "red": {"hue": 0.0, "saturation": 1.0, "lightness": 0.0}}
    assert sanitize_hsl_adjust({"red": "bad", "ultraviolet": {"hue": 0.5}}) is None
    assert sanitize_hsl_adjust("red") is None
