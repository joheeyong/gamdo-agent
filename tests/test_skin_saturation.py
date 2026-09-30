"""인물 피부가 잿빛이 되던 문제의 회귀 테스트.

기본 레시피에서 인물 사진은 전부 saturation −0.184(바닥 −0.15 + 레시피 −0.08, 게인 0.8)가
걸렸다. 채도 측정(HSV 평균)이 옷·배경의 선명한 색에 끌려 올라가 "과한 채도"로 읽혔고,
그 값이 피부에도 똑같이 걸려 피부 채도가 13~27% 빠졌다 (c2·meir 피부가 잿빛).
  - 측정: 인물이면 측정분의 채도 낮추기는 절반만 둔다 (레퍼런스가 있으면 그대로).
  - 렌더: 음수 saturation은 피부색 화소에 절반만 건다 (흑백은 영향 없음).
"""

import cv2
import numpy as np
from PIL import Image

import image_processor as ip
from param_engine import build_params_with_comment

_SKIN_RGB = (214, 160, 130)      # 밝은 피부 (LAB 색상각 ~55°, C* ~25)
_DEEP_SKIN_RGB = (150, 100, 75)  # 짙은 피부
_BLUE_RGB = (30, 60, 200)        # 선명한 파란 옷
_GREEN_RGB = (60, 150, 70)       # 풀밭


def _portrait(h=240, w=240) -> Image.Image:
    """가운데 피부, 주변은 선명한 파란 옷 — 채도 측정이 높게 나오는 인물 사진."""
    rng = np.random.default_rng(0)
    arr = np.zeros((h, w, 3), np.float32)
    arr[:] = _BLUE_RGB
    arr[60:180, 70:170] = _SKIN_RGB
    arr += rng.normal(0, 2, arr.shape)
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))


def _chroma(rgb) -> float:
    px = np.asarray(rgb, np.uint8).reshape(-1, 1, 3)
    lab = cv2.cvtColor(px.astype(np.float32) / 255.0, cv2.COLOR_RGB2LAB)
    return float(np.hypot(lab[..., 1], lab[..., 2]).mean())


def _patch_chroma(img: Image.Image, box) -> float:
    y0, y1, x0, x1 = box
    return _chroma(np.asarray(img.convert("RGB"))[y0:y1, x0:x1])


def _flat(rgb, size=64) -> Image.Image:
    return Image.fromarray(np.full((size, size, 3), rgb, np.uint8))


def test_skin_weight_covers_skin_not_blue_green_or_gray():
    for rgb in (_SKIN_RGB, _DEEP_SKIN_RGB):
        lab = cv2.cvtColor(np.full((1, 1, 3), rgb, np.uint8), cv2.COLOR_RGB2LAB).astype(np.float32)
        assert ip._skin_tone_weight(lab[..., 0], lab[..., 1], lab[..., 2]).min() > 0.8
    for rgb in (_BLUE_RGB, _GREEN_RGB, (128, 128, 128), (250, 120, 0)):
        lab = cv2.cvtColor(np.full((1, 1, 3), rgb, np.uint8), cv2.COLOR_RGB2LAB).astype(np.float32)
        assert ip._skin_tone_weight(lab[..., 0], lab[..., 1], lab[..., 2]).max() < 0.2, rgb


def test_negative_saturation_spares_skin_but_mutes_the_rest():
    for sat in (-0.184, -0.35):          # 기본 레시피 바닥, 뮤트 계열(korean_gamsung) 바닥
        skin_before, blue_before = _chroma(_SKIN_RGB), _chroma(_BLUE_RGB)
        skin_after = _patch_chroma(ip._apply_lab_adjustments(_flat(_SKIN_RGB), saturation=sat), (0, 64, 0, 64))
        blue_after = _patch_chroma(ip._apply_lab_adjustments(_flat(_BLUE_RGB), saturation=sat), (0, 64, 0, 64))
        # 피부는 채도 감소의 절반 남짓만, 옷은 거의 그대로 다 뺀다
        assert skin_after >= skin_before * (1.0 + sat * 0.7)
        assert blue_after <= blue_before * (1.0 + sat * 0.8)


def test_positive_saturation_is_unchanged_on_skin():
    img = _flat(_SKIN_RGB)
    after = _patch_chroma(ip._apply_lab_adjustments(img, saturation=0.2), (0, 64, 0, 64))
    assert after > _chroma(_SKIN_RGB) * 1.15   # 보호는 낮출 때만 — 올릴 때는 그대로


def test_monochrome_is_still_fully_gray_on_skin():
    out = np.asarray(ip._apply_lab_adjustments(_flat(_SKIN_RGB), saturation=-1.0), np.int16)
    assert np.abs(out[..., 0] - out[..., 2]).max() <= 1


def test_vivid_clothes_portrait_does_not_hit_the_saturation_floor():
    img = _portrait()
    p_portrait, _ = build_params_with_comment(img, {}, {"subjectType": "인물"})
    p_object, _ = build_params_with_comment(img, {}, {"subjectType": "사물"})
    # 같은 사진이라도 사물이면 예전처럼 바닥(−0.184), 인물이면 측정분이 절반
    assert p_object["saturation"] <= -0.18
    assert -0.14 <= p_portrait["saturation"] < 0      # 레시피의 차분함(−0.08)은 남는다


def test_default_portrait_skin_chroma_is_retained():
    img = _portrait()
    for profile in ({}, {"trendCategory": "korean_gamsung", "styleSource": "manual"}):
        p, _ = build_params_with_comment(img, profile, {"subjectType": "인물"})
        out = ip._apply_lab_adjustments(img, saturation=p["saturation"])
        ratio = _patch_chroma(out, (80, 160, 90, 150)) / _patch_chroma(img, (80, 160, 90, 150))
        assert ratio > 0.88, (profile, p["saturation"], ratio)   # 예전: 기본 0.82, 뮤트 0.65


def test_bw_portrait_is_still_monochrome():
    p, _ = build_params_with_comment(_portrait(), {"trendCategory": "bw_grain", "styleSource": "manual"},
                                     {"subjectType": "인물"})
    assert p["saturation"] == -1.0


def test_reference_portrait_keeps_measured_desaturation():
    """레퍼런스(그 사람 피드의 실제 채도)가 있으면 그게 취향이다 — 덜어내지 않는다."""
    img = _portrait()
    ref = {"brightness": 0.5, "contrast": 0.6, "saturation": 0.25, "warmth": 0.05}
    p_portrait, _ = build_params_with_comment(img, {}, {"subjectType": "인물"}, reference=ref)
    p_object, _ = build_params_with_comment(img, {}, {"subjectType": "사물"}, reference=ref)
    assert p_portrait["saturation"] == p_object["saturation"]
