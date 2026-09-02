"""영역별 보정 합성 회귀 테스트.

과거 버그: 얼굴 마스크를 이미지 크기 기준의 좁은 가우시안으로만 풀고, 영역을
순서대로 덧칠했다. background 마스크는 얼굴의 여집합이라 얼굴 바로 밖에서
알파가 1.0이었고, 그 덧칠이 얼굴 테두리의 보정을 원본으로 되돌렸다.
결과적으로 얼굴 가운데만 톤이 살고 윤곽·목은 그대로여서 얼굴이 겉돌았다.

MediaPipe 없이 합성 랜드마크로 마스크를 만들고, 균일한 회색 이미지에
보정을 걸어 실제로 어디에 얼마나 적용됐는지 측정한다.
"""

import cv2
import numpy as np
import pytest
from PIL import Image

from image_processor import (
    _FACE_TONE_LIMIT,
    _SKIN_FACE_OVAL,
    apply_regional_transforms,
    build_face_skin_mask,
)
from tests.test_skin_mask import IMG_H, IMG_W, _landmarks

GRAY = 128
FACE_WIDTHS = [250, 400, 900]


def _setup(face_w: int):
    pt = _landmarks(face_w)
    face = build_face_skin_mask(pt, IMG_H, IMG_W, mode="tone")
    regions = {"face": face, "background": cv2.bitwise_not(face)}
    base = Image.fromarray(np.full((IMG_H, IMG_W, 3), GRAY, np.uint8))
    oval = np.zeros((IMG_H, IMG_W), np.uint8)
    cv2.fillPoly(oval, [np.array([pt(i) for i in _SKIN_FACE_OVAL], np.int32)], 255)
    return pt, regions, base, oval


def _luma(img: Image.Image) -> np.ndarray:
    return np.array(img, np.float32)[:, :, 0]


@pytest.mark.parametrize("face_w", FACE_WIDTHS)
def test_face_tone_reaches_whole_oval(face_w):
    """배경 보정이 함께 걸려도 얼굴 전체가 같은 세기로 밝아져야 한다."""
    _, regions, base, oval = _setup(face_w)
    out = _luma(apply_regional_transforms(
        base, regions,
        {"face": {"brightness": 0.3}, "background": {"brightness": -0.1}},
    ))
    inside = out[oval > 0]
    peak = inside.max() - GRAY
    assert peak > 5, f"얼굴이 거의 밝아지지 않았다 (peak {peak:.1f})"
    full = (inside - GRAY > peak * 0.9).mean()
    assert full > 0.97, f"얼굴 {face_w}px에서 90% 이상 적용된 면적 {full:.1%}"


@pytest.mark.parametrize("face_w", FACE_WIDTHS)
def test_boundary_feather_scales_with_face(face_w):
    """경계 전이폭은 이미지 크기가 아니라 얼굴 크기에 비례해야 한다.

    이미지 기준으로 풀던 시절에는 얼굴이 크든 작든 12~13px이라,
    큰 얼굴에서는 톤이 뚝 끊기는 타원 테두리가 보였다.
    """
    _, regions, base, _ = _setup(face_w)
    out = _luma(apply_regional_transforms(base, regions, {"face": {"brightness": 0.3}}))
    row = out[int(IMG_H * 0.45)] - GRAY
    peak = row.max()
    band = np.where((row > peak * 0.1) & (row < peak * 0.9))[0]
    left = band[band < IMG_W // 2]
    width = left.max() - left.min() + 1
    assert width > face_w * 0.06, f"얼굴 {face_w}px 경계가 {width}px로 급하다"


@pytest.mark.parametrize("face_w", FACE_WIDTHS)
def test_face_tone_carries_into_neck(face_w):
    """턱선에서 톤이 끊기지 않고 목 쪽으로 이어져야 한다."""
    _, regions, base, _ = _setup(face_w)
    out = _luma(apply_regional_transforms(base, regions, {"face": {"brightness": 0.3}}))
    cx = IMG_W // 2
    chin = int(np.where(regions["face"][:, cx] > 0)[0].max())
    inside = out[chin - int(face_w * 0.1), cx] - GRAY
    neck = out[chin + int(face_w * 0.05), cx] - GRAY
    assert neck > inside * 0.4, f"턱 아래 보정이 {neck / max(inside, 1e-6):.0%}로 급감"


def test_face_tone_is_clamped():
    """모델이 과한 얼굴 톤을 보내도 상한에서 잘려야 한다."""
    _, regions, base, oval = _setup(400)
    strong = _luma(apply_regional_transforms(base, regions, {"face": {"brightness": 1.0}}))
    limit = _luma(apply_regional_transforms(
        base, regions, {"face": {"brightness": _FACE_TONE_LIMIT}}))
    assert strong[oval > 0].max() == pytest.approx(limit[oval > 0].max(), abs=1.0)


def test_regions_do_not_cancel_each_other():
    """겹치는 페더에서 앞선 영역의 보정이 원본으로 되돌아가면 안 된다."""
    _, regions, base, oval = _setup(400)
    face_only = _luma(apply_regional_transforms(base, regions, {"face": {"brightness": 0.3}}))
    with_bg = _luma(apply_regional_transforms(
        base, regions,
        {"face": {"brightness": 0.3}, "background": {"brightness": -0.3}},
    ))
    core = oval > 0
    assert with_bg[core].mean() == pytest.approx(face_only[core].mean(), abs=1.0)


def test_feather_uses_largest_face_not_total_area():
    """단체 사진에서 얼굴 수만큼 페더가 커지면 안 된다.

    마스크 전체 면적으로 크기를 재던 시절에는 얼굴이 늘어날수록
    √n배씩 램프가 번져 배경까지 얼굴 톤이 물들었다.
    """
    from image_processor import _soften_region_mask

    one = build_face_skin_mask(_landmarks(300, cx_frac=0.3), IMG_H, IMG_W, mode="tone")
    two = cv2.bitwise_or(
        one, build_face_skin_mask(_landmarks(300, cx_frac=0.7), IMG_H, IMG_W, mode="tone"))
    spread_one = (_soften_region_mask(one, "face") > 0.02).sum() - cv2.countNonZero(one)
    spread_two = (_soften_region_mask(two, "face") > 0.02).sum() - cv2.countNonZero(two)
    assert spread_two < spread_one * 2.4, (
        f"얼굴 2명에서 램프 면적 {spread_two}가 1명 {spread_one}의 2.4배를 넘는다")
