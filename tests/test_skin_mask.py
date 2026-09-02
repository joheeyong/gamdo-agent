"""얼굴 마스크 커버리지 회귀 테스트.

MediaPipe 없이 랜드마크 좌표만 합성해서 마스크 로직만 검증한다.
과거 버그: 침식 크기가 이미지 기준(짧은 변의 2.5%)이라 얼굴이 작게 찍힌
사진일수록 마스크가 줄어, 얼굴 한가운데만 보정되고 나머지는 원본으로 남았다.
"""

import math

import cv2
import numpy as np
import pytest

from image_processor import (
    _LEFT_EYE,
    _LEFT_EYEBROW,
    _LIPS,
    _RIGHT_EYE,
    _RIGHT_EYEBROW,
    _SKIN_FACE_OVAL,
    build_face_skin_mask,
)

IMG_W, IMG_H = 1568, 1176


def _landmarks(face_w_px: float, cx_frac: float = 0.5, cy_frac: float = 0.45):
    """해부학적으로 그럴듯한 위치에 필요한 인덱스만 배치한다."""
    cx, cy = IMG_W * cx_frac, IMG_H * cy_frac
    rx = face_w_px / 2.0
    ry = rx * 1.32
    pts: dict[int, tuple[float, float]] = {}

    for k, idx in enumerate(_SKIN_FACE_OVAL):
        a = -math.pi / 2 + 2 * math.pi * k / len(_SKIN_FACE_OVAL)
        pts[idx] = (cx + rx * math.cos(a), cy + ry * math.sin(a))
    pts[234] = (cx - rx, cy)
    pts[454] = (cx + rx, cy)

    def ellipse(ids, ecx, ecy, erx, ery):
        for k, idx in enumerate(ids):
            a = 2 * math.pi * k / len(ids)
            pts[idx] = (ecx + erx * math.cos(a), ecy + ery * math.sin(a))

    ellipse(_LEFT_EYE, cx - rx * 0.42, cy - ry * 0.18, rx * 0.26, ry * 0.10)
    ellipse(_RIGHT_EYE, cx + rx * 0.42, cy - ry * 0.18, rx * 0.26, ry * 0.10)
    ellipse(_LEFT_EYEBROW, cx - rx * 0.44, cy - ry * 0.36, rx * 0.30, ry * 0.05)
    ellipse(_RIGHT_EYEBROW, cx + rx * 0.44, cy - ry * 0.36, rx * 0.30, ry * 0.05)
    ellipse(_LIPS, cx, cy + ry * 0.52, rx * 0.36, ry * 0.11)
    for idx, (dx, dy) in {
        6: (0, -0.14), 197: (0, -0.05), 195: (0, 0.05), 5: (0, 0.14),
        4: (0, 0.22), 1: (0, 0.26), 2: (0, 0.30),
        98: (-0.14, 0.28), 327: (0.14, 0.28), 168: (0, -0.18),
    }.items():
        pts[idx] = (cx + rx * dx, cy + ry * dy)

    return lambda i: (int(pts[i][0]), int(pts[i][1]))


def _oval_area(pt) -> int:
    oval = np.zeros((IMG_H, IMG_W), np.uint8)
    cv2.fillPoly(oval, [np.array([pt(i) for i in _SKIN_FACE_OVAL], np.int32)], 255)
    return cv2.countNonZero(oval)


FACE_WIDTHS = [250, 400, 600, 900]


@pytest.mark.parametrize("face_w", FACE_WIDTHS)
def test_tone_mask_covers_whole_face(face_w):
    """톤 보정용 마스크는 얼굴을 거의 다 덮어야 한다.

    볼만 밝히고 눈두덩·입술은 그대로 두면 얼굴이 얼룩덜룩해진다.
    """
    pt = _landmarks(face_w)
    mask = build_face_skin_mask(pt, IMG_H, IMG_W, mode="tone")
    coverage = cv2.countNonZero(mask) / _oval_area(pt)
    assert coverage > 0.90, f"얼굴 {face_w}px에서 톤 마스크 커버리지 {coverage:.1%}"


@pytest.mark.parametrize("face_w", FACE_WIDTHS)
def test_texture_mask_keeps_most_skin(face_w):
    """질감용 마스크는 이목구비만 빼고 피부를 남겨야 한다."""
    pt = _landmarks(face_w)
    mask = build_face_skin_mask(pt, IMG_H, IMG_W, mode="texture")
    coverage = cv2.countNonZero(mask) / _oval_area(pt)
    assert coverage > 0.60, f"얼굴 {face_w}px에서 질감 마스크 커버리지 {coverage:.1%}"


def test_coverage_does_not_depend_on_face_size():
    """같은 얼굴이 크게 찍히든 작게 찍히든 보정 범위는 같아야 한다.

    이미지 기준 침식을 쓰던 시절에는 250px 얼굴 44.6%, 900px 얼굴 74.9%로
    30%p나 벌어졌다.
    """
    for mode in ("tone", "texture"):
        ratios = [
            cv2.countNonZero(build_face_skin_mask(
                _landmarks(fw), IMG_H, IMG_W, mode=mode)) / _oval_area(_landmarks(fw))
            for fw in FACE_WIDTHS
        ]
        spread = max(ratios) - min(ratios)
        assert spread < 0.06, f"{mode} 커버리지 편차 {spread:.1%}: {ratios}"


def test_texture_mask_excludes_eyes_and_lips():
    """눈동자·입술 중심은 질감 마스크에서 빠져 있어야 한다."""
    pt = _landmarks(600)
    mask = build_face_skin_mask(pt, IMG_H, IMG_W, mode="texture")
    for name, ids in (("왼눈", _LEFT_EYE), ("오른눈", _RIGHT_EYE), ("입술", _LIPS)):
        xs = [pt(i)[0] for i in ids]
        ys = [pt(i)[1] for i in ids]
        cx, cy = sum(xs) // len(xs), sum(ys) // len(ys)
        assert mask[cy, cx] == 0, f"{name} 중심이 질감 마스크에 포함됨"
