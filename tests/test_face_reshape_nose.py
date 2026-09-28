"""얼굴 슬림 보정이 코를 가로로 눌러 각지게 만들던 문제 회귀 테스트.

윤곽만 옮기고 코에는 제어점이 없으면, 워프가 주변 볼의 변위를 코까지
보간해 버린다. face_slim은 dx만 주고 dy는 0이라 그 결과가 "가로로만 눌리기"다.
"""

import numpy as np
import pytest

from image_processor import (
    _FACE_CONTOUR_LEFT,
    _FACE_CONTOUR_RIGHT,
    _JAW_LEFT,
    _JAW_RIGHT,
    _LEFT_EYE_CONTOUR,
    _NOSE_ANCHORS,
    _RIGHT_EYE_CONTOUR,
    _build_reshape_controls,
)

# 합성 얼굴 — 정면, 중심 (300, 300)
CX, CY, FW, FH = 300.0, 300.0, 260.0, 340.0
N_LANDMARKS = 478


def _synthetic_pt(idx: int) -> tuple[float, float]:
    """랜드마크 인덱스를 그럴듯한 좌표로 매핑한다.

    윤곽 인덱스는 얼굴 타원 위에, 코 인덱스는 중앙 세로선 근처에 둔다.
    실제 배치와 정확히 같을 필요는 없다 — 검증 대상은 "코에 제어점이
    있느냐"가 만드는 변위장의 차이이지 좌표의 사실성이 아니다.
    """
    if idx in _NOSE_ANCHORS:
        k = _NOSE_ANCHORS.index(idx)
        return CX + (k % 5 - 2) * 9.0, CY - 45.0 + (k // 5) * 22.0

    contour = _FACE_CONTOUR_LEFT + [i for i in _FACE_CONTOUR_RIGHT if i != 152]
    if idx in contour:
        k = contour.index(idx)
        side = -1 if k < len(_FACE_CONTOUR_LEFT) else 1
        n = len(_FACE_CONTOUR_LEFT)
        t = np.pi * (0.5 + side * ((k % n) / (n - 1)) * 0.9)
        return CX + side * (FW / 2) * abs(np.sin(t)), CY - (FH / 2) * np.cos(t)

    for eye, ex in ((_LEFT_EYE_CONTOUR, CX + 60), (_RIGHT_EYE_CONTOUR, CX - 60)):
        if idx in eye:
            a = 2 * np.pi * eye.index(idx) / len(eye)
            return ex + 22 * np.cos(a), CY - 70 + 11 * np.sin(a)

    return CX, CY  # 그 밖의 인덱스는 쓰이지 않는다


def _jacobian(src, dst, size=600):
    """제어점이 만드는 역워프 변위장의 국소 배율을 구한다.

    image_processor._mls_similarity_warp와 같은 역거리가중(IDW) 계산.
    반환: (가로 배율 맵, 세로 배율 맵). 1.0이면 그 지점은 변형 없음.
    """
    src = np.asarray(src, np.float32)
    dst = np.asarray(dst, np.float32)
    ys, xs = np.mgrid[0:size, 0:size].astype(np.float32)
    flat = np.stack([xs, ys], -1).reshape(-1, 2)

    wts = np.empty((len(src), flat.shape[0]), np.float32)
    for i in range(len(src)):
        d = flat - dst[i]
        wts[i] = 1.0 / (np.sum(d * d, axis=1) + 1e-6)

    disp = np.zeros_like(flat)
    for i in range(len(src)):
        disp += wts[i][:, None] * (src[i] - dst[i])[None, :]
    disp /= wts.sum(0)[:, None]

    mx = disp[:, 0].reshape(size, size)
    my = disp[:, 1].reshape(size, size)
    return 1 + np.gradient(mx, axis=1), 1 + np.gradient(my, axis=0)


def _anisotropy(jx, jy, cx, cy, half):
    sl = (slice(int(cy - half), int(cy + half)), slice(int(cx - half), int(cx + half)))
    return abs(jx[sl].mean() - jy[sl].mean())


def test_코_앵커는_움직이는_인덱스와_겹치지_않는다():
    """같은 인덱스가 이동점과 고정점 양쪽에 들어가면 서로 상쇄된다."""
    moved = set(
        _FACE_CONTOUR_LEFT + _FACE_CONTOUR_RIGHT + _JAW_LEFT + _JAW_RIGHT
        + _LEFT_EYE_CONTOUR + _RIGHT_EYE_CONTOUR
    )
    assert moved.isdisjoint(_NOSE_ANCHORS)


def test_face_slim이_코_앵커를_변위0으로_넣는다():
    src, dst = _build_reshape_controls(_synthetic_pt, N_LANDMARKS, 0.7, 0.0, 0.0)
    anchored = [
        (s, d) for s, d in zip(src, dst)
        if s == d
    ]
    assert len(anchored) == len(_NOSE_ANCHORS)


def test_보정이_꺼져_있으면_앵커도_넣지_않는다():
    """제어점이 코 앵커뿐이면 워프가 무의미하게 돌아간다."""
    src, dst = _build_reshape_controls(_synthetic_pt, N_LANDMARKS, 0.0, 0.0, 0.0)
    assert src == [] and dst == []


def test_랜드마크가_모자라면_있는_것만_쓴다():
    """iris(468~)가 없는 결과에서도 인덱스 초과로 터지지 않아야 한다."""
    full, _ = _build_reshape_controls(_synthetic_pt, N_LANDMARKS, 0.7, 0.0, 0.0)
    short, short_dst = _build_reshape_controls(_synthetic_pt, 200, 0.7, 0.0, 0.0)

    dropped = sum(1 for i in _NOSE_ANCHORS if i >= 200)
    assert len(short) == len(short_dst) == len(full) - dropped


def test_코가_가로로_눌리지_않는다():
    """앵커가 있으면 코의 비등방(가로만 눌림)이 볼보다 훨씬 작아야 한다."""
    src, dst = _build_reshape_controls(_synthetic_pt, N_LANDMARKS, 0.7, 0.0, 0.0)
    jx, jy = _jacobian(src, dst)

    nose = _anisotropy(jx, jy, CX, CY, 30)
    cheek = _anisotropy(jx, jy, CX - 85, CY, 25)

    # 코는 거의 원래 비율을 지키고
    assert nose < 0.03, f"코 비등방 {nose:.3f} — 가로로 눌리고 있다"
    # 슬림 압축은 볼에서 일어나야 한다
    assert cheek > nose * 3, f"볼 {cheek:.3f} vs 코 {nose:.3f} — 압축 위치가 잘못됐다"


def test_앵커를_빼면_코가_눌린다():
    """회귀 방지 — 이 테스트가 실패하면 앵커가 효과를 잃은 것이다."""
    src, dst = _build_reshape_controls(_synthetic_pt, N_LANDMARKS, 0.7, 0.0, 0.0)
    n = len(_NOSE_ANCHORS)
    jx, jy = _jacobian(src[:-n], dst[:-n])   # 앵커 제외 = 예전 동작
    assert _anisotropy(jx, jy, CX, CY, 30) > 0.08
