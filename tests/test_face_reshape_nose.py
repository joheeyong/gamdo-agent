"""얼굴 슬림 보정이 코를 가로로 눌러 각지게 만들던 문제 회귀 테스트.

윤곽만 옮기고 코에는 제어점이 없으면, 변위장이 주변 볼의 변위를 코까지
번지게 한다. face_slim은 가로 변위가 대부분이라 그 결과가 "가로로만 눌리기"다.
코에 변위 0인 고정점을 넣어 막는다.
"""

import numpy as np

from _synthetic_face import N_LANDMARKS, face_pt
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

PT = face_pt(0.0)
SIZE = 600


def _jacobian(c, size=SIZE):
    """변위장의 국소 배율 (가로, 세로). 1.0이면 그 지점은 변형 없음."""
    xs = np.arange(size, dtype=np.float32)
    ys = np.arange(size, dtype=np.float32)
    dx, dy = c.field(xs, ys)
    return 1 + np.gradient(dx, axis=1), 1 + np.gradient(dy, axis=0)


def _anisotropy(jx, jy, cx, cy, half):
    sl = (slice(int(cy - half), int(cy + half)), slice(int(cx - half), int(cx + half)))
    return abs(jx[sl].mean() - jy[sl].mean())


def _nose_center():
    pts = np.array([PT(i) for i in _NOSE_ANCHORS])
    return pts.mean(axis=0)


def _is_nose_anchor(c, k):
    return any(np.allclose(c.pts[k], PT(i), atol=1e-6) for i in _NOSE_ANCHORS)


def test_코_앵커는_움직이는_인덱스와_겹치지_않는다():
    """같은 인덱스가 이동점과 고정점 양쪽에 들어가면 서로 상쇄된다."""
    moved = set(
        _FACE_CONTOUR_LEFT + _FACE_CONTOUR_RIGHT + _JAW_LEFT + _JAW_RIGHT
        + _LEFT_EYE_CONTOUR + _RIGHT_EYE_CONTOUR
    )
    assert moved.isdisjoint(_NOSE_ANCHORS)


def test_face_slim이_코_앵커를_변위0으로_넣는다():
    c = _build_reshape_controls(PT, N_LANDMARKS, 0.7, 0.0, 0.0)
    anchored = [k for k in range(len(c.pts)) if _is_nose_anchor(c, k)]
    assert len(anchored) == len(_NOSE_ANCHORS)
    assert all(np.allclose(c.disp[k], 0.0) for k in anchored)


def test_보정이_꺼져_있으면_아무것도_하지_않는다():
    """제어점이 고정점뿐이면 워프가 무의미하게 돌아간다."""
    c = _build_reshape_controls(PT, N_LANDMARKS, 0.0, 0.0, 0.0)
    assert not c.active and c.pts == [] and c.eyes == []


def test_랜드마크가_모자라면_있는_것만_쓴다():
    """iris(468~)가 없는 결과에서도 인덱스 초과로 터지지 않아야 한다."""
    full = _build_reshape_controls(PT, N_LANDMARKS, 0.7, 0.0, 0.3)
    short = _build_reshape_controls(PT, 468, 0.7, 0.0, 0.3)
    assert full.active and short.active
    assert len(short.pts) == len(short.disp)


def test_코가_가로로_눌리지_않는다():
    """고정점이 있으면 코의 비등방(가로만 눌림)이 볼보다 훨씬 작아야 한다."""
    c = _build_reshape_controls(PT, N_LANDMARKS, 0.7, 0.0, 0.0)
    jx, jy = _jacobian(c)
    nx, ny = _nose_center()
    cheek_x = PT(_FACE_CONTOUR_LEFT[3])[0]

    nose = _anisotropy(jx, jy, nx, ny, 12)
    cheek = _anisotropy(jx, jy, cheek_x, ny + 30, 12)
    assert nose < 0.03, f"코 비등방 {nose:.3f} — 가로로 눌리고 있다"
    assert cheek > nose * 3, f"볼 {cheek:.3f} vs 코 {nose:.3f} — 압축 위치가 잘못됐다"


def test_코_앵커를_빼면_코가_더_눌린다():
    """회귀 방지 — 이 테스트가 실패하면 코 고정점이 효과를 잃은 것이다."""
    c = _build_reshape_controls(PT, N_LANDMARKS, 0.7, 0.0, 0.0)
    nx, ny = _nose_center()
    with_anchor = _anisotropy(*_jacobian(c), nx, ny, 12)

    keep = [k for k in range(len(c.pts)) if not _is_nose_anchor(c, k)]
    c.pts = [c.pts[k] for k in keep]
    c.disp = [c.disp[k] for k in keep]
    without = _anisotropy(*_jacobian(c), nx, ny, 12)
    assert without > with_anchor * 2, (without, with_anchor)
