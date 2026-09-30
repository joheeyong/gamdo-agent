"""얼굴/체형 보정이 '티 나지 않게' 동작하는지 — 예전에 실제로 났던 문제들의 회귀 테스트.

예전 동작 (실사진에서 확인):
  - 얼굴 슬림이 배경까지 9~12px 휘게 했다 (문틀·벽 모서리가 볼 옆에서 굽음)
  - 눈 확대가 얼굴 전체와 배경을 출렁이게 했다
  - 볼을 줄이면 헤어라인·관자놀이가 따라 들어갔다
  - 턱선 보정이 턱끝을 끌어내려 V자로 만들었다
  - 살짝 돌린 얼굴에서 한쪽 볼만 3배 넘게 깎였다
"""

import numpy as np
import pytest

from _synthetic_face import N_LANDMARKS, face_pt
from image_processor import (
    _FACE_CONTOUR_LEFT,
    _FACE_CONTOUR_RIGHT,
    _FACE_UPPER_ANCHORS,
    _build_reshape_controls,
    _eye_bulge,
    _support_weight,
)

SIZE = 600


def _disp_at(c, x, y):
    dx, dy = c.field(np.array([x], np.float32), np.array([y], np.float32))
    return float(dx[0, 0]), float(dy[0, 0])


def _mag_at(c, pt_xy):
    dx, dy = _disp_at(c, *pt_xy)
    return (dx * dx + dy * dy) ** 0.5


# ── 배경 보호 ──


def test_실루엣_밖으로_가면_변위_가중치가_0이_된다():
    person = np.zeros((200, 200), np.float32)
    person[50:150, 60:140] = 1.0
    w = _support_weight(person, band=20.0)
    assert w[100, 100] == pytest.approx(1.0)          # 인물 안
    assert w[100, 150] > 0.0                           # 바로 밖 띠: 채움용으로 조금 늘어남
    assert w[100, 170] == 0.0 and w[10, 10] == 0.0     # band 밖 배경: 변위 0
    assert np.all(np.diff(w[100, 140:170]) <= 1e-6)    # 밖으로 갈수록 단조 감소


def test_분할이_없으면_가중치도_없다():
    assert _support_weight(None, 10.0) is None
    assert _support_weight(np.zeros((10, 10), np.float32), 10.0) is None


# ── 눈 확대 ──


def test_눈_확대는_타원_밖을_전혀_움직이지_않는다():
    xs = np.arange(200, dtype=np.float32)
    ys = np.arange(200, dtype=np.float32)
    dx, dy = _eye_bulge(xs, ys, (100.0, 100.0), 30.0, 20.0, 0.0, 0.22)
    yy, xx = np.mgrid[0:200, 0:200]
    outside = ((xx - 100) / 30.0) ** 2 + ((yy - 100) / 20.0) ** 2 >= 1.0
    assert np.abs(dx[outside]).max() == 0.0 and np.abs(dy[outside]).max() == 0.0
    assert np.abs(dx).max() > 0.5                      # 안쪽은 실제로 확대됨


def test_눈_확대는_접히지_않는다():
    """반경 방향 사상이 단조여야 눈꺼풀이 겹치거나 뒤집히지 않는다."""
    xs = np.arange(100, 131, dtype=np.float32)
    dx, _ = _eye_bulge(xs, np.array([100.0], np.float32), (100.0, 100.0), 30.0, 20.0, 0.0, 0.22)
    src_x = xs + dx[0]                                  # 출력 x → 원본 x
    assert np.all(np.diff(src_x) > 0)


def test_얼굴_보정에서_눈_확대는_얼굴_폭_대비_작게_제한된다():
    pt = face_pt(0.0)
    c = _build_reshape_controls(pt, N_LANDMARKS, 0.0, 0.0, 1.0)
    assert c.eyes and not c.pts                         # 눈만 — 윤곽 장은 없음
    face_w = abs(pt(454)[0] - pt(234)[0])
    xs = np.arange(SIZE, dtype=np.float32)
    dx, dy = c.field(xs, xs)
    assert np.hypot(dx, dy).max() < 0.03 * face_w       # 최대치에서도 얼굴 폭의 3% 미만


# ── 고정점 ──


# 광대점(234·454) 바로 위 옆 관자놀이. 광대가 조금 움직이므로 여기는 부드럽게
# 이어지는 전이 구간이다 (끊으면 경계가 꺾인다). 이마·헤어라인은 고정.
_TEMPLE_SIDE = {127, 356, 162, 389}


def test_볼을_줄여도_이마_헤어라인은_움직이지_않는다():
    pt = face_pt(0.0)
    c = _build_reshape_controls(pt, N_LANDMARKS, 1.0, 1.0, 0.0)
    cheek = max(_mag_at(c, pt(i)) for i in _FACE_CONTOUR_LEFT[2:5])
    for idx in _FACE_UPPER_ANCHORS:
        limit = 0.35 if idx in _TEMPLE_SIDE else 0.02
        assert _mag_at(c, pt(idx)) < limit * cheek, idx


def test_턱선_보정이_턱끝을_끌어내리지_않는다():
    pt = face_pt(0.0)
    c = _build_reshape_controls(pt, N_LANDMARKS, 0.0, 1.0, 0.0)
    _, chin_dy = _disp_at(c, *pt(152))
    jaw = max(_mag_at(c, pt(i)) for i in _FACE_CONTOUR_LEFT[4:8])
    assert abs(chin_dy) < 0.15 * jaw


# ── 얼굴 방향 ──


def test_정면_얼굴은_좌우를_같게_줄인다():
    pt = face_pt(0.0)
    c = _build_reshape_controls(pt, N_LANDMARKS, 0.6, 0.0, 0.0)
    left = [_mag_at(c, pt(i)) for i in _FACE_CONTOUR_LEFT[2:6]]
    right = [_mag_at(c, pt(i)) for i in _FACE_CONTOUR_RIGHT[2:6]]
    assert sum(left) == pytest.approx(sum(right), rel=0.1)


def test_돌린_얼굴도_한쪽_볼만_깎지_않는다():
    pt = face_pt(20.0)
    c = _build_reshape_controls(pt, N_LANDMARKS, 0.6, 0.0, 0.0)
    left = max(_mag_at(c, pt(i)) for i in _FACE_CONTOUR_LEFT[2:6])
    right = max(_mag_at(c, pt(i)) for i in _FACE_CONTOUR_RIGHT[2:6])
    assert min(left, right) > 0.0
    assert max(left, right) / min(left, right) < 2.5   # 예전 실사진: 26.1 / 8.1 ≈ 3.2


def test_거울상_얼굴은_같은_양만큼_보정된다():
    a = _build_reshape_controls(face_pt(20.0), N_LANDMARKS, 0.6, 0.4, 0.4)
    b = _build_reshape_controls(face_pt(20.0, mirror=True), N_LANDMARKS, 0.6, 0.4, 0.4)
    # 랜드마크 번호는 얼굴 기준이라 거울상에서도 방향 부호는 같고 크기만 같아야 한다
    assert abs(a.yaw) == pytest.approx(abs(b.yaw), abs=1e-6)
    assert a.max_move == pytest.approx(b.max_move, rel=1e-3)
    assert [e[4] for e in a.eyes] == pytest.approx([e[4] for e in b.eyes], rel=1e-3)


def test_많이_돌린_얼굴은_보정을_줄인다():
    front = _build_reshape_controls(face_pt(0.0), N_LANDMARKS, 0.6, 0.0, 0.0)
    turned = _build_reshape_controls(face_pt(40.0), N_LANDMARKS, 0.6, 0.0, 0.0)
    assert turned.max_move < front.max_move
