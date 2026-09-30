"""얼굴·체형 워프의 성긴 격자 변위장 회귀 테스트.

변위장은 성긴 격자(_field_step px)에서 계산해 선형 보간한 뒤 remap 한 번으로
적용한다 (_remap_region). 여기서는 그 근사가
  1) 모든 픽셀에서 계산한 변위장과 사실상 같고
  2) 반 픽셀 밀림 없이 정렬돼 있으며
  3) step=1이면 조밀한 계산과 수치적으로 같고
  4) 최종 출력 화질이 조밀한 계산과 구별되지 않는다
는 것을 확인한다.
"""

import cv2
import numpy as np

from _synthetic_face import N_LANDMARKS, face_pt, textured_image
from image_processor import (
    _build_reshape_controls,
    _field_step,
    _gaussian_field,
    _remap_region,
)


def _face_ctrl():
    return _build_reshape_controls(face_pt(), N_LANDMARKS, 0.5, 0.5, 0.5)


def _coord_image(h, w):
    """각 픽셀 값이 자기 (x, y) 좌표인 float32 이미지.

    선형 램프는 쌍선형 보간으로 정확히 복원되므로, 이 이미지를 워프하면
    출력 값이 곧 remap 좌표(= 좌표 + 변위)가 된다.
    """
    ys, xs = np.mgrid[0:h, 0:w].astype(np.float32)
    return np.dstack([xs, ys, np.zeros_like(xs)])


ROI = (60, 60, 540, 560)
H, W = 600, 600


def test_기본_격자_간격은_성긴_격자다():
    assert _field_step(260.0) > 1
    assert _field_step(1500.0) <= 12


def test_격자_1이면_조밀한_계산과_같다():
    ctrl = _face_ctrl()
    out, (dx, dy) = _remap_region(_coord_image(H, W), ROI, ctrl.field, 1, None, 1.0)
    x0, y0, x1, y1 = ROI
    ref_x, ref_y = ctrl.field(np.arange(x0, x1, dtype=np.float32), np.arange(y0, y1, dtype=np.float32))
    # ROI 가장자리 램프(4%) 안쪽만 비교 — 거기서는 장이 이미 0이다
    inner = (slice(24, -24), slice(24, -24))
    np.testing.assert_allclose(dx[inner], ref_x[inner], atol=1e-4)
    np.testing.assert_allclose(dy[inner], ref_y[inner], atol=1e-4)
    # 출력 좌표 = 원래 좌표 + 변위 (remap이 정확히 그 자리를 읽는다)
    ys, xs = np.mgrid[y0:y1, x0:x1].astype(np.float32)
    np.testing.assert_allclose(out[y0:y1, x0:x1, 0][inner], (xs + dx)[inner], atol=2e-3)
    np.testing.assert_allclose(out[y0:y1, x0:x1, 1][inner], (ys + dy)[inner], atol=2e-3)


def test_성긴_격자_변위장이_조밀한_계산과_거의_같다():
    ctrl = _face_ctrl()
    step = _field_step(ctrl.scale)
    assert step > 1
    _, dense = _remap_region(_coord_image(H, W), ROI, ctrl.field, 1, None, 1.0)
    _, coarse = _remap_region(_coord_image(H, W), ROI, ctrl.field, step, None, 1.0)
    err = np.hypot(dense[0] - coarse[0], dense[1] - coarse[1])
    mag = np.hypot(*dense)
    assert float(mag.max()) > 2.0  # 비교할 만큼 변형이 있어야 의미가 있다
    # 가우시안 장은 σ(얼굴 폭의 11%) 규모로 매끄러워 선형 보간 오차가 아주 작다
    assert float(err.max()) < 0.1, float(err.max())
    assert float(err.mean()) < 0.01, float(err.mean())


def test_격자_보간에_반_픽셀_밀림이_없다():
    """변위장이 어디서나 상수면 보간된 장도 정확히 그 상수여야 한다.

    격자점을 픽셀 중심 규약과 다르게 두면 가장자리에서 어긋나거나
    전체가 반 픽셀 밀린다. 격자 간격으로 나누어떨어지지 않는 크기로 본다.
    """
    h, w = 203, 157

    def const(xs, ys):
        shape = (len(ys), len(xs))
        return np.full(shape, 2.5, np.float32), np.full(shape, -1.25, np.float32)

    for step in (3, 7, 12):
        out, (dx, dy) = _remap_region(_coord_image(h, w), (0, 0, w, h), const, step, None, 1.0)
        edge = max(2, int(min(w, h) * 0.04))
        inner = (slice(edge, -edge), slice(edge, -edge))
        np.testing.assert_allclose(dx[inner], 2.5, atol=1e-5)
        np.testing.assert_allclose(dy[inner], -1.25, atol=1e-5)
        ys, xs = np.mgrid[0:h, 0:w].astype(np.float32)
        sl = (slice(edge, -edge - 3), slice(edge, -edge - 3))
        np.testing.assert_allclose((out[..., 0] - xs)[sl], 2.5, atol=1e-3)
        np.testing.assert_allclose((out[..., 1] - ys)[sl], -1.25, atol=1e-3)


def test_가우시안_장은_제어점_변위를_넘치지_않는다():
    """가중 평균은 제어점 변위와 0의 볼록 결합 — 오버슈트가 없어야 한다."""
    rng = np.random.default_rng(3)
    pts = rng.uniform(50, 250, (20, 2))
    disp = rng.uniform(-6, 6, (20, 2))
    xs = np.arange(0, 300, dtype=np.float32)
    dx, dy = _gaussian_field(xs, xs, pts, disp, 25.0)
    assert float(dx.max()) <= disp[:, 0].max() + 1e-4
    assert float(dx.min()) >= disp[:, 0].min() - 1e-4
    assert float(dy.max()) <= disp[:, 1].max() + 1e-4
    assert float(dy.min()) >= disp[:, 1].min() - 1e-4


def test_블렌딩까지_포함한_출력이_조밀한_계산과_구별되지_않는다():
    img = textured_image(H, W, seed=1)
    img = cv2.GaussianBlur(img, (0, 0), 1.0)  # 사진처럼 약간 부드럽게
    ctrl = _face_ctrl()
    coarse, _ = _remap_region(img, ROI, ctrl.field, _field_step(ctrl.scale), None, 1.0)
    dense, _ = _remap_region(img, ROI, ctrl.field, 1, None, 1.0)
    mse = np.mean((dense.astype(np.float64) - coarse.astype(np.float64)) ** 2)
    psnr = 10 * np.log10(255 ** 2 / max(mse, 1e-12))
    assert psnr > 45, psnr
