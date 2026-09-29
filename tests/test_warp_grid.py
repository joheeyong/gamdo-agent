"""얼굴·체형 워프의 성긴 격자 변위장 회귀 테스트.

예전 워프는 출력 픽셀 전부에 대해 (제어점 수 × 픽셀 수) 가중치 배열을 만들어
3413x2560 사진에서 수백 MB, 워프 한 번에 1.5초가 걸렸다. 이제 변위장을
성긴 격자(_WARP_GRID_STEP px)에서 계산해 선형 보간한다. 여기서는 그 근사가
  1) 모든 픽셀에서 계산한 값(예전 방식)과 사실상 같고
  2) 반 픽셀 밀림 없이 정렬돼 있으며
  3) grid_step=1이면 예전 계산과 수치적으로 같다
는 것을 확인한다.
"""

import cv2
import numpy as np

from image_processor import _WARP_GRID_STEP, _mls_similarity_warp, _warp_with_mask


def _face_like_controls(cx=300.0, cy=320.0, fw=260.0, fh=340.0):
    """얼굴 보정과 비슷한 제어점 — 윤곽은 안쪽으로, 눈은 바깥으로, 코는 고정."""
    src, dst = [], []
    for k in range(21):  # 윤곽 (슬림: dx만)
        t = np.pi * (0.05 + 0.9 * k / 20)
        for side in (-1, 1):
            px = cx + side * (fw / 2) * np.sin(t)
            py = cy - (fh / 2) * np.cos(t)
            src.append([px, py])
            dst.append([px + (cx - px) * 0.056, py])
    for ex in (cx - 60, cx + 60):  # 눈 (방사형 확대)
        for k in range(16):
            a = 2 * np.pi * k / 16
            px, py = ex + 22 * np.cos(a), cy - 70 + 11 * np.sin(a)
            src.append([px, py])
            dst.append([px + (px - ex) * 0.054, py + (py - (cy - 70)) * 0.054])
    for k in range(24):  # 코 고정점
        p = [cx + (k % 5 - 2) * 9.0, cy - 45.0 + (k // 5) * 22.0]
        src.append(p)
        dst.append(p)
    return np.array(src, np.float32), np.array(dst, np.float32)


def _coord_image(h, w):
    """각 픽셀 값이 자기 (x, y) 좌표인 float32 이미지.

    선형 램프는 쌍선형 보간으로 정확히 복원되므로, 이 이미지를 워프하면
    출력 값이 곧 remap 좌표(= 좌표 + 변위)가 된다.
    """
    ys, xs = np.mgrid[0:h, 0:w].astype(np.float32)
    return np.dstack([xs, ys, np.zeros_like(xs)])


def _reference_maps(h, w, src, dst):
    """예전 구현 그대로의 (N, H*W) 가중치 계산 — 작은 크기에서만 쓴다."""
    ys, xs = np.mgrid[0:h, 0:w].astype(np.float32)
    flat = np.stack([xs, ys], -1).reshape(-1, 2)
    wts = np.zeros((len(src), flat.shape[0]), np.float32)
    for i in range(len(src)):
        d = flat - dst[i]
        wts[i] = 1.0 / (np.sum(d ** 2, axis=1) + 1e-6)
    disp = np.zeros_like(flat)
    for i in range(len(src)):
        disp += wts[i, :, None] * (src[i] - dst[i])[None, :]
    disp /= np.sum(wts, axis=0)[:, None]
    m = flat + disp
    return m[:, 0].reshape(h, w), m[:, 1].reshape(h, w)


def test_기본_격자_간격은_성긴_격자다():
    assert _WARP_GRID_STEP > 1


def test_격자_1이면_예전_계산과_같다():
    h, w = 240, 200
    src, dst = _face_like_controls(cx=100, cy=130, fw=120, fh=160)
    ref_x, ref_y = _reference_maps(h, w, src, dst)
    out = _mls_similarity_warp(_coord_image(h, w), src, dst, grid_step=1)
    # BORDER_REPLICATE 때문에 좌표가 이미지 밖을 가리키는 가장자리는 뺀다
    inner = (slice(4, -4), slice(4, -4))
    np.testing.assert_allclose(out[..., 0][inner], ref_x[inner], atol=2e-3)
    np.testing.assert_allclose(out[..., 1][inner], ref_y[inner], atol=2e-3)


def test_성긴_격자_변위장이_조밀한_계산과_거의_같다():
    h, w = 640, 600
    src, dst = _face_like_controls()
    dense = _mls_similarity_warp(_coord_image(h, w), src, dst, grid_step=1)
    coarse = _mls_similarity_warp(_coord_image(h, w), src, dst)
    inner = (slice(8, -8), slice(8, -8))
    err = np.abs(dense[..., :2] - coarse[..., :2])[inner]
    # 변위 자체는 최대 ~7px. 오차는 제어점 바로 위의 뾰족한 봉우리 끝에서만
    # 커지고(간격 2에서 실측 0.12px), 나머지는 그보다 훨씬 작아야 한다.
    assert float(err.max()) < 0.2, float(err.max())
    assert float(np.percentile(err, 99.9)) < 0.05, float(np.percentile(err, 99.9))
    assert float(err.mean()) < 0.002, float(err.mean())


def test_격자_보간에_반_픽셀_밀림이_없다():
    """제어점이 모두 같은 만큼 움직이면 변위장은 어디서나 그 상수여야 한다.

    격자점을 픽셀 중심 규약과 다르게 두면 이 상수 이동이 가장자리에서
    어긋나거나 전체가 반 픽셀 밀린다.
    """
    h, w = 203, 157  # 격자 간격으로 나누어떨어지지 않는 크기
    rng = np.random.default_rng(3)
    src = rng.uniform(10, 150, (12, 2)).astype(np.float32)
    dst = src - np.float32([2.5, -1.25])
    out = _mls_similarity_warp(_coord_image(h, w), src, dst)
    ys, xs = np.mgrid[0:h, 0:w].astype(np.float32)
    inner = (slice(3, -3), slice(4, -4))
    np.testing.assert_allclose((out[..., 0] - xs)[inner], 2.5, atol=1e-3)
    np.testing.assert_allclose((out[..., 1] - ys)[inner], -1.25, atol=1e-3)


def test_블렌딩까지_포함한_출력이_조밀한_계산과_구별되지_않는다(monkeypatch):
    h, w = 640, 600
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    base = 120 + 50 * np.sin(xx / 23) + 40 * np.cos(yy / 17)
    fine = cv2.GaussianBlur(np.random.default_rng(1).normal(0, 30, (h, w)).astype(np.float32),
                            (0, 0), 1.0)
    img = np.clip(np.dstack([base + fine, base, base - fine]), 0, 255).astype(np.uint8)
    src, dst = _face_like_controls()
    roi = (160, 140, 280, 360)

    coarse = _warp_with_mask(img, src, dst, roi)
    # _warp_with_mask는 격자 간격을 받지 않는다 — 기본 인자를 1로 바꿔 조밀한 기준을 만든다
    monkeypatch.setattr(_mls_similarity_warp, "__defaults__", (1.0, 1))
    dense = _warp_with_mask(img, src, dst, roi)

    mse = np.mean((dense.astype(np.float64) - coarse.astype(np.float64)) ** 2)
    psnr = 10 * np.log10(255 ** 2 / max(mse, 1e-12))
    assert psnr > 45, psnr
