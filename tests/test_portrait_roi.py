"""인물 보정의 ROI 처리 회귀 테스트.

피부 스무딩·잡티 제거·다리 늘리기·배경 흐림 합성은 예전에 사진 전체를
처리했다. 이제 결과가 달라질 수 있는 영역만 잘라 처리한다. 잘라낸 경계에서
필터 반경이 모자라면 경계선이 생기므로, 여기서는 예전의 전체 처리 방식과
화소 단위로 같은지 확인한다 (MediaPipe 없이 합성 마스크·랜드마크로).
"""

import types

import cv2
import numpy as np
import pytest
from PIL import Image

import image_processor as ip

H, W = 720, 960


def _skin_image(seed: int = 0) -> np.ndarray:
    """피부톤 바탕 + 질감 + 잡티처럼 붉은 점 몇 개."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    base = np.dstack([
        190 + 20 * np.sin(xx / 70),
        150 + 15 * np.cos(yy / 55),
        130 + 10 * np.sin((xx + yy) / 90),
    ])
    fine = cv2.GaussianBlur(rng.normal(0, 18, (H, W, 3)).astype(np.float32), (0, 0), 1.0)
    img = base + fine
    for cx, cy in [(430, 300), (520, 360), (470, 420), (560, 280)]:
        cv2.circle(img, (cx, cy), 5, (215, 95, 95), -1)
    return np.clip(img, 0, 255).astype(np.uint8)


def _face_mask() -> np.ndarray:
    mask = np.zeros((H, W), np.uint8)
    cv2.ellipse(mask, (490, 350), (150, 190), 0, 0, 360, 255, -1)
    return mask


@pytest.fixture
def fake_skin(monkeypatch):
    mask = _face_mask()
    monkeypatch.setattr(ip, "_get_skin_mask", lambda *a, **k: mask.copy())
    return mask


# ── 예전 구현 (전체 사진 처리) — 비교 기준 ──


def _old_skin_smoothing(arr: np.ndarray, skin_mask: np.ndarray, intensity: float) -> np.ndarray:
    arr_bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    xs, ys = np.where(skin_mask > 0)[1], np.where(skin_mask > 0)[0]
    face_size = max(max(1, int(xs.max() - xs.min())), max(1, int(ys.max() - ys.min())))
    d = int(np.clip(face_size * 0.020 * (0.6 + intensity), 3, 15))
    sigma_color = 18 + intensity * 26
    sigma_space = float(np.clip(face_size * 0.035, 8, 60))
    smoothed = cv2.bilateralFilter(arr_bgr, d, sigma_color, sigma_space)
    feather = max(5, int(face_size * 0.03)) | 1
    alpha = cv2.GaussianBlur(skin_mask, (feather, feather), 0)
    alpha = (alpha.astype(np.float32) / 255.0 * intensity)[:, :, np.newaxis]
    blended = arr_bgr.astype(np.float32) * (1.0 - alpha) + smoothed.astype(np.float32) * alpha
    return cv2.cvtColor(np.clip(blended, 0, 255).astype(np.uint8), cv2.COLOR_BGR2RGB)


def _old_blemish_removal(arr: np.ndarray, skin_mask: np.ndarray, intensity: float) -> np.ndarray:
    arr_bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    blemish = ip._detect_blemishes(arr_bgr, skin_mask, intensity)
    if cv2.countNonZero(blemish) == 0:
        return arr
    inpainted = ip._inpaint_blemishes(arr_bgr, blemish)
    short_side = min(arr_bgr.shape[:2])
    feather = max(5, int(short_side * 0.004)) | 1
    blend = cv2.GaussianBlur(blemish, (feather, feather), 0)
    fill = min(1.0, 0.6 + intensity * 0.4)
    alpha = (blend.astype(np.float32) / 255.0 * fill)[:, :, np.newaxis]
    out = arr_bgr.astype(np.float32) * (1.0 - alpha) + inpainted.astype(np.float32) * alpha
    return cv2.cvtColor(np.clip(out, 0, 255).astype(np.uint8), cv2.COLOR_BGR2RGB)


# ── 테스트 ──


@pytest.mark.parametrize("intensity", [0.2, 0.45, 1.0])
def test_피부_스무딩_ROI_처리가_전체_처리와_같다(fake_skin, intensity):
    arr = _skin_image()
    out = np.array(ip.apply_skin_smoothing(Image.fromarray(arr), intensity))
    ref = _old_skin_smoothing(arr, fake_skin, intensity)
    assert np.array_equal(out, ref)
    # 얼굴 밖은 한 화소도 건드리지 않는다
    far = cv2.dilate(fake_skin, np.ones((61, 61), np.uint8)) == 0
    assert np.array_equal(out[far], arr[far])


def test_피부_스무딩_얼굴이_가장자리에_걸려도_같다(monkeypatch):
    """마스크 상자가 사진 경계에 닿으면 필터 패딩이 잘린다 — 경계 처리도 같아야 한다."""
    mask = np.zeros((H, W), np.uint8)
    cv2.ellipse(mask, (40, 30), (120, 150), 0, 0, 360, 255, -1)
    monkeypatch.setattr(ip, "_get_skin_mask", lambda *a, **k: mask.copy())
    arr = _skin_image(1)
    out = np.array(ip.apply_skin_smoothing(Image.fromarray(arr), 0.6))
    assert np.array_equal(out, _old_skin_smoothing(arr, mask, 0.6))


@pytest.mark.parametrize("intensity", [0.35, 1.0])
def test_잡티_제거_ROI_처리가_전체_처리와_같다(fake_skin, intensity):
    arr = _skin_image()
    ref = _old_blemish_removal(arr, fake_skin, intensity)
    assert not np.array_equal(ref, arr), "합성 잡티가 감지되지 않아 비교가 무의미하다"
    out = np.array(ip.apply_blemish_removal(Image.fromarray(arr), intensity))
    assert np.array_equal(out, ref)


def test_잡티_판정_크기는_잘라낸_조각이_아니라_원본_기준이다():
    """short_side를 넘기면 잘린 조각에서도 원본과 같은 크기 기준을 쓴다."""
    arr_bgr = cv2.cvtColor(_skin_image(), cv2.COLOR_RGB2BGR)
    mask = _face_mask()
    full = ip._detect_blemishes(arr_bgr, mask, 0.6)
    m = ip._blemish_margin(min(H, W))
    x, y, w, h = cv2.boundingRect(mask)
    y1, y2, x1, x2 = max(0, y - m), min(H, y + h + m), max(0, x - m), min(W, x + w + m)
    crop = ip._detect_blemishes(arr_bgr[y1:y2, x1:x2], mask[y1:y2, x1:x2], 0.6,
                                short_side=min(H, W))
    assert np.array_equal(crop, full[y1:y2, x1:x2])


def test_알파_블렌딩이_numpy_계산과_비트_단위로_같다():
    rng = np.random.default_rng(5)
    fg = rng.integers(0, 256, (517, 389, 3), dtype=np.uint8)  # 묶음 크기로 안 나눠지는 높이
    bg = cv2.GaussianBlur(fg, (0, 0), 3)
    alpha = cv2.GaussianBlur(rng.random((517, 389), dtype=np.float32), (0, 0), 4)
    ref = np.clip(
        fg.astype(np.float32) * alpha[:, :, None]
        + bg.astype(np.float32) * (1.0 - alpha[:, :, None]), 0, 255,
    ).astype(np.uint8)
    assert np.array_equal(ip._alpha_blend(fg, bg, alpha), ref)


# ── 다리 늘리기 ──


def _fake_pose(hip_y: float) -> types.SimpleNamespace:
    lms = [types.SimpleNamespace(x=0.5, y=0.2, visibility=1.0) for _ in range(33)]
    lms[23] = types.SimpleNamespace(x=0.45, y=hip_y, visibility=1.0)
    lms[24] = types.SimpleNamespace(x=0.55, y=hip_y + 0.004, visibility=1.0)
    lms[27] = types.SimpleNamespace(x=0.45, y=0.95, visibility=0.9)
    lms[28] = types.SimpleNamespace(x=0.55, y=0.95, visibility=0.9)
    return types.SimpleNamespace(pose_landmarks=[lms])


class _PoseCache:
    def __init__(self, results):
        self.results = results

    def get_pose_landmarks(self, arr):
        return self.results


def _old_leg_stretch(arr: np.ndarray, hip_y: float, leg_stretch: float) -> np.ndarray:
    h, w = arr.shape[:2]
    stretch_factor = 1.0 + leg_stretch * 0.25
    map_y = np.zeros((h, w), dtype=np.float32)
    map_x = np.arange(w, dtype=np.float32)[np.newaxis, :].repeat(h, axis=0)
    hip_y_int = int(hip_y)
    for row in range(h):
        if row <= hip_y_int:
            map_y[row, :] = row
        else:
            map_y[row, :] = min(h - 1, hip_y + (row - hip_y) / stretch_factor)
    mask = np.zeros((h, w), dtype=np.float32)
    mask[hip_y_int:, :] = 1.0
    transition = max(10, int(h * 0.03))
    for row in range(max(0, hip_y_int - transition), min(h, hip_y_int + transition)):
        t = (row - (hip_y_int - transition)) / (2 * transition)
        mask[row, :] = max(0.0, min(1.0, t))
    stretched = cv2.remap(arr, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    m3 = mask[:, :, np.newaxis]
    return (arr.astype(np.float32) * (1 - m3) + stretched.astype(np.float32) * m3).astype(np.uint8)


@pytest.mark.parametrize("hip_norm", [0.55, 0.02, 0.999, -0.03])  # 음수: 힙이 프레임 밖
def test_다리_늘리기가_예전_행_루프와_같다(monkeypatch, hip_norm):
    monkeypatch.setattr(ip, "pose_model_path", lambda: "stub")
    arr = _skin_image(2)
    results = _fake_pose(hip_norm)
    out = np.array(ip.apply_body_reshape(Image.fromarray(arr), leg_stretch=0.4,
                                         cache=_PoseCache(results)))
    lms = results.pose_landmarks[0]
    hip_y = (lms[23].y * H + lms[24].y * H) / 2.0
    assert np.array_equal(out, _old_leg_stretch(arr, hip_y, 0.4))
