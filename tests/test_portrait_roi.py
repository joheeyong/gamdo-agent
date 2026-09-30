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


def _fake_face(mask: np.ndarray, face_w: float = 450.0) -> "ip._SkinFace":
    """사진 전체를 ROI로 쓰는 가짜 얼굴. 윤곽 = 피부 = mask, 금지 구역 없음."""
    return ip._SkinFace(0, 0, mask.shape[1], mask.shape[0], mask.copy(), mask.copy(),
                        np.zeros_like(mask), face_w)


@pytest.fixture
def fake_skin(monkeypatch):
    mask = _face_mask()
    monkeypatch.setattr(ip, "_get_skin_faces", lambda *a, **k: [_fake_face(mask)])
    return mask


# ── 테스트 ──
#
# 예전에는 여기서 ROI 처리 결과가 "예전 전체 처리(양방향 필터 / 짧은 변 기준
# 잡티 탐지)"와 화소 단위로 같은지 봤다. 그 알고리즘 자체가 헤어라인 덧칠·
# 밀랍 피부의 원인이라 주파수 분리·얼굴 폭 기준 탐지로 바꿨고, 기준 구현과의
# 비트 일치 대신 "마스크 밖은 한 화소도 안 바뀐다"는 ROI 처리의 핵심 성질을 본다.
# (새 동작의 품질 검증은 tests/test_skin_retouch.py)


@pytest.mark.parametrize("intensity", [0.2, 0.45, 1.0])
def test_피부_스무딩은_피부_마스크_밖을_건드리지_않는다(fake_skin, intensity):
    arr = _skin_image()
    out = np.array(ip.apply_skin_smoothing(Image.fromarray(arr), intensity))
    assert not np.array_equal(out, arr)
    outside = fake_skin == 0
    assert np.array_equal(out[outside], arr[outside])


def test_피부_스무딩_얼굴이_가장자리에_걸려도_마스크_밖은_그대로(monkeypatch):
    """마스크 상자가 사진 경계에 닿아도 (패딩이 잘려도) 문제없어야 한다."""
    mask = np.zeros((H, W), np.uint8)
    cv2.ellipse(mask, (40, 30), (120, 150), 0, 0, 360, 255, -1)
    monkeypatch.setattr(ip, "_get_skin_faces", lambda *a, **k: [_fake_face(mask)])
    arr = _skin_image(1)
    out = np.array(ip.apply_skin_smoothing(Image.fromarray(arr), 0.6))
    assert not np.array_equal(out, arr)
    assert np.array_equal(out[mask == 0], arr[mask == 0])


@pytest.mark.parametrize("intensity", [0.35, 1.0])
def test_잡티_제거는_붉은_점만_지우고_피부_밖은_그대로(fake_skin, intensity):
    arr = _skin_image()
    out = np.array(ip.apply_blemish_removal(Image.fromarray(arr), intensity))
    assert np.array_equal(out[fake_skin == 0], arr[fake_skin == 0])
    # 합성 잡티(붉은 원)의 붉은 기가 주변 피부 쪽으로 돌아와야 한다
    for cx, cy in [(430, 300), (520, 360), (470, 420), (560, 280)]:
        before = arr[cy, cx].astype(int)
        after = out[cy, cx].astype(int)
        assert after[1] - before[1] > 25, f"({cx},{cy}) 잡티가 남음: {before} → {after}"


def test_잡티_크기_기준은_사진이_아니라_얼굴_폭이다():
    """같은 점(지름 11px)이라도 얼굴이 크면 잡티, 얼굴이 작으면(점이 얼굴 폭의
    7%면) 잡티가 아니다.

    예전에는 사진 짧은 변 기준이라 3413x2560 사진의 폭 450px 얼굴에서
    지름 90px 덩어리(앞머리 끝)까지 잡티로 봤다.
    """
    arr = _skin_image()
    mask = _face_mask()
    spots = [(430, 300), (520, 360), (470, 420), (560, 280)]
    big = ip._detect_blemishes(arr, mask, mask, face_w=450.0, intensity=0.6)
    small = ip._detect_blemishes(arr, mask, mask, face_w=150.0, intensity=0.6)
    assert all(big[y, x] for x, y in spots)
    assert not any(small[y, x] for x, y in spots)


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


@pytest.mark.parametrize("hip_norm", [0.55, 0.02, 0.999, -0.03])  # 음수: 힙이 프레임 밖
def test_다리_늘리기는_힙_위를_건드리지_않고_크기를_유지한다(monkeypatch, hip_norm):
    """예전 행 루프는 힙 아래를 통째로 늘려 코트 자락·손이 늘고 발이 잘렸다.
    새 사상은 허벅지 중간~발목만 늘리므로 힙 위 행은 그대로여야 한다."""
    monkeypatch.setattr(ip, "pose_model_path", lambda: "stub")
    arr = _skin_image(2)
    results = _fake_pose(hip_norm)
    out = np.array(ip.apply_body_reshape(Image.fromarray(arr), leg_stretch=0.4,
                                         cache=_PoseCache(results)))
    assert out.shape == arr.shape
    lms = results.pose_landmarks[0]
    hip_y = (lms[23].y * H + lms[24].y * H) / 2.0
    top = int(max(0.0, hip_y) * 0.9)
    assert np.array_equal(out[:top], arr[:top])


def _leg_map(h=1000, hip=400.0, ankle=850.0, foot=900.0, v=0.45):
    return ip._leg_row_map(h, hip, ankle, foot, v)


def test_다리_사상은_접히지_않고_사진_높이_안에서_발을_지킨다():
    src, first = _leg_map()
    assert np.all(np.diff(src) >= -1e-4)          # 행 순서가 뒤집히지 않음
    assert src[-1] == pytest.approx(999.0, abs=1.0)  # 맨 아래 행 = 원본 맨 아래 (발이 밀려 잘리지 않음)
    assert first >= 400 - 0.05 * 450 - 1          # 힙 위는 그대로


def test_다리_사상은_정강이만_늘리고_발_크기는_그대로다():
    src, _ = _leg_map()
    slope = np.gradient(src)
    assert slope[650:800].mean() < 0.97            # 정강이: 원본 행을 천천히 읽음 = 늘어남
    assert slope[870:890].mean() == pytest.approx(1.0, abs=0.03)  # 발: 배율 1
    assert slope[:350].max() == pytest.approx(1.0, abs=1e-6)      # 상체: 변화 없음


def test_다리_사상은_값이_작으면_아무것도_하지_않는다():
    assert _leg_map(v=0.0) is None
    assert ip._leg_row_map(1000, 400.0, 420.0, 430.0, 0.45) is None  # 다리가 너무 짧음
