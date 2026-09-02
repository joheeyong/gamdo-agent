"""배경 흐림 — 인물 마스크와 흐림 세기 검증.

과거 버그 둘:
1. 인물을 "얼굴 타원 + 그 아래 몸통 타원"으로 근사했다. 팔을 들거나 앉은
   자세에서는 팔다리가 흐려지고 어깨 옆 배경은 선명하게 남았다.
2. 흐림 계수를 sigma가 아니라 커널 크기에 곱했다. OpenCV가 커널에서 유도하는
   sigma는 그 6분의 1이라, 기본 세기(0.19)에서 sigma가 0.8 — 인물 사진마다
   자동으로 켜지는 기능이 사실상 아무 일도 하지 않았다.

인물 분할 모델은 네트워크로 받아 오므로, 모델이 필요한 검사는 로컬에 파일이
있을 때만 돈다. 마스크를 직접 넣어 검증하는 부분은 모델 없이 항상 돈다.
"""

import cv2
import numpy as np
import pytest
from PIL import Image

from image_processor import (
    _BLUR_MIN_SUBJECT,
    _BLUR_SIGMA_RATIO,
    MediaPipeCache,
    _blur_background,
    apply_background_blur,
    person_model_path,
)

H, W = 600, 800


def _textured() -> np.ndarray:
    """사진 비슷한 배열 — 저주파 구조 + 몇 픽셀 단위의 잔디테일.

    순수 노이즈는 3px 커널에도 전부 날아가 지표가 안 되고, 매끄러운 사인파만
    쓰면 반대로 흐림에 거의 반응하지 않는다. 실제 사진처럼 둘을 섞는다.
    """
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    base = 120 + 50 * np.sin(xx / 60) + 30 * np.cos(yy / 45)
    fine = np.random.default_rng(7).normal(0, 40, (H, W)).astype(np.float32)
    fine = cv2.GaussianBlur(fine, (0, 0), 1.2)   # 2~4px 규모의 디테일
    base = base + fine
    return np.clip(np.stack([base, base * 0.95, base * 0.9], -1), 0, 255).astype(np.uint8)


def _half_mask() -> np.ndarray:
    """왼쪽 절반이 인물."""
    alpha = np.zeros((H, W), np.float32)
    alpha[:, : W // 2] = 1.0
    return alpha


def _detail(arr: np.ndarray, region) -> float:
    gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F)[region].var())


class _StubCache:
    """get_person_mask만 흉내내는 캐시 — 모델 없이 합성 경로를 검증한다."""

    def __init__(self, mask):
        self._mask = mask

    def get_person_mask(self, arr_rgb):
        return self._mask


def test_only_the_background_is_blurred():
    arr = _textured()
    out = _blur_background(arr, _half_mask(), 1.0)
    subject = (slice(None), slice(0, W // 2 - 40))
    background = (slice(None), slice(W // 2 + 40, None))
    assert _detail(out, subject) == pytest.approx(_detail(arr, subject), rel=0.02)
    assert _detail(out, background) < _detail(arr, background) * 0.5


@pytest.mark.parametrize("strength", [0.19, 0.5, 1.0])
def test_default_strength_is_actually_visible(strength):
    """기본 세기에서도 배경이 눈에 보이게 바뀌어야 한다.

    예전 방식은 같은 strength에서 배경 변화가 1.2레벨(보이지 않음)이었다.
    """
    arr = _textured()
    out = _blur_background(arr, _half_mask(), strength)
    bg = slice(None), slice(W // 2 + 40, None)
    changed = np.abs(out.astype(np.int16) - arr.astype(np.int16))[bg].mean()
    assert changed > 2.0, f"strength {strength}에서 배경 변화가 {changed:.2f}레벨뿐"


def test_stronger_means_blurrier():
    arr = _textured()
    bg = slice(None), slice(W // 2 + 40, None)
    deltas = [
        np.abs(_blur_background(arr, _half_mask(), s).astype(np.int16)
               - arr.astype(np.int16))[bg].mean()
        for s in (0.2, 0.6, 1.0)
    ]
    assert deltas[0] < deltas[1] < deltas[2], deltas


def test_sigma_scales_with_the_image_not_the_kernel():
    """계수는 sigma에 곱해야 한다. 커널에 곱하면 실제 sigma가 6분의 1이 된다."""
    sigma = min(H, W) * _BLUR_SIGMA_RATIO * 1.0
    kernel_derived = 0.3 * ((max(3, int(min(H, W) * 0.012)) | 1) - 1) * 0.5 + 0.8
    assert sigma > kernel_derived * 3


def test_no_mask_means_no_change():
    """모델을 못 받았으면 예전 타원 근사로 되돌리지 않고 그대로 둔다."""
    img = Image.fromarray(_textured())
    assert apply_background_blur(img, 0.5, _StubCache(None)) is img


def test_photo_without_a_person_is_left_alone():
    """사람 없는 사진에서는 확률이 0에 가깝다 — 면적으로 판단해야 한다."""
    faint = np.full((H, W), _BLUR_MIN_SUBJECT * 0.1, np.float32)
    faint[0, :20] = 0.9   # 최댓값으로 판단하면 오탐하는 잔점
    img = Image.fromarray(_textured())
    assert apply_background_blur(img, 0.5, _StubCache(faint)) is img


def test_subject_is_blurred_when_coverage_is_enough():
    img = Image.fromarray(_textured())
    out = apply_background_blur(img, 0.8, _StubCache(_half_mask()))
    assert out is not img
    bg = slice(None), slice(W // 2 + 60, None)
    assert _detail(np.array(out), bg) < _detail(np.array(img), bg) * 0.6


def test_zero_strength_is_a_no_op():
    img = Image.fromarray(_textured())
    assert apply_background_blur(img, 0.0, _StubCache(_half_mask())) is img


@pytest.mark.skipif(person_model_path() is None,
                    reason="인물 분할 모델이 로컬에 없음 (테스트에서 네트워크를 쓰지 않는다)")
def test_real_segmenter_returns_a_probability_field():
    with MediaPipeCache() as cache:
        mask = cache.get_person_mask(_textured())
    assert mask is not None
    assert mask.dtype == np.float32 and mask.shape == (H, W)
    assert 0.0 <= float(mask.min()) and float(mask.max()) <= 1.0
