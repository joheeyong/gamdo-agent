"""계조 보호 — 하드 클립 대신 끝을 접는지 검증한다.

과거: LAB 채널을 np.clip으로 잘라서, 범위를 넘친 값이 전부 한 값에 붙었다.
실측으로 contrast +0.15만으로도 사진에 따라 화소의 10~37%가 새로 클립됐다.
L 채널에서는 밝은 부분이 평평한 판이 되고, a/b에서는 한 채널만 붙어 색상이 돈다.
"""

import cv2
import numpy as np
import pytest
from PIL import Image

from image_processor import _SOFT_KNEE, _apply_lab_adjustments, _soft_limit


def test_soft_limit_is_monotonic_and_bounded():
    x = np.linspace(-200.0, 500.0, 20001, dtype=np.float32)
    y = _soft_limit(x)
    assert np.all(np.diff(y) >= -1e-6), "순서가 뒤바뀌면 계조가 반전된다"
    assert 0.0 <= float(y.min()) and float(y.max()) <= 255.0


def test_soft_limit_leaves_the_middle_untouched():
    """무릎 안쪽은 건드리지 않아야 한다 — 사진 대부분이 이 구간이다."""
    mid = np.linspace(_SOFT_KNEE + 1, 255.0 - _SOFT_KNEE - 1, 500, dtype=np.float32)
    assert np.allclose(_soft_limit(mid), mid)


def test_overshoot_stays_inside_and_keeps_order():
    """범위를 넘긴 값들도 서로 구분돼야 한다 (하드 클립은 전부 255가 된다).

    아주 큰 값은 tanh가 포화해 255에 닿는다 — 현실의 오버슈트 범위
    (대비·밝기가 만드는 260~300 정도)에서 구분되면 충분하다.
    """
    over = np.array([256.0, 265.0, 280.0, 300.0], dtype=np.float32)
    y = _soft_limit(over)
    assert np.all(y <= 255.0) and np.all(y > 240.0)
    assert len(np.unique(y)) == len(over), "넘친 값들이 한 값으로 뭉쳤다"


def _crushed(img: Image.Image) -> tuple[float, float]:
    arr = np.asarray(img, np.uint8)
    lab = cv2.cvtColor(cv2.cvtColor(arr, cv2.COLOR_RGB2BGR), cv2.COLOR_BGR2LAB)
    L = lab[:, :, 0]
    return float((L <= 1).mean()), float((L >= 254).mean())


def _textured() -> Image.Image:
    yy, xx = np.mgrid[0:400, 0:600].astype(np.float32)
    base = 70 + 55 * np.sin(xx / 70) + 35 * np.cos(yy / 50)
    return Image.fromarray(
        np.clip(np.stack([base, base * 0.95, base * 0.9], -1), 0, 255).astype(np.uint8))


def test_strong_contrast_does_not_block_up_the_blacks(monkeypatch):
    """대비를 세게 걸어도 하드 클립보다 계조가 살아 있어야 한다."""
    import image_processor as ip

    img = _textured()
    kw = dict(brightness=0.0, contrast=0.35, clarity=0.0, dehaze=0.0,
              highlights=0.0, shadows=0.0, saturation=0.0, temperature=0.0,
              tone_curve_preset="linear", tone_curve_strength=0.0, tone_curve_points=None)

    soft_lo, soft_hi = _crushed(_apply_lab_adjustments(img, **kw))
    monkeypatch.setattr(ip, "_soft_limit", lambda x, knee=None: np.clip(x, 0, 255))
    hard_lo, hard_hi = _crushed(_apply_lab_adjustments(img, **kw))

    assert soft_lo <= hard_lo and soft_hi <= hard_hi
    assert (soft_lo + soft_hi) < (hard_lo + hard_hi) * 0.85, (
        f"클립 개선이 미미하다: 하드 {hard_lo + hard_hi:.4f} → 소프트 {soft_lo + soft_hi:.4f}")
