"""하늘 감지가 위치·모양을 보는지 검증한다.

과거 버그: 색만 봤다. 상단 가중치로 걸러 보려 했지만 가중치의 하한(0.4)이
통과 문턱(0.3)보다 커서 아무것도 못 걸렀다 — 프레임 어디에 있든 파란 픽셀이면
하늘로 잡혔고, 파란 셔츠·물·파란 벽에 하늘용 보정(밝기↓ 채도↑)이 걸렸다.
"""

import cv2
import numpy as np
import pytest
from PIL import Image

from image_processor import _SKY_MAX_TOP, detect_regions

H, W = 900, 1200
SKY_RGB = (120, 170, 225)
GRAY_RGB = (110, 108, 105)


def _scene(paint) -> Image.Image:
    arr = np.full((H, W, 3), GRAY_RGB, np.uint8)
    paint(arr)
    return Image.fromarray(arr)


def _sky_ratio(img: Image.Image) -> float:
    return cv2.countNonZero(detect_regions(img)["sky"]) / (H * W)


def test_sky_at_the_top_is_detected():
    img = _scene(lambda a: a.__setitem__(slice(0, int(H * 0.4)), SKY_RGB))
    assert _sky_ratio(img) > 0.3


def test_sky_partly_hidden_at_the_very_top_is_still_detected():
    """처마·나뭇가지가 맨 위를 조금 가려도 하늘은 하늘이다."""
    top = int(H * _SKY_MAX_TOP * 0.5)
    img = _scene(lambda a: a.__setitem__(slice(top, int(H * 0.45)), SKY_RGB))
    assert _sky_ratio(img) > 0.3


@pytest.mark.parametrize("name, paint", [
    ("아래쪽 물·파란 셔츠",
     lambda a: a.__setitem__(slice(int(H * 0.55), None), SKY_RGB)),
    ("가운데 파란 물체",
     lambda a: a.__setitem__(
         (slice(int(H * 0.3), int(H * 0.7)), slice(int(W * 0.3), int(W * 0.7))), SKY_RGB)),
    ("위쪽의 좁은 파란 간판",
     lambda a: a.__setitem__(
         (slice(0, int(H * 0.3)), slice(int(W * 0.4), int(W * 0.5))), SKY_RGB)),
    ("파란 것이 없는 사진", lambda a: None),
])
def test_blue_that_is_not_sky_is_rejected(name, paint):
    assert _sky_ratio(_scene(paint)) == 0.0, name
