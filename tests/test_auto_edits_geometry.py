"""모델이 준 좌표가 기하 보정 뒤에도 짚은 곳을 가리키는지 검증한다.

과거 버그: autoEdits를 기하 보정 → 요소 제거 → 크롭 순서로 적용했다.
모델은 원본 사진을 보고 0~1 좌표를 짚는데, 키스톤·수평 보정이 먼저 프레임을
회전시키고 잘라내 그 좌표가 다른 곳을 가리켰다 (수평 3° + 키스톤 0.15에서
x=0.10 → 0.12). 작은 제거 박스는 대상을 통째로 놓쳤다.
"""

import numpy as np
import pytest
from PIL import Image

from image_processor import (
    _INPAINT_MAX_AREA,
    _INPAINT_MAX_TOTAL_AREA,
    _map_normalized_box,
    apply_auto_edits,
)

W, H = 1440, 1080
MARK = {"x": 0.68, "y": 0.20, "width": 0.10, "height": 0.10}
GEOMETRY = {"keystone": 0.15, "straighten": 3.0}
RED = (230, 40, 40)


def _marked(box=MARK) -> Image.Image:
    """회색 배경에 빨간 표식 하나 — 좌표가 어디로 갔는지 추적하기 위한 것."""
    arr = np.full((H, W, 3), 110, np.uint8)
    left, top = int(box["x"] * W), int(box["y"] * H)
    right = int((box["x"] + box["width"]) * W)
    bottom = int((box["y"] + box["height"]) * H)
    arr[top:bottom, left:right] = RED
    return Image.fromarray(arr)


def _redness(img: Image.Image) -> float:
    arr = np.array(img, np.int16)
    return float(((arr[:, :, 0] > 170) & (arr[:, :, 1] < 110)).mean())


def test_crop_lands_on_the_marker_after_geometry():
    """기하 보정으로 프레임이 바뀌어도 크롭이 짚은 대상을 담아야 한다."""
    out = apply_auto_edits(_marked(), {**GEOMETRY, "crop": MARK})
    assert _redness(out) > 0.85, f"크롭 결과에서 표식이 {_redness(out):.0%}만 잡혔다"


def test_removal_hits_the_marker_after_geometry():
    """제거는 기하 보정 앞에서 일어나므로 표식이 완전히 사라져야 한다.

    예전 순서에서는 보정된 프레임에 원본 좌표를 그대로 적용해 표식이
    그대로 남았다 (측정: 제거 후에도 빨간 비율 1.12%).
    """
    assert _redness(_marked()) > 0.005
    out = apply_auto_edits(_marked(), {**GEOMETRY, "remove_areas": [MARK]})
    assert _redness(out) < 0.001, f"표식이 {_redness(out):.2%} 남았다"


def test_no_geometry_leaves_coordinates_alone():
    out = apply_auto_edits(_marked(), {"crop": MARK})
    assert _redness(out) > 0.99


def test_map_box_is_identity_without_geometry():
    assert _map_normalized_box(MARK, None, (W, H), (W, H)) == MARK


def test_map_box_returns_none_when_pushed_off_frame():
    """변환으로 프레임 밖으로 나간 박스는 크롭하지 않는다."""
    far_right = np.array([[1.0, 0.0, W * 2.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    assert _map_normalized_box(MARK, far_right, (W, H), (W, H)) is None


def test_map_box_rejects_non_numeric_coordinates():
    assert _map_normalized_box({"x": "왼쪽", "y": 0.1}, None, (W, H), (W, H)) is None


def test_oversized_removal_is_skipped():
    """넓은 영역을 메우면 얼룩이 된다 — 지우지 않고 남기는 편이 낫다."""
    big = {"x": 0.1, "y": 0.1, "width": 0.5, "height": 0.5}
    out = apply_auto_edits(_marked(big), {"remove_areas": [big]})
    assert _redness(out) == pytest.approx(_redness(_marked(big)), abs=0.001)


def test_removal_total_area_is_capped():
    """상한 이하의 작은 영역을 여러 개 보내 우회하지 못해야 한다."""
    side = (_INPAINT_MAX_AREA * 0.9) ** 0.5
    boxes = [{"x": 0.02 + i * (side + 0.01), "y": 0.4, "width": side, "height": side}
             for i in range(6)]
    arr = np.full((H, W, 3), 110, np.uint8)
    for b in boxes:
        arr[int(b["y"] * H):int((b["y"] + b["height"]) * H),
            int(b["x"] * W):int((b["x"] + b["width"]) * W)] = RED
    before = _redness(Image.fromarray(arr))
    after = _redness(apply_auto_edits(Image.fromarray(arr), {"remove_areas": boxes}))
    removed = before - after
    assert removed > 0, "하나도 지우지 않았다"
    assert removed < _INPAINT_MAX_TOTAL_AREA * 1.2, f"누적 {removed:.1%}를 지웠다"


def test_small_removal_still_works():
    small = {"x": 0.4, "y": 0.4, "width": 0.06, "height": 0.06}
    out = apply_auto_edits(_marked(small), {"remove_areas": [small]})
    assert _redness(out) < 0.001
