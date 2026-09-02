"""국소 보정(local_*) 영역과 영역별 대비 피벗 테스트.

모델이 좌표로 짚은 영역을 마스크로 만들고 그 안에만 보정을 거는 경로다.
하늘/얼굴/배경처럼 감지로 찾는 영역과 달리 기하 정보만으로 만들어지므로
MediaPipe 없이 전부 검증할 수 있다.
"""

import cv2
import numpy as np
import pytest
from PIL import Image

from image_processor import (
    _LOCAL_LIMITS,
    apply_regional_transforms,
    build_local_regions,
)

W, H = 800, 600
SIZE = (W, H)
GRAY = 120


def _flat(value: int = GRAY) -> Image.Image:
    return Image.fromarray(np.full((H, W, 3), value, np.uint8))


def _luma(img: Image.Image) -> np.ndarray:
    lab = cv2.cvtColor(np.array(img, np.uint8), cv2.COLOR_RGB2LAB)
    return lab[:, :, 0].astype(np.float32)


AREA = {"x": 0.5, "y": 0.2, "width": 0.25, "height": 0.3}


def test_rect_mask_lands_on_the_given_box():
    masks = build_local_regions(SIZE, {"local_0": {"area": AREA, "shape": "rect"}})
    mask = masks["local_0"]
    assert mask.shape == (H, W)
    xs, ys = np.where(mask > 0)[1], np.where(mask > 0)[0]
    assert (xs.min(), xs.max() + 1) == (int(0.5 * W), int(0.75 * W))
    assert (ys.min(), ys.max() + 1) == (int(0.2 * H), int(0.5 * H))


def test_ellipse_is_the_default_shape():
    """shape를 안 주면 타원 — 사각 마스크는 경계가 눈에 띈다."""
    masks = build_local_regions(SIZE, {"local_0": {"area": AREA}})
    box_area = (0.25 * W) * (0.3 * H)
    filled = cv2.countNonZero(masks["local_0"])
    assert filled == pytest.approx(box_area * np.pi / 4, rel=0.05)


@pytest.mark.parametrize("spec, why", [
    ({}, "area 없음"),
    ({"area": {"x": 0.1, "y": 0.1, "width": 0.02, "height": 0.02}}, "너무 작음"),
    ({"area": {"x": 0.0, "y": 0.0, "width": 1.0, "height": 0.9}}, "너무 큼"),
    ({"area": {"x": "왼쪽", "y": 0.1, "width": 0.2, "height": 0.2}}, "좌표가 숫자가 아님"),
])
def test_bad_areas_are_dropped(spec, why):
    """좌표를 잘못 짚었으면 아무것도 안 하는 게 낫다."""
    assert build_local_regions(SIZE, {"local_0": spec}) == {}, why


def test_at_most_four_areas():
    specs = {f"local_{i}": {"area": AREA} for i in range(7)}
    assert len(build_local_regions(SIZE, specs)) == 4


def test_local_params_are_clamped():
    """모델이 과한 값을 줘도 파라미터별 상한에서 잘려야 한다."""
    regions = build_local_regions(SIZE, {"local_0": {"area": AREA, "shape": "rect"}})
    base = _flat()
    strong = _luma(apply_regional_transforms(
        base, regions, {"local_0": {"area": AREA, "shape": "rect", "brightness": 1.0}}))
    limit = _luma(apply_regional_transforms(
        base, regions,
        {"local_0": {"area": AREA, "shape": "rect",
                     "brightness": _LOCAL_LIMITS["brightness"]}}))
    assert strong.max() == pytest.approx(limit.max(), abs=1.0)


def test_meta_keys_are_not_treated_as_params():
    """area/shape/reason은 보정값이 아니다 — 이것만 있으면 사진이 그대로여야 한다."""
    spec = {"area": AREA, "shape": "rect", "feather": 0.5, "reason": "창문이 날아감"}
    regions = build_local_regions(SIZE, {"local_0": spec})
    base = _flat()
    out = apply_regional_transforms(base, regions, {"local_0": spec})
    assert np.array_equal(np.array(out), np.array(base))


def test_feather_widens_with_the_feather_value():
    """feather 값이 크면 경계가 더 넓게 풀려야 한다."""
    spec = {"area": AREA, "shape": "rect", "brightness": 0.3}
    regions = build_local_regions(SIZE, {"local_0": spec})
    widths = []
    for feather in (0.0, 1.0):
        out = _luma(apply_regional_transforms(
            _flat(), regions, {"local_0": {**spec, "feather": feather}}))
        row = out[int(0.35 * H)]
        row = row - row[0]          # 프레임 왼쪽 끝은 보정 밖 — 그것을 기준으로
        peak = row.max()
        band = np.where((row > peak * 0.1) & (row < peak * 0.9))[0]
        left = band[band < 0.5 * W]
        widths.append(left.max() - left.min() + 1)
    assert widths[1] > widths[0] * 1.5, f"feather 0.0 → {widths[0]}px, 1.0 → {widths[1]}px"


def test_local_wins_over_background():
    """국소 보정은 모델이 좌표까지 짚은 지시라 배경 보정보다 위에 얹혀야 한다."""
    spec = {"area": AREA, "shape": "rect", "brightness": 0.4}
    regions = {
        "background": np.full((H, W), 255, np.uint8),
        **build_local_regions(SIZE, {"local_0": spec}),
    }
    out = _luma(apply_regional_transforms(
        _flat(), regions,
        {"background": {"brightness": -0.4}, "local_0": spec},
    ))
    cx, cy = int(0.625 * W), int(0.35 * H)
    assert out[cy, cx] > GRAY, "국소 영역이 배경 보정에 덮였다"
    assert out[cy, 10] < GRAY, "배경이 보정되지 않았다"


def test_region_contrast_pivots_on_the_region_not_the_frame():
    """평탄한 영역에 대비를 걸면 그 영역의 밝기는 그대로여야 한다.

    전체 평균을 피벗으로 쓰던 시절에는 프레임에 밝은 하늘이 있으면
    어두운 영역이 통째로 14레벨 어두워졌다.
    """
    arr = np.full((H, W, 3), 70, np.uint8)
    arr[: int(H * 0.3)] = 225           # 밝은 하늘이 전체 평균을 끌어올린다
    img = Image.fromarray(arr)
    spec = {"area": {"x": 0.1, "y": 0.6, "width": 0.3, "height": 0.3},
            "shape": "rect", "contrast": 0.3}
    regions = build_local_regions(SIZE, {"local_0": spec})
    out = _luma(apply_regional_transforms(img, regions, {"local_0": spec}))
    before = _luma(img)
    cx, cy = int(0.25 * W), int(0.75 * H)
    assert out[cy, cx] == pytest.approx(before[cy, cx], abs=2.0)
