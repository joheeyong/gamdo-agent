"""얼굴 랜드마크 인덱스 짝이 맞는지 검증한다.

과거 버그: 눈 확대에서 왼눈 윤곽에 오른눈 홍채 중심이 짝지어져 있었다.
확대의 중심이 반대편 눈에 있으니 방사형 확대가 아니라 "옆으로 끌어당기기"가
되어, 눈이 찢어지고 두 눈 사이의 코까지 딸려가 한쪽 콧구멍만 커졌다.

인덱스 규약은 MediaPipe가 제공하는 연결 상수를 기준으로 확인한다.
"""

import pytest

from image_processor import (
    _LEFT_EYE,
    _LEFT_EYE_CENTER,
    _LEFT_EYE_CONTOUR,
    _RIGHT_EYE,
    _RIGHT_EYE_CENTER,
    _RIGHT_EYE_CONTOUR,
)

mp_conn = pytest.importorskip(
    "mediapipe.tasks.python.vision.face_landmarker"
).FaceLandmarksConnections


def _ids(connections) -> set[int]:
    out: set[int] = set()
    for c in connections:
        out.add(c.start)
        out.add(c.end)
    return out


MP_LEFT = _ids(mp_conn.FACE_LANDMARKS_LEFT_EYE)
MP_RIGHT = _ids(mp_conn.FACE_LANDMARKS_RIGHT_EYE)


def test_eye_contours_match_mediapipe_sides():
    """윤곽 집합이 MediaPipe의 좌우 정의와 같아야 한다."""
    assert set(_LEFT_EYE_CONTOUR) == MP_LEFT
    assert set(_RIGHT_EYE_CONTOUR) == MP_RIGHT


def test_iris_center_belongs_to_same_eye_as_contour():
    """홍채 중심과 윤곽이 같은 눈이어야 한다.

    MediaPipe 규약: 468~472 = LEFT iris, 473~477 = RIGHT iris.
    각 구간의 첫 인덱스가 중심이다.
    """
    assert _LEFT_EYE_CENTER == 468, "LEFT 윤곽에는 LEFT 홍채 중심이 와야 한다"
    assert _RIGHT_EYE_CENTER == 473, "RIGHT 윤곽에는 RIGHT 홍채 중심이 와야 한다"
    assert 468 <= _LEFT_EYE_CENTER <= 472
    assert 473 <= _RIGHT_EYE_CENTER <= 477


def test_left_and_right_eye_sets_are_disjoint():
    """좌우가 겹치면 한쪽 변형이 다른 쪽을 끌고 간다."""
    assert not set(_LEFT_EYE_CONTOUR) & set(_RIGHT_EYE_CONTOUR)
    assert not set(_LEFT_EYE) & set(_RIGHT_EYE)


def test_eye_sets_are_same_size():
    """제어점 개수가 다르면 MLS 워프가 한쪽으로 치우친다."""
    assert len(_LEFT_EYE_CONTOUR) == len(_RIGHT_EYE_CONTOUR)
    assert len(_LEFT_EYE) == len(_RIGHT_EYE)


def test_skin_mask_eye_sets_also_match_mediapipe():
    """피부 마스크가 쓰는 눈 집합도 같은 규약을 따라야 한다.

    (마스크는 제외 영역이라 좌우 라벨이 결과를 바꾸지는 않지만,
     같은 규약을 쓰지 않으면 다음 수정에서 또 헷갈린다.)
    """
    assert set(_LEFT_EYE) | set(_RIGHT_EYE) == MP_LEFT | MP_RIGHT
