"""얼굴 보정 테스트용 합성 랜드마크.

실제 MediaPipe 배치를 흉내 낸 3D 머리 모형을 y축으로 돌려 화면에 투영한다.
  - 윤곽(광대 234/454 → 턱끝 152)은 머리 옆면(깊이 0 근처)에,
  - 코·미간·턱끝 같은 중심선은 머리 앞면(깊이 +R)에 있어,
고개를 돌리면 실제 사진처럼 중심선이 먼 쪽으로 치우쳐 보인다.
정확한 해부학이 목적이 아니다 — 보정 코드가 쓰는 인덱스가 그럴듯한 자리에
있고, 좌우 대칭·회전이 실제와 같은 방향으로 작동하는 것이 목적이다.
"""

import math
import types

import numpy as np

from image_processor import (
    _FACE_CONTOUR_LEFT,
    _FACE_CONTOUR_RIGHT,
    _FACE_UPPER_ANCHORS,
    _LEFT_EYE_CENTER,
    _LEFT_EYE_CONTOUR,
    _NOSE_ANCHORS,
    _RIGHT_EYE_CENTER,
    _RIGHT_EYE_CONTOUR,
)

CX, CY = 300.0, 300.0   # 화면상 머리 중심
R = 130.0               # 머리 반폭 (정면 얼굴 폭 260px)
FH = 170.0              # 광대 높이 → 턱끝까지
N_LANDMARKS = 478


def _model() -> dict[int, tuple[float, float, float]]:
    """정면 3D 좌표 (x: 화면 오른쪽 +, y: 아래 +, z: 카메라 쪽 +)."""
    m: dict[int, tuple[float, float, float]] = {}

    def front(x: float) -> float:
        # 머리 단면을 타원으로 — 가장자리(x=±R)는 깊이 0, 중심은 +R
        return R * math.sqrt(max(0.0, 1.0 - (x / R) ** 2))

    # 윤곽: 광대(위) → 턱끝. MediaPipe에서 234 쪽 윤곽은 화면 왼쪽, 454 쪽은 오른쪽.
    n = len(_FACE_CONTOUR_LEFT)
    for side, contour in ((-1.0, _FACE_CONTOUR_LEFT), (1.0, _FACE_CONTOUR_RIGHT)):
        for k, idx in enumerate(contour):
            t = k / (n - 1)                     # 0 = 광대, 1 = 턱끝
            ang = t * math.pi / 2
            x = side * R * math.cos(ang) * (1.0 - 0.25 * t)
            y = -20.0 + (FH + 20.0) * math.sin(ang)
            # 윤곽은 볼 옆면 — 턱끝으로 갈수록 앞으로 나온다
            z = 0.15 * R + t * 0.7 * R
            m[idx] = (x, y, z)
    m[152] = (0.0, FH, 0.85 * R)

    # 이마·관자놀이 (얼굴 타원 윗부분)
    nu = len(_FACE_UPPER_ANCHORS)
    for k, idx in enumerate(_FACE_UPPER_ANCHORS):
        ang = math.pi * k / (nu - 1)            # 0 = 화면 왼쪽 관자놀이, π = 오른쪽
        x = -R * math.cos(ang) * 0.97
        y = -20.0 - 150.0 * math.sin(ang)
        m[idx] = (x, y, front(x) * 0.8)

    # 코 (168 미간도 코 고정점 목록에 있다 — 중심선 기준이라 아래에서 다시 둔다)
    for k, idx in enumerate(_NOSE_ANCHORS):
        x = (k % 5 - 2) * 9.0
        y = -45.0 + (k // 5) * 22.0
        m[idx] = (x, y, R + 15.0 - abs(x))
    m[168] = (0.0, -60.0, R)

    # 눈: RIGHT_EYE(33...)가 화면 왼쪽, LEFT_EYE(362...)가 화면 오른쪽
    for contour, center_idx, ex in (
        (_RIGHT_EYE_CONTOUR, _RIGHT_EYE_CENTER, -55.0),
        (_LEFT_EYE_CONTOUR, _LEFT_EYE_CENTER, 55.0),
    ):
        for k, idx in enumerate(contour):
            a = 2 * math.pi * k / len(contour)
            x, y = ex + 24.0 * math.cos(a), -70.0 + 10.0 * math.sin(a)
            m[idx] = (x, y, front(x) * 0.9)
        m[center_idx] = (ex, -70.0, front(ex) * 0.9 + 3.0)
    # 눈꼬리(33/133, 362/263)는 눈 윤곽의 좌우 끝으로 다시 맞춘다
    for idx, x in ((33, -79.0), (133, -31.0), (362, 31.0), (263, 79.0)):
        m[idx] = (x, -70.0, front(x) * 0.9)

    # 입
    for idx, x, y in ((61, -40.0, 90.0), (291, 40.0, 90.0), (0, 0.0, 78.0), (17, 0.0, 104.0)):
        m[idx] = (x, y, front(x) * 0.95)
    return m


_MODEL = _model()


def face_pt(yaw_deg: float = 0.0, mirror: bool = False):
    """pt(idx) -> (x, y). yaw_deg > 0이면 454 쪽(화면 오른쪽)이 카메라에 가까워진다.

    mirror=True는 화면 좌우를 뒤집은 사진 (같은 얼굴을 거울로 본 것).
    """
    th = math.radians(yaw_deg)
    c, s = math.cos(th), math.sin(th)

    def pt(idx: int) -> tuple[float, float]:
        x, y, z = _MODEL.get(idx, (0.0, 0.0, R))
        # y축 회전: 앞면(z > 0)이 -x 쪽으로 치우친다 → 454 쪽이 넓어 보인다
        xr = x * c - z * s
        if mirror:
            xr = -xr
        return CX + xr, CY + y

    return pt


def landmark_result(pt, w: int, h: int, n: int = N_LANDMARKS):
    """apply_face_reshape가 받는 FaceLandmarker 결과 모양 (정규화 좌표)."""
    lms = []
    for i in range(n):
        x, y = pt(i)
        lms.append(types.SimpleNamespace(x=x / w, y=y / h, z=0.0))
    return types.SimpleNamespace(face_landmarks=[lms])


def textured_image(h: int, w: int, seed: int = 0) -> np.ndarray:
    """어느 방향으로 밀려도 픽셀 값이 바뀌는 무늬 (평탄한 곳이 없다)."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    base = 128 + 60 * np.sin(xx / 7.0) + 50 * np.cos(yy / 5.0)
    noise = rng.normal(0, 20, (h, w)).astype(np.float32)
    img = np.dstack([base + noise, base - noise, 255 - base])
    return np.clip(img, 0, 255).astype(np.uint8)
