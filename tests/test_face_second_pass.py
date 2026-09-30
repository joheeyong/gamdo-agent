"""전신 사진의 작은 얼굴 2차 감지 + 피부 설명문 회귀 테스트.

FaceLandmarker의 얼굴 검출기는 근거리용이라 사진 전체를 작게 줄여 본다.
2560px 전신 사진의 폭 100~200px 얼굴은 거기서 사라져 0개가 나왔고
(glover·hoshide·la·store 실측), 피부 보정·얼굴 밝기·얼굴 워프가 전부 빠졌다.
그런데 설명문은 "피부는 자연스럽게 정리했어요"라고 말했다.

  1) 1차 감지가 0개면 포즈의 코·귀 주변을 잘라 다시 찾고, 좌표를 전체 사진으로 되돌린다
  2) 최소 얼굴 폭(_SKIN_MIN_FACE_PX)보다 작은 얼굴은 버린다
  3) 포즈 크롭이 겹쳐 같은 얼굴을 두 번 넣지 않는다
  4) 얼굴이 없으면 설명문에서 피부 문장을 뺀다 (서버가 감지 결과를 넘긴다)
"""

import types

import numpy as np
import pytest
from PIL import Image

import image_processor as ip
from param_engine import build_params_with_comment, describe_params

pytestmark = pytest.mark.skipif(ip.mp is None, reason="mediapipe 없음")

H, W = 1000, 800


def _lm(x, y, z=0.0):
    return types.SimpleNamespace(x=x, y=y, z=z, visibility=1.0, presence=1.0)


def _pose(nose=(0.5, 0.2), ear_half=0.06, shoulder_half=0.15):
    pts = [_lm(nose[0], nose[1] + 0.2) for _ in range(33)]
    pts[0] = _lm(*nose)
    pts[7] = _lm(nose[0] - ear_half, nose[1])
    pts[8] = _lm(nose[0] + ear_half, nose[1])
    pts[11] = _lm(nose[0] - shoulder_half, nose[1] + 0.1)
    pts[12] = _lm(nose[0] + shoulder_half, nose[1] + 0.1)
    return pts


def _face_in_crop(half_w):
    """크롭 좌표에서 가운데 놓인 얼굴 — 234/454가 광대 양끝, 1이 코끝."""
    lms = [_lm(0.5, 0.5) for _ in range(478)]
    lms[234] = _lm(0.5 - half_w, 0.5)
    lms[454] = _lm(0.5 + half_w, 0.5)
    lms[1] = _lm(0.5, 0.55)
    return lms


class _FakeFace:
    """전체 사진에서는 못 찾고, 크롭에서는 가운데에 얼굴 하나를 찾는 가짜 랜드마커."""

    def __init__(self, half_w=0.15):
        self.half_w = half_w
        self.sizes = []

    def detect(self, image):
        self.sizes.append((image.height, image.width))
        if (image.height, image.width) == (H, W):
            return types.SimpleNamespace(face_landmarks=[])
        return types.SimpleNamespace(face_landmarks=[_face_in_crop(self.half_w)])

    def close(self):
        pass


def _cache(monkeypatch, face, poses):
    cache = ip.MediaPipeCache()
    monkeypatch.setattr(cache, "_get_face_landmarker", lambda: face)
    result = types.SimpleNamespace(pose_landmarks=poses) if poses else None
    monkeypatch.setattr(cache, "get_pose_landmarks", lambda arr: result)
    return cache


def _img():
    rng = np.random.default_rng(0)
    return rng.integers(0, 255, (H, W, 3), dtype=np.uint8)


def test_1차_감지가_0개면_포즈_머리_주변을_잘라_다시_찾고_좌표를_되돌린다(monkeypatch):
    face = _FakeFace()
    cache = _cache(monkeypatch, face, [_pose()])
    res = cache.get_face_landmarks(_img())

    assert res is not None and len(res.face_landmarks) == 1
    # 머리 크기 = max(귀 사이 96px, 어깨 240px × 0.45 = 108px) → 크롭 한 변 432px
    side = int(round(ip._FACE_CROP_HEADS * 108))
    assert face.sizes == [(H, W), (side, side)]
    x0 = round(0.5 * W - side / 2)
    y0 = max(0, round(0.2 * H - side / 2))  # 위쪽 끝에 걸리면 사진 안으로 민다
    lms = res.face_landmarks[0]
    assert lms[234].x == pytest.approx((x0 + 0.35 * side) / W)
    assert lms[454].x == pytest.approx((x0 + 0.65 * side) / W)
    assert lms[1].y == pytest.approx((y0 + 0.55 * side) / H)
    # 사진 좌표로 돌린 얼굴이 원래 머리 자리(코 0.5, 0.2 근처)에 있다
    assert abs(lms[1].x - 0.5) < 0.01 and abs(lms[1].y - 0.2) < 0.05

    # 결과는 캐시된다 — 같은 사진에서 다시 감지하지 않는다
    cache.get_face_landmarks(_img())
    assert len(face.sizes) == 2


def test_2차_감지_얼굴도_기존_소비자가_그대로_쓴다(monkeypatch):
    cache = _cache(monkeypatch, _FakeFace(), [_pose()])
    pts = ip._face_point_sets(_img(), cache=cache)
    assert pts is not None and len(pts) == 1
    face_w = abs(pts[0](454)[0] - pts[0](234)[0])
    assert face_w == pytest.approx(0.3 * 432, abs=2)


def test_최소_얼굴_폭보다_작은_얼굴은_버린다(monkeypatch):
    # 크롭 432px × 0.1 = 43px < 64px
    cache = _cache(monkeypatch, _FakeFace(half_w=0.05), [_pose()])
    assert cache.get_face_landmarks(_img()) is None


def test_머리가_아주_작으면_크롭_감지를_하지_않는다(monkeypatch):
    face = _FakeFace()
    cache = _cache(monkeypatch, face, [_pose(ear_half=0.01, shoulder_half=0.02)])
    assert cache.get_face_landmarks(_img()) is None
    assert face.sizes == [(H, W)]


def test_포즈가_없으면_2차_감지도_없다(monkeypatch):
    face = _FakeFace()
    cache = _cache(monkeypatch, face, None)
    assert cache.get_face_landmarks(_img()) is None
    assert face.sizes == [(H, W)]


def test_겹치는_포즈_크롭에서_같은_얼굴을_두_번_넣지_않는다(monkeypatch):
    cache = _cache(monkeypatch, _FakeFace(), [_pose(), _pose(nose=(0.505, 0.2))])
    res = cache.get_face_landmarks(_img())
    assert len(res.face_landmarks) == 1


def test_떨어진_두_사람은_각각_찾는다(monkeypatch):
    poses = [_pose(nose=(0.3, 0.2)), _pose(nose=(0.7, 0.6))]
    cache = _cache(monkeypatch, _FakeFace(), poses)
    res = cache.get_face_landmarks(_img())
    assert len(res.face_landmarks) == 2


def test_포즈_감지_예외는_얼굴_없음으로_처리한다(monkeypatch):
    cache = ip.MediaPipeCache()
    monkeypatch.setattr(cache, "_get_face_landmarker", lambda: _FakeFace())

    def boom(arr):
        raise RuntimeError("graph error")

    monkeypatch.setattr(cache, "get_pose_landmarks", boom)
    assert cache.get_face_landmarks(_img()) is None


def test_has_retouchable_face는_최소_폭_기준을_따른다(monkeypatch):
    img = Image.fromarray(_img())
    big = _cache(monkeypatch, _FakeFace(), [_pose()])
    assert ip.has_retouchable_face(img, cache=big) is True
    none = _cache(monkeypatch, _FakeFace(), None)
    assert ip.has_retouchable_face(img, cache=none) is False


# ── 설명문 ──

_STATS = {
    "brightness": 0.5, "highlight_clip": 0.0, "sharpness": 0.5,
}


def _params(**kw):
    p = {
        "brightness": 0.0, "highlights": 0.0, "shadows": 0.0, "dehaze": 0.0,
        "contrast": 0.0, "sharpness": 0.0, "saturation": 0.0, "temperature": 0.0,
        "skin_smoothing": 0.3, "grain": 0.0,
        "toneCurve": {"preset": "linear", "strength": 0.0},
    }
    p.update(kw)
    return p


def test_얼굴이_없으면_설명문에_피부를_말하지_않는다():
    text = describe_params(_STATS, "", "인물", _params(), 1.0, face_detected=False)
    assert "피부" not in text
    assert text.endswith("어요")


def test_얼굴이_있거나_모르면_예전처럼_피부를_말한다():
    assert "피부" in describe_params(_STATS, "", "인물", _params(), 1.0, face_detected=True)
    assert "피부" in describe_params(_STATS, "", "인물", _params(), 1.0)


def test_build_params_with_comment가_감지_결과를_설명문에_넘긴다():
    img = Image.fromarray(np.full((96, 96, 3), 128, np.uint8))
    analysis = {"subjectType": "인물"}
    params, with_face = build_params_with_comment(img, None, dict(analysis), face_detected=True)
    params2, no_face = build_params_with_comment(img, None, dict(analysis), face_detected=False)
    assert params["skin_smoothing"] >= 0.1
    assert params == params2  # 값은 그대로, 설명만 바뀐다
    assert "피부" in with_face
    assert "피부" not in no_face


def test_서버가_얼굴_감지_결과를_설명문까지_전달한다(monkeypatch):
    import base64
    import io

    import server

    seen = {}
    real_build = server.build_params_with_comment

    def spy_build(*a, **kw):
        seen["face"] = kw.get("face_detected")
        return real_build(*a, **kw)

    monkeypatch.setattr(server, "build_params_with_comment", spy_build)
    monkeypatch.setattr(server, "has_retouchable_face", lambda img, cache=None: False)
    monkeypatch.setattr(server, "transform_photo", lambda **kw: {"subjectType": "인물"})
    monkeypatch.setattr(server, "get_reference_image_paths", lambda uid: [])
    buf = io.BytesIO()
    Image.fromarray(np.full((64, 64, 3), 128, np.uint8)).save(buf, format="JPEG")
    req = server.AnalyzeAndTransformRequest(image_base64=base64.b64encode(buf.getvalue()).decode())
    resp = server._run_analyze_and_transform(req)
    assert resp.success, resp.error
    assert seen["face"] is False
    assert "피부" not in (resp.params_comment or "")
