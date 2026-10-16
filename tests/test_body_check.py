"""체형 보정 세기 측정과 자체 검사 (body_check).

- 세기: 모델 제안은 상한, 실제 세기는 관절·실루엣으로 잰 비율이 목표에 모자란 만큼
- 자체 검사: 배경 직선이 휘거나 다른 사람이 변형되면 줄이거나 끈다
"""

import types

import cv2
import numpy as np
import pytest
from PIL import Image

import body_check as bc
import image_processor as ip

W, H = 600, 1000


# ── 순수 계산 ──


def test_다리_비율이_이미_충분하면_늘리지_않는다():
    # (950-420)/(950-0) = 0.558 ≥ 0.54
    assert bc.leg_need(0.0, 420.0, 900.0, 950.0) == 0.0


def test_다리가_짧아_보이면_목표_비율까지_늘린다():
    top, hip, ankle, foot = 0.0, 520.0, 900.0, 950.0
    v = bc.leg_need(top, hip, ankle, foot)
    assert 0.0 < v <= 0.5
    if v < 0.5:  # 상한에 걸리지 않았다면 늘린 뒤 비율이 목표와 같아야 한다
        extra = v * ip._LEG_STRETCH_GAIN * (ankle - hip)
        ratio = (foot - hip + extra) / (foot - top + extra)
        assert ratio == pytest.approx(bc._TARGET_LEG_RATIO, abs=1e-6)


def test_다리_보정은_상한을_넘지_않는다():
    assert bc.leg_need(0.0, 800.0, 900.0, 950.0) == pytest.approx(0.5)


def test_다리_측정이_말이_안_되면_None():
    assert bc.leg_need(500.0, 400.0, 300.0, 200.0) is None


def test_허리가_이미_좁으면_줄이지_않는다():
    assert bc.waist_need(0.80) == 0.0


def test_허리는_목표까지_모자란_만큼만_줄인다():
    ratio = 0.89
    v = bc.waist_need(ratio)
    assert 0.0 < v < 0.5
    assert ratio * (1.0 - v * ip._WAIST_SLIM_GAIN) == pytest.approx(bc._TARGET_WAIST_RATIO, abs=1e-6)


@pytest.mark.parametrize("requested,ratio,expect_zero", [
    (0.3, 0.95, True),    # 이미 어깨가 넓은데 넓히라는 제안
    (-0.3, 1.0, True),    # 이미 균형인데 좁히라는 제안
    (0.3, 0.70, False),   # 좁은 어깨 넓히기
    (-0.3, 1.40, False),  # 넓은 어깨 좁히기
])
def test_어깨는_측정이_허락하는_방향만(requested, ratio, expect_zero):
    v = bc.shoulder_allowed(requested, ratio)
    if expect_zero:
        assert v == 0.0
    else:
        assert np.sign(v) == np.sign(requested)
        assert abs(v) <= abs(requested)


def _line_points(n=24):
    t = np.linspace(0.0, 1.0, n)
    return np.stack([np.full(n, 50.0), 20.0 + 160.0 * t], axis=1)  # x=50 수직선


def test_평행이동과_회전은_휨이_아니다():
    pts = _line_points()
    dx = np.full((200, 100), 3.0, np.float32)
    dy = np.zeros_like(dx)
    assert bc.line_bend(pts, np.array([1.0, 0.0]), dx, dy, (0, 0)) == pytest.approx(0.0, abs=1e-5)
    # 아래로 갈수록 고르게 더 밀림 = 직선이 기울 뿐
    dx = np.repeat(np.linspace(0.0, 4.0, 200, dtype=np.float32)[:, None], 100, axis=1)
    assert bc.line_bend(pts, np.array([1.0, 0.0]), dx, dy, (0, 0)) < 0.05


def test_가운데만_밀리면_휨이다():
    pts = _line_points()
    rows = np.arange(200, dtype=np.float32)
    bump = 4.0 * np.exp(-0.5 * ((rows - 100.0) / 20.0) ** 2)
    dx = np.repeat(bump[:, None], 100, axis=1).astype(np.float32)
    dy = np.zeros_like(dx)
    assert bc.line_bend(pts, np.array([1.0, 0.0]), dx, dy, (0, 0)) > 1.5


# ── 합성 인물 ──


def _lm(x, y, v=1.0):
    return types.SimpleNamespace(x=x, y=y, visibility=v)


def _pose(hip_y=0.52, foot_y=0.93, arms_on_waist=False, x_shift=0.0, scale=1.0):
    """정면으로 선 사람 한 명 (정규화 좌표)."""
    cx = 0.5 + x_shift
    lms = [_lm(cx, 0.3, 0.0) for _ in range(33)]
    lms[0] = _lm(cx, 0.12)
    lms[11] = _lm(cx - 0.08 * scale, 0.25)
    lms[12] = _lm(cx + 0.08 * scale, 0.25)
    lms[23] = _lm(cx - 0.05 * scale, hip_y)
    lms[24] = _lm(cx + 0.05 * scale, hip_y)
    lms[27] = _lm(cx - 0.04 * scale, foot_y - 0.03)
    lms[28] = _lm(cx + 0.04 * scale, foot_y - 0.03)
    for i in (29, 30, 31, 32):
        lms[i] = _lm(cx, foot_y)
    if arms_on_waist:
        lms[15] = _lm(cx - 0.095 * scale, 0.42)
        lms[16] = _lm(cx + 0.095 * scale, 0.42)
    return lms


def _person_mask(hip_y=0.52, foot_y=0.93, torso_half=0.09, top=0.06):
    m = np.zeros((H, W), np.float32)
    cx = W // 2
    # 머리
    cv2.ellipse(m, (cx, int((top + 0.07) * H)), (int(0.06 * W), int(0.07 * H)), 0, 0, 360, 1.0, -1)
    # 몸통(어깨~힙)과 다리
    tw = int(torso_half * W)
    m[int(0.2 * H):int(hip_y * H) + 1, cx - tw:cx + tw] = 1.0
    m[int(hip_y * H):int(foot_y * H), cx - int(0.07 * W):cx + int(0.07 * W)] = 1.0
    return m


class _Cache:
    def __init__(self, poses, mask):
        self.poses = poses
        self.mask = mask

    def get_pose_landmarks(self, arr):
        if not self.poses:
            return None
        return types.SimpleNamespace(pose_landmarks=self.poses)

    def get_person_mask(self, arr):
        return self.mask


def _img(mask, line_x=None):
    arr = np.full((H, W, 3), 200, np.uint8)
    arr[mask > 0.5] = (150, 120, 110)
    if line_x is not None:
        arr[150:700, line_x - 2:line_x + 2] = 30  # 진한 수직 문틀
    return Image.fromarray(arr)


def test_다리가_짧아_보이는_사람은_다리를_늘린다():
    mask = _person_mask()
    cal = bc.calibrate_body_reshape(_img(mask), 0.4, 0.0, 0.0, cache=_Cache([_pose()], mask))
    assert cal.measure.leg_ratio is not None and cal.measure.leg_ratio < bc._TARGET_LEG_RATIO
    assert 0.0 < cal.leg_stretch <= 0.4


def test_다리_비율이_좋은_사람은_모델이_제안해도_늘리지_않는다():
    mask = _person_mask(hip_y=0.45)
    cal = bc.calibrate_body_reshape(_img(mask), 0.4, 0.0, 0.0, cache=_Cache([_pose(hip_y=0.45)], mask))
    assert cal.measure.leg_ratio >= bc._TARGET_LEG_RATIO
    assert cal.leg_stretch == 0.0


def test_모델이_제안하지_않으면_측정이_모자라도_늘리지_않는다():
    mask = _person_mask()
    cal = bc.calibrate_body_reshape(_img(mask), 0.0, 0.0, 0.3, cache=_Cache([_pose()], mask))
    assert cal.leg_stretch == 0.0


def test_늘어나는_구간에_다른_사람이_있으면_다리를_끈다():
    mask = _person_mask()
    other = _pose(x_shift=0.4, scale=0.5)
    cal = bc.calibrate_body_reshape(_img(mask), 0.4, 0.0, 0.0, cache=_Cache([_pose(), other], mask))
    assert cal.leg_stretch == 0.0
    assert any("다른 사람" in n for n in cal.notes)


def test_사람이_없으면_체형_보정을_모두_끈다():
    mask = np.zeros((H, W), np.float32)
    cal = bc.calibrate_body_reshape(_img(mask), 0.4, 0.2, 0.3, cache=_Cache([], mask))
    assert (cal.leg_stretch, cal.shoulder_width, cal.waist_slim) == (0.0, 0.0, 0.0)


def test_실루엣이_없으면_측정_없이_낮은_상한만_적용한다():
    cal = bc.calibrate_body_reshape(_img(_person_mask()), 0.45, 0.0, 0.45,
                                    cache=_Cache([_pose()], None))
    assert cal.leg_stretch <= bc._UNMEASURED_CAP
    assert cal.waist_slim <= bc._UNMEASURED_CAP


def test_팔이_허리를_가리면_허리를_재지_않는다():
    mask = _person_mask()
    cal = bc.calibrate_body_reshape(_img(mask), 0.0, 0.0, 0.45,
                                    cache=_Cache([_pose(arms_on_waist=True)], mask))
    assert cal.measure.waist_ratio is None
    assert cal.waist_slim <= bc._UNMEASURED_CAP


# 배경 휨은 인물이 화면을 크게 채울 때 생긴다. 작은 인물은 허리를 최대로 줄여도
# 몸이 2~3px만 움직이고, 실루엣 밖 변위가 좁은 띠에 묶여 배경이 1px도 휘지 않는다.
_BIG = dict(torso_half=0.25)


def _big_pose():
    return _pose(scale=2.6)


def test_허리_옆_배경_직선이_휘면_허리_보정을_줄이거나_끈다():
    mask = _person_mask(**_BIG)
    edge_x = W // 2 + int(0.25 * W)
    clean = bc.calibrate_body_reshape(_img(mask), 0.0, 0.0, 0.5, cache=_Cache([_big_pose()], mask))
    framed = bc.calibrate_body_reshape(_img(mask, line_x=edge_x + 6), 0.0, 0.0, 0.5,
                                       cache=_Cache([_big_pose()], mask))
    assert clean.waist_slim > 0.0
    assert framed.waist_slim < clean.waist_slim
    assert any("배경" in n for n in framed.notes)


def test_작은_인물은_배경이_휘지_않아_그대로_둔다():
    mask = _person_mask()
    edge_x = W // 2 + int(0.09 * W)
    cal = bc.calibrate_body_reshape(_img(mask, line_x=edge_x + 10), 0.0, 0.0, 0.5,
                                    cache=_Cache([_pose()], mask))
    assert cal.waist_slim == pytest.approx(bc.waist_need(cal.measure.waist_ratio))
    assert not any("배경" in n for n in cal.notes)


def test_다른_사람이_허리_변형에_밀리면_허리_보정을_끈다():
    mask = _person_mask(**_BIG)
    edge_x = W // 2 + int(0.25 * W)
    # 허리 바로 옆에 붙어 선 사람 — 보이는 관절 하나가 실루엣 4px 바깥에 있다
    neighbor = [_lm((edge_x + 4) / W, 0.42, 1.0 if i == 0 else 0.0) for i in range(33)]
    cal = bc.calibrate_body_reshape(_img(mask), 0.0, 0.0, 0.5,
                                    cache=_Cache([_big_pose(), neighbor], mask))
    assert cal.waist_slim == 0.0
    assert any("다른 사람" in n for n in cal.notes)


# ── 서버 연결 ──


def test_서버는_다듬은_값을_응답과_분석_기록에_함께_남긴다(monkeypatch):
    import server

    def fake(img, leg, shoulder, waist, cache=None):
        return bc.BodyCalibration(0.1, 0.0, 0.0, notes=["test"])

    monkeypatch.setattr(server, "calibrate_body_reshape", fake)
    params = {"leg_stretch": 0.4, "shoulder_width": 0.2, "waist_slim": 0.3}
    analysis = {"recommendedParams": {"reshapeParams": dict(params)}}
    server._calibrate_body(params, None, None, analysis, "t")
    assert (params["leg_stretch"], params["shoulder_width"], params["waist_slim"]) == (0.1, 0.0, 0.0)
    assert analysis["recommendedParams"]["reshapeParams"]["leg_stretch"] == 0.1


def test_검사가_실패하면_체형만_끄고_나머지는_그대로(monkeypatch):
    import server

    def boom(*a, **k):
        raise RuntimeError("mediapipe down")

    monkeypatch.setattr(server, "calibrate_body_reshape", boom)
    params = {"leg_stretch": 0.4, "shoulder_width": 0.2, "waist_slim": 0.3, "brightness": 0.1}
    server._calibrate_body(params, None, None, {}, "t")
    assert (params["leg_stretch"], params["shoulder_width"], params["waist_slim"]) == (0.0, 0.0, 0.0)
    assert params["brightness"] == 0.1


def test_체형_값이_없으면_검사하지_않는다(monkeypatch):
    import server

    called = []
    monkeypatch.setattr(server, "calibrate_body_reshape", lambda *a, **k: called.append(1))
    server._calibrate_body({"brightness": 0.2}, None, None, {}, "t")
    assert not called
