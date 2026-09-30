"""유리·렌즈 반사를 국소 보정으로 어둡게만 하던 문제의 회귀 테스트.

사용자 보고: 창문 너머 도시 사진에서 천장 조명의 유리 반사가 지워지지 않고 이상하게
변했다. 모델은 반사를 "유리에 비친 조명 반사"로 정확히 짚었지만 remove_areas가 아니라
local_* (톤 조정)로 처리했다. 작은 반사·얼룩 영역은 지우기(인페인팅)로 옮긴다.
"""

import base64
import io

import numpy as np
from fastapi.testclient import TestClient
from PIL import Image, ImageDraw

import server
from param_engine import _prune_region_params


def _analysis(reason, area):
    return {
        "autoEdits": {"remove_areas": []},
        "regionParams": {"local_0": {"area": area, "shape": "ellipse", "feather": 0.5,
                                     "reason": reason, "brightness": -0.2}},
    }


def test_small_reflection_region_becomes_a_removal():
    a = _analysis("유리에 비친 조명 반사", {"x": 0.1, "y": 0.1, "width": 0.12, "height": 0.08})
    _prune_region_params(a, {"brightness": 0.0}, False, 0.8)
    assert not (a["regionParams"] or {}).get("local_0")
    assert a["autoEdits"]["remove_areas"] == [{"x": 0.1, "y": 0.1, "width": 0.12, "height": 0.08}]


def test_blown_window_recovery_stays_a_local_edit():
    a = _analysis("왼쪽 창문이 날아감", {"x": 0.1, "y": 0.1, "width": 0.12, "height": 0.08})
    _prune_region_params(a, {"brightness": 0.0}, False, 0.8)
    assert a["regionParams"]["local_0"]["brightness"] == -0.2
    assert a["autoEdits"]["remove_areas"] == []


def test_large_reflection_is_not_inpainted():
    # 인페인팅 한도(한 영역 2%)를 넘으면 지우지 않고 국소 보정으로 둔다
    a = _analysis("유리 반사", {"x": 0.1, "y": 0.1, "width": 0.3, "height": 0.2})
    _prune_region_params(a, {"brightness": 0.0}, False, 0.8)
    assert a["regionParams"]["local_0"]
    assert a["autoEdits"]["remove_areas"] == []


def test_missing_auto_edits_is_created_and_request_not_mutated():
    region = {"area": {"x": 0.5, "y": 0.2, "width": 0.1, "height": 0.05}, "reason": "렌즈 플레어 얼룩",
              "brightness": -0.1}
    a = {"regionParams": {"local_0": region}}
    _prune_region_params(a, {}, False, 0.8)
    assert a["autoEdits"]["remove_areas"][0]["x"] == 0.5
    assert region["brightness"] == -0.1      # 원본 요청 딕셔너리는 그대로


def test_reflection_is_actually_removed_end_to_end(monkeypatch):
    """서버 전체 경로: 하늘 그라데이션 위의 밝은 타원 반사가 사라져야 한다."""
    w, h = 400, 300
    y = np.linspace(0, 1, h)[:, None, None]
    sky = (np.array([150, 160, 170]) * (1 - y) + np.array([200, 205, 210]) * y) * np.ones((1, w, 1))
    img = Image.fromarray(sky.astype(np.uint8))
    ImageDraw.Draw(img).ellipse([60, 40, 120, 70], fill=(250, 245, 230))    # 조명 반사
    buf = io.BytesIO(); img.save(buf, "PNG")

    # 반사에 딱 맞춘 박스 (면적 1.8% — 인페인팅 한도 2% 안)
    area = {"x": 58 / w, "y": 38 / h, "width": 64 / w, "height": 34 / h}
    analysis = {"subjectType": "풍경", "autoEdits": {"remove_areas": []},
                "regionParams": {"local_0": {"area": area, "shape": "ellipse", "feather": 0.5,
                                             "reason": "유리에 비친 조명 반사", "brightness": -0.25}}}
    monkeypatch.setattr(server, "APP_TOKEN", "")
    monkeypatch.setattr(server, "transform_photo", lambda **_: analysis)
    monkeypatch.setattr(server, "get_reference_image_paths", lambda _uid: [])
    monkeypatch.setattr(server, "measure_reference_target", lambda _paths: None)
    monkeypatch.setattr(server, "detect_tilt_angle", lambda _img: None)
    monkeypatch.setattr(server, "estimate_keystone", lambda _img: 0.0)
    r = TestClient(server.app).post("/api/analyze-and-transform",
                                    json={"image_base64": base64.b64encode(buf.getvalue()).decode()})
    body = r.json()
    assert body["success"], body.get("error")
    assert body["analysis"]["autoEdits"]["remove_areas"]
    out = np.asarray(Image.open(io.BytesIO(base64.b64decode(body["image_base64"]))).convert("L"), np.float32)
    spot = out[45:66, 70:111]                  # 반사가 있던 자리 (프레임 그대로)
    around = out[45:66, 150:191]               # 같은 높이의 하늘
    assert abs(float(spot.mean()) - float(around.mean())) < 12   # 반사(≈+50 밝음)가 메워짐


def _sky_with_reflection():
    w, h = 400, 300
    y = np.linspace(0, 1, h)[:, None, None]
    sky = (np.array([150, 160, 170]) * (1 - y) + np.array([200, 205, 210]) * y) * np.ones((1, w, 1))
    img = Image.fromarray(sky.astype(np.uint8))
    ImageDraw.Draw(img).ellipse([60, 40, 110, 72], fill=(250, 245, 230))
    return img


def test_tight_box_covering_top_half_still_removes_whole_reflection():
    """실사진: 모델 박스 높이가 반사의 절반이라 윗부분만 메워지고 반사가 남았다."""
    from image_processor import apply_object_removal
    img = _sky_with_reflection()
    w, h = img.size
    box = {"x": 62 / w, "y": 42 / h, "width": 46 / w, "height": 13 / h}   # 위쪽 절반만
    out = np.asarray(apply_object_removal(img, [box]).convert("L"), np.float32)
    spot = out[42:71, 64:107]
    around = out[42:71, 200:243]
    assert abs(float(spot.mean()) - float(around.mean())) < 10
    assert float(spot.max()) < float(around.max()) + 20


def test_removal_does_not_touch_far_background():
    from image_processor import apply_object_removal
    img = _sky_with_reflection()
    w, h = img.size
    box = {"x": 62 / w, "y": 42 / h, "width": 46 / w, "height": 13 / h}
    src = np.asarray(img, np.int16)
    out = np.asarray(apply_object_removal(img, [box]), np.int16)
    assert np.abs(out[150:, :] - src[150:, :]).max() == 0
    assert np.abs(out[:, 250:] - src[:, 250:]).max() == 0


def test_featureless_box_falls_back_to_modest_expansion():
    """지울 덩어리가 없으면 원래 박스를 조금만 넓힌다 (크게 뭉개지 않는다)."""
    from image_processor import _removal_region_mask
    lum = np.full((300, 400), 128, np.float32)
    m = _removal_region_mask(lum, 100, 100, 140, 120)
    ys, xs = np.nonzero(m)
    assert xs.min() >= 90 and xs.max() <= 150 and ys.min() >= 95 and ys.max() <= 125


def _building_edge_scene():
    """왼쪽 하늘, 오른쪽 어두운 건물 외벽(밝은 창문 띠), 경계에 걸친 조명 반사."""
    w, h = 400, 300
    y = np.linspace(0, 1, h)[:, None]
    a = np.zeros((h, w, 3), np.float32)
    a[:] = (np.array([185, 190, 195]) * (1 - y) + np.array([205, 208, 210]) * y)[:, None, :]
    a[:, 200:] = (80, 90, 105)
    for yy in range(0, h, 20):
        a[yy:yy + 6, 200:] = (120, 140, 170)
    a[:, 200:203] = (60, 60, 60)
    img = Image.fromarray(a.astype(np.uint8))
    ImageDraw.Draw(img).ellipse([178, 60, 228, 88], fill=(248, 245, 235))
    return img


def test_reflection_on_building_edge_does_not_smear_the_building():
    """실사진: 건물 모서리에 걸친 반사를 지우며 넓은 어두운 외벽까지 덩어리로 잡아 뭉갰다."""
    from image_processor import apply_object_removal
    img = _building_edge_scene()
    w, h = img.size
    box = {"x": 180 / w, "y": 62 / h, "width": 46 / w, "height": 13 / h}   # 위쪽 절반
    out = apply_object_removal(img, [box])
    lum = np.asarray(out.convert("L"), np.float32)
    assert lum[64:85, 185:222].max() < 215                     # 반사(≈246)는 사라짐
    d = np.abs(np.asarray(out, np.int16) - np.asarray(img, np.int16)).max(-1)
    ys, xs = np.nonzero(d > 8)
    assert xs.min() >= 172 and xs.max() <= 236                 # 반사 둘레 몇 px 밖은 그대로
    assert ys.min() >= 52 and ys.max() <= 96


def test_mask_does_not_run_along_a_bright_corner_line():
    """실사진: 반사에 붙은 건물 모서리의 가늘고 밝은 세로선을 따라 마스크가 위아래로 뻗어
    모서리가 휘었다. 두툼한 반사만 지우고 가는 선은 마스크에 넣지 않는다."""
    from image_processor import _removal_region_mask
    arr = np.asarray(_building_edge_scene()).copy()
    arr[:, 199:204] = (240, 240, 240)                         # 햇빛 받은 모서리선
    img = Image.fromarray(arr)
    ImageDraw.Draw(img).ellipse([178, 60, 228, 88], fill=(248, 245, 235))
    lum = np.asarray(img.convert("L"), np.float32)
    m = _removal_region_mask(lum, 180, 62, 226, 75)
    assert m[70, 203] and m[82, 203]                          # 반사는 지운다 (박스 아래 절반 포함)
    assert not m[40:54, 196:206].any() and not m[96:, 196:206].any()   # 위·아래 모서리선은 남긴다


def test_warm_reflection_tip_on_bright_sky_is_included():
    """실사진: 박스 밖으로 삐져나온 반사 끝자락이 밝은 하늘 옆이라 밝기로는 안 잡혀 남았다.
    반사는 크림색이라 색 방향으로 잡는다."""
    from image_processor import _removal_region_mask
    import cv2
    a = np.zeros((300, 400, 3), np.uint8)
    a[:] = (232, 234, 236)                                   # 밝고 무채색인 하늘
    a[:, 200:] = (150, 165, 190)                             # 푸른 건물
    img = Image.fromarray(a)
    ImageDraw.Draw(img).ellipse([176, 60, 246, 88], fill=(250, 238, 200))
    rgb = np.asarray(img)
    lum = np.asarray(img.convert("L"), np.float32)
    ab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)[..., 1:].astype(np.float32) - 128.0
    box = (196, 62, 244, 86)                                 # 왼쪽 끝자락(176~196)이 빠진 박스
    assert _removal_region_mask(lum, *box, ab)[70:80, 182:194].all()
    assert not _removal_region_mask(lum, *box)[70:80, 182:194].all()   # 색 없이는 놓친다
    assert not _removal_region_mask(lum, *box, ab)[:, 250:].any()      # 건물 쪽으로 번지지 않음
