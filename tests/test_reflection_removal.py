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
