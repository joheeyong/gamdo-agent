"""구도를 바꾸는 자동 편집(크롭·인스타 비율·미세 수평)의 서버 게이팅.

과거: 프롬프트가 크롭을 "적극 추천"하라고 해서 실사진 7장 중 6장에 4:5 비율,
5장에 크롭이 붙었고 서버는 그대로 적용했다. 4:3 가로 사진의 좌우 40%가 말없이
잘려 사용자가 잡은 구도가 사라졌다. 지금은:
  - 비율은 인스타가 받지 않는 사진(3:4보다 길거나 1.91:1보다 넓음)에만 자동 적용
  - 모델 크롭은 프레임을 60% 이상 남기면 자동 적용(apply_suggested_crop=True, 앱에서
    원래 구도로 되돌릴 수 있음), 그보다 과하면 suggested_* 제안으로만
  - 크롭을 자동 적용할 때 모델 비율(4:5 등)은 겹쳐 얹지 않는다
  - 수평 보정은 감지기 하한(0.4°) 이상이면 한다 (1°로 올렸다가 사용자 보고로 되돌림)
"""

import base64
import io

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

import server
from image_processor import (
    apply_auto_edits,
    gate_auto_edits,
    required_instagram_ratio,
    suggest_crop,
)


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(server, "APP_TOKEN", "")
    return TestClient(server.app)


def _b64(size) -> str:
    rng = np.random.default_rng(0)
    arr = np.full((size[1], size[0], 3), 140, np.float32) + rng.normal(0, 8, (size[1], size[0], 3))
    buf = io.BytesIO()
    Image.fromarray(arr.clip(0, 255).astype(np.uint8)).save(buf, format="JPEG")
    return base64.b64encode(buf.getvalue()).decode()


def _size_of(b64: str) -> tuple[int, int]:
    return Image.open(io.BytesIO(base64.b64decode(b64))).size


def _analyze(client, monkeypatch, size, auto_edits, subject="풍경", tilt=None):
    analysis = {"subjectType": subject, "autoEdits": dict(auto_edits)}
    monkeypatch.setattr(server, "transform_photo", lambda **_: analysis)
    monkeypatch.setattr(server, "get_reference_image_paths", lambda _uid: [])
    monkeypatch.setattr(server, "measure_reference_target", lambda _paths: None)
    monkeypatch.setattr(server, "detect_tilt_angle", lambda _img: tilt)
    monkeypatch.setattr(server, "estimate_keystone", lambda _img: 0.0)
    r = client.post("/api/analyze-and-transform", json={"image_base64": _b64(size)})
    body = r.json()
    assert body["success"], body.get("error")
    return body["analysis"]["autoEdits"], _size_of(body["image_base64"])


# ── 비율 ──


@pytest.mark.parametrize("size", [(400, 300), (300, 400), (360, 240), (300, 300), (382, 200)])
def test_postable_aspect_needs_no_ratio(size):
    """4:3·3:4·3:2·1:1·1.91:1은 인스타가 그대로 받는다."""
    assert required_instagram_ratio(size) is None


def test_too_tall_and_too_wide_get_nearest_supported_ratio():
    assert required_instagram_ratio((270, 480)) == "3:4"      # 9:16
    assert required_instagram_ratio((300, 450)) == "3:4"      # 2:3
    assert required_instagram_ratio((600, 200)) == "1.91:1"   # 3:1 파노라마


@pytest.mark.parametrize("size", [(400, 300), (300, 400)])
def test_moderate_crop_is_applied_but_model_ratio_is_not_stacked(client, monkeypatch, size):
    """불필요한 가장자리를 정리하는 크롭(면적 0.72)은 적용하되, 4:5를 겹쳐 얹지 않는다."""
    edits, out = _analyze(client, monkeypatch, size, {
        "crop": {"x": 0.12, "y": 0.0, "width": 0.72, "height": 1.0},
        "instagram_ratio": "4:5",
    }, subject="카페/일상")
    assert edits["apply_suggested_crop"] is True
    assert edits["suggested_crop"]["width"] == pytest.approx(0.72)
    assert "suggested_ratio" not in edits and "instagram_ratio" not in edits
    assert out[0] == pytest.approx(size[0] * 0.72, abs=2)
    # 세로 사진은 크롭 결과가 3:4보다 길어지면 인스타 범위로 높이만 다시 맞춘다
    assert out[1] == size[1] or out[0] / out[1] == pytest.approx(0.75, abs=0.01)


@pytest.mark.parametrize("size", [(400, 300), (300, 400)])
def test_aggressive_crop_stays_a_suggestion(client, monkeypatch, size):
    """프레임의 60% 미만만 남기는 크롭은 구도를 통째로 바꾸므로 제안으로만."""
    edits, out = _analyze(client, monkeypatch, size, {
        "crop": {"x": 0.2, "y": 0.1, "width": 0.6, "height": 0.7},
        "instagram_ratio": "4:5",
    }, subject="카페/일상")
    assert out == size
    assert "apply_suggested_crop" not in edits
    assert edits["suggested_crop"]["width"] == pytest.approx(0.6)
    assert edits["suggested_ratio"] == "4:5"


def test_very_tall_photo_is_cropped_to_3_4(client, monkeypatch):
    edits, out = _analyze(client, monkeypatch, (270, 480), {"instagram_ratio": "1:1"})
    assert edits["instagram_ratio"] == "3:4"
    assert out[0] / out[1] == pytest.approx(0.75, abs=0.01)
    assert edits["suggested_ratio"] == "1:1"


def test_tall_portrait_is_not_cropped_vertically(client, monkeypatch):
    """인물은 위아래를 자르지 않는다 — 필요한 비율이 붙어도 프레임 유지."""
    edits, out = _analyze(client, monkeypatch, (270, 480), {}, subject="인물")
    assert out == (270, 480)
    assert edits["allow_vertical_crop"] is False


def test_noop_ratio_suggestion_is_dropped():
    # 인물 3:4 사진에 4:5 → 위아래를 잘라야 해서 실제로는 아무 일도 안 일어난다
    edits = gate_auto_edits({"instagram_ratio": "4:5"}, (300, 400), allow_vertical_crop=False)
    assert "suggested_ratio" not in edits
    # 이미 그 비율인 사진
    edits = gate_auto_edits({"instagram_ratio": "4:5"}, (400, 500), allow_vertical_crop=True)
    assert "suggested_ratio" not in edits


# ── 크롭 제안 ──


def test_tiny_crop_suggestion_is_ignored():
    # 실측: food(0.88x0.85=0.748)는 남고, 가장자리만 깎는 0.95x0.95는 버린다
    assert suggest_crop({"x": 0.02, "y": 0.02, "width": 0.95, "height": 0.95}) is None
    assert suggest_crop({"x": 0.06, "y": 0.1, "width": 0.88, "height": 0.85}) is not None
    # 인물은 높이가 1로 고정되므로 폭 0.9 크롭은 면적 0.9 → 버린다
    assert suggest_crop({"x": 0.05, "y": 0.1, "width": 0.9, "height": 0.8},
                        allow_vertical_crop=False) is None


def test_crop_suggestion_is_clamped_like_apply_smart_crop():
    box = suggest_crop({"x": 0.9, "y": 0.0, "width": 0.1, "height": 0.5})
    assert box["width"] == pytest.approx(0.3) and box["x"] == pytest.approx(0.7)
    assert suggest_crop({"x": "왼쪽"}) is None
    assert suggest_crop({"x": float("nan"), "width": 0.5, "height": 0.5}) is None
    assert suggest_crop(None) is None


def test_suggestion_is_not_applied_unless_requested():
    img = Image.new("RGB", (400, 300))
    edits = {"suggested_crop": {"x": 0.25, "y": 0.0, "width": 0.5, "height": 1.0},
             "suggested_ratio": "4:5"}
    assert apply_auto_edits(img, edits).size == (400, 300)
    applied = apply_auto_edits(img, {**edits, "apply_suggested_crop": True})
    # 200x300 크롭 → 4:5 중앙 크롭 → 200x250
    assert applied.size == (200, 250)


def test_applied_suggestion_without_ratio_rechecks_postable_range():
    """좁은 세로 크롭은 인스타 범위를 벗어나므로 크롭 뒤 프레임으로 3:4를 다시 맞춘다."""
    img = Image.new("RGB", (400, 300))
    edits = {"suggested_crop": {"x": 0.3, "y": 0.0, "width": 0.3, "height": 1.0},
             "apply_suggested_crop": True}
    out = apply_auto_edits(img, edits)
    assert out.size[0] == 120
    assert out.size[0] / out.size[1] == pytest.approx(0.75, abs=0.01)


# ── 수평 ──


def test_small_measured_tilt_is_corrected(client, monkeypatch):
    # 지평선이 있으면 0.6°도 눈에 띈다 — 예전처럼 잡아 준다
    edits, out = _analyze(client, monkeypatch, (400, 300), {"straighten": 2.0}, tilt=0.6)
    assert edits["straighten"] == pytest.approx(0.6)
    assert out != (400, 300)


def test_measured_tilt_below_detector_floor_is_skipped(client, monkeypatch):
    # 측정이 0.3°로 '거의 수평'이라 확신하면 모델 예비값(2°)으로 넘어가지 않는다
    edits, out = _analyze(client, monkeypatch, (400, 300), {"straighten": 2.0}, tilt=0.3)
    assert edits["straighten"] is None
    assert out == (400, 300)


def test_tiny_model_tilt_is_skipped(client, monkeypatch):
    edits, out = _analyze(client, monkeypatch, (400, 300), {"straighten": -0.3})
    assert edits["straighten"] is None
    assert out == (400, 300)


def test_real_tilt_is_still_corrected(client, monkeypatch):
    edits, out = _analyze(client, monkeypatch, (400, 300), {}, tilt=2.5)
    assert edits["straighten"] == 2.5
    assert out != (400, 300)


# ── 옛 기록 호환 ──


def test_apply_transform_replays_old_stored_auto_edits(client):
    """게이팅 전에 저장된 기록(crop·4:5가 이미 적용된 결과)은 그대로 재현돼야 한다.

    앱은 기록을 열 때 저장된 autoEdits를 apply-transform으로 되돌려 보낸다.
    여기서 게이팅하면 기록의 After·저장본이 예전에 보던 것과 달라진다.
    """
    old = {"crop": {"x": 0.14, "y": 0.0, "width": 0.72, "height": 1.0},
           "instagram_ratio": "4:5", "allow_vertical_crop": False, "straighten": 0.6}
    r = client.post("/api/apply-transform", json={
        "image_base64": _b64((400, 300)), "preview": True, "auto_edits": old,
    })
    body = r.json()
    assert body["success"], body.get("error")
    w, h = _size_of(body["image_base64"])
    assert w / h == pytest.approx(0.8, abs=0.02)


def test_apply_transform_with_suggestion_flag(client):
    edits = {"suggested_crop": {"x": 0.25, "y": 0.0, "width": 0.5, "height": 1.0},
             "allow_vertical_crop": True}
    base = client.post("/api/apply-transform", json={
        "image_base64": _b64((400, 300)), "preview": True, "auto_edits": edits}).json()
    assert _size_of(base["image_base64"]) == (400, 300)
    on = client.post("/api/apply-transform", json={
        "image_base64": _b64((400, 300)), "preview": True,
        "auto_edits": {**edits, "apply_suggested_crop": True}}).json()
    # 200x300(2:3)은 인스타 범위 밖이라 크롭 뒤 3:4로 다시 맞춘다
    assert _size_of(on["image_base64"]) == (200, 266)
