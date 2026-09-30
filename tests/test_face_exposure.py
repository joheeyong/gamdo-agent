"""노출은 화면 평균이 아니라 얼굴·장면의 의도를 따른다.

- 검은 배경 인물: 평균이 어둡다고 brightness·highlights·대비를 올려 얼굴이
  날아가면 안 된다 (실측 walker 피부 L 62 → 시네마틱 86·플래시 88).
- 하이키(흰 접시 음식·흰 배경): 평균이 높다고 노출을 내리거나 흰색을 누르면
  안 된다 (실측 음식 L 80 → 71, p99 99 → 93).
- bright_airy는 기본보다 밝아야 한다.

얼굴 감지(MediaPipe)는 합성 이미지에서 되지 않으므로 알려진 얼굴 마스크로 대신한다.
"""

import numpy as np
import cv2
import pytest
from PIL import Image

import param_engine
from image_processor import analysis_to_transform_params, apply_all_transforms
from param_engine import _FaceTone, build_params_with_comment

W, H = 480, 600
FACE = (240, 170, 60, 78)  # 중심 x, y, 반축 x, y


def _face_mask(w: int = W, h: int = H) -> np.ndarray:
    m = np.zeros((h, w), np.uint8)
    sx, sy = w / W, h / H
    cx, cy, ax, ay = FACE
    cv2.ellipse(m, (int(cx * sx), int(cy * sy)), (int(ax * sx), int(ay * sy)), 0, 0, 360, 255, -1)
    return m > 0


def _portrait(bg: tuple[int, int, int], skin: tuple[int, int, int]) -> Image.Image:
    rng = np.random.default_rng(0)
    arr = np.empty((H, W, 3), np.float32)
    arr[:] = bg
    arr[_face_mask()] = skin
    # 옷 (어두운 파랑)
    arr[300:, 120:360] = (20, 40, 110)
    arr += rng.normal(0, 2.0, arr.shape)
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))


def _lab_l(img: Image.Image) -> np.ndarray:
    return cv2.cvtColor(np.asarray(img.convert("RGB")), cv2.COLOR_RGB2LAB)[..., 0].astype(np.float32) / 255.0


def _face_l(img: Image.Image) -> float:
    return float(np.median(_lab_l(img)[_face_mask(*img.size)]))


@pytest.fixture
def known_face(monkeypatch):
    """_measure_face_tone을 알려진 마스크로 대신한다."""
    def fake(img):
        small = img.convert("RGB").resize((W // 2, H // 2), Image.BILINEAR)
        m = _face_mask(W // 2, H // 2)
        return _FaceTone(small, m, float(np.median(_lab_l(small)[m])))
    monkeypatch.setattr(param_engine, "_measure_face_tone", fake)


def _render(img, style, subject):
    prof = {} if style == "default" else {"trendCategory": style, "styleSource": "manual"}
    analysis = {"subjectType": subject}
    analysis["recommendedParams"], _ = build_params_with_comment(img, prof, analysis)
    params = analysis_to_transform_params(analysis)
    return apply_all_transforms(img, **params), analysis["recommendedParams"]


STYLES = ["default", "cinematic_moody", "flash_digicam", "bright_airy", "clean_minimal", "warm_film"]


@pytest.mark.parametrize("style", STYLES)
def test_low_key_face_not_blown(known_face, style):
    img = _portrait(bg=(4, 4, 6), skin=(190, 145, 125))  # 검은 배경, 피부 L ≈ 0.64
    before = _face_l(img)
    out, params = _render(img, style, "인물")
    after = _face_l(out)
    lift = param_engine._TREND_RECIPES.get(style, {}).get("face_lift", param_engine._FACE_LIFT_CAP)
    # 레시피가 허용한 폭 + 렌더 오차(분할 토닝·그레인 등) 안에서만 밝아진다
    assert after - before <= lift + 0.025, (style, before, after, params)


def test_low_key_face_without_cap_would_blow(monkeypatch):
    """얼굴 정보가 없으면(예전 동작) 같은 사진에서 얼굴이 크게 밝아진다 — 테스트가 의미 있는지 확인."""
    monkeypatch.setattr(param_engine, "_measure_face_tone", lambda img: None)
    img = _portrait(bg=(4, 4, 6), skin=(190, 145, 125))
    out, _ = _render(img, "cinematic_moody", "인물")
    assert _face_l(out) - _face_l(img) > 0.08


def test_dark_face_is_lifted_toward_target(known_face):
    img = _portrait(bg=(90, 90, 95), skin=(120, 88, 72))  # 얼굴 L ≈ 0.42
    out, params = _render(img, "default", "인물")
    assert params["brightness"] > 0.05
    assert _face_l(out) - _face_l(img) > 0.04


def test_high_key_portrait_face_decides_exposure(known_face):
    """흰 배경 인물: 평균이 높아도 얼굴이 적정이면 노출을 내리지 않는다."""
    img = _portrait(bg=(242, 242, 244), skin=(180, 135, 115))
    out, params = _render(img, "default", "인물")
    assert params["brightness"] >= 0.0
    assert params["highlights"] >= 0.0
    assert _face_l(out) >= _face_l(img) - 0.02


def _high_key_food() -> Image.Image:
    rng = np.random.default_rng(1)
    arr = np.full((H, W, 3), 238.0, np.float32)           # 흰 접시·배경
    cv2.ellipse(arr, (240, 300), (200, 150), 0, 0, 360, (250, 250, 250), -1)
    arr[230:370, 160:330] = (120, 80, 50)                 # 케이크
    arr[330:370, 160:330] = (60, 30, 25)
    arr += rng.normal(0, 2.0, arr.shape)
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))


@pytest.mark.parametrize("style", ["default", "bright_airy", "clean_minimal"])
def test_high_key_not_darkened(style):
    img = _high_key_food()
    out, params = _render(img, style, "음식")
    assert params["brightness"] >= 0.0, params
    assert params["highlights"] >= 0.0, params
    l0, l1 = _lab_l(img), _lab_l(out)
    assert l1.mean() >= l0.mean() - 0.02
    # 흰색이 연회색으로 눌리지 않는다 (소프트 필름 커브의 바랜 끝 정도만 허용)
    assert np.percentile(l1, 99) >= np.percentile(l0, 99) - 0.035


def _mid_scene() -> Image.Image:
    rng = np.random.default_rng(2)
    base = np.linspace(60, 200, W, dtype=np.float32)[None, :, None]
    arr = np.broadcast_to(base, (H, W, 3)).copy() * np.array([1.0, 0.95, 0.85], np.float32)
    arr += rng.normal(0, 3.0, arr.shape)
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))


@pytest.mark.parametrize("make", [_mid_scene, _high_key_food])
def test_bright_airy_brighter_than_default(make):
    img = make()
    out_def, p_def = _render(img, "default", "사물")
    out_ba, p_ba = _render(img, "bright_airy", "사물")
    # 하이키 사진은 둘 다 노출을 내리지 않아 brightness가 같을 수 있다 — 밝기는
    # bright 커브가 낸다. 판단은 결과 이미지의 밝기로 한다.
    assert p_ba["brightness"] >= p_def["brightness"]
    assert _lab_l(out_ba).mean() > _lab_l(out_def).mean() + 0.01
