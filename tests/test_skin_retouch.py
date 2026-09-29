"""피부 보정(잡티 제거·스무딩)이 자연스러운지 보는 회귀 테스트.

과거 증상 (NASA 우주인 사진, 기본 레시피 잡티 0.35 + 스무딩 0.28):
  1. 앞머리 끝을 따라 피부색 띠가 딱딱한 경계로 칠해지고, 관자놀이 머리카락에
     피부색 얼룩이 번졌다. 원인: 랜드마크 얼굴 윤곽(이마 위쪽이 머리카락 속까지
     올라간다)을 그대로 피부로 보고, 사진 짧은 변 기준 크기(최대 지름 90px)로
     잡티를 찾아 머리카락 끝을 통째로 인페인팅했다.
  2. 모공·주근깨가 사라져 밀랍처럼 보였다.
  3. 모델의 얼굴 영역 보정(face.skin_smoothing 등)이 전역 값 위에 한 번 더 걸렸다.

MediaPipe 없이 랜드마크를 합성해서 본다.
"""

import math

import cv2
import numpy as np
import pytest
from PIL import Image

import image_processor as ip
import jobs
from param_engine import build_params_with_comment

IMG_W, IMG_H = 1200, 1100
FACE_W = 400
SKIN = (205, 160, 135)
HAIR = (150, 105, 60)        # 붉은 기가 도는 갈색 — 예전 탐지(A·B 편차)에 걸리는 색


def _landmarks(face_w: float = FACE_W, cx: float = IMG_W / 2, cy: float = IMG_H * 0.52):
    rx = face_w / 2.0
    ry = rx * 1.32
    pts: dict[int, tuple[float, float]] = {}
    for k, idx in enumerate(ip._SKIN_FACE_OVAL):
        a = -math.pi / 2 + 2 * math.pi * k / len(ip._SKIN_FACE_OVAL)
        pts[idx] = (cx + rx * math.cos(a), cy + ry * math.sin(a))
    pts[234] = (cx - rx, cy)
    pts[454] = (cx + rx, cy)

    def ellipse(ids, ecx, ecy, erx, ery):
        for k, idx in enumerate(ids):
            a = 2 * math.pi * k / len(ids)
            pts[idx] = (ecx + erx * math.cos(a), ecy + ery * math.sin(a))

    ellipse(ip._LEFT_EYE, cx - rx * 0.42, cy - ry * 0.18, rx * 0.26, ry * 0.10)
    ellipse(ip._RIGHT_EYE, cx + rx * 0.42, cy - ry * 0.18, rx * 0.26, ry * 0.10)
    ellipse(ip._LEFT_EYEBROW, cx - rx * 0.44, cy - ry * 0.36, rx * 0.30, ry * 0.05)
    ellipse(ip._RIGHT_EYEBROW, cx + rx * 0.44, cy - ry * 0.36, rx * 0.30, ry * 0.05)
    ellipse(ip._LIPS, cx, cy + ry * 0.52, rx * 0.36, ry * 0.11)
    for idx, (dx, dy) in {
        6: (0, -0.14), 4: (0, 0.22), 1: (0, 0.26), 2: (0, 0.30),
        98: (-0.14, 0.28), 327: (0.14, 0.28), 64: (-0.12, 0.25), 294: (0.12, 0.25),
        48: (-0.10, 0.20), 278: (0.10, 0.20),
    }.items():
        pts[idx] = (cx + rx * dx, cy + ry * dy)
    return lambda i: (int(pts[i][0]), int(pts[i][1]))


def _oval(pt) -> np.ndarray:
    m = np.zeros((IMG_H, IMG_W), np.uint8)
    cv2.fillPoly(m, [np.array([pt(i) for i in ip._SKIN_FACE_OVAL], np.int32)], 255)
    return m


def _portrait(seed: int = 0, face_w: float = FACE_W):
    """피부(질감+얼룩+붉은 잡티) + 앞머리(가닥·피부색 하이라이트 포함) 합성 인물.

    반환: (이미지, 머리카락 마스크, 잡티 중심들, 이마 한가운데 가닥 마스크)
    """
    pt = _landmarks(face_w)
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:IMG_H, 0:IMG_W].astype(np.float32)
    img = np.empty((IMG_H, IMG_W, 3), np.float32)
    img[:] = (120, 120, 125)
    oval = _oval(pt) > 0
    blotch = cv2.GaussianBlur(rng.normal(0, 1, (IMG_H, IMG_W)).astype(np.float32), (0, 0), face_w * 0.02)
    blotch *= 6.0 / (blotch[oval].std() + 1e-6)
    fine = cv2.GaussianBlur(rng.normal(0, 7, (IMG_H, IMG_W, 3)).astype(np.float32), (0, 0), 0.8)
    for c in range(3):
        img[:, :, c][oval] = SKIN[c]
    img[oval] += np.stack([blotch * 1.0, blotch * 0.6, blotch * 0.5], axis=2)[oval] + fine[oval]

    # 붉은 잡티 — 볼
    rx, ry = face_w / 2, face_w / 2 * 1.32
    cx, cy = IMG_W / 2, IMG_H * 0.52
    spots = [(int(cx - rx * 0.45), int(cy + ry * 0.15)), (int(cx + rx * 0.5), int(cy + ry * 0.2))]
    for sx, sy in spots:
        cv2.circle(img, (sx, sy), max(3, int(face_w * 0.012)), (215, 110, 100), -1)

    # 앞머리: 헤어라인 아래로 들쭉날쭉 내려온 가닥들 + 피부색에 가까운 하이라이트 가닥
    hairline = cy - ry * 0.55
    hair = (yy < hairline + 12 * np.sin(xx / 23.0)).astype(np.uint8) * 255
    for k in range(-12, 13):
        x0 = int(cx + k * face_w * 0.04)
        length = int(face_w * (0.06 + 0.05 * ((k * 7) % 3)))
        cv2.line(hair, (x0, int(hairline) - 5), (x0 + 6, int(hairline) + length), 255, 3)
    hair_b = hair > 0
    img[hair_b] = HAIR
    # 하이라이트 가닥 (피부색과 거의 같다)
    hl = np.zeros_like(hair)
    for k in range(-20, 21, 2):
        x0 = int(cx + k * face_w * 0.025)
        cv2.line(hl, (x0, int(hairline) - 120), (x0 + 10, int(hairline) - 8), 255, 2)
    img[(hl > 0) & hair_b] = (210, 165, 130)

    # 이마 한가운데 떨어진 머리카락 한 가닥 (길쭉함 — 잡티가 아니다)
    stray = np.zeros_like(hair)
    cv2.line(stray, (int(cx - rx * 0.1), int(cy - ry * 0.47)),
             (int(cx + rx * 0.15), int(cy - ry * 0.44)), 255, 2)
    img[stray > 0] = HAIR

    img = np.clip(img, 0, 255).astype(np.uint8)
    return img, hair_b, spots, stray > 0, pt


@pytest.fixture
def synthetic(monkeypatch):
    img, hair, spots, stray, pt = _portrait()
    monkeypatch.setattr(ip, "face_model_path", lambda: "stub")
    monkeypatch.setattr(ip, "_face_point_sets", lambda *a, **k: [pt])
    return img, hair, spots, stray, pt


# ── 마스크 ──


def test_피부_마스크는_얼굴_윤곽을_넘지_않고_머리카락을_뺀다(synthetic):
    img, hair, _, _, pt = synthetic
    faces = ip._get_skin_faces(img)
    assert faces and len(faces) == 1
    f = faces[0]
    core = np.zeros((IMG_H, IMG_W), np.uint8)
    core[f.y0:f.y1, f.x0:f.x1] = f.core
    oval = _oval(pt)
    assert cv2.countNonZero(core) > 0.4 * cv2.countNonZero(oval)
    # 윤곽 밖은 0
    assert cv2.countNonZero(cv2.bitwise_and(core, cv2.bitwise_not(oval))) == 0
    # 머리카락(하이라이트 가닥 포함)은 윤곽 안에 있어도 피부가 아니다
    hair_in_oval = hair & (oval > 0)
    assert hair_in_oval.sum() > 1000, "합성 앞머리가 윤곽 안에 들어와야 테스트가 의미 있다"
    assert (core[hair_in_oval] > 0).sum() == 0

    # 블렌딩 알파도 윤곽 밖은 정확히 0
    alpha = np.zeros((IMG_H, IMG_W), np.float32)
    alpha[f.y0:f.y1, f.x0:f.x1] = ip._skin_alpha(f)
    assert alpha[oval == 0].max() == 0.0
    # 머리카락 위의 알파는 사실상 0 (페더 꼬리 2% 미만)
    assert alpha[hair_in_oval].max() < 0.02


# ── 잡티 제거 ──


def test_잡티_제거는_헤어라인의_머리카락과_길쭉한_가닥을_건드리지_않는다(synthetic):
    img, hair, spots, stray, pt = synthetic
    out = np.array(ip.apply_blemish_removal(Image.fromarray(img), 1.0))
    diff = np.abs(out.astype(int) - img.astype(int)).sum(axis=2)

    # 머리카락 (윤곽 밖/안 모두) — 한 화소도 바뀌면 안 된다
    near_hair = cv2.dilate(hair.astype(np.uint8) * 255, np.ones((5, 5), np.uint8)) > 0
    assert diff[near_hair].max() == 0
    # 이마 한가운데 가닥도 그대로
    assert diff[cv2.dilate(stray.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0].max() == 0
    # 진짜 잡티는 지운다 (G 채널이 주변 피부 쪽으로 올라온다)
    for sx, sy in spots:
        assert int(out[sy, sx, 1]) - int(img[sy, sx, 1]) > 20


def test_잡티_제거_총량에_상한이_있다(synthetic):
    """주근깨 밭 / 점이 많은 얼굴에서 얼굴 전체를 덧칠하지 않는다."""
    img, _, _, _, pt = synthetic
    rng = np.random.default_rng(3)
    many = img.copy()
    oval = _oval(pt)
    ys, xs = np.nonzero(cv2.erode(oval, np.ones((81, 81), np.uint8)))
    for k in rng.choice(len(xs), 200, replace=False):
        cv2.circle(many, (int(xs[k]), int(ys[k])), 3, (200, 120, 105), -1)
    out = np.array(ip.apply_blemish_removal(Image.fromarray(many), 1.0))
    changed = (np.abs(out.astype(int) - many.astype(int)).sum(axis=2) > 6).sum()
    assert changed < 0.05 * cv2.countNonZero(oval)


# ── 스무딩 ──


def _bands(img: np.ndarray, mask: np.ndarray, face_w: float) -> tuple[float, float]:
    """피부 안의 (고주파 질감 std, 중주파 얼룩 std) — L 채널."""
    lum = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)[:, :, 0].astype(np.float32)
    s1, s2 = face_w * 0.01, face_w * 0.045
    g1 = cv2.GaussianBlur(lum, (0, 0), s1)
    g2 = cv2.GaussianBlur(lum, (0, 0), s2)
    return float((lum - g1)[mask].std()), float((g1 - g2)[mask].std())


@pytest.mark.parametrize("intensity, min_texture", [(0.28, 0.85), (0.6, 0.72), (1.0, 0.60)])
def test_스무딩은_질감을_남기고_얼룩만_줄인다(synthetic, intensity, min_texture):
    img, _, _, _, pt = synthetic
    out = np.array(ip.apply_skin_smoothing(Image.fromarray(img), intensity))
    f = ip._get_skin_faces(img)[0]
    core = np.zeros((IMG_H, IMG_W), np.uint8)
    core[f.y0:f.y1, f.x0:f.x1] = f.core
    inner = cv2.erode(core, np.ones((41, 41), np.uint8)) > 0

    tex0, mid0 = _bands(img, inner, FACE_W)
    tex1, mid1 = _bands(out, inner, FACE_W)
    assert tex1 / tex0 >= min_texture, f"질감 {tex1 / tex0:.0%}만 남음"
    # 얼룩은 실제로 줄어야 한다 (intensity에 비례)
    assert mid1 / mid0 <= 1.0 - 0.5 * intensity * 0.6, f"얼룩 {mid1 / mid0:.0%}"


def test_스무딩은_머리카락과_윤곽_밖을_건드리지_않는다(synthetic):
    img, hair, _, _, pt = synthetic
    out = np.array(ip.apply_skin_smoothing(Image.fromarray(img), 1.0))
    diff = np.abs(out.astype(int) - img.astype(int)).sum(axis=2)
    assert diff[_oval(pt) == 0].max() == 0
    # 머리카락 위: 페더 꼬리로 1~2 레벨 이상 바뀌면 안 된다
    assert diff[hair].max() <= 3


def test_작은_얼굴은_건드리지_않는다(monkeypatch):
    img, _, _, _, pt = _portrait(face_w=50)
    monkeypatch.setattr(ip, "face_model_path", lambda: "stub")
    monkeypatch.setattr(ip, "_face_point_sets", lambda *a, **k: [pt])
    src = Image.fromarray(img)
    assert ip.apply_skin_smoothing(src, 1.0) is src
    assert ip.apply_blemish_removal(src, 1.0) is src


# ── 파라미터 / 토글 ──

PROFILE = {
    "trendCategory": "clean_minimal",
    "editingStyle": {"filterTendency": "moderate", "skinRetouchLevel": "auto"},
}


@pytest.fixture
def photo():
    rng = np.random.default_rng(0)
    a = np.full((600, 400, 3), 150, np.float32) + rng.normal(0, 6, (600, 400, 3))
    return Image.fromarray(a.clip(0, 255).astype(np.uint8))


def _analysis(face=None):
    face = face if face is not None else {
        "brightness": 0.12, "skin_smoothing": 0.4, "blemish_removal": 0.3,
    }
    return {"subjectType": "인물", "regionParams": {"face": dict(face), "sky": None}}


def test_피부_보정_끄면_전역과_얼굴_영역_모두_0(photo):
    analysis = _analysis()
    params, _ = build_params_with_comment(photo, PROFILE, analysis, skin_retouch_enabled=False)
    assert params["skin_smoothing"] == 0.0
    assert params["blemish_removal"] == 0.0
    face = analysis["regionParams"]["face"]
    assert face["skin_smoothing"] == 0.0 and face["blemish_removal"] == 0.0
    assert face["brightness"] == 0.12  # 톤 보정은 그대로

    # analyze_to_transform_params가 만든 최종 파라미터도 0
    analysis["recommendedParams"] = params
    tp = ip.analysis_to_transform_params(analysis)
    assert tp["skin_smoothing"] == 0.0 and tp["blemish_removal"] == 0.0


def test_피부_보정_기본값은_켜짐(photo):
    params, _ = build_params_with_comment(photo, PROFILE, _analysis({}))
    assert params["skin_smoothing"] > 0.0 and params["blemish_removal"] > 0.0


def test_얼굴_영역_잡티_스무딩은_전역에_합쳐져_두_번_걸리지_않는다(photo):
    base, _ = build_params_with_comment(photo, PROFILE, _analysis({}))
    analysis = _analysis({"brightness": 0.1, "skin_smoothing": 0.9, "blemish_removal": 0.9})
    params, _ = build_params_with_comment(photo, PROFILE, analysis)
    face = analysis["regionParams"]["face"]
    assert face["skin_smoothing"] == 0.0 and face["blemish_removal"] == 0.0
    # 전역은 영역 값 쪽으로 올라가되 상한에서 멈춘다
    assert base["skin_smoothing"] < params["skin_smoothing"] <= 0.45
    assert base["blemish_removal"] < params["blemish_removal"] <= 0.5


def test_얼굴_영역_보정은_요청_원본을_바꾸지_않는다(photo):
    face = {"skin_smoothing": 0.4}
    regions = {"face": face}
    analysis = {"subjectType": "인물", "regionParams": regions}
    build_params_with_comment(photo, PROFILE, analysis, skin_retouch_enabled=False)
    assert face == {"skin_smoothing": 0.4}
    assert regions["face"] is face


def test_피부_보정_토글은_작업_중복키를_바꾼다(monkeypatch):
    """토글을 바꾸고 다시 요청하면 이전 결과(피부 보정 적용본)를 돌려주면 안 된다."""
    import server

    keys = []

    class _Store:
        def submit(self, key, owner, fn):
            keys.append(key)
            return type("S", (), {"job_id": "j" * 16, "status": "queued"})()

    monkeypatch.setattr(server, "_job_store", _Store())
    monkeypatch.setattr(server, "_check_user", lambda ctx, uid: None)
    ctx = type("C", (), {"uid": "u1"})()
    monkeypatch.setattr(server, "_ctx_uid", lambda c: "u1")
    body = {"image_base64": "AAAA", "style_profile": {}, "user_id": "u1"}
    server.api_start_analyze_job(server.AnalyzeAndTransformRequest(**body), ctx)
    server.api_start_analyze_job(
        server.AnalyzeAndTransformRequest(**body, skin_retouch_enabled=False), ctx)
    server.api_start_analyze_job(server.AnalyzeAndTransformRequest(**body), ctx)
    assert keys[0] != keys[1]
    assert keys[0] == keys[2]
    assert isinstance(jobs.dedupe_key("a", True), str)


def test_피부_보정_토글이_분석_경로까지_전달된다(monkeypatch):
    import server

    seen = {}

    def fake_build(img, profile, analysis, reference=None, reshape_enabled=False,
                   skin_retouch_enabled=True):
        seen["skin"] = skin_retouch_enabled
        raise RuntimeError("stop")

    monkeypatch.setattr(server, "build_params_with_comment", fake_build)
    monkeypatch.setattr(server, "transform_photo", lambda **kw: {"subjectType": "인물"})
    monkeypatch.setattr(server, "get_reference_image_paths", lambda uid: [])
    import base64
    import io
    buf = io.BytesIO()
    Image.fromarray(np.full((64, 64, 3), 128, np.uint8)).save(buf, format="JPEG")
    req = server.AnalyzeAndTransformRequest(
        image_base64=base64.b64encode(buf.getvalue()).decode(), skin_retouch_enabled=False)
    resp = server._run_analyze_and_transform(req)
    assert resp.success is False
    assert seen["skin"] is False
