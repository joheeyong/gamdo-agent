"""스타일 프리셋의 톤 모양 — 서로 구분되는 룩, 눌린 흰색·우유빛 검정·분필 얼굴·안개 낀 야경 방지.

리뷰 실측(256px 평균 ΔE): 기본↔웜 필름 3.2, 깔끔↔밝고 화사 3.1, 웜 필름↔골든아워 3.7 —
거의 같은 사진이었다. 한국 감성은 양 끝이 같이 눌려(p5 18·p99 85) 회색 필름,
소프트 파스텔은 흰색이 회색(p99 83), 흑백은 어두운 배경 인물의 얼굴이 하얗게 떴고
(피부 L 63 → 76), 야간 거리 사진은 검정이 들려(p1 0.8 → 9) 안개처럼 보였다.
"""

import cv2
import numpy as np
from PIL import Image

import image_processor as ip
from image_processor import _apply_lab_adjustments, analysis_to_transform_params, apply_all_transforms
from param_engine import _TREND_RECIPES, build_params_with_comment, dark_scene_factor, measure_image_stats
from test_trend_recipes import _scene


def _lowkey_portrait() -> Image.Image:
    """어두운 배경 앞의 인물 — 얼굴(따뜻한 피부) + 짙은 파랑 옷."""
    rng = np.random.default_rng(1)
    arr = np.full((320, 240, 3), 0.03, np.float32)
    arr[60:200, 70:170] = (0.80, 0.58, 0.48)
    arr[200:, 40:200] = (0.10, 0.14, 0.35)
    arr[20:50, 180:230] = (0.90, 0.90, 0.88)
    arr += rng.normal(0, 0.01, arr.shape)
    return Image.fromarray((np.clip(arr, 0, 1) * 255).astype(np.uint8))


def _night_street() -> Image.Image:
    """노이즈가 적은 깨끗한 야경 — low_light(노이즈 기준)로는 잡히지 않는다."""
    rng = np.random.default_rng(2)
    y = np.linspace(0, 1, 240)[:, None]
    x = np.linspace(0, 1, 320)[None, :]
    base = 0.02 + 0.35 * (y * 0.3 + x * 0.7) ** 2
    arr = np.stack([base, base * 0.95, base * 1.1], -1)
    arr[100:130, 40:80] = (0.95, 0.80, 0.50)
    arr[60:90, 200:260] = (0.90, 0.30, 0.25)
    arr[0:40] = 0.01
    arr += rng.normal(0, 0.004, arr.shape)
    return Image.fromarray((np.clip(arr, 0, 1) * 255).astype(np.uint8))


def _render(img, trend, subject="혼합"):
    profile = None if trend == "default" else {"trendCategory": trend, "styleSource": "manual"}
    params, _ = build_params_with_comment(img, profile, {"subjectType": subject})
    out = apply_all_transforms(img, **analysis_to_transform_params({"recommendedParams": params}))
    return params, out


def _lab(img) -> np.ndarray:
    return cv2.cvtColor(np.asarray(img).astype(np.float32) / 255.0, cv2.COLOR_RGB2LAB)


def _mean_de(a, b) -> float:
    return float(np.linalg.norm(_lab(a) - _lab(b), axis=2).mean())


# ── 서로 비슷했던 쌍이 구분된다 ──


def test_formerly_similar_styles_are_distinct():
    # 합성 이미지 두 장의 평균 ΔE 하한 (예전 값). 실사진 10장에서는 5.1 / 5.1 / 4.5 (예전 3.4 / 3.3 / 3.3)
    pairs = [("default", "warm_film", 3.2),        # 예전 2.5
             ("clean_minimal", "bright_airy", 4.5),  # 예전 3.9
             ("warm_film", "golden_hour", 4.3)]      # 예전 3.8
    imgs = [(_scene(), "혼합"), (_lowkey_portrait(), "인물")]
    for a, b, floor in pairs:
        des = [_mean_de(_render(img, a, s)[1], _render(img, b, s)[1]) for img, s in imgs]
        assert np.mean(des) >= floor, (a, b, des)


def test_warm_film_carries_visible_split_toning():
    lo, hi = _TREND_RECIPES["warm_film"]["split"]["shadow"], _TREND_RECIPES["warm_film"]["split"]["highlight"]
    assert lo[1] >= 0.2 and hi[1] >= 0.3
    assert 150 <= lo[0] <= 220          # 청록 그림자 (골든아워의 보라 그림자와 다르다)
    assert 25 <= hi[0] <= 60            # 호박색 하이라이트


def test_clean_minimal_is_neutral_without_grain():
    r = _TREND_RECIPES["clean_minimal"]
    assert r["grain"] == 0.0 and r["temperature"] == 0.0 and "split" not in r


# ── 한국 감성: 흰색은 맑게, 바닥만 띄우고, 채도·혈색은 지킨다 ──


def test_korean_gamsung_keeps_white_ceiling_and_lifts_only_floor():
    assert _TREND_RECIPES["korean_gamsung"]["highlight_ceiling"] >= 0.95
    img = _scene()
    src = _lab(img)[..., 0]
    params, out = _render(img, "korean_gamsung")
    l = _lab(out)[..., 0]
    assert np.percentile(l, 99) >= np.percentile(src, 99) - 1.0   # 예전: −3.6
    assert np.percentile(l, 1) > np.percentile(src, 1) + 3.0      # 바랜 검정은 스타일
    assert params["contrast"] >= -0.08                            # 전역 대비로 양 끝을 누르지 않는다
    assert params["saturation"] >= -0.16                          # 예전: −0.35


def test_korean_gamsung_does_not_cool_warm_skin_much():
    warm = Image.new("RGB", (200, 200), (235, 150, 120))
    params, _ = build_params_with_comment(
        warm, {"trendCategory": "korean_gamsung", "styleSource": "manual"}, {"subjectType": "인물"})
    assert params["temperature"] >= -0.06    # 예전: 코랄 재킷 인물 −0.22 → 피부 회색


# ── 소프트 파스텔: 흰색이 회색이 되지 않고 검정이 우유빛이 되지 않는다 ──


def test_soft_pastel_whites_stay_clean_and_blacks_not_milky():
    assert _TREND_RECIPES["soft_pastel"]["highlight_ceiling"] >= 0.96
    img = _scene()
    src = _lab(img)[..., 0]
    _, out = _render(img, "soft_pastel")
    l = _lab(out)[..., 0]
    assert np.percentile(l, 99) >= np.percentile(src, 99) - 2.0   # 예전: −5.0
    assert np.percentile(l, 1) <= np.percentile(src, 1) + 11.0    # 예전: +13.5


# ── 흑백: 어두운 배경 인물의 얼굴이 분필처럼 뜨지 않는다 ──


def test_bw_face_on_dark_background_not_chalk_white():
    img = _lowkey_portrait()
    face = (slice(80, 180), slice(90, 150))
    _, bw = _render(img, "bw_grain", "인물")
    _, color = _render(img, "default", "인물")
    bw_l = _lab(bw)[..., 0][face].mean()
    color_l = _lab(color)[..., 0][face].mean()
    assert bw_l <= color_l + 6.0   # 예전: +10.1 (81.6 vs 71.5)


def test_mono_mix_barely_lifts_warm_skin():
    skin = Image.new("RGB", (64, 64), (205, 150, 125))
    before = _lab(skin)[..., 0].mean()
    after = _lab(_apply_lab_adjustments(skin, saturation=-1.0))[..., 0].mean()
    # b(노랑) 쪽 믹스만으로 +1.0, a 0.08이 +0.7 — 예전 믹스(a 0.22)는 +2.5
    assert 0.0 <= after - before <= 2.0
    assert ip._MONO_MIX_B >= 0.1                 # 파란 하늘은 여전히 어둡게 누른다


# ── 어두운 장면: 바랜 검정이 안개가 되지 않는다 ──


def test_dark_scene_factor():
    assert dark_scene_factor(measure_image_stats(_night_street())) > 0.8
    assert dark_scene_factor(measure_image_stats(_scene())) == 0.0
    assert dark_scene_factor(measure_image_stats(_scene()), low_light=True) == 1.0


def test_night_scene_keeps_deep_blacks_in_film_styles():
    img = _night_street()
    src_p1 = np.percentile(_lab(img)[..., 0], 1)
    for trend, limit in (("default", 3.0), ("warm_film", 3.0), ("korean_gamsung", 5.0), ("soft_pastel", 7.0)):
        _, out = _render(img, trend)
        p1 = np.percentile(_lab(out)[..., 0], 1)
        assert p1 <= src_p1 + limit, (trend, p1)   # 예전: 기본 +5.4, 파스텔 +13.9


def test_bright_scene_still_gets_faded_film_floor():
    """어두운 장면 완화가 밝은 사진의 소프트 필름 바닥까지 없애지는 않는다."""
    img = _scene()
    params, _ = _render(img, "default")
    assert params["toneCurve"]["strength"] >= 0.5


def test_split_toning_leaves_deepest_black_neutral():
    """쉐도우 색은 깊은 검정에서 사라진다 — 검은 배경이 청록·분홍 판이 되지 않는다."""
    img = Image.new("RGB", (32, 32), (2, 2, 2))
    img.paste((60, 60, 60), (0, 0, 32, 16))
    out = _lab(ip.apply_split_toning(img, 195, 0.3, 45, 0.3))
    chroma = np.hypot(out[..., 1], out[..., 2])
    assert chroma[20:, :].mean() < 1.5          # 거의 검정: 중립
    assert chroma[:12, :].mean() > 3.0          # 짙은 그림자: 색이 입혀진다
