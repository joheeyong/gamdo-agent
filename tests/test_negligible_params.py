"""거의 아무것도 안 하는 값 — 0으로 정리하고, 약한 값이 절삭 때문에 '다른 일'을 하지 않게.

실측(실사진 7장): auto_wb 0.05~0.06은 렌더러의 1% 데드존에 걸려 ΔE 0, 선명감 ±0.016은
ΔE 0.1~0.2, 선명도 0.04의 '효과'는 대부분 uint8 절삭으로 화면 전체가 0.5레벨 어두워진
것이었다. 그런데도 앱의 '적용된 변형'에는 전부 칩으로 떴다.
"""

import numpy as np
import pytest
from PIL import Image

import image_processor as ip
from param_engine import (
    _MIN_BLEND_EFFECT,
    _MIN_EFFECT,
    _drop_negligible,
    build_params_with_comment,
)


def _noise_img(seed=0, size=(160, 240), cast=(1.0, 1.0, 1.0)) -> Image.Image:
    rng = np.random.default_rng(seed)
    h, w = size
    y = np.linspace(0, 1, h)[:, None, None]
    x = np.linspace(0, 1, w)[None, :, None]
    base = 30 + 190 * (0.6 * y + 0.4 * x) + rng.normal(0, 12, (h, w, 3))
    arr = base * np.array(cast)[None, None, :]
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))


def _params(**over):
    p = {k: 0.0 for k in _MIN_EFFECT}
    p.update(auto_wb=0.0,
             toneCurve={"preset": "soft_film", "strength": 0.5, "points": None},
             splitToning={"shadow": {"hue": 30, "strength": 0.0},
                          "highlight": {"hue": 200, "strength": 0.0}})
    p.update(over)
    return p


# ── 임계값 정리 ──

def test_values_below_the_floor_become_zero_and_the_rest_stay():
    over = {}
    for k, floor in _MIN_EFFECT.items():
        over[k] = -floor * 0.6 if k != "dehaze" else floor * 0.6
    p = _params(**over)
    _drop_negligible(p, _noise_img())
    assert all(p[k] == 0.0 for k in _MIN_EFFECT)

    kept = {k: (-floor if k != "dehaze" else floor) for k, floor in _MIN_EFFECT.items()}
    p = _params(**kept)
    _drop_negligible(p, _noise_img())
    assert all(p[k] == kept[k] for k in _MIN_EFFECT), "임계값 이상은 부호·크기 그대로"


def test_recent_server_values_are_dropped():
    """실서버 응답에서 본 값들: 선명감 ±0.016, 선명도 0.04, 채도 0.016, 색온도 −0.023."""
    p = _params(clarity=-0.016, sharpness=0.04, saturation=0.016, temperature=-0.023,
                brightness=0.01, shadows=-0.007, highlights=0.28, vignette=0.096)
    _drop_negligible(p, _noise_img())
    assert p["clarity"] == p["sharpness"] == p["saturation"] == p["temperature"] == 0.0
    assert p["brightness"] == p["shadows"] == 0.0
    assert p["highlights"] == 0.28 and p["vignette"] == 0.096


def test_blend_strengths_and_reference_curve():
    p = _params(toneCurve={"preset": "soft_film", "strength": _MIN_BLEND_EFFECT * 0.5, "points": None},
                splitToning={"shadow": {"hue": 30, "strength": 0.02},
                             "highlight": {"hue": 200, "strength": 0.2}})
    _drop_negligible(p, _noise_img())
    assert p["toneCurve"]["strength"] == 0.0
    assert p["splitToning"]["shadow"]["strength"] == 0.0
    assert p["splitToning"]["highlight"]["strength"] == 0.2

    # 레퍼런스 곡선은 제어점 자체가 효과다 — 건드리지 않는다
    ref = _params(toneCurve={"preset": "reference", "strength": 0.01, "points": [(0, 0), (1, 1)]})
    _drop_negligible(ref, _noise_img())
    assert ref["toneCurve"]["strength"] == 0.01


def test_auto_wb_follows_the_renderer_deadzone():
    """렌더러가 건너뛰는 세기(모든 게인 1% 안쪽)만 0으로 — 색이 틀어진 사진은 약해도 남긴다."""
    neutral = _noise_img()
    p = _params(auto_wb=0.06)
    _drop_negligible(p, neutral)
    assert p["auto_wb"] == 0.0
    assert ip.apply_all_transforms(neutral, auto_wb=0.06).tobytes() == neutral.tobytes()

    tinted = _noise_img(cast=(1.0, 0.85, 0.65))
    p = _params(auto_wb=0.3)
    _drop_negligible(p, tinted)
    assert p["auto_wb"] == 0.3
    assert ip.apply_auto_white_balance(tinted, 0.3).tobytes() != tinted.tobytes()


@pytest.mark.parametrize("seed", range(4))
def test_engine_output_has_no_negligible_values(seed):
    img = _noise_img(seed, cast=(1.0, 0.95 + 0.02 * seed, 0.9))
    for trend in ("", "warm_film", "clean_bright"):
        params, _ = build_params_with_comment(img, {"trendCategory": trend})
        for k, floor in _MIN_EFFECT.items():
            assert params[k] == 0.0 or abs(params[k]) >= floor, (trend, k, params[k])


# ── 약한 값이 절삭 때문에 화면 전체를 바꾸지 않는다 ──

def _mean_shift(a: Image.Image, b: Image.Image) -> float:
    return float(np.abs((np.asarray(b, np.float32) - np.asarray(a, np.float32)).mean(axis=(0, 1))).max())


def test_weak_sharpness_does_not_darken_the_frame():
    img = _noise_img()
    assert _mean_shift(img, ip.apply_sharpness(img, 0.04)) < 0.05   # 예전 0.5
    assert _mean_shift(img, ip.apply_sharpness(img, -0.04)) < 0.05


def test_white_balance_rounds_instead_of_truncating():
    img = _noise_img(cast=(1.0, 0.97, 0.93))
    out = np.asarray(ip.apply_auto_white_balance(img, 0.5), np.float32)
    gains = [max(0.75, min(1.25, 1.0 + (g - 1.0) * 0.5)) for g in ip.estimate_illuminant(img)]
    src = np.asarray(img, np.float32)
    for c in range(3):
        expected = np.clip(np.rint(src[..., c] * np.float32(gains[c])), 0, 255)
        assert abs(out[..., c].mean() - expected.mean()) < 0.02   # 절삭이면 ≈ −0.5


@pytest.mark.parametrize("step", [
    lambda im: ip.apply_hsl_adjust(im, {"purple": {"hue": 0.0, "saturation": 0.02, "lightness": 0.0}}),
    lambda im: ip.apply_split_toning(im, 30, 0.0101, 0, 0.0),
])
def test_color_steps_round_trip_without_loss(step):
    """영향이 거의 없는 조정은 결과도 거의 그대로여야 한다 (예전: 왕복만으로 ΔE 0.2~0.6)."""
    img = _noise_img()   # 따뜻한 회색 계열 — purple 채널이 닿지 않는다
    diff = np.abs(np.asarray(step(img), np.float32) - np.asarray(img, np.float32))
    assert diff.mean() < 0.15 and _mean_shift(img, step(img)) < 0.1


def test_vignette_leaves_the_center_alone():
    img = _noise_img(size=(200, 200))
    out = np.asarray(ip.apply_vignette(img, 0.1), np.int16)
    src = np.asarray(img, np.int16)
    c = slice(70, 130)
    assert np.abs(out[c, c] - src[c, c]).max() <= 1   # 예전: 가운데까지 평균 0.5 어두워짐


def test_soft_limit_is_continuous_at_the_edge():
    """화소 하나가 255를 살짝 넘는다고 무릎 구간 전체가 튀면 안 된다 (예전: 240→245.6)."""
    base = np.array([231.0, 240.0, 243.0, 250.0, 254.0], np.float32)
    a = ip._soft_limit(np.append(base, 255.0).astype(np.float32))
    b = ip._soft_limit(np.append(base, 255.01).astype(np.float32))
    assert np.abs(a[:5] - b[:5]).max() < 0.05
    lo_a = ip._soft_limit(np.array([0.0, 10.0, 20.0], np.float32))
    lo_b = ip._soft_limit(np.array([-0.01, 10.0, 20.0], np.float32))
    assert np.abs(lo_a[1:] - lo_b[1:]).max() < 0.05


def test_tone_curve_keeps_sub_level_changes():
    """톤 커브 앞의 1레벨 미만 변화가 uint8 인덱싱에 잘려 사라지지 않는다."""
    ramp = np.tile(np.linspace(140, 250, 256, dtype=np.float32), (32, 1))
    img = Image.fromarray(np.stack([ramp] * 3, -1).astype(np.uint8))
    kw = dict(tone_curve_preset="soft_film", tone_curve_strength=0.5)
    base = np.asarray(ip._apply_lab_adjustments(img, **kw), np.float32)
    lifted = np.asarray(ip._apply_lab_adjustments(img, highlights=0.02, **kw), np.float32)
    assert lifted.mean() - base.mean() > 0.2
