"""선명한 파랑이 보정 뒤 보라로 돌던 문제의 회귀 테스트.

기본 레시피(saturation −0.184 + brightness +0.16 + 톤 커브)에서 로열블루 드레스가
RGB (6,24,122) → (48,42,123)로 R>G가 됐다. CIELAB a·b를 같은 비율로 줄이거나 a·b를
둔 채 L만 올리면 CIELAB 색상각은 그대로지만, CIELAB은 파랑 영역에서 색상이 고르지
않아 눈에는 보라로 보인다 (Oklab 색상각 +15°).
L·채도 연산 뒤 Oklab 색상각을 연산 전 값으로 되돌린다 ([ip._keep_oklab_hue]).
색온도처럼 일부러 색상을 옮기는 연산은 그대로 둔다.
"""

import numpy as np
from PIL import Image

import image_processor as ip

_ROYAL = (6, 24, 122)
_C5_DEFAULT = dict(brightness=0.16, highlights=0.28, shadows=0.036, saturation=-0.184,
                   tone_curve_preset="soft_film", tone_curve_strength=0.52)


def _flat(rgb, size=48) -> Image.Image:
    return Image.fromarray(np.full((size, size, 3), rgb, np.uint8))


def _mean_rgb(img: Image.Image) -> np.ndarray:
    return np.asarray(img.convert("RGB"), np.float64).reshape(-1, 3).mean(0)


def _oklab(rgb) -> np.ndarray:
    c = np.asarray(rgb, np.float64) / 255.0
    lin = np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)
    lms = np.cbrt(lin @ ip._OK_M1.T.astype(np.float64))
    return lms @ ip._OK_M2.T.astype(np.float64)


def _hue(rgb) -> float:
    o = _oklab(rgb)
    return float(np.degrees(np.arctan2(o[2], o[1])) % 360.0)


def _hue_diff(a: float, b: float) -> float:
    return abs((a - b + 180.0) % 360.0 - 180.0)


def _chroma(rgb) -> float:
    o = _oklab(rgb)
    return float(np.hypot(o[1], o[2]))


def test_oklab_roundtrip_is_exact_for_in_gamut_colors():
    rgb = np.random.default_rng(0).integers(0, 256, (32, 32, 3)).astype(np.uint8)
    lab = ip._to_lab8f(Image.fromarray(rgb))
    l, a, b = ip._keep_oklab_hue((lab[..., 0], lab[..., 1], lab[..., 2]),
                                 (lab[..., 0], lab[..., 1], lab[..., 2]))
    back = np.asarray(ip._from_lab8f(np.dstack([l, a, b]).astype(np.float32)), np.int16)
    assert np.abs(back - rgb).max() <= 1


def test_royal_blue_stays_blue_under_default_desaturation_and_brightening():
    for kw in (dict(saturation=-0.184, brightness=0.16), _C5_DEFAULT):
        out = _mean_rgb(ip._apply_lab_adjustments(_flat(_ROYAL), **kw))
        assert _hue_diff(_hue(out), _hue(_ROYAL)) < 3.0, (kw, out)
        assert out[0] < out[1], (kw, out)              # R<G — 보라가 아니다 (예전 48,42)
        assert _chroma(out) < _chroma(_ROYAL)           # 채도는 여전히 낮아진다


def test_saturated_primaries_keep_hue_under_desaturation():
    for rgb in ((30, 60, 200), (20, 30, 90), (200, 30, 30), (40, 150, 60), (230, 130, 30)):
        out = _mean_rgb(ip._apply_lab_adjustments(_flat(rgb), saturation=-0.3))
        assert _hue_diff(_hue(out), _hue(rgb)) < 2.0, (rgb, out)
        assert 0.6 < _chroma(out) / _chroma(rgb) < 0.8, (rgb, out)


def test_brightening_alone_keeps_blue_hue():
    for rgb in (_ROYAL, (10, 49, 97)):
        out = _mean_rgb(ip._apply_lab_adjustments(_flat(rgb), brightness=0.3))
        assert _hue_diff(_hue(out), _hue(rgb)) < 2.0, (rgb, out)
        assert _oklab(out)[0] > _oklab(rgb)[0] + 0.03     # 실제로 밝아진다


def test_temperature_hue_shift_is_kept():
    base = _mean_rgb(ip._apply_lab_adjustments(_flat(_ROYAL), temperature=0.3))
    both = _mean_rgb(ip._apply_lab_adjustments(_flat(_ROYAL), temperature=0.3, brightness=0.16))
    assert _hue_diff(_hue(base), _hue(_ROYAL)) > 3.0      # 색온도는 색상을 옮긴다
    assert _hue_diff(_hue(both), _hue(base)) < 1.5        # 밝기는 그 위에 색상을 더 돌리지 않는다


def test_oversaturation_stays_in_gamut_without_hue_flip():
    for rgb in (_ROYAL, (200, 30, 30), (40, 150, 60)):
        out = _mean_rgb(ip._apply_lab_adjustments(_flat(rgb), saturation=0.3))
        assert _chroma(out) > _chroma(rgb)
        assert _hue_diff(_hue(out), _hue(rgb)) < 3.0, (rgb, out)


def test_monochrome_is_still_neutral_with_brightness():
    rgb = np.random.default_rng(1).integers(0, 256, (40, 40, 3)).astype(np.uint8)
    out = np.asarray(ip._apply_lab_adjustments(Image.fromarray(rgb), saturation=-1.0,
                                               brightness=0.16), np.int16)
    assert (out.max(axis=2) - out.min(axis=2)).max() <= 1


def test_sky_and_skin_relief_still_apply():
    """하늘·피부의 채도 덜 빼기는 채도(C) 쪽이라 색상 고정과 겹치지 않는다."""
    h, w = 240, 320
    arr = np.zeros((h, w, 3), np.uint8)
    arr[:100] = (100, 150, 228)          # 위쪽 넓은 하늘
    arr[100:] = (124, 122, 118)
    arr[150:210, 30:110] = (30, 60, 200)     # 파란 옷
    arr[150:210, 200:280] = (214, 160, 130)  # 피부
    src = Image.fromarray(arr)
    out = np.asarray(ip._apply_lab_adjustments(src, saturation=-0.3, brightness=0.1), np.float64)

    def ratio(y0, y1, x0, x1):
        a = arr[y0:y1, x0:x1].reshape(-1, 3).mean(0)
        b = out[y0:y1, x0:x1].reshape(-1, 3).mean(0)
        return _chroma(b) / _chroma(a), _hue_diff(_hue(b), _hue(a))

    sky, sky_dh = ratio(10, 60, 40, 280)
    blue, blue_dh = ratio(160, 200, 40, 100)
    skin, _ = ratio(160, 200, 210, 270)
    assert sky > blue + 0.1              # 하늘은 덜 뺀다
    assert skin > blue + 0.08            # 피부도 덜 뺀다
    assert sky_dh < 2.0 and blue_dh < 2.0


def test_crushed_shadows_do_not_turn_blue():
    """대비·채도 올리기로 아주 어두운 곳의 L만 내려가 선형 RGB가 음수가 된 화소.

    그 값 그대로 Oklab 채도를 재면 실제보다 훨씬 커서, 색상을 되돌릴 때 숲 그림자가
    파랗게 떴다 (flash_digicam 풍경). 화면에 나갈 색(채널별 자르기) 기준으로 잰다.
    """
    arr = np.zeros((64, 64, 3), np.uint8)
    arr[:, :32] = (23, 26, 31)
    arr[:, 32:] = (210, 210, 205)
    kw = dict(contrast=0.5, shadows=-0.3, saturation=0.31, tone_curve_preset="flash",
              tone_curve_strength=0.5)
    out = np.asarray(ip._apply_lab_adjustments(Image.fromarray(arr), **kw), np.float64)
    dark = out[:, :32].reshape(-1, 3).mean(0)
    assert dark.max() < 25
    assert dark[2] - dark[0] < 16, dark   # 채널 자르기만 하던 예전 결과 (0,1,13) 수준
