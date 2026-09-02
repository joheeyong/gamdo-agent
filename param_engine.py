"""보정 파라미터 산출 엔진 — 측정값 + 스타일 프로필 규칙으로 슬라이더 값을 계산한다.

기존에는 Claude가 사진을 눈으로 보고 brightness/contrast/saturation 같은 수치를
직접 추천했다. 두 가지 문제가 있었다:

1. 정확도 — 1024px로 줄인 JPEG를 눈대중해서 "밝기 +0.12"를 정하는 것보다,
   히스토그램에서 실제 밝기를 재고 목표값과의 차이를 계산하는 쪽이 정확하다.
2. 지연 — recommendedParams 블록(HSL 8채널 포함)이 응답에서 가장 큰 덩어리였고,
   출력 토큰 수가 곧 응답 시간이다.

그래서 수치는 여기서 계산하고, 모델에게는 눈이 필요한 판단
(피사체 종류, 분위기, 구도, 영역별 보정)만 맡긴다.

프롬프트에 표로 적혀 있던 트렌드·피사체별 레시피가 이 파일의 규칙 테이블이다.
"""

from __future__ import annotations

import logging
import os
from typing import Any

import cv2
import numpy as np
from PIL import Image

from image_processor import estimate_noise_sigma

log = logging.getLogger("gamdo-agent")

# 측정 시 이미지를 이 크기로 줄인다 — 통계값은 해상도에 거의 무관하다.
_MEASURE_MAX_PX = 512


# ── 이미지 측정 ──


def _center_crop(img: Image.Image, size: int) -> Image.Image:
    """원본 해상도를 유지한 채 가운데 정사각 영역을 잘라낸다."""
    w, h = img.size
    if w <= size and h <= size:
        return img
    side = min(size, w, h)
    left, top = (w - side) // 2, (h - side) // 2
    return img.crop((left, top, left + side, top + side))


def measure_image_stats(img: Image.Image) -> dict[str, float]:
    """사진의 실제 상태를 측정한다. 모든 값은 0~1 (warmth만 -1~1).

    - brightness: Rec.709 휘도 평균
    - contrast: 휘도의 5~95 백분위 폭 (표준편차보다 극단값에 덜 흔들린다)
    - saturation: HSV 채도 평균
    - warmth: R-B 균형. 양수면 웜톤
    - highlight_clip / shadow_crush: 날아간·뭉갠 픽셀 비율
    - highlight_p95: 밝은 끝(95백분위)의 위치. 하이라이트를 누를 여지가 있는지
    - haze: Dark Channel Prior 평균. 높을수록 뿌옇다
    - sharpness: 라플라시안 분산을 0~1로 정규화
    """
    small = img.convert("RGB")
    w, h = small.size
    if max(w, h) > _MEASURE_MAX_PX:
        ratio = _MEASURE_MAX_PX / max(w, h)
        small = small.resize((max(1, int(w * ratio)), max(1, int(h * ratio))), Image.BILINEAR)

    arr = np.asarray(small, dtype=np.float32)
    r, g, b = arr[..., 0], arr[..., 1], arr[..., 2]

    luma = 0.2126 * r + 0.7152 * g + 0.0722 * b
    p5, p95 = np.percentile(luma, [5, 95])

    hsv = cv2.cvtColor(arr.astype(np.uint8), cv2.COLOR_RGB2HSV)
    saturation = float(hsv[..., 1].mean()) / 255.0

    gray = luma.astype(np.uint8)
    lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())

    # Dark Channel Prior: 국소 최소값이 클수록 안개가 낀 사진
    dark_channel = cv2.erode(arr.min(axis=2).astype(np.uint8), np.ones((9, 9), np.uint8))
    haze = float(dark_channel.mean()) / 255.0

    # 색 틀어짐이 사진 전체에 고른지 — 조명 탓인지 장면 탓인지 가른다.
    # 백열등 실내는 밝은 곳도 어두운 곳도 다 누렇지만(고름 → 교정 대상),
    # 노을은 하늘만 붉고 그늘은 그렇지 않다(고르지 않음 → 장면의 색).
    lo, hi = np.percentile(luma, [33, 67])
    dark_px, bright_px = luma <= lo, luma >= hi
    warm_dark = float((r[dark_px] - b[dark_px]).mean()) / 128.0 if dark_px.any() else 0.0
    warm_bright = float((r[bright_px] - b[bright_px]).mean()) / 128.0 if bright_px.any() else 0.0
    if warm_dark * warm_bright <= 0:
        cast_uniformity = 0.0          # 부호가 다르면 조명 탓이 아니다
    else:
        lo_mag, hi_mag = sorted((abs(warm_dark), abs(warm_bright)))
        cast_uniformity = lo_mag / hi_mag if hi_mag > 1e-6 else 0.0

    return {
        "brightness": float(luma.mean()) / 255.0,
        "contrast": float(p95 - p5) / 255.0,
        "saturation": saturation,
        "warmth": float(r.mean() - b.mean()) / 128.0,
        "highlight_p95": float(p95) / 255.0,
        "highlight_clip": float((luma > 250).mean()),
        "shadow_crush": float((luma < 6).mean()),
        "haze": haze,
        # 라플라시안 분산 500 정도면 충분히 선명한 사진으로 본다
        "sharpness": min(1.0, lap_var / 500.0),
        # 노이즈는 반드시 원본 해상도에서 잰다 — 축소하면 이웃 화소가 평균되어
        # 노이즈가 사라져 버린다. 비용을 아끼려 가운데 일부만 잘라 본다.
        "noise": estimate_noise_sigma(_center_crop(img, 512)),
        "cast_uniformity": round(cast_uniformity, 3),
    }


def extract_dominant_colors(img: Image.Image, k: int = 5) -> list[str]:
    """k-means로 실제 대표 색 k개를 뽑아 hex로 반환한다 (큰 군집 순).

    모델이 색을 눈대중해 hex를 지어내던 것을 대체한다.
    """
    small = img.convert("RGB")
    w, h = small.size
    if max(w, h) > 160:
        ratio = 160 / max(w, h)
        small = small.resize((max(1, int(w * ratio)), max(1, int(h * ratio))), Image.BILINEAR)

    pixels = np.asarray(small, dtype=np.float32).reshape(-1, 3)
    if len(pixels) < k:
        k = max(1, len(pixels))

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
    _, labels, centers = cv2.kmeans(
        pixels, k, None, criteria, 3, cv2.KMEANS_PP_CENTERS
    )

    counts = np.bincount(labels.flatten(), minlength=k)
    order = np.argsort(-counts)
    return [
        "#{:02X}{:02X}{:02X}".format(*(int(round(c)) for c in centers[i]))
        for i in order
    ]


def measure_color_analysis(img: Image.Image) -> dict[str, Any]:
    """colorAnalysis 중 측정 가능한 필드를 계산한다.

    colorHarmony / paletteDescription은 서술이라 모델이 담당한다.
    """
    stats = measure_image_stats(img)
    warmth = stats["warmth"]
    if warmth > 0.06:
        temperature = "warm"
    elif warmth < -0.06:
        temperature = "cool"
    else:
        temperature = "neutral"

    return {
        "dominantColors": extract_dominant_colors(img),
        "colorTemperature": temperature,
        "saturationLevel": round(stats["saturation"], 3),
        "brightnessLevel": round(stats["brightness"], 3),
    }


# ── 스타일 프로필 → 목표값 ──

_TONE_TARGETS = {
    "cool": -0.30,
    "slightly_cool": -0.15,
    "neutral": 0.0,
    "slightly_warm": 0.15,
    "warm": 0.30,
    "mixed": 0.05,
}

_LEVEL5 = {"very_low": 0, "low": 1, "medium": 2, "high": 3, "very_high": 4}

# 5단계 성향 → 목표 측정값
_SATURATION_TARGETS = [0.22, 0.30, 0.38, 0.47, 0.56]
_BRIGHTNESS_TARGETS = [0.36, 0.43, 0.50, 0.57, 0.64]
_CONTRAST_TARGETS = [0.48, 0.58, 0.68, 0.78, 0.88]

# 보정 강도 → 전체 게인
_FILTER_GAIN = {
    "none": 0.0,
    "minimal": 0.45,
    "moderate": 0.75,
    "strong": 1.0,
    "very_strong": 1.3,
    "auto": 0.8,
}

_GRAIN_LEVELS = {"none": 0.0, "subtle": 0.12, "moderate": 0.22, "heavy": 0.35, "film": 0.28}
_VIGNETTE_LEVELS = {"none": 0.0, "subtle": 0.10, "moderate": 0.20, "strong": 0.30}
_SKIN_LEVELS = {"none": 0.0, "light": 0.18, "moderate": 0.30, "heavy": 0.45}


def _level_index(value: str | None, default: int = 2) -> int:
    return _LEVEL5.get(value or "", default)


# ── 트렌드·피사체 레시피 ──
#
# 프롬프트에 산문으로 적혀 있던 규칙을 그대로 옮긴 것이다.
# 여기 없는 키는 0으로 본다.

_TREND_RECIPES: dict[str, dict[str, Any]] = {
    "warm_film": {
        "temperature": 0.18, "shadows": 0.28, "highlights": -0.25, "saturation": -0.08,
        "tone_curve": ("film", 0.60), "grain": 0.20,
        "split": {"shadow": (255, 0.15), "highlight": (30, 0.15)},
    },
    "korean_gamsung": {
        "temperature": 0.10, "shadows": 0.30, "highlights": -0.28, "saturation": -0.12,
        "clarity": -0.08, "tone_curve": ("fade", 0.40), "grain": 0.08,
    },
    "cinematic_moody": {
        "temperature": 0.05, "shadows": 0.15, "highlights": -0.30, "saturation": -0.10,
        "clarity": 0.20, "vignette": 0.22, "tone_curve": ("high_contrast", 0.50), "grain": 0.28,
        "split": {"shadow": (210, 0.30), "highlight": (30, 0.22)},
    },
    "bright_airy": {
        "temperature": 0.08, "shadows": 0.35, "highlights": -0.30, "saturation": -0.05,
        "clarity": 0.05, "vignette": 0.0, "tone_curve": ("bright", 0.40), "grain": 0.05,
    },
    "golden_hour": {
        "temperature": 0.22, "shadows": 0.25, "highlights": -0.28, "saturation": 0.0,
        "tone_curve": ("film", 0.45), "grain": 0.12,
        "split": {"shadow": (270, 0.12), "highlight": (40, 0.15)},
    },
    "clean_minimal": {
        "temperature": 0.05, "shadows": 0.20, "highlights": -0.20, "saturation": -0.05,
        "clarity": 0.05, "vignette": 0.0, "tone_curve": ("linear", 0.0), "grain": 0.0,
    },
}

# 트렌드를 모를 때 쓰는 2025-2026 공통 베이스라인
_DEFAULT_RECIPE: dict[str, Any] = {
    "temperature": 0.10, "shadows": 0.25, "highlights": -0.25, "saturation": -0.08,
    "tone_curve": ("s_curve", 0.25), "grain": 0.05,
}

_SUBJECT_RECIPES: dict[str, dict[str, Any]] = {
    "인물": {
        "clarity": -0.02, "sharpness": 0.05, "vignette": 0.12,
        "blemish_removal": 0.35, "skin_smoothing": 0.28, "dehaze": 0.0,
        "tone_curve": ("s_curve", 0.30),
    },
    "풍경": {"clarity": 0.18, "sharpness": 0.12, "vignette": 0.03, "use_haze": True},
    "음식": {
        "clarity": 0.25, "sharpness": 0.22, "vignette": 0.18,
        "saturation": 0.06, "temperature": 0.25,
    },
    "카페/일상": {"clarity": -0.10, "contrast": -0.05, "vignette": 0.10},
    "사물": {"clarity": 0.15, "sharpness": 0.10, "vignette": 0.15},
    "동물": {"clarity": 0.12, "sharpness": 0.15, "vignette": 0.10},
    "혼합": {},
}


# 측정에서 나온 교정 성분 하나가 낼 수 있는 최대치
_CORRECTION_BAND = 0.35

# 노출은 다른 축보다 좁게 잡는다. 평균 휘도는 장면마다 정당하게 다르다 —
# 설경·흰 벽 카페·역광은 원래 높고 야경은 원래 낮다. 목표 평균에 억지로
# 맞추면 잘 찍은 밝은 사진이 전부 중간 회색으로 눌린다.
# 목표에서 이 폭 안이면 노출이 맞은 것으로 보고 손대지 않는다.
_EXPOSURE_DEADZONE = 0.06
# 데드존을 벗어났을 때 노출 교정이 낼 수 있는 최대치
_EXPOSURE_BAND = 0.20

# 화이트밸런스 강도 1.0이 실제로 중화하는 비율 (채널 게인 상한 때문에 1은 아니다)
_AWB_NEUTRALIZE = 0.8


def _clamp(v: float, lo: float = -1.0, hi: float = 1.0) -> float:
    return round(max(lo, min(hi, v)), 3)


def _band(v: float, limit: float = _CORRECTION_BAND) -> float:
    """측정 기반 교정값을 ±limit로 묶는다."""
    return max(-limit, min(limit, v))


def _deadzone(diff: float, width: float) -> float:
    """목표와의 차이에서 width 안쪽은 0으로 죽이고, 벗어난 만큼만 남긴다."""
    if abs(diff) <= width:
        return 0.0
    return diff - width if diff > 0 else diff + width


def _ramp(v: float, lo: float, hi: float) -> float:
    """lo 이하면 0, hi 이상이면 1, 사이는 선형. 0~1."""
    if hi <= lo:
        return 0.0
    return max(0.0, min(1.0, (v - lo) / (hi - lo)))


def build_recommended_params(
    img: Image.Image,
    style_profile: dict[str, Any] | None,
    analysis: dict[str, Any] | None = None,
    reference: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """[build_params_with_comment]에서 파라미터만 꺼내는 단축 함수."""
    params, _ = build_params_with_comment(img, style_profile, analysis, reference)
    return params


def build_params_with_comment(
    img: Image.Image,
    style_profile: dict[str, Any] | None,
    analysis: dict[str, Any] | None = None,
    reference: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], str]:
    """측정값 + 프로필 + 모델의 스타일 방향으로 recommendedParams와 설명을 만든다.

    파라미터의 형식은 기존에 모델이 내려주던 recommendedParams와 동일해서
    [analysis_to_transform_params]가 그대로 소비할 수 있다.

    설명은 왜 이 값이 나왔는지를 한 문장으로 적은 것이다. 값을 정한 근거가
    여기 다 있으므로 모델에게 따로 물어볼 필요가 없다.
    """
    profile = normalize_style_profile(style_profile)
    analysis = analysis or {}

    stats = measure_image_stats(img)

    color_pref = profile.get("colorPreference") or {}
    editing = profile.get("editingStyle") or {}
    trend = profile.get("trendCategory") or ""
    subject = str(analysis.get("subjectType") or "").strip()

    recipe = dict(_TREND_RECIPES.get(trend, _DEFAULT_RECIPE))
    subject_recipe = _SUBJECT_RECIPES.get(subject, {})

    # 게인: 보정 강도 성향이 전체 세기를 정한다
    gain = _FILTER_GAIN.get(editing.get("filterTendency") or "auto", 0.8)

    # 화이트밸런스 세기는 색온도 계산보다 먼저 정해져야 한다 (아래에서 참조)
    auto_wb_strength = round(0.65 * stats["cast_uniformity"], 3)

    # ── 측정값 ↔ 목표값 차이로 노출계 3형제를 정한다 ──
    if reference:
        # 사용자가 실제로 올리는 사진에서 잰 값. 카테고리 추정보다 정확하다.
        target_brightness = reference["brightness"]
        target_contrast = reference["contrast"]
        target_saturation = reference["saturation"]
        target_warmth = reference["warmth"]
    else:
        target_brightness = _BRIGHTNESS_TARGETS[_level_index(color_pref.get("brightnessTendency"))]
        target_contrast = _CONTRAST_TARGETS[_level_index(color_pref.get("contrast"))]
        target_saturation = _SATURATION_TARGETS[_level_index(color_pref.get("saturationTendency"))]
        target_warmth = _TONE_TARGETS.get(color_pref.get("preferredTones") or "neutral", 0.0)

    # ── 장면에 따라 기준을 바꾼다 ──
    scene = detect_scene(stats, img)

    # 차이를 슬라이더 범위로 옮기는 배율. 측정 스케일과 슬라이더 스케일이 달라 실측으로 맞춘 값들이다.
    #
    # 측정 성분은 ±_CORRECTION_BAND로 묶는다. 측정은 "현재 사진을 목표 쪽으로 당기는"
    # 교정이지 스타일이 아니다. 밴드가 없으면 거의 무채색인 사진 한 장이
    # saturation +1.0 같은 값을 만들어 이미지를 태워버린다. 스타일의 세기는
    # 아래 레시피가 담당한다.
    # 노출만은 "목표 평균에 맞추기"가 아니라 "빗나갔을 때만 당기기"다.
    # 데드존 밖으로 나간 만큼만, 그것도 좁은 밴드 안에서 움직인다.
    brightness = _band(
        _deadzone(target_brightness - stats["brightness"], _EXPOSURE_DEADZONE) * 1.2,
        _EXPOSURE_BAND,
    )
    contrast = _band((target_contrast - stats["contrast"]) * 1.6)
    saturation = _band((target_saturation - stats["saturation"]) * 1.8)
    # 화이트밸런스가 먼저 중립으로 당기므로, 그 뒤에 남는 웜니스를 기준으로 잡는다.
    # 원본 warmth를 그대로 쓰면 같은 편차를 두 번 보정하게 된다.
    warmth_after_wb = stats["warmth"] * (1.0 - _AWB_NEUTRALIZE * auto_wb_strength)
    temperature = _band((target_warmth - warmth_after_wb) * 1.2)

    # 레시피의 방향성을 더한다 (트렌드 → 피사체 순으로 덮어씀)
    def pick(key: str, default: float = 0.0) -> float:
        if key in subject_recipe:
            return float(subject_recipe[key])
        return float(recipe.get(key, default))

    saturation += float(recipe.get("saturation", 0.0)) + float(subject_recipe.get("saturation", 0.0))
    temperature += float(recipe.get("temperature", 0.0)) + float(subject_recipe.get("temperature", 0.0))
    contrast += float(subject_recipe.get("contrast", 0.0))

    shadows = float(recipe.get("shadows", 0.0))
    # 레시피의 하이라이트 억제는 누를 밝은 부분이 있을 때만 뜻이 있다.
    # 밝은 끝이 낮은 사진에 그대로 걸면 하이라이트가 아니라 중간톤이 눌려
    # 사진 전체가 어두워진다. p95가 올라온 만큼만 비례해 적용한다.
    highlights = float(recipe.get("highlights", 0.0)) * _ramp(
        stats["highlight_p95"], 0.80, 0.95
    )

    if scene["backlit"]:
        # 역광: 피사체가 실루엣으로 남는다. 쉐도우를 크게 들어올리고
        # 날아간 배경을 눌러 준다.
        shadows += 0.25
        highlights -= 0.15
    if scene["low_light"]:
        # 저조도: 대비를 세우면 노이즈와 뭉갬이 같이 도드라진다
        contrast *= 0.6

    # 날아간 하이라이트·뭉갠 쉐도우가 많으면 그만큼 더 되살린다
    if stats["highlight_clip"] > 0.02:
        highlights -= min(0.25, stats["highlight_clip"] * 4.0)
    if stats["shadow_crush"] > 0.02:
        shadows += min(0.25, stats["shadow_crush"] * 4.0)

    clarity = pick("clarity")
    sharpness = pick("sharpness")
    vignette = pick("vignette")

    # 이미 흐린 사진이면 선명도를 올리고, 충분히 선명하면 건드리지 않는다
    if stats["sharpness"] < 0.25:
        sharpness += 0.15
    elif stats["sharpness"] > 0.75:
        sharpness = min(sharpness, 0.05)

    # 노이즈 제거: 실제 측정된 노이즈량에 비례하되, 쉐도우를 많이 들어올릴수록
    # 어두운 곳 노이즈가 더 드러나므로 그만큼 세게 잡는다.
    noise = stats["noise"]
    if noise < 2.5:
        denoise_strength = 0.0
    else:
        denoise_strength = min(1.0, (noise - 2.5) / 12.0)
        denoise_strength = min(1.0, denoise_strength + max(0.0, shadows) * 0.4)
    if scene["low_light"]:
        denoise_strength = min(1.0, denoise_strength + 0.2)

    # 안개 제거는 풍경에서 실제로 뿌옇게 측정될 때만.
    #
    # 다크 채널 평균(stats["haze"])만 보면 안 된다. 그 값은 "가장 어두운 채널이
    # 얼마나 들려 있나"라서 맑은 날 하늘이나 흰 벽처럼 어두운 물체가 없는
    # 밝은 사진이면 무조건 높게 나온다 — 실측에서 맑은 하늘 0.55, 진짜 안개
    # 0.63으로 거의 구분이 안 됐다. 그대로 믿으면 화창한 풍경마다 dehaze가
    # 최대로 걸려 사진이 통째로 어두워진다.
    #
    # 안개는 세 가지가 동시에 성립할 때다: 검은 점이 들리고(veil),
    # 대비가 눌리고(flat), 색이 빠진다(washed). 하나라도 아니면 안개가 아니다.
    dehaze = 0.0
    if subject_recipe.get("use_haze"):
        veil = _ramp(stats["haze"], 0.45, 0.75)
        flat = _ramp(0.55 - stats["contrast"], 0.0, 0.25)
        washed = 1.0 - _ramp(stats["saturation"], 0.22, 0.40)
        dehaze = min(0.35, veil * flat * washed * 0.7)

    # 사람 사진이 아니면 피부 보정은 하지 않는다
    is_portrait = subject == "인물"
    skin_level = editing.get("skinRetouchLevel") or "auto"
    if not is_portrait:
        blemish = skin_smoothing = 0.0
    elif skin_level == "auto":
        blemish = float(subject_recipe.get("blemish_removal", 0.0))
        skin_smoothing = float(subject_recipe.get("skin_smoothing", 0.0))
    else:
        skin_smoothing = _SKIN_LEVELS.get(skin_level, 0.0)
        blemish = min(1.0, skin_smoothing * 1.2)

    # 그레인·비네팅은 프로필이 명시하면 프로필이 이긴다
    grain_pref = editing.get("grainPreference") or "auto"
    grain = float(recipe.get("grain", 0.0)) if grain_pref == "auto" else _GRAIN_LEVELS.get(grain_pref, 0.0)

    vignette_pref = editing.get("vignettePreference") or "auto"
    if vignette_pref != "auto":
        vignette = _VIGNETTE_LEVELS.get(vignette_pref, 0.0)

    tone_preset, tone_strength = subject_recipe.get(
        "tone_curve", recipe.get("tone_curve", ("linear", 0.0))
    )
    # 트렌드가 톤 커브를 지정했으면 트렌드를 우선한다 (피사체보다 스타일이 상위)
    if "tone_curve" in recipe and trend in _TREND_RECIPES:
        tone_preset, tone_strength = recipe["tone_curve"]

    # 레퍼런스가 있으면 프리셋 대신 그 사람 사진의 밝기 분포를 따라간다.
    # 프리셋은 "필름이면 이런 곡선"이라는 일반론이고, 이쪽은 그 사람의 실제 곡선이다.
    tone_points = build_reference_tone_curve(img, reference, 0.45 * gain)
    if tone_points:
        tone_preset, tone_strength = "reference", 1.0

    split = recipe.get("split") or {}
    shadow_hue, shadow_str = split.get("shadow", (0, 0.0))
    hi_hue, hi_str = split.get("highlight", (0, 0.0))

    params: dict[str, Any] = {
        # 게인이 걸리는 항목 — 보정 강도 성향에 비례해 세진다
        "brightness": _clamp(brightness * gain),
        "contrast": _clamp(contrast * gain),
        "clarity": _clamp(clarity * gain),
        "dehaze": _clamp(dehaze * gain, 0.0, 1.0),
        "highlights": _clamp(highlights * gain),
        "shadows": _clamp(shadows * gain),
        "saturation": _clamp(saturation * gain),
        "temperature": _clamp(temperature * gain),
        "sharpness": _clamp(sharpness * gain),
        "vignette": _clamp(vignette * gain),
        "grain": _clamp(grain * gain, 0.0, 1.0),
        # 촬영 결함 교정은 취향(보정 강도)과 무관하므로 게인을 곱하지 않는다
        "auto_wb": _clamp(auto_wb_strength, 0.0, 1.0),
        "denoise": _clamp(denoise_strength, 0.0, 1.0),
        # 배경 흐림은 인물에서만. 얼굴이 없으면 apply 단계에서 무시된다.
        "background_blur": _clamp(0.25 * gain if is_portrait else 0.0, 0.0, 1.0),
        # 피부 보정은 사용자가 고른 강도 그대로 — 필터 게인을 곱하지 않는다
        "blemish_removal": _clamp(blemish, 0.0, 1.0),
        "skin_smoothing": _clamp(skin_smoothing, 0.0, 1.0),
        "toneCurve": {
            "preset": tone_preset,
            # 레퍼런스 곡선은 이미 strength만큼 섞어서 만들었으므로 그대로 태운다
            "strength": 1.0 if tone_points else _clamp(float(tone_strength) * gain, 0.0, 1.0),
            "points": tone_points,
        },
        "splitToning": {
            "shadow": {"hue": shadow_hue, "strength": _clamp(shadow_str * gain, 0.0, 1.0)},
            "highlight": {"hue": hi_hue, "strength": _clamp(hi_str * gain, 0.0, 1.0)},
        },
    }

    # 얼굴/체형은 눈이 필요한 판단이라 모델이 인물 사진에서만 제안한다.
    # 스키마상 최상위에 오지만, 옛 응답 형식(recommendedParams 안)도 받아준다.
    reshape = analysis.get("reshapeParams")
    if not isinstance(reshape, dict):
        legacy = analysis.get("recommendedParams")
        if isinstance(legacy, dict):
            reshape = legacy.get("reshapeParams")
    if isinstance(reshape, dict) and is_portrait:
        params["reshapeParams"] = _clamp_reshape(reshape)

    log.info(
        "param_engine: subject=%s trend=%s gain=%.2f | measured b=%.2f c=%.2f s=%.2f w=%.2f haze=%.2f sharp=%.2f",
        subject or "-", trend or "-", gain,
        stats["brightness"], stats["contrast"], stats["saturation"],
        stats["warmth"], stats["haze"], stats["sharpness"],
    )
    return params, describe_params(stats, trend, subject, params, gain)


# 워프 계수가 커진 만큼 모델이 범위를 벗어난 값을 주면 얼굴이 뭉개진다.
# 프롬프트 권장 상한(0.5)에서 한 번 더 자른다.
_RESHAPE_MAX = 0.5
_RESHAPE_KEYS = (
    "face_slim", "jaw_sharpen", "eye_enlarge",
    "leg_stretch", "shoulder_width", "waist_slim",
)


def _clamp_reshape(reshape: dict[str, Any]) -> dict[str, float]:
    """얼굴/체형 값을 안전 범위로 자른다. shoulder_width만 음수를 허용한다."""
    out: dict[str, float] = {}
    for key in _RESHAPE_KEYS:
        raw = reshape.get(key)
        if not isinstance(raw, (int, float)):
            continue
        lo = -_RESHAPE_MAX if key == "shoulder_width" else 0.0
        out[key] = _clamp(float(raw), lo, _RESHAPE_MAX)
    return out


# ── 적용된 변형 코멘트 ──

_TREND_LABELS = {
    "warm_film": "웜 필름",
    "korean_gamsung": "한국 감성",
    "cinematic_moody": "시네마틱",
    "bright_airy": "밝은 감성",
    "golden_hour": "골든아워",
    "clean_minimal": "클린 미니멀",
}

_CURVE_LABELS = {
    "film": "필름 커브",
    "s_curve": "S커브",
    "fade": "페이드",
    "high_contrast": "강한 대비 커브",
    "bright": "밝은 커브",
}


def describe_params(
    stats: dict[str, float],
    trend: str,
    subject: str,
    params: dict[str, Any],
    gain: float,
) -> str:
    """왜 이 보정값이 나왔는지 한 문장으로 설명한다.

    값을 정한 근거(측정값·프로필·피사체)가 모두 여기 있으므로
    모델에게 설명을 시키지 않는다 — 공짜이고, 실제 근거와 어긋날 일도 없다.
    """
    if gain <= 0.0:
        return "보정 없음 설정이라 원본 톤을 그대로 두었어요"

    is_portrait = subject == "인물" and params["skin_smoothing"] >= 0.1

    # ── 측정에서 나온 이유 (눈에 띄는 것부터) ──
    reasons: list[str] = []

    if params["brightness"] >= 0.10 and stats["brightness"] < 0.45:
        reasons.append("어두워서 밝기를 올리고")
    elif params["brightness"] <= -0.10 and stats["brightness"] > 0.55:
        reasons.append("밝게 찍혀 노출을 낮추고")

    if stats["highlight_clip"] > 0.03 and params["highlights"] < -0.1:
        reasons.append("날아간 밝은 부분을 눌러 주고")
    elif stats["shadow_crush"] > 0.03 and params["shadows"] > 0.1:
        reasons.append("뭉친 어두운 부분을 살리고")
    elif params["dehaze"] >= 0.10:
        reasons.append("뿌연 기운을 걷어내고")
    elif params["contrast"] >= 0.15:
        reasons.append("밋밋한 대비를 세우고")
    elif params["contrast"] <= -0.15:
        reasons.append("센 대비를 부드럽게 눌러")
    elif params["sharpness"] >= 0.12 and stats["sharpness"] < 0.3:
        reasons.append("흐린 초점을 다듬고")
    elif params["saturation"] <= -0.12:
        reasons.append("과한 채도를 덜어내고")
    elif params["saturation"] >= 0.12:
        reasons.append("빠진 색을 채우고")

    # 인물이면 피부 문장이 뒤에 붙으므로 이유는 하나만 남겨 길이를 맞춘다
    reasons = reasons[: 1 if is_portrait else 2]

    # ── 스타일 마무리 ──
    trend_label = _TREND_LABELS.get(trend)
    lead = f"{trend_label} 톤에 맞춰 " if trend_label else ""

    if params["temperature"] >= 0.12:
        warm_adj, warm_adv = "따뜻한", "따뜻하게"
    elif params["temperature"] <= -0.12:
        warm_adj, warm_adv = "차가운", "차갑게"
    else:
        warm_adj = warm_adv = ""

    curve = _CURVE_LABELS.get(params["toneCurve"]["preset"])
    if params["toneCurve"]["strength"] < 0.15:
        curve = None

    if curve and warm_adj:
        # "따뜻한 강한 대비 커브"처럼 수식이 세 마디 이상 겹칠 때만 쉼표로 끊는다
        tail = (
            f"{lead}{warm_adv}, {curve}를 얹었어요"
            if curve.count(" ") >= 2
            else f"{lead}{warm_adj} {curve}를 얹었어요"
        )
    elif curve:
        tail = f"{lead}{curve}를 얹었어요"
    elif warm_adv:
        tail = f"{lead}{warm_adv} 맞췄어요" if lead else f"{warm_adv} 톤을 맞췄어요"
    elif lead:
        tail = f"{lead}전체 톤을 정리했어요"
    else:
        tail = "전체 톤을 정리했어요"

    if is_portrait:
        tail = tail.replace("어요", "고, 피부는 자연스럽게 정리했어요")

    if not reasons:
        return tail
    return ", ".join(reasons) + " " + tail


# ── 기울기 측정 ──

# 이 각도를 넘어서면 의도한 구도(네덜란드 앵글 등)로 보고 건드리지 않는다.
_MAX_TILT = 8.0
# 이보다 작으면 회전으로 잃는 해상도가 이득보다 크다.
_MIN_TILT = 0.4


def detect_tilt_angle(img: Image.Image, max_angle: float = _MAX_TILT) -> float | None:
    """사진의 기울기를 재서 교정 각도(도)를 반환한다. 확신이 없으면 None.

    수평선(지평선·수면·테이블 모서리)과 수직선(건물·기둥·문틀)은
    실제로 수평·수직이라는 전제로, 검출된 직선들이 그 축에서 얼마나
    벗어났는지를 길이로 가중한 중앙값으로 추정한다.

    모델에게 눈대중시키는 것보다 정확하고, 근거(선의 개수·일관성)로
    확신 여부를 판단할 수 있다.
    """
    small = img.convert("L")
    w, h = small.size
    if max(w, h) > 900:
        ratio = 900 / max(w, h)
        w, h = max(1, int(w * ratio)), max(1, int(h * ratio))
        small = small.resize((w, h), Image.BILINEAR)

    gray = np.asarray(small, dtype=np.uint8)
    edges = cv2.Canny(gray, 60, 180, apertureSize=3)

    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 720,          # 0.25도 해상도
        threshold=60,
        minLineLength=int(min(w, h) * 0.25),
        maxLineGap=int(min(w, h) * 0.02) + 2,
    )
    if lines is None:
        return None

    deviations: list[float] = []
    weights: list[float] = []

    # OpenCV 버전에 따라 (N,1,4) 또는 (N,4)로 온다
    for x1, y1, x2, y2 in np.asarray(lines).reshape(-1, 4):
        dx, dy = float(x2 - x1), float(y2 - y1)
        length = float(np.hypot(dx, dy))
        if length < 1.0:
            continue

        # -90 ~ +90도로 정규화 (선은 방향이 없다)
        angle = (np.degrees(np.arctan2(dy, dx)) + 90.0) % 180.0 - 90.0

        if abs(angle) <= max_angle:
            deviation = angle                      # 수평선의 어긋남
        elif abs(abs(angle) - 90.0) <= max_angle:
            deviation = angle - 90.0 if angle > 0 else angle + 90.0  # 수직선의 어긋남
        else:
            continue                               # 대각선은 기준이 되지 못한다

        deviations.append(deviation)
        weights.append(length)

    if len(deviations) < 3:
        return None

    dev = np.array(deviations)
    wgt = np.array(weights)

    # 길이 가중 중앙값 — 긴 선(지평선·건물 모서리)이 짧은 잡선보다 신뢰도가 높다
    order = np.argsort(dev)
    dev, wgt = dev[order], wgt[order]
    cumulative = np.cumsum(wgt)
    median = float(dev[int(np.searchsorted(cumulative, cumulative[-1] / 2.0))])

    if abs(median) < _MIN_TILT:
        return None

    # 확신 판정: 추정치 근처(±1도)에 모인 선이 전체 길이의 절반은 되어야 한다.
    # 그렇지 않으면 선들이 제각각이라는 뜻이고, 그때 회전하면 오히려 망친다.
    agreeing = wgt[np.abs(dev - median) <= 1.0].sum()
    if agreeing < wgt.sum() * 0.5:
        log.info("tilt: 선들이 일관되지 않아 보정하지 않음 (median=%.2f°)", median)
        return None

    # apply_straighten(θ)를 적용하면 측정 편차가 θ만큼 줄어든다.
    # 따라서 편차를 0으로 만들려면 편차값을 그대로 넘기면 된다. (실측으로 확인)
    return round(median, 2)


def prefix_tilt_comment(comment: str, tilt: float | None) -> str:
    """보정 코멘트 앞에 수평 보정 사실을 덧붙인다."""
    if tilt is None or abs(tilt) < _MIN_TILT:
        return comment
    return f"{abs(tilt):.1f}° 기울어 있어 수평을 맞추고, {comment}"


# ── 레퍼런스 사진 기반 목표 ──
#
# 스타일 프로필의 5단계 카테고리("보통", "높음")는 두 번의 추측 위에 서 있다.
# 모델이 피드를 눈으로 보고 고른 등급이고, 그 등급이 가리키는 수치는 검증된 적
# 없는 상수다. 사용자의 대표 사진은 서버에 이미 저장돼 있으니 그냥 재면 된다.

_LUMA_PERCENTILES = [1, 5, 10, 25, 50, 75, 90, 95, 99]

# (경로, mtime) → 통계. 요청마다 다시 읽지 않기 위한 캐시.
_ref_cache: dict[tuple, dict[str, Any]] = {}


def _luma_percentiles(img: Image.Image) -> list[float]:
    """밝기 분포의 대표 지점들 (0~1). 톤 커브 매칭의 기준이 된다."""
    small = img.convert("RGB")
    w, h = small.size
    if max(w, h) > 384:
        r = 384 / max(w, h)
        small = small.resize((max(1, int(w * r)), max(1, int(h * r))), Image.BILINEAR)
    a = np.asarray(small, dtype=np.float32)
    luma = 0.2126 * a[..., 0] + 0.7152 * a[..., 1] + 0.0722 * a[..., 2]
    return [float(v) / 255.0 for v in np.percentile(luma, _LUMA_PERCENTILES)]


def measure_reference_target(ref_paths: list[str]) -> dict[str, Any] | None:
    """대표 사진들을 재서 '이 사용자가 실제로 올리는 사진'의 수치를 낸다.

    반환: brightness / contrast / saturation / warmth 평균 + 밝기 분포 백분위.
    읽을 수 있는 사진이 없으면 None (호출 쪽에서 카테고리 방식으로 폴백).
    """
    if not ref_paths:
        return None

    try:
        key = tuple(sorted((p, os.path.getmtime(p)) for p in ref_paths))
    except OSError:
        key = tuple(sorted(ref_paths))
    if key in _ref_cache:
        return _ref_cache[key]

    stats_list: list[dict[str, float]] = []
    pct_list: list[list[float]] = []
    for path in ref_paths[:5]:
        try:
            img = Image.open(path).convert("RGB")
        except Exception as exc:
            log.warning("reference photo unreadable %s: %s", path, exc)
            continue
        stats_list.append(measure_image_stats(img))
        pct_list.append(_luma_percentiles(img))

    if not stats_list:
        return None

    target = {
        key_: float(np.mean([s[key_] for s in stats_list]))
        for key_ in ("brightness", "contrast", "saturation", "warmth")
    }
    target["luma_percentiles"] = [float(v) for v in np.mean(pct_list, axis=0)]
    target["count"] = len(stats_list)

    log.info(
        "reference target from %d photos: b=%.3f c=%.3f s=%.3f w=%+.3f",
        target["count"], target["brightness"], target["contrast"],
        target["saturation"], target["warmth"],
    )
    _ref_cache[key] = target
    return target


def feed_compatibility(stats: dict[str, float], reference: dict[str, Any] | None) -> int | None:
    """이 사진이 사용자 피드와 얼마나 어울리는지 0~100. 레퍼런스가 없으면 None.

    예전에는 모델이 눈대중으로 매기던 점수인데, 이제 실제 거리로 계산한다.
    """
    if not reference:
        return None

    # 각 축의 "완전히 다르다"고 볼 만한 차이로 정규화한다
    scale = {"brightness": 0.20, "contrast": 0.30, "saturation": 0.20, "warmth": 0.35}
    diffs = [
        abs(stats[k] - reference[k]) / scale[k]
        for k in ("brightness", "contrast", "saturation", "warmth")
    ]
    distance = float(np.sqrt(np.mean(np.square(diffs))))
    return int(round(100 * max(0.0, 1.0 - min(1.0, distance))))


def build_reference_tone_curve(
    img: Image.Image,
    reference: dict[str, Any] | None,
    strength: float,
) -> list[tuple[float, float]] | None:
    """사진의 밝기 분포를 레퍼런스 분포 쪽으로 옮기는 톤 커브 제어점.

    평균값 네 개로는 "필름 룩"처럼 곡선 형태로 정의되는 취향을 담을 수 없다.
    같은 평균 밝기라도 쉐도우가 들려 있는지 아닌지가 인상을 가른다.
    백분위끼리 대응시켜 곡선을 만들고, strength만큼만 섞는다 —
    100% 적용하면 사진의 원래 명암 구조가 통째로 사라진다.
    """
    if not reference or strength < 0.01:
        return None
    ref_pct = reference.get("luma_percentiles")
    if not ref_pct:
        return None

    src_pct = _luma_percentiles(img)
    xs = [0.0] + src_pct + [1.0]
    ys = [0.0] + list(ref_pct) + [1.0]

    # strength만큼만 이동 + 단조 증가 보장 (톤 반전 방지)
    points: list[tuple[float, float]] = []
    prev_y = -1.0
    for x, y in zip(xs, ys):
        blended = x * (1.0 - strength) + y * strength
        blended = max(prev_y + 1e-4, min(1.0, blended))
        prev_y = blended
        points.append((round(float(x), 4), round(float(blended), 4)))

    # x가 겹치는 점 제거 (보간이 깨진다)
    deduped: list[tuple[float, float]] = []
    for x, y in points:
        if not deduped or x > deduped[-1][0] + 1e-4:
            deduped.append((x, y))
    return deduped if len(deduped) >= 3 else None


# ── 장면 판단 ──


def detect_scene(stats: dict[str, float], img: Image.Image) -> dict[str, bool]:
    """보정 기준을 바꿔야 하는 촬영 상황을 가려낸다.

    - backlit: 역광. 배경만 밝고 피사체가 어둡다 → 쉐도우를 크게 올려야 한다
    - low_light: 저조도/야간 → 노이즈를 더 잡고 대비를 과하게 올리지 않는다
    """
    small = img.convert("L")
    w, h = small.size
    if max(w, h) > 320:
        r = 320 / max(w, h)
        small = small.resize((max(1, int(w * r)), max(1, int(h * r))), Image.BILINEAR)
    luma = np.asarray(small, dtype=np.float32) / 255.0

    # 가운데 절반(피사체가 있을 자리)과 바깥 테두리의 밝기 차
    gh, gw = luma.shape
    center = luma[gh // 4: gh * 3 // 4, gw // 4: gw * 3 // 4]
    border = np.concatenate([
        luma[: gh // 8].ravel(), luma[gh * 7 // 8:].ravel(),
        luma[:, : gw // 8].ravel(), luma[:, gw * 7 // 8:].ravel(),
    ])
    backlit = bool(
        center.mean() < border.mean() - 0.14
        and border.mean() > 0.55
        and center.mean() < 0.45
    )

    low_light = bool(stats["brightness"] < 0.32 and stats["noise"] > 3.0)

    if backlit or low_light:
        log.info("scene: backlit=%s low_light=%s (center=%.2f border=%.2f)",
                 backlit, low_light, center.mean(), border.mean())
    return {"backlit": backlit, "low_light": low_light}


# ── 스타일 프로필 정규화 ──
#
# 프로필을 만드는 쪽(ANALYZE_USER_PROMPT)과 읽는 쪽(이 파일), 그리고 앱의
# 수동 편집 화면이 서로 다른 어휘를 써 왔다. 생성 쪽을 읽는 쪽 어휘에 맞췄지만,
# Firebase에 이미 저장된 프로필은 옛 어휘로 남아 있다. 여기서 흡수한다.

_LEGACY_FILTER = {"heavy": "strong"}          # 생성 쪽에만 있던 값
_LEGACY_TONE = {"warm": "warm", "cool": "cool", "neutral": "neutral", "mixed": "mixed"}


def normalize_style_profile(profile: dict[str, Any] | None) -> dict[str, Any]:
    """옛 어휘로 저장된 프로필을 현재 어휘로 옮긴다 (원본은 건드리지 않는다).

    조용히 폴백시키면 사용자가 고른 성향이 전부 '보통'으로 뭉개진다.
    실제로 filterTendency='heavy'가 게인 0.8(auto)로 떨어지고 있었다.
    """
    if not profile:
        return {}

    out = dict(profile)
    color = dict(out.get("colorPreference") or {})
    editing = dict(out.get("editingStyle") or {})

    filt = editing.get("filterTendency")
    if filt in _LEGACY_FILTER:
        log.info("profile: filterTendency %r → %r", filt, _LEGACY_FILTER[filt])
        editing["filterTendency"] = _LEGACY_FILTER[filt]

    # 옛 프로필은 대비를 editingStyle.contrastLevel에 담았다
    if "contrast" not in color and editing.get("contrastLevel"):
        color["contrast"] = editing["contrastLevel"]

    tone = color.get("preferredTones")
    if tone and tone not in _TONE_TARGETS:
        # colorTemperature(옛 필드)라도 있으면 그쪽을 쓴다
        fallback = color.get("colorTemperature")
        color["preferredTones"] = _LEGACY_TONE.get(tone) or (
            fallback if fallback in _TONE_TARGETS else "neutral"
        )

    out["colorPreference"] = color
    out["editingStyle"] = editing
    return out
