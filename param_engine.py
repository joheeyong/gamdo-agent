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
from typing import Any

import cv2
import numpy as np
from PIL import Image

log = logging.getLogger("gamdo-agent")

# 측정 시 이미지를 이 크기로 줄인다 — 통계값은 해상도에 거의 무관하다.
_MEASURE_MAX_PX = 512


# ── 이미지 측정 ──


def measure_image_stats(img: Image.Image) -> dict[str, float]:
    """사진의 실제 상태를 측정한다. 모든 값은 0~1 (warmth만 -1~1).

    - brightness: Rec.709 휘도 평균
    - contrast: 휘도의 5~95 백분위 폭 (표준편차보다 극단값에 덜 흔들린다)
    - saturation: HSV 채도 평균
    - warmth: R-B 균형. 양수면 웜톤
    - highlight_clip / shadow_crush: 날아간·뭉갠 픽셀 비율
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

    return {
        "brightness": float(luma.mean()) / 255.0,
        "contrast": float(p95 - p5) / 255.0,
        "saturation": saturation,
        "warmth": float(r.mean() - b.mean()) / 128.0,
        "highlight_clip": float((luma > 250).mean()),
        "shadow_crush": float((luma < 6).mean()),
        "haze": haze,
        # 라플라시안 분산 500 정도면 충분히 선명한 사진으로 본다
        "sharpness": min(1.0, lap_var / 500.0),
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


def _clamp(v: float, lo: float = -1.0, hi: float = 1.0) -> float:
    return round(max(lo, min(hi, v)), 3)


def _band(v: float, limit: float = _CORRECTION_BAND) -> float:
    """측정 기반 교정값을 ±limit로 묶는다."""
    return max(-limit, min(limit, v))


def build_recommended_params(
    img: Image.Image,
    style_profile: dict[str, Any] | None,
    analysis: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """[build_params_with_comment]에서 파라미터만 꺼내는 단축 함수."""
    params, _ = build_params_with_comment(img, style_profile, analysis)
    return params


def build_params_with_comment(
    img: Image.Image,
    style_profile: dict[str, Any] | None,
    analysis: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], str]:
    """측정값 + 프로필 + 모델의 스타일 방향으로 recommendedParams와 설명을 만든다.

    파라미터의 형식은 기존에 모델이 내려주던 recommendedParams와 동일해서
    [analysis_to_transform_params]가 그대로 소비할 수 있다.

    설명은 왜 이 값이 나왔는지를 한 문장으로 적은 것이다. 값을 정한 근거가
    여기 다 있으므로 모델에게 따로 물어볼 필요가 없다.
    """
    profile = style_profile or {}
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

    # ── 측정값 ↔ 목표값 차이로 노출계 3형제를 정한다 ──
    target_brightness = _BRIGHTNESS_TARGETS[_level_index(color_pref.get("brightnessTendency"))]
    target_contrast = _CONTRAST_TARGETS[_level_index(color_pref.get("contrast"))]
    target_saturation = _SATURATION_TARGETS[_level_index(color_pref.get("saturationTendency"))]
    target_warmth = _TONE_TARGETS.get(color_pref.get("preferredTones") or "neutral", 0.0)

    # 차이를 슬라이더 범위로 옮기는 배율. 측정 스케일과 슬라이더 스케일이 달라 실측으로 맞춘 값들이다.
    #
    # 측정 성분은 ±_CORRECTION_BAND로 묶는다. 측정은 "현재 사진을 목표 쪽으로 당기는"
    # 교정이지 스타일이 아니다. 밴드가 없으면 거의 무채색인 사진 한 장이
    # saturation +1.0 같은 값을 만들어 이미지를 태워버린다. 스타일의 세기는
    # 아래 레시피가 담당한다.
    brightness = _band((target_brightness - stats["brightness"]) * 2.2)
    contrast = _band((target_contrast - stats["contrast"]) * 1.6)
    saturation = _band((target_saturation - stats["saturation"]) * 1.8)
    temperature = _band((target_warmth - stats["warmth"]) * 1.2)

    # 레시피의 방향성을 더한다 (트렌드 → 피사체 순으로 덮어씀)
    def pick(key: str, default: float = 0.0) -> float:
        if key in subject_recipe:
            return float(subject_recipe[key])
        return float(recipe.get(key, default))

    saturation += float(recipe.get("saturation", 0.0)) + float(subject_recipe.get("saturation", 0.0))
    temperature += float(recipe.get("temperature", 0.0)) + float(subject_recipe.get("temperature", 0.0))
    contrast += float(subject_recipe.get("contrast", 0.0))

    shadows = float(recipe.get("shadows", 0.0))
    highlights = float(recipe.get("highlights", 0.0))

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

    # 안개 제거는 풍경에서 실제로 뿌옇게 측정될 때만
    dehaze = 0.0
    if subject_recipe.get("use_haze") and stats["haze"] > 0.25:
        dehaze = min(0.5, (stats["haze"] - 0.25) * 1.6)

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
        # 피부 보정은 사용자가 고른 강도 그대로 — 필터 게인을 곱하지 않는다
        "blemish_removal": _clamp(blemish, 0.0, 1.0),
        "skin_smoothing": _clamp(skin_smoothing, 0.0, 1.0),
        "toneCurve": {
            "preset": tone_preset,
            "strength": _clamp(float(tone_strength) * gain, 0.0, 1.0),
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
        params["reshapeParams"] = reshape

    log.info(
        "param_engine: subject=%s trend=%s gain=%.2f | measured b=%.2f c=%.2f s=%.2f w=%.2f haze=%.2f sharp=%.2f",
        subject or "-", trend or "-", gain,
        stats["brightness"], stats["contrast"], stats["saturation"],
        stats["warmth"], stats["haze"], stats["sharpness"],
    )
    return params, describe_params(stats, trend, subject, params, gain)


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
