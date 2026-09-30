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
import math
import os
from typing import Any

import cv2
import numpy as np
from PIL import Image

from image_processor import (
    _HSL_CHANNELS, _INPAINT_MAX_AREA, cast_pixel_weight, estimate_illuminant,
    estimate_noise_sigma, vivid_blue_weight,
)

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
    - warmth: R-B 균형(원색·피부·하늘을 덜어 낸 회색 면 기준). 양수면 웜톤
    - highlight_clip / shadow_crush: 날아간·뭉갠 픽셀 비율
    - highlight_p95: 밝은 끝(95백분위)의 위치. 하이라이트를 누를 여지가 있는지
    - shadow_p05: 어두운 끝(5백분위)의 위치. 쉐도우를 들어올릴 여지가 있는지
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
    # 선명한 하늘색 화소는 빼고 잰다. 파란 하늘(밝음)과 그늘(어두움)이 함께 파래서
    # 고른 파란 캐스트로 잡히고, 화이트밸런스가 하늘을 회색으로 만들었다.
    # 원색 옷·소품·피부도 뺀다 ([cast_pixel_weight]) — 코랄 재킷·붉은 국기가 화면을
    # 채운 인물이 '고른 웜 캐스트'(0.85)로 읽혀 피부가 청록·회색으로 식었다.
    cast_w = cast_pixel_weight(arr)
    use = cast_w > 0.25
    if use.mean() < 0.05:
        use = np.ones_like(use)
    rb = r - b
    lo, hi = np.percentile(luma[use], [33, 67])
    dark_px, bright_px = (luma <= lo) & use, (luma >= hi) & use

    def _wmean(sel: np.ndarray) -> float:
        wsum = float(cast_w[sel].sum())
        return float((rb[sel] * cast_w[sel]).sum()) / wsum / 128.0 if wsum > 1e-6 else 0.0

    warm_dark = _wmean(dark_px)
    warm_bright = _wmean(bright_px)
    if warm_dark * warm_bright <= 0:
        cast_uniformity = 0.0          # 부호가 다르면 조명 탓이 아니다
    else:
        lo_mag, hi_mag = sorted((abs(warm_dark), abs(warm_bright)))
        cast_uniformity = lo_mag / hi_mag if hi_mag > 1e-6 else 0.0

    return {
        "brightness": float(luma.mean()) / 255.0,
        "contrast": float(p95 - p5) / 255.0,
        "saturation": saturation,
        # 하늘을 뺀 R−B. 넓은 파란 하늘이 사진 전체를 '차갑다'고 읽히게 해서
        # 웜톤이 최대치(+0.2)까지 얹히고 하늘이 탁해졌다 (캐스트 판단과 같은 기준).
        # 원색·피부를 덜어 낸 가중 평균 — 옷 색이 아니라 빛의 색을 잰다.
        "warmth": float((rb * cast_w).sum()) / max(float(cast_w.sum()), 1e-6) / 128.0,
        "highlight_p95": float(p95) / 255.0,
        "shadow_p05": float(p5) / 255.0,
        "highlight_clip": float((luma > 250).mean()),
        # 밝은 화소(휘도 0.75 이상)·거의 흰 화소(0.90 이상)의 비율 — 하이키·흰 배경 판정용
        "bright_share": float((luma >= 0.75 * 255).mean()),
        "white_share": float((luma >= 0.90 * 255).mean()),
        "shadow_crush": float((luma < 6).mean()),
        "haze": haze,
        # 라플라시안 분산 500 정도면 충분히 선명한 사진으로 본다
        "sharpness": min(1.0, lap_var / 500.0),
        # 노이즈는 반드시 원본 해상도에서 잰다 — 축소하면 이웃 화소가 평균되어
        # 노이즈가 사라져 버린다. 비용을 아끼려 가운데 일부만 잘라 보되,
        # 512px 조각은 음식 접사처럼 가운데가 온통 질감인 사진에서 매끈한
        # 영역이 없어 질감을 노이즈로 읽는다. 2048px(12MP에서도 수십 ms)로 본다.
        "noise": estimate_noise_sigma(_center_crop(img, 2048)),
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

# 어둡지 않은 사진의 denoise 상한, 그리고 denoise 1.0일 때 레시피 그레인을 덜어 내는 비율
_DENOISE_CAP_BRIGHT = 0.5
_DENOISE_GRAIN_TRADE = 0.7
_GRAIN_LEVELS = {"none": 0.0, "subtle": 0.12, "moderate": 0.22, "heavy": 0.35, "film": 0.28}
_VIGNETTE_LEVELS = {"none": 0.0, "subtle": 0.10, "moderate": 0.20, "strong": 0.30}
_SKIN_LEVELS = {"none": 0.0, "light": 0.18, "moderate": 0.30, "heavy": 0.45}


def _level_index(value: str | None, default: int = 2) -> int:
    return _LEVEL5.get(value or "", default)


# ── 트렌드·피사체 레시피 ──
#
# 프롬프트에 산문으로 적혀 있던 규칙을 그대로 옮긴 것이다.
# 여기 없는 키는 0으로 본다.
#
# 키 설명 (값이 없으면 괄호 안의 기본 동작):
#   shadow_floor / highlight_ceiling — 어두운 끝(p5)·밝은 끝(p95)의 목표 위치
#   crush_blacks — "always": 바닥이 목표보다 떠 있으면 눌러 내린다 (깊은 그림자가
#                  스타일인 트렌드). "hazy": 사진이 실제로 뿌옇고 평평할 때만
#                  누른다 — 바랜 검정이 스타일인 필름 계열. (always)
#   contrast_target / saturation_target / warmth_target — 프로필에 해당 성향이 없거나
#                  사용자가 스타일을 직접 골랐을 때(styleSource=manual) 쓰는 측정 목표
#                  (medium 0.68 / 0.38 / 중립 0.0)
#   saturation_range / temperature_range — contrast_range와 같은 방식의 채도·색온도 교정 범위
#   contrast_range — 측정 기반 대비 교정의 (하한, 상한). 필름 계열은 대비를
#                    크게 세우지 않는다. (±밴드)
#   contrast / brightness — 측정과 무관하게 더하는 방향성
#   brightness_target — 프로필에 밝기 성향이 없거나 수동 선택일 때 쓰는 평균 밝기 목표
#                  (medium 0.50). 인물이면 얼굴 밝기 목표도 이만큼 옮긴다.
#   face_lift — 인물 얼굴(피부 L)을 원본보다 밝게 올려도 되는 폭 (0.04 ≈ L +4).
#                  플래시처럼 밝게 튀어나온 얼굴이 스타일인 트렌드만 넓힌다.
#   clarity / sharpness — 피사체 레시피 값에 더해진다
#   clarity_cap / sharpness_cap — 피사체 레시피가 더해진 뒤의 상한. 음식·풍경
#                    레시피가 필름 룩을 쨍한 HDR로 덮어쓰지 않게 한다. (0.25)
#   vignette — 스타일로서의 비네팅 (피사체 레시피에는 없다)
#   background_blur — 인물 배경 흐림 (없으면 0 — 기본으로는 걸지 않는다)
#   dark_contrast_relief — 어두운 장면(dark_scene_factor)에서 양수 대비를 덜어내는 비율 (0)
#   monochrome — True면 완전한 흑백 (saturation −1.0, 색 보정 전부 끔)
#   split — 스플릿 토닝 {"shadow": (hue, 세기), "highlight": (hue, 세기)}

_TREND_RECIPES: dict[str, dict[str, Any]] = {
    "warm_film": {
        "warmth_target": 0.12, "saturation_target": 0.28, "saturation_range": (-0.20, 0.04),
        "temperature_range": (-0.06, 0.25),
        "temperature": 0.16, "shadow_floor": 0.12, "highlight_ceiling": 0.91, "saturation": -0.14,
        "crush_blacks": "hazy", "contrast_target": 0.62, "contrast_range": (-0.20, 0.08),
        "clarity_cap": 0.05, "sharpness_cap": 0.08,
        "tone_curve": ("film", 0.60), "grain": 0.20,
        # 필름 인화의 색 분리: 호박색 하이라이트 + 청록 그림자. 예전 (255 0.12 / 35 0.15)은
        # 게인 뒤 a·b 2~3 시프트라 기본 레시피와 거의 같았다 (평균 ΔE 3.2).
        # 전역 웜톤은 골든아워(채도·온도)와 겹치므로, 따뜻함을 하이라이트 쪽에 싣는다.
        "split": {"shadow": (195, 0.34), "highlight": (45, 0.38)},
    },
    "korean_gamsung": {
        # 들린 바닥 + 맑은 흰색 + 순한 채도. 예전(천장 0.89·대비 하한 −0.25·채도 −0.44)은
        # 전역 대비를 깎아 양 끝이 같이 눌리고(p5 18·p99 85) 채도가 ×0.84, 피부 채도가
        # ×0.71이 되어 "회색 필름"이었다. 바닥만 커브로 띄우고 흰색은 거의 그대로 둔다.
        # 피부의 혈색이 남도록 식히는 쪽 색온도를 좁게 묶는다 (코랄 재킷 인물 −0.22).
        "warmth_target": 0.04, "saturation_target": 0.26, "saturation_range": (-0.09, 0.0),
        "temperature_range": (-0.06, 0.15),
        "temperature": 0.0, "shadow_floor": 0.13, "highlight_ceiling": 0.95, "saturation": -0.10,
        "crush_blacks": "hazy", "contrast_target": 0.60, "contrast_range": (-0.08, 0.05),
        "clarity": -0.08, "clarity_cap": 0.0, "sharpness_cap": 0.06,
        "tone_curve": ("gamsung", 0.75), "grain": 0.10,
        # 뽀얀 톤: 민트빛 그림자 + 복숭아빛 하이라이트를 옅게
        "split": {"shadow": (185, 0.26), "highlight": (15, 0.20)},
    },
    "cinematic_moody": {
        "temperature": 0.05, "shadow_floor": 0.04, "highlight_ceiling": 0.93, "saturation": -0.10,
        "clarity": 0.20, "vignette": 0.22, "tone_curve": ("high_contrast", 0.50), "grain": 0.25,
        "split": {"shadow": (210, 0.30), "highlight": (30, 0.22)},
    },
    # 예전(2010년대 후반) 유행 — 유지하되 새 사용자에게 권하지는 않는다
    "bright_airy": {
        # 밝은 커브를 제대로 태워(0.40 → 0.85) 중간톤을 띄우고 대비를 낮춘다.
        # 예전에는 깔끔(clean_minimal)과 평균 ΔE 3.1 — 거의 같은 사진이었다.
        # 밝기 목표(0.60)는 사진마다 노출을 맞추는 값이고, 커브가 이미 중간톤을 띄우므로
        # 고정 밝기 가산(+0.05)은 두지 않는다 — 둘을 합치면 밝기가 이중으로 오른다.
        "saturation_target": 0.30, "temperature_range": (-0.08, 0.15),
        "brightness_target": 0.60, "face_lift": 0.06,
        "temperature": 0.06, "shadow_floor": 0.15, "highlight_ceiling": 0.97, "saturation": -0.14,
        "crush_blacks": "hazy", "contrast_target": 0.64, "contrast_range": (-0.06, 0.04),
        "clarity_cap": 0.05, "sharpness_cap": 0.08,
        "vignette": 0.0, "tone_curve": ("bright", 0.85), "grain": 0.05,
    },
    "golden_hour": {
        "warmth_target": 0.20,
        "temperature": 0.22, "shadow_floor": 0.09, "highlight_ceiling": 0.93, "saturation": 0.0,
        "crush_blacks": "hazy", "clarity_cap": 0.08, "sharpness_cap": 0.10,
        "tone_curve": ("film", 0.45), "grain": 0.12,
        "split": {"shadow": (270, 0.12), "highlight": (40, 0.15)},
    },
    "clean_minimal": {
        # 뉴트럴: 색을 얹지 않고(온도 상수 0) 진한 검정·맑은 흰색, 그레인 없음.
        "warmth_target": 0.0, "temperature_range": (-0.12, 0.10), "brightness_target": 0.55,
        "temperature": 0.0, "shadow_floor": 0.03, "highlight_ceiling": 0.97, "saturation": 0.0,
        "contrast_target": 0.74,
        "clarity": 0.05, "vignette": 0.0, "tone_curve": ("linear", 0.0), "grain": 0.0,
    },
    # 정면 플래시 스냅 / Y2K 디카: 밝게 튀어나온 피사체, 깊은 바닥, 또렷한 로컬 대비,
    # 뉴트럴~쿨한 화이트밸런스, 가장자리가 떨어지는 조명, 디지털 노이즈 같은 그레인.
    "flash_digicam": {
        "warmth_target": -0.02, "saturation_target": 0.40,
        "temperature": -0.12, "shadow_floor": 0.03, "highlight_ceiling": 0.97, "saturation": 0.04,
        "contrast_target": 0.80, "contrast": 0.08,
        "clarity": 0.16, "sharpness": 0.08, "clarity_cap": 0.30,
        "vignette": 0.24, "tone_curve": ("flash", 0.60), "grain": 0.18,
        "wb_boost": 0.25, "face_lift": 0.07,
    },
    # 핑크·피치 파스텔: 들린 그림자, 낮은 대비, 부드러운 하이라이트.
    "soft_pastel": {
        "warmth_target": 0.04, "saturation_target": 0.30, "saturation_range": (-0.15, 0.0),
        # 흰색은 맑게(천장 0.96), 바닥은 커브로만 띄운다. 예전에는 전역 대비 −0.36까지
        # 깎아 검정이 우유빛(p1 22)·흰색이 회색(p99 83)이 됐다.
        "temperature": 0.03, "shadow_floor": 0.12, "highlight_ceiling": 0.96, "saturation": -0.10,
        "crush_blacks": "hazy", "contrast_target": 0.56, "contrast_range": (-0.16, 0.0),
        "contrast": -0.03, "brightness": 0.06,
        "clarity": -0.10, "clarity_cap": -0.02, "sharpness_cap": 0.04,
        "vignette": 0.0, "tone_curve": ("pastel", 0.90), "grain": 0.06,
        "temperature_range": (-0.08, 0.15),
        "split": {"shadow": (335, 0.42), "highlight": (5, 0.36)},
    },
    # 흑백 + 필름 그레인. 대비는 적당히, 그레인은 눈에 보이게.
    "bw_grain": {
        # 전역 대비는 화면 평균을 축으로 늘인다. 어두운 배경 인물에서는 축이 바닥 근처라
        # 얼굴만 위로 밀려 하얗게 떴다 (+0.18~0.31 → 피부 L 63 → 77). 어두운 장면에서만
        # 양수 대비를 덜어낸다 (dark_contrast_relief) — 풍경 흑백의 대비는 그대로.
        "monochrome": True, "shadow_floor": 0.04, "highlight_ceiling": 0.95,
        "contrast_target": 0.74, "contrast": 0.04, "dark_contrast_relief": 0.7,
        "clarity": 0.08, "vignette": 0.12,
        "tone_curve": ("bw", 0.60), "grain": 0.30,
    },
}

# 트렌드를 모를 때(스타일 프로필 없음·custom) 쓰는 2026 공통 베이스라인
# "소프트 필름 내추럴": 살짝 바랜 검정, 순한 대비, 옅은 웜톤, 고운 그레인.
#
# 예전 기본값은 S커브 + 바닥 누르기였다. 스타일 프로필이 없는 사용자는 전부
# 이 레시피를 탔는데 (실측 38건 중 38건), 평균 shadows −0.18·contrast +0.18·
# clarity +0.10·sharpness +0.17로 대부분의 사진이 쨍하고 무거워졌다 — 2026년의
# 아날로그 필름·뮤트 톤 흐름과 정반대다.
_DEFAULT_RECIPE: dict[str, Any] = {
    "temperature": 0.07, "shadow_floor": 0.07, "highlight_ceiling": 0.93, "saturation": -0.08,
    "crush_blacks": "hazy", "contrast_target": 0.62, "contrast_range": (-0.12, 0.08),
    "warmth_target": 0.06, "saturation_target": 0.32, "saturation_range": (-0.15, 0.10),
    "temperature_range": (-0.10, 0.20),
    "clarity_cap": 0.05, "sharpness_cap": 0.08,
    "tone_curve": ("soft_film", 0.65), "grain": 0.16,
}

# 흑백 트렌드 이름 (describe_params·테스트가 참조)
_MONO_TRENDS = frozenset(k for k, v in _TREND_RECIPES.items() if v.get("monochrome"))

# 비네팅은 피사체 레시피에 두지 않는다 — 스타일(트렌드)이나 사용자 취향
# (vignettePreference)이 원할 때만 건다. 예전에는 피사체마다 0.03~0.18을 넣어
# 모든 사진의 모서리가 어두워졌다 (실측 7장 전부 0.09~0.14, ΔE 2~3).
# 2026년의 내추럴 톤에서는 스타일이 아닌 비네팅이 오래된 필터처럼 보인다.
_SUBJECT_RECIPES: dict[str, dict[str, Any]] = {
    # 인물의 톤 커브는 트렌드(또는 기본 레시피)가 정한다. 예전에는 여기서 S커브 0.30을
    # 강제해 기본 레시피 인물 사진이 전부 S커브를 탔다.
    "인물": {
        "clarity": -0.02, "sharpness": 0.05,
        "blemish_removal": 0.35, "skin_smoothing": 0.28, "dehaze": 0.0,
    },
    "풍경": {"clarity": 0.18, "sharpness": 0.12, "use_haze": True},
    # 음식은 질감이 맛이라 필름 룩에서도 조금 더 또렷하게 남긴다 (texture_bonus)
    "음식": {
        "clarity": 0.25, "sharpness": 0.22,
        "saturation": 0.06, "temperature": 0.15, "texture_bonus": 0.06,
    },
    "카페/일상": {"clarity": -0.10, "contrast": -0.05},
    "사물": {"clarity": 0.15, "sharpness": 0.10},
    "동물": {"clarity": 0.12, "sharpness": 0.15},
    "혼합": {},
}


# 측정에서 나온 교정 성분 하나가 낼 수 있는 최대치
_CORRECTION_BAND = 0.35

# temperature_range가 없는 레시피의 측정 색온도 하한. 예전에는 ±밴드(−0.35)까지
# 열려 있어, 따뜻하게 읽힌 인물(실측 코랄 재킷)에 −0.24~−0.33이 걸려 피부가
# 잿빛이 됐다. 스타일의 쿨 성향은 레시피 상수(flash_digicam −0.12)로 따로 더해진다.
_TEMP_COOL_FLOOR = -0.12

# 측정분에 레시피 상수를 더한 뒤의 상한. 게인을 곱하기 전에 한 번 더 묶는다.
_STYLE_BAND = 0.45

# 인물 사진에서 측정분(목표 채도와의 차이)의 채도 낮추기 중 남기는 몫 ([compute] 참고)
_PORTRAIT_DESAT_MEASURE_KEEP = 0.5

# 목표 warmth가 이 이상이면(웜 필름·골든아워·웜 취향) 스타일이 웜톤을 요구하는 것으로
# 보고 장면의 빛을 이유로 웜톤을 덜어내지 않는다.
_WARM_STYLE_TARGET = 0.12
# WB 뒤 warmth가 이 구간에 있으면 양수 temperature를 선형으로 덜어낸다 (이미 따뜻한 사진)
_WARM_SCENE_LO, _WARM_SCENE_HI = 0.15, 0.40
# WB 뒤 warmth가 이 구간(음수 크기)이면 고르지 않은 쿨 캐스트를 의도된 빛으로 본다
_COOL_SCENE_LO, _COOL_SCENE_HI = 0.04, 0.12
# cast_uniformity가 이 구간 위면 조명 캐스트로 본다. 곱셈형 캐스트는 어두운 곳에서
# R-B 차가 작아 고름이 0.3~0.4로 나온다 (실측 상점가 인물 0.65, 새벽 호수 0.09).
_CAST_UNIFORM_LO, _CAST_UNIFORM_HI = 0.10, 0.40

# 전역 웜톤이 이만큼 걸리면 피부 채널(orange) 채도 부스트·얼굴 영역 temperature를
# 덜어낸다 — 같은 피부에 웜톤이 세 번 쌓이지 않게 ([_damp_warm_stacking]).
_STACK_T_LO, _STACK_T_HI = 0.05, 0.20
_STACK_ORANGE_DAMP = 0.6

# 흰 화소(휘도 250 초과)가 이 비율이면 색온도를 바닥까지 줄인다.
_WHITE_TEMPERATURE_TOLERANCE = 0.25
# 그래도 남기는 최소 비율 — 취향을 완전히 버리지는 않는다.
_WHITE_TEMPERATURE_FLOOR = 0.35

# 쉐도우 바닥 / 하이라이트 천장 차이(0~1)를 슬라이더 값으로 옮기는 배율.
#
# 4.2였을 때는 차이가 0.083만 넘으면 밴드(±0.35)에 박혔다. 실제 사진의 차이는
# 0.01~0.36 범위라, 스윕 1,960건에서 shadows가 78.6%, highlights가 67.9%
# 상한에 붙었다 — "측정 기반"이라면서 사실상 사진과 무관한 상수였다.
# 1.5면 차이 0.23까지 비례하므로 대부분의 사진에서 값이 실제로 움직인다.
_SHADOW_LIFT_GAIN = 1.5

# 하이라이트도 같은 이유로 같은 배율을 쓴다 ([_SHADOW_LIFT_GAIN] 참고)
_HIGHLIGHT_GAIN = 1.5

# 날아간 화소가 이 비율에 이르면 하이라이트를 끌어내리지 않는다.
# 눌러도 디테일은 돌아오지 않고 흰색만 회색이 된다.
_HIGHLIGHT_CLIP_TOLERANCE = 0.12

# 노출은 다른 축보다 좁게 잡는다. 평균 휘도는 장면마다 정당하게 다르다 —
# 설경·흰 벽 카페·역광은 원래 높고 야경은 원래 낮다. 목표 평균에 억지로
# 맞추면 잘 찍은 밝은 사진이 전부 중간 회색으로 눌린다.
# 목표에서 이 폭 안이면 노출이 맞은 것으로 보고 손대지 않는다.
_EXPOSURE_DEADZONE = 0.06
# 데드존을 벗어났을 때 노출 교정이 낼 수 있는 최대치
_EXPOSURE_BAND = 0.20

# 하이키(밝은 화소 비율이 이 구간 위) 사진은 평균이 높아도 노출을 내리지 않는다.
# 흰 접시 음식·흰 벽 카페·흰 배경 인물은 원래 밝다. 평균 목표(0.50)를 좇으면
# 실측: 음식 L 80→71, 흰 배경 인물 78→73 — 흰색이 회색이 됐다 (bright_airy 포함).
_HIGH_KEY_LO, _HIGH_KEY_HI = 0.25, 0.50
# 거의 흰 화소(휘도 0.90 이상) 비율이 이 구간 위면 흰 배경으로 보고
# 하이라이트 천장·대비 측정으로 밝은 끝을 끌어내리지 않는다 (p99 98~100 → 91~93).
_WHITE_BG_LO, _WHITE_BG_HI = 0.03, 0.12

# 인물: 노출은 화면 평균이 아니라 얼굴 밝기로 정한다 (LAB L/100 중앙값).
# 검은 배경 인물(평균 L 7)은 평균으로 보면 "어둡다"라서 brightness +0.16·
# highlights +0.28·대비 +0.28이 걸려 얼굴이 L 62→86으로 날아갔다.
# 얼굴이 이 목표 ± 구간 안이면 노출이 맞은 것으로 본다. 위쪽은 넓게 둔다 —
# 밝은 피부·플래시 인물은 원래 밝다.
_FACE_L_TARGET = 0.62
_FACE_L_BELOW, _FACE_L_ABOVE = 0.07, 0.12
# 얼굴 밝기가 원본보다 이만큼 넘게 오르거나(레시피 face_lift로 조정) 내려가지 않게,
# 톤 파라미터를 합친 결과를 작은 사본으로 시뮬레이션해 넘치는 성분을 줄인다.
_FACE_LIFT_CAP = 0.04
_FACE_DROP_CAP = 0.05
# 얼굴 감지·시뮬레이션 해상도 (긴 변)
_FACE_DETECT_PX = 1024
_FACE_SIM_PX = 384

# 화이트밸런스 강도 1.0이 실제로 중화하는 비율 (채널 게인 상한 때문에 1은 아니다)
_AWB_NEUTRALIZE = 0.8

# 캐스트가 완전히 고를 때의 자동 화이트밸런스 세기
_AWB_BASE = 0.65
# 캐스트 방향이 사용자 취향과 일치할 때 교정을 덜어내는 비율.
# 1.0이면 취향과 같은 방향의 캐스트는 아예 건드리지 않는다.
_AWB_TASTE_RELIEF = 0.85


def _clamp(v: float, lo: float = -1.0, hi: float = 1.0) -> float:
    """범위 안으로 자른다. NaN·무한은 0(보정 없음)으로 본다.

    min/max는 NaN을 만나면 다른 쪽 인자를 그대로 돌려준다 — min(1.0, nan)은 1.0이다.
    그래서 모델이 NaN을 보내면 값이 버려지는 게 아니라 **최대 보정**이 됐다.
    실측: {"face_slim": NaN} → 0.5(상한), hslAdjust 전 채널 NaN → 전부 상한.
    json.loads가 표준 밖의 NaN·Infinity 리터럴을 그대로 받아들이므로 실제로 닿는다.
    """
    try:
        v = float(v)
    except (TypeError, ValueError):
        return 0.0
    if not math.isfinite(v):
        return 0.0
    return round(max(lo, min(hi, v)), 3)


def _band(v: float, limit: float = _CORRECTION_BAND) -> float:
    """측정 기반 교정값을 ±limit로 묶는다. NaN은 0으로 본다 ([_clamp] 참고)."""
    try:
        v = float(v)
    except (TypeError, ValueError):
        return 0.0
    if not math.isfinite(v):
        return 0.0
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


def haze_flatness(stats: dict[str, float]) -> float:
    """사진이 실제로 뿌옇고 평평한 정도 (0~1).

    바닥(p5)이 높이 떠 있고 대비(p95-p5)가 낮을 때만 1에 가까워진다.
    바닥만 높은 사진(밝은 하이키·흰 배경)이나 대비만 낮은 사진(원래 어두운 장면)은
    뿌연 게 아니다. 다크 채널(stats["haze"])이 높으면 조금 더 믿는다.
    """
    lifted = _ramp(stats["shadow_p05"], 0.10, 0.24)
    flat = _ramp(0.66 - stats["contrast"], 0.0, 0.22)
    veil = _ramp(stats.get("haze", 0.0), 0.35, 0.65)
    return round(min(1.0, lifted * flat * (0.7 + 0.3 * veil) * 1.4), 3)


# 어두운 장면(야경·어두운 배경 인물)에서 바랜 검정을 덜어내는 비율.
# 밝은 사진에서 살짝 뜬 검정은 필름 느낌이지만, 화면 대부분이 어두운 사진에서는
# 같은 들림이 넓은 면적에 깔려 안개처럼 보인다. 실측: 야간 거리(sf) 기본 레시피
# p1 L 0.8 → 9.0 — 쉐도우 목표 + 커브 바닥 + 음수 대비가 한 방향으로 쌓였다.
# (노이즈가 적은 깨끗한 야경은 low_light로 잡히지 않아 밝기·바닥으로 따로 본다)
_DARK_SCENE_BRIGHT = (0.22, 0.32)   # 평균 휘도가 이 구간 아래로 갈수록 어두운 장면
_DARK_SCENE_FLOOR = (0.03, 0.08)    # p5가 이 구간 아래면 원래 검정이 깊은 사진
_DARK_FLOOR_RELIEF = 0.7            # 쉐도우 목표 바닥을 덜어내는 비율
_DARK_CURVE_RELIEF = 0.5            # 바닥을 띄우는 커브의 세기를 덜어내는 비율
_DARK_CONTRAST_RELIEF = 0.6         # 음수 대비(바닥을 들어 올림)를 덜어내는 비율
# 검정을 띄우는 톤 커브 프리셋 (x=0의 출력이 0보다 큰 것)
_LIFTED_CURVES = frozenset({"film", "fade", "bright", "soft_film", "pastel", "gamsung"})


def dark_scene_factor(stats: dict[str, float], low_light: bool = False) -> float:
    """사진이 어두운 장면인 정도 (0~1). 저조도로 판정됐으면 1."""
    if low_light:
        return 1.0
    dark = 1.0 - _ramp(stats["brightness"], *_DARK_SCENE_BRIGHT)
    deep = 1.0 - _ramp(stats["shadow_p05"], *_DARK_SCENE_FLOOR)
    return round(dark * deep, 3)

class _FaceTone:
    """노출 판단용 얼굴 피부 정보 — 작은 사본과 그 위의 피부 마스크."""

    __slots__ = ("small", "mask", "l0")

    def __init__(self, small: Image.Image, mask: np.ndarray, l0: float):
        self.small = small    # 시뮬레이션용 RGB 사본 (긴 변 _FACE_SIM_PX)
        self.mask = mask      # bool — 얼굴 피부 (눈·눈썹·입술 제외)
        self.l0 = l0          # 원본 피부 L 중앙값 (0~1)


def _face_l(img: Image.Image, mask: np.ndarray) -> float:
    lab_l = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2LAB)[..., 0]
    return float(np.median(lab_l[mask])) / 255.0


def _measure_face_tone(img: Image.Image) -> _FaceTone | None:
    """얼굴을 찾아 피부 밝기를 잰다. 없거나 실패하면 None (평균 기준으로 돌아간다)."""
    try:
        import image_processor as ip

        det = img.convert("RGB")
        if max(det.size) > _FACE_DETECT_PX:
            det.thumbnail((_FACE_DETECT_PX, _FACE_DETECT_PX), Image.BILINEAR)
        arr = np.ascontiguousarray(np.asarray(det))
        h, w = arr.shape[:2]
        with ip.MediaPipeCache() as cache:
            point_sets = ip._face_point_sets(arr, cache=cache)
        if not point_sets:
            return None
        mask = np.zeros((h, w), np.uint8)
        for pt in point_sets:
            mask = cv2.bitwise_or(mask, ip.build_face_skin_mask(pt, h, w))
        small = det.copy()
        small.thumbnail((_FACE_SIM_PX, _FACE_SIM_PX), Image.BILINEAR)
        m = cv2.resize(mask, small.size, interpolation=cv2.INTER_AREA) > 127
        if m.sum() < 30:
            return None
        return _FaceTone(small, m, _face_l(small, m))
    except Exception:
        log.exception("param_engine: face tone measurement failed")
        return None


_FACE_TONE_KEYS = ("brightness", "highlights", "shadows", "contrast")


def _simulate_face_l(face: _FaceTone, p: dict[str, float], tone: dict[str, Any]) -> float:
    """톤 파라미터를 작은 사본에 실제 렌더 함수로 적용해 얼굴 L을 잰다."""
    from image_processor import _apply_lab_adjustments

    out = _apply_lab_adjustments(
        face.small,
        highlights=p["highlights"], shadows=p["shadows"],
        tone_curve_preset=tone.get("preset") or "linear",
        tone_curve_strength=float(tone.get("strength") or 0.0),
        tone_curve_points=tone.get("points"),
        brightness=p["brightness"], contrast=p["contrast"], clarity=p["clarity"],
    )
    return _face_l(out, face.mask)


def _cap_face_tone(params: dict[str, Any], face: _FaceTone, lo: float, hi: float) -> None:
    """합친 톤 보정 뒤 얼굴 L이 [lo, hi]를 벗어나면 넘치게 만든 성분을 줄인다.

    brightness·highlights·대비는 각각 따로 보면 온건해도 한 얼굴에 겹친다.
    대비는 화면 평균을 축으로 늘리므로 검은 배경이면 얼굴(평균보다 훨씬 밝다)을
    크게 끌어올린다. 그래서 개별 값이 아니라 실제 렌더 결과로 판단한다.
    톤 커브는 스타일 자체라 건드리지 않고, 그것만으로도 넘치면 brightness로 되돌린다.
    """
    tone = params.get("toneCurve") or {}
    cur = {k: float(params.get(k, 0.0)) for k in _FACE_TONE_KEYS + ("clarity",)}
    f = _simulate_face_l(face, cur, tone)
    if lo <= f <= hi:
        return
    up = f > hi
    bound = hi if up else lo

    def ok(v: float) -> bool:
        return v <= bound if up else v >= bound

    base = dict(cur, **{k: 0.0 for k in _FACE_TONE_KEYS})
    f_base = _simulate_face_l(face, base, tone)
    # 얼굴을 넘치는 방향으로 민 성분만 줄인다
    culprits = []
    for k in _FACE_TONE_KEYS:
        if abs(cur[k]) < 1e-6:
            continue
        effect = _simulate_face_l(face, dict(base, **{k: cur[k]}), tone) - f_base
        if (effect > 0.002) if up else (effect < -0.002):
            culprits.append(k)

    def scaled(s: float) -> dict[str, float]:
        return dict(cur, **{k: cur[k] * s for k in culprits})

    if culprits:
        s_lo, s_hi = 0.0, 1.0
        if ok(_simulate_face_l(face, scaled(0.0), tone)):
            for _ in range(7):
                mid = (s_lo + s_hi) / 2
                if ok(_simulate_face_l(face, scaled(mid), tone)):
                    s_lo = mid
                else:
                    s_hi = mid
        cur = scaled(s_lo)
    if not ok(_simulate_face_l(face, cur, tone)):
        # 톤 커브·남은 성분만으로도 넘친다 — brightness로 되돌린다
        b0 = cur["brightness"]
        b_lo, b_hi = (b0, b0 - _EXPOSURE_BAND) if up else (b0, b0 + _EXPOSURE_BAND)
        for _ in range(7):
            mid = (b_lo + b_hi) / 2
            if ok(_simulate_face_l(face, dict(cur, brightness=mid), tone)):
                b_hi = mid
            else:
                b_lo = mid
        cur["brightness"] = b_hi
    log.info("param_engine: face L %.3f → capped to [%.3f, %.3f] — %s",
             f, lo, hi, {k: round(cur[k], 3) for k in _FACE_TONE_KEYS})
    for k in _FACE_TONE_KEYS:
        params[k] = _clamp(cur[k])


def build_recommended_params(
    img: Image.Image,
    style_profile: dict[str, Any] | None,
    analysis: dict[str, Any] | None = None,
    reference: dict[str, Any] | None = None,
    reshape_enabled: bool = False,
    skin_retouch_enabled: bool = True,
) -> dict[str, Any]:
    """[build_params_with_comment]에서 파라미터만 꺼내는 단축 함수."""
    params, _ = build_params_with_comment(
        img, style_profile, analysis, reference, reshape_enabled,
        skin_retouch_enabled=skin_retouch_enabled,
    )
    return params


def build_params_with_comment(
    img: Image.Image,
    style_profile: dict[str, Any] | None,
    analysis: dict[str, Any] | None = None,
    reference: dict[str, Any] | None = None,
    reshape_enabled: bool = False,
    skin_retouch_enabled: bool = True,
    face_detected: bool | None = None,
) -> tuple[dict[str, Any], str]:
    """측정값 + 프로필 + 모델의 스타일 방향으로 recommendedParams와 설명을 만든다.

    skin_retouch_enabled가 False면(앱의 '피부 보정' 토글) 전역 잡티·스무딩과
    analysis["regionParams"]["face"]의 잡티·스무딩을 모두 0으로 만든다.
    얼굴 영역 보정은 모델이 주고 서버와 앱이 analysis에서 그대로 꺼내 쓰므로
    analysis 쪽 값도 여기서 고쳐 둔다 (얼굴 밝기 등 톤 보정은 그대로 둔다).

    파라미터의 형식은 기존에 모델이 내려주던 recommendedParams와 동일해서
    [analysis_to_transform_params]가 그대로 소비할 수 있다.

    설명은 왜 이 값이 나왔는지를 한 문장으로 적은 것이다. 값을 정한 근거가
    여기 다 있으므로 모델에게 따로 물어볼 필요가 없다.

    face_detected: 질감 보정이 걸릴 얼굴이 사진에 있는지 (호출 측이 감지한 결과).
    False면 설명문에서 피부 문장을 뺀다 — 뒷모습·먼 전신처럼 얼굴이 없으면 피부
    보정은 아무것도 하지 않는데 "피부는 자연스럽게 정리했어요"라고 말하고 있었다.
    None(모름)이면 예전처럼 파라미터만 보고 판단한다.
    """
    profile = normalize_style_profile(style_profile)
    analysis = analysis or {}

    stats = measure_image_stats(img)

    color_pref = profile.get("colorPreference") or {}
    editing = profile.get("editingStyle") or {}
    trend = str(profile.get("trendCategory") or "").strip()
    subject = str(analysis.get("subjectType") or "").strip()
    manual = profile.get("styleSource") == "manual"

    # 알 수 없는 트렌드(custom·빈 값·앱이 새로 보낸 값)는 기본 레시피
    if trend and trend not in _TREND_RECIPES:
        log.info("param_engine: unknown trendCategory %r → default recipe", trend)
        trend = ""
    recipe = dict(_TREND_RECIPES.get(trend, _DEFAULT_RECIPE))
    subject_recipe = _SUBJECT_RECIPES.get(subject, {})
    mono = bool(recipe.get("monochrome"))

    # 사용자가 설정에서 스타일을 직접 골랐으면(styleSource=manual) 그 스타일이
    # 피드 측정보다 우선한다. 레퍼런스(대표 사진)는 "자동(내 피드 기준)"일 때의 목표다.
    if manual and trend and reference:
        log.info("param_engine: manual style %s — ignoring feed reference", trend)
        reference = None

    # 게인: 보정 강도 성향이 전체 세기를 정한다
    gain = _FILTER_GAIN.get(editing.get("filterTendency") or "auto", 0.8)

    # ── 측정값 ↔ 목표값 차이로 노출계 3형제를 정한다 ──
    if reference:
        # 사용자가 실제로 올리는 사진에서 잰 값. 카테고리 추정보다 정확하다.
        target_brightness = reference["brightness"]
        target_contrast = reference["contrast"]
        target_saturation = reference["saturation"]
        target_warmth = reference["warmth"]
    else:
        if color_pref.get("brightnessTendency") in _LEVEL5 and not (manual and "brightness_target" in recipe):
            target_brightness = _BRIGHTNESS_TARGETS[_level_index(color_pref.get("brightnessTendency"))]
        else:
            # 밝기 성향이 없거나 스타일을 직접 골랐으면 레시피 목표 (bright_airy는 더 밝게)
            target_brightness = float(recipe.get("brightness_target", _BRIGHTNESS_TARGETS[2]))
        if color_pref.get("contrast") in _LEVEL5 and not (manual and "contrast_target" in recipe):
            target_contrast = _CONTRAST_TARGETS[_level_index(color_pref.get("contrast"))]
        else:
            # 프로필에 대비 성향이 없으면(수동 선택·프로필 없음) 레시피가 정한다
            target_contrast = float(recipe.get("contrast_target", _CONTRAST_TARGETS[2]))
        # 프로필에 성향이 없거나(수동 선택·프로필 없음) 사용자가 스타일을 직접
        # 골랐으면 레시피의 목표를 쓴다. 중립 목표를 그대로 두면 웜 필름을 골라도
        # 따뜻한 사진이 차갑게 당겨지고, 뮤트 톤을 골라도 채도가 올라간다.
        if color_pref.get("saturationTendency") in _LEVEL5 and not (manual and "saturation_target" in recipe):
            target_saturation = _SATURATION_TARGETS[_level_index(color_pref.get("saturationTendency"))]
        else:
            target_saturation = float(recipe.get("saturation_target", _SATURATION_TARGETS[2]))
        if color_pref.get("preferredTones") in _TONE_TARGETS and not (manual and "warmth_target" in recipe):
            target_warmth = _TONE_TARGETS[color_pref["preferredTones"]]
        else:
            target_warmth = float(recipe.get("warmth_target", 0.0))

    # ── 자동 화이트밸런스 ──
    #
    # 고른 캐스트를 조명 탓으로 보고 중립으로 당긴다. 그런데 "고르다"는 것만으로는
    # 잘못 잡힌 화이트밸런스와 의도한 빛을 가릴 수 없다. 따뜻한 실내 조명·골든아워·
    # 촛불·텅스텐은 화면 전체를 고르게 물들이는데, 그게 그 사진의 빛이다.
    # 사용자가 원하는 방향과 같은 캐스트까지 걷어내면, 장면의 빛을 지운 자리에
    # temperature로 일반적인 색조를 덧칠하는 셈이 된다 — 두 보정이 서로 싸운다.
    #
    # 실측: 따뜻한 조명의 반려동물 사진(cast_uniformity 0.79)에서 auto_wb 0.51이
    # 걸려 LAB a가 9 내려가고 황금색 털이 회녹색이 됐다. 프로필은 slightly_warm,
    # 트렌드는 warm_film — 사용자가 원하는 바로 그 방향의 캐스트였다.
    #
    # 그래서 캐스트가 목표 방향과 일치하는 만큼 교정을 덜어낸다. 방향이 반대면
    # (쿨한 사진 + 웜 취향) 그대로 다 교정한다 — 그건 정말 고쳐야 할 캐스트다.
    #
    # 색온도 계산이 이 값을 참조하므로 그보다 먼저 정해져야 한다.
    taste_agreement = 0.0
    if stats["warmth"] * target_warmth > 0:
        taste_agreement = min(1.0, abs(stats["warmth"]) / abs(target_warmth))
    auto_wb_strength = round(
        _AWB_BASE * stats["cast_uniformity"] * (1.0 - _AWB_TASTE_RELIEF * taste_agreement),
        3,
    )
    # 플래시 스냅은 조명이 카메라 쪽 흰 빛 하나다 — 실내 조명의 캐스트를 더 걷어낸다
    if recipe.get("wb_boost"):
        auto_wb_strength = round(min(1.0, auto_wb_strength
                                     + float(recipe["wb_boost"]) * stats["cast_uniformity"]), 3)

    # ── 장면에 따라 기준을 바꾼다 ──
    scene = detect_scene(stats, img)
    dark = dark_scene_factor(stats, scene["low_light"])

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
    # 하이키·흰 배경 정도 (0~1)
    high_key = _ramp(stats.get("bright_share", 0.0), _HIGH_KEY_LO, _HIGH_KEY_HI)
    white_bg = max(high_key, _ramp(stats.get("white_share", 0.0), _WHITE_BG_LO, _WHITE_BG_HI))
    # 인물이면 노출은 얼굴이 정한다. 목표 평균의 성향(레시피·프로필)만큼 얼굴 목표도 옮긴다.
    face = (_measure_face_tone(img)
            if subject == "인물" and gain > 0.0 else None)
    face_target = _FACE_L_TARGET + (target_brightness - _BRIGHTNESS_TARGETS[2]) * 0.5
    if face is not None:
        want = min(max(face.l0, face_target - _FACE_L_BELOW), face_target + _FACE_L_ABOVE)
        if abs(want - face.l0) > 1e-3 and 0.02 < face.l0 < 0.98:
            # brightness는 L 감마(1/(1+b), 음수면 1-1.5b)다 — 얼굴을 want로 옮기는 값을 바로 푼다
            g = math.log(want) / math.log(face.l0)
            brightness = _band(1.0 / g - 1.0 if g <= 1.0 else (1.0 - g) / 1.5, _EXPOSURE_BAND)
        else:
            brightness = 0.0
        log.info("param_engine: face L %.3f target %.3f → brightness %.3f",
                 face.l0, face_target, brightness)
    elif brightness < 0:
        # 하이키는 원래 밝다 — 평균이 높다고 내리지 않는다
        brightness *= 1.0 - high_key
    contrast = _band((target_contrast - stats["contrast"]) * 1.6)
    # 필름 계열은 측정 차이가 커도 대비·채도를 크게 세우지(또는 죽이지) 않는다
    if "contrast_range" in recipe:
        lo, hi = recipe["contrast_range"]
        contrast = max(float(lo), min(float(hi), contrast))
    saturation = _band((target_saturation - stats["saturation"]) * 1.8)
    if "saturation_range" in recipe:
        lo, hi = recipe["saturation_range"]
        saturation = max(float(lo), min(float(hi), saturation))
    # 인물 사진에서 "채도가 목표보다 높다"는 측정은 대개 옷·배경·소품의 색이다
    # (피부를 빼고 재도 값이 거의 같다 — 실측 c1 0.50→0.57, c2 0.44→0.44, meir 0.54→0.55).
    # 그건 고칠 결함이 아니라 찍은 사람이 고른 색이라, 측정분의 채도 낮추기는 절반만 둔다
    # (범위로 묶은 뒤에 — 먼저 줄이면 선명한 사진은 여전히 범위 바닥에 붙는다).
    # 전부 두면 기본 레시피 인물이 전부 바닥(−0.15)에 붙어 피부까지 잿빛이 됐다.
    # 레퍼런스(그 사람 피드의 실제 채도)가 있으면 그게 취향이라 그대로 따른다.
    if saturation < 0 and subject == "인물" and not reference:
        saturation *= _PORTRAIT_DESAT_MEASURE_KEEP
    # 화이트밸런스가 먼저 중립으로 당기므로, 그 뒤에 남는 웜니스를 기준으로 잡는다.
    # 원본 warmth를 그대로 쓰면 같은 편차를 두 번 보정하게 된다.
    warmth_after_wb = stats["warmth"] * (1.0 - _AWB_NEUTRALIZE * auto_wb_strength)
    temperature = _band((target_warmth - warmth_after_wb) * 1.2)
    # 필름 계열은 따뜻한 사진을 크게 식히지 않는다 — 피부가 회색으로 죽는다.
    # 범위가 없는 레시피도 식히는 쪽은 _TEMP_COOL_FLOOR까지만 (고른 캐스트는 auto_wb가 맡는다).
    lo, hi = recipe.get("temperature_range", (_TEMP_COOL_FLOOR, _CORRECTION_BAND))
    temperature = max(float(lo), min(float(hi), temperature))

    # 레시피의 방향성을 더한다 (트렌드 → 피사체 순으로 덮어씀)
    def pick(key: str, default: float = 0.0) -> float:
        if key in subject_recipe:
            return float(subject_recipe[key])
        return float(recipe.get(key, default))

    # 레시피 상수를 더한 뒤 스타일 밴드로 다시 묶는다.
    #
    # 예전에는 여기서 더한 값에 바로 게인(최대 1.3)을 곱했다. 트렌드와 피사체
    # 레시피가 같은 방향이면 상수만 0.47까지 쌓이고, 측정분 0.35가 더해진 뒤
    # 1.3배가 되어 temperature가 1.0에 박혔다 — 사진이 통째로 오렌지가 된다.
    # 실측: (fog, golden_hour, 음식, very_strong) → temperature +1.000.
    saturation = _band(
        saturation + float(recipe.get("saturation", 0.0))
        + float(subject_recipe.get("saturation", 0.0)), _STYLE_BAND)
    # 장면의 원래 빛을 존중한다 — 스타일이 웜톤을 요구하지 않을 때만(목표 warmth <
    # _WARM_STYLE_TARGET: 기본 레시피·뉴트럴 계열) 적용한다.
    #  - 이미 따뜻한 사진은 더 덥히지 않는다. 실측: 라테(warmth 0.35)에 음식 레시피
    #    0.25가 그대로 얹혀 temperature +0.18.
    #  - 차가운 쪽으로 기울었는데 캐스트가 고르지 않으면(cast_uniformity < 0.4) 조명
    #    탓이 아니라 장면의 색이다 — 새벽 호수·파란 수트·흐린 날. 덥히지 않는다.
    #    실측: 라벤더빛 새벽 호수(warmth −0.14, 고름 0.09)·파란 수트 인물 2장이
    #    전부 temperature +0.216을 받아 가장 큰 보정(ΔE 3~4.3)이 됐다.
    #    고른 캐스트(그늘·형광등)는 그대로 교정한다.
    temperature = _band(
        temperature + float(recipe.get("temperature", 0.0))
        + float(subject_recipe.get("temperature", 0.0)), _STYLE_BAND)
    # 덜어내기만 한다(양수 → 0 쪽). 식히는 값은 건드리지 않는다.
    if target_warmth < _WARM_STYLE_TARGET and temperature > 0:
        temperature *= 1.0 - _ramp(warmth_after_wb, _WARM_SCENE_LO, _WARM_SCENE_HI)
        cool_scene = (_ramp(-warmth_after_wb, _COOL_SCENE_LO, _COOL_SCENE_HI)
                      * (1.0 - _ramp(stats["cast_uniformity"], _CAST_UNIFORM_LO, _CAST_UNIFORM_HI)))
        temperature *= 1.0 - cool_scene
    contrast = _band(contrast + float(recipe.get("contrast", 0.0))
                     + float(subject_recipe.get("contrast", 0.0)), _STYLE_BAND)
    brightness = _band(brightness + float(recipe.get("brightness", 0.0)), _STYLE_BAND)

    # 흰 영역이 넓으면 "대비가 너무 높다"는 측정도 믿을 수 없다.
    #
    # contrast는 p95-p5로 재는데, 흰 배경이 p95를 1.0에 고정시키므로 흰 배경
    # 사진은 무조건 대비가 높게 나온다. 그걸 목표로 끌어내리면 배경의 흰색이
    # 회색으로 눌린다 — 노출을 고친 게 아니라 배경을 더럽힌 것이다.
    # 실측: 흰 배경 인물 사진에서 contrast -0.41이 걸려 배경이 회베이지가 됐다.
    if contrast < 0 and stats["highlight_clip"] > 0:
        contrast = round(contrast * max(
            _WHITE_TEMPERATURE_FLOOR,
            1.0 - stats["highlight_clip"] / _WHITE_TEMPERATURE_TOLERANCE), 3)
    # 날아가지 않은 흰 배경(휘도 0.90~0.98)도 p95를 끌어올려 같은 오판을 만든다
    if contrast < 0:
        contrast = round(contrast * (1.0 - white_bg), 3)

    # 흰 영역이 넓은 사진에서는 색온도를 덜 얹는다.
    #
    # 흰 배경 스튜디오 사진·흰 벽·흰 옷은 그 자체가 기준 백색이다. 그 위에 취향의
    # 웜톤을 그대로 올리면 색이 물들 것이 없는 영역이라 시프트가 100% 드러나,
    # 보정이 아니라 얼룩처럼 보인다. 실측: 흰 배경 인물 사진에 temperature
    # +0.47이 걸려 배경이 베이지가 됐다. 색이 있는 사진에서는 같은 값이
    # 자연스럽게 묻힌다 — 그래서 흰 화소 비율에 따라서만 줄인다.
    if stats["highlight_clip"] > 0:
        keep = max(_WHITE_TEMPERATURE_FLOOR,
                   1.0 - stats["highlight_clip"] / _WHITE_TEMPERATURE_TOLERANCE)
        temperature = round(temperature * keep, 3)

    # 쉐도우는 "무조건 들어올린다"가 아니라 "바닥이 목표보다 낮으면 올린다"다.
    # 트렌드 상수(+0.15~0.35)를 모든 사진에 붙이면, 이미 어두운 끝이 열려 있는
    # 사진까지 들려서 다이내믹 레인지가 눌리고 입체감이 사라진다.
    # 레퍼런스가 있으면 그 사람 사진의 실제 바닥을, 없으면 트렌드의 목표를 쓴다.
    if reference and reference.get("luma_percentiles"):
        target_floor = float(reference["luma_percentiles"][1])   # p5
    else:
        target_floor = float(recipe.get("shadow_floor", 0.08))
        target_floor *= 1.0 - _DARK_FLOOR_RELIEF * dark
    shadows = _band((target_floor - stats["shadow_p05"]) * _SHADOW_LIFT_GAIN)
    # 바닥이 목표보다 떠 있을 때 눌러 내릴지.
    #
    # 필름 계열(기본 레시피 포함)은 살짝 바랜 검정이 곧 스타일이다. 예전 기본값은
    # 떠 있는 바닥을 무조건 눌러(실측 평균 shadows −0.18) 모든 사진을 무겁게 만들었다.
    # 이제는 사진이 실제로 뿌옇고 평평할 때만 — 바닥이 높고(p5) 대비가 낮을 때만 —
    # 그 정도만큼 누른다. 레퍼런스(그 사람의 실제 바닥)가 있으면 그쪽을 믿는다.
    if (shadows < 0 and not reference
            and recipe.get("crush_blacks", "always") == "hazy"):
        shadows *= haze_flatness(stats)
    # 하이라이트도 쉐도우와 대칭으로 다룬다. 예전에는 "억제만" 했는데,
    # 밝은 끝이 낮아 흐릿한 사진은 눌러서 더 평평해질 뿐이다. 그런 사진은
    # 오히려 올려야 밝은 액센트가 생겨 입체감이 산다.
    if reference and reference.get("luma_percentiles"):
        target_ceiling = float(reference["luma_percentiles"][7])  # p95
    else:
        target_ceiling = float(recipe.get("highlight_ceiling", 0.94))
    highlights = _band(
        (target_ceiling - stats["highlight_p95"]) * _HIGHLIGHT_GAIN
    )
    # 밝은 끝을 끌어내리는 방향은 조심해야 한다. 이미 255에 붙은 화소는 눌러도
    # 디테일이 돌아오지 않고 흰색이 회색이 될 뿐이다 (아래 주석에 남아 있는,
    # 클리핑 보너스를 제거한 이유와 같다).
    #
    # 게다가 흰 배경 스튜디오 사진·하늘·흰 옷처럼 "원래 흰" 영역이 넓으면
    # p95는 노출과 무관하게 1.0으로 측정된다. 그러면 천장 목표를 좇는 것 자체가
    # 틀린 판단이 된다 — 흰 배경을 회색으로 만들라는 지시가 되어 버린다.
    #
    # 실측: 흰 배경 인물 사진(p95 1.0, 날아간 화소 38%)에서 highlights -0.46이
    # 걸려 배경이 탁한 갈색이 됐다. 날아간 비율만큼 이 방향을 덜어낸다.
    if highlights < 0 and stats["highlight_clip"] > 0:
        keep = max(0.0, 1.0 - stats["highlight_clip"] / _HIGHLIGHT_CLIP_TOLERANCE)
        highlights = round(highlights * keep, 3)

    if scene["backlit"]:
        # 역광: 피사체가 실루엣으로 남는다. 쉐도우를 크게 들어올리고
        # 날아간 배경을 눌러 준다.
        shadows += 0.25
        highlights -= 0.15
    if scene["low_light"]:
        # 저조도: 대비를 세우면 노이즈와 뭉갬이 같이 도드라진다
        contrast *= 0.6
    if contrast < 0:
        # 음수 대비는 평균을 축으로 검정을 들어 올린다 — 어두운 장면에서는 안개가 된다
        contrast *= 1.0 - _DARK_CONTRAST_RELIEF * dark
    else:
        contrast *= 1.0 - float(recipe.get("dark_contrast_relief", 0.0)) * dark

    # 하이키·흰 배경: 밝은 끝이 흰 배경 자체다. 천장 목표(역광 보정 포함)로 누르면
    # 흰색이 연회색이 될 뿐이다.
    if highlights < 0:
        highlights = round(highlights * (1.0 - white_bg), 3)

    # 날아간 하이라이트·뭉갠 쉐도우가 많으면 그만큼 더 되살린다
    # 클리핑 보너스는 제거했다. 천장 목표가 이미 "너무 밝다"를 다루는데
    # 밴드(±0.35) 밖에서 또 빼면 총 -0.43까지 가서 하이라이트가 회색으로
    # 뭉개졌다. 완전히 날아간(=255) 화소는 눌러도 디테일이 돌아오지 않는다.


    # 선명감·선명도는 트렌드 방향 + 피사체 방향을 더한 뒤 트렌드 상한으로 묶는다.
    # 상한이 없으면 음식(0.25/0.22)·풍경(0.18/0.12) 레시피가 필름 룩을 쨍한
    # HDR로 덮어쓴다. 음식은 질감이 맛이라 상한을 조금 풀어 준다.
    texture_bonus = float(subject_recipe.get("texture_bonus", 0.0))
    clarity = min(
        float(recipe.get("clarity", 0.0)) + float(subject_recipe.get("clarity", 0.0)),
        float(recipe.get("clarity_cap", 0.25)) + texture_bonus,
    )
    sharpness = min(
        float(recipe.get("sharpness", 0.0)) + float(subject_recipe.get("sharpness", 0.0)),
        float(recipe.get("sharpness_cap", 0.25)) + texture_bonus,
    )
    # 비네팅은 스타일의 일부인 트렌드(플래시·시네마틱)가 피사체 값보다 약해지지 않게
    vignette = pick("vignette")
    if "vignette" in recipe:
        vignette = max(float(recipe["vignette"]), vignette) if recipe["vignette"] > 0 else float(recipe["vignette"])

    # 흰 배경 사진에 비네팅을 얹으면 모서리의 흰색이 회색으로 죽는다.
    # 스튜디오 흰 배경·흰 벽에서는 스타일이 아니라 렌즈 결함처럼 보인다.
    if stats["highlight_clip"] > 0:
        vignette = round(vignette * max(
            _WHITE_TEMPERATURE_FLOOR,
            1.0 - stats["highlight_clip"] / _WHITE_TEMPERATURE_TOLERANCE), 3)

    # 이미 흐린 사진이면 선명도를 올리고, 충분히 선명하면 건드리지 않는다
    if stats["sharpness"] < 0.25:
        # 초점 결함 교정은 취향과 별개지만, 필름 계열에서는 반만 — 그레인이 얹히므로
        sharpness += 0.15 if "sharpness_cap" not in recipe else 0.08
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
    else:
        # 밝은 사진의 노이즈는 대개 옅다. 세게 지우면 음식·천·나뭇결 질감이
        # 먼저 사라져 플라스틱처럼 보이므로, 정말 어두운 사진이 아니면 절반까지만.
        denoise_strength = min(denoise_strength, _DENOISE_CAP_BRIGHT)

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
    # 노이즈를 지운 만큼 레시피 그레인을 덜어 낸다 — 지우고 다시 뿌리는 건
    # 모순이고, 뭉개진 면 위의 그레인은 필름이 아니라 노이즈로 읽힌다.
    # 사용자가 직접 고른 그레인 강도는 그대로 둔다.
    if grain_pref == "auto" and denoise_strength > 0.0:
        grain *= 1.0 - _DENOISE_GRAIN_TRADE * min(1.0, denoise_strength)

    vignette_pref = editing.get("vignettePreference") or "auto"
    if vignette_pref != "auto":
        vignette = _VIGNETTE_LEVELS.get(vignette_pref, 0.0)

    # 톤 커브는 트렌드(또는 기본 레시피)가 정한다 — 피사체보다 스타일이 상위다.
    tone_preset, tone_strength = recipe.get(
        "tone_curve", subject_recipe.get("tone_curve", ("linear", 0.0))
    )
    if tone_preset in _LIFTED_CURVES:
        tone_strength = float(tone_strength) * (1.0 - _DARK_CURVE_RELIEF * dark)

    # 레퍼런스가 있으면 프리셋 대신 그 사람 사진의 밝기 분포를 따라간다.
    # 프리셋은 "필름이면 이런 곡선"이라는 일반론이고, 이쪽은 그 사람의 실제 곡선이다.
    tone_points = build_reference_tone_curve(img, reference, 0.45 * gain)
    if tone_points:
        tone_preset, tone_strength = "reference", 1.0

    # 흑백: 레퍼런스 곡선은 컬러 사진의 밝기 분포라 흑백 커브를 대신하지 않는다
    if mono and tone_points:
        tone_points = None
        tone_preset, tone_strength = recipe["tone_curve"]

    split = recipe.get("split") or {}
    shadow_hue, shadow_str = split.get("shadow", (0, 0.0))
    hi_hue, hi_str = split.get("highlight", (0, 0.0))

    # 사용자가 "보정 없음"을 골랐으면 정말로 아무것도 하지 않는다.
    #
    # 게인이 0이면 취향 기반 값들은 전부 0이 되지만, "촬영 결함 교정은 취향과
    # 무관하다"는 이유로 게인을 곱하지 않는 항목들(auto_wb·denoise·잡티·스무딩)은
    # 그대로 남아 있었다. 실측: filterTendency="none" 392건 중 350건이 사진을
    # 바꿨고, 야간 인물에서는 denoise 1.0 + skin_smoothing 0.45 + 잡티 0.54가
    # 걸려 피부와 천 질감이 전부 사라졌다. 그런데 설명문은 "보정 없음 설정이라
    # 원본 톤을 그대로 두었어요"라고 말하고 있었다.
    if gain <= 0.0:
        auto_wb_strength = 0.0
        denoise_strength = 0.0
        blemish = 0.0
        skin_smoothing = 0.0
        dehaze = 0.0

    # 앱의 '피부 보정' 토글이 꺼져 있으면 피부 질감은 건드리지 않는다.
    if not skin_retouch_enabled:
        blemish = 0.0
        skin_smoothing = 0.0

    # 얼굴 영역 보정의 잡티·스무딩을 전역 값 하나로 합친다 (두 번 겹치지 않게).
    blemish, skin_smoothing = _merge_face_texture(
        analysis, blemish, skin_smoothing,
        enabled=skin_retouch_enabled and gain > 0.0 and skin_level != "none",
        fold=is_portrait,
    )

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
        # 배경 흐림은 기본으로 걸지 않는다 — 스타일이 원할 때(레시피의
        # background_blur)만, 인물에서만. 예전에는 인물 사진 전부에 0.25×게인을
        # 걸어 스튜디오 배경·건물까지 가짜 보케로 뭉갰다 (실측 인물 4장 전부 0.2).
        # 앱의 수동 편집 슬라이더는 이 값과 무관하게 그대로 동작한다.
        "background_blur": _clamp(
            float(recipe.get("background_blur", 0.0)) * gain if is_portrait else 0.0,
            0.0, 1.0),
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

    # 흑백: 채도를 끝까지 빼서(−1.0) 완전한 무채색으로 만든다. 게인을 곱하면
    # −0.8이 되어 색이 20% 남는다. 색온도·화이트밸런스·스플릿은 흑백에서 의미가 없고,
    # 남겨 두면 흑백 위에 색을 다시 칠한다 (HSL은 회색 화소를 빨강으로 본다).
    if mono and gain > 0.0:
        params["saturation"] = -1.0
        params["temperature"] = 0.0
        params["auto_wb"] = 0.0
        params["splitToning"] = {
            "shadow": {"hue": 0, "strength": 0.0},
            "highlight": {"hue": 0, "strength": 0.0},
        }

    # 색계열별 조정은 측정으로 나오지 않는 판단이라 모델 값을 그대로 쓴다.
    # 없으면 키를 넣지 않는다 — analysis_to_transform_params가 None으로 본다.
    hsl = None if mono else _clamp_hsl(analysis.get("hslAdjust"), gain)
    # 사진에 거의 없는 색·사진 대부분을 덮는 색(사실상 전역 채도)을 걸러낸다.
    hsl = _prune_hsl(hsl, img)
    # 영역 보정도 효과 없는 값·전역과 겹치는 값을 걸러 analysis에 되돌려 둔다
    # (서버와 앱이 analysis["regionParams"]를 그대로 꺼내 쓴다).
    _prune_region_params(analysis, params, is_portrait, gain)
    if hsl:
        params["hslAdjust"] = hsl
        log.info("param_engine: hslAdjust from model — %s", list(hsl.keys()))
    _damp_warm_stacking(params, analysis)
    if is_portrait:
        _guard_skin_cooling(params, img)

    # 얼굴/체형은 눈이 필요한 판단이라 모델이 인물 사진에서만 제안한다.
    # 스키마상 최상위에 오지만, 옛 응답 형식(recommendedParams 안)도 받아준다.
    reshape = analysis.get("reshapeParams")
    if not isinstance(reshape, dict):
        legacy = analysis.get("recommendedParams")
        if isinstance(legacy, dict):
            reshape = legacy.get("reshapeParams")
    # 몸의 비율을 바꾸는 변형은 사용자가 설정에서 켰을 때만 한다.
    # 토글이 앱에만 저장되고 서버로 오지 않아, 꺼 둔 사람도 체형이 변형됐다.
    if isinstance(reshape, dict) and is_portrait and reshape_enabled:
        params["reshapeParams"] = _clamp_reshape(reshape)

    # 얼굴 밝기가 합친 톤 보정으로 넘치게 오르거나 내려가지 않게 한다.
    # 원래 어둡거나 밝은 얼굴은 목표 구간까지는 움직일 수 있다.
    if face is not None:
        lift = float(recipe.get("face_lift", _FACE_LIFT_CAP))
        face_hi = max(face.l0, face_target - _FACE_L_BELOW) + lift
        face_lo = min(face.l0, face_target + _FACE_L_ABOVE) - _FACE_DROP_CAP
        _cap_face_tone(params, face, face_lo, face_hi)

    # 눈에 안 보이는 값은 0으로 — 렌더 단계를 건너뛰고, 앱의 '적용된 변형'에도
    # 실제로 한 보정만 남는다 (게인까지 곱한 최종값 기준).
    _drop_negligible(params, img)

    log.info(
        "param_engine: subject=%s trend=%s source=%s gain=%.2f | measured b=%.2f c=%.2f s=%.2f w=%.2f haze=%.2f sharp=%.2f",
        subject or "-", trend or "default", profile.get("styleSource") or "-", gain,
        stats["brightness"], stats["contrast"], stats["saturation"],
        stats["warmth"], stats["haze"], stats["sharpness"],
    )
    return params, describe_params(stats, trend, subject, params, gain, face_detected=face_detected)


# 이 값보다 작으면 결과가 눈에 띄게 달라지지 않는 파라미터의 최소 효과 크기.
#
# 실사진 7장(카페·음식·풍경·인물 4)에서 "그 값 vs 0"의 색차를 쟀다(1280px,
# float LAB ΔE76). 임계값 바로 아래는 평균 ΔE 0.5 이하·p99 1.6 이하 — 8비트 출력
# 양자화 수준이다. 예전 서버 응답에 흔했던 값: 선명감 ±0.016(ΔE 0.1~0.2),
# 선명도 0.04(평균 0.01~0.05), 채도 0.016, 색온도 −0.023, 밝기 0.01, 비네팅 0.024 미만.
# 효과가 세기에 비례하지 않는 항목(노이즈 제거·피부·배경 흐림·체형)과
# 모델이 정하는 HSL은 여기서 건드리지 않는다.
_MIN_EFFECT = {
    "brightness": 0.015,
    "contrast": 0.015,
    "saturation": 0.025,
    "temperature": 0.025,
    "highlights": 0.025,
    "shadows": 0.025,
    "clarity": 0.04,
    "sharpness": 0.06,
    "vignette": 0.02,
    "dehaze": 0.02,
}
# 톤 커브·스플릿 토닝 세기
_MIN_BLEND_EFFECT = 0.03
# apply_auto_white_balance는 모든 채널 게인이 1%p 안쪽이면 아무것도 하지 않는다
_AWB_NOOP_GAIN = 0.01


def _drop_negligible(params: dict[str, Any], img: Image.Image) -> None:
    """효과가 보이지 않는 값을 0으로 만든다 (params를 제자리에서 고친다)."""
    for key, floor in _MIN_EFFECT.items():
        if key in params and abs(params[key]) < floor:
            params[key] = 0.0

    tone = params.get("toneCurve") or {}
    if not tone.get("points") and float(tone.get("strength") or 0.0) < _MIN_BLEND_EFFECT:
        tone["strength"] = 0.0
    for side in (params.get("splitToning") or {}).values():
        if float(side.get("strength") or 0.0) < _MIN_BLEND_EFFECT:
            side["strength"] = 0.0

    # auto_wb는 세기가 아니라 "세기 × 이 사진의 색 틀어짐"이 효과다. 렌더러와
    # 같은 계산으로 모든 게인이 1% 안쪽(=렌더러가 건너뜀)이면 0으로 적는다.
    # 실측: 0.05~0.06은 7장 전부 이 경우였다(ΔE 0).
    strength = float(params.get("auto_wb") or 0.0)
    if 0.0 < strength:
        gains = estimate_illuminant(img)
        if all(abs(max(0.75, min(1.25, 1.0 + (g - 1.0) * strength)) - 1.0) < _AWB_NOOP_GAIN
               for g in gains):
            params["auto_wb"] = 0.0


# HSL 한 채널이 낼 수 있는 최대치. apply_hsl_adjust에서 1.0은 색상 ±30°,
# 채도·밝기는 ±80(255 기준)이라 그대로 태우면 색이 튄다.
_HSL_LIMITS = {"hue": 0.35, "saturation": 0.55, "lightness": 0.45}


def _clamp_hsl(raw: Any, gain: float) -> dict[str, dict[str, float]] | None:
    """모델이 준 hslAdjust를 안전 범위로 자르고 보정 강도를 곱한다.

    색계열 단위 판단("이 초록이 탁하다", "하늘의 파랑만 채도를")은 히스토그램
    측정으로 나오지 않는다. 그래서 이 값만은 모델이 정하고, 여기서는 범위와
    세기만 관리한다. 알 수 없는 색 이름과 0에 가까운 값은 버린다.
    """
    if not isinstance(raw, dict):
        return None
    out: dict[str, dict[str, float]] = {}
    for channel, adj in raw.items():
        if channel not in _HSL_CHANNELS or not isinstance(adj, dict):
            continue
        vals: dict[str, float] = {}
        for key, limit in _HSL_LIMITS.items():
            try:
                value = float(adj.get(key, 0.0))
            except (TypeError, ValueError):
                continue
            value = _clamp(value * gain, -limit, limit)
            if abs(value) >= 0.01:
                vals[key] = value
        if vals:
            out[channel] = vals
    return out or None


# HSL 채널 거르기 기준. 점유율 = apply_hsl_adjust와 같은 색상·채도 마스크의 평균.
# 실측(7장): 원본에 0.3~1.0%뿐인 색(음식 yellow, 거리 인물 orange)에 준 값은
# 그 색을 잡는 게 아니라 전역 색온도로 데워진 화소를 한 번 더 물들였다
# (walker orange: 원본 점유 0.8%인데 ΔE 1.26). 반대로 81%를 덮는 색(카페 orange)은
# 사실상 전역 채도라 전역 채도·색온도 위에 겹쳐 쌓인다.
_HSL_MIN_SHARE = 0.02
_HSL_GLOBAL_SHARE = 0.50
_HSL_GLOBAL_DAMP = 0.5
_HSL_MAX_CHANNELS = 2


def _hsl_channel_shares(img: Image.Image) -> dict[str, float]:
    """채널별로 apply_hsl_adjust가 움직일 화소의 비율(마스크 평균, 0~1)."""
    small = img.convert("RGB")
    small.thumbnail((256, 256))
    hsv = cv2.cvtColor(np.asarray(small)[:, :, ::-1], cv2.COLOR_BGR2HSV).astype(np.float32)
    h_ch, gate = hsv[:, :, 0], np.clip(hsv[:, :, 1] / 40.0, 0.0, 1.0)
    shares: dict[str, float] = {}
    for channel, (lo, hi) in _HSL_CHANNELS.items():
        if lo > hi:
            half = ((180 - lo) + hi) / 2.0
            d = np.abs(h_ch - (lo + half) % 180)
            dist = np.minimum(d, 180.0 - d)
        else:
            half = (hi - lo) / 2.0
            dist = np.abs(h_ch - (lo + hi) / 2.0)
        mask = np.clip(1.0 - dist / max(half + 5, 1), 0.0, 1.0) * gate
        shares[channel] = float(mask.mean())
    return shares


def _prune_hsl(
    hsl: dict[str, dict[str, float]] | None, img: Image.Image,
) -> dict[str, dict[str, float]] | None:
    """사진에 거의 없는 색은 버리고, 대부분을 덮는 색은 줄이고, 채널 수를 묶는다."""
    if not hsl:
        return hsl
    shares = _hsl_channel_shares(img)
    kept: list[tuple[float, str, dict[str, float]]] = []
    for channel, vals in hsl.items():
        share = shares.get(channel, 0.0)
        if share < _HSL_MIN_SHARE:
            log.info("param_engine: hsl %s 버림 — 점유 %.1f%%", channel, share * 100)
            continue
        if share >= _HSL_GLOBAL_SHARE:
            vals = {k: v * _HSL_GLOBAL_DAMP for k, v in vals.items()}
            vals = {k: v for k, v in vals.items() if abs(v) >= 0.01}
            if not vals:
                continue
        kept.append((share * max(abs(v) for v in vals.values()), channel, vals))
    kept.sort(key=lambda item: -item[0])
    out = {channel: vals for _, channel, vals in kept[:_HSL_MAX_CHANNELS]}
    return out or None


# 영역 보정 거르기 기준.
# 실측(7장): 인물 4장 전부 face 밝기(0.04~0.10)가 같은 방향의 전역 밝기(0.13~0.16)
# 위에 얹혔고, background는 "하늘도 얼굴 피부도 아닌 전부"라 인물의 몸·옷까지
# 포함한다(astro 98%, hammock 91%). 그래서 background 채도·대비는 전역 보정과 같은
# 방향으로 한 번 더 걸린 사실상의 전역 보정이었다 (astro ΔE 3.64 — 주황 우주복이
# 바래고 얼굴만 원래 채도로 남았다). 국소 보정 두 개가 겹친 채 반대 방향으로
# 밝기를 당기기도 했다 (store: 아치 −0.1 / 상반신 +0.15).
_REGION_MIN_EFFECT = 0.05
_REGION_GLOBAL_DAMP = 0.5
_REGION_LOCAL_MAX = 2
_REGION_OVERLAP = 0.5
_REGION_META = frozenset({"area", "shape", "feather", "reason"})
_REGION_TONE_KEYS = ("brightness", "contrast", "saturation", "temperature",
                     "highlights", "shadows")


def _local_box(spec: dict[str, Any]) -> tuple[float, float, float, float] | None:
    area = spec.get("area")
    if not isinstance(area, dict):
        return None
    try:
        x, y = float(area.get("x", 0.0)), float(area.get("y", 0.0))
        w, h = float(area.get("width", 0.0)), float(area.get("height", 0.0))
    except (TypeError, ValueError):
        return None
    if not all(math.isfinite(v) for v in (x, y, w, h)) or w <= 0 or h <= 0:
        return None
    return x, y, x + w, y + h


def _overlap_of_smaller(a: tuple[float, ...], b: tuple[float, ...]) -> float:
    iw = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    ih = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    smaller = min((a[2] - a[0]) * (a[3] - a[1]), (b[2] - b[0]) * (b[3] - b[1]))
    return iw * ih / smaller if smaller > 0 else 0.0


def _nonzero(raw: Any) -> bool:
    try:
        return abs(float(raw)) >= 0.01
    except (TypeError, ValueError):
        return False


# 국소 보정의 이유(reason)가 이런 말이면, 톤을 만질 대상이 아니라 지울 대상이다.
# 실측: 창문 너머 도시 사진에서 모델이 천장 조명의 유리 반사를 "유리에 비친 조명 반사"로
# 정확히 짚고도 local_* 로 어둡게만 해, 반사가 사라지지 않고 타원 얼룩으로 남았다.
_REMOVAL_REASON_WORDS = ("반사", "비친", "비쳐", "비침", "글레어", "플레어", "먼지",
                         "얼룩", "물방울", "반점", "자국")


def _local_to_removal(
    analysis: dict[str, Any], name: str, spec: dict[str, Any],
) -> bool:
    """반사·얼룩을 짚은 작은 국소 보정을 remove_areas(인페인팅)로 옮긴다. 옮겼으면 True.

    인페인팅은 작은 영역만 자연스럽게 메우므로 image_processor와 같은 크기 한도를 쓴다.
    그보다 크면 지우지 않고 원래대로 국소 보정으로 둔다.
    """
    reason = str(spec.get("reason") or "")
    if not any(w in reason for w in _REMOVAL_REASON_WORDS):
        return False
    box = _local_box(spec)
    if box is None:
        return False
    x0, y0 = max(0.0, box[0]), max(0.0, box[1])
    x1, y1 = min(1.0, box[2]), min(1.0, box[3])
    w, h = x1 - x0, y1 - y0
    if w <= 0 or h <= 0 or w * h > _INPAINT_MAX_AREA:
        return False
    auto = analysis.get("autoEdits")
    auto = dict(auto) if isinstance(auto, dict) else {}
    areas = [a for a in (auto.get("remove_areas") or []) if isinstance(a, dict)]
    areas.append({"x": round(x0, 4), "y": round(y0, 4), "width": round(w, 4), "height": round(h, 4)})
    auto["remove_areas"] = areas
    analysis["autoEdits"] = auto
    log.info("param_engine: region %s → remove_areas (%s, 면적 %.2f%%)", name, reason, w * h * 100)
    return True


def _prune_region_params(
    analysis: dict[str, Any], params: dict[str, Any], is_portrait: bool, gain: float,
) -> None:
    """analysis["regionParams"]에서 효과가 없거나 전역 보정과 겹치는 값을 뺀다.

    요청 원본 딕셔너리는 건드리지 않고 새 딕셔너리로 바꿔 넣는다. 남는 게
    없으면 None (스키마상 허용되는 값)."""
    regions = analysis.get("regionParams")
    if not isinstance(regions, dict):
        return
    if gain <= 0.0:
        # "보정 없음"이면 영역 보정도 하지 않는다
        analysis["regionParams"] = None
        return

    out: dict[str, dict[str, Any]] = {}
    local_boxes: list[tuple[tuple[float, ...], float]] = []
    for name in sorted(regions, key=lambda n: (n.startswith("local"), n)):
        spec = regions.get(name)
        if not isinstance(spec, dict):
            continue
        if name == "background" and is_portrait:
            log.info("param_engine: region background 버림 — 인물의 몸까지 덮는 전역 보정")
            continue
        if name.startswith("local") and _local_to_removal(analysis, name, spec):
            continue
        vals: dict[str, Any] = {}
        for key, raw in spec.items():
            if key in _REGION_META or key in _FACE_TEXTURE_KEYS:
                # 잡티·스무딩은 _merge_face_texture가 이미 정리했다
                vals[key] = raw
                continue
            try:
                v = float(raw)
            except (TypeError, ValueError):
                continue
            if not math.isfinite(v):
                continue
            g = params.get(key)
            same_dir = (
                key in _REGION_TONE_KEYS and isinstance(g, (int, float))
                and abs(g) >= 0.03 and g * v > 0
            )
            if name == "face" and key == "brightness" and same_dir and abs(g) >= _REGION_MIN_EFFECT:
                continue   # 전역 밝기가 이미 같은 쪽으로 얼굴을 올린다
            if name == "background" and same_dir:
                v *= _REGION_GLOBAL_DAMP
            if name == "sky" and key == "saturation" and v < 0:
                # 하늘은 채도를 지키거나 높이는 자리다 (프롬프트도 그렇게 요구한다).
                # 모델이 −0.1을 줘서 전역 채도 감소와 겹쳐 하늘이 회색으로 빠졌다.
                continue
            if abs(v) < _REGION_MIN_EFFECT:
                continue
            vals[key] = v
        if not any(k not in _REGION_META and _nonzero(vals[k]) for k in vals):
            continue
        if name.startswith("local"):
            box = _local_box(vals)
            if box is None:
                continue
            if len(local_boxes) >= _REGION_LOCAL_MAX:
                log.info("param_engine: region %s 버림 — 국소 보정 %d개 한도", name, _REGION_LOCAL_MAX)
                continue
            b = float(vals.get("brightness", 0.0))
            if any(_overlap_of_smaller(box, ob) > _REGION_OVERLAP and b * obr < 0
                   for ob, obr in local_boxes):
                log.info("param_engine: region %s 버림 — 겹친 국소 보정과 밝기 방향이 반대", name)
                continue
            local_boxes.append((box, b))
        out[name] = vals

    if out != regions:
        log.info("param_engine: regionParams %s → %s", list(regions), list(out))
    analysis["regionParams"] = out or None

def _damp_warm_stacking(params: dict[str, Any], analysis: dict[str, Any]) -> None:
    """전역 temperature가 이미 웜이면 피부에 겹치는 웜 보정을 덜어낸다.

    전역 웜톤 + hslAdjust.orange 채도 부스트 + regionParams.face temperature가
    같은 피부에 차례로 걸려 인물 사진이 전부 같은 주황 피부로 수렴했다.
    - orange 채도 양수: 전역 웜의 세기에 따라 최대 _STACK_ORANGE_DAMP만큼 줄인다.
    - face temperature 양수: 전역이 이미 준 몫을 뺀다 (0 아래로는 내리지 않는다).
    음수(식히기·채도 빼기)는 겹침이 아니므로 그대로 둔다. analysis["regionParams"]는
    새 딕셔너리로 바꿔 넣는다 (요청 원본을 건드리지 않게).
    """
    t = float(params.get("temperature") or 0.0)
    if t < _STACK_T_LO:
        return
    hsl = params.get("hslAdjust")
    orange = hsl.get("orange") if isinstance(hsl, dict) else None
    if isinstance(orange, dict) and orange.get("saturation", 0.0) > 0:
        keep = 1.0 - _STACK_ORANGE_DAMP * _ramp(t, _STACK_T_LO, _STACK_T_HI)
        sat = round(orange["saturation"] * keep, 3)
        new_orange = {k: sat if k == "saturation" else v for k, v in orange.items()
                      if k != "saturation" or sat >= 0.01}
        new_hsl = {k: new_orange if k == "orange" else v for k, v in hsl.items()
                   if k != "orange" or new_orange}
        if new_hsl:
            params["hslAdjust"] = new_hsl
        else:
            params.pop("hslAdjust", None)
    regions = analysis.get("regionParams")
    face = regions.get("face") if isinstance(regions, dict) else None
    if isinstance(face, dict):
        try:
            face_t = float(face.get("temperature") or 0.0)
        except (TypeError, ValueError):
            face_t = 0.0
        if math.isfinite(face_t) and face_t > 0:
            new_regions = dict(regions)
            new_regions["face"] = {**face, "temperature": round(max(0.0, face_t - t), 3)}
            analysis["regionParams"] = new_regions


# 화이트밸런스 + 식히는 temperature가 피부의 a*(붉음)·b*(노랑)를 이 비율 아래로
# 떨어뜨리면 두 값을 함께 줄인다. 실측: 원본 피부 a* 13.8 b* 15.4가 a* −3.5(청록)·
# a* 0.4 b* 5(잿빛)가 됐다. 뉴트럴~쿨 스타일도 피부가 빨간기를 잃으면 병색으로 보인다.
_SKIN_A_KEEP = 0.70
_SKIN_B_KEEP = 0.55
# 피부색 화소가 화면의 이 비율보다 적으면 판단하지 않는다
_SKIN_GUARD_MIN_FRAC = 0.005


def _skin_ab_after(skin_rgb: np.ndarray, gains: tuple[float, float, float],
                   wb: float, temperature: float) -> tuple[float, float]:
    """피부 화소(N×3, 0~255)에 렌더러와 같은 WB 게인과 temperature를 건 뒤의 평균 a*, b*."""
    rgb = skin_rgb.copy()
    if wb >= 0.01:
        for c, g in enumerate(gains):
            rgb[:, c] *= max(0.75, min(1.25, 1.0 + (g - 1.0) * min(1.0, wb)))
    lab = cv2.cvtColor(np.clip(np.rint(rgb), 0, 255).astype(np.uint8)[None], cv2.COLOR_RGB2LAB)[0]
    a = lab[:, 1].astype(np.float32) - 128.0
    b = lab[:, 2].astype(np.float32) - 128.0
    # adjust_color_temperature: b += 15t, a += 4.5t (8비트 LAB 눈금)
    return float(a.mean() + 4.5 * temperature), float(b.mean() + 15.0 * temperature)


def _guard_skin_cooling(params: dict[str, Any], img: Image.Image) -> None:
    """계획된 auto_wb + 음수 temperature가 피부를 청록·잿빛으로 식히면 함께 덜어낸다.

    렌더 전에 피부색 화소(색상각 20~65°, a* 양수, 밝기 대비 채도가 옷 원색만큼
    높지 않은 화소)만 골라 같은 게인·시프트를 걸어 보고, a*가 _SKIN_A_KEEP·b*가
    _SKIN_B_KEEP 아래로 떨어지면 둘을 같은 비율로 줄인다(이분 탐색). 양수
    temperature(덥히기)와 피부를 덥히는 WB는 건드리지 않는다.
    """
    wb = float(params.get("auto_wb") or 0.0)
    temp = float(params.get("temperature") or 0.0)
    cool_t = min(0.0, temp)
    if wb < 0.01 and cool_t > -0.01:
        return
    small = img.convert("RGB")
    small.thumbnail((256, 256), Image.BILINEAR)
    arr = np.asarray(small, dtype=np.float32).reshape(-1, 3)
    lab = cv2.cvtColor(arr.astype(np.uint8)[None], cv2.COLOR_RGB2LAB)[0].astype(np.float32)
    L = lab[:, 0] * (100.0 / 255.0)
    a, b = lab[:, 1] - 128.0, lab[:, 2] - 128.0
    hue = np.degrees(np.arctan2(b, a))
    rel = np.hypot(a, b) / (L + 10.0)
    skin = (hue > 20) & (hue < 65) & (a > 5) & (rel > 0.12) & (rel < 0.60) & (L > 25) & (L < 92)
    if skin.mean() < _SKIN_GUARD_MIN_FRAC:
        return
    skin_rgb = arr[skin]
    a0, b0 = float(a[skin].mean()), float(b[skin].mean())
    gains = estimate_illuminant(img) if wb >= 0.01 else (1.0, 1.0, 1.0)

    def ok(k: float) -> bool:
        a1, b1 = _skin_ab_after(skin_rgb, gains, wb * k, cool_t * k)
        return a1 >= _SKIN_A_KEEP * a0 and b1 >= _SKIN_B_KEEP * b0

    if ok(1.0):
        return
    lo, hi = 0.0, 1.0
    for _ in range(8):
        mid = (lo + hi) / 2
        if ok(mid):
            lo = mid
        else:
            hi = mid
    log.info("param_engine: skin guard — auto_wb %.2f→%.2f, temperature %.3f→%.3f (skin a %.1f b %.1f)",
             wb, wb * lo, temp, cool_t * lo if temp < 0 else temp, a0, b0)
    params["auto_wb"] = round(wb * lo, 3)
    if temp < 0:
        params["temperature"] = round(temp * lo, 3)


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


# 얼굴 영역 보정(regionParams.face)의 잡티·스무딩을 전역 값에 합칠 때의 상한.
# 두 패스는 같은 피부 마스크에 차례로 걸린다 — 따로 두면 모델이 준 face 값이
# 기본 레시피 위에 한 번 더 얹혀 효과가 두 배가 됐다.
_FACE_MERGED_SMOOTHING_MAX = 0.45
_FACE_MERGED_BLEMISH_MAX = 0.50
_FACE_TEXTURE_KEYS = ("blemish_removal", "skin_smoothing")


def _merge_face_texture(
    analysis: dict[str, Any],
    blemish: float,
    skin_smoothing: float,
    enabled: bool,
    fold: bool,
) -> tuple[float, float]:
    """regionParams.face의 잡티·스무딩을 전역 파라미터로 옮기고 영역 쪽은 0으로 둔다.

    - enabled=False (피부 보정 꺼짐·보정 없음): 영역 쪽도 0.
    - fold=True (인물): 전역 = max(전역, 영역)을 상한으로 자른 값. 합치지 않고
      더하면 같은 피부에 두 번 걸린다.
    - fold=False (인물 아님): 전역 피부 보정이 0이라 겹칠 게 없다 — 영역 값을 그대로 둔다.

    analysis["regionParams"]는 새 딕셔너리로 바꿔 넣는다 (요청 원본을 건드리지 않게).
    """
    regions = analysis.get("regionParams")
    face = regions.get("face") if isinstance(regions, dict) else None
    if not isinstance(face, dict):
        return blemish, skin_smoothing
    if enabled and not fold:
        return blemish, skin_smoothing

    def _num(key: str) -> float:
        try:
            v = float(face.get(key) or 0.0)
        except (TypeError, ValueError):
            return 0.0
        return v if math.isfinite(v) else 0.0

    if enabled:
        # 영역 값은 상한까지만 전역을 끌어올린다 (사용자가 고른 강도를 낮추지는 않는다)
        blemish = max(blemish, min(_num("blemish_removal"), _FACE_MERGED_BLEMISH_MAX))
        skin_smoothing = max(
            skin_smoothing, min(_num("skin_smoothing"), _FACE_MERGED_SMOOTHING_MAX)
        )

    new_face = dict(face)
    for key in _FACE_TEXTURE_KEYS:
        if key in new_face:
            new_face[key] = 0.0
    new_regions = dict(regions)
    new_regions["face"] = new_face
    analysis["regionParams"] = new_regions
    return blemish, skin_smoothing


# ── 적용된 변형 코멘트 ──

_TREND_LABELS = {
    "warm_film": "웜 필름",
    "korean_gamsung": "한국 감성",
    "cinematic_moody": "시네마틱",
    "bright_airy": "밝은 감성",
    "golden_hour": "골든아워",
    "clean_minimal": "클린 미니멀",
    "flash_digicam": "플래시 스냅",
    "soft_pastel": "소프트 파스텔",
    "bw_grain": "흑백 필름",
}

# 트렌드 없이 기본 레시피를 탔을 때의 이름
_DEFAULT_LABEL = "소프트 필름"

# 모든 라벨이 모음으로 끝난다 — 조사를 "와/를"로 고정해 쓴다 (describe_params)
_CURVE_LABELS = {
    "film": "필름 커브",
    "s_curve": "S커브",
    "fade": "페이드",
    "high_contrast": "강한 대비 커브",
    "bright": "밝은 커브",
    "soft_film": "부드러운 필름 커브",
    "flash": "플래시 커브",
    "pastel": "파스텔 커브",
    "bw": "흑백 커브",
    "gamsung": "감성 페이드",
}


def describe_params(
    stats: dict[str, float],
    trend: str,
    subject: str,
    params: dict[str, Any],
    gain: float,
    face_detected: bool | None = None,
) -> str:
    """왜 이 보정값이 나왔는지 한 문장으로 설명한다.

    값을 정한 근거(측정값·프로필·피사체)가 모두 여기 있으므로
    모델에게 설명을 시키지 않는다 — 공짜이고, 실제 근거와 어긋날 일도 없다.
    face_detected=False면 피부 보정이 걸릴 얼굴이 없으므로 피부 문장을 쓰지 않는다.
    """
    if gain <= 0.0:
        return "보정 없음 설정이라 원본 톤을 그대로 두었어요"

    is_portrait = (
        subject == "인물" and params["skin_smoothing"] >= 0.1 and face_detected is not False
    )
    mono = params["saturation"] <= -0.99
    recipe = _TREND_RECIPES.get(trend, _DEFAULT_RECIPE)
    hazy_only = recipe.get("crush_blacks", "always") == "hazy"

    # ── 측정에서 나온 이유 (눈에 띄는 것부터) ──
    reasons: list[str] = []

    if params["brightness"] >= 0.10 and stats["brightness"] < 0.45:
        reasons.append("어두워서 밝기를 올리고")
    elif params["brightness"] <= -0.10 and stats["brightness"] > 0.55:
        reasons.append("밝게 찍혀 노출을 낮추고")

    if stats["highlight_clip"] > 0.03 and params["highlights"] < -0.1:
        reasons.append("날아간 밝은 부분을 눌러 주고")
    elif params["shadows"] >= 0.10:
        reasons.append("뭉친 어두운 부분을 살리고")
    elif params["shadows"] <= -0.10:
        if hazy_only:
            # 필름 계열은 뿌연 사진에서만 바닥을 누른다 ([haze_flatness])
            reasons.append("뿌옇게 뜬 바닥을 살짝 눌러 또렷하게 하고")
        else:
            reasons.append("떠 있는 검정을 눌러 깊이를 주고")
    elif params["dehaze"] >= 0.10:
        reasons.append("뿌연 기운을 걷어내고")
    elif params["contrast"] >= 0.15:
        reasons.append("밋밋한 대비를 세우고")
    elif params["contrast"] <= -0.10:
        reasons.append("센 대비를 부드럽게 풀고")
    elif params["sharpness"] >= 0.12 and stats["sharpness"] < 0.3:
        reasons.append("흐린 초점을 다듬고")
    elif not mono and params["saturation"] <= -0.12:
        reasons.append("과한 채도를 덜어내고")
    elif not mono and params["saturation"] >= 0.12:
        reasons.append("빠진 색을 채우고")

    if mono:
        # 흑백 전환이 가장 큰 변화다 — 맨 앞에 둔다
        reasons.insert(0, "흑백으로 옮기고")

    # 인물이면 피부 문장이 뒤에 붙으므로 이유는 하나만 남겨 길이를 맞춘다
    reasons = reasons[: 1 if is_portrait else 2]

    # ── 스타일 마무리 ──
    trend_label = _TREND_LABELS.get(trend) or (_DEFAULT_LABEL if trend not in _TREND_RECIPES else None)
    lead = f"{trend_label} 톤에 맞춰 " if trend_label else ""

    warm_adj = warm_adv = ""
    if not mono:
        if params["temperature"] >= 0.12:
            warm_adj, warm_adv = "따뜻한", "따뜻하게"
        elif params["temperature"] <= -0.08:
            warm_adj, warm_adv = "깨끗하고 차가운", "차갑게"

    curve = _CURVE_LABELS.get(params["toneCurve"]["preset"])
    if params["toneCurve"]["strength"] < 0.15:
        curve = None
    grain_val = params.get("grain", 0.0)
    grain = "필름 그레인" if grain_val >= 0.18 else "고운 그레인" if grain_val >= 0.07 else None

    # 목적어: "커브와 그레인을" / "커브를" / "그레인을" (라벨은 모두 모음·ㄴ 받침으로 끝남)
    if curve and grain:
        obj = f"{curve}와 {grain}을"
    elif curve:
        obj = f"{curve}를"
    elif grain:
        obj = f"{grain}을"
    else:
        obj = ""

    if obj and warm_adj:
        # 수식이 길게 겹치면 부사로 떼어 낸다 ("따뜻하게, 부드러운 필름 커브와 …")
        tail = (
            f"{lead}{warm_adv}, {obj} 얹었어요"
            if obj.count(" ") >= 2 or " " in warm_adj
            else f"{lead}{warm_adj} {obj} 얹었어요"
        )
    elif obj:
        tail = f"{lead}{obj} 얹었어요"
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

    # 노이즈 1.5는 블록 추정(질감 제외) 눈금이다. 예전 전체 평균 추정의 3.0과
    # 같은 사진들(어두운 야간 인물 noguchi 2.05·walker 1.85)을 저조도로 잡는다.
    low_light = bool(stats["brightness"] < 0.32 and stats["noise"] > 1.5)

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
