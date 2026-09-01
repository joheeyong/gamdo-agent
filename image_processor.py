"""이미지 변형 엔진 — Pillow + OpenCV 기반 순수 함수 모듈."""

from __future__ import annotations

import base64
import io
import logging
import os
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter

log = logging.getLogger("gamdo-agent")


# ── 톤 커브 프리셋 (5개 제어점: input → output, 0.0~1.0) ──

TONE_CURVE_PRESETS: dict[str, list[tuple[float, float]]] = {
    "linear": [(0, 0), (0.25, 0.25), (0.5, 0.5), (0.75, 0.75), (1, 1)],
    "s_curve": [(0, 0), (0.25, 0.18), (0.5, 0.5), (0.75, 0.82), (1, 1)],
    "film": [(0, 0.05), (0.25, 0.22), (0.5, 0.52), (0.75, 0.78), (1, 0.95)],
    "fade": [(0, 0.08), (0.25, 0.28), (0.5, 0.50), (0.75, 0.72), (1, 0.92)],
    "high_contrast": [(0, 0), (0.25, 0.12), (0.5, 0.5), (0.75, 0.88), (1, 1)],
    "bright": [(0, 0.04), (0.25, 0.30), (0.5, 0.56), (0.75, 0.80), (1, 1)],
}


# MediaPipe는 선택적 의존성 — 없으면 잡티 제거 비활성화
try:
    import mediapipe as mp

    _MP_AVAILABLE = True
    # FaceLandmarker 모델 파일 경로 (패키지 내 번들 또는 자동 다운로드)
    _MP_MODEL_PATH: str | None = None
    _model_candidate = os.path.join(
        os.path.dirname(mp.__file__), "models", "face_landmarker.task"
    )
    if os.path.exists(_model_candidate):
        _MP_MODEL_PATH = _model_candidate
    else:
        try:
            import urllib.request

            os.makedirs(os.path.dirname(_model_candidate), exist_ok=True)
            urllib.request.urlretrieve(
                "https://storage.googleapis.com/mediapipe-models/"
                "face_landmarker/face_landmarker/float16/latest/face_landmarker.task",
                _model_candidate,
            )
            _MP_MODEL_PATH = _model_candidate
            log.info("Downloaded face_landmarker model to %s", _model_candidate)
        except Exception as exc:
            log.warning("Failed to download face_landmarker model: %s", exc)
except ImportError:
    mp = None  # type: ignore[assignment]
    _MP_AVAILABLE = False
    _MP_MODEL_PATH = None
    log.warning("mediapipe not installed — blemish removal disabled")


# ── MediaPipe Pose Landmarker 초기화 ──
_POSE_MODEL_PATH: str | None = None
if _MP_AVAILABLE and mp is not None:
    _pose_candidate = os.path.join(
        os.path.dirname(mp.__file__), "models", "pose_landmarker_lite.task"
    )
    if os.path.exists(_pose_candidate):
        _POSE_MODEL_PATH = _pose_candidate
    else:
        try:
            import urllib.request as _urllib_req

            os.makedirs(os.path.dirname(_pose_candidate), exist_ok=True)
            _urllib_req.urlretrieve(
                "https://storage.googleapis.com/mediapipe-models/"
                "pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task",
                _pose_candidate,
            )
            _POSE_MODEL_PATH = _pose_candidate
            log.info("Downloaded pose_landmarker model to %s", _pose_candidate)
        except Exception as _exc:
            log.warning("Failed to download pose_landmarker model: %s", _exc)


# ── Base64 ↔ PIL Image 변환 ──


def decode_base64_image(b64: str) -> Image.Image:
    """Base64 문자열을 PIL Image로 디코딩."""
    data = base64.b64decode(b64)
    return Image.open(io.BytesIO(data)).convert("RGB")


def encode_image_base64(img: Image.Image, fmt: str = "JPEG", quality: int = 90) -> str:
    """PIL Image를 base64 문자열로 인코딩."""
    buf = io.BytesIO()
    img.save(buf, format=fmt, quality=quality)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


# ── 개별 변형 함수 ──


def adjust_brightness(img: Image.Image, factor: float) -> Image.Image:
    """밝기 조절. factor: -1.0 ~ +1.0 (0 = 원본).

    LAB L(밝기) 채널에 감마 보정을 적용한다.
    A/B(색상) 채널은 그대로 유지하므로 파란 하늘 같은
    채색 영역의 색상 정보가 완전히 보존된다.
    """
    if abs(factor) < 0.01:
        return img

    arr = np.array(img, dtype=np.uint8)
    arr_bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    lab = cv2.cvtColor(arr_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)

    l_ch = lab[:, :, 0] / 255.0

    # factor → gamma 변환: +1.0 → gamma 0.4(밝게), -1.0 → gamma 2.5(어둡게)
    gamma = 1.0 / (1.0 + factor) if factor >= 0 else 1.0 - factor * 1.5
    gamma = max(0.2, min(5.0, gamma))

    l_ch = np.power(l_ch, gamma)
    lab[:, :, 0] = np.clip(l_ch * 255.0, 0, 255)

    result_bgr = cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2BGR)
    result_rgb = cv2.cvtColor(result_bgr, cv2.COLOR_BGR2RGB)

    return Image.fromarray(result_rgb)


def adjust_contrast(img: Image.Image, factor: float) -> Image.Image:
    """대비 조절. factor: -1.0 ~ +1.0 (0 = 원본).

    LAB 색공간의 L(밝기) 채널에서만 대비를 조절하여,
    채도와 색상 정보를 보존한다. 기존 RGB 전체 대비 감소는
    채도까지 떨어뜨려 하늘 같은 채색 영역을 회색으로 만들었다.
    """
    if abs(factor) < 0.01:
        return img

    arr = np.array(img, dtype=np.uint8)
    arr_bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    lab = cv2.cvtColor(arr_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)

    l_ch = lab[:, :, 0]
    mean_l = np.mean(l_ch)

    # L 채널에서만 대비 조절: factor > 0 → 중간값에서 멀어짐 / < 0 → 가까워짐
    l_ch = mean_l + (l_ch - mean_l) * (1.0 + factor)
    lab[:, :, 0] = np.clip(l_ch, 0, 255)

    result_bgr = cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2BGR)
    result_rgb = cv2.cvtColor(result_bgr, cv2.COLOR_BGR2RGB)

    return Image.fromarray(result_rgb)


def adjust_clarity(img: Image.Image, factor: float) -> Image.Image:
    """선명감(Clarity) 조절. factor: -1.0 ~ +1.0 (0 = 원본).

    Lightroom의 Clarity와 동일 개념 — 중간톤의 로컬 대비만 강화한다.
    LAB L채널에서 가우시안 블러(로컬 평균)를 빼 하이패스 디테일을 추출하고,
    중간톤 마스크를 적용하여 밝은/어두운 극단은 건드리지 않는다.
    A/B 채널은 보존되므로 색상 왜곡이 없다.
    """
    if abs(factor) < 0.01:
        return img

    arr = np.array(img, dtype=np.uint8)
    arr_bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    lab = cv2.cvtColor(arr_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)

    l_ch = lab[:, :, 0]

    # 로컬 평균 (큰 커널 가우시안 블러 → 저주파 성분)
    h, w = l_ch.shape[:2]
    ksize = max(31, int(min(h, w) * 0.05)) | 1  # 해상도 적응형 커널
    l_blur = cv2.GaussianBlur(l_ch, (ksize, ksize), 0)

    # 하이패스 디테일 = 원본 - 로컬 평균
    detail = l_ch - l_blur

    # 중간톤 마스크: 밝기 64~192 구간에 높은 가중치 (극단은 보호)
    midtone_mask = 1.0 - np.abs(l_ch - 128.0) / 128.0
    midtone_mask = np.clip(midtone_mask * 1.5, 0.0, 1.0)

    # factor 비례로 디테일 증폭 (양수: 로컬 대비 강화, 음수: 소프트)
    l_ch = l_ch + detail * factor * 1.5 * midtone_mask
    lab[:, :, 0] = np.clip(l_ch, 0, 255)

    result_bgr = cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2BGR)
    result_rgb = cv2.cvtColor(result_bgr, cv2.COLOR_BGR2RGB)

    return Image.fromarray(result_rgb)


def apply_dehaze(img: Image.Image, factor: float) -> Image.Image:
    """안개 제거(Dehaze). factor: -1.0 ~ +1.0 (0 = 원본).

    Dark Channel Prior 기반 디헤이즈.
    양수: 안개/연무 제거 (대비·채도 복원), 음수: 안개 추가 (몽환적 효과).
    LAB 색공간에서 L채널은 디헤이즈, A/B 채널은 채도 복원을 수행한다.
    """
    if abs(factor) < 0.01:
        return img

    arr = np.array(img, dtype=np.uint8)
    arr_bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)

    if factor > 0:
        # ── 양수: Dark Channel Prior 디헤이즈 ──
        b, g, r = cv2.split(arr_bgr)
        # Dark channel: 각 픽셀 주변에서 RGB 최솟값의 로컬 최솟값
        dark = np.minimum(np.minimum(b, g), r).astype(np.float32)
        ksize = max(7, int(min(arr_bgr.shape[:2]) * 0.01)) | 1
        dark = cv2.erode(dark, np.ones((ksize, ksize), np.uint8))

        # Atmospheric light 추정: dark channel 상위 0.1% 밝기의 평균
        num_pixels = dark.size
        n_bright = max(1, int(num_pixels * 0.001))
        flat_dark = dark.flatten()
        indices = np.argpartition(flat_dark, -n_bright)[-n_bright:]
        # 해당 인덱스들에서 원본의 밝기 평균
        arr_f = arr_bgr.astype(np.float32)
        flat_img = arr_f.reshape(-1, 3)
        atm = flat_img[indices].mean(axis=0)  # [B, G, R]
        atm = np.clip(atm, 1.0, 255.0)

        # Transmission 추정
        norm = arr_f / atm[np.newaxis, np.newaxis, :]
        dark_norm = np.min(norm, axis=2)
        dark_norm_blur = cv2.GaussianBlur(dark_norm, (ksize * 2 + 1, ksize * 2 + 1), 0)
        # factor 비례로 제거 강도 조절 (0.0~0.95)
        omega = factor * 0.95
        transmission = 1.0 - omega * dark_norm_blur
        transmission = np.clip(transmission, 0.1, 1.0)

        # Scene radiance 복원
        t = transmission[:, :, np.newaxis]
        result_f = (arr_f - atm) / t + atm
        result_bgr = np.clip(result_f, 0, 255).astype(np.uint8)
    else:
        # ── 음수: 안개 추가 (화이트 쪽으로 블렌딩) ──
        lab = cv2.cvtColor(arr_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
        haze_amount = abs(factor)
        # L채널을 밝게 + 균일화 → 안개 효과
        lab[:, :, 0] = lab[:, :, 0] * (1.0 - haze_amount * 0.4) + 200.0 * haze_amount * 0.4
        # A/B 채널을 128(무채색)쪽으로 → 탈채도
        lab[:, :, 1] = lab[:, :, 1] * (1.0 - haze_amount * 0.3) + 128.0 * haze_amount * 0.3
        lab[:, :, 2] = lab[:, :, 2] * (1.0 - haze_amount * 0.3) + 128.0 * haze_amount * 0.3
        lab[:, :, 0] = np.clip(lab[:, :, 0], 0, 255)
        lab[:, :, 1] = np.clip(lab[:, :, 1], 0, 255)
        lab[:, :, 2] = np.clip(lab[:, :, 2], 0, 255)
        result_bgr = cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2BGR)

    result_rgb = cv2.cvtColor(result_bgr, cv2.COLOR_BGR2RGB)
    return Image.fromarray(result_rgb)


def adjust_saturation(img: Image.Image, factor: float) -> Image.Image:
    """채도 조절. factor: -1.0 ~ +1.0 (0 = 원본).

    LAB 색공간에서 A/B(색상) 채널만 조절하여 밝기를 건드리지 않는다.
    RGB 기반 Color enhance는 밝기와 채도가 커플링되어
    채도 감소 시 하늘 같은 밝은 색상을 회색으로 만든다.
    """
    if abs(factor) < 0.01:
        return img

    arr = np.array(img, dtype=np.uint8)
    arr_bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    lab = cv2.cvtColor(arr_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)

    # A, B 채널 (128이 중심점)
    lab[:, :, 1] = 128.0 + (lab[:, :, 1] - 128.0) * (1.0 + factor)
    lab[:, :, 2] = 128.0 + (lab[:, :, 2] - 128.0) * (1.0 + factor)

    lab[:, :, 1] = np.clip(lab[:, :, 1], 0, 255)
    lab[:, :, 2] = np.clip(lab[:, :, 2], 0, 255)

    result_bgr = cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2BGR)
    result_rgb = cv2.cvtColor(result_bgr, cv2.COLOR_BGR2RGB)

    return Image.fromarray(result_rgb)


def adjust_color_temperature(img: Image.Image, factor: float) -> Image.Image:
    """색온도 조절. factor: -1.0(쿨톤) ~ +1.0(웜톤), 0 = 원본.

    LAB 색공간의 B 채널(파랑-노랑 축)에서 조절하여
    밝기와 채도를 보존하면서 색온도만 변경한다.
    RGB 직접 곱셈은 밝은 파란 하늘에서 B채널을 깎아
    색상 정보를 손실시킨다.
    """
    if abs(factor) < 0.01:
        return img

    arr = np.array(img, dtype=np.uint8)
    arr_bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    lab = cv2.cvtColor(arr_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)

    # LAB B 채널: 높을수록 노랑(웜), 낮을수록 파랑(쿨)
    # factor +1.0 → B채널 +15 (웜톤), factor -1.0 → B채널 -15 (쿨톤)
    shift = factor * 15.0
    lab[:, :, 2] = np.clip(lab[:, :, 2] + shift, 0, 255)

    # A 채널도 미세하게 (웜톤은 살짝 마젠타 방향)
    lab[:, :, 1] = np.clip(lab[:, :, 1] + shift * 0.3, 0, 255)

    result_bgr = cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2BGR)
    result_rgb = cv2.cvtColor(result_bgr, cv2.COLOR_BGR2RGB)

    return Image.fromarray(result_rgb)


def adjust_highlights(img: Image.Image, factor: float) -> Image.Image:
    """하이라이트(밝은 영역) 조절. factor: -1.0 ~ +1.0 (0 = 원본).

    LAB L(밝기) 채널에서만 밝은 영역을 조절하여,
    색상 정보(A/B)를 보존한다. RGB에서 직접 밝기를 더하면
    파란 하늘 같은 채색 영역의 색상 비율이 깨진다.
    """
    if abs(factor) < 0.01:
        return img

    arr = np.array(img, dtype=np.uint8)
    arr_bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    lab = cv2.cvtColor(arr_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)

    l_ch = lab[:, :, 0]

    # 밝은 영역 마스크 (L > 128 부분에 그라데이션)
    threshold = 128.0
    mask = np.clip((l_ch - threshold) / threshold, 0.0, 1.0)

    # L 채널에서만 조절
    l_ch = l_ch + factor * 60.0 * mask
    lab[:, :, 0] = np.clip(l_ch, 0, 255)

    result_bgr = cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2BGR)
    result_rgb = cv2.cvtColor(result_bgr, cv2.COLOR_BGR2RGB)

    return Image.fromarray(result_rgb)


def adjust_shadows(img: Image.Image, factor: float) -> Image.Image:
    """쉐도우(어두운 영역) 조절. factor: -1.0 ~ +1.0 (0 = 원본).

    LAB L(밝기) 채널에서만 어두운 영역을 조절하여,
    색상 정보(A/B)를 보존한다. 양수: 밝게, 음수: 더 어둡게.
    """
    if abs(factor) < 0.01:
        return img

    arr = np.array(img, dtype=np.uint8)
    arr_bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    lab = cv2.cvtColor(arr_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)

    l_ch = lab[:, :, 0]

    # 어두운 영역 마스크 (L < 128 부분에 그라데이션)
    threshold = 128.0
    mask = np.clip((threshold - l_ch) / threshold, 0.0, 1.0)

    # L 채널에서만 조절
    l_ch = l_ch + factor * 60.0 * mask
    lab[:, :, 0] = np.clip(l_ch, 0, 255)

    result_bgr = cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2BGR)
    result_rgb = cv2.cvtColor(result_bgr, cv2.COLOR_BGR2RGB)

    return Image.fromarray(result_rgb)


def apply_tone_curve(
    img: Image.Image, preset: str = "linear", strength: float = 0.0
) -> Image.Image:
    """톤 커브 적용. preset: 프리셋 이름, strength: 0.0~1.0.

    프리셋의 5개 제어점으로 256단계 LUT를 생성하고,
    identity curve와 블렌딩하여 LAB L채널에 적용한다.
    A/B(색상) 채널은 보존되므로 색상 왜곡이 없다.
    """
    if strength < 0.01:
        return img

    points = TONE_CURVE_PRESETS.get(preset)
    if points is None or preset == "linear":
        return img

    # 제어점에서 256단계 LUT 생성 (선형 보간)
    x_pts = np.array([p[0] for p in points], dtype=np.float64)
    y_pts = np.array([p[1] for p in points], dtype=np.float64)

    x_256 = np.linspace(0.0, 1.0, 256)
    curve = np.interp(x_256, x_pts, y_pts)

    # identity curve와 strength로 블렌딩
    identity = x_256
    blended = identity * (1.0 - strength) + curve * strength

    # 0~255 정수 LUT
    lut = np.clip(blended * 255.0, 0, 255).astype(np.uint8)

    # LAB L채널에 LUT 적용
    arr = np.array(img, dtype=np.uint8)
    arr_bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    lab = cv2.cvtColor(arr_bgr, cv2.COLOR_BGR2LAB)

    lab[:, :, 0] = lut[lab[:, :, 0]]

    result_bgr = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    result_rgb = cv2.cvtColor(result_bgr, cv2.COLOR_BGR2RGB)

    return Image.fromarray(result_rgb)


def apply_split_toning(
    img: Image.Image,
    shadow_hue: float = 0.0,
    shadow_strength: float = 0.0,
    highlight_hue: float = 0.0,
    highlight_strength: float = 0.0,
) -> Image.Image:
    """스플릿 토닝 — 쉐도우와 하이라이트에 각각 다른 색조를 입힌다.

    LAB 색공간에서 L(밝기) 기준으로 쉐도우/하이라이트를 분리하고,
    각 영역의 A/B 채널을 hue 방향으로 시프트한다.

    hue: 0~360 (색상환 각도). 0=빨강, 30=오렌지, 60=노랑, 120=녹색,
         180=시안, 210=틸, 240=파랑, 270=보라, 300=마젠타, 330=핑크
    strength: 0.0~1.0 (색조 강도)
    """
    if shadow_strength < 0.01 and highlight_strength < 0.01:
        return img

    arr = np.array(img, dtype=np.uint8)
    arr_bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    lab = cv2.cvtColor(arr_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)

    l_ch = lab[:, :, 0]  # 0~255
    a_ch = lab[:, :, 1]  # 128 중심
    b_ch = lab[:, :, 2]  # 128 중심

    # hue(도) → LAB A/B 시프트 변환
    # LAB A축: +빨강/-녹색, B축: +노랑/-파랑
    def _hue_to_ab_shift(hue_deg: float) -> tuple[float, float]:
        rad = np.radians(hue_deg)
        # 색상환에서 LAB A/B 방향으로 매핑
        # A: cos(hue) 방향 (0도=빨강 → +A)
        # B: sin(hue) 방향 (90도=노랑 → +B)
        # 보정: 색상환 0도=빨강은 LAB에서 +A 방향
        a_shift = np.cos(rad)   # 빨강(+)/녹색(-)
        b_shift = np.sin(rad)   # 노랑(+)/파랑(-)
        return float(a_shift), float(b_shift)

    # 쉐도우 처리 (L < 128 영역, 부드러운 그라데이션)
    if shadow_strength >= 0.01:
        shadow_mask = np.clip((128.0 - l_ch) / 128.0, 0.0, 1.0)
        a_s, b_s = _hue_to_ab_shift(shadow_hue)
        intensity_s = shadow_strength * 25.0  # 최대 A/B 시프트 25
        a_ch = a_ch + a_s * intensity_s * shadow_mask
        b_ch = b_ch + b_s * intensity_s * shadow_mask

    # 하이라이트 처리 (L > 128 영역, 부드러운 그라데이션)
    if highlight_strength >= 0.01:
        highlight_mask = np.clip((l_ch - 128.0) / 128.0, 0.0, 1.0)
        a_h, b_h = _hue_to_ab_shift(highlight_hue)
        intensity_h = highlight_strength * 25.0
        a_ch = a_ch + a_h * intensity_h * highlight_mask
        b_ch = b_ch + b_h * intensity_h * highlight_mask

    lab[:, :, 1] = np.clip(a_ch, 0, 255)
    lab[:, :, 2] = np.clip(b_ch, 0, 255)

    result_bgr = cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2BGR)
    result_rgb = cv2.cvtColor(result_bgr, cv2.COLOR_BGR2RGB)

    return Image.fromarray(result_rgb)


# ── HSL 선택적 색상 조절 ──

# 8색 채널의 HSV Hue 범위 (OpenCV HSV: H 0~180)
_HSL_CHANNELS: dict[str, tuple[int, int]] = {
    "red":     (170, 10),   # 170~180 + 0~10 (wrap-around)
    "orange":  (10, 25),
    "yellow":  (25, 40),
    "green":   (40, 80),
    "cyan":    (80, 100),
    "blue":    (100, 130),
    "purple":  (130, 155),
    "magenta": (155, 170),
}


def apply_hsl_adjust(
    img: Image.Image,
    hsl_params: dict[str, dict[str, float]] | None = None,
) -> Image.Image:
    """선택적 색상(HSL) 조절 — 특정 색상만 H/S/L 개별 조절.

    hsl_params: {
      "red":    {"hue": -1~1, "saturation": -1~1, "lightness": -1~1},
      "orange": {"hue": -1~1, ...},
      ...
    }
    hue: 색상 시프트 (-1.0~1.0, ±30도), saturation: 채도, lightness: 밝기
    """
    if not hsl_params:
        return img

    # 조절이 필요한 채널만 필터링
    active = {
        ch: adj for ch, adj in hsl_params.items()
        if ch in _HSL_CHANNELS and isinstance(adj, dict) and any(
            abs(adj.get(k, 0.0)) >= 0.01 for k in ("hue", "saturation", "lightness")
        )
    }
    if not active:
        return img

    arr = np.array(img, dtype=np.uint8)
    arr_bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    hsv = cv2.cvtColor(arr_bgr, cv2.COLOR_BGR2HSV).astype(np.float32)

    h_ch = hsv[:, :, 0]  # 0~180
    s_ch = hsv[:, :, 1]  # 0~255
    v_ch = hsv[:, :, 2]  # 0~255

    for channel_name, adj in active.items():
        h_shift = adj.get("hue", 0.0)
        s_shift = adj.get("saturation", 0.0)
        l_shift = adj.get("lightness", 0.0)

        lo, hi = _HSL_CHANNELS[channel_name]

        # 색상 마스크 생성 (부드러운 경계)
        if lo > hi:
            # wrap-around (red: 170~180 + 0~10)
            dist = np.minimum(
                np.minimum(np.abs(h_ch - lo), np.abs(h_ch - (lo - 180))),
                np.minimum(np.abs(h_ch - hi), np.abs(h_ch - (hi + 180))),
            )
            half_range = ((180 - lo) + hi) / 2.0
            center = (lo + half_range) % 180
            # 이 경우 직접 계산: wrap-around 거리
            d1 = np.abs(h_ch - center)
            d2 = 180.0 - d1
            dist = np.minimum(d1, d2)
            mask = np.clip(1.0 - dist / max(half_range + 5, 1), 0.0, 1.0)
        else:
            center = (lo + hi) / 2.0
            half_range = (hi - lo) / 2.0
            dist = np.abs(h_ch - center)
            # feather: 경계에서 5도 더 부드럽게
            mask = np.clip(1.0 - dist / max(half_range + 5, 1), 0.0, 1.0)

        # 저채도 픽셀 제외 (무채색은 색상 조절 의미 없음)
        sat_gate = np.clip(s_ch / 40.0, 0.0, 1.0)
        mask = mask * sat_gate

        # Hue 시프트 (±30도, OpenCV 스케일 ±15)
        if abs(h_shift) >= 0.01:
            h_ch = h_ch + h_shift * 15.0 * mask
            h_ch = np.mod(h_ch, 180.0)

        # Saturation 조절
        if abs(s_shift) >= 0.01:
            s_ch = s_ch + s_shift * 80.0 * mask

        # Lightness(Value) 조절
        if abs(l_shift) >= 0.01:
            v_ch = v_ch + l_shift * 80.0 * mask

    hsv[:, :, 0] = np.clip(h_ch, 0, 179)
    hsv[:, :, 1] = np.clip(s_ch, 0, 255)
    hsv[:, :, 2] = np.clip(v_ch, 0, 255)

    result_bgr = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
    result_rgb = cv2.cvtColor(result_bgr, cv2.COLOR_BGR2RGB)

    return Image.fromarray(result_rgb)


def apply_vignette(img: Image.Image, intensity: float) -> Image.Image:
    """비네팅 효과. intensity: -1.0 ~ +1.0 (0 = 없음).

    양수: 가장자리를 어둡게 (클래식 비네팅)
    음수: 가장자리를 밝게 (역비네팅)

    LAB L채널에서만 밝기를 조절하여 가장자리 색상(하늘 파란색 등)을
    보존하면서 자연스러운 비네팅을 적용한다.
    """
    if abs(intensity) < 0.01:
        return img

    arr = np.array(img, dtype=np.uint8)
    arr_bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    lab = cv2.cvtColor(arr_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)

    h, w = lab.shape[:2]

    # 타원형 그라데이션 마스크 생성
    cy, cx = h / 2, w / 2
    y_coords, x_coords = np.ogrid[:h, :w]
    dist = np.sqrt(((x_coords - cx) / cx) ** 2 + ((y_coords - cy) / cy) ** 2)

    # 중심 0 ~ 가장자리 1 → 부드러운 감쇠 커브
    mask = np.clip(dist - 0.4, 0, 1.0) / 0.6
    mask = mask ** 1.5  # 감쇠 커브를 더 부드럽게

    # L채널에서만 밝기 조절 (색상 보존)
    l_ch = lab[:, :, 0]
    l_ch = l_ch - intensity * 80.0 * mask
    lab[:, :, 0] = np.clip(l_ch, 0, 255)

    result_bgr = cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2BGR)
    result_rgb = cv2.cvtColor(result_bgr, cv2.COLOR_BGR2RGB)

    return Image.fromarray(result_rgb)


def apply_grain(img: Image.Image, intensity: float) -> Image.Image:
    """필름 그레인 효과. intensity: 0.0(없음) ~ 1.0(강한 노이즈).

    밝기 채널에만 모노크롬 노이즈를 추가하여 자연스러운 필름 느낌을 만든다.
    """
    if intensity < 0.01:
        return img

    arr = np.array(img, dtype=np.float32)
    h, w = arr.shape[:2]

    # 모노크롬 노이즈 생성 (RGB 동일 → 컬러 노이즈 방지)
    strength = intensity * 40.0  # 최대 40 밝기값 편차
    noise = np.random.normal(0, strength, (h, w)).astype(np.float32)
    noise = noise[:, :, np.newaxis]

    # 밝은 영역보다 중간톤에 그레인이 더 잘 보이도록 가중치
    gray = np.mean(arr, axis=2, keepdims=True) / 255.0
    weight = 1.0 - np.abs(gray - 0.5) * 1.2  # 중간톤에서 최대
    weight = np.clip(weight, 0.3, 1.0)

    adjusted = arr + noise * weight
    adjusted = np.clip(adjusted, 0, 255)

    return Image.fromarray(adjusted.astype(np.uint8))


def apply_skin_smoothing(img: Image.Image, intensity: float) -> Image.Image:
    """피부 보정 (bilateral filter). intensity: 0.0 ~ 1.0."""
    if intensity < 0.01:
        return img

    arr = np.array(img)
    arr_bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)

    # intensity에 비례하여 필터 강도 조절
    d = int(5 + intensity * 10)           # diameter: 5 ~ 15
    sigma_color = 20 + intensity * 55     # 20 ~ 75
    sigma_space = 20 + intensity * 55     # 20 ~ 75

    smoothed = cv2.bilateralFilter(arr_bgr, d, sigma_color, sigma_space)

    # 원본과 블렌딩하여 자연스럽게
    blended = cv2.addWeighted(arr_bgr, 1.0 - intensity, smoothed, intensity, 0)
    result_rgb = cv2.cvtColor(blended, cv2.COLOR_BGR2RGB)

    return Image.fromarray(result_rgb)


def apply_sharpness(img: Image.Image, factor: float) -> Image.Image:
    """선명도 조절. factor: -1.0 ~ +1.0 (0 = 원본)."""
    enhancer = ImageEnhance.Sharpness(img)
    return enhancer.enhance(1.0 + factor)


# ── 잡티 제거 (Blemish Removal) ──

# MediaPipe Face Mesh 피부 영역 인덱스 (볼, 이마, 턱, 코 등 / 눈·입·눈썹 제외)
_SKIN_FACE_OVAL = [
    10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379,
    378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127,
    162, 21, 54, 103, 67, 109,
]
_LEFT_EYE = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
_RIGHT_EYE = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
_LIPS = [
    61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 308, 324, 318, 402,
    317, 14, 87, 178, 88, 95,
]
_LEFT_EYEBROW = [70, 63, 105, 66, 107, 55, 65, 52, 53, 46]
_RIGHT_EYEBROW = [300, 293, 334, 296, 336, 285, 295, 282, 283, 276]


def _get_skin_mask(img_rgb: np.ndarray) -> np.ndarray | None:
    """MediaPipe FaceLandmarker로 피부 영역 마스크를 생성한다.

    반환: 얼굴 경계에서 충분히 안쪽으로 침식된 피부 마스크, 얼굴 미감지 시 None.
    다중 얼굴이면 모든 얼굴의 마스크를 합친다.
    """
    if not _MP_AVAILABLE or _MP_MODEL_PATH is None:
        return None

    h, w = img_rgb.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)

    base_options = mp.tasks.BaseOptions(model_asset_path=_MP_MODEL_PATH)
    options = mp.tasks.vision.FaceLandmarkerOptions(
        base_options=base_options,
        num_faces=5,
        min_face_detection_confidence=0.5,
        min_face_presence_confidence=0.5,
    )
    landmarker = mp.tasks.vision.FaceLandmarker.create_from_options(options)
    try:
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
        results = landmarker.detect(mp_image)
    finally:
        landmarker.close()

    if not results.face_landmarks:
        return None

    for face_lms in results.face_landmarks:
        def _idx_to_pt(idx: int) -> tuple[int, int]:
            lm = face_lms[idx]
            return int(lm.x * w), int(lm.y * h)

        # 얼굴 윤곽 마스크
        oval_pts = np.array([_idx_to_pt(i) for i in _SKIN_FACE_OVAL], dtype=np.int32)
        face_mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(face_mask, [oval_pts], 255)

        # 제외 영역 (눈, 입, 눈썹) — 넉넉하게 확장하여 제외
        for region in (_LEFT_EYE, _RIGHT_EYE, _LIPS, _LEFT_EYEBROW, _RIGHT_EYEBROW):
            pts = np.array([_idx_to_pt(i) for i in region], dtype=np.int32)
            cv2.fillConvexPoly(face_mask, pts, 0)

        # 코 영역 일부 제외 (콧구멍·코 다리 — 자연스러운 음영이 잡티로 오감지됨)
        nose_bridge = [6, 197, 195, 5, 4]
        nose_tip = [1, 2, 98, 327, 168]
        for region in (nose_bridge, nose_tip):
            pts = np.array([_idx_to_pt(i) for i in region], dtype=np.int32)
            cv2.fillConvexPoly(face_mask, pts, 0)

        # ★ 핵심: 경계에서 충분히 안쪽으로 침식 — 턱선·이마선 근처 오감지 방지
        short_side = min(h, w)
        erode_px = max(8, int(short_side * 0.025))
        erode_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (erode_px, erode_px)
        )
        face_mask = cv2.erode(face_mask, erode_kernel, iterations=1)

        mask = cv2.bitwise_or(mask, face_mask)

    return mask


def _detect_blemishes(
    img_bgr: np.ndarray,
    skin_mask: np.ndarray,
    intensity: float,
) -> np.ndarray:
    """LAB A·B 채널(색상 이상치)로 잡티를 탐지한다.

    밝기(L) 편차는 음영·조명이므로 무시하고,
    A·B 채널(빨강-녹색, 노랑-파랑)만으로 색상 이상 점을 탐지한다.
    반환: 잡티 영역이 255인 단채널 마스크.
    """
    h, w = img_bgr.shape[:2]
    short_side = min(h, w)
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)

    # A·B 채널만 사용 (L 채널 제외 → 음영·조명 무시)
    a_ch = lab[:, :, 1].astype(np.float32)
    b_ch = lab[:, :, 2].astype(np.float32)

    # 해상도 적응형 커널 — 잡티보다 훨씬 큰 스케일의 로컬 평균
    ksize = max(31, int(short_side * 0.06)) | 1
    a_mean = cv2.GaussianBlur(a_ch, (ksize, ksize), 0)
    b_mean = cv2.GaussianBlur(b_ch, (ksize, ksize), 0)

    # 색상 편차 (A·B만)
    diff = np.sqrt((a_ch - a_mean) ** 2 + (b_ch - b_mean) ** 2)

    # 높은 임계값: intensity 0.3 → 20, 1.0 → 10
    threshold = 22.0 - intensity * 12.0
    threshold = max(threshold, 8.0)

    blemish_mask = (diff > threshold).astype(np.uint8) * 255
    blemish_mask = cv2.bitwise_and(blemish_mask, skin_mask)

    # 모폴로지로 노이즈 제거
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    blemish_mask = cv2.morphologyEx(blemish_mask, cv2.MORPH_OPEN, kernel, iterations=2)

    # 개별 잡티 크기 필터 — 작은 점만 (점·여드름 크기)
    min_area = max(4, int((short_side * 0.002) ** 2))
    max_area = int((short_side * 0.02) ** 2)  # 2% 이하 — 아주 작은 점만
    contours, _ = cv2.findContours(blemish_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filtered_mask = np.zeros_like(blemish_mask)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if min_area <= area <= max_area:
            cv2.drawContours(filtered_mask, [cnt], -1, 255, cv2.FILLED)

    return filtered_mask


def _inpaint_blemishes(img_bgr: np.ndarray, blemish_mask: np.ndarray) -> np.ndarray:
    """OpenCV INPAINT_TELEA로 잡티 영역을 복원한다."""
    return cv2.inpaint(img_bgr, blemish_mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA)


def apply_blemish_removal(img: Image.Image, intensity: float) -> Image.Image:
    """잡티 자동 제거. intensity: 0.0(비활성) ~ 1.0(최대 감도).

    파이프라인:
      1. MediaPipe Face Mesh로 피부 마스크 (경계 침식)
      2. LAB A·B 채널 색상 이상치로 잡티 탐지
      3. OpenCV inpainting으로 잡티 영역만 복원
      4. 잡티 영역만 마스크 기반 블렌딩
    """
    if intensity < 0.01:
        return img

    arr_rgb = np.array(img)
    arr_bgr = cv2.cvtColor(arr_rgb, cv2.COLOR_RGB2BGR)

    # 1. 피부 마스크 (얼굴 미감지 → 원본 반환)
    skin_mask = _get_skin_mask(arr_rgb)
    if skin_mask is None:
        return img

    # 2. 잡티 탐지
    blemish_mask = _detect_blemishes(arr_bgr, skin_mask, intensity)

    # 빈 마스크 → 깨끗한 피부, 원본 반환
    if cv2.countNonZero(blemish_mask) == 0:
        return img

    # 3. 인페인팅
    inpainted = _inpaint_blemishes(arr_bgr, blemish_mask)

    # 4. 잡티 픽셀만 교체 (마스크 경계를 살짝 블러하여 자연스럽게)
    blend_mask = cv2.GaussianBlur(blemish_mask, (5, 5), 0)
    alpha = (blend_mask.astype(np.float32) / 255.0 * intensity)[:, :, np.newaxis]

    result_bgr = arr_bgr.astype(np.float32) * (1.0 - alpha) + inpainted.astype(np.float32) * alpha
    result_bgr = np.clip(result_bgr, 0, 255).astype(np.uint8)
    result_rgb = cv2.cvtColor(result_bgr, cv2.COLOR_BGR2RGB)

    return Image.fromarray(result_rgb)


# ── AI 자동 편집 (autoEdits) ──


def apply_smart_crop(img: Image.Image, crop: dict) -> Image.Image:
    """AI가 추천한 영역으로 이미지를 크롭(줌인)한다.

    crop: {"x": 0~1, "y": 0~1, "width": 0~1, "height": 0~1} (정규화 좌표)
    """
    try:
        w, h = img.size
        x = max(0.0, min(1.0, float(crop.get("x", 0))))
        y = max(0.0, min(1.0, float(crop.get("y", 0))))
        cw = max(0.05, min(1.0, float(crop.get("width", 1))))
        ch = max(0.05, min(1.0, float(crop.get("height", 1))))

        left = int(x * w)
        top = int(y * h)
        right = min(w, int((x + cw) * w))
        bottom = min(h, int((y + ch) * h))

        if right - left < 50 or bottom - top < 50:
            return img

        return img.crop((left, top, right, bottom))
    except Exception:
        return img


def apply_instagram_ratio(img: Image.Image, ratio: str) -> Image.Image:
    """인스타그램 최적 비율로 중앙 크롭한다.

    ratio: "4:5" (피드 최적) 또는 "1:1" (정사각형)
    """
    try:
        w, h = img.size
        if ratio == "4:5":
            target = 4 / 5
        elif ratio == "1:1":
            target = 1.0
        else:
            return img

        current = w / h
        if abs(current - target) < 0.02:
            return img  # 이미 비슷한 비율

        if current > target:
            # 가로가 더 넓음 → 좌우 크롭
            new_w = int(h * target)
            left = (w - new_w) // 2
            return img.crop((left, 0, left + new_w, h))
        else:
            # 세로가 더 길음 → 상하 크롭
            new_h = int(w / target)
            top = (h - new_h) // 2
            return img.crop((0, top, w, top + new_h))
    except Exception:
        return img


def apply_object_removal(img: Image.Image, areas: list[dict]) -> Image.Image:
    """AI가 지정한 영역의 불필요한 요소를 인페인팅으로 제거한다.

    areas: [{"x": 0~1, "y": 0~1, "width": 0~1, "height": 0~1}, ...]
    """
    if not areas:
        return img

    try:
        arr_rgb = np.array(img)
        arr_bgr = cv2.cvtColor(arr_rgb, cv2.COLOR_RGB2BGR)
        h, w = arr_bgr.shape[:2]

        mask = np.zeros((h, w), dtype=np.uint8)
        for area in areas:
            ax = max(0.0, min(1.0, float(area.get("x", 0))))
            ay = max(0.0, min(1.0, float(area.get("y", 0))))
            aw = max(0.0, min(1.0, float(area.get("width", 0))))
            ah = max(0.0, min(1.0, float(area.get("height", 0))))

            left = int(ax * w)
            top = int(ay * h)
            right = min(w, int((ax + aw) * w))
            bottom = min(h, int((ay + ah) * h))

            if right - left < 2 or bottom - top < 2:
                continue

            # 영역이 이미지의 30% 이상이면 무시 (안전장치)
            if (right - left) * (bottom - top) > w * h * 0.3:
                continue

            mask[top:bottom, left:right] = 255

        if cv2.countNonZero(mask) == 0:
            return img

        # 인페인팅 반경을 영역 크기에 비례하게 설정
        short_side = min(h, w)
        inpaint_radius = max(5, int(short_side * 0.01))
        inpainted = cv2.inpaint(arr_bgr, mask, inpaint_radius, cv2.INPAINT_TELEA)

        result_rgb = cv2.cvtColor(inpainted, cv2.COLOR_BGR2RGB)
        return Image.fromarray(result_rgb)
    except Exception:
        return img


def apply_straighten(img: Image.Image, angle: float) -> Image.Image:
    """이미지 수평 보정. angle: 회전 각도 (도 단위, 시계방향 양수).

    회전 후 생기는 검은 여백을 자동으로 크롭하여 깔끔한 결과를 반환한다.
    안전장치: ±15도 초과 시 의도적 기울기로 판단하여 무시.
    """
    if abs(angle) < 0.1:
        return img

    # 안전장치: 극단적 각도는 의도적 구도로 판단
    if abs(angle) > 15.0:
        log.warning("straighten: angle %.1f° exceeds ±15° limit, skipping", angle)
        return img

    try:
        w, h = img.size
        arr = np.array(img)
        arr_bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)

        # 이미지 중심 기준 회전
        center = (w / 2, h / 2)
        rot_mat = cv2.getRotationMatrix2D(center, angle, 1.0)

        # 회전 후 전체 이미지가 들어가도록 캔버스 확장
        cos_a = abs(rot_mat[0, 0])
        sin_a = abs(rot_mat[0, 1])
        new_w = int(h * sin_a + w * cos_a)
        new_h = int(h * cos_a + w * sin_a)

        rot_mat[0, 2] += (new_w - w) / 2
        rot_mat[1, 2] += (new_h - h) / 2

        rotated = cv2.warpAffine(
            arr_bgr, rot_mat, (new_w, new_h),
            flags=cv2.INTER_LANCZOS4,
            borderMode=cv2.BORDER_REPLICATE,
        )

        # 검은 여백 없이 원본 영역만 크롭 (내접 직사각형)
        rad = abs(angle) * np.pi / 180.0
        if w > h:
            crop_w = int(w * cos_a - h * sin_a)
            crop_h = int(h * cos_a - w * sin_a)
        else:
            crop_w = int(w * cos_a - h * sin_a)
            crop_h = int(h * cos_a - w * sin_a)

        # 내접 직사각형이 유효하지 않으면 간단한 비율 축소
        if crop_w <= 0 or crop_h <= 0:
            shrink = cos_a
            crop_w = int(w * shrink)
            crop_h = int(h * shrink)

        cx, cy = new_w // 2, new_h // 2
        left = max(0, cx - crop_w // 2)
        top = max(0, cy - crop_h // 2)
        right = min(new_w, left + crop_w)
        bottom = min(new_h, top + crop_h)

        if right - left < 50 or bottom - top < 50:
            return img

        cropped = rotated[top:bottom, left:right]
        result_rgb = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)

        log.info("straighten: rotated %.1f°, size %dx%d → %dx%d",
                 angle, w, h, right - left, bottom - top)
        return Image.fromarray(result_rgb)

    except Exception as exc:
        log.warning("straighten failed: %s", exc)
        return img


def apply_auto_edits(img: Image.Image, auto_edits: dict) -> Image.Image:
    """AI autoEdits를 순서대로 적용한다.

    순서: 수평 보정 → 불필요 요소 제거 → 스마트 크롭 → 인스타 비율
    (수평 보정을 가장 먼저 해야 이후 크롭이 정확함)
    """
    # 0. 수평 보정 (기울기 교정)
    straighten = auto_edits.get("straighten")
    if straighten is not None:
        try:
            angle = float(straighten)
            img = apply_straighten(img, angle)
        except (TypeError, ValueError):
            pass

    # 1. 불필요한 요소 제거
    remove_areas = auto_edits.get("remove_areas")
    if remove_areas and isinstance(remove_areas, list):
        img = apply_object_removal(img, remove_areas)

    # 2. 스마트 크롭 (줌/리프레임)
    crop = auto_edits.get("crop")
    if crop and isinstance(crop, dict):
        img = apply_smart_crop(img, crop)

    # 3. 인스타그램 비율 크롭
    ig_ratio = auto_edits.get("instagram_ratio")
    if ig_ratio and isinstance(ig_ratio, str):
        img = apply_instagram_ratio(img, ig_ratio)

    return img


# ── 영역별 스마트 보정 ──


def detect_regions(img: Image.Image) -> dict[str, np.ndarray]:
    """HSV 기반 하늘 감지 + MediaPipe 얼굴 감지 + 나머지=배경.

    반환: {"sky": mask, "face": mask, "background": mask}
    각 마스크는 0~255 uint8 단채널. 영역이 없으면 해당 키가 빈 마스크(전체 0).
    """
    arr_rgb = np.array(img, dtype=np.uint8)
    h, w = arr_rgb.shape[:2]
    arr_bgr = cv2.cvtColor(arr_rgb, cv2.COLOR_RGB2BGR)
    hsv = cv2.cvtColor(arr_bgr, cv2.COLOR_BGR2HSV)

    # ── 하늘 감지: HSV 범위 + 이미지 상단 가중치 ──
    # H: 90~130 (파란~시안), S: 30+, V: 100+
    lower_sky = np.array([90, 30, 100], dtype=np.uint8)
    upper_sky = np.array([130, 255, 255], dtype=np.uint8)
    sky_color_mask = cv2.inRange(hsv, lower_sky, upper_sky)

    # 이미지 상단 50%에 가중치 부여 (하늘은 대부분 위쪽)
    top_weight = np.zeros((h, w), dtype=np.float32)
    for row in range(h):
        weight = max(0.0, 1.0 - (row / (h * 0.5)))
        top_weight[row, :] = weight
    # 상단 가중치를 적용한 하늘 마스크
    sky_weighted = (sky_color_mask.astype(np.float32) / 255.0) * (0.4 + 0.6 * top_weight)
    sky_mask = (sky_weighted > 0.3).astype(np.uint8) * 255

    # 모폴로지로 노이즈 제거 및 영역 연결
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    sky_mask = cv2.morphologyEx(sky_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    sky_mask = cv2.morphologyEx(sky_mask, cv2.MORPH_OPEN, kernel, iterations=1)

    # 하늘 영역이 이미지의 5% 미만이면 하늘 없음으로 처리
    sky_ratio = cv2.countNonZero(sky_mask) / (h * w)
    if sky_ratio < 0.05:
        sky_mask = np.zeros((h, w), dtype=np.uint8)

    # ── 얼굴/피부 감지: 기존 _get_skin_mask() 재사용 ──
    face_mask = _get_skin_mask(arr_rgb)
    if face_mask is None:
        face_mask = np.zeros((h, w), dtype=np.uint8)

    # ── 배경: 하늘도 얼굴도 아닌 나머지 ──
    combined = cv2.bitwise_or(sky_mask, face_mask)
    background_mask = cv2.bitwise_not(combined)

    return {
        "sky": sky_mask,
        "face": face_mask,
        "background": background_mask,
    }


def apply_regional_transforms(
    img: Image.Image,
    regions: dict[str, np.ndarray],
    region_params: dict[str, dict[str, float]],
) -> Image.Image:
    """영역별로 다른 보정을 적용한 뒤 마스크 경계를 블렌딩.

    region_params 예시:
    {
      "sky": {"brightness": 0.1, "saturation": -0.1, "temperature": 0.0},
      "face": {"brightness": 0.1, "blemish_removal": 0.3, "skin_smoothing": 0.2},
      "background": {"brightness": 0.0, "contrast": 0.1, "saturation": -0.05}
    }
    """
    arr_rgb = np.array(img, dtype=np.float32)

    # 영역별 변형 적용 후 블렌딩
    result = arr_rgb.copy()

    # 사용 가능한 변형 함수 맵
    transform_funcs = {
        "brightness": adjust_brightness,
        "contrast": adjust_contrast,
        "saturation": adjust_saturation,
        "temperature": adjust_color_temperature,
        "highlights": adjust_highlights,
        "shadows": adjust_shadows,
        "blemish_removal": apply_blemish_removal,
        "skin_smoothing": apply_skin_smoothing,
        "sharpness": apply_sharpness,
    }

    for region_name, params in region_params.items():
        if not params or region_name not in regions:
            continue

        mask = regions[region_name]
        if cv2.countNonZero(mask) == 0:
            continue

        # 이 영역에 해당하는 변형을 순서대로 적용
        region_img = img.copy()
        for param_name, value in params.items():
            if abs(value) < 0.01:
                continue
            func = transform_funcs.get(param_name)
            if func is not None:
                region_img = func(region_img, value)

        # 마스크 경계를 가우시안 블러로 부드럽게
        blur_size = max(21, int(min(mask.shape) * 0.03)) | 1
        soft_mask = cv2.GaussianBlur(mask, (blur_size, blur_size), 0)
        alpha = (soft_mask.astype(np.float32) / 255.0)[:, :, np.newaxis]

        region_arr = np.array(region_img, dtype=np.float32)
        result = result * (1.0 - alpha) + region_arr * alpha

    return Image.fromarray(np.clip(result, 0, 255).astype(np.uint8))


# ── 얼굴/체형 보정 (MLS Warp) ──

# 얼굴 윤곽 랜드마크 인덱스 (MediaPipe 478개 중 양쪽 볼·턱선)
_FACE_CONTOUR_LEFT = [234, 93, 132, 58, 172, 136, 150, 149, 176, 148, 152]
_FACE_CONTOUR_RIGHT = [454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152]

# 턱선 인덱스 (V라인)
_JAW_LEFT = [172, 136, 150, 149, 176, 148]
_JAW_RIGHT = [397, 365, 379, 378, 400, 377]
_JAW_TIP = [152]

# 눈 인덱스 (방사형 확대용)
_LEFT_EYE_CENTER = 468   # iris center (478 mesh)
_RIGHT_EYE_CENTER = 473
_LEFT_EYE_CONTOUR = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
_RIGHT_EYE_CONTOUR = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]


def _mls_similarity_warp(
    img: np.ndarray,
    src_pts: np.ndarray,
    dst_pts: np.ndarray,
    alpha: float = 1.0,
) -> np.ndarray:
    """MLS (Moving Least Squares) Similarity Warp.

    src_pts, dst_pts: (N, 2) float32 배열 — (x, y) 좌표.
    원본 이미지의 각 픽셀을 역워프하여 매핑한다.
    경계 접합선 없이 부드러운 변형을 생성한다.
    """
    h, w = img.shape[:2]
    n = len(src_pts)
    if n < 2:
        return img

    # 출력 이미지의 픽셀 좌표 그리드
    ys, xs = np.mgrid[0:h, 0:w].astype(np.float32)
    grid = np.stack([xs, ys], axis=-1)  # (H, W, 2)
    flat_grid = grid.reshape(-1, 2)  # (H*W, 2)
    num_pixels = flat_grid.shape[0]

    # 각 제어점의 가중치 계산: w_i = 1 / |p_i - v|^(2*alpha)
    # (N, num_pixels) 가중치 배열
    weights = np.zeros((n, num_pixels), dtype=np.float32)
    for i in range(n):
        diff = flat_grid - dst_pts[i]  # (num_pixels, 2)
        dist_sq = np.sum(diff ** 2, axis=1) + 1e-6  # (num_pixels,)
        weights[i] = 1.0 / (dist_sq ** alpha)

    # 가중 합계
    w_sum = np.sum(weights, axis=0)  # (num_pixels,)

    # 가중 평균 displacement
    disp = np.zeros_like(flat_grid)  # (num_pixels, 2)
    for i in range(n):
        d = src_pts[i] - dst_pts[i]  # (2,)
        disp += weights[i, :, np.newaxis] * d[np.newaxis, :]

    disp = disp / w_sum[:, np.newaxis]

    # 역워프: 목적 좌표에서 원본 좌표 계산
    map_xy = flat_grid + disp
    map_x = map_xy[:, 0].reshape(h, w).astype(np.float32)
    map_y = map_xy[:, 1].reshape(h, w).astype(np.float32)

    return cv2.remap(img, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


def _warp_with_mask(
    img: np.ndarray,
    src_pts: np.ndarray,
    dst_pts: np.ndarray,
    roi: tuple[int, int, int, int] | None = None,
) -> np.ndarray:
    """MLS 워프 + 가우시안 마스크 블렌딩으로 원본과 자연스럽게 합성.

    roi: (x, y, w, h) — 워프 영향 영역. None이면 전체.
    """
    warped = _mls_similarity_warp(img, src_pts, dst_pts)

    if roi is None:
        return warped

    h, w = img.shape[:2]
    rx, ry, rw, rh = roi
    # 안전한 범위
    rx = max(0, rx)
    ry = max(0, ry)
    rw = min(w - rx, rw)
    rh = min(h - ry, rh)

    # 타원형 마스크 (부드러운 경계)
    mask = np.zeros((h, w), dtype=np.float32)
    cv2.ellipse(
        mask,
        center=(rx + rw // 2, ry + rh // 2),
        axes=(rw // 2, rh // 2),
        angle=0, startAngle=0, endAngle=360,
        color=1.0, thickness=-1,
    )
    blur_size = max(31, int(min(rw, rh) * 0.3)) | 1
    mask = cv2.GaussianBlur(mask, (blur_size, blur_size), 0)
    mask = mask[:, :, np.newaxis]

    return (img.astype(np.float32) * (1 - mask) + warped.astype(np.float32) * mask).astype(np.uint8)


def apply_face_reshape(
    img: Image.Image,
    face_slim: float = 0.0,
    jaw_sharpen: float = 0.0,
    eye_enlarge: float = 0.0,
) -> Image.Image:
    """얼굴 보정 — MediaPipe 478 랜드마크 기반 MLS 워프.

    face_slim: 0~1 (얼굴 양쪽을 중심축 방향으로)
    jaw_sharpen: 0~1 (턱 V라인)
    eye_enlarge: 0~1 (눈 확대)

    얼굴 미감지 시 원본 반환. 다중 얼굴은 각각 독립 적용.
    """
    if face_slim < 0.01 and jaw_sharpen < 0.01 and eye_enlarge < 0.01:
        return img

    if not _MP_AVAILABLE or _MP_MODEL_PATH is None:
        return img

    arr_rgb = np.array(img, dtype=np.uint8)
    h, w = arr_rgb.shape[:2]

    base_options = mp.tasks.BaseOptions(model_asset_path=_MP_MODEL_PATH)
    options = mp.tasks.vision.FaceLandmarkerOptions(
        base_options=base_options,
        num_faces=5,
        min_face_detection_confidence=0.5,
        min_face_presence_confidence=0.5,
    )
    landmarker = mp.tasks.vision.FaceLandmarker.create_from_options(options)
    try:
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=arr_rgb)
        results = landmarker.detect(mp_image)
    finally:
        landmarker.close()

    if not results.face_landmarks:
        return img

    result_arr = arr_rgb.copy()

    for face_lms in results.face_landmarks:
        def _pt(idx: int) -> tuple[float, float]:
            lm = face_lms[idx]
            return lm.x * w, lm.y * h

        src_all: list[list[float]] = []
        dst_all: list[list[float]] = []

        # 얼굴 중심축 (코 중앙)
        nose_tip = _pt(1)
        cx = nose_tip[0]

        # ── face_slim: 볼 양쪽을 중심 방향으로 ──
        if face_slim >= 0.01:
            strength = face_slim * 0.08  # 최대 8% 이동
            for idx in _FACE_CONTOUR_LEFT:
                px, py = _pt(idx)
                dx = (cx - px) * strength
                src_all.append([px, py])
                dst_all.append([px + dx, py])
            for idx in _FACE_CONTOUR_RIGHT:
                if idx == 152:
                    continue  # 턱 끝은 중복
                px, py = _pt(idx)
                dx = (cx - px) * strength
                src_all.append([px, py])
                dst_all.append([px + dx, py])

        # ── jaw_sharpen: 턱선을 V자로 ──
        if jaw_sharpen >= 0.01:
            strength_x = jaw_sharpen * 0.06
            strength_y = jaw_sharpen * 0.03
            jaw_tip_x, jaw_tip_y = _pt(152)
            for idx in _JAW_LEFT:
                px, py = _pt(idx)
                dx = (cx - px) * strength_x
                dy = (jaw_tip_y - py) * strength_y
                src_all.append([px, py])
                dst_all.append([px + dx, py + dy])
            for idx in _JAW_RIGHT:
                px, py = _pt(idx)
                dx = (cx - px) * strength_x
                dy = (jaw_tip_y - py) * strength_y
                src_all.append([px, py])
                dst_all.append([px + dx, py + dy])

        # ── eye_enlarge: 눈 윤곽 방사형 확대 ──
        if eye_enlarge >= 0.01:
            strength = eye_enlarge * 0.12  # 최대 12% 확대
            # 왼쪽 눈
            if len(face_lms) > _LEFT_EYE_CENTER:
                ecx, ecy = _pt(_LEFT_EYE_CENTER)
            else:
                # iris 랜드마크 없으면 눈 중앙 계산
                pts = [_pt(i) for i in _LEFT_EYE_CONTOUR]
                ecx = sum(p[0] for p in pts) / len(pts)
                ecy = sum(p[1] for p in pts) / len(pts)
            for idx in _LEFT_EYE_CONTOUR:
                px, py = _pt(idx)
                dx = (px - ecx) * strength
                dy = (py - ecy) * strength
                src_all.append([px, py])
                dst_all.append([px + dx, py + dy])

            # 오른쪽 눈
            if len(face_lms) > _RIGHT_EYE_CENTER:
                ecx, ecy = _pt(_RIGHT_EYE_CENTER)
            else:
                pts = [_pt(i) for i in _RIGHT_EYE_CONTOUR]
                ecx = sum(p[0] for p in pts) / len(pts)
                ecy = sum(p[1] for p in pts) / len(pts)
            for idx in _RIGHT_EYE_CONTOUR:
                px, py = _pt(idx)
                dx = (px - ecx) * strength
                dy = (py - ecy) * strength
                src_all.append([px, py])
                dst_all.append([px + dx, py + dy])

        if not src_all:
            continue

        src_pts = np.array(src_all, dtype=np.float32)
        dst_pts = np.array(dst_all, dtype=np.float32)

        # 얼굴 바운딩 영역 계산 (마스크용)
        all_face_pts = [_pt(i) for i in range(min(len(face_lms), 468))]
        fxs = [p[0] for p in all_face_pts]
        fys = [p[1] for p in all_face_pts]
        margin = int(min(h, w) * 0.05)
        roi = (
            max(0, int(min(fxs)) - margin),
            max(0, int(min(fys)) - margin),
            min(w, int(max(fxs) - min(fxs)) + 2 * margin),
            min(h, int(max(fys) - min(fys)) + 2 * margin),
        )

        result_arr = _warp_with_mask(result_arr, src_pts, dst_pts, roi)

    return Image.fromarray(result_arr)


def apply_body_reshape(
    img: Image.Image,
    leg_stretch: float = 0.0,
    shoulder_width: float = 0.0,
    waist_slim: float = 0.0,
) -> Image.Image:
    """체형 보정 — MediaPipe Pose 33 랜드마크 기반.

    leg_stretch: 0~1 (힙 아래 수직 스트레칭)
    shoulder_width: -1~1 (음수=좁게, 양수=넓게)
    waist_slim: 0~1 (허리 양쪽을 안쪽으로)

    바디 미감지 시 원본 반환. 다중 바디는 가장 신뢰도 높은 것만.
    """
    if abs(leg_stretch) < 0.01 and abs(shoulder_width) < 0.01 and abs(waist_slim) < 0.01:
        return img

    if not _MP_AVAILABLE or _POSE_MODEL_PATH is None:
        return img

    arr_rgb = np.array(img, dtype=np.uint8)
    h, w = arr_rgb.shape[:2]

    base_options = mp.tasks.BaseOptions(model_asset_path=_POSE_MODEL_PATH)
    options = mp.tasks.vision.PoseLandmarkerOptions(
        base_options=base_options,
        num_poses=3,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
    )
    landmarker = mp.tasks.vision.PoseLandmarker.create_from_options(options)
    try:
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=arr_rgb)
        results = landmarker.detect(mp_image)
    finally:
        landmarker.close()

    if not results.pose_landmarks:
        return img

    # 가장 큰(키가 큰) 바디만 선택
    best_pose = None
    best_height = 0.0
    for pose_lms in results.pose_landmarks:
        ys = [lm.y for lm in pose_lms]
        pose_h = max(ys) - min(ys)
        if pose_h > best_height:
            best_height = pose_h
            best_pose = pose_lms

    if best_pose is None:
        return img

    pose = best_pose

    def _pt(idx: int) -> tuple[float, float]:
        lm = pose[idx]
        return lm.x * w, lm.y * h

    def _vis(idx: int) -> float:
        return pose[idx].visibility if hasattr(pose[idx], 'visibility') else 1.0

    result_arr = arr_rgb.copy()

    # ── leg_stretch: 힙 아래 영역 수직 스트레칭 ──
    if leg_stretch >= 0.01:
        # Pose 랜드마크: 23=왼쪽 힙, 24=오른쪽 힙, 27=왼쪽 발목, 28=오른쪽 발목
        left_ankle_vis = _vis(27)
        right_ankle_vis = _vis(28)

        if left_ankle_vis >= 0.5 or right_ankle_vis >= 0.5:
            lhip = _pt(23)
            rhip = _pt(24)
            hip_y = (lhip[1] + rhip[1]) / 2.0

            # 힙 아래 영역을 수직 스케일링 (단순 remap)
            stretch_factor = 1.0 + leg_stretch * 0.15  # 최대 15% 늘리기

            # remap: hip_y 위는 그대로, 아래는 스트레칭
            map_y = np.zeros((h, w), dtype=np.float32)
            map_x = np.arange(w, dtype=np.float32)[np.newaxis, :].repeat(h, axis=0)

            hip_y_int = int(hip_y)
            for row in range(h):
                if row <= hip_y_int:
                    map_y[row, :] = row
                else:
                    # 목적 row → 원본 row (역워프)
                    orig_row = hip_y + (row - hip_y) / stretch_factor
                    map_y[row, :] = min(h - 1, orig_row)

            # 경계 블렌딩 마스크
            mask = np.zeros((h, w), dtype=np.float32)
            mask[hip_y_int:, :] = 1.0
            # 힙 주변 부드러운 전환
            transition = max(10, int(h * 0.03))
            for row in range(max(0, hip_y_int - transition), min(h, hip_y_int + transition)):
                t = (row - (hip_y_int - transition)) / (2 * transition)
                mask[row, :] = max(0.0, min(1.0, t))

            stretched = cv2.remap(result_arr, map_x, map_y, cv2.INTER_LINEAR,
                                  borderMode=cv2.BORDER_REPLICATE)
            mask_3d = mask[:, :, np.newaxis]
            result_arr = (result_arr.astype(np.float32) * (1 - mask_3d)
                          + stretched.astype(np.float32) * mask_3d).astype(np.uint8)

    # ── shoulder_width: 어깨 너비 조절 ──
    if abs(shoulder_width) >= 0.01:
        # Pose: 11=왼쪽 어깨, 12=오른쪽 어깨
        ls = _pt(11)
        rs = _pt(12)
        mid_x = (ls[0] + rs[0]) / 2.0
        mid_y = (ls[1] + rs[1]) / 2.0
        strength = shoulder_width * 0.06  # 최대 6%

        src_pts = np.array([
            [ls[0], ls[1]],
            [rs[0], rs[1]],
        ], dtype=np.float32)
        dst_pts = np.array([
            [ls[0] + (ls[0] - mid_x) * strength, ls[1]],
            [rs[0] + (rs[0] - mid_x) * strength, rs[1]],
        ], dtype=np.float32)

        shoulder_w = abs(rs[0] - ls[0])
        shoulder_h = shoulder_w * 0.6
        roi = (
            max(0, int(min(ls[0], rs[0]) - shoulder_w * 0.3)),
            max(0, int(mid_y - shoulder_h)),
            min(w, int(shoulder_w + shoulder_w * 0.6)),
            min(h, int(shoulder_h * 2)),
        )
        result_arr = _warp_with_mask(result_arr, src_pts, dst_pts, roi)

    # ── waist_slim: 허리 양쪽을 안쪽으로 ──
    if waist_slim >= 0.01:
        # Pose: 23=왼쪽 힙, 24=오른쪽 힙, 11=왼쪽 어깨, 12=오른쪽 어깨
        lhip = _pt(23)
        rhip = _pt(24)
        ls = _pt(11)
        rs = _pt(12)

        # 허리 위치 = 어깨와 힙의 중간
        waist_y = (ls[1] + lhip[1]) / 2.0
        mid_x = (lhip[0] + rhip[0]) / 2.0
        waist_left_x = min(ls[0], lhip[0])
        waist_right_x = max(rs[0], rhip[0])

        strength = waist_slim * 0.06  # 최대 6%

        src_pts = np.array([
            [waist_left_x, waist_y],
            [waist_right_x, waist_y],
        ], dtype=np.float32)
        dst_pts = np.array([
            [waist_left_x + (mid_x - waist_left_x) * strength, waist_y],
            [waist_right_x + (mid_x - waist_right_x) * strength, waist_y],
        ], dtype=np.float32)

        waist_w = abs(waist_right_x - waist_left_x)
        waist_h = abs(lhip[1] - ls[1])
        roi = (
            max(0, int(waist_left_x - waist_w * 0.2)),
            max(0, int(waist_y - waist_h * 0.4)),
            min(w, int(waist_w + waist_w * 0.4)),
            min(h, int(waist_h * 0.8)),
        )
        result_arr = _warp_with_mask(result_arr, src_pts, dst_pts, roi)

    return Image.fromarray(result_arr)


# ── 통합 변형 ──


def apply_all_transforms(
    img: Image.Image,
    brightness: float = 0.0,
    contrast: float = 0.0,
    clarity: float = 0.0,
    dehaze: float = 0.0,
    highlights: float = 0.0,
    shadows: float = 0.0,
    saturation: float = 0.0,
    temperature: float = 0.0,
    blemish_removal: float = 0.0,
    skin_smoothing: float = 0.0,
    vignette: float = 0.0,
    sharpness: float = 0.0,
    grain: float = 0.0,
    tone_curve_preset: str = "linear",
    tone_curve_strength: float = 0.0,
    split_shadow_hue: float = 0.0,
    split_shadow_strength: float = 0.0,
    split_highlight_hue: float = 0.0,
    split_highlight_strength: float = 0.0,
    hsl_adjust: dict[str, dict[str, float]] | None = None,
    face_slim: float = 0.0,
    jaw_sharpen: float = 0.0,
    eye_enlarge: float = 0.0,
    leg_stretch: float = 0.0,
    shoulder_width: float = 0.0,
    waist_slim: float = 0.0,
) -> Image.Image:
    """모든 변형을 순서대로 적용.

    순서: 얼굴 보정 → 체형 보정 → 하이라이트 → 쉐도우 → 톤 커브 → 밝기 → 대비
          → 선명감(Clarity) → 디헤이즈 → 색온도 → 채도 → HSL 선택적 색상
          → 스플릿 토닝 → 잡티 제거 → 피부보정 → 비네팅 → 선명도 → 그레인

    핵심 원칙:
    - 기하학적 변형(reshape)이 모든 색감 보정보다 먼저 실행
    - 하이라이트/쉐도우를 먼저 적용하여 다이나믹 레인지를 확보한 뒤 밝기 조절
    - 톤 커브를 밝기/대비 전에 적용하여 커브의 특성이 보존됨
    - Clarity는 대비 직후에 적용 (글로벌 대비 위에 로컬 대비 추가)
    - Dehaze는 색온도 전에 적용 (안개 제거로 복원된 색감에 색온도 적용)
    - 색온도를 대비 뒤에 적용하여 색상 변환의 클리핑 최소화
    - HSL 선택적 색상은 전체 채도 뒤, 스플릿 토닝 전에 적용
    - 스플릿 토닝은 색온도/채도 뒤에 적용하여 기본 색감 위에 색조를 입힘
    - 모든 밝기/대비/하이라이트/쉐도우는 LAB L채널에서 처리하여 색상 보존
    - 그레인은 최종 단계 (선명도 보정에 의해 노이즈가 강조되지 않도록)
    """
    result = img
    # ── 기하학적 변형(reshape)이 색감 보정보다 먼저 ──
    result = apply_face_reshape(result, face_slim, jaw_sharpen, eye_enlarge)
    result = apply_body_reshape(result, leg_stretch, shoulder_width, waist_slim)
    # ── 색감 보정 ──
    result = adjust_highlights(result, highlights)
    result = adjust_shadows(result, shadows)
    result = apply_tone_curve(result, tone_curve_preset, tone_curve_strength)
    result = adjust_brightness(result, brightness)
    result = adjust_contrast(result, contrast)
    result = adjust_clarity(result, clarity)
    result = apply_dehaze(result, dehaze)
    result = adjust_color_temperature(result, temperature)
    result = adjust_saturation(result, saturation)
    result = apply_hsl_adjust(result, hsl_adjust)
    result = apply_split_toning(
        result, split_shadow_hue, split_shadow_strength,
        split_highlight_hue, split_highlight_strength,
    )
    result = apply_blemish_removal(result, blemish_removal)
    result = apply_skin_smoothing(result, skin_smoothing)
    result = apply_vignette(result, vignette)
    result = apply_sharpness(result, sharpness)
    result = apply_grain(result, grain)
    return result


# ── AI 분석 → 변형 파라미터 자동 계산 ──


def analysis_to_transform_params(analysis: dict[str, Any]) -> dict[str, float]:
    """AI 분석 JSON의 recommendedParams를 슬라이더 초기값으로 사용한다.

    AI가 사진을 직접 보고 추천한 값을 그대로 사용하고,
    recommendedParams가 없으면 기본값(0.0)으로 폴백한다.
    """
    default_params: dict[str, float] = {
        "brightness": 0.0,
        "contrast": 0.0,
        "clarity": 0.0,
        "dehaze": 0.0,
        "highlights": 0.0,
        "shadows": 0.0,
        "saturation": 0.0,
        "temperature": 0.0,
        "blemish_removal": 0.0,
        "skin_smoothing": 0.0,
        "vignette": 0.0,
        "sharpness": 0.0,
        "grain": 0.0,
    }

    _split_defaults = {
        "split_shadow_hue": 0.0,
        "split_shadow_strength": 0.0,
        "split_highlight_hue": 0.0,
        "split_highlight_strength": 0.0,
    }

    recommended = analysis.get("recommendedParams", {})
    if not recommended or not isinstance(recommended, dict):
        return {
            **default_params,
            "tone_curve_preset": "linear", "tone_curve_strength": 0.0,
            **_split_defaults,
            "hsl_adjust": None,
        }

    params: dict[str, Any] = {}
    for key, default in default_params.items():
        raw = recommended.get(key, default)
        try:
            val = float(raw)
        except (TypeError, ValueError):
            val = default
        # 범위 클램핑
        if key in ("blemish_removal", "skin_smoothing"):
            val = max(0.0, min(1.0, val))
        else:
            val = max(-1.0, min(1.0, val))
        params[key] = round(val, 3)

    # 톤 커브 파싱
    tone_curve = recommended.get("toneCurve", {})
    if isinstance(tone_curve, dict):
        preset = tone_curve.get("preset", "linear")
        if preset not in TONE_CURVE_PRESETS:
            preset = "linear"
        params["tone_curve_preset"] = preset
        try:
            strength = float(tone_curve.get("strength", 0.0))
        except (TypeError, ValueError):
            strength = 0.0
        params["tone_curve_strength"] = round(max(0.0, min(1.0, strength)), 3)
    else:
        params["tone_curve_preset"] = "linear"
        params["tone_curve_strength"] = 0.0

    # 스플릿 토닝 파싱
    split_toning = recommended.get("splitToning", {})
    if isinstance(split_toning, dict):
        shadow = split_toning.get("shadow", {})
        highlight = split_toning.get("highlight", {})
        if isinstance(shadow, dict):
            try:
                params["split_shadow_hue"] = round(float(shadow.get("hue", 0.0)) % 360.0, 1)
            except (TypeError, ValueError):
                params["split_shadow_hue"] = 0.0
            try:
                params["split_shadow_strength"] = round(max(0.0, min(1.0, float(shadow.get("strength", 0.0)))), 3)
            except (TypeError, ValueError):
                params["split_shadow_strength"] = 0.0
        else:
            params["split_shadow_hue"] = 0.0
            params["split_shadow_strength"] = 0.0
        if isinstance(highlight, dict):
            try:
                params["split_highlight_hue"] = round(float(highlight.get("hue", 0.0)) % 360.0, 1)
            except (TypeError, ValueError):
                params["split_highlight_hue"] = 0.0
            try:
                params["split_highlight_strength"] = round(max(0.0, min(1.0, float(highlight.get("strength", 0.0)))), 3)
            except (TypeError, ValueError):
                params["split_highlight_strength"] = 0.0
        else:
            params["split_highlight_hue"] = 0.0
            params["split_highlight_strength"] = 0.0
    else:
        params.update(_split_defaults)

    # HSL 선택적 색상 파싱
    hsl_raw = recommended.get("hslAdjust")
    if isinstance(hsl_raw, dict):
        valid_channels = set(_HSL_CHANNELS.keys())
        hsl_parsed: dict[str, dict[str, float]] = {}
        for ch_name, ch_adj in hsl_raw.items():
            if ch_name not in valid_channels or not isinstance(ch_adj, dict):
                continue
            parsed_adj: dict[str, float] = {}
            for k in ("hue", "saturation", "lightness"):
                try:
                    v = float(ch_adj.get(k, 0.0))
                except (TypeError, ValueError):
                    v = 0.0
                parsed_adj[k] = round(max(-1.0, min(1.0, v)), 3)
            if any(abs(v) >= 0.01 for v in parsed_adj.values()):
                hsl_parsed[ch_name] = parsed_adj
        params["hsl_adjust"] = hsl_parsed if hsl_parsed else None
    else:
        params["hsl_adjust"] = None

    # 얼굴/체형 보정 파싱
    reshape = recommended.get("reshapeParams", {})
    if isinstance(reshape, dict):
        for rkey, rrange in [
            ("face_slim", (0.0, 1.0)),
            ("jaw_sharpen", (0.0, 1.0)),
            ("eye_enlarge", (0.0, 1.0)),
            ("leg_stretch", (0.0, 1.0)),
            ("waist_slim", (0.0, 1.0)),
        ]:
            try:
                rv = float(reshape.get(rkey, 0.0))
            except (TypeError, ValueError):
                rv = 0.0
            params[rkey] = round(max(rrange[0], min(rrange[1], rv)), 3)

        # shoulder_width: -1.0 ~ 1.0
        try:
            sw = float(reshape.get("shoulder_width", 0.0))
        except (TypeError, ValueError):
            sw = 0.0
        params["shoulder_width"] = round(max(-1.0, min(1.0, sw)), 3)
    else:
        params["face_slim"] = 0.0
        params["jaw_sharpen"] = 0.0
        params["eye_enlarge"] = 0.0
        params["leg_stretch"] = 0.0
        params["shoulder_width"] = 0.0
        params["waist_slim"] = 0.0

    return params


