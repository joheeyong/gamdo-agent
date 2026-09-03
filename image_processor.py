"""이미지 변형 엔진 — Pillow + OpenCV 기반 순수 함수 모듈."""

from __future__ import annotations

import base64
import hashlib
import io
import logging
import os
import tempfile
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
except ImportError:
    mp = None  # type: ignore[assignment]
    _MP_AVAILABLE = False
    log.warning("mediapipe not installed — blemish removal disabled")


# ── MediaPipe 모델 파일 해석 ──
#
# 경로를 import 시점에 한 번만 계산하면, 그 뒤 프로젝트 디렉터리가 옮겨지거나
# 가상환경이 재설치될 때 문자열만 남고 파일은 사라진 상태가 된다
# (MediaPipe는 create_from_options 시점에 경로로 파일을 다시 연다).
# 그래서 사용 시점마다 존재를 확인하고, 없으면 다시 받아 자가 복구한다.

_MODEL_SPECS: dict[str, tuple[str, str]] = {
    "face": (
        "face_landmarker.task",
        "https://storage.googleapis.com/mediapipe-models/"
        "face_landmarker/face_landmarker/float16/latest/face_landmarker.task",
    ),
    "pose": (
        "pose_landmarker_lite.task",
        "https://storage.googleapis.com/mediapipe-models/"
        "pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task",
    ),
    # 인물 분할 — 배경 흐림에서 사람과 배경을 가른다. 249KB, 1080p에서 3.4ms.
    "person": (
        "selfie_segmenter.tflite",
        "https://storage.googleapis.com/mediapipe-models/"
        "image_segmenter/selfie_segmenter/float16/latest/selfie_segmenter.tflite",
    ),
}

# 내려받은 모델을 둘 곳. site-packages는 uv sync 한 번에 날아가므로 쓰지 않는다.
# 이름 앞에 점을 붙여 같은 디렉터리의 models.py 모듈과 헷갈리지 않게 한다.
_MODEL_CACHE_DIR = os.path.abspath(
    os.environ.get(
        "GAMDO_MODEL_DIR",
        os.path.join(os.path.dirname(os.path.abspath(__file__)), ".model_cache"),
    )
)

_model_path_cache: dict[str, str] = {}


def _candidate_model_paths(filename: str) -> list[str]:
    """모델 파일을 찾을 후보 경로 (우선순위 순)."""
    paths = [os.path.join(_MODEL_CACHE_DIR, filename)]
    if mp is not None:
        # 패키지에 번들되어 있으면 그것도 사용한다
        paths.append(os.path.join(os.path.dirname(mp.__file__), "models", filename))
    return paths


def _download_model(url: str, dest: str) -> None:
    """모델을 임시 파일로 받은 뒤 원자적으로 교체한다.

    중간에 끊겨도 손상된 파일이 남지 않게 한다.
    """
    import urllib.request

    os.makedirs(os.path.dirname(dest), exist_ok=True)
    # 임시 이름은 고유해야 한다. 고정 이름(.part)이면 첫 요청 두 개가 동시에
    # 들어올 때 서로의 임시 파일을 지운다 — FastAPI는 동기 엔드포인트를
    # 스레드풀에서 병렬로 돌리므로 실제로 겹칠 수 있다.
    fd, tmp = tempfile.mkstemp(
        prefix=os.path.basename(dest) + ".", suffix=".part",
        dir=os.path.dirname(dest),
    )
    os.close(fd)
    try:
        urllib.request.urlretrieve(url, tmp)
        if os.path.getsize(tmp) < 1024:
            raise OSError(f"downloaded file too small ({os.path.getsize(tmp)} bytes)")
        os.replace(tmp, dest)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


def _resolve_model_path(kind: str) -> str | None:
    """MediaPipe 모델 경로를 반환한다 (없으면 내려받고, 실패 시 None)."""
    if not _MP_AVAILABLE or mp is None:
        return None

    cached = _model_path_cache.get(kind)
    if cached is not None and os.path.exists(cached):
        return cached
    if cached is not None:
        # 경로가 사라졌다 — 프로젝트 이동이나 venv 재설치. 다시 찾는다.
        log.warning("Model file vanished at %s — re-resolving", cached)
        _model_path_cache.pop(kind, None)

    filename, url = _MODEL_SPECS[kind]
    candidates = _candidate_model_paths(filename)

    for path in candidates:
        if os.path.exists(path):
            _model_path_cache[kind] = path
            return path

    dest = candidates[0]
    try:
        _download_model(url, dest)
    except Exception as exc:
        log.warning("Failed to download %s model: %s", kind, exc)
        return None

    log.info("Downloaded %s model to %s", kind, dest)
    _model_path_cache[kind] = dest
    return dest


def face_model_path() -> str | None:
    """FaceLandmarker 모델 경로 (사용 시점에 확인·복구)."""
    return _resolve_model_path("face")


def pose_model_path() -> str | None:
    """PoseLandmarker 모델 경로 (사용 시점에 확인·복구)."""
    return _resolve_model_path("pose")


def person_model_path() -> str | None:
    """인물 분할(ImageSegmenter) 모델 경로 (사용 시점에 확인·복구)."""
    return _resolve_model_path("person")


# ── 요청 스코프 MediaPipe 캐시 ──


class MediaPipeCache:
    """요청 단위로 MediaPipe 모델 인스턴스 + 랜드마크 결과를 캐시한다.

    사용법::

        with MediaPipeCache() as cache:
            face_results = cache.get_face_landmarks(arr_rgb)
            pose_results = cache.get_pose_landmarks(arr_rgb)

    - 모델 인스턴스는 첫 호출 시 생성하고 컨텍스트 종료까지 재사용
    - 동일 이미지(바이트 해시 기준)에 대한 감지 결과를 캐시하여 중복 호출 제거
    - 컨텍스트 종료 시 모든 모델 인스턴스를 close()하여 메모리 누수 방지
    """

    def __init__(self) -> None:
        self._face_landmarker: Any | None = None
        self._pose_landmarker: Any | None = None
        self._person_segmenter: Any | None = None
        self._face_results_cache: dict[str, Any] = {}
        self._pose_results_cache: dict[str, Any] = {}
        self._person_mask_cache: dict[str, Any] = {}

    def __enter__(self) -> "MediaPipeCache":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    def close(self) -> None:
        """모든 모델 인스턴스를 닫고 캐시를 비운다."""
        if self._face_landmarker is not None:
            try:
                self._face_landmarker.close()
            except Exception:
                pass
            self._face_landmarker = None
        if self._pose_landmarker is not None:
            try:
                self._pose_landmarker.close()
            except Exception:
                pass
            self._pose_landmarker = None
        self._close_person_segmenter()
        self._face_results_cache.clear()
        self._pose_results_cache.clear()
        self._person_mask_cache.clear()

    def _close_person_segmenter(self) -> None:
        """분할기 인스턴스를 닫고 슬롯을 비운다.

        잘못된 입력을 한 번 먹은 인스턴스는 다음 호출에서 영구히 멈춘다
        (내부 그래프가 에러 상태로 남는다). 그래서 예외가 나면 재사용하지 않고
        버리고 다시 만든다.
        """
        if self._person_segmenter is not None:
            try:
                self._person_segmenter.close()
            except Exception:
                pass
            self._person_segmenter = None

    @staticmethod
    def _image_key(arr_rgb: np.ndarray) -> str:
        """이미지 배열의 빠른 해시 키를 반환한다."""
        # 전체 데이터 해시 대신 shape + 샘플 바이트로 빠른 키 생성
        h, w = arr_rgb.shape[:2]
        # 균등 간격 샘플링 (최대 ~4KB)
        step_h = max(1, h // 32)
        step_w = max(1, w // 32)
        sample = arr_rgb[::step_h, ::step_w].tobytes()
        digest = hashlib.md5(sample, usedforsecurity=False).hexdigest()
        return f"{h}x{w}_{digest}"

    def _get_face_landmarker(self) -> Any:
        """FaceLandmarker 인스턴스를 반환 (없으면 생성)."""
        if self._face_landmarker is None:
            model_path = face_model_path()
            if model_path is None or mp is None:
                return None
            base_options = mp.tasks.BaseOptions(model_asset_path=model_path)
            options = mp.tasks.vision.FaceLandmarkerOptions(
                base_options=base_options,
                num_faces=5,
                min_face_detection_confidence=0.5,
                min_face_presence_confidence=0.5,
            )
            self._face_landmarker = mp.tasks.vision.FaceLandmarker.create_from_options(options)
        return self._face_landmarker

    def _get_pose_landmarker(self) -> Any:
        """PoseLandmarker 인스턴스를 반환 (없으면 생성)."""
        if self._pose_landmarker is None:
            model_path = pose_model_path()
            if model_path is None or mp is None:
                return None
            base_options = mp.tasks.BaseOptions(model_asset_path=model_path)
            options = mp.tasks.vision.PoseLandmarkerOptions(
                base_options=base_options,
                num_poses=3,
                min_pose_detection_confidence=0.5,
                min_pose_presence_confidence=0.5,
            )
            self._pose_landmarker = mp.tasks.vision.PoseLandmarker.create_from_options(options)
        return self._pose_landmarker

    def _get_person_segmenter(self) -> Any:
        """ImageSegmenter 인스턴스를 반환 (없으면 생성)."""
        if self._person_segmenter is None:
            model_path = person_model_path()
            if model_path is None or mp is None:
                return None
            base_options = mp.tasks.BaseOptions(model_asset_path=model_path)
            options = mp.tasks.vision.ImageSegmenterOptions(
                base_options=base_options,
                # 0/255 이진 마스크(category_mask)보다 확률 마스크가 4배 빠르고,
                # 경계가 부드러워 블렌딩에 그대로 쓸 수 있다.
                output_confidence_masks=True,
                output_category_mask=False,
            )
            self._person_segmenter = mp.tasks.vision.ImageSegmenter.create_from_options(
                options
            )
        return self._person_segmenter

    def get_person_mask(self, arr_rgb: np.ndarray) -> np.ndarray | None:
        """인물 확률 마스크를 돌려준다 — float32 (h, w), 0.0~1.0.

        모델이 없거나 분할에 실패하면 None. 입력 해상도로 이미 업샘플되어
        나오므로 크기를 맞출 필요가 없다.
        """
        key = self._image_key(arr_rgb)
        if key in self._person_mask_cache:
            return self._person_mask_cache[key]

        segmenter = self._get_person_segmenter()
        if segmenter is None:
            self._person_mask_cache[key] = None
            return None

        try:
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=arr_rgb)
            result = segmenter.segment(mp_image)
            masks = getattr(result, "confidence_masks", None)
            if not masks:
                self._person_mask_cache[key] = None
                return None
            # numpy_view()는 owndata=False인 읽기 전용 뷰라 반드시 복사해 둔다
            mask = np.array(masks[0].numpy_view(), dtype=np.float32).squeeze()
        except Exception as exc:
            # 한 번 실패한 인스턴스는 다음 호출에서 멈춘다 — 버리고 다시 만든다
            log.warning("person segmentation failed: %s", exc)
            self._close_person_segmenter()
            self._person_mask_cache[key] = None
            return None

        self._person_mask_cache[key] = mask
        return mask

    def get_face_landmarks(self, arr_rgb: np.ndarray) -> Any:
        """얼굴 랜드마크 감지 결과를 캐시에서 반환하거나 새로 감지한다.

        반환: FaceLandmarkerResult (face_landmarks 속성 포함), 또는 감지 실패 시 None.
        """
        key = self._image_key(arr_rgb)
        if key in self._face_results_cache:
            return self._face_results_cache[key]

        landmarker = self._get_face_landmarker()
        if landmarker is None:
            self._face_results_cache[key] = None
            return None

        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=arr_rgb)
        results = landmarker.detect(mp_image)

        if not results.face_landmarks:
            self._face_results_cache[key] = None
            return None

        self._face_results_cache[key] = results
        return results

    def get_pose_landmarks(self, arr_rgb: np.ndarray) -> Any:
        """포즈 랜드마크 감지 결과를 캐시에서 반환하거나 새로 감지한다.

        반환: PoseLandmarkerResult (pose_landmarks 속성 포함), 또는 감지 실패 시 None.
        """
        key = self._image_key(arr_rgb)
        if key in self._pose_results_cache:
            return self._pose_results_cache[key]

        landmarker = self._get_pose_landmarker()
        if landmarker is None:
            self._pose_results_cache[key] = None
            return None

        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=arr_rgb)
        results = landmarker.detect(mp_image)

        if not results.pose_landmarks:
            self._pose_results_cache[key] = None
            return None

        self._pose_results_cache[key] = results
        return results


# ── Base64 ↔ PIL Image 변환 ──


def decode_base64_image(b64: str) -> Image.Image:
    """Base64 문자열을 PIL Image로 디코딩."""
    data = base64.b64decode(b64)
    return Image.open(io.BytesIO(data)).convert("RGB")


def encode_image_base64(img: Image.Image, fmt: str = "JPEG", quality: int = 92) -> str:
    """PIL Image를 base64 문자열로 인코딩.

    이 결과가 곧 사용자가 저장하는 사진이다. 90 → 92는 용량 대비 이득이
    크지 않지만(실측 누적 오차 3.94 → 3.68), 업로드에서 이미 한 번 JPEG를
    거친 뒤라 마지막 단계는 조금 여유를 둔다.
    화질 손실의 주범은 압축이 아니라 해상도였다 (12MP → 1.8MP).
    """
    buf = io.BytesIO()
    img.save(buf, format=fmt, quality=quality, optimize=True)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


# ── 촬영 결함 교정 (색 보정 이전 단계) ──


def estimate_noise_sigma(img: Image.Image) -> float:
    """이미지의 노이즈 표준편차를 추정한다 (0~255 스케일).

    Immerkær(1996)의 라플라시안 기반 추정 — 평탄한 영역의 고주파 성분만
    남기는 3x3 커널로 합성곱한 뒤 평균 절대값을 취한다. 사진 내용(엣지)에
    거의 영향을 받지 않아 별도 마스킹 없이 쓸 수 있다.
    """
    gray = np.asarray(img.convert("L"), dtype=np.float32)
    h, w = gray.shape
    if h < 8 or w < 8:
        return 0.0

    kernel = np.array([[1, -2, 1], [-2, 4, -2], [1, -2, 1]], dtype=np.float32)
    conv = cv2.filter2D(gray, cv2.CV_32F, kernel)
    sigma = float(np.abs(conv).mean()) * np.sqrt(np.pi / 2.0) / 6.0
    return round(sigma, 3)


def apply_denoise(img: Image.Image, strength: float) -> Image.Image:
    """노이즈를 줄인다. strength 0.0~1.0.

    휘도(Y)는 약하게, 색차(CrCb)는 강하게 지운다. 색 얼룩이 먼저 눈에 띄고,
    색차는 세게 뭉개도 디테일 손실이 거의 보이지 않기 때문이다.
    쉐도우 리프팅 전에 적용해야 어두운 곳 노이즈가 증폭되지 않는다.
    """
    if strength < 0.01:
        return img

    strength = max(0.0, min(1.0, strength))
    arr = np.asarray(img.convert("RGB"), dtype=np.uint8)

    try:
        ycrcb = cv2.cvtColor(arr, cv2.COLOR_RGB2YCrCb)
        y, cr, cb = cv2.split(ycrcb)

        # 휘도: NLM의 h는 노이즈 표준편차와 같은 눈금이라, 실제 측정치에
        # 비례해 잡아야 한다. 고정값을 쓰면 노이즈가 큰 사진에서 아무 효과가
        # 없고(h가 너무 작음) 깨끗한 사진에서는 디테일만 뭉갠다.
        sigma = estimate_noise_sigma(img)
        h_luma = float(np.clip(sigma * (0.5 + 1.0 * strength), 1.0, 15.0))
        y = cv2.fastNlMeansDenoising(y, None, h=h_luma, templateWindowSize=7,
                                     searchWindowSize=15)

        # 색차: 절반 해상도에서 강하게 뭉갠 뒤 되돌린다 (빠르고 티가 안 난다)
        ch, cw = cr.shape
        small = (max(1, cw // 2), max(1, ch // 2))
        blur_px = int(3 + 6 * strength) | 1
        cr = cv2.resize(cv2.medianBlur(cv2.resize(cr, small, interpolation=cv2.INTER_AREA), blur_px),
                        (cw, ch), interpolation=cv2.INTER_LINEAR)
        cb = cv2.resize(cv2.medianBlur(cv2.resize(cb, small, interpolation=cv2.INTER_AREA), blur_px),
                        (cw, ch), interpolation=cv2.INTER_LINEAR)

        result = cv2.cvtColor(cv2.merge([y, cr, cb]), cv2.COLOR_YCrCb2RGB)
        log.info("denoise: sigma=%.1f strength=%.2f (luma h=%.1f, chroma blur=%dpx)",
                 sigma, strength, h_luma, blur_px)
        return Image.fromarray(result)
    except Exception as exc:
        log.warning("denoise failed: %s", exc)
        return img


def estimate_illuminant(img: Image.Image) -> tuple[float, float, float]:
    """장면의 조명 색을 추정해 중립으로 만드는 RGB 게인을 반환한다.

    Shades-of-Gray (Minkowski p=6) — 순수 Gray World는 한 색이 넓게 깔린
    사진(잔디밭, 파란 하늘)에서 그 색을 회색으로 만들어 버리는데,
    p-노름을 쓰면 밝은 픽셀에 가중이 실려 그 실패가 완화된다.
    """
    small = img.convert("RGB")
    w, h = small.size
    if max(w, h) > 256:
        ratio = 256 / max(w, h)
        small = small.resize((max(1, int(w * ratio)), max(1, int(h * ratio))), Image.BILINEAR)

    arr = np.asarray(small, dtype=np.float32) / 255.0
    p = 6.0
    norms = np.array([
        (np.power(arr[..., c], p).mean()) ** (1.0 / p) for c in range(3)
    ])
    norms[norms < 1e-6] = 1e-6

    gains = norms.mean() / norms
    return float(gains[0]), float(gains[1]), float(gains[2])


def apply_auto_white_balance(img: Image.Image, strength: float) -> Image.Image:
    """색이 틀어진 사진을 중립 쪽으로 당긴다. strength 0.0~1.0.

    전부 보정하지 않는다 — 노을이나 골든아워의 따뜻함까지 지워 버리기
    때문이다. strength로 부분 보정하고, 채널당 게인도 ±25%로 묶는다.
    프로필이 원하는 색온도는 이 위에 temperature로 다시 얹힌다.
    """
    if strength < 0.01:
        return img

    strength = max(0.0, min(1.0, strength))
    gr, gg, gb = estimate_illuminant(img)

    # 부분 적용 + 채널당 상한
    gains = []
    for g in (gr, gg, gb):
        g = 1.0 + (g - 1.0) * strength
        gains.append(max(0.75, min(1.25, g)))

    if all(abs(g - 1.0) < 0.01 for g in gains):
        return img

    arr = np.asarray(img.convert("RGB"), dtype=np.float32)
    for c, g in enumerate(gains):
        arr[..., c] *= g

    log.info("auto_wb: strength=%.2f gains=(%.3f, %.3f, %.3f)", strength, *gains)
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))


def estimate_keystone(img: Image.Image, max_correction: float = 0.35) -> float:
    """수직 원근 왜곡(키스톤)의 세기를 추정한다. -1~1, 확신 없으면 0.

    건물을 아래에서 올려다보면 위쪽이 좁아진다. 화면 좌우의 "수직에 가까운"
    선들이 위로 갈수록 서로 모이는지를 보고 그 정도를 잰다.
    양수면 위가 좁다(올려다봄), 음수면 아래가 좁다(내려다봄).
    """
    gray = np.asarray(img.convert("L"), dtype=np.uint8)
    h, w = gray.shape
    if max(h, w) > 900:
        scale = 900 / max(h, w)
        gray = cv2.resize(gray, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
        h, w = gray.shape

    edges = cv2.Canny(gray, 60, 180, apertureSize=3)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 720, threshold=60,
                            minLineLength=int(h * 0.30),
                            maxLineGap=int(h * 0.02) + 2)
    if lines is None:
        return 0.0

    left_tilt: list[tuple[float, float]] = []   # (기울기, 길이)
    right_tilt: list[tuple[float, float]] = []

    for x1, y1, x2, y2 in np.asarray(lines).reshape(-1, 4):
        dx, dy = float(x2 - x1), float(y2 - y1)
        length = float(np.hypot(dx, dy))
        if abs(dy) < 1e-3:
            continue
        angle = abs(np.degrees(np.arctan2(dy, dx)))
        # 수직에서 1.2~25도 벗어난 선만 — 완전한 수직은 왜곡 정보가 없고,
        # 많이 기운 선은 지붕·계단 같은 진짜 사선이다.
        if not (1.2 <= abs(angle - 90.0) <= 25.0):
            continue
        # 위로 갈수록 안쪽으로 기우는 정도 (x가 y에 대해 변하는 비율)
        slope = dx / dy
        cx = (x1 + x2) / 2.0
        (left_tilt if cx < w / 2 else right_tilt).append((slope, length))

    if len(left_tilt) < 2 or len(right_tilt) < 2:
        return 0.0

    def weighted_mean(items: list[tuple[float, float]]) -> float:
        vals = np.array([v for v, _ in items])
        wts = np.array([wt for _, wt in items])
        return float((vals * wts).sum() / wts.sum())

    left_slope = weighted_mean(left_tilt)
    right_slope = weighted_mean(right_tilt)

    # 위가 좁으면 왼쪽 선은 오른쪽으로, 오른쪽 선은 왼쪽으로 기운다.
    convergence = (right_slope - left_slope) / 2.0

    # 기울기(dx/dy)를 [apply_keystone]이 쓰는 단위로 바꾼다.
    # 한 변이 전체 높이에 걸쳐 convergence*h 만큼 안으로 들어오므로,
    # 좁아진 쪽을 그만큼 넓히려면 폭 대비 2*convergence*h/w 가 필요하다.
    amount = 2.0 * convergence * h / w
    if abs(amount) < 0.02:
        return 0.0

    return round(float(np.clip(amount, -max_correction, max_correction)), 3)


def _translation(dx: float, dy: float) -> np.ndarray:
    """평행이동 3x3 행렬. 크롭을 좌표 변환으로 표현할 때 쓴다."""
    return np.array(
        [[1.0, 0.0, float(dx)], [0.0, 1.0, float(dy)], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )


def _compose_geometry(
    base: np.ndarray | None, added: np.ndarray | None
) -> np.ndarray | None:
    """기하 변환을 순서대로 합성한다. None은 "손대지 않음"이다."""
    if added is None:
        return base
    return added if base is None else added @ base


def apply_keystone(img: Image.Image, amount: float) -> Image.Image:
    """수직 원근을 편다. amount는 [estimate_keystone]의 반환값."""
    return _keystone_with_matrix(img, amount)[0]


def _keystone_with_matrix(
    img: Image.Image, amount: float
) -> tuple[Image.Image, np.ndarray | None]:
    """[apply_keystone]과 같지만 원본→결과 좌표 변환 행렬도 돌려준다.

    위쪽(또는 아래쪽) 변을 늘려 좌우 수직선을 평행하게 만든 뒤,
    회전 보정과 같이 검은 여백 없이 내접 영역만 남긴다.

    행렬이 필요한 이유: 모델이 준 좌표(제거 영역·크롭)는 원본 프레임 기준인데
    이 보정이 프레임을 바꿔 놓는다. 행렬로 좌표를 같이 옮겨야 짚은 곳에 맞는다.
    손대지 않았으면 None을 돌려준다.
    """
    if abs(amount) < 0.02:
        return img, None

    amount = float(np.clip(amount, -0.35, 0.35))
    arr = np.asarray(img.convert("RGB"))
    h, w = arr.shape[:2]

    # 좁아진 쪽 변을 그만큼 넓힌다
    shift = abs(amount) * w * 0.5
    if amount > 0:      # 위가 좁다 → 위를 넓힌다
        src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
        dst = np.float32([[-shift, 0], [w + shift, 0], [w, h], [0, h]])
    else:               # 아래가 좁다 → 아래를 넓힌다
        src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
        dst = np.float32([[0, 0], [w, 0], [w + shift, h], [-shift, h]])

    matrix = cv2.getPerspectiveTransform(src, dst)
    out_w = int(w + 2 * shift)
    matrix[0, 2] += shift
    warped = cv2.warpPerspective(
        arr, matrix, (out_w, h),
        flags=cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_REPLICATE,
    )

    # 넓힌 만큼 가장자리는 늘어난 화소라 잘라낸다
    crop = int(shift)
    if out_w - 2 * crop < 50:
        return img, None
    result = warped[:, crop:out_w - crop]

    log.info("keystone: amount=%.3f, %dx%d → %dx%d", amount, w, h,
             result.shape[1], result.shape[0])
    return Image.fromarray(result), _translation(-crop, 0) @ matrix


# 인물로 인정할 최소 면적 (확률 0.5 넘는 픽셀의 비율).
# 분할 모델은 사람이 없는 사진에도 빈 마스크가 아니라 0에 가까운 확률장을 낸다.
# 실측: 인물 사진 0.43~0.55, 사람 없는 사진 0.00008 — 최댓값으로 판단하면
# (사람 없는 사진에서도 0.58까지 튄다) 오탐이 나므로 면적으로 판단한다.
# 배경 흐림을 걸 만한 최소 인물 비중(화면 면적 대비).
# 0.5%였을 때는 풍경 속 작은 사람 하나로 사진 전체가 흐려졌다.
# 가짜 보케는 인물이 주인공일 때만 자연스럽다.
_BLUR_MIN_SUBJECT = 0.08
# 이 비중 이상이면 온전한 세기로 건다. 사이 구간은 선형으로 올린다.
_BLUR_FULL_SUBJECT = 0.18

# strength 1.0에서의 흐림 반경(sigma)을 짧은 변 대비로. 실측 sigma:
#   0.19(기본) → 2.5,  0.5 → 6.5,  1.0 → 13.0
# 예전에는 이 계수를 sigma가 아니라 커널 크기에 곱했다. OpenCV가 커널에서
# 유도하는 sigma는 그 6분의 1이라, 기본 세기에서 sigma가 0.8 — 눈에 보이지
# 않았다. 인물 사진마다 켜지는 기능이 사실상 아무 일도 하지 않았다.
_BLUR_SIGMA_RATIO = 0.012


def _blur_background(
    arr: np.ndarray, alpha: np.ndarray, strength: float
) -> np.ndarray:
    """alpha(1=인물)를 써서 배경만 흐린 배열을 돌려준다."""
    h, w = arr.shape[:2]
    sigma = max(1.0, min(h, w) * _BLUR_SIGMA_RATIO * strength)

    # sigma가 크면 원본 해상도의 가우시안이 비싸다. 흐림은 저주파라
    # 축소해서 흐리고 되돌려도 결과가 같다 (실측 14ms → 8ms).
    shrink = min(1.0, 8.0 / sigma)
    if shrink < 1.0:
        small = cv2.resize(arr, None, fx=shrink, fy=shrink,
                           interpolation=cv2.INTER_AREA)
        small = cv2.GaussianBlur(small, (0, 0), sigma * shrink)
        blurred = cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)
    else:
        blurred = cv2.GaussianBlur(arr, (0, 0), sigma)

    a = alpha[:, :, np.newaxis]
    out = arr.astype(np.float32) * a + blurred.astype(np.float32) * (1.0 - a)
    return np.clip(out, 0, 255).astype(np.uint8)


def background_blur_scale(coverage: float) -> float:
    """인물 비중에 따른 배경 흐림 세기 배율 (0~1).

    실제 아웃포커스는 피사체가 화면을 채울 때 생긴다. 대략적인 구도별 비중:
      얼굴 클로즈업 30~60% / 상반신 20~35% / 전신 10~20% / 풍경 속 사람 1~5%

    풍경 속 사람에게 걸면 사진의 거의 전부가 흐려진다. 그래서 비중이
    낮으면 아예 걸지 않고, 사이 구간은 갑자기 튀지 않게 선형으로 올린다.
    """
    if coverage < _BLUR_MIN_SUBJECT:
        return 0.0
    if coverage >= _BLUR_FULL_SUBJECT:
        return 1.0
    span = _BLUR_FULL_SUBJECT - _BLUR_MIN_SUBJECT
    return float((coverage - _BLUR_MIN_SUBJECT) / span)


def apply_background_blur(
    img: Image.Image,
    strength: float,
    cache: "MediaPipeCache | None" = None,
) -> Image.Image:
    """인물 뒤 배경을 흐린다. strength 0.0~1.0. 사람이 없으면 그대로 둔다.

    MediaPipe 인물 분할(selfie_segmenter)로 사람 모양 그대로의 확률 마스크를
    받아 그 바깥을 흐린다.

    예전에는 얼굴 랜드마크에서 "얼굴 타원 + 그 아래 몸통 타원"을 그려 인물로
    삼았다. 팔을 들거나 앉은 자세, 전신샷, 옆으로 선 구도에서는 팔다리가
    배경으로 판정돼 흐려지고, 반대로 어깨 옆 배경은 타원 안이라 선명하게
    남았다. 사람 모양은 타원이 아니다.
    """
    if strength < 0.01:
        return img

    arr = np.asarray(img.convert("RGB"), dtype=np.uint8)

    ctx = cache if cache is not None else MediaPipeCache()
    try:
        mask = ctx.get_person_mask(arr)
    finally:
        if cache is None:
            ctx.close()

    if mask is None:
        # 모델을 받지 못했거나 분할이 실패했다. 예전의 타원 근사로 되돌리지
        # 않는다 — 팔다리를 흐리는 잘못된 마스크보다 아무것도 안 하는 게 낫다.
        log.info("background_blur: 인물 분할을 쓸 수 없어 건너뜀")
        return img

    coverage = float((mask > 0.5).mean())
    scale = background_blur_scale(coverage)
    if scale <= 0.0:
        log.info(
            "background_blur: 인물 면적 %.1f%% — 근접샷이 아니라 건너뜀 "
            "(최소 %.0f%%)", coverage * 100, _BLUR_MIN_SUBJECT * 100,
        )
        return img
    strength *= scale

    # 확률 마스크는 이미 경계가 부드럽다(내부 해상도 256px에서 업샘플된다).
    # 애매한 영역의 잔점만 가볍게 눌러 준다.
    smooth = max(1.0, min(arr.shape[:2]) * 0.003)
    alpha = cv2.GaussianBlur(mask, (0, 0), smooth)

    out = _blur_background(arr, alpha, strength)
    log.info("background_blur: strength=%.2f(x%.2f) 인물 %.1f%% sigma=%.1f",
             strength, scale, coverage * 100,
             max(1.0, min(arr.shape[:2]) * _BLUR_SIGMA_RATIO * strength))
    return Image.fromarray(out)


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


def adjust_contrast(
    img: Image.Image,
    factor: float,
    pivot: float | None = None,
) -> Image.Image:
    """대비 조절. factor: -1.0 ~ +1.0 (0 = 원본).

    LAB 색공간의 L(밝기) 채널에서만 대비를 조절하여,
    채도와 색상 정보를 보존한다. 기존 RGB 전체 대비 감소는
    채도까지 떨어뜨려 하늘 같은 채색 영역을 회색으로 만들었다.

    pivot: 대비를 벌리는 기준 밝기(L, 0~255). None이면 이미지 전체 평균.
    영역별 보정에서는 반드시 그 영역 안의 평균을 넘겨야 한다. 전체 평균을
    쓰면 영역 밖의 밝기가 기준을 끌고 가, 평탄한 영역에 대비를 걸었을 뿐인데
    영역 전체가 밝아지거나 어두워진다.
    """
    if abs(factor) < 0.01:
        return img

    arr = np.array(img, dtype=np.uint8)
    arr_bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    lab = cv2.cvtColor(arr_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)

    l_ch = lab[:, :, 0]
    mean_l = float(np.mean(l_ch)) if pivot is None else float(pivot)

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
    ksize = max(9, int(min(h, w) * 0.015)) | 1  # 해상도 적응형 커널
    l_blur = cv2.GaussianBlur(l_ch, (ksize, ksize), 0)

    # 하이패스 디테일 = 원본 - 로컬 평균
    detail = l_ch - l_blur

    # 중간톤 마스크: 모든 톤에 최소 30% 적용, 중간톤에 100% 적용
    midtone_mask = 0.3 + 0.7 * (1.0 - np.abs(l_ch - 128.0) / 128.0)

    # factor 비례로 디테일 증폭 (양수: 로컬 대비 강화, 음수: 소프트)
    l_ch = l_ch + detail * factor * 0.8 * midtone_mask
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

    # 밝은 영역 마스크 — 시그모이드 기반 부드러운 전환
    # 중심점 160: 진짜 하이라이트 영역에 집중, 폭 40: 부드러운 그라데이션
    normalized = (l_ch - 160.0) / 40.0
    mask = 1.0 / (1.0 + np.exp(-normalized))

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

    # 어두운 영역 마스크 — 시그모이드 기반 부드러운 전환
    # 중심점 96: 진짜 쉐도우 영역에 집중, 폭 40: 부드러운 그라데이션
    normalized = (96.0 - l_ch) / 40.0
    mask = 1.0 / (1.0 + np.exp(-normalized))

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


# 그레인·샤픈의 기준 해상도.
#
# 앱은 미리보기를 800px, 저장을 2560px로 렌더한다. 효과의 크기를 화소 단위로
# 고정하면 2560px에서 만든 것은 화면에 맞게 줄이는 순간 평균되어 사라진다 —
# 미리보기에서 고른 그레인·선명도가 저장본에는 없다.
# 실측: 같은 값에서 저장본의 그레인이 미리보기의 0.28~0.31배, 샤픈이 0.15~0.26배.
_EFFECT_REFERENCE_PX = 800


def apply_grain(img: Image.Image, intensity: float) -> Image.Image:
    """필름 그레인 효과. intensity: 0.0(없음) ~ 1.0(강한 노이즈).

    밝기 채널에만 모노크롬 노이즈를 추가하여 자연스러운 필름 느낌을 만든다.
    알갱이 크기는 해상도에 비례해 미리보기와 저장본의 체감을 맞춘다.
    """
    if intensity < 0.01:
        return img

    arr = np.array(img, dtype=np.float32)
    h, w = arr.shape[:2]
    sigma_target = intensity * 40.0   # 최대 40 밝기값 편차

    # 기준 해상도에서 1화소짜리 노이즈를 만들고 원본 크기로 늘린다.
    scale = max(1.0, min(h, w) / _EFFECT_REFERENCE_PX)
    nh, nw = max(1, int(round(h / scale))), max(1, int(round(w / scale)))

    # 씨앗을 사진에서 끌어온다. 무작위로 두면 같은 사진을 두 번 렌더할 때마다
    # 그레인이 달라져 미리보기와 저장본이 절대 일치하지 않는다.
    digest = hashlib.md5(
        f"{h}x{w}".encode()
        + arr[::max(1, h // 16), ::max(1, w // 16)].astype(np.uint8).tobytes(),
        usedforsecurity=False,
    ).hexdigest()[:8]
    noise = np.random.default_rng(int(digest, 16)).normal(
        0.0, sigma_target, (nh, nw)
    ).astype(np.float32)

    if (nh, nw) != (h, w):
        noise = cv2.resize(noise, (w, h), interpolation=cv2.INTER_LINEAR)
        # 확대하면 이웃이 섞여 진폭이 줄어든다. 목표 편차로 되돌린다.
        actual = float(noise.std())
        if actual > 1e-6:
            noise *= sigma_target / actual

    noise = noise[:, :, np.newaxis]

    # 밝은 영역보다 중간톤에 그레인이 더 잘 보이도록 가중치
    gray = np.mean(arr, axis=2, keepdims=True) / 255.0
    weight = 1.0 - np.abs(gray - 0.5) * 1.2  # 중간톤에서 최대
    weight = np.clip(weight, 0.3, 1.0)

    adjusted = arr + noise * weight
    adjusted = np.clip(adjusted, 0, 255)

    return Image.fromarray(adjusted.astype(np.uint8))


def apply_skin_smoothing(
    img: Image.Image,
    intensity: float,
    cache: MediaPipeCache | None = None,
) -> Image.Image:
    """피부 보정 (bilateral filter). intensity: 0.0 ~ 1.0.

    피부 마스크 안에서만 섞는다 — 예전에는 이미지 전체에 블렌딩해서
    머리카락·눈동자·배경 디테일까지 같이 뭉개졌다.
    얼굴 미감지 시에는 뭉갤 피부가 없으므로 원본을 그대로 반환한다.

    cache가 제공되면 MediaPipe 모델/결과 캐시를 재사용한다.
    """
    if intensity < 0.01:
        return img

    arr = np.array(img)
    arr_bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)

    skin_mask = _get_skin_mask(arr, cache=cache)
    if skin_mask is None or cv2.countNonZero(skin_mask) == 0:
        return img

    # 커널 크기는 얼굴 크기에 맞춘다. 픽셀 고정값을 쓰면 클로즈업에서는
    # 효과가 거의 없고 얼굴이 작게 찍힌 사진에서는 뭉개진다 — 같은 설정인데
    # 결과가 달라 보이는 원인이었다.
    xs, ys = np.where(skin_mask > 0)[1], np.where(skin_mask > 0)[0]
    face_w = max(1, int(xs.max() - xs.min()))
    face_h = max(1, int(ys.max() - ys.min()))
    face_size = max(face_w, face_h)

    d = int(np.clip(face_size * 0.020 * (0.6 + intensity), 3, 15))
    sigma_color = 18 + intensity * 26     # 색상 병합 범위 — 얼굴 크기와 무관
    sigma_space = float(np.clip(face_size * 0.035, 8, 60))

    smoothed = cv2.bilateralFilter(arr_bgr, d, sigma_color, sigma_space)

    # 피부 안에서만, 마스크 경계는 부드럽게 — 얼굴 윤곽에 선이 생기지 않게.
    # 페더 폭도 얼굴 기준이라야 경계가 늘 비슷하게 자연스럽다.
    feather = max(5, int(face_size * 0.03)) | 1
    alpha = cv2.GaussianBlur(skin_mask, (feather, feather), 0)
    alpha = (alpha.astype(np.float32) / 255.0 * intensity)[:, :, np.newaxis]

    blended = arr_bgr.astype(np.float32) * (1.0 - alpha) + smoothed.astype(np.float32) * alpha
    result_rgb = cv2.cvtColor(np.clip(blended, 0, 255).astype(np.uint8), cv2.COLOR_BGR2RGB)

    return Image.fromarray(result_rgb)


def apply_sharpness(img: Image.Image, factor: float) -> Image.Image:
    """선명도 조절. factor: -1.0 ~ +1.0 (0 = 원본).

    언샤프 마스크. 예전에는 PIL의 ImageEnhance.Sharpness를 썼는데 고정 3x3
    커널이라 반경이 항상 1화소였다. 2560px 저장본에서는 800px 미리보기 때의
    3분의 1 크기 디테일만 건드려 효과가 거의 사라졌다 (실측 0.15~0.26배).
    반경을 해상도에 비례시켜 둘의 체감을 맞춘다.
    """
    if abs(factor) < 0.01:
        return img

    arr = np.array(img, dtype=np.float32)
    h, w = arr.shape[:2]
    sigma = max(0.6, 0.8 * min(h, w) / _EFFECT_REFERENCE_PX)
    blurred = cv2.GaussianBlur(arr, (0, 0), sigma)

    if factor > 0:
        out = arr + (arr - blurred) * factor
    else:
        # 음수는 흐리게 — 디테일을 빼는 대신 흐린 쪽으로 섞는다.
        # 그냥 뺐다가는 -1.0에서 디테일이 반전돼 윤곽이 이중으로 보인다.
        out = arr * (1.0 + factor) - blurred * factor
    return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8))


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


def _get_skin_mask(
    img_rgb: np.ndarray,
    cache: MediaPipeCache | None = None,
    for_blemish: bool = False,
    mode: str = "texture",
) -> np.ndarray | None:
    """MediaPipe FaceLandmarker로 피부 영역 마스크를 생성한다.

    반환: 얼굴 경계에서 충분히 안쪽으로 침식된 피부 마스크, 얼굴 미감지 시 None.
    다중 얼굴이면 모든 얼굴의 마스크를 합친다.

    cache가 제공되면 모델 인스턴스와 감지 결과를 캐시에서 재사용한다.
    """
    model_path = face_model_path()
    if model_path is None:
        return None

    h, w = img_rgb.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)

    if cache is not None:
        results = cache.get_face_landmarks(img_rgb)
        if results is None:
            return None
    else:
        base_options = mp.tasks.BaseOptions(model_asset_path=model_path)
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

        mask = cv2.bitwise_or(
            mask,
            build_face_skin_mask(_idx_to_pt, h, w, for_blemish=for_blemish, mode=mode),
        )

    return mask


def build_face_skin_mask(
    pt,
    h: int,
    w: int,
    for_blemish: bool = False,
    mode: str = "texture",
) -> np.ndarray:
    """랜드마크 좌표에서 얼굴 마스크 하나를 만든다. MediaPipe와 분리해 테스트 가능.

    [pt]는 랜드마크 인덱스를 (x, y) 픽셀 좌표로 바꾸는 함수다.

    mode에 따라 두 가지 마스크를 만든다:
    - "texture" — 질감을 건드리는 보정(피부 스무딩, 잡티 제거)용.
      눈·눈썹·입술을 뚫어 둔다. 그것까지 뭉개면 안 되니까.
    - "tone" — 밝기·색온도처럼 톤을 바꾸는 보정용. 얼굴 전체를 덮는다.
      볼만 밝히고 눈두덩과 입술은 그대로 두면 얼굴이 얼룩덜룩해진다.

    바깥 윤곽 축소와 이목구비 구멍 확장은 서로 다른 값이어야 한다.
    예전에는 침식 한 번으로 둘을 함께 처리한 데다 그 크기가 이미지 기준
    (짧은 변의 2.5%)이라, 얼굴이 작게 찍힌 사진에서는 남는 영역이
    얼굴 한가운데 조각들뿐이었다. 이제 둘 다 얼굴 크기에 비례한다.
    """
    oval_pts = np.array([pt(i) for i in _SKIN_FACE_OVAL], dtype=np.int32)
    face_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(face_mask, [oval_pts], 255)

    # 얼굴 너비 — 광대 양끝(234, 454) 사이 거리
    lx, _ = pt(234)
    rx, _ = pt(454)
    face_w = max(1.0, abs(rx - lx))

    if mode == "tone":
        # 톤 보정은 얼굴 전체에 고르게 — 윤곽만 아주 살짝 줄여
        # 머리카락·배경이 물리지 않게 한다. 블렌딩 시 경계는 어차피 부드러워진다.
        oval_shrink = max(1, int(face_w * 0.015))
        return cv2.erode(
            face_mask,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (oval_shrink * 2 + 1,) * 2),
        )

    # 이목구비 구멍: 볼록 껍질로 채운다.
    # fillConvexPoly는 입술처럼 오목한 윤곽에서 결과가 어긋난다.
    features = np.zeros((h, w), dtype=np.uint8)
    regions = [_LEFT_EYE, _RIGHT_EYE, _LIPS, _LEFT_EYEBROW, _RIGHT_EYEBROW]
    if for_blemish:
        # 잡티 탐지에서만 콧구멍 주변을 뺀다 — 자연스러운 음영이 잡티로 오감지된다.
        # 피부 보정에서는 코도 피부이므로 남긴다.
        regions.append([1, 2, 98, 327])
    for region in regions:
        pts = cv2.convexHull(np.array([pt(i) for i in region], dtype=np.int32))
        cv2.fillPoly(features, [pts], 255)

    feature_pad = max(2, int(face_w * 0.02))
    features = cv2.dilate(
        features,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (feature_pad * 2 + 1,) * 2),
    )

    oval_shrink = max(2, int(face_w * 0.03))
    face_mask = cv2.erode(
        face_mask,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (oval_shrink * 2 + 1,) * 2),
    )

    return cv2.bitwise_and(face_mask, cv2.bitwise_not(features))


def _detect_blemishes(
    img_bgr: np.ndarray,
    skin_mask: np.ndarray,
    intensity: float,
) -> np.ndarray:
    """LAB A·B 채널 밴드패스로 잡티를 탐지한다.

    밝기(L) 편차는 음영·조명이므로 무시하고,
    A·B 채널(빨강-녹색, 노랑-파랑)만으로 색상 이상 점을 탐지한다.

    잡티 크기의 성분만 남기려면 고주파(피부 노이즈)와 저주파(조명 그라데이션)를
    함께 걷어내야 한다. 예전에는 잡티와 비슷한 크기(짧은 변 2%)의 로컬 평균만
    빼서, 평균이 잡티를 같이 포함해 편차가 스스로 상쇄되고 픽셀 노이즈는 그대로
    통과했다 — 합성 피부 테스트에서 검출이 0이던 원인이다.

    반환: 잡티 영역이 255인 단채널 마스크.
    """
    h, w = img_bgr.shape[:2]
    short_side = min(h, w)
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)

    # A·B 채널만 사용 (L 채널 제외 → 음영·조명 무시)
    a_ch = lab[:, :, 1].astype(np.float32)
    b_ch = lab[:, :, 2].astype(np.float32)

    # 신호: 잡티 크기 정도로만 살짝 평활 → 픽셀 노이즈·필름 그레인을 걷어낸다
    sig_sigma = max(1.0, short_side * 0.004)
    a_sig = cv2.GaussianBlur(a_ch, (0, 0), sig_sigma)
    b_sig = cv2.GaussianBlur(b_ch, (0, 0), sig_sigma)

    # 배경: 잡티보다 훨씬 넓은 창 → 얼굴 전체의 색조·조명 그라데이션
    bg_ksize = max(31, int(short_side * 0.08)) | 1
    a_bg = cv2.GaussianBlur(a_ch, (bg_ksize, bg_ksize), 0)
    b_bg = cv2.GaussianBlur(b_ch, (bg_ksize, bg_ksize), 0)

    # 색상 편차 (A·B 밴드패스)
    diff = np.sqrt((a_sig - a_bg) ** 2 + (b_sig - b_bg) ** 2)

    # 밴드패스 후 스케일: 정상 피부는 1 이하, 눈에 보이는 잡티가 3~7이다.
    # intensity 0.35(자동 기본) → 3.95(뚜렷한 것만), 1.0 → 2.0(옅은 것까지)
    threshold = max(2.0, 5.0 - intensity * 3.0)

    blemish_mask = (diff > threshold).astype(np.uint8) * 255
    blemish_mask = cv2.bitwise_and(blemish_mask, skin_mask)

    # 모폴로지로 노이즈 제거
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    blemish_mask = cv2.morphologyEx(blemish_mask, cv2.MORPH_OPEN, kernel, iterations=2)

    # 개별 잡티 크기 필터 — 작은 점만 (점·여드름 크기)
    min_area = max(4, int((short_side * 0.002) ** 2))
    max_area = int((short_side * 0.035) ** 2)  # 점·여드름 크기 상한
    contours, _ = cv2.findContours(blemish_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filtered_mask = np.zeros_like(blemish_mask)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if min_area <= area <= max_area:
            cv2.drawContours(filtered_mask, [cnt], -1, 255, cv2.FILLED)

    # 임계값을 넘는 건 잡티의 코어뿐이고 번진 테두리는 남는다. 그대로 인페인팅하면
    # 그 테두리 색을 다시 안쪽으로 끌어와 잡티가 옅게 남는다 — 코어를 넓혀
    # 잡티 전체를 덮는다. (크기 필터 뒤에 해야 max_area에 걸리지 않는다.)
    dilate_px = max(3, int(short_side * 0.01)) | 1
    filtered_mask = cv2.dilate(
        filtered_mask,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate_px, dilate_px)),
    )
    # 확장분이 입술·눈 경계로 새지 않도록 피부 영역으로 다시 자른다
    return cv2.bitwise_and(filtered_mask, skin_mask)


def _inpaint_blemishes(img_bgr: np.ndarray, blemish_mask: np.ndarray) -> np.ndarray:
    """OpenCV INPAINT_NS(Navier-Stokes)로 잡티 영역을 복원한다."""
    # 반경이 잡티보다 작으면 가운데가 덜 채워진다 — 해상도에 맞춰 키운다
    radius = max(3, int(min(img_bgr.shape[:2]) * 0.004))
    return cv2.inpaint(img_bgr, blemish_mask, inpaintRadius=radius, flags=cv2.INPAINT_NS)


def apply_blemish_removal(
    img: Image.Image,
    intensity: float,
    cache: MediaPipeCache | None = None,
) -> Image.Image:
    """잡티 자동 제거. intensity: 0.0(비활성) ~ 1.0(최대 감도).

    파이프라인:
      1. MediaPipe Face Mesh로 피부 마스크 (경계 침식)
      2. LAB A·B 채널 색상 이상치로 잡티 탐지
      3. OpenCV inpainting으로 잡티 영역만 복원
      4. 잡티 영역만 마스크 기반 블렌딩

    cache가 제공되면 MediaPipe 모델/결과 캐시를 재사용한다.
    """
    if intensity < 0.01:
        return img

    arr_rgb = np.array(img)
    arr_bgr = cv2.cvtColor(arr_rgb, cv2.COLOR_RGB2BGR)

    # 1. 피부 마스크 (얼굴 미감지 → 원본 반환)
    skin_mask = _get_skin_mask(arr_rgb, cache=cache, for_blemish=True)
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
    #    intensity는 "무엇을 잡티로 볼지"의 감도다. 잡티라고 판정한 뒤에
    #    절반만 지울 이유는 없으므로 채움 강도는 따로 둔다.
    short_side = min(arr_bgr.shape[:2])
    feather = max(5, int(short_side * 0.004)) | 1
    blend_mask = cv2.GaussianBlur(blemish_mask, (feather, feather), 0)
    fill = min(1.0, 0.6 + intensity * 0.4)
    alpha = (blend_mask.astype(np.float32) / 255.0 * fill)[:, :, np.newaxis]

    result_bgr = arr_bgr.astype(np.float32) * (1.0 - alpha) + inpainted.astype(np.float32) * alpha
    result_bgr = np.clip(result_bgr, 0, 255).astype(np.uint8)
    result_rgb = cv2.cvtColor(result_bgr, cv2.COLOR_BGR2RGB)

    return Image.fromarray(result_rgb)


# ── AI 자동 편집 (autoEdits) ──


# 크롭으로 남길 최소 비율. 프롬프트가 모델에게 약속한 값과 같다.
#
# 예전 하한은 0.05였고 그 외에는 절대 50px 가드뿐이었다. 프롬프트는 크롭을
# "적극 추천", "과감하게 줌인"하라고 밀어붙이므로, 모델이 작은 박스를 주면
# 4000x3000 사진이 240x180으로 잘려 나왔다 (50px 가드는 썸네일만 지킨다).
_CROP_MIN_SIDE = 0.3


def apply_smart_crop(
    img: Image.Image, crop: dict, allow_vertical_crop: bool = True
) -> Image.Image:
    """AI가 추천한 영역으로 이미지를 크롭(줌인)한다.

    crop: {"x": 0~1, "y": 0~1, "width": 0~1, "height": 0~1} (정규화 좌표)

    allow_vertical_crop=False면 위아래를 자르지 않는다. 인물 사진에서 이 크롭이
    머리와 발을 잘라 다리가 짧아 보이게 하던 경로다 — apply_instagram_ratio만
    이 플래그를 받고 있어서 여기로 새어 나갔다.
    """
    try:
        w, h = img.size
        x = max(0.0, min(1.0, float(crop.get("x", 0))))
        y = max(0.0, min(1.0, float(crop.get("y", 0))))
        cw = max(_CROP_MIN_SIDE, min(1.0, float(crop.get("width", 1))))
        ch = max(_CROP_MIN_SIDE, min(1.0, float(crop.get("height", 1))))
        if not allow_vertical_crop:
            y, ch = 0.0, 1.0

        left = int(x * w)
        top = int(y * h)
        right = min(w, int((x + cw) * w))
        bottom = min(h, int((y + ch) * h))

        if right - left < 50 or bottom - top < 50:
            return img

        return img.crop((left, top, right, bottom))
    except Exception:
        return img


def apply_instagram_ratio(
    img: Image.Image, ratio: str, allow_vertical_crop: bool = True
) -> Image.Image:
    """인스타그램 최적 비율로 중앙 크롭한다.

    ratio: "4:5" (피드 최적) 또는 "1:1" (정사각형)

    allow_vertical_crop=False면 위아래를 자르지 않는다. 전신 인물 사진에서
    가운데를 기준으로 위아래를 자르면 머리와 발이 잘려 다리가 짧아 보인다.
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
            if not allow_vertical_crop:
                return img
            new_h = int(w / target)
            top = (h - new_h) // 2
            return img.crop((0, top, w, top + new_h))
    except Exception:
        return img


# 인페인팅으로 메울 수 있는 최대 크기 (프레임 면적 대비).
#
# cv2.inpaint(TELEA)는 경계에서 안쪽으로 값을 밀어 넣는 방식이라 구멍이 깊어질수록
# 근거 없이 지어낸 픽셀이 된다. 합성 텍스처로 실측한 복원 오차(RMSE):
#   0.3% → 10,  1% → 12,  2% → 17,  3% → 21,  10% → 28,  30% → 37
# 예전 상한은 30%였다. 그 크기를 메우면 사진 한복판에 문질러 놓은 얼룩이 남아,
# 거슬리는 요소를 지우려다 사진을 더 망친다. 못 지우고 남는 편이 낫다.
_INPAINT_MAX_AREA = 0.02        # 영역 하나
_INPAINT_MAX_TOTAL_AREA = 0.05  # 전체 합 — 작은 영역 여러 개로 우회하지 못하게

# 인페인팅 이웃 반경. 실측상 3~20 사이에서 오차 차이가 1레벨 미만인데
# 시간은 20배 벌어진다 (2% 영역에서 12ms vs 235ms). 작게 고정한다.
_INPAINT_RADIUS = 5


def apply_object_removal(img: Image.Image, areas: list[dict]) -> Image.Image:
    """AI가 지정한 영역의 불필요한 요소를 인페인팅으로 제거한다.

    areas: [{"x": 0~1, "y": 0~1, "width": 0~1, "height": 0~1}, ...]
    좌표는 원본 프레임 기준이다 — [apply_auto_edits]가 기하 보정보다 먼저 부른다.
    """
    if not areas:
        return img

    try:
        arr_rgb = np.array(img)
        arr_bgr = cv2.cvtColor(arr_rgb, cv2.COLOR_RGB2BGR)
        h, w = arr_bgr.shape[:2]
        frame_area = float(h * w)

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

            box_area = (right - left) * (bottom - top)
            if box_area > frame_area * _INPAINT_MAX_AREA:
                log.info("object_removal: 영역 %.1f%%가 상한 %.0f%% 초과 — 건너뜀",
                         box_area / frame_area * 100, _INPAINT_MAX_AREA * 100)
                continue
            if (cv2.countNonZero(mask) + box_area) > frame_area * _INPAINT_MAX_TOTAL_AREA:
                log.info("object_removal: 누적 면적이 상한 %.0f%% 초과 — 나머지 건너뜀",
                         _INPAINT_MAX_TOTAL_AREA * 100)
                break

            mask[top:bottom, left:right] = 255

        if cv2.countNonZero(mask) == 0:
            return img

        inpainted = cv2.inpaint(arr_bgr, mask, _INPAINT_RADIUS, cv2.INPAINT_TELEA)

        log.info("object_removal: %d개 영역, 총 %.2f%% 메움",
                 len(areas), cv2.countNonZero(mask) / frame_area * 100)
        result_rgb = cv2.cvtColor(inpainted, cv2.COLOR_BGR2RGB)
        return Image.fromarray(result_rgb)
    except Exception:
        return img


def apply_straighten(img: Image.Image, angle: float) -> Image.Image:
    """이미지 수평 보정. angle: 회전 각도 (도 단위, 시계방향 양수)."""
    return _straighten_with_matrix(img, angle)[0]


def _straighten_with_matrix(
    img: Image.Image, angle: float
) -> tuple[Image.Image, np.ndarray | None]:
    """[apply_straighten]과 같지만 원본→결과 좌표 변환 행렬도 돌려준다.

    회전 후 생기는 검은 여백을 자동으로 크롭하여 깔끔한 결과를 반환한다.
    안전장치: ±15도 초과 시 의도적 기울기로 판단하여 무시.
    손대지 않았으면 행렬은 None이다.
    """
    if abs(angle) < 0.1:
        return img, None

    # 안전장치: 극단적 각도는 의도적 구도로 판단
    if abs(angle) > 15.0:
        log.warning("straighten: angle %.1f° exceeds ±15° limit, skipping", angle)
        return img, None

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
            return img, None

        cropped = rotated[top:bottom, left:right]
        result_rgb = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)

        log.info("straighten: rotated %.1f°, size %dx%d → %dx%d",
                 angle, w, h, right - left, bottom - top)
        rotation = np.vstack([rot_mat, [0.0, 0.0, 1.0]])
        return Image.fromarray(result_rgb), _translation(-left, -top) @ rotation

    except Exception as exc:
        log.warning("straighten failed: %s", exc)
        return img, None


def _map_normalized_box(
    box: dict,
    matrix: np.ndarray | None,
    src_size: tuple[int, int],
    dst_size: tuple[int, int],
) -> dict | None:
    """정규화 좌표 박스를 기하 보정 뒤의 프레임 좌표로 옮긴다.

    모델은 원본 사진을 보고 0~1 좌표를 짚는다. 그 사이에 키스톤·수평 보정이
    프레임을 회전시키고 잘라내므로, 같은 0~1 값이 다른 곳을 가리키게 된다.
    박스의 네 꼭짓점을 변환 행렬로 옮긴 뒤 축에 평행한 외접 사각형을 취한다.

    matrix가 None이면 프레임이 그대로라 박스도 그대로다.
    보정으로 프레임 밖으로 밀려나 남는 영역이 거의 없으면 None.
    """
    try:
        x = float(box.get("x", 0.0))
        y = float(box.get("y", 0.0))
        bw = float(box.get("width", 0.0))
        bh = float(box.get("height", 0.0))
    except (TypeError, ValueError):
        return None

    if matrix is None:
        return {"x": x, "y": y, "width": bw, "height": bh}

    sw, sh = src_size
    dw, dh = dst_size
    corners = np.array([[
        [x * sw, y * sh],
        [(x + bw) * sw, y * sh],
        [(x + bw) * sw, (y + bh) * sh],
        [x * sw, (y + bh) * sh],
    ]], dtype=np.float32)
    moved = cv2.perspectiveTransform(corners, matrix.astype(np.float64))[0]

    x0 = float(np.clip(moved[:, 0].min(), 0, dw))
    x1 = float(np.clip(moved[:, 0].max(), 0, dw))
    y0 = float(np.clip(moved[:, 1].min(), 0, dh))
    y1 = float(np.clip(moved[:, 1].max(), 0, dh))
    if x1 - x0 < 2 or y1 - y0 < 2:
        return None

    return {"x": x0 / dw, "y": y0 / dh,
            "width": (x1 - x0) / dw, "height": (y1 - y0) / dh}


def apply_auto_edits(
    img: Image.Image, auto_edits: dict, allow_vertical_crop: bool = True
) -> Image.Image:
    """AI autoEdits를 순서대로 적용한다.

    순서: 불필요 요소 제거 → 원근 보정 → 수평 보정 → 스마트 크롭 → 인스타 비율

    요소 제거가 맨 앞인 이유: 모델은 원본 사진을 보고 좌표를 짚는데, 기하 보정이
    프레임을 회전시키고 잘라내 그 좌표를 밀어 놓는다 (수평 3° + 키스톤 0.15에서
    x=0.10이 0.12로 이동 — 작은 박스는 대상 자체를 벗어난다). 인페인팅은 프레임과
    무관한 픽셀 편집이라 앞으로 옮기면 짚은 곳에 정확히 걸린다.

    크롭은 기하 보정 뒤에 남는다. 먼저 자르면 회전 보정이 그 프레임을 다시 잘라
    의도한 구도가 틀어진다. 대신 좌표를 변환 행렬로 함께 옮겨, 원본 기준으로
    짚은 박스가 새 프레임에서도 같은 곳을 가리키게 한다.

    allow_vertical_crop은 auto_edits 안에 같은 이름의 키가 있으면 그것을 따른다.
    분석 시점에만 피사체가 무엇인지 알 수 있는데, 저장·미리보기는 그때 만든
    autoEdits를 앱이 되돌려 보내 다시 적용하는 구조다. 판단을 딕셔너리에
    실어 두면 어느 경로로 들어와도 같은 결정이 적용된다.
    """
    src_size = img.size
    recorded = auto_edits.get("allow_vertical_crop")
    if isinstance(recorded, bool):
        allow_vertical_crop = recorded

    # 1. 불필요한 요소 제거 — 원본 좌표계에서 (기하 보정 전에)
    remove_areas = auto_edits.get("remove_areas")
    if remove_areas and isinstance(remove_areas, list):
        img = apply_object_removal(img, remove_areas)

    # 2-a. 수직 원근 보정 (키스톤)
    geometry: np.ndarray | None = None
    keystone = auto_edits.get("keystone")
    if keystone is not None:
        try:
            img, matrix = _keystone_with_matrix(img, float(keystone))
            geometry = _compose_geometry(geometry, matrix)
        except (TypeError, ValueError):
            pass

    # 2-b. 수평 보정 (기울기 교정)
    straighten = auto_edits.get("straighten")
    if straighten is not None:
        try:
            img, matrix = _straighten_with_matrix(img, float(straighten))
            geometry = _compose_geometry(geometry, matrix)
        except (TypeError, ValueError):
            pass

    # 3. 스마트 크롭 (줌/리프레임) — 원본 좌표를 새 프레임으로 옮겨서
    crop = auto_edits.get("crop")
    if crop and isinstance(crop, dict):
        mapped = _map_normalized_box(crop, geometry, src_size, img.size)
        if mapped is None:
            log.info("smart_crop: 기하 보정 후 크롭 영역이 남지 않음 — 건너뜀")
        else:
            img = apply_smart_crop(img, mapped, allow_vertical_crop)

    # 4. 인스타그램 비율 크롭
    ig_ratio = auto_edits.get("instagram_ratio")
    if ig_ratio and isinstance(ig_ratio, str):
        img = apply_instagram_ratio(img, ig_ratio, allow_vertical_crop)

    return img


# ── 영역별 스마트 보정 ──


# 하늘로 인정할 덩어리의 조건.
# top: 덩어리의 윗변이 프레임 위쪽 이 비율 안에서 시작해야 한다
#      (0이 아니라 여유를 두는 이유 — 처마·나뭇가지가 맨 위를 가릴 수 있다)
# width: 옆으로 이만큼 넓어야 한다. 위쪽에 걸린 파란 간판·표지판을 걸러낸다
_SKY_MAX_TOP = 0.15
_SKY_MIN_WIDTH = 0.15


def _keep_sky_like_components(mask: np.ndarray) -> np.ndarray:
    """마스크에서 하늘처럼 생긴 덩어리만 남긴다.

    조건: 프레임 위쪽에서 시작하고(윗변이 상단 15% 안), 옆으로 넓다(폭 15% 이상).
    """
    if cv2.countNonZero(mask) == 0:
        return mask

    h, w = mask.shape[:2]
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    kept = np.zeros((h, w), dtype=np.uint8)
    for idx in range(1, count):
        top = stats[idx, cv2.CC_STAT_TOP]
        width = stats[idx, cv2.CC_STAT_WIDTH]
        if top <= h * _SKY_MAX_TOP and width >= w * _SKY_MIN_WIDTH:
            kept[labels == idx] = 255
    return kept


def detect_regions(
    img: Image.Image,
    cache: MediaPipeCache | None = None,
) -> dict[str, np.ndarray]:
    """HSV 기반 하늘 감지 + MediaPipe 얼굴 감지 + 나머지=배경.

    반환: {"sky": mask, "face": mask, "background": mask}
    각 마스크는 0~255 uint8 단채널. 영역이 없으면 해당 키가 빈 마스크(전체 0).

    cache가 제공되면 MediaPipe 모델/결과 캐시를 재사용한다.
    """
    arr_rgb = np.array(img, dtype=np.uint8)
    h, w = arr_rgb.shape[:2]
    arr_bgr = cv2.cvtColor(arr_rgb, cv2.COLOR_RGB2BGR)
    hsv = cv2.cvtColor(arr_bgr, cv2.COLOR_BGR2HSV)

    # ── 하늘 감지 ──
    # H: 90~130 (파란~시안), S: 30+, V: 100+
    lower_sky = np.array([90, 30, 100], dtype=np.uint8)
    upper_sky = np.array([130, 255, 255], dtype=np.uint8)
    sky_mask = cv2.inRange(hsv, lower_sky, upper_sky)

    # 모폴로지로 노이즈 제거 및 영역 연결
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    sky_mask = cv2.morphologyEx(sky_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    sky_mask = cv2.morphologyEx(sky_mask, cv2.MORPH_OPEN, kernel, iterations=1)

    # 위치·모양으로 걸러낸다. 색만 보면 파란 셔츠·물·파란 벽·유리창이 다 하늘로
    # 잡히고, 거기에 하늘용 보정(밝기↓ 채도↑)이 걸린다.
    #
    # 예전에는 상단 가중치로 걸러 보려 했지만 실제로는 아무것도 못 걸렀다.
    # 가중치의 하한이 0.4인데 통과 문턱이 0.3이라, 프레임 어디에 있든 색만
    # 맞으면 통과했다.
    #
    # 하늘은 (1) 프레임 위쪽에서 시작하고 (2) 옆으로 넓다. 두 조건을 다 만족하는
    # 덩어리만 남긴다. 창문 너머로만 보이는 하늘은 놓치지만, 옷을 하늘로
    # 착각해 색을 틀어 놓는 것보다 낫다.
    sky_mask = _keep_sky_like_components(sky_mask)

    # 하늘 영역이 이미지의 5% 미만이면 하늘 없음으로 처리
    if cv2.countNonZero(sky_mask) / (h * w) < 0.05:
        sky_mask = np.zeros((h, w), dtype=np.uint8)

    # ── 얼굴 감지 ──
    # 영역별 보정의 face는 밝기·색온도 같은 톤 조절이 주 용도라 얼굴 전체를 덮는
    # 마스크를 쓴다. 눈·입술을 뚫어 둔 질감용 마스크로 톤을 바꾸면 얼룩이 진다.
    face_mask = _get_skin_mask(arr_rgb, cache=cache, mode="tone")
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


# ── 영역별 보정 블렌딩 ──

# 톤을 바꾸는 파라미터. 이웃 영역과 값이 벌어지면 경계가 그대로 드러난다.
_REGION_TONE_PARAMS = (
    "brightness", "contrast", "saturation", "temperature", "highlights", "shadows",
)

# 얼굴 톤 보정의 상한. 모델은 -1.0~1.0을 주지만 얼굴 마스크는 목·귀·머리카락을
# 포함하지 않으므로, 이 범위를 넘겨 밝히면 페더를 아무리 넓혀도 얼굴만
# 오려 붙인 것처럼 겉돈다. 전체 보정으로 올릴 몫은 슬라이더 쪽에 있다.
_FACE_TONE_LIMIT = 0.18

# 얼굴 질감 보정(잡티·스무딩)의 상한. 전역 슬라이더가 같은 픽셀에 한 번 더
# 적용하므로 두 패스가 겹친다.
_FACE_TEXTURE_LIMIT = 0.5

# 국소 보정(local_*)의 파라미터별 상한. 얼굴과 달리 "날아간 창문을 살린다"처럼
# 의도가 분명한 교정이라 더 크게 허용하되, 한 영역이 사진을 지배하지는 못하게 한다.
# 하늘·배경의 상한. 얼굴만 묶여 있었고 이쪽은 프롬프트가 -1.0~1.0을 허용했다.
# 실측: sky brightness -1.0 → 하늘 평균 167에서 98로, background -1.0 → 165에서
# 105로. 손대지 않은 피사체만 남고 주변이 터널처럼 어두워진다.
_SKY_LIMIT = 0.35
_BACKGROUND_LIMIT = 0.30

_LOCAL_LIMITS: dict[str, float] = {
    "brightness": 0.50,
    "highlights": 0.60,
    "shadows": 0.60,
    "contrast": 0.35,
    "saturation": 0.35,
    "temperature": 0.35,
    "sharpness": 0.50,
}

# 영역 딕셔너리에서 보정값이 아니라 마스크를 만드는 데 쓰이는 키.
# 변형 함수를 찾을 때 건너뛴다.
_REGION_META_KEYS = frozenset({"area", "shape", "feather", "reason"})

# 국소 보정 영역 이름의 접두사. 모델은 local_0, local_1... 로 준다.
_LOCAL_PREFIX = "local"

# 국소 보정 영역의 개수·크기 한도.
_LOCAL_MAX_COUNT = 4
_LOCAL_MIN_AREA = 0.003   # 프레임 대비 — 이보다 작으면 마스크를 풀면 사라진다
_LOCAL_MAX_AREA = 0.50    # 이보다 크면 국소 보정이 아니라 전체 보정이다

# 텍스처를 건드리는 보정 — 비싸서 미리보기에서는 뺀다.
_REGION_TEXTURE_PARAMS = ("blemish_removal", "skin_smoothing")

# 영역 합성 우선순위 (작을수록 위). background는 하늘도 얼굴도 아닌 여집합이라
# 항상 맨 아래여야 한다. 얼굴은 페더가 밖으로 뻗으므로 맨 위에 얹는다.
# 국소 보정은 모델이 좌표까지 짚은 구체적 지시라 하늘·배경보다 위에 둔다.
_REGION_PRIORITY = {"face": 0, "sky": 2, "background": 3}
_LOCAL_PRIORITY = 1


def _region_priority(region_name: str) -> int:
    """합성 순서를 돌려준다. local_0, local_1... 은 모두 같은 층이다."""
    if region_name.startswith(_LOCAL_PREFIX):
        return _LOCAL_PRIORITY
    return _REGION_PRIORITY.get(region_name, _LOCAL_PRIORITY)


# 얼굴 마스크를 알파로 풀 때의 페더 크기 (영역 등가 반지름 대비).
# 얼굴 크기에 비례해야 작게 찍힌 얼굴도 같은 정도로 부드러워진다.
_FACE_FEATHER_SIGMA = 0.12
# 블러 전에 마스크를 밖으로 넓히는 양 (sigma 대비).
# 넓히지 않고 그냥 블러하면 얼굴 테두리의 알파가 깎여, 얼굴 한가운데만
# 톤이 살고 윤곽은 원본으로 남는다.
_FACE_FEATHER_GROW = 2.0

# 국소 보정 페더. 모델이 짚은 영역을 넘어 번지면 안 되니 얼굴보다 좁게 잡고,
# 넓힘도 sigma의 1배까지만 — 안쪽 알파는 1.0로 유지되면서 바깥 번짐은 억제된다.
_LOCAL_FEATHER_SIGMA = 0.16
_LOCAL_FEATHER_GROW = 1.0


def build_local_regions(
    size: tuple[int, int],
    region_params: dict[str, dict[str, Any]],
) -> dict[str, np.ndarray]:
    """region_params의 local_* 항목에서 국소 보정 마스크를 만든다.

    모델이 좌표로 짚은 영역(밝기가 날아간 창문, 그늘에 묻힌 피사체 등)을
    마스크로 바꾼다. 하늘·얼굴처럼 감지로 찾는 영역과 달리 기하 정보만
    있으면 되므로 MediaPipe가 필요 없다.

    각 항목의 형태::

        "local_0": {
            "area": {"x": 0.55, "y": 0.1, "width": 0.2, "height": 0.35},
            "shape": "rect" | "ellipse",
            "feather": 0.0~1.0,
            "reason": "왼쪽 창문이 날아감",
            "highlights": -0.45, "brightness": -0.15
        }

    좌표는 정규화(0~1)이고 area/shape/feather/reason은 보정값이 아니다.
    범위를 벗어나거나 너무 작고 큰 영역은 조용히 버린다 — 모델이 좌표를
    잘못 짚었을 때 사진 절반에 보정이 걸리는 것보다 아무것도 안 하는 게 낫다.
    """
    w, h = size
    masks: dict[str, np.ndarray] = {}
    if not region_params:
        return masks

    names = sorted(n for n in region_params if n.startswith(_LOCAL_PREFIX))
    for name in names:
        if len(masks) >= _LOCAL_MAX_COUNT:
            log.info("local region %s: 개수 한도(%d) 초과 — 버림", name, _LOCAL_MAX_COUNT)
            continue

        spec = region_params.get(name)
        if not isinstance(spec, dict):
            continue
        area = spec.get("area")
        if not isinstance(area, dict):
            log.info("local region %s: area 없음 — 버림", name)
            continue

        try:
            ax = float(area.get("x", 0.0))
            ay = float(area.get("y", 0.0))
            aw = float(area.get("width", 0.0))
            ah = float(area.get("height", 0.0))
        except (TypeError, ValueError):
            log.info("local region %s: area 좌표를 읽을 수 없음 — 버림", name)
            continue

        ax = min(max(ax, 0.0), 1.0)
        ay = min(max(ay, 0.0), 1.0)
        aw = min(max(aw, 0.0), 1.0 - ax)
        ah = min(max(ah, 0.0), 1.0 - ay)

        ratio = aw * ah
        if ratio < _LOCAL_MIN_AREA or ratio > _LOCAL_MAX_AREA:
            log.info("local region %s: 면적 %.1f%%가 허용 범위(%.1f~%.0f%%) 밖 — 버림",
                     name, ratio * 100, _LOCAL_MIN_AREA * 100, _LOCAL_MAX_AREA * 100)
            continue

        left, top = int(ax * w), int(ay * h)
        right, bottom = int((ax + aw) * w), int((ay + ah) * h)
        if right - left < 2 or bottom - top < 2:
            continue

        mask = np.zeros((h, w), dtype=np.uint8)
        if str(spec.get("shape", "ellipse")).lower() == "rect":
            mask[top:bottom, left:right] = 255
        else:
            cv2.ellipse(
                mask,
                center=((left + right) // 2, (top + bottom) // 2),
                axes=(max(1, (right - left) // 2), max(1, (bottom - top) // 2)),
                angle=0, startAngle=0, endAngle=360, color=255, thickness=-1,
            )
        masks[name] = mask
        log.info("local region %s: %s %dx%d @ (%d,%d) — %s",
                 name, spec.get("shape", "ellipse"), right - left, bottom - top,
                 left, top, spec.get("reason") or "(이유 없음)")

    return masks


def _soften_region_mask(
    mask: np.ndarray,
    region_name: str,
    feather_scale: float = 1.0,
) -> np.ndarray:
    """영역 마스크(0/255)를 블렌딩용 알파(0.0~1.0)로 바꾼다.

    얼굴은 턱선·헤어라인이 실제 경계가 아니라 마스크의 끝일 뿐이므로,
    마스크를 밖으로 넓힌 뒤 얼굴 크기에 비례해 길게 푼다. 그래야 톤이
    목·귀까지 이어져 얼굴만 밝은 타원으로 보이지 않는다.

    국소 보정도 같은 이유로 영역 크기에 비례해 풀되, 모델이 짚은 범위를
    크게 벗어나지 않도록 얼굴보다 좁게 잡는다.

    하늘·배경은 지평선처럼 실제 경계를 따르므로 좁게 유지한다.
    넓게 풀면 건물이나 인물 윤곽에 후광이 생긴다.
    """
    is_face = region_name == "face"
    is_local = region_name.startswith(_LOCAL_PREFIX)
    if not (is_face or is_local):
        blur_size = max(21, int(min(mask.shape[:2]) * 0.03)) | 1
        return cv2.GaussianBlur(mask, (blur_size, blur_size), 0).astype(np.float32) / 255.0

    # 여러 명이 찍힌 사진에서 마스크 전체 면적을 쓰면 얼굴 수만큼 페더가
    # 커진다. 가장 큰 얼굴 하나의 크기를 기준으로 삼는다.
    count, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    area = float(stats[1:, cv2.CC_STAT_AREA].max()) if count > 1 else 0.0
    radius = max(8.0, float(np.sqrt(area / np.pi)))

    sigma_ratio = _FACE_FEATHER_SIGMA if is_face else _LOCAL_FEATHER_SIGMA
    grow_ratio = _FACE_FEATHER_GROW if is_face else _LOCAL_FEATHER_GROW
    sigma = max(4.0, radius * sigma_ratio * feather_scale)
    grow = max(1, int(sigma * grow_ratio))

    # 얼굴이 크면 넓히기·풀기 모두 원본 해상도에서 비싸다 (1400px 얼굴에서 0.7초).
    # 램프는 부드러운 저주파라 축소해서 만들고 되돌려도 눈에 차이가 없다.
    shrink = min(1.0, 8.0 / sigma)
    work = mask
    if shrink < 1.0:
        work = cv2.resize(mask, None, fx=shrink, fy=shrink,
                          interpolation=cv2.INTER_AREA)
        sigma *= shrink
        grow = max(1, int(grow * shrink))

    work = cv2.dilate(
        work,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (grow * 2 + 1,) * 2),
    )
    ksize = int(sigma * 6) | 1
    work = cv2.GaussianBlur(work, (ksize, ksize), sigma)

    if work.shape[:2] != mask.shape[:2]:
        work = cv2.resize(work, (mask.shape[1], mask.shape[0]),
                          interpolation=cv2.INTER_LINEAR)
    return work.astype(np.float32) / 255.0


def _limit_region_value(region_name: str, param_name: str, value: float) -> float:
    """영역별 보정값을 그 영역이 자연스럽게 감당할 수 있는 범위로 자른다."""
    if region_name == "face":
        if param_name in _REGION_TONE_PARAMS:
            return float(np.clip(value, -_FACE_TONE_LIMIT, _FACE_TONE_LIMIT))
        if param_name in _REGION_TEXTURE_PARAMS:
            # 전역 슬라이더에서도 같은 픽셀에 한 번 더 걸리므로, 여기서 1.0을
            # 허용하면 두 번 겹쳐 피부가 밀랍처럼 된다.
            return float(np.clip(value, 0.0, _FACE_TEXTURE_LIMIT))
        return value
    if region_name == "sky" and param_name in _REGION_TONE_PARAMS:
        return float(np.clip(value, -_SKY_LIMIT, _SKY_LIMIT))
    if region_name == "background" and param_name in _REGION_TONE_PARAMS:
        return float(np.clip(value, -_BACKGROUND_LIMIT, _BACKGROUND_LIMIT))
    if region_name.startswith(_LOCAL_PREFIX):
        limit = _LOCAL_LIMITS.get(param_name)
        if limit is not None:
            return float(np.clip(value, -limit, limit))
    return value


def _feather_scale(spec: dict[str, Any]) -> float:
    """모델이 준 feather(0~1)를 페더 배율(0.5~1.5)로 바꾼다. 없으면 1.0."""
    raw = spec.get("feather")
    if raw is None:
        return 1.0
    try:
        return 0.5 + float(np.clip(float(raw), 0.0, 1.0))
    except (TypeError, ValueError):
        return 1.0


def apply_regional_transforms(
    img: Image.Image,
    regions: dict[str, np.ndarray],
    region_params: dict[str, dict[str, float]],
    cache: MediaPipeCache | None = None,
    preview: bool = False,
) -> Image.Image:
    """영역별로 다른 보정을 적용한 뒤 마스크 경계를 블렌딩.

    region_params 예시:
    {
      "sky": {"brightness": 0.1, "saturation": -0.1, "temperature": 0.0},
      "face": {"brightness": 0.1, "blemish_removal": 0.3, "skin_smoothing": 0.2},
      "background": {"brightness": 0.0, "contrast": 0.1, "saturation": -0.05},
      "local_0": {"area": {...}, "shape": "rect", "highlights": -0.45}
    }

    local_* 영역의 마스크는 [build_local_regions]로 만들어 regions에 넣어 둔다.

    preview가 True면 잡티 제거·피부 스무딩은 건너뛴다 (비용이 크다).
    톤 보정은 그대로 적용해 미리보기와 저장본의 색이 갈리지 않게 한다.

    cache가 제공되면 MediaPipe 모델/결과 캐시를 재사용한다.
    """
    arr_rgb = np.array(img, dtype=np.float32)

    # 사용 가능한 변형 함수 맵
    # blemish_removal은 cache를 전달해야 하므로 lambda로 래핑
    transform_funcs: dict[str, Any] = {
        "brightness": adjust_brightness,
        "contrast": adjust_contrast,
        "saturation": adjust_saturation,
        "temperature": adjust_color_temperature,
        "highlights": adjust_highlights,
        "shadows": adjust_shadows,
        "blemish_removal": lambda img_, val: apply_blemish_removal(img_, val, cache=cache),
        "skin_smoothing": lambda img_, val: apply_skin_smoothing(img_, val, cache=cache),
        "sharpness": apply_sharpness,
    }

    # 영역별 결과와 알파를 먼저 모은다 — 순서대로 덧칠하면 안 된다.
    layers: list[tuple[str, np.ndarray, np.ndarray]] = []

    for region_name, params in region_params.items():
        if not params or region_name not in regions:
            continue

        mask = regions[region_name]
        if cv2.countNonZero(mask) == 0:
            continue

        # 이 영역에 해당하는 변형을 순서대로 적용
        region_img = img.copy()
        applied = False
        for param_name, raw in params.items():
            if param_name in _REGION_META_KEYS:
                continue
            if preview and param_name in _REGION_TEXTURE_PARAMS:
                continue
            try:
                value = float(raw)
            except (TypeError, ValueError):
                continue

            limited = _limit_region_value(region_name, param_name, value)
            if limited != value:
                log.info("regional: %s %s %.2f → %.2f (상한)",
                         region_name, param_name, value, limited)
                value = limited
            if abs(value) < 0.01:
                continue

            func = transform_funcs.get(param_name)
            if func is None:
                continue
            if param_name == "contrast":
                # 대비의 기준 밝기는 이 영역 안의 평균이어야 한다. 전체 평균을
                # 쓰면 평탄한 영역에 대비를 걸었을 뿐인데 영역이 통째로
                # 밝아지거나 어두워진다 (밝은 하늘이 기준을 끌어올린다).
                region_img = func(region_img, value, pivot=_region_mean_l(img, mask))
            else:
                region_img = func(region_img, value)
            applied = True

        if not applied:
            continue

        layers.append((
            region_name,
            np.array(region_img, dtype=np.float32),
            _soften_region_mask(mask, region_name, _feather_scale(params)),
        ))

    if not layers:
        return img

    # 우선순위 순으로 알파 합성한다. 앞선 영역이 덮은 만큼만 뒤 영역에 남긴다.
    #
    # 예전에는 영역을 차례로 덧칠했다. 각 영역을 매번 원본에서 새로 계산하니
    # 나중 영역이 앞선 영역의 보정을 경계에서 원본으로 되돌려 놓았다.
    # 게다가 background 마스크는 얼굴의 여집합이라 얼굴 바로 밖에서 알파가
    # 1.0이었고, 얼굴 페더를 얼마나 넓혀도 경계에서 배경 보정이 이겼다.
    # face를 먼저 얹어야 얼굴 톤이 목·귀 쪽으로 실제로 번져 나간다.
    layers.sort(key=lambda item: _region_priority(item[0]))

    result = arr_rgb.copy()
    remaining = np.ones(arr_rgb.shape[:2], dtype=np.float32)
    for _, region_arr, alpha in layers:
        weight = (alpha * remaining)[:, :, np.newaxis]
        result += (region_arr - arr_rgb) * weight
        remaining *= 1.0 - alpha

    return Image.fromarray(np.clip(result, 0, 255).astype(np.uint8))


def _region_mean_l(img: Image.Image, mask: np.ndarray) -> float:
    """마스크 안 픽셀의 평균 L(0~255)을 돌려준다. 비면 전체 평균."""
    lab = cv2.cvtColor(cv2.cvtColor(np.array(img, dtype=np.uint8), cv2.COLOR_RGB2BGR),
                       cv2.COLOR_BGR2LAB)
    l_ch = lab[:, :, 0]
    if mask.shape[:2] != l_ch.shape[:2] or cv2.countNonZero(mask) == 0:
        return float(l_ch.mean())
    return float(cv2.mean(l_ch, mask=mask)[0])

# ── 얼굴/체형 보정 (MLS Warp) ──

# 얼굴 윤곽 랜드마크 인덱스 (MediaPipe 478개 중 양쪽 볼·턱선)
_FACE_CONTOUR_LEFT = [234, 93, 132, 58, 172, 136, 150, 149, 176, 148, 152]
_FACE_CONTOUR_RIGHT = [454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152]

# 턱선 인덱스 (V라인)
_JAW_LEFT = [172, 136, 150, 149, 176, 148]
_JAW_RIGHT = [397, 365, 379, 378, 400, 377]
_JAW_TIP = [152]

# 눈 인덱스 (방사형 확대용)
# 눈 확대는 눈 중심에서 바깥으로 밀어내는 변형이라, 윤곽과 중심이 반드시
# 같은 눈이어야 한다. 예전에는 왼눈 윤곽에 오른눈 홍채 중심이 짝지어져 있어
# 확대가 아니라 "반대편 눈 쪽으로 끌어당기기"가 됐고, 두 눈 사이의 코까지
# 딸려가 한쪽 콧구멍만 커지는 결과가 나왔다.
#
# MediaPipe 규약 (FaceLandmarksConnections로 확인):
#   LEFT_EYE  = 362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398
#   RIGHT_EYE = 33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246
#   홍채는 468~472가 LEFT, 473~477이 RIGHT이며 각 첫 번째가 중심이다.
_LEFT_EYE_CENTER = 468    # LEFT iris 중심
_RIGHT_EYE_CENTER = 473   # RIGHT iris 중심
_LEFT_EYE_CONTOUR = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
_RIGHT_EYE_CONTOUR = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]


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
    ROI가 주어지면 패딩된 ROI 영역만 워프하여 성능 향상 + 배경 아티팩트 방지.
    """
    if roi is None:
        return _mls_similarity_warp(img, src_pts, dst_pts)

    h, w = img.shape[:2]
    rx, ry, rw, rh = roi
    rx = max(0, rx)
    ry = max(0, ry)
    rw = min(w - rx, rw)
    rh = min(h - ry, rh)

    if rw <= 0 or rh <= 0:
        return img

    # ROI를 패딩하여 블렌딩 경계가 자연스럽도록 (블러 커널 크기의 1.5배)
    blur_size = max(31, int(min(rw, rh) * 0.3)) | 1
    pad = blur_size
    crop_x1 = max(0, rx - pad)
    crop_y1 = max(0, ry - pad)
    crop_x2 = min(w, rx + rw + pad)
    crop_y2 = min(h, ry + rh + pad)
    crop_w = crop_x2 - crop_x1
    crop_h = crop_y2 - crop_y1

    # ROI 영역만 잘라내서 워프 (제어점도 ROI 좌표계로 변환)
    crop = img[crop_y1:crop_y2, crop_x1:crop_x2].copy()
    offset = np.array([crop_x1, crop_y1], dtype=np.float32)
    crop_src = src_pts - offset
    crop_dst = dst_pts - offset

    warped_crop = _mls_similarity_warp(crop, crop_src, crop_dst)

    # 타원형 마스크 (ROI 좌표계)
    mask = np.zeros((crop_h, crop_w), dtype=np.float32)
    # 타원 중심과 축을 crop 좌표계로 변환
    ellipse_cx = rx + rw // 2 - crop_x1
    ellipse_cy = ry + rh // 2 - crop_y1
    cv2.ellipse(
        mask,
        center=(ellipse_cx, ellipse_cy),
        axes=(rw // 2, rh // 2),
        angle=0, startAngle=0, endAngle=360,
        color=1.0, thickness=-1,
    )
    mask = cv2.GaussianBlur(mask, (blur_size, blur_size), 0)
    mask = mask[:, :, np.newaxis]

    # 블렌딩 후 원본에 합성
    blended_crop = (
        crop.astype(np.float32) * (1 - mask)
        + warped_crop.astype(np.float32) * mask
    ).astype(np.uint8)

    result = img.copy()
    result[crop_y1:crop_y2, crop_x1:crop_x2] = blended_crop
    return result


def apply_face_reshape(
    img: Image.Image,
    face_slim: float = 0.0,
    jaw_sharpen: float = 0.0,
    eye_enlarge: float = 0.0,
    cache: MediaPipeCache | None = None,
) -> Image.Image:
    """얼굴 보정 — MediaPipe 478 랜드마크 기반 MLS 워프.

    face_slim: 0~1 (얼굴 양쪽을 중심축 방향으로)
    jaw_sharpen: 0~1 (턱 V라인)
    eye_enlarge: 0~1 (눈 확대)

    얼굴 미감지 시 원본 반환. 다중 얼굴은 각각 독립 적용.
    cache가 제공되면 MediaPipe 모델/결과 캐시를 재사용한다.
    """
    if face_slim < 0.01 and jaw_sharpen < 0.01 and eye_enlarge < 0.01:
        return img

    model_path = face_model_path()
    if model_path is None:
        return img

    arr_rgb = np.array(img, dtype=np.uint8)
    h, w = arr_rgb.shape[:2]

    if cache is not None:
        results = cache.get_face_landmarks(arr_rgb)
        if results is None:
            return img
    else:
        base_options = mp.tasks.BaseOptions(model_asset_path=model_path)
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

        # 얼굴 중심축.
        # 코끝(1)을 쓰면 얼굴이 조금만 돌아가도 축이 한쪽으로 치우쳐
        # 볼 슬림·턱선 이동량이 좌우로 달라진다. 광대 양끝(234, 454)의
        # 중점은 고개 방향에 훨씬 덜 흔들린다.
        left_cheek = _pt(234)
        right_cheek = _pt(454)
        cx = (left_cheek[0] + right_cheek[0]) / 2.0

        # ── face_slim: 볼 양쪽을 중심 방향으로 ──
        if face_slim >= 0.01:
            strength = face_slim * 0.14  # 최대 14% 이동
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
            strength_x = jaw_sharpen * 0.10
            strength_y = jaw_sharpen * 0.05
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
            strength = eye_enlarge * 0.18  # 최대 18% 확대
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
        # 얼굴 경계 상자 + 여백. 폭/높이를 이미지 전체 크기로 자르면
        # (min(w, ...)) x가 0으로 밀린 만큼을 반영하지 못해 블렌딩 타원이
        # 얼굴에서 어긋나고, 얼굴의 한쪽만 변형된다.
        margin = int(min(h, w) * 0.05)
        rx = max(0, int(min(fxs)) - margin)
        ry = max(0, int(min(fys)) - margin)
        rx2 = min(w, int(max(fxs)) + margin)
        ry2 = min(h, int(max(fys)) + margin)
        roi = (rx, ry, max(0, rx2 - rx), max(0, ry2 - ry))

        result_arr = _warp_with_mask(result_arr, src_pts, dst_pts, roi)

    return Image.fromarray(result_arr)


def apply_body_reshape(
    img: Image.Image,
    leg_stretch: float = 0.0,
    shoulder_width: float = 0.0,
    waist_slim: float = 0.0,
    cache: MediaPipeCache | None = None,
) -> Image.Image:
    """체형 보정 — MediaPipe Pose 33 랜드마크 기반.

    leg_stretch: 0~1 (힙 아래 수직 스트레칭)
    shoulder_width: -1~1 (음수=좁게, 양수=넓게)
    waist_slim: 0~1 (허리 양쪽을 안쪽으로)

    바디 미감지 시 원본 반환. 다중 바디는 가장 신뢰도 높은 것만.
    cache가 제공되면 MediaPipe 모델/결과 캐시를 재사용한다.
    """
    if abs(leg_stretch) < 0.01 and abs(shoulder_width) < 0.01 and abs(waist_slim) < 0.01:
        return img

    model_path = pose_model_path()
    if model_path is None:
        return img

    arr_rgb = np.array(img, dtype=np.uint8)
    h, w = arr_rgb.shape[:2]

    if cache is not None:
        results = cache.get_pose_landmarks(arr_rgb)
        if results is None:
            return img
    else:
        base_options = mp.tasks.BaseOptions(model_asset_path=model_path)
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
            stretch_factor = 1.0 + leg_stretch * 0.25  # 최대 25% 늘리기

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
        # Pose: 11=왼쪽 어깨, 12=오른쪽 어깨, 23=왼쪽 힙, 24=오른쪽 힙
        ls = _pt(11)
        rs = _pt(12)
        mid_x = (ls[0] + rs[0]) / 2.0
        mid_y = (ls[1] + rs[1]) / 2.0
        strength = shoulder_width * 0.10  # 최대 10%

        # 어깨 위 1/3 지점과 어깨 사이 보간점 추가 → 자연스러운 워프
        neck_y = mid_y - abs(rs[0] - ls[0]) * 0.25  # 어깨 위 목 부근
        below_y = mid_y + abs(rs[0] - ls[0]) * 0.3   # 어깨 아래

        src_list = [
            [ls[0], ls[1]],  # 왼쪽 어깨
            [rs[0], rs[1]],  # 오른쪽 어깨
            # 어깨-목 사이 보간점 (강도 50%)
            [(ls[0] + mid_x) / 2, neck_y],
            [(rs[0] + mid_x) / 2, neck_y],
            # 고정 앵커 (목 중앙, 어깨 아래 중앙) — 변형되지 않아야 할 점
            [mid_x, neck_y],
            [mid_x, below_y],
        ]
        dst_list = [
            [ls[0] + (ls[0] - mid_x) * strength, ls[1]],
            [rs[0] + (rs[0] - mid_x) * strength, rs[1]],
            [(ls[0] + mid_x) / 2 + ((ls[0] + mid_x) / 2 - mid_x) * strength * 0.3, neck_y],
            [(rs[0] + mid_x) / 2 + ((rs[0] + mid_x) / 2 - mid_x) * strength * 0.3, neck_y],
            [mid_x, neck_y],      # 고정
            [mid_x, below_y],     # 고정
        ]

        src_pts = np.array(src_list, dtype=np.float32)
        dst_pts = np.array(dst_list, dtype=np.float32)

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

        strength = waist_slim * 0.11  # 최대 11%

        # 허리 위/아래 보간점 + 고정 앵커 추가 → 자연스러운 수직 그라데이션
        above_y = (ls[1] + waist_y) / 2.0   # 어깨-허리 중간
        below_y = (waist_y + lhip[1]) / 2.0  # 허리-힙 중간

        src_list = [
            # 허리 중심 (주 제어점 — 강도 100%)
            [waist_left_x, waist_y],
            [waist_right_x, waist_y],
            # 허리 위 보간 (강도 40%)
            [waist_left_x, above_y],
            [waist_right_x, above_y],
            # 허리 아래 보간 (강도 40%)
            [waist_left_x, below_y],
            [waist_right_x, below_y],
            # 고정 앵커 (중심축, 어깨, 힙 — 변형되지 않아야 할 점)
            [mid_x, waist_y],
            [ls[0], ls[1]],
            [rs[0], rs[1]],
            [lhip[0], lhip[1]],
            [rhip[0], rhip[1]],
        ]
        dst_list = [
            # 허리 중심 (주 변형)
            [waist_left_x + (mid_x - waist_left_x) * strength, waist_y],
            [waist_right_x + (mid_x - waist_right_x) * strength, waist_y],
            # 허리 위 보간 (40% 강도)
            [waist_left_x + (mid_x - waist_left_x) * strength * 0.4, above_y],
            [waist_right_x + (mid_x - waist_right_x) * strength * 0.4, above_y],
            # 허리 아래 보간 (40% 강도)
            [waist_left_x + (mid_x - waist_left_x) * strength * 0.4, below_y],
            [waist_right_x + (mid_x - waist_right_x) * strength * 0.4, below_y],
            # 고정 앵커 (원래 위치 유지)
            [mid_x, waist_y],
            [ls[0], ls[1]],
            [rs[0], rs[1]],
            [lhip[0], lhip[1]],
            [rhip[0], rhip[1]],
        ]

        src_pts = np.array(src_list, dtype=np.float32)
        dst_pts = np.array(dst_list, dtype=np.float32)

        waist_w = abs(waist_right_x - waist_left_x)
        waist_h = abs(lhip[1] - ls[1])
        roi = (
            max(0, int(waist_left_x - waist_w * 0.3)),
            max(0, int(min(ls[1], above_y) - waist_h * 0.1)),
            min(w, int(waist_w + waist_w * 0.6)),
            min(h, int(waist_h * 1.2)),
        )
        result_arr = _warp_with_mask(result_arr, src_pts, dst_pts, roi)

    return Image.fromarray(result_arr)


# ── LAB 일괄 보정 헬퍼 (색 공간 변환 1회) ──


# 계조를 접어 넣는 무릎의 폭 (0~255 기준).
#
# 하드 클립은 범위를 넘친 값을 전부 한 값에 붙여 버린다. L 채널에서는 밝은
# 부분이 평평한 회색 판이 되고, a/b 채널에서는 한 채널만 붙어 색상이 돌아간다.
# 실측: contrast +0.15만으로도 사진에 따라 화소의 10~37%가 새로 클립됐고,
# 채도가 최대에 붙은 화소가 10%p 늘어난 경우가 있었다.
_SOFT_KNEE = 24.0


def _soft_limit(x: np.ndarray, knee: float = _SOFT_KNEE) -> np.ndarray:
    """0~255를 넘으려는 값을 끝에서 접어 넣는다 (하드 클립 대신).

    범위를 벗어난 값이 없으면 아무것도 하지 않는다. 무조건 무릎을 적용하면
    원래 255였던 순백이 249로 내려가 흰 배경이 회색으로 보인다.
    넘친 값이 있을 때만, 넘친 폭에 맞춰 위쪽 [255-knee, 최댓값]을
    [255-knee, 255]로 부드럽게 눌러 담는다. 순서는 유지된다(단조).
    """
    y = np.array(x, dtype=np.float32, copy=True)

    peak = float(y.max()) if y.size else 0.0
    if peak > 255.0:
        edge = 255.0 - knee
        up = y > edge
        t = (y[up] - edge) / (peak - edge)
        y[up] = edge + knee * (1.0 - (1.0 - t) ** 2)

    floor = float(y.min()) if y.size else 0.0
    if floor < 0.0:
        edge = knee
        dn = y < edge
        t = (edge - y[dn]) / (edge - floor)
        y[dn] = edge - knee * (1.0 - (1.0 - t) ** 2)

    return np.clip(y, 0.0, 255.0)


def _apply_lab_adjustments(
    img: Image.Image,
    highlights: float = 0.0,
    shadows: float = 0.0,
    tone_curve_preset: str = "linear",
    tone_curve_strength: float = 0.0,
    tone_curve_points: list | None = None,
    brightness: float = 0.0,
    contrast: float = 0.0,
    clarity: float = 0.0,
    dehaze: float = 0.0,
    temperature: float = 0.0,
    saturation: float = 0.0,
) -> Image.Image:
    """LAB 기반 색감 보정을 한 번의 색 공간 변환 안에서 일괄 처리한다.

    기존에 개별 함수가 각각 PIL→uint8→BGR→LAB→float32 변환을 반복하면서
    발생하던 양자화 손실(float32→uint8→float32 반복)을 제거한다.

    처리 순서 (apply_all_transforms의 기존 순서 유지):
      highlights → shadows → tone_curve → brightness → contrast
      → clarity → dehaze → temperature → saturation

    dehaze의 양수(Dark Channel Prior)는 BGR 공간이 필요하므로,
    BGR 단계에서 먼저 처리한 뒤 LAB로 진입한다:
      1단계: dehaze 양수 → BGR에서 Dark Channel Prior 적용
      2단계: BGR → LAB 변환 (1회)
      3단계: highlights ~ clarity (L 채널)
      4단계: dehaze 음수 + temperature + saturation (LAB)
      5단계: LAB → RGB 역변환 (1회)
    """
    # 모든 조정이 불필요하면 즉시 반환
    if (abs(highlights) < 0.01 and abs(shadows) < 0.01
            and (tone_curve_strength < 0.01
                 or (tone_curve_points is None
                     and (tone_curve_preset == "linear"
                          or tone_curve_preset not in TONE_CURVE_PRESETS)))
            and abs(brightness) < 0.01 and abs(contrast) < 0.01
            and abs(clarity) < 0.01 and abs(dehaze) < 0.01
            and abs(temperature) < 0.01 and abs(saturation) < 0.01):
        return img

    arr = np.array(img, dtype=np.uint8)
    arr_bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)

    # ── dehaze 양수(Dark Channel Prior)는 BGR에서 먼저 처리 ──
    if dehaze > 0.01:
        b, g, r = cv2.split(arr_bgr)
        dark = np.minimum(np.minimum(b, g), r).astype(np.float32)
        ksize_dh = max(7, int(min(arr_bgr.shape[:2]) * 0.01)) | 1
        dark = cv2.erode(dark, np.ones((ksize_dh, ksize_dh), np.uint8))

        num_pixels = dark.size
        n_bright = max(1, int(num_pixels * 0.001))
        flat_dark = dark.flatten()
        indices = np.argpartition(flat_dark, -n_bright)[-n_bright:]
        arr_f = arr_bgr.astype(np.float32)
        flat_img = arr_f.reshape(-1, 3)
        atm = flat_img[indices].mean(axis=0)
        atm = np.clip(atm, 1.0, 255.0)

        norm = arr_f / atm[np.newaxis, np.newaxis, :]
        dark_norm = np.min(norm, axis=2)
        dark_norm_blur = cv2.GaussianBlur(
            dark_norm, (ksize_dh * 2 + 1, ksize_dh * 2 + 1), 0
        )
        omega = dehaze * 0.95
        transmission = 1.0 - omega * dark_norm_blur
        transmission = np.clip(transmission, 0.1, 1.0)

        t = transmission[:, :, np.newaxis]
        result_f = (arr_f - atm) / t + atm
        arr_bgr = np.clip(result_f, 0, 255).astype(np.uint8)

    # ── BGR → LAB (float32) 1회 변환 ──
    lab = cv2.cvtColor(arr_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)

    l_ch = lab[:, :, 0]
    a_ch = lab[:, :, 1]
    b_ch = lab[:, :, 2]

    # ── 1. Highlights (L 채널) ──
    if abs(highlights) >= 0.01:
        hl_mask = np.clip((l_ch - 128.0) / 128.0, 0.0, 1.0)
        l_ch = l_ch + highlights * 60.0 * hl_mask

    # ── 2. Shadows (L 채널) ──
    if abs(shadows) >= 0.01:
        sh_mask = np.clip((128.0 - l_ch) / 128.0, 0.0, 1.0)
        l_ch = l_ch + shadows * 60.0 * sh_mask

    # ── 3. Tone Curve (L 채널 LUT) ──
    if tone_curve_strength >= 0.01:
        # 명시적 제어점이 오면 프리셋보다 우선한다 — 레퍼런스 사진에서
        # 뽑아낸 곡선을 그대로 태우기 위한 통로다.
        tc_points = tone_curve_points or TONE_CURVE_PRESETS.get(tone_curve_preset)
        if tc_points is not None and (tone_curve_points or tone_curve_preset != "linear"):
            x_pts = np.array([p[0] for p in tc_points], dtype=np.float64)
            y_pts = np.array([p[1] for p in tc_points], dtype=np.float64)
            x_256 = np.linspace(0.0, 1.0, 256)
            curve = np.interp(x_256, x_pts, y_pts)
            identity = x_256
            blended_curve = identity * (1.0 - tone_curve_strength) + curve * tone_curve_strength
            # float32 LUT (0~255)
            tc_lut = np.clip(blended_curve * 255.0, 0, 255).astype(np.float32)
            # l_ch를 uint8 인덱스로 변환하여 LUT 적용, 결과는 float32 유지
            l_idx = np.clip(l_ch, 0, 255).astype(np.uint8)
            l_ch = tc_lut[l_idx]

    # ── 4. Brightness (L 채널 감마 보정) ──
    if abs(brightness) >= 0.01:
        l_norm = np.clip(l_ch / 255.0, 0.0, 1.0)
        gamma = 1.0 / (1.0 + brightness) if brightness >= 0 else 1.0 - brightness * 1.5
        gamma = max(0.2, min(5.0, gamma))
        l_ch = np.power(l_norm, gamma) * 255.0

    # ── 5. Contrast (L 채널) ──
    if abs(contrast) >= 0.01:
        mean_l = np.mean(l_ch)
        l_ch = mean_l + (l_ch - mean_l) * (1.0 + contrast)

    # ── 6. Clarity (L 채널 로컬 대비) ──
    if abs(clarity) >= 0.01:
        h_img, w_img = l_ch.shape[:2]
        ksize_cl = max(31, int(min(h_img, w_img) * 0.05)) | 1
        l_blur = cv2.GaussianBlur(l_ch, (ksize_cl, ksize_cl), 0)
        detail = l_ch - l_blur
        midtone_mask = 1.0 - np.abs(l_ch - 128.0) / 128.0
        midtone_mask = np.clip(midtone_mask * 1.5, 0.0, 1.0)
        l_ch = l_ch + detail * clarity * 1.5 * midtone_mask

    # ── 7. Dehaze 음수 (안개 추가 — LAB 기반) ──
    if dehaze < -0.01:
        haze_amount = abs(dehaze)
        l_ch = l_ch * (1.0 - haze_amount * 0.4) + 200.0 * haze_amount * 0.4
        a_ch = a_ch * (1.0 - haze_amount * 0.3) + 128.0 * haze_amount * 0.3
        b_ch = b_ch * (1.0 - haze_amount * 0.3) + 128.0 * haze_amount * 0.3

    # ── 8. Temperature (B 채널 + A 채널 미세 조정) ──
    if abs(temperature) >= 0.01:
        shift = temperature * 15.0
        b_ch = b_ch + shift
        a_ch = a_ch + shift * 0.3

    # ── 9. Saturation (A, B 채널) ──
    if abs(saturation) >= 0.01:
        a_ch = 128.0 + (a_ch - 128.0) * (1.0 + saturation)
        b_ch = 128.0 + (b_ch - 128.0) * (1.0 + saturation)

    # ── LAB → BGR → RGB 1회 역변환 ──
    # 하드 클립 대신 끝을 접는다 ([_soft_limit] 참고).
    lab[:, :, 0] = _soft_limit(l_ch)
    lab[:, :, 1] = _soft_limit(a_ch)
    lab[:, :, 2] = _soft_limit(b_ch)

    bgr_out = cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2BGR)
    rgb_out = cv2.cvtColor(bgr_out, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb_out)


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
    auto_wb: float = 0.0,
    denoise: float = 0.0,
    background_blur: float = 0.0,
    tone_curve_points: list | None = None,
    preview: bool = False,
    cache: MediaPipeCache | None = None,
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

    cache가 제공되면 MediaPipe 모델/결과를 재사용한다.
    제공되지 않으면 내부에서 자동 생성하여 요청 내 중복 호출을 제거한다.
    """
    # ── 촬영 결함 교정: 색 보정보다 먼저 ──
    # 화이트밸런스를 먼저 잡아야 프로필의 색온도가 "중립 위에 얹는 취향"이 되고,
    # 노이즈를 먼저 지워야 뒤따르는 쉐도우 리프팅이 그것을 증폭시키지 않는다.
    if auto_wb >= 0.01:
        img = apply_auto_white_balance(img, auto_wb)
    if denoise >= 0.01:
        img = apply_denoise(img, denoise)

    # MediaPipe가 필요한 변형이 있는지 판단
    # reshape은 preview 모드에서도 적용하므로 preview 조건 제거
    # blemish_removal만 preview 시 스킵 (비용이 높으므로)
    needs_mp = (
        face_slim >= 0.01
        or jaw_sharpen >= 0.01
        or eye_enlarge >= 0.01
        or leg_stretch >= 0.01
        or abs(shoulder_width) >= 0.01
        or waist_slim >= 0.01
        or (not preview and blemish_removal >= 0.01)
        # 스무딩도 피부 마스크가 필요하다. preview에서도 켜 두어야
        # 미리보기와 최종 결과가 갈리지 않는다 (얼굴 감지는 캐시로 1회).
        or skin_smoothing >= 0.01
    )

    # 배경 흐림도 얼굴 감지가 필요하다
    needs_mp = needs_mp or background_blur >= 0.01

    if needs_mp and cache is None:
        # 캐시가 없으면 자동 생성하여 함수 내에서 모델/결과 재사용
        with MediaPipeCache() as auto_cache:
            result = _apply_all_transforms_impl(
                img, brightness, contrast, clarity, dehaze, highlights, shadows,
                saturation, temperature, blemish_removal, skin_smoothing, vignette,
                sharpness, grain, tone_curve_preset, tone_curve_strength,
                split_shadow_hue, split_shadow_strength, split_highlight_hue,
                split_highlight_strength, hsl_adjust, face_slim, jaw_sharpen,
                eye_enlarge, leg_stretch, shoulder_width, waist_slim, preview,
                auto_cache, tone_curve_points=tone_curve_points,
            )
            # 배경 흐림은 색 보정이 끝난 뒤 — 그래야 선명도 보정이
            # 흐려 둔 배경을 다시 살려내지 않는다.
            if background_blur >= 0.01:
                result = apply_background_blur(result, background_blur, auto_cache)
            return result
    else:
        result = _apply_all_transforms_impl(
            img, brightness, contrast, clarity, dehaze, highlights, shadows,
            saturation, temperature, blemish_removal, skin_smoothing, vignette,
            sharpness, grain, tone_curve_preset, tone_curve_strength,
            split_shadow_hue, split_shadow_strength, split_highlight_hue,
            split_highlight_strength, hsl_adjust, face_slim, jaw_sharpen,
            eye_enlarge, leg_stretch, shoulder_width, waist_slim, preview,
            cache, tone_curve_points=tone_curve_points,
        )
        if background_blur >= 0.01:
            result = apply_background_blur(result, background_blur, cache)
        return result


def _apply_all_transforms_impl(
    img: Image.Image,
    brightness: float,
    contrast: float,
    clarity: float,
    dehaze: float,
    highlights: float,
    shadows: float,
    saturation: float,
    temperature: float,
    blemish_removal: float,
    skin_smoothing: float,
    vignette: float,
    sharpness: float,
    grain: float,
    tone_curve_preset: str,
    tone_curve_strength: float,
    split_shadow_hue: float,
    split_shadow_strength: float,
    split_highlight_hue: float,
    split_highlight_strength: float,
    hsl_adjust: dict[str, dict[str, float]] | None,
    face_slim: float,
    jaw_sharpen: float,
    eye_enlarge: float,
    leg_stretch: float,
    shoulder_width: float,
    waist_slim: float,
    preview: bool,
    cache: MediaPipeCache | None,
    tone_curve_points: list | None = None,
) -> Image.Image:
    """apply_all_transforms의 내부 구현. cache를 전달받아 MediaPipe 재사용.

    파이프라인 단계:
      Phase 1: 기하학적 변형 (RGB/PIL 기반 — 기존대로)
      Phase 2: LAB 색감 보정 (float32 LAB에서 한 번에 처리)
      Phase 3: HSL 선택적 색상 (HSV 기반)
      Phase 4: 스플릿 토닝 (LAB 기반 — 별도)
      Phase 5: 텍스처/디테일 (RGB/PIL 기반)
    """
    result = img

    # ── Phase 1: 기하학적 변형 (RGB/PIL 기반) ──
    # preview 모드에서도 reshape 파라미터가 0이 아니면 적용한다.
    # (슬라이더 미리보기에서 얼굴/체형 보정 결과를 즉시 확인할 수 있도록)
    needs_face = face_slim >= 0.01 or jaw_sharpen >= 0.01 or eye_enlarge >= 0.01
    needs_body = leg_stretch >= 0.01 or abs(shoulder_width) >= 0.01 or waist_slim >= 0.01
    if needs_face:
        result = apply_face_reshape(result, face_slim, jaw_sharpen, eye_enlarge, cache=cache)
    if needs_body:
        result = apply_body_reshape(result, leg_stretch, shoulder_width, waist_slim, cache=cache)

    # ── Phase 2: LAB 색감 보정 (float32 LAB에서 한 번에 처리) ──
    # highlights, shadows, tone_curve, brightness, contrast, clarity,
    # dehaze, temperature, saturation을 1회 색 공간 변환으로 통합
    result = _apply_lab_adjustments(
        result,
        highlights=highlights,
        shadows=shadows,
        tone_curve_preset=tone_curve_preset,
        tone_curve_strength=tone_curve_strength,
        tone_curve_points=tone_curve_points,
        brightness=brightness,
        contrast=contrast,
        clarity=clarity,
        dehaze=dehaze,
        temperature=temperature,
        saturation=saturation,
    )

    # ── Phase 3: HSL 선택적 색상 (HSV 기반) ──
    result = apply_hsl_adjust(result, hsl_adjust)

    # ── Phase 4: 스플릿 토닝 (LAB 기반 — 별도) ──
    result = apply_split_toning(
        result, split_shadow_hue, split_shadow_strength,
        split_highlight_hue, split_highlight_strength,
    )

    # ── Phase 5: 텍스처/디테일 (RGB/PIL 기반) ──
    if not preview:
        # 잡티 제거는 MediaPipe 재호출이 필요하므로 미리보기에서 스킵
        result = apply_blemish_removal(result, blemish_removal, cache=cache)

    result = apply_skin_smoothing(result, skin_smoothing, cache=cache)
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
        "auto_wb": 0.0,
        "denoise": 0.0,
        "background_blur": 0.0,
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
            "tone_curve_points": None,
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
        if key in ("blemish_removal", "skin_smoothing", "auto_wb", "denoise",
                   "background_blur"):
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
        raw_points = tone_curve.get("points")
        if isinstance(raw_points, list) and len(raw_points) >= 2:
            try:
                pts = [(float(x), float(y)) for x, y in raw_points]
                params["tone_curve_points"] = sorted(pts)
            except (TypeError, ValueError):
                params["tone_curve_points"] = None
        else:
            params["tone_curve_points"] = None
    else:
        params["tone_curve_preset"] = "linear"
        params["tone_curve_strength"] = 0.0
        params["tone_curve_points"] = None

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


