"""이미지 변형 엔진 — Pillow + OpenCV 기반 순수 함수 모듈."""

from __future__ import annotations

import base64
import hashlib
import io
import logging
import math
import os
import tempfile
import threading
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
    # ── 2026 트렌드 커브 ──
    # 소프트 필름: 검정을 살짝 띄우고(바랜 바닥) 흰색 끝을 부드럽게 접는다.
    # S커브와 달리 중간톤 기울기가 1보다 조금 낮아 대비가 순해진다.
    "soft_film": [(0, 0.07), (0.25, 0.255), (0.5, 0.505), (0.75, 0.755), (1, 0.955)],
    # 정면 플래시 스냅: 중간~밝은 톤을 세워 피사체가 튀어나오고, 바닥은 그대로 깊다.
    "flash": [(0, 0), (0.25, 0.20), (0.5, 0.53), (0.75, 0.84), (1, 1)],
    # 소프트 파스텔: 바닥을 크게 띄우고 전체를 밝은 쪽으로 눌러 담는다 (저대비).
    "pastel": [(0, 0.10), (0.25, 0.32), (0.5, 0.575), (0.75, 0.795), (1, 0.955)],
    # 흑백 그레인: 필름 인화 같은 적당한 대비, 검정·흰색 끝만 살짝 접는다.
    "bw": [(0, 0.035), (0.25, 0.21), (0.5, 0.50), (0.75, 0.80), (1, 0.975)],
}

# saturation이 이 값 이하면 "흑백 변환"으로 본다 (채도를 끝까지 뺀 것).
_MONO_SATURATION = -0.99
# 흑백 변환의 색→밝기 믹스 (LAB a·b 편차 → L 가산).
# 채널 평균으로 뽑으면 피부가 칙칙해진다. 흑백 필름에 옅은 주황 필터를 끼운 것처럼
# 따뜻한 색(피부·입술)은 살짝 밝게, 파랑(하늘)은 살짝 어둡게 옮긴다.
_MONO_MIX_A = 0.22
_MONO_MIX_B = 0.14
_MONO_MIX_LIMIT = 18.0


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


# ── MediaPipe 모델 인스턴스 풀 ──
#
# 예전에는 요청마다 모델을 새로 만들고 요청 끝에 close()했다. 만드는 건
# 10~50ms로 싸지만 close()가 모델 하나에 0.25~0.35초씩 걸린다(내부 그래프
# 종료 대기 — 실측, M3 Pro). 인물 요청은 얼굴·포즈·분할 모델을 다 쓰므로
# 요청마다 0.6~0.9초를 정리에만 썼다.
#
# 그래서 다 쓴 인스턴스를 닫지 않고 풀에 돌려 두었다가 다음 요청이 다시 쓴다.
# MediaPipe 태스크 객체는 스레드 안전하지 않으므로 한 인스턴스는 한 번에 한
# 요청(MediaPipeCache)만 빌려 쓴다. 스레드 로컬로 두지 않는 이유: 서버는
# 동기 엔드포인트를 anyio 스레드풀(최대 40개)에서 돌리므로, 스레드마다 모델을
# 붙이면 동시 처리 수(GAMDO_MAX_CONCURRENT)와 무관하게 인스턴스가 스레드 수만큼
# 쌓인다. 빌려 쓰는 풀이면 동시에 쓰이는 수 이상으로는 늘지 않는다.
#
# IMAGE 모드 태스크는 호출 간 상태가 없어 재사용해도 결과가 같다
# (tests/test_mediapipe_pool.py에서 확인).

# 종류별로 보관할 유휴 인스턴스 수. 동시 처리 상한과 같게 둔다 — 그 이상은
# 동시에 쓰일 일이 없다. 넘치는 인스턴스는 닫는다.
_MP_POOL_MAX_IDLE = max(1, int(os.environ.get("GAMDO_MAX_CONCURRENT", "3")))

_mp_pool_lock = threading.Lock()
_mp_pool: dict[str, list[Any]] = {"face": [], "pose": [], "person": []}


def _mp_pool_acquire(kind: str) -> Any | None:
    """풀에서 유휴 인스턴스를 하나 꺼낸다. 없으면 None."""
    with _mp_pool_lock:
        idle = _mp_pool[kind]
        return idle.pop() if idle else None


def _mp_close_quietly(instance: Any) -> None:
    try:
        instance.close()
    except Exception:
        pass


def _mp_pool_release(kind: str, instance: Any) -> None:
    """다 쓴 인스턴스를 풀에 돌려준다. 풀이 차 있으면 닫는다."""
    with _mp_pool_lock:
        idle = _mp_pool[kind]
        if len(idle) < _MP_POOL_MAX_IDLE:
            idle.append(instance)
            return
    _mp_close_quietly(instance)


def _mp_pool_clear() -> None:
    """풀의 유휴 인스턴스를 모두 닫는다 (테스트·종료용)."""
    with _mp_pool_lock:
        items = [inst for lst in _mp_pool.values() for inst in lst]
        for lst in _mp_pool.values():
            lst.clear()
    for inst in items:
        _mp_close_quietly(inst)


# ── 요청 스코프 MediaPipe 캐시 ──


class MediaPipeCache:
    """요청 단위로 MediaPipe 모델 인스턴스 + 랜드마크 결과를 캐시한다.

    사용법::

        with MediaPipeCache() as cache:
            face_results = cache.get_face_landmarks(arr_rgb)
            pose_results = cache.get_pose_landmarks(arr_rgb)

    - 모델 인스턴스는 첫 호출 시 풀에서 빌리고(없으면 생성) 컨텍스트 종료까지 재사용
    - 동일 이미지(바이트 해시 기준)에 대한 감지 결과를 캐시하여 중복 호출 제거
    - 컨텍스트 종료 시 인스턴스를 풀에 돌려준다 (닫는 비용이 커서 닫지 않는다)
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
        """빌린 모델 인스턴스를 풀에 돌려주고 결과 캐시를 비운다."""
        if self._face_landmarker is not None:
            _mp_pool_release("face", self._face_landmarker)
            self._face_landmarker = None
        if self._pose_landmarker is not None:
            _mp_pool_release("pose", self._pose_landmarker)
            self._pose_landmarker = None
        if self._person_segmenter is not None:
            _mp_pool_release("person", self._person_segmenter)
            self._person_segmenter = None
        self._face_results_cache.clear()
        self._pose_results_cache.clear()
        self._person_mask_cache.clear()

    def _close_person_segmenter(self) -> None:
        """분할기 인스턴스를 닫고 슬롯을 비운다 (풀에 돌려주지 않는다).

        잘못된 입력을 한 번 먹은 인스턴스는 다음 호출에서 영구히 멈춘다
        (내부 그래프가 에러 상태로 남는다). 그래서 예외가 나면 재사용하지 않고
        버리고 다시 만든다.
        """
        if self._person_segmenter is not None:
            _mp_close_quietly(self._person_segmenter)
            self._person_segmenter = None

    def _discard_landmarker(self, kind: str) -> None:
        """감지 중 예외가 난 랜드마커를 버린다 — 분할기와 같은 이유로 풀에 돌려주지 않는다."""
        attr = "_face_landmarker" if kind == "face" else "_pose_landmarker"
        inst = getattr(self, attr)
        if inst is not None:
            _mp_close_quietly(inst)
            setattr(self, attr, None)

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
            self._face_landmarker = _mp_pool_acquire("face")
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
            self._pose_landmarker = _mp_pool_acquire("pose")
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
            self._person_segmenter = _mp_pool_acquire("person")
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
        try:
            results = landmarker.detect(mp_image)
        except Exception:
            self._discard_landmarker("face")
            raise

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
        try:
            results = landmarker.detect(mp_image)
        except Exception:
            self._discard_landmarker("pose")
            raise

        if not results.pose_landmarks:
            self._pose_results_cache[key] = None
            return None

        self._pose_results_cache[key] = results
        return results


# ── Base64 ↔ PIL Image 변환 ──


# 디코딩 전에 거절할 픽셀 수. 헤더만 읽고 판단하므로 압축 폭탄(작은 파일이
# 수억 화소로 풀리는 PNG 등)을 메모리에 올리기 전에 막는다. Pillow 기본 경고선
# (약 89MP)보다 낮게 둔다 — 파이프라인은 한 장에 원본의 수십 배를 쓴다.
_MAX_DECODE_PIXELS = 50_000_000

# 처리 해상도 상한. 앱(flutter_image_compress minWidth/minHeight=2560)은 짧은 변을
# 2560 이하로 맞춰 보내고 긴 변은 비율대로 둔다 (4:3 → 3413x2560).
# 짧은 변 상한은 그 계약과 같게 두어 앱 사진은 건드리지 않는다.
# 픽셀 상한은 2560 기준 약 2.4:1 비율까지 원본 그대로 통과시키고,
# 그보다 긴 파노라마만 줄인다 (2560px 한 장 피크 ~1.1GB, 동시 3장).
_MAX_SHORT_EDGE = 2560
_MAX_PROCESS_PIXELS = 16_000_000


def decode_base64_image(b64: str) -> Image.Image:
    """Base64 문자열을 PIL Image로 디코딩.

    앱 밖에서 들어온 요청은 해상도 제한이 없으므로 서버에서도 막는다.
    """
    data = base64.b64decode(b64)
    img = Image.open(io.BytesIO(data))
    w, h = img.size
    if w <= 0 or h <= 0 or w * h > _MAX_DECODE_PIXELS:
        raise ValueError(f"처리할 수 없는 이미지 크기입니다 ({w}x{h})")
    img = img.convert("RGB")

    scale = min(1.0, _MAX_SHORT_EDGE / min(w, h), (_MAX_PROCESS_PIXELS / (w * h)) ** 0.5)
    if scale < 1.0:
        new_size = (max(1, int(w * scale)), max(1, int(h * scale)))
        log.info("decode: %dx%d → %dx%d (해상도 상한)", w, h, *new_size)
        img = img.resize(new_size, Image.LANCZOS)
    return img


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


def _alpha_blend(fg: np.ndarray, bg: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    """fg * alpha + bg * (1 - alpha)를 0~255로 잘라 uint8로 돌려준다.

    fg, bg: (H, W, C) uint8, alpha: (H, W) float32.
    numpy로 한 번에 쓰면 (H, W, 1) 알파를 브로드캐스트하며 전체 크기 float32 임시
    배열이 여러 개 생긴다(3413x2560에서 개당 100MB). 행 묶음 단위로 cv2.multiply를
    쓰면 같은 float32 곱·합이라 결과가 비트 단위로 같고, 임시 메모리는 묶음
    크기로 줄며 2배 이상 빠르다 (tests/test_portrait_roi.py).
    """
    h = fg.shape[0]
    channels = fg.shape[2]
    out = np.empty(fg.shape, dtype=np.uint8)
    step = 256
    for y in range(0, h, step):
        a = np.ascontiguousarray(alpha[y:y + step], dtype=np.float32)
        a3 = cv2.merge([a] * channels)
        acc = cv2.multiply(fg[y:y + step], a3, dtype=cv2.CV_32F)
        np.subtract(1.0, a3, out=a3)
        acc += cv2.multiply(bg[y:y + step], a3, dtype=cv2.CV_32F)
        np.clip(acc, 0, 255, out=acc)
        out[y:y + step] = acc
    return out


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

    return _alpha_blend(arr, blurred, alpha)


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
    """피부 보정 (주파수 분리). intensity: 0.0 ~ 1.0.

    예전에는 양방향 필터 결과를 intensity만큼 섞었다. 모공·주근깨 같은 질감이
    얼룩과 같은 비율로 지워져 강도를 조금만 올려도 밀랍처럼 보였다.

    이제 피부를 세 층으로 나눈다 (가우시안 차이, 피부 화소만으로 평균).
      - 저주파 (얼굴 폭의 4.5% 이상): 얼굴 입체감·조명 — 그대로 둔다.
      - 중주파 (1%~4.5%): 붉은 기·얼룩·울퉁불퉁함 — intensity만큼 줄인다.
        진폭이 큰 성분(콧방울·팔자 음영처럼 구조적인 그림자)은 덜 줄인다.
      - 고주파 (1% 미만): 모공·주근깨·잔털 — 대부분 남긴다
        (intensity 1.0에서도 65%, 기본값 0.2대에서는 90% 이상).

    마스크는 [_get_skin_faces]의 피부 확정 영역이고, 경계는 안쪽으로 물려
    부드럽게 푼다 — 머리카락·눈썹·입술에는 닿지 않는다.
    얼굴 미감지·너무 작은 얼굴은 원본을 그대로 반환한다.
    """
    if intensity < 0.01:
        return img

    arr = np.array(img)
    faces = _get_skin_faces(arr, cache=cache)
    if not faces:
        return img

    changed = False
    for face in faces:
        changed |= _smooth_face(arr, face, min(1.0, intensity))
    return Image.fromarray(arr) if changed else img


# 주파수 분리 스무딩 계수 (얼굴 폭 대비 / intensity 1.0 기준).
_SMOOTH_FINE_SIGMA = 0.010    # 고주파/중주파 경계
_SMOOTH_BASE_SIGMA = 0.045    # 중주파/저주파 경계
_SMOOTH_MID_REDUCE = 1.2      # 중주파(얼룩) 감쇠 = intensity × 이 값 (최대 1)
_SMOOTH_FINE_REDUCE = 0.35    # 고주파(질감) 최대 감쇠 — 1.0에서도 65%는 남긴다
_SMOOTH_MID_KNEE = 12.0       # 이보다 진폭이 큰 중주파는 구조(음영)로 보고 덜 줄인다


def _masked_blur(img: np.ndarray, weights: np.ndarray, sigma: float) -> np.ndarray:
    """가중치(피부=1)가 있는 화소만으로 평균한 가우시안 블러.

    그냥 흐리면 피부 가장자리에 머리카락·눈썹 색이 섞여 들어와 경계에 후광이 생긴다.
    시그마가 크면 줄여서 흐린 뒤 되돌린다 — 결과가 저주파라 차이가 없고 훨씬 빠르다.
    """
    h, w = img.shape[:2]
    num = img * weights[:, :, np.newaxis]
    scale = 1.0 if sigma <= 3.0 else 2.5 / sigma
    if scale < 1.0:
        sw, sh = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
        num = cv2.resize(num, (sw, sh), interpolation=cv2.INTER_AREA)
        den = cv2.resize(weights, (sw, sh), interpolation=cv2.INTER_AREA)
    else:
        den = weights
    s = sigma * scale
    num = cv2.GaussianBlur(num, (0, 0), s)
    den = cv2.GaussianBlur(den, (0, 0), s)
    out = num / np.maximum(den, 1e-3)[:, :, np.newaxis]
    if scale < 1.0:
        out = cv2.resize(out, (w, h), interpolation=cv2.INTER_LINEAR)
    return out


def _skin_alpha(face: "_SkinFace") -> np.ndarray:
    """피부 확정 영역을 블렌딩 알파(0~1, float32)로 푼다.

    먼저 안쪽으로 2.5σ 깎고 σ로 흐린다 — 알파의 꼬리가 확정 영역 밖으로 거의
    나가지 않는다(1% 미만). 마지막에 얼굴 윤곽으로 한 번 더 잘라 둔다.
    """
    sigma = max(1.0, face.face_w * 0.012)
    grow = max(1, int(round(sigma * 2.5)))
    core = _shrink(face.core, grow)
    alpha = cv2.GaussianBlur(core.astype(np.float32) / 255.0, (0, 0), sigma)
    alpha *= face.oval.astype(np.float32) / 255.0
    return alpha


def _smooth_face(arr: np.ndarray, face: "_SkinFace", intensity: float) -> bool:
    """얼굴 하나의 피부를 제자리에서 주파수 분리로 정리한다. 바뀌었으면 True."""
    if cv2.countNonZero(face.core) == 0:
        return False
    alpha = _skin_alpha(face)
    ys, xs = np.nonzero(alpha > 1e-3)
    if ys.size == 0:
        return False
    # 알파가 닿는 상자 + 블러 반경만큼만 계산한다
    fw = face.face_w
    s_fine = max(0.8, fw * _SMOOTH_FINE_SIGMA)
    s_base = max(3.0, fw * _SMOOTH_BASE_SIGMA)
    pad = int(math.ceil(s_base * 3)) + 2
    rh, rw = face.core.shape[:2]
    by1, by2 = max(0, ys.min() - pad), min(rh, ys.max() + 1 + pad)
    bx1, bx2 = max(0, xs.min() - pad), min(rw, xs.max() + 1 + pad)

    y0, x0 = face.y0 + by1, face.x0 + bx1
    y1, x1 = face.y0 + by2, face.x0 + bx2
    roi = arr[y0:y1, x0:x1].astype(np.float32)
    weights = face.core[by1:by2, bx1:bx2].astype(np.float32) / 255.0
    a = alpha[by1:by2, bx1:bx2][:, :, np.newaxis]

    low_fine = _masked_blur(roi, weights, s_fine)
    low_base = _masked_blur(roi, weights, s_base)
    mid = low_fine - low_base
    fine = roi - low_fine

    amp = np.abs(mid).mean(axis=2, keepdims=True)
    mid_gain = min(1.0, _SMOOTH_MID_REDUCE * intensity) / (1.0 + (amp / _SMOOTH_MID_KNEE) ** 2)
    fine_gain = _SMOOTH_FINE_REDUCE * intensity

    out = roi - a * (mid_gain * mid + fine_gain * fine)
    arr[y0:y1, x0:x1] = np.clip(out + 0.5, 0, 255).astype(np.uint8)
    return True


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


# 콧방울·콧구멍 — 자연스러운 음영이 잡티로 오감지된다 (잡티 탐지에서만 뺀다)
_NOSE_WINGS = [1, 2, 98, 327, 64, 294, 48, 278, 4]

# 이 폭(px)보다 작은 얼굴은 질감 보정을 하지 않는다. 피부가 몇십 화소뿐이라
# 결과가 보이지 않고, 랜드마크 오차가 마스크 폭과 비슷해 머리카락을 건드리기 쉽다.
_SKIN_MIN_FACE_PX = 64


class _SkinFace:
    """얼굴 하나의 질감 보정용 마스크 (얼굴 상자 ROI 좌표)."""

    __slots__ = ("x0", "y0", "x1", "y1", "core", "oval", "guard", "face_w")

    def __init__(self, x0, y0, x1, y1, core, oval, guard, face_w):
        self.x0, self.y0, self.x1, self.y1 = x0, y0, x1, y1
        self.core = core      # uint8 0/255 — 피부로 확정된 영역
        self.oval = oval      # uint8 0/255 — 얼굴 윤곽 (이 밖으로는 절대 나가지 않는다)
        self.guard = guard    # uint8 0/255 — 이목구비·콧방울 주변 (잡티 탐지 제외)
        self.face_w = face_w  # 얼굴 폭(px) — 모든 크기 기준


def _face_point_sets(
    img_rgb: np.ndarray,
    cache: MediaPipeCache | None = None,
) -> list | None:
    """얼굴마다 랜드마크 인덱스 → (x, y) 픽셀 좌표 함수를 돌려준다. 미감지 시 None."""
    model_path = face_model_path()
    if model_path is None:
        return None

    h, w = img_rgb.shape[:2]
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

    point_sets = []
    for face_lms in results.face_landmarks:
        def _idx_to_pt(idx: int, _lms=face_lms) -> tuple[int, int]:
            lm = _lms[idx]
            return int(lm.x * w), int(lm.y * h)
        point_sets.append(_idx_to_pt)
    return point_sets


def _get_skin_mask(
    img_rgb: np.ndarray,
    cache: MediaPipeCache | None = None,
    for_blemish: bool = False,
    mode: str = "texture",
) -> np.ndarray | None:
    """얼굴 마스크를 사진 크기로 돌려준다. 얼굴 미감지 시 None.

    - mode="tone": 영역별 톤 보정(detect_regions)용 — 얼굴 윤곽 전체.
    - mode="texture": 질감 보정이 실제로 건드리는 피부 확정 영역
      ([_get_skin_faces]와 같다. for_blemish면 잡티 탐지 영역 — 경계에서 더 물린다).
    다중 얼굴이면 모든 얼굴의 마스크를 합친다.
    """
    h, w = img_rgb.shape[:2]
    if mode == "tone":
        point_sets = _face_point_sets(img_rgb, cache=cache)
        if point_sets is None:
            return None
        mask = np.zeros((h, w), dtype=np.uint8)
        for pt in point_sets:
            mask = cv2.bitwise_or(mask, build_face_skin_mask(pt, h, w, mode="tone"))
        return mask

    faces = _get_skin_faces(img_rgb, cache=cache)
    if faces is None:
        return None
    mask = np.zeros((h, w), dtype=np.uint8)
    for face in faces:
        part = _blemish_search_mask(face) if for_blemish else face.core
        region = mask[face.y0:face.y1, face.x0:face.x1]
        np.maximum(region, part, out=region)
    return mask


def _get_skin_faces(
    img_rgb: np.ndarray,
    cache: MediaPipeCache | None = None,
) -> list[_SkinFace] | None:
    """질감 보정(스무딩·잡티)이 쓸 얼굴별 피부 마스크. 얼굴 미감지 시 None.

    얼굴 상자만 잘라 [refine_skin_mask]로 만든다. 너무 작은 얼굴은 뺀다.
    """
    point_sets = _face_point_sets(img_rgb, cache=cache)
    if point_sets is None:
        return None

    h, w = img_rgb.shape[:2]
    faces: list[_SkinFace] = []
    for pt in point_sets:
        oval_pts = np.array([pt(i) for i in _SKIN_FACE_OVAL], dtype=np.int32)
        face_w = float(abs(pt(454)[0] - pt(234)[0]))
        ox, oy, ow, oh = cv2.boundingRect(oval_pts)
        if min(face_w, ow) < _SKIN_MIN_FACE_PX:
            log.info("skin: 얼굴 폭 %dpx — 너무 작아 질감 보정 생략", int(face_w))
            continue
        pad = max(4, int(face_w * 0.05))
        x0, y0 = max(0, ox - pad), max(0, oy - pad)
        x1, y1 = min(w, ox + ow + pad), min(h, oy + oh + pad)
        if x1 - x0 < 8 or y1 - y0 < 8:
            continue

        def _roi_pt(idx: int, _pt=pt, _x0=x0, _y0=y0) -> tuple[int, int]:
            x, y = _pt(idx)
            return x - _x0, y - _y0

        core, oval, guard = refine_skin_mask(img_rgb[y0:y1, x0:x1], _roi_pt, face_w)
        if cv2.countNonZero(core) == 0:
            continue
        faces.append(_SkinFace(x0, y0, x1, y1, core, oval, guard, face_w))
    return faces


def _hull_mask(pt, ids, h: int, w: int) -> np.ndarray:
    m = np.zeros((h, w), dtype=np.uint8)
    hull = cv2.convexHull(np.array([pt(i) for i in ids], dtype=np.int32))
    cv2.fillPoly(m, [hull], 255)
    return m


def _grow(mask: np.ndarray, radius: float) -> np.ndarray:
    """원판 팽창과 같은 결과를 거리 변환으로 — 반지름이 커도 비용이 같다.

    얼굴 폭의 10%(큰 얼굴이면 100px 넘는 원판)로 cv2.dilate를 돌리면 수십 ms씩 걸린다.
    """
    if radius <= 0:
        return mask.copy()
    dist = cv2.distanceTransform(cv2.bitwise_not(mask), cv2.DIST_L2, 5)
    return ((dist <= radius) * 255).astype(np.uint8)


def _shrink(mask: np.ndarray, radius: float) -> np.ndarray:
    """원판 침식과 같은 결과를 거리 변환으로."""
    if radius <= 0:
        return mask.copy()
    dist = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
    return ((dist > radius) * 255).astype(np.uint8)


def _disk(radius: int) -> np.ndarray:
    radius = max(1, int(radius))
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (radius * 2 + 1,) * 2)


def refine_skin_mask(
    roi_rgb: np.ndarray,
    pt,
    face_w: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """랜드마크 얼굴 윤곽 ∩ 피부색 ∩ '얼굴 가운데서 이어지는 영역'.

    [build_face_skin_mask]의 윤곽은 이마 위쪽이 머리카락 속까지 올라간다
    (랜드마크 10번은 두개골 윤곽이라 앞머리가 있으면 그 위에 찍힌다).
    예전에는 이 윤곽을 그대로 피부로 보고 잡티를 찾아, 앞머리 끝·관자놀이의
    머리카락을 잡티로 판정해 피부색으로 덧칠했다.

    1. 피부색 모델: 눈 아래~입 사이(머리카락이 거의 없는 볼·코)에서 LAB 중앙값과
       편차를 잰다. 밝은 쪽 편차(하이라이트)는 관대하게, 어두운 쪽은 엄격하게.
    2. 피부가 아닌 화소가 얼굴 가운데(콧대)에서 바깥으로 뻗는 광선 위에 일정 길이
       (얼굴 폭 3%) 이상 쌓이면, 그 너머는 피부색이어도 뺀다 — 앞머리 사이로
       비치는 금발 하이라이트가 피부색과 거의 같아 색만으로는 못 거른다.
       눈·눈썹·입술·코는 광선을 막지 않는다 (이목구비 너머 이마·볼을 살린다).

    pt는 ROI 좌표를 돌려줘야 한다. 반환: (core, oval, guard) — ROI 크기 uint8 0/255.
    guard는 잡티 탐지 금지 구역(이목구비·콧방울 주변)이다.
    """
    h, w = roi_rgb.shape[:2]
    geo = build_face_skin_mask(pt, h, w, mode="texture")
    oval = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(oval, [np.array([pt(i) for i in _SKIN_FACE_OVAL], dtype=np.int32)], 255)
    # 잡티 탐지 금지 구역: 눈(눈물샘 쪽 붉은 살)·눈썹·입술·콧방울 주변
    # 눈꼬리는 속눈썹·아이라인·붉은 살이 랜드마크 밖으로 나와 있어 더 넓게 막는다.
    guard = _hull_mask(pt, _NOSE_WINGS, h, w)
    for region in (_LIPS, _LEFT_EYEBROW, _RIGHT_EYEBROW):
        guard |= _hull_mask(pt, region, h, w)
    guard = _grow(guard, face_w * 0.06)
    eyes = _hull_mask(pt, _LEFT_EYE, h, w) | _hull_mask(pt, _RIGHT_EYE, h, w)
    guard |= _grow(eyes, face_w * 0.10)
    if cv2.countNonZero(geo) == 0:
        return geo, oval, guard

    lab = cv2.cvtColor(roi_rgb.astype(np.float32) * (1.0 / 255.0), cv2.COLOR_RGB2Lab)

    # 1. 피부색 표본 — 눈 아래 ~ 입꼬리 위
    eye_y = (pt(145)[1] + pt(374)[1]) // 2
    mouth_y = (pt(61)[1] + pt(291)[1]) // 2
    sample = geo.copy()
    if mouth_y - eye_y > 4:
        sample[:max(0, eye_y)] = 0
        sample[max(0, mouth_y):] = 0
    if cv2.countNonZero(sample) < 50:
        sample = geo
    px = lab[sample > 0]
    med = np.median(px, axis=0)
    mad = np.median(np.abs(px - med), axis=0) * 1.4826
    spread = np.maximum(mad, np.array([6.0, 1.5, 1.5], dtype=np.float32))

    dl = (lab[:, :, 0] - med[0]) / spread[0]
    dl_w = np.where(dl < 0, 0.5, 0.12).astype(np.float32)
    da = (lab[:, :, 1] - med[1]) / spread[1]
    db = (lab[:, :, 2] - med[2]) / spread[2]
    dist_raw = np.sqrt(dl_w * dl * dl + da * da + db * db)
    # 주근깨·모공 한 점이 아니라 넓은 영역의 색으로 판정한다
    dist = cv2.GaussianBlur(dist_raw, (0, 0), max(1.0, face_w * 0.012))
    non_skin = dist > 3.0

    # 이마로 흘러내린 가는 머리카락은 넓게 흐린 색으로는 안 잡힌다. 좁게 흐린 색으로
    # 튀는 성분 중 크거나(머리카락 덩어리에 붙은 가닥) 길쭉한 것만 뺀다.
    # 주근깨·점은 작고 둥글어 남는다 — 이것까지 빼면 주근깨마다 보정 구멍이 난다.
    strand_mask = np.zeros((h, w), dtype=np.uint8)
    # 반사광(밝고 채도가 빠진 곳)은 가닥이 아니다 — 피부보다 어두운 곳만 본다.
    fine_sigma = max(0.7, face_w * 0.003)
    strands = (cv2.GaussianBlur(dist_raw, (0, 0), fine_sigma) > 4.0) & (
        cv2.GaussianBlur(dl, (0, 0), fine_sigma) < -0.5
    )
    strands = strands.astype(np.uint8) * 255
    s_count, s_labels, s_stats, _ = cv2.connectedComponentsWithStats(strands, connectivity=8)
    if s_count > 1:
        min_len = max(4.0, face_w * 0.03)
        excl = np.zeros(s_count, dtype=bool)
        long_side = np.maximum(s_stats[:, cv2.CC_STAT_WIDTH], s_stats[:, cv2.CC_STAT_HEIGHT])
        for i in np.nonzero(long_side >= min_len)[0]:
            if i == 0:
                continue
            if s_stats[i, cv2.CC_STAT_AREA] >= min_len * min_len:
                excl[i] = True
                continue
            x, y, bw, bh = s_stats[i, :4]
            m = cv2.moments((s_labels[y:y + bh, x:x + bw] == i).astype(np.uint8), binaryImage=True)
            if m["m00"] <= 0:
                continue
            mu20, mu02, mu11 = m["mu20"] / m["m00"], m["mu02"] / m["m00"], m["mu11"] / m["m00"]
            common = math.sqrt(max(0.0, ((mu20 - mu02) / 2) ** 2 + mu11 ** 2))
            l1 = (mu20 + mu02) / 2 + common
            l2 = max(1e-6, (mu20 + mu02) / 2 - common)
            excl[i] = math.sqrt(l1 / l2) > 3.0
        excl[0] = False
        strand_mask = cv2.dilate((excl[s_labels] * 255).astype(np.uint8), _disk(2))
        non_skin |= strand_mask > 0

    # 2. 광선 차단 — 이목구비와 얼굴 가운데는 중립
    features = np.zeros((h, w), dtype=np.uint8)
    for region in (_LEFT_EYE, _RIGHT_EYE, _LIPS, _LEFT_EYEBROW, _RIGHT_EYEBROW):
        features |= _hull_mask(pt, region, h, w)
    features = _grow(features, face_w * 0.04)
    central = _hull_mask(
        pt, _LEFT_EYE + _RIGHT_EYE + _LIPS + _LEFT_EYEBROW + _RIGHT_EYEBROW + _NOSE_WINGS + [6],
        h, w,
    )
    neutral = features | _grow(central, face_w * 0.03)
    blockers = (non_skin & (neutral == 0)).astype(np.uint8) * 255

    cx, cy = pt(6)  # 콧대 (두 눈 사이)
    cx = float(min(max(cx, 0), w - 1))
    cy = float(min(max(cy, 0), h - 1))
    radius = int(math.hypot(max(cx, w - cx), max(cy, h - cy))) + 2
    n_ang = int(np.clip(2 * math.pi * radius / 3, 360, 1440))
    polar = cv2.warpPolar(blockers, (radius, n_ang), (cx, cy), radius,
                          cv2.WARP_POLAR_LINEAR | cv2.INTER_NEAREST)
    run = np.cumsum(polar > 127, axis=1, dtype=np.int32)
    blocked = ((run > max(2.0, face_w * 0.03)) * 255).astype(np.uint8)
    blocked = cv2.warpPolar(blocked, (w, h), (cx, cy), radius,
                            cv2.WARP_POLAR_LINEAR | cv2.WARP_INVERSE_MAP | cv2.INTER_NEAREST)
    # 역변환은 광선 사이 틈을 남길 수 있다 — 살짝 넓혀 메운다
    blocked = cv2.dilate(blocked, _disk(max(1, face_w * 0.005)))

    allowed = geo & (blocked == 0).astype(np.uint8) * 255
    core = allowed & (~non_skin).astype(np.uint8) * 255
    # 하이라이트·작은 점 때문에 생긴 구멍은 메우고(허용 영역 안에서만), 부스러기는 버린다
    # 가닥 자리는 닫기로 다시 메우지 않는다
    allowed_close = allowed & cv2.bitwise_not(strand_mask)
    core = cv2.morphologyEx(core, cv2.MORPH_CLOSE, _disk(face_w * 0.02)) & allowed_close
    core = cv2.morphologyEx(core, cv2.MORPH_OPEN, _disk(max(1, face_w * 0.006)))

    # 볼·이마의 반사광(하이라이트)은 채도가 빠져 피부색 모델 밖으로 나간다.
    # 피부에 완전히 둘러싸인 '밝은' 구멍만 메운다 — 어두운 구멍(콧구멍·점)과
    # 허용 영역 경계에 닿은 구멍(머리카락)은 그대로 둔다.
    holes = cv2.bitwise_and(allowed_close, cv2.bitwise_not(core))
    count, labels, stats, _ = cv2.connectedComponentsWithStats(holes, connectivity=8)
    if count > 1:
        outside = cv2.dilate(cv2.bitwise_not(allowed_close), _disk(1)) > 0
        touching = np.unique(labels[outside])
        fill = np.zeros(count, dtype=bool)
        mean_l = np.bincount(labels.ravel(), weights=lab[:, :, 0].ravel(), minlength=count)
        mean_l /= np.maximum(stats[:, cv2.CC_STAT_AREA], 1)
        fill[1:] = mean_l[1:] > med[0]
        fill[touching] = False
        fill[0] = False
        core = cv2.bitwise_or(core, (fill[labels] * 255).astype(np.uint8))

    count, labels, stats, _ = cv2.connectedComponentsWithStats(core, connectivity=8)
    if count > 1:
        min_area = max(16, int(cv2.countNonZero(geo) * 0.02))
        keep = np.zeros(count, dtype=bool)
        keep[1:] = stats[1:, cv2.CC_STAT_AREA] >= min_area
        core = (keep[labels] * 255).astype(np.uint8)
    return core, oval, guard


# ── 잡티 제거 ──

_BLEMISH_MAX_COUNT = 15        # 얼굴 하나에서 지우는 최대 개수 (강한 것부터)
_BLEMISH_MAX_AREA_RATIO = 0.01  # 얼굴 피부 면적 대비 총 상한
_BLEMISH_FIELD_NEIGHBORS = 4    # 이만큼 이웃이 붙어 있으면 주근깨 밭으로 보고 남긴다


def _blemish_search_mask(face: _SkinFace) -> np.ndarray:
    """잡티를 찾는 영역 — 피부 확정 영역에서 얼굴 폭 4%만큼 더 안쪽.

    헤어라인·눈썹·입술·콧방울 경계에서 떨어뜨려 두면, 경계의 머리카락 끝이나
    음영이 잡티 후보로 들어오지 않는다.
    """
    strict = _shrink(face.core, max(2, face.face_w * 0.04))
    return cv2.bitwise_and(strict, cv2.bitwise_not(face.guard))


def _detect_blemishes(
    roi_rgb: np.ndarray,
    core: np.ndarray,
    search: np.ndarray,
    face_w: float,
    intensity: float,
) -> np.ndarray:
    """작고 둥글고 떨어져 있는 붉은/어두운 점만 잡티로 고른다.

    - 크기 기준은 얼굴 폭이다. 예전에는 사진 짧은 변 기준이라(최대 지름 3.5%)
      3413x2560 사진에서 폭 450px 얼굴이면 얼굴 폭 5분의 1짜리 덩어리까지
      '잡티'였다 — 앞머리 끝이 통째로 덧칠된 원인.
    - 길쭉한 성분(머리카락·주름)은 뺀다 (2차 모멘트 장단축비 > 2).
    - 이웃이 많은 점은 주근깨 밭이다 — 자연스러운 개성이라 남긴다.
    - 배경(국소 평균)은 피부 화소만으로 잰다. 머리카락이 섞이면 경계의
      피부가 통째로 튀는 값이 된다.

    반환: 지울 영역(코어, 확장 전) uint8 0/255.
    """
    empty = np.zeros(core.shape, dtype=np.uint8)
    if cv2.countNonZero(search) == 0:
        return empty

    lab = cv2.cvtColor(roi_rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    weights = core.astype(np.float32) / 255.0
    bg = _masked_blur(lab, weights, max(3.0, face_w * 0.04))
    sig = cv2.GaussianBlur(lab, (0, 0), max(0.7, face_w * 0.0035))
    d_l = sig[:, :, 0] - bg[:, :, 0]
    d_a = sig[:, :, 1] - bg[:, :, 1]
    d_b = sig[:, :, 2] - bg[:, :, 2]
    # 붉은 기(a+)가 주 신호. 갈색(b+)·어두움은 주근깨와 겹치므로 약하게만 본다.
    score = np.sqrt(np.maximum(d_a, 0) ** 2 + 0.4 * d_b ** 2) + 0.15 * np.maximum(-d_l, 0)
    score[search == 0] = 0

    # intensity 0.3 → 4.6(뚜렷한 것만), 1.0 → 2.5(옅은 것까지)
    threshold = max(2.5, 5.5 - intensity * 3.0)
    min_d = max(2.0, face_w * 0.006)
    max_d = max(4.0, face_w * 0.035)
    cand = (score > threshold).astype(np.uint8) * 255
    # 잡티보다 작은 부스러기(화소 노이즈·그레인)를 먼저 걷어낸다. 그대로 두면
    # 잡티에 달라붙어 모양이 일그러지고 이웃 수도 부풀린다.
    cand = cv2.morphologyEx(cand, cv2.MORPH_OPEN, _disk(max(1, int(min_d / 2))))
    count, labels, stats, centroids = cv2.connectedComponentsWithStats(cand, connectivity=8)
    if count <= 1:
        return empty

    min_area = max(3, int(math.pi / 4 * min_d * min_d))
    max_area = int(math.pi / 4 * max_d * max_d)

    field_r2 = (face_w * 0.06) ** 2
    # 성분별 최대 점수
    peaks = np.zeros(count, dtype=np.float32)
    np.maximum.at(peaks, labels.ravel(), score.ravel())
    # 이웃으로 셀 점: 잡티 크기 이상인 것만 (화소 노이즈·그레인 부스러기는 빼고)
    sized = stats[:, cv2.CC_STAT_AREA] >= min_area
    sized[0] = False
    chosen: list[tuple[float, int]] = []
    for i in range(1, count):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if not (min_area <= area <= max_area):
            continue
        x, y, bw, bh = stats[i, :4]
        comp = labels[y:y + bh, x:x + bw] == i
        m = cv2.moments(comp.astype(np.uint8), binaryImage=True)
        if m["m00"] <= 0:
            continue
        mu20, mu02, mu11 = m["mu20"] / m["m00"], m["mu02"] / m["m00"], m["mu11"] / m["m00"]
        common = math.sqrt(max(0.0, ((mu20 - mu02) / 2) ** 2 + mu11 ** 2))
        l1 = (mu20 + mu02) / 2 + common
        l2 = (mu20 + mu02) / 2 - common
        if l2 <= 1e-6 or math.sqrt(l1 / l2) > 2.0:
            continue  # 길쭉함 — 머리카락·주름
        if area / float(bw * bh) < 0.4:
            continue  # 가지 친 모양
        # 주근깨 밭: 비슷한 세기(절반 이상)의 점이 가까이 여럿 모여 있다.
        # 노이즈 속에 홀로 튀는 뾰루지는 주변 점들이 훨씬 약하다.
        peak = float(peaks[i])
        similar = sized & (peaks >= peak * 0.5)
        similar[i] = False
        d2 = ((centroids[similar] - centroids[i]) ** 2).sum(axis=1)
        if int((d2 < field_r2).sum()) >= _BLEMISH_FIELD_NEIGHBORS:
            continue
        chosen.append((peak, i))

    if not chosen:
        return empty
    chosen.sort(reverse=True)
    budget = cv2.countNonZero(core) * _BLEMISH_MAX_AREA_RATIO
    keep = np.zeros(count, dtype=bool)
    used = 0
    for peak, i in chosen[:_BLEMISH_MAX_COUNT]:
        area = int(stats[i, cv2.CC_STAT_AREA])
        if used + area > budget:
            continue
        keep[i] = True
        used += area
    return (keep[labels] * 255).astype(np.uint8)


def _remove_blemishes_face(arr: np.ndarray, face: _SkinFace, intensity: float) -> bool:
    """얼굴 하나의 잡티를 제자리에서 지운다. 지운 게 있으면 True."""
    roi = arr[face.y0:face.y1, face.x0:face.x1]
    spots = _detect_blemishes(roi, face.core, _blemish_search_mask(face), face.face_w, intensity)
    if cv2.countNonZero(spots) == 0:
        return False

    fw = face.face_w
    # 임계값을 넘는 건 잡티의 코어뿐이다 — 번진 테두리까지 덮게 조금 넓힌다
    grow = max(1, int(round(fw * 0.006)))
    fill_mask = cv2.dilate(spots, _disk(grow))
    radius = max(2, int(round(fw * 0.01)))
    inpainted = cv2.inpaint(roi, fill_mask, inpaintRadius=radius, flags=cv2.INPAINT_TELEA)

    # 경계는 부드럽게 — 한 번 더 넓힌 마스크를 흐려 알파로 쓴다 (안쪽은 1)
    soft = cv2.dilate(fill_mask, _disk(grow)).astype(np.float32) / 255.0
    soft = cv2.GaussianBlur(soft, (0, 0), max(0.8, grow))
    soft = np.minimum(1.0, soft * 1.5)
    # 확장분이 피부 밖으로 새지 않게
    soft *= face.core.astype(np.float32) / 255.0
    fill = min(1.0, 0.8 + intensity * 0.2)
    a = (soft * fill)[:, :, np.newaxis]
    out = roi.astype(np.float32) * (1.0 - a) + inpainted.astype(np.float32) * a
    roi[:] = np.clip(out + 0.5, 0, 255).astype(np.uint8)
    return True


def apply_blemish_removal(
    img: Image.Image,
    intensity: float,
    cache: MediaPipeCache | None = None,
) -> Image.Image:
    """잡티 자동 제거. intensity: 0.0(비활성) ~ 1.0(최대 감도).

    파이프라인 (얼굴마다, 얼굴 상자 안에서만):
      1. 피부 확정 마스크 ([refine_skin_mask]) — 머리카락·이목구비 제외
      2. 경계에서 얼굴 폭 4% 안쪽에서만, 작고 둥글고 외따로 있는 점을 고른다
      3. 인페인팅 후 부드러운 알파로 섞는다

    얼굴 미감지·너무 작은 얼굴이면 원본을 그대로 반환한다.
    """
    if intensity < 0.01:
        return img

    arr = np.array(img)
    faces = _get_skin_faces(arr, cache=cache)
    if not faces:
        return img

    changed = False
    for face in faces:
        changed |= _remove_blemishes_face(arr, face, min(1.0, intensity))
    return Image.fromarray(arr) if changed else img


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
        # 시작점을 안쪽으로 당긴다. 폭만 하한으로 올리고 x를 두면 오른쪽이
        # 프레임 끝에서 잘려 하한이 무력해진다 (x=0.9, w=0.1 → 1000px가 100px).
        x = min(x, 1.0 - cw)
        y = min(y, 1.0 - ch)

        left = int(x * w)
        top = int(y * h)
        right = min(w, int((x + cw) * w))
        bottom = min(h, int((y + ch) * h))

        if right - left < 50 or bottom - top < 50:
            return img

        return img.crop((left, top, right, bottom))
    except Exception:
        return img


# 인스타그램이 받는 세로형·정사각 비율 (가로/세로).
# 3:4는 2025년 1월 프로필 그리드가 3:4로 바뀌고 5월부터 3:4 게시물이 지원되면서 추가했다.
_INSTAGRAM_RATIOS: dict[str, float] = {
    "3:4": 3 / 4,
    "4:5": 4 / 5,
    "1:1": 1.0,
}


def apply_instagram_ratio(
    img: Image.Image, ratio: str, allow_vertical_crop: bool = True
) -> Image.Image:
    """인스타그램 최적 비율로 중앙 크롭한다.

    ratio: "3:4" (2025년부터 프로필 그리드 비율, 1080x1440), "4:5" (피드 세로),
           "1:1" (정사각형). 공백·"x" 구분자("3x4")도 받는다.

    allow_vertical_crop=False면 위아래를 자르지 않는다. 전신 인물 사진에서
    가운데를 기준으로 위아래를 자르면 머리와 발이 잘려 다리가 짧아 보인다.
    """
    try:
        w, h = img.size
        target = _INSTAGRAM_RATIOS.get(str(ratio).strip().lower().replace("x", ":"))
        if target is None:
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
# 적용하므로 두 패스가 겹친다. 분석 경로는 param_engine이 영역 값을 전역에 합치고
# 영역 쪽을 0으로 보내므로, 여기 걸리는 건 옛 analysis를 들고 있는 앱이 보낸
# apply-transform 정도다 — 겹쳐도 과하지 않게 낮게 묶는다.
_FACE_TEXTURE_LIMIT = 0.3

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
        if not all(math.isfinite(v) for v in (ax, ay, aw, ah)):
            # NaN은 min/max를 그대로 통과해 면적 검사도 빠져나간다
            log.info("local region %s: area 좌표가 유한하지 않음 — 버림", name)
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
            if not math.isfinite(value):
                # np.clip은 NaN을 그대로 돌려줘 상한을 통과한다
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

        # uint8로 들고 있다가 합성할 때 float로 바꾼다 — 영역마다 전체 크기
        # float32(3413x2560에서 100MB)를 동시에 들고 있지 않게. 값은 같다.
        layers.append((
            region_name,
            np.array(region_img, dtype=np.uint8),
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
    for _, region_u8, alpha in layers:
        weight = (alpha * remaining)[:, :, np.newaxis]
        # (region - 원본) * weight를 제자리 연산으로 — 같은 float32 연산 순서라
        # 결과는 그대로이고 전체 크기 임시 배열이 하나로 준다.
        delta = region_u8.astype(np.float32)
        delta -= arr_rgb
        delta *= weight
        result += delta
        del delta
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

# ── 얼굴/체형 보정 (가우시안 국소 워프) ──

# 얼굴 윤곽 랜드마크 인덱스 (MediaPipe 478개 중 양쪽 볼·턱선)
_FACE_CONTOUR_LEFT = [234, 93, 132, 58, 172, 136, 150, 149, 176, 148, 152]
_FACE_CONTOUR_RIGHT = [454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152]

# 턱선 인덱스 (V라인)
_JAW_LEFT = [172, 136, 150, 149, 176, 148]
_JAW_RIGHT = [397, 365, 379, 378, 400, 377]
_JAW_TIP = [152]

# 코 고정점 — 움직이지 않는 제어점으로 넣는다.
#
# 얼굴 보정은 윤곽·턱·눈만 옮기고 코에는 손대지 않는데, 그래서 오히려
# 코가 망가졌다. 워프는 제어점 사이를 보간하므로, 코처럼 제어점이 하나도
# 없는 영역은 주변 볼이 끌고 가는 대로 딸려간다. face_slim은 dx만 주고
# dy는 0이라 그 결과가 "가로로만 눌리기"다 — 실측: face_slim 0.7에서
# 코 야코비안 가로 1.105 / 세로 1.000 (비등방 0.105). 세로는 그대로인데
# 가로만 10% 좁아지니 콧대가 각지게 선다.
#
# 코 위에 변위 0인 점을 박아 두면 압축이 코를 비켜 볼로 몰린다.
# 실측: 코 비등방 0.105 → 0.018, 볼 0.101 → 0.148 (슬림이 일어나야 할 곳).
#
# 인덱스는 MediaPipe FaceLandmarksConnections.FACE_LANDMARKS_NOSE 원본.
# 윤곽·턱·눈 인덱스와 겹치지 않는다 (test_face_reshape_nose.py에서 검증).
_NOSE_ANCHORS = [
    1, 2, 4, 5, 6, 19, 45, 48, 64, 94, 97, 98, 115, 168, 195, 197,
    220, 275, 278, 294, 326, 327, 344, 440,
]

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

# 윤곽을 옮기면 안 되는 얼굴 부위 — 변위 0인 고정점으로 넣는다.
#   이마·관자놀이(얼굴 타원 윗부분): 볼을 줄여도 헤어라인이 딸려오지 않게
#   눈꼬리·입꼬리·입술 중앙: 볼 슬림이 눈을 가로로 좁히거나 웃는 입을 누르지 않게
# 눈 확대는 따로 계산하는 국소 확대(아래 _eye_bulge)라 이 고정점과 부딪히지 않는다.
_FACE_UPPER_ANCHORS = [127, 162, 21, 54, 103, 67, 109, 10, 338, 297, 332, 284, 251, 389, 356]
_FACE_FEATURE_ANCHORS = [33, 133, 362, 263, 61, 291, 0, 17]

# 윤곽점별 슬림 비중 (_FACE_CONTOUR_LEFT/RIGHT 순서: 광대 → 볼 → 턱 → 턱끝).
# 예전에는 광대부터 턱끝 옆까지 똑같이 14%씩 당겨 광대와 관자놀이 경계,
# 턱끝 양옆이 함께 꺾였다. 사진관 보정은 아랫볼·턱선이 주로 줄고 광대는 조금,
# 턱끝은 제자리다 (152는 0 = 고정점).
_SLIM_PROFILE = [0.45, 0.75, 1.0, 1.0, 1.0, 0.9, 0.75, 0.6, 0.45, 0.3, 0.0]

# 턱선 비중 (_JAW_LEFT/RIGHT 순서: 턱각 → 턱끝 옆).
# 예전 jaw_sharpen은 턱선 전체를 안쪽 10% + 턱끝 높이 쪽으로 5% 끌어내려
# 턱끝이 뾰족한 V자가 됐다. 지금은 턱각만 안쪽으로 다듬고 턱끝 쪽으로 갈수록 줄인다.
_JAW_PROFILE = [1.0, 1.0, 0.8, 0.55, 0.3, 0.15]

# ── 보정 강도 보정(calibration) ──
#
# 값 1.0(앱 슬라이더 최대)에서의 실제 효과. 모델이 주로 쓰는 0.2~0.35가
# "티 나지 않게 조금"이 되도록 예전보다 약 절반으로 낮췄다.
#   face_slim   1.0 → 아랫볼 윤곽이 얼굴 중심선까지 거리의 약 8% 안쪽으로 (예전 14%)
#   jaw_sharpen 1.0 → 턱각이 약 7% 안쪽으로 (예전 10% + 아래로 5%)
#   eye_enlarge 1.0 → 눈 중심 배율 약 1.28배 (0.3 → 1.07배, 0.45 → 1.11배), 눈 밖으로 갈수록 0
#               (예전: 윤곽 18% + 얼굴 전체 출렁임). 0.10이던 게인은 0.45에서 최대 1px라 안 보였다.
# 이 값들은 제어점에 주는 목표치이고, 가우시안 가중 평균이 주변 고정점과 섞으며
# 조금 덜어낸다 — _FACE_FIELD_COMP가 그만큼을 되돌린다 (실측으로 맞춤).
_FACE_SLIM_GAIN = 0.08
_JAW_SHARPEN_GAIN = 0.07
_EYE_ENLARGE_GAIN = 0.22
_FACE_FIELD_COMP = 1.35

# 변형 가능한 배경 띠의 폭 하한 (사진 짧은 변 대비). 인물 실루엣 밖으로는
# 이 띠(또는 최대 이동량의 4배 중 큰 쪽) 안에서만 변위가 0으로 줄어든다.
_SUPPORT_BAND_MIN = 0.01


def _gaussian_field(
    xs: np.ndarray,
    ys: np.ndarray,
    pts: np.ndarray,
    disp: np.ndarray,
    sigma: np.ndarray | float,
    far: float = 3.0,
) -> tuple[np.ndarray, np.ndarray]:
    """제어점 변위를 가우시안 가중 평균으로 (ys × xs) 격자에 펼친다.

    d(v) = Σ g_i(v) d_i / (Σ g_i(v) + g_far),  g_i = exp(-|v - p_i|² / 2σ_i²)

    예전 워프는 역거리가중(1/d², Shepard)이었다. 그 가중은 꼬리가 길어서
    제어점에서 멀어질수록 변위가 0이 아니라 "모든 제어점 변위의 평균"으로
    수렴한다. 그래서 눈 확대만 해도 볼·턱·머리카락·배경까지 수 px씩 밀렸고
    (실측: eye_enlarge 0.45에서 인물 밖 3% 지점 배경 6.6px), 그 변형을
    타원 마스크로 잘라내니 경계가 얼룩처럼 보였다.
    가우시안 가중은 몇 σ 밖에서 사라지고, g_far(= 거리 far·σ의 가중)가
    "먼 곳은 변위 0"인 보이지 않는 고정점 역할을 한다. 결과는 제어점 변위와
    0의 볼록 결합이라 목표치를 넘치는 일(오버슈트)도 없다.

    가중이 x·y로 분리되므로 제어점마다 (W,)·(H,) 두 벡터의 외적만 더한다.
    pts: (N, 2) 출력 좌표, disp: (N, 2) 역워프 변위(출력 → 원본), sigma: (N,) 또는 스칼라.
    """
    xs = np.asarray(xs, dtype=np.float32)
    ys = np.asarray(ys, dtype=np.float32)
    pts = np.asarray(pts, dtype=np.float32).reshape(-1, 2)
    disp = np.asarray(disp, dtype=np.float32).reshape(-1, 2)
    sig = np.broadcast_to(np.asarray(sigma, dtype=np.float32), (len(pts),))
    shape = (ys.shape[0], xs.shape[0])
    den = np.full(shape, np.exp(-0.5 * far * far), dtype=np.float32)
    num_x = np.zeros(shape, dtype=np.float32)
    num_y = np.zeros(shape, dtype=np.float32)
    for (px, py), (dx, dy), s in zip(pts, disp, sig):
        inv = np.float32(-0.5 / max(float(s), 1e-3) ** 2)
        gx = np.exp(inv * (xs - px) ** 2)
        gy = np.exp(inv * (ys - py) ** 2)
        wgt = np.outer(gy, gx)
        den += wgt
        if dx != 0.0:
            num_x += wgt * dx
        if dy != 0.0:
            num_y += wgt * dy
    return num_x / den, num_y / den


def _eye_bulge(
    xs: np.ndarray,
    ys: np.ndarray,
    center: tuple[float, float],
    rx: float,
    ry: float,
    angle: float,
    amount: float,
) -> tuple[np.ndarray, np.ndarray]:
    """눈 하나를 국소 확대하는 역워프 변위.

    원본 = c + (q - c)(1 - a·k(ρ)),  k(ρ) = (1 - ρ²)² (ρ < 1), ρ = 눈 좌표계 타원 반경.
    변위는 타원(눈꺼풀·눈 주변) 밖에서 정확히 0이다. 예전에는 눈 윤곽점만
    바깥으로 밀고 나머지를 보간에 맡겨 얼굴 전체가 따라 움직였다.
    반경 방향 사상의 기울기는 1 - a(1-ρ²)(1-5ρ²) ≥ 1 - a라 a < 1이면 단조 — 접힘이 없다
    (eye_enlarge 1.0에서 a = 0.22).
    """
    cx, cy = center
    ca, sa = math.cos(angle), math.sin(angle)
    X = (np.asarray(xs, dtype=np.float32) - cx)[np.newaxis, :]
    Y = (np.asarray(ys, dtype=np.float32) - cy)[:, np.newaxis]
    u = X * ca + Y * sa
    v = -X * sa + Y * ca
    rho2 = (u / rx) ** 2 + (v / ry) ** 2
    k = np.clip(1.0 - rho2, 0.0, None) ** 2
    f = np.float32(-amount) * k
    return X * f, Y * f


def _support_weight(person: np.ndarray | None, band: float) -> np.ndarray | None:
    """인물 실루엣 안은 1, 밖으로 band px에 걸쳐 0으로 줄어드는 가중치.

    변위장에 곱해서 배경(문틀·벽 모서리·수평선)이 인물 옆에서 휘지 않게 한다.
    윤곽이 옮겨 간 자리를 채우려면 실루엣 바로 밖의 좁은 띠는 늘어나야 하므로
    0/1로 자르지 않고 band 폭으로 부드럽게 줄인다 (band ≥ 최대 이동량의 4배라 접힘 없음).
    """
    if person is None:
        return None
    fg = (person > 0.5).astype(np.uint8)
    if not fg.any():
        return None
    if fg.all():
        return np.ones(fg.shape, dtype=np.float32)
    dist = cv2.distanceTransform(1 - fg, cv2.DIST_L2, 3)
    t = np.clip(1.0 - dist / max(band, 1.0), 0.0, 1.0)
    return (t * t * (3.0 - 2.0 * t)).astype(np.float32)


def _field_step(scale: float) -> int:
    """변위장을 계산할 성긴 격자 간격. 장이 σ(얼굴 폭의 ~10%) 규모로 매끄러워
    얼굴 폭의 1.5% 간격이면 충분하다 (선형 보간 오차 < 0.1px)."""
    return int(np.clip(scale * 0.015, 2, 12))


def _remap_region(
    arr: np.ndarray,
    roi: tuple[int, int, int, int],
    field_fn,
    step: int,
    person: np.ndarray | None,
    band: float,
) -> tuple[np.ndarray, tuple[np.ndarray, np.ndarray] | None]:
    """roi (x0, y0, x1, y1) 안에서 변위장을 만들어 한 번의 remap으로 적용한다.

    field_fn(xs, ys) -> (dx, dy): 원본 좌표계 격자 좌표를 받아 역워프 변위를 준다.
    반환: (결과 배열, roi 안의 최종 변위장 또는 None).
    """
    h, w = arr.shape[:2]
    x0, y0, x1, y1 = roi
    x0, y0 = max(0, int(x0)), max(0, int(y0))
    x1, y1 = min(w, int(math.ceil(x1))), min(h, int(math.ceil(y1)))
    rw, rh = x1 - x0, y1 - y0
    if rw < 4 or rh < 4:
        return arr, None

    gw = max(2, -(-rw // step))
    gh = max(2, -(-rh // step))
    # 격자점 j를 픽셀 중심 규약에 맞춰 둔다 — cv2.resize(INTER_LINEAR)가
    # 격자점 사이를 정확히 선형 보간하도록 (반 픽셀 밀림 없음).
    gxs = x0 + (np.arange(gw, dtype=np.float32) + 0.5) * (rw / gw) - 0.5
    gys = y0 + (np.arange(gh, dtype=np.float32) + 0.5) * (rh / gh) - 0.5
    cdx, cdy = field_fn(gxs, gys)
    dx = cv2.resize(np.ascontiguousarray(cdx, dtype=np.float32), (rw, rh), interpolation=cv2.INTER_LINEAR)
    dy = cv2.resize(np.ascontiguousarray(cdy, dtype=np.float32), (rw, rh), interpolation=cv2.INTER_LINEAR)

    # ROI 가장자리는 어떤 경우에도 변위 0 — 장이 이미 0에 수렴하지만 안전장치.
    edge = max(2, int(min(rw, rh) * 0.04))
    ramp_x = np.clip(np.minimum(np.arange(rw), np.arange(rw)[::-1]) / edge, 0, 1).astype(np.float32)
    ramp_y = np.clip(np.minimum(np.arange(rh), np.arange(rh)[::-1]) / edge, 0, 1).astype(np.float32)
    taper = np.outer(ramp_y, ramp_x)
    if person is not None:
        sup = _support_weight(person[y0:y1, x0:x1], band)
        if sup is not None:
            taper *= sup
    dx *= taper
    dy *= taper

    map_x = dx + np.arange(x0, x1, dtype=np.float32)[np.newaxis, :]
    map_y = dy + np.arange(y0, y1, dtype=np.float32)[:, np.newaxis]
    out = arr.copy()
    out[y0:y1, x0:x1] = cv2.remap(arr, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    return out, (dx, dy)


class _FaceControls:
    """얼굴 한 개의 변형 정의 — 윤곽 제어점(가우시안 장) + 눈 국소 확대."""

    def __init__(self) -> None:
        self.pts: list[list[float]] = []    # 출력 좌표
        self.disp: list[list[float]] = []   # 역워프 변위 (0이면 고정점)
        self.eyes: list[tuple] = []          # (center, rx, ry, angle, amount)
        self.sigma = 1.0
        self.scale = 1.0                     # 얼굴 폭 (px)
        self.max_move = 0.0                  # 최대 이동량 (px)
        self.yaw = 0.0                       # -1~1, 부호 = 가까운 쪽

    @property
    def active(self) -> bool:
        return self.max_move > 0.0 or bool(self.eyes)

    def field(self, xs: np.ndarray, ys: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if self.max_move > 0.0:
            dx, dy = _gaussian_field(xs, ys, np.array(self.pts), np.array(self.disp), self.sigma)
        else:
            dx = np.zeros((len(ys), len(xs)), np.float32)
            dy = np.zeros_like(dx)
        for center, rx, ry, ang, amt in self.eyes:
            ex, ey = _eye_bulge(xs, ys, center, rx, ry, ang, amt)
            dx += ex
            dy += ey
        return dx, dy


def _face_yaw(pt) -> float:
    """고개 돌림 정도 (-1~1). 코 중심선에서 양쪽 광대(234, 454)까지 거리의 비대칭.

    정면 0. 코 중심선이 머리 앞쪽 표면에 있어 작은 회전에도 크게 반응한다
    (대략 tan θ — 15° ≈ ±0.25, 25° ≈ ±0.45). 부호는 454 쪽이 가까우면 +.
    """
    top = np.array(pt(168), np.float64)
    chin = np.array(pt(152), np.float64)
    m = chin - top
    n_ = np.linalg.norm(m)
    if n_ < 1e-6:
        return 0.0
    nrm = np.array([m[1], -m[0]]) / n_
    dl = abs(float(np.dot(np.array(pt(234)) - top, nrm)))
    dr = abs(float(np.dot(np.array(pt(454)) - top, nrm)))
    if dl + dr < 1e-6:
        return 0.0
    return (dr - dl) / (dr + dl)


def _build_reshape_controls(
    pt,
    n_landmarks: int,
    face_slim: float,
    jaw_sharpen: float,
    eye_enlarge: float,
) -> _FaceControls:
    """얼굴 한 개의 변형 정의를 만든다.

    pt(idx) -> (x, y): 랜드마크 인덱스를 픽셀 좌표로 바꾸는 함수.
    MediaPipe에 의존하지 않아 합성 좌표로 단독 검증할 수 있다.

    - 윤곽은 얼굴 중심선(미간 168 – 턱끝 152)에 수직으로, 그 선까지 거리에 비례해 옮긴다.
      예전에는 광대 양끝의 중점을 축으로 좌우 같은 px만큼 옮겨, 돌아간 얼굴에서
      좁게 보이는 먼 쪽 볼이 비율상 더 많이 눌렸다.
    - 양쪽 광대 끝의 이동량은 머리 반폭 기준으로 같게 — 돌아간 얼굴에서 한쪽만 깎이지 않게.
      먼 쪽은 조금 덜(yaw 0.25에서 0.8배) 옮기고, 거의 옆모습이면 슬림·턱·눈을 끈다.
    - 윤곽 바깥에 변위 0인 고정점 띠를 둘러 배경·머리카락·목이 따라 휘지 않게 한다.
    """
    c = _FaceControls()
    face_slim = float(np.clip(face_slim, 0.0, 1.0))
    jaw_sharpen = float(np.clip(jaw_sharpen, 0.0, 1.0))
    eye_enlarge = float(np.clip(eye_enlarge, 0.0, 1.0))
    if face_slim < 0.01 and jaw_sharpen < 0.01 and eye_enlarge < 0.01:
        return c

    P = lambda i: np.array(pt(i), dtype=np.float64)  # noqa: E731
    top, chin = P(168), P(152)
    axis = chin - top
    axis_len = float(np.linalg.norm(axis))
    face_w = float(np.linalg.norm(P(454) - P(234)))
    if axis_len < 1e-3 or face_w < 1e-3:
        return c
    down = axis / axis_len
    lateral = np.array([down[1], -down[0]])  # 중심선에 수직
    c.scale = face_w
    c.sigma = 0.11 * face_w
    yaw = _face_yaw(pt)
    c.yaw = yaw

    # 좌우 이동량 기준 — 머리 반폭 (234·454 사이 거리의 절반).
    # 코 중심선은 머리 앞쪽 표면에 있어서, 고개가 조금만 돌아가도 먼 쪽으로
    # 치우쳐 보인다 (hammock 실측: 중심선에서 234까지 238px, 454까지 418px, 약 15°).
    # 중심선까지 거리에 비례해 옮기면 가까운 쪽 볼이 먼 쪽의 1.8배 움직이고,
    # 먼 쪽 감쇠까지 겹치면 3배가 넘어 한쪽 볼만 깎인 얼굴이 된다.
    # 실제 얼굴을 3D로 좁히면 양쪽 실루엣은 화면에서 거의 같은 px만큼 들어온다.
    # 그래서 각 쪽 윤곽점의 "그 쪽 볼 폭 대비 위치"(0=중심선, 1=광대 끝)에
    # 머리 반폭을 곱해 양쪽 광대 끝의 목표 이동량을 같게 맞춘다.
    lat_l = float(np.dot(P(234) - top, lateral))
    lat_r = float(np.dot(P(454) - top, lateral))
    half_w = 0.5 * abs(lat_r - lat_l)

    # 먼 쪽은 보이는 볼 폭이 좁아 같은 px도 더 크게 눌린다 — 살짝만 덜 옮긴다
    # (yaw 0.25에서 0.8배). 많이 돌아간 얼굴(|yaw| 0.5~0.75, 약 27°~37°)은 전체를
    # 줄여 가다 끈다 — 먼 쪽 윤곽이 코 뒤로 숨어 랜드마크를 믿기 어렵다.
    far_factor = float(np.clip(1.0 - 0.8 * abs(yaw), 0.5, 1.0))
    global_factor = float(np.clip((0.75 - abs(yaw)) / 0.25, 0.0, 1.0))
    near_sign = float(np.sign(lat_r) * np.sign(yaw))  # yaw > 0: 454 쪽이 가깝다

    def side_scale(s: float) -> float:
        """lateral 좌표 s인 윤곽점의 이동 기준 거리 (px, 부호 없음)."""
        side_extent = abs(lat_r) if np.sign(s) == np.sign(lat_r) else abs(lat_l)
        if side_extent < 1e-3:
            return 0.0
        is_far = yaw != 0.0 and np.sign(s) == -near_sign
        return abs(s) / side_extent * half_w * global_factor * (far_factor if is_far else 1.0)

    moves: dict[int, np.ndarray] = {}

    def add_move(idx: int, weight: float, gain: float) -> None:
        if idx >= n_landmarks or weight <= 0.0:
            return
        p = P(idx)
        s = float(np.dot(p - top, lateral))
        mv = -np.sign(s) * side_scale(s) * gain * weight * lateral
        moves[idx] = moves.get(idx, np.zeros(2)) + mv

    if face_slim >= 0.01:
        g = face_slim * _FACE_SLIM_GAIN * _FACE_FIELD_COMP
        for contour in (_FACE_CONTOUR_LEFT, _FACE_CONTOUR_RIGHT):
            for idx, wt in zip(contour, _SLIM_PROFILE):
                add_move(idx, wt, g)
    if jaw_sharpen >= 0.01:
        g = jaw_sharpen * _JAW_SHARPEN_GAIN * _FACE_FIELD_COMP
        for jaw in (_JAW_LEFT, _JAW_RIGHT):
            for idx, wt in zip(jaw, _JAW_PROFILE):
                add_move(idx, wt, g)

    moves = {i: m for i, m in moves.items() if float(np.linalg.norm(m)) > 1e-3}
    if moves:
        c.max_move = max(float(np.linalg.norm(m)) for m in moves.values()) / _FACE_FIELD_COMP
        delta = max(4.0 * c.max_move, 0.10 * face_w)
        center = (top + chin) / 2.0
        for idx, mv in moves.items():
            p = P(idx)
            # 역워프 변위는 옮겨 간 자리(출력 좌표)에 둔다
            c.pts.append(list(p + mv))
            c.disp.append(list(-mv))
        # 고정점: 이동한 윤곽 바깥 띠 + 턱끝 아래(목)
        moved_contour = [i for i in (_FACE_CONTOUR_LEFT + _FACE_CONTOUR_RIGHT) if i < n_landmarks]
        for idx in dict.fromkeys(moved_contour):
            p = P(idx)
            out = p - center
            nrm = float(np.linalg.norm(out))
            if nrm < 1e-6:
                continue
            c.pts.append(list(p + out / nrm * delta))
            c.disp.append([0.0, 0.0])
        c.pts.append(list(chin + down * delta))
        c.disp.append([0.0, 0.0])
        # 고정점: 코, 이마·관자놀이, 눈꼬리·입꼬리
        for idx in list(_NOSE_ANCHORS) + _FACE_UPPER_ANCHORS + _FACE_FEATURE_ANCHORS:
            if idx < n_landmarks and idx not in moves:
                c.pts.append(list(P(idx)))
                c.disp.append([0.0, 0.0])

    if eye_enlarge >= 0.01:
        amount = eye_enlarge * _EYE_ENLARGE_GAIN * global_factor
        if amount > 1e-4:
            for center_idx, contour, corners in (
                (_LEFT_EYE_CENTER, _LEFT_EYE_CONTOUR, (362, 263)),
                (_RIGHT_EYE_CENTER, _RIGHT_EYE_CONTOUR, (133, 33)),
            ):
                if any(i >= n_landmarks for i in corners):
                    continue
                a, b = P(corners[0]), P(corners[1])
                ew = float(np.linalg.norm(b - a))
                if ew < 2.0:
                    continue
                if n_landmarks > center_idx:
                    ec = P(center_idx)
                else:
                    ec = np.mean([P(i) for i in contour if i < n_landmarks], axis=0)
                ang = math.atan2(b[1] - a[1], b[0] - a[0])
                c.eyes.append(((float(ec[0]), float(ec[1])), 0.85 * ew, 0.65 * ew, ang, amount))

    return c


def _cached_person_mask(cache: Any, arr_rgb: np.ndarray) -> np.ndarray | None:
    """캐시에 인물 분할이 있으면 쓴다 (없거나 실패하면 None → 실루엣 제한 없이 고정점만)."""
    getter = getattr(cache, "get_person_mask", None) if cache is not None else None
    if getter is None:
        return None
    try:
        mask = getter(arr_rgb)
    except Exception as exc:  # 분할 실패가 보정 전체를 막지 않게
        log.warning("reshape: person mask unavailable: %s", exc)
        return None
    if mask is None or mask.shape[:2] != arr_rgb.shape[:2]:
        return None
    return mask


def apply_face_reshape(
    img: Image.Image,
    face_slim: float = 0.0,
    jaw_sharpen: float = 0.0,
    eye_enlarge: float = 0.0,
    cache: MediaPipeCache | None = None,
) -> Image.Image:
    """얼굴 보정 — MediaPipe 478 랜드마크 기반 국소 워프.

    face_slim: 0~1 (아랫볼·턱선을 얼굴 중심선 쪽으로)
    jaw_sharpen: 0~1 (턱각을 안쪽으로 — 턱끝은 고정)
    eye_enlarge: 0~1 (눈 주변만 국소 확대)

    변형은 얼굴 윤곽 바로 바깥의 고정점 띠와 인물 실루엣(분할 마스크) 안에서
    끝난다 — 배경의 직선이 볼 옆에서 휘지 않는다. 얼굴 미감지 시 원본 반환.
    다중 얼굴은 각각 독립 적용. cache가 있으면 MediaPipe 모델/결과를 재사용한다.
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

    person = _cached_person_mask(cache, arr_rgb)
    result_arr = arr_rgb

    for face_lms in results.face_landmarks:
        def _pt(idx: int) -> tuple[float, float]:
            lm = face_lms[idx]
            return lm.x * w, lm.y * h

        ctrl = _build_reshape_controls(_pt, len(face_lms), face_slim, jaw_sharpen, eye_enlarge)
        if not ctrl.active:
            continue

        # 변형이 닿는 범위: 얼굴 랜드마크 상자 + 고정점 띠 + 가우시안 꼬리
        fxs = [face_lms[i].x * w for i in range(min(len(face_lms), 468))]
        fys = [face_lms[i].y * h for i in range(min(len(face_lms), 468))]
        margin = max(4.0 * ctrl.max_move, 0.10 * ctrl.scale) + 3.5 * ctrl.sigma
        roi = (min(fxs) - margin, min(fys) - margin, max(fxs) + margin, max(fys) + margin)
        band = max(4.0 * ctrl.max_move, _SUPPORT_BAND_MIN * min(h, w))
        result_arr, _ = _remap_region(
            result_arr, roi, ctrl.field, _field_step(ctrl.scale), person, band,
        )

    if result_arr is arr_rgb:
        return img
    return Image.fromarray(result_arr)


# ── 체형 보정 강도 (값 1.0에서) ──
#   leg_stretch    1.0 → 엉덩이~발목 다리 길이 +10% (예전: 힙 아래 전체 +25%, 발이 잘림)
#   waist_slim     1.0 → 허리 실루엣이 반폭의 8% 안쪽으로 (예전 11%, 관절점을 밀어 팔·배경이 휨)
#   shoulder_width ±1.0 → 어깨선이 어깨 반폭의 ±6% (예전 ±10%, 목·턱까지 끌려감)
_LEG_STRETCH_GAIN = 0.10
_WAIST_SLIM_GAIN = 0.08
_SHOULDER_GAIN = 0.06
_BODY_FIELD_COMP = 1.4
# 다리를 늘린 만큼 발 아래 바닥을 최대 이 비율까지 눌러 발이 프레임 밖으로 밀리지 않게 한다
_FLOOR_MAX_COMPRESS = 0.3


def _smooth_plateau(t: np.ndarray, a: float, b: float, ramp_in: float, ramp_out: float) -> np.ndarray:
    """[a, b]에서 1, 양끝 ramp 폭에 걸쳐 0으로 매끄럽게 (smoothstep) 줄어드는 창."""
    def ss(x):
        x = np.clip(x, 0.0, 1.0)
        return x * x * (3.0 - 2.0 * x)
    up = ss((t - a) / max(ramp_in, 1e-6))
    dn = 1.0 - ss((t - (b - ramp_out)) / max(ramp_out, 1e-6))
    return up * dn


def _leg_row_map(
    h: int, hip_y: float, ankle_y: float, foot_y: float, leg_stretch: float,
) -> tuple[np.ndarray, int] | None:
    """다리 늘리기의 행 사상 (출력 행 → 원본 행)과 처음 바뀌는 행을 만든다.

    예전에는 힙 아래 전체를 1 + 0.25·값 배로 늘려, 엉덩이·손·코트 자락까지 늘어나고
    발이 사진 아래로 밀려 잘렸다 (0.45에서 발끝이 프레임 끝에 닿음).
    지금은
      - 허벅지 중간~발목(정강이 위주)만 늘리고, 엉덩이·손 높이는 그대로
      - 발은 크기 그대로 아래로 옮기고
      - 그만큼 발 아래 바닥을 눌러(최대 30%) 사진 높이 안에서 발을 지킨다.
    행 단위 사상이라 배경의 수직선은 수직, 수평선은 수평 그대로다 (휘지 않음).
    밀도(행 간격)를 매끄러운 창으로 바꾸므로 비스듬한 선에도 꺾임이 없다.
    """
    L = ankle_y - hip_y
    if L < 0.05 * h or leg_stretch < 0.01:
        return None
    extra = float(np.clip(leg_stretch, 0.0, 1.0)) * _LEG_STRETCH_GAIN * L

    t = np.arange(h, dtype=np.float64) + 0.5
    stretch = _smooth_plateau(t, hip_y + 0.15 * L, ankle_y - 0.02 * L, 0.2 * L, 0.15 * L)
    floor_top = foot_y + 0.01 * h
    floor_len = h - floor_top
    if floor_len > 0.02 * h:
        floor = _smooth_plateau(t, floor_top, h + floor_len, 0.3 * floor_len, 1e-6)
        # 바닥 행 간격이 최대 _FLOOR_MAX_COMPRESS만큼만 줄도록 늘릴 양을 제한
        extra = min(extra, _FLOOR_MAX_COMPRESS * float(floor.sum()))
    else:
        floor = None  # 발이 이미 프레임 끝이면 늘어난 만큼 아래로 밀려 나간다
    if extra < 0.5 or stretch.sum() < 1.0:
        return None

    density = stretch * (extra / stretch.sum())
    if floor is not None and floor.sum() > 1.0:
        density = density - floor * (extra / floor.sum())
    # 원본 행 경계 y → 출력 위치 y + Σ밀도
    edges_src = np.arange(h + 1, dtype=np.float64)
    edges_out = np.concatenate([[0.0], np.cumsum(1.0 + density)])
    out_rows = np.arange(h, dtype=np.float64) + 0.5
    src = np.interp(out_rows, edges_out, edges_src) - 0.5
    src = np.clip(src, 0.0, h - 1).astype(np.float32)
    changed = np.flatnonzero(np.abs(src - (out_rows - 0.5)) > 1e-3)
    if changed.size == 0:
        return None
    return src, int(changed[0])


def _torso_edge(person: np.ndarray | None, center: np.ndarray, direction: np.ndarray,
                guess: float) -> float:
    """몸통 중심에서 direction으로 실루엣 가장자리까지 거리.

    분할 마스크를 따라가 처음 배경이 나오는 곳. 팔이 몸통에 붙어 끝을 못 찾으면
    (또는 마스크가 없으면) 관절 폭으로 추정한 guess를 쓴다.
    """
    if person is None:
        return guess
    h, w = person.shape[:2]
    ss = np.arange(0.6 * guess, 1.3 * guess, 1.0)
    xs = np.round(center[0] + direction[0] * ss).astype(int)
    ys = np.round(center[1] + direction[1] * ss).astype(int)
    ok = (xs >= 0) & (xs < w) & (ys >= 0) & (ys < h)
    if not ok.any():
        return guess
    vals = np.zeros(len(ss), np.float32)
    vals[ok] = person[ys[ok], xs[ok]]
    off = np.flatnonzero(vals < 0.5)
    if off.size == 0:
        return guess
    return float(ss[off[0]])


def _build_body_controls(
    pt, vis, person: np.ndarray | None, shoulder_width: float, waist_slim: float,
) -> tuple[list, list, float, float, float]:
    """허리·어깨 변형 제어점. 반환: (pts, disp, sigma, 최대 이동량, 어깨 반폭)."""
    pts: list[list[float]] = []
    disp: list[list[float]] = []
    P = lambda i: np.array(pt(i), dtype=np.float64)  # noqa: E731
    ls, rs, lh, rh = P(11), P(12), P(23), P(24)
    ms, mh = (ls + rs) / 2.0, (lh + rh) / 2.0
    sh_half = float(np.linalg.norm(ls - rs)) / 2.0
    hip_half = float(np.linalg.norm(lh - rh)) / 2.0
    torso = mh - ms
    t_len = float(np.linalg.norm(torso))
    if sh_half < 2.0 or t_len < 2.0:
        return pts, disp, 1.0, 0.0, sh_half
    down = torso / t_len
    lateral = np.array([down[1], -down[0]])
    sigma = 0.22 * sh_half
    moves: list[tuple[np.ndarray, np.ndarray]] = []
    anchors: list[np.ndarray] = []

    if waist_slim >= 0.01 and min(vis(23), vis(24)) >= 0.5:
        g = float(np.clip(waist_slim, 0.0, 1.0)) * _WAIST_SLIM_GAIN * _BODY_FIELD_COMP
        for tl in (0.3, 0.42, 0.54, 0.66, 0.78, 0.9, 1.0):
            prof = 0.5 * (1.0 + math.cos(math.pi * (tl - 0.62) / 0.38)) if abs(tl - 0.62) < 0.38 else 0.0
            c = ms + torso * tl
            guess = (1.0 - tl) * 0.85 * sh_half + tl * 1.6 * hip_half
            anchors.append(c)
            for sgn in (-1.0, 1.0):
                d = lateral * sgn
                e = _torso_edge(person, c, d, guess)
                edge = c + d * e
                if prof > 0.0:
                    moves.append((edge, -d * e * g * prof))
                else:
                    anchors.append(edge)
        # 팔은 움직이지 않는다 — 팔꿈치·손목 고정 (허리 옆 팔이 휘던 문제)
        for i in (13, 14, 15, 16):
            if vis(i) >= 0.5:
                anchors.append(P(i))

    if abs(shoulder_width) >= 0.01 and min(vis(11), vis(12)) >= 0.5:
        g = float(np.clip(shoulder_width, -1.0, 1.0)) * _SHOULDER_GAIN * _BODY_FIELD_COMP
        for s_pt, e_idx, w_idx in ((ls, 13, 15), (rs, 14, 16)):
            o = s_pt - ms
            o = o / max(float(np.linalg.norm(o)), 1e-6)
            mv = o * sh_half * g
            # 어깨 관절과 바깥 어깨선이 함께, 팔은 통째로 따라가며 아래로 갈수록 덜
            moves.append((s_pt, mv))
            moves.append((s_pt + o * 0.18 * sh_half, mv))
            if vis(e_idx) >= 0.5:
                moves.append((P(e_idx), mv * 0.6))
            if vis(w_idx) >= 0.5:
                moves.append((P(w_idx), mv * 0.3))
        # 목·가슴 중앙·얼굴은 제자리 (예전에는 턱까지 끌려갔다)
        anchors += [ms - down * 0.35 * sh_half, ms + down * 0.5 * sh_half, P(0), P(9), P(10)]

    if not moves:
        return pts, disp, sigma, 0.0, sh_half
    max_move = max(float(np.linalg.norm(m)) for _, m in moves) / _BODY_FIELD_COMP
    delta = max(4.0 * max_move, 0.2 * sh_half)
    body_c = (ms + mh) / 2.0
    for p, mv in moves:
        pts.append(list(p + mv))
        disp.append(list(-mv))
        out = p - body_c
        nrm = float(np.linalg.norm(out))
        if nrm > 1e-6:
            pts.append(list(p + out / nrm * delta))
            disp.append([0.0, 0.0])
    for a in anchors:
        pts.append(list(a))
        disp.append([0.0, 0.0])
    return pts, disp, sigma, max_move, sh_half


def apply_body_reshape(
    img: Image.Image,
    leg_stretch: float = 0.0,
    shoulder_width: float = 0.0,
    waist_slim: float = 0.0,
    cache: MediaPipeCache | None = None,
) -> Image.Image:
    """체형 보정 — MediaPipe Pose 33 랜드마크 + 인물 분할 기반.

    leg_stretch: 0~1 (허벅지~발목 세로 늘리기, 발 아래 바닥으로 흡수)
    shoulder_width: -1~1 (음수=좁게, 양수=넓게)
    waist_slim: 0~1 (허리 실루엣을 안쪽으로)

    바디 미감지 시 원본 반환. 다중 바디는 가장 큰 것만.
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
        v = getattr(pose[idx], "visibility", None)
        return 1.0 if v is None else float(v)

    result_arr = arr_rgb

    # ── 허리·어깨: 실루엣 기준 국소 워프 (한 번의 remap) ──
    if abs(shoulder_width) >= 0.01 or waist_slim >= 0.01:
        person = _cached_person_mask(cache, arr_rgb)
        pts, disp, sigma, max_move, sh_half = _build_body_controls(
            _pt, _vis, person, shoulder_width, waist_slim,
        )
        if max_move > 0.0:
            arr_pts = np.array(pts, np.float32)
            arr_disp = np.array(disp, np.float32)
            margin = 3.5 * sigma
            roi = (arr_pts[:, 0].min() - margin, arr_pts[:, 1].min() - margin,
                   arr_pts[:, 0].max() + margin, arr_pts[:, 1].max() + margin)
            band = max(4.0 * max_move, _SUPPORT_BAND_MIN * min(h, w))
            result_arr, _ = _remap_region(
                result_arr, roi,
                lambda xs, ys: _gaussian_field(xs, ys, arr_pts, arr_disp, sigma),
                _field_step(2.0 * sh_half), person, band,
            )

    # ── leg_stretch: 행 단위 세로 늘리기 ──
    if leg_stretch >= 0.01:
        # Pose: 23/24 힙, 27/28 발목, 29/30 뒤꿈치, 31/32 발끝
        ankles = [i for i in (27, 28) if _vis(i) >= 0.5]
        if ankles:
            hip_y = (_pt(23)[1] + _pt(24)[1]) / 2.0
            ankle_y = max(_pt(i)[1] for i in ankles)
            feet = [_pt(i)[1] for i in (29, 30, 31, 32) if _vis(i) >= 0.3]
            foot_y = max(feet) if feet else ankle_y + 0.06 * (ankle_y - hip_y)
            foot_y = max(foot_y, ankle_y)
            mapping = _leg_row_map(h, hip_y, ankle_y, foot_y, leg_stretch)
            if mapping is not None:
                row_map, r0 = mapping
                map_y = np.repeat(row_map[r0:, np.newaxis], w, axis=1)
                map_x = np.repeat(np.arange(w, dtype=np.float32)[np.newaxis, :], h - r0, axis=0)
                if result_arr is arr_rgb:
                    result_arr = arr_rgb.copy()
                result_arr[r0:] = cv2.remap(result_arr, map_x, map_y, cv2.INTER_LINEAR,
                                            borderMode=cv2.BORDER_REPLICATE)

    if result_arr is arr_rgb:
        return img
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
    if saturation <= _MONO_SATURATION:
        # 흑백 변환 — 색을 밝기로 옮긴 뒤 a·b를 완전히 중립으로 둔다.
        # 스케일(1+saturation)로 두면 -0.99가 1% 색을 남겨 완전한 흑백이 아니다.
        mix = (a_ch - 128.0) * _MONO_MIX_A + (b_ch - 128.0) * _MONO_MIX_B
        l_ch = l_ch + np.clip(mix, -_MONO_MIX_LIMIT, _MONO_MIX_LIMIT)
        a_ch = np.full_like(a_ch, 128.0)
        b_ch = np.full_like(b_ch, 128.0)
    elif abs(saturation) >= 0.01:
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


def _finite_float(raw: Any, default: float = 0.0) -> float:
    """숫자로 읽되 NaN·무한은 default로 본다.

    min/max는 NaN을 만나면 다른 쪽 인자를 돌려준다 — min(1.0, nan)은 1.0이다.
    클램핑만으로는 NaN이 버려지지 않고 **최대 보정**이 된다. json.loads가 표준
    밖의 NaN·Infinity 리터럴을 받아들이므로 클라이언트 JSON에서 실제로 닿는다.
    """
    try:
        val = float(raw)
    except (TypeError, ValueError):
        return default
    return val if math.isfinite(val) else default


def sanitize_tone_curve_points(raw: Any) -> list[tuple[float, float]] | None:
    """톤 커브 제어점을 [(x, y), ...] 오름차순으로 정리한다. 쓸 수 없으면 None.

    np.interp는 x가 오름차순이라고 가정하고 검사하지 않는다 — 뒤섞인 점은
    오류 없이 엉뚱한 곡선을 만든다. 숫자가 아니거나 NaN인 점은 버린다.
    """
    if not isinstance(raw, (list, tuple)):
        return None
    pts: list[tuple[float, float]] = []
    for item in raw:
        if not isinstance(item, (list, tuple)) or len(item) != 2:
            return None
        try:
            px, py = float(item[0]), float(item[1])
        except (TypeError, ValueError):
            return None
        if not (math.isfinite(px) and math.isfinite(py)):
            return None
        pts.append((px, py))
    if len(pts) < 2:
        return None
    return sorted(pts)


def sanitize_hsl_adjust(raw: Any) -> dict[str, dict[str, float]] | None:
    """HSL 조절값을 알려진 채널·키만 남기고 -1~1로 자른다. 비면 None."""
    if not isinstance(raw, dict):
        return None
    valid_channels = set(_HSL_CHANNELS.keys())
    hsl_parsed: dict[str, dict[str, float]] = {}
    for ch_name, ch_adj in raw.items():
        if ch_name not in valid_channels or not isinstance(ch_adj, dict):
            continue
        parsed_adj: dict[str, float] = {}
        for k in ("hue", "saturation", "lightness"):
            v = _finite_float(ch_adj.get(k, 0.0))
            parsed_adj[k] = round(max(-1.0, min(1.0, v)), 3)
        if any(abs(v) >= 0.01 for v in parsed_adj.values()):
            hsl_parsed[ch_name] = parsed_adj
    return hsl_parsed if hsl_parsed else None


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
        val = _finite_float(recommended.get(key, default), default)
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
        strength = _finite_float(tone_curve.get("strength", 0.0))
        params["tone_curve_strength"] = round(max(0.0, min(1.0, strength)), 3)
        params["tone_curve_points"] = sanitize_tone_curve_points(tone_curve.get("points"))
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
            params["split_shadow_hue"] = round(_finite_float(shadow.get("hue", 0.0)) % 360.0, 1)
            params["split_shadow_strength"] = round(
                max(0.0, min(1.0, _finite_float(shadow.get("strength", 0.0)))), 3
            )
        else:
            params["split_shadow_hue"] = 0.0
            params["split_shadow_strength"] = 0.0
        if isinstance(highlight, dict):
            params["split_highlight_hue"] = round(_finite_float(highlight.get("hue", 0.0)) % 360.0, 1)
            params["split_highlight_strength"] = round(
                max(0.0, min(1.0, _finite_float(highlight.get("strength", 0.0)))), 3
            )
        else:
            params["split_highlight_hue"] = 0.0
            params["split_highlight_strength"] = 0.0
    else:
        params.update(_split_defaults)

    # HSL 선택적 색상 파싱
    params["hsl_adjust"] = sanitize_hsl_adjust(recommended.get("hslAdjust"))

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
            rv = _finite_float(reshape.get(rkey, 0.0))
            params[rkey] = round(max(rrange[0], min(rrange[1], rv)), 3)

        # shoulder_width: -1.0 ~ 1.0
        sw = _finite_float(reshape.get("shoulder_width", 0.0))
        params["shoulder_width"] = round(max(-1.0, min(1.0, sw)), 3)
    else:
        params["face_slim"] = 0.0
        params["jaw_sharpen"] = 0.0
        params["eye_enlarge"] = 0.0
        params["leg_stretch"] = 0.0
        params["shoulder_width"] = 0.0
        params["waist_slim"] = 0.0

    return params


