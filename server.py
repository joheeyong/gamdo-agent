"""GAMDO Agent Server — FastAPI + Claude Code CLI."""

import base64
import logging
import math
import os
import re
import threading
from dataclasses import dataclass
from urllib.parse import urlencode

from dotenv import load_dotenv
from fastapi import Depends, FastAPI, Header, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse, RedirectResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

import httpx

import auth
import jobs
from models import (
    AnalyzeAndTransformRequest,
    AnalyzeAndTransformResponse,
    AnalyzeUserRequest,
    AnalyzeUserResponse,
    ApplyTransformRequest,
    ApplyTransformResponse,
    AutoTransformRequest,
    AutoTransformResponse,
    FirebaseTokenResponse,
    InstagramExchangeTokenRequest,
    InstagramExchangeTokenResponse,
    InstagramMediaRequest,
    InstagramMediaResponse,
    InstagramStoriesRequest,
    InstagramStoriesResponse,
    JobStartResponse,
    JobStatusResponse,
    ReferenceImagesResponse,
    SessionRequest,
    SessionResponse,
    TransformPhotoRequest,
    TransformPhotoResponse,
)
from claude_client import analyze_user, get_reference_image_paths, transform_photo
from param_engine import (
    build_params_with_comment,
    detect_tilt_angle,
    feed_compatibility,
    measure_color_analysis,
    measure_image_stats,
    measure_reference_target,
    prefix_tilt_comment,
)
from image_processor import (
    MediaPipeCache,
    estimate_keystone,
    analysis_to_transform_params,
    apply_all_transforms,
    apply_auto_edits,
    apply_regional_transforms,
    build_local_regions,
    decode_base64_image,
    detect_regions,
    encode_image_base64,
    gate_auto_edits,
    has_retouchable_face,
    sanitize_hsl_adjust,
    sanitize_tone_curve_points,
)

load_dotenv()

logging.basicConfig(level=logging.INFO)
# httpx는 INFO에서 요청 URL을 쿼리째 남긴다. Instagram 호출은 access_token과
# client_secret을 쿼리로 보내므로 그대로 두면 비밀값이 로그에 찍힌다.
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
log = logging.getLogger("gamdo-agent")

# 쿼리·본문에 실리는 비밀값. httpx 예외 메시지에 요청 URL이 통째로 들어간다.
# 인가 code는 쿼리(code=...) 형태만 가린다. Instagram 에러 JSON의 "code": 400은
# 에러 번호라 남겨 둔다.
_SECRET_PARAM_RE = re.compile(
    r"((?:access_token|client_secret|refresh_token|session_token|firebase_token)[\"']?\s*[=:]\s*[\"']?|\bcode=)"
    r"[^&\s\"',}]+",
    re.IGNORECASE,
)


def _redact(text: object) -> str:
    """로그·에러 응답에 남기기 전에 토큰류 값을 가린다."""
    return _SECRET_PARAM_RE.sub(r"\1***", str(text))

def _docs_kwargs() -> dict:
    """API 문서(/docs, /redoc, /openapi.json) 노출 여부.

    문서 페이지는 인증 없이 열려 엔드포인트·요청 스키마를 모두 보여 준다.
    운영에서는 끄고, 개발할 때만 GAMDO_ENABLE_DOCS=1로 켠다.
    """
    if os.getenv("GAMDO_ENABLE_DOCS", "").strip().lower() in auth._TRUTHY:
        return {}
    return {"docs_url": None, "redoc_url": None, "openapi_url": None}


app = FastAPI(title="GAMDO Agent", version="0.1.0", **_docs_kwargs())

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# GZip 압축: 500바이트 이상 응답을 자동 gzip 압축
# CORSMiddleware 뒤에 추가하여 CORS 헤더가 먼저 설정된 후 압축 적용
app.add_middleware(GZipMiddleware, minimum_size=500)


# 요청 본문 상한. 가장 큰 정상 요청은 앱이 보내는 사진 한 장(최대 5MB JPEG →
# base64 약 6.7MB)이라 넉넉히 잡는다. 상한이 없으면 수백 MB짜리 본문 하나가
# JSON 파싱과 base64 디코딩에서 메모리를 몇 배로 부풀린다.
_MAX_BODY_BYTES = int(float(os.getenv("GAMDO_MAX_BODY_MB", "50")) * 1024 * 1024)


class _BodySizeLimitMiddleware:
    """Content-Length가 상한을 넘으면 읽기 전에 413. 길이를 속이거나 chunked로
    보내는 경우를 위해 실제로 읽은 바이트도 센다."""

    def __init__(self, app, max_bytes: int):
        self.app = app
        self.max_bytes = max_bytes

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        for name, value in scope.get("headers", []):
            if name == b"content-length":
                try:
                    declared = int(value)
                except ValueError:
                    declared = 0
                if declared > self.max_bytes:
                    response = JSONResponse(
                        {"detail": "Request body too large"}, status_code=413
                    )
                    await response(scope, receive, send)
                    return

        received = 0

        async def limited_receive():
            nonlocal received
            message = await receive()
            if message["type"] == "http.request":
                received += len(message.get("body", b""))
                if received > self.max_bytes:
                    # FastAPI는 본문 읽기 중 난 HTTPException을 그대로 올려 보낸다
                    raise StarletteHTTPException(
                        status_code=413, detail="Request body too large"
                    )
            return message

        await self.app(scope, limited_receive, send)


# 마지막에 추가한 미들웨어가 가장 바깥이다 — 본문을 읽기 전에 거른다.
app.add_middleware(_BodySizeLimitMiddleware, max_bytes=_MAX_BODY_BYTES)

# 이미지 처리는 메모리를 많이 쓴다 (2560px 한 장에 피크 ~1.1GB).
# FastAPI는 동기 엔드포인트를 기본 40개 스레드까지 동시에 돌리므로,
# 제한이 없으면 동시 요청 몇 건에 프로세스가 죽는다.
_HEAVY_SLOTS = int(os.getenv("GAMDO_MAX_CONCURRENT", "3"))
_heavy_semaphore = threading.Semaphore(_HEAVY_SLOTS)

APP_TOKEN = os.getenv("APP_TOKEN", "")
INSTAGRAM_CLIENT_ID = os.getenv("INSTAGRAM_CLIENT_ID", "")
INSTAGRAM_CLIENT_SECRET = os.getenv("INSTAGRAM_CLIENT_SECRET", "")


# ── 인증 ──
#
# 규칙 (앱과의 계약, scratchpad/auth_contract.md):
# - APP_TOKEN과 일치하는 Bearer는 서비스 토큰 — 항상 통과, uid 제한 없음.
# - 유효한 세션 토큰이면 uid를 얻고, 요청의 user_id가 그와 다르면 403.
# - 세션이 없거나 무효면: GAMDO_AUTH_REQUIRED가 켜져 있으면 401,
#   아니면(과도기) 예전 동작 — APP_TOKEN이 설정돼 있으면 401, 없으면 통과.
#   과도기 통과를 막지 않는 이유는 헤더를 보내지 않는 기존 앱 빌드 때문이다.


class AuthError(Exception):
    """인증·인가 실패. 전용 핸들러가 {"success", "error", "error_code"} 형식으로 답한다."""

    def __init__(self, status_code: int, error_code: str, message: str):
        super().__init__(message)
        self.status_code = status_code
        self.error_code = error_code
        self.message = message


@app.exception_handler(AuthError)
async def _auth_error_handler(_request: Request, exc: AuthError):
    return JSONResponse(
        status_code=exc.status_code,
        content={"success": False, "error": exc.message, "error_code": exc.error_code},
    )


@dataclass(frozen=True)
class AuthContext:
    uid: str | None = None   # 유효한 세션의 Instagram user_id
    service: bool = False    # APP_TOKEN(서비스 토큰)으로 들어온 요청


def _session_invalid() -> AuthError:
    return AuthError(401, "session_invalid", "세션이 없거나 만료되었습니다")


def _authenticate(authorization: str | None, path: str = "") -> AuthContext:
    """Authorization 헤더를 해석한다. 토큰 값은 로그에 남기지 않는다.

    compare_digest를 쓰는 이유: ==는 앞에서부터 비교하다 처음 다른 곳에서
    멈춰, 응답 시간으로 토큰을 한 글자씩 알아낼 여지를 준다.
    """
    token = auth.parse_bearer(authorization)

    if token and APP_TOKEN and auth.tokens_equal(token, APP_TOKEN):
        return AuthContext(service=True)

    uid = auth.verify_session(token) if token else None
    if uid:
        return AuthContext(uid=uid)

    if auth.auth_required():
        raise _session_invalid()

    # 과도기: 기존 앱 빌드 호환
    if APP_TOKEN:
        raise _session_invalid()
    log.warning("auth: 세션 없는 요청을 과도기 규칙으로 통과시킴 (%s)", path or "-")
    return AuthContext()


def require_auth(request: Request, authorization: str | None = Header(None)) -> AuthContext:
    """FastAPI 의존성 — 인증 면제가 아닌 엔드포인트에 붙인다."""
    return _authenticate(authorization, request.url.path)


def require_session(authorization: str | None = Header(None)) -> AuthContext:
    """세션이 반드시 있어야 하는 엔드포인트용 (과도기에도). 서비스 토큰도 안 된다 — uid가 없다."""
    uid = auth.verify_session(auth.parse_bearer(authorization))
    if not uid:
        raise _session_invalid()
    return AuthContext(uid=uid)


def _check_user(ctx: AuthContext | None, user_id: str | None) -> None:
    """요청의 user_id가 세션 uid와 다르면 403. 세션이 없는 요청(서비스·과도기)은 검사하지 않는다.

    ctx가 AuthContext가 아니면(테스트가 엔드포인트 함수를 직접 부른 경우) 건너뛴다 —
    HTTP로 들어온 요청에는 FastAPI가 항상 require_auth 결과를 넣는다.
    """
    if not isinstance(ctx, AuthContext):
        return
    if ctx.uid and user_id and user_id != ctx.uid:
        raise AuthError(403, "forbidden_user", "다른 사용자의 데이터에는 접근할 수 없습니다")


@app.get("/health")
def health():
    return {"status": "ok", "service": "gamdo-agent"}


@app.post("/api/analyze-user", response_model=AnalyzeUserResponse)
def api_analyze_user(
    req: AnalyzeUserRequest,
    ctx: AuthContext = Depends(require_auth),
):
    """사용자의 게시글/피드/스토리를 분석하여 스타일 프로필을 반환합니다."""
    _check_user(ctx, req.user_id)

    try:
        log.info(
            "analyze-user: posts=%d, feeds=%d, stories=%d",
            len(req.posts), len(req.feeds), len(req.stories),
        )

        result = analyze_user(
            posts=[p.model_dump() for p in req.posts],
            feeds=[f.model_dump() for f in req.feeds],
            stories=[s.model_dump() for s in req.stories],
            user_id=req.user_id,
        )

        log.info("analyze-user: success")
        return AnalyzeUserResponse(success=True, data=result)

    except Exception as e:
        log.exception("analyze-user failed")
        return AnalyzeUserResponse(success=False, error=str(e))


@app.post("/api/transform-photo", response_model=TransformPhotoResponse)
def api_transform_photo(
    req: TransformPhotoRequest,
    ctx: AuthContext = Depends(require_auth),
):
    """사용자 스타일 프로필에 맞춰 사진 보정 가이드를 반환합니다."""
    try:
        log.info("transform-photo: style=%s", req.style_profile.get("primaryStyle", "unknown"))

        result = transform_photo(
            style_profile=req.style_profile,
            image_base64=req.image_base64,
            media_type=req.media_type,
        )

        log.info("transform-photo: success")
        return TransformPhotoResponse(success=True, data=result)

    except Exception as e:
        log.exception("transform-photo failed")
        return TransformPhotoResponse(success=False, error=str(e))


# ── 분석 + 변형 통합 API ──


# 자동 수평 보정의 하한(도). 기울기 감지기(param_engine._MIN_TILT)와 같은 값이다.
# 한때 1.0°로 올렸다가, 지평선·건물 모서리가 있는 사진의 0.5~0.9° 기울기를
# 더는 잡지 못한다는 사용자 보고로 되돌렸다 — 직선이 있으면 이 정도도 눈에 띈다.
_MIN_AUTO_STRAIGHTEN = 0.4


def _run_analyze_and_transform(
    req: AnalyzeAndTransformRequest,
    on_stage=None,
) -> AnalyzeAndTransformResponse:
    """분석 + 변형 본체. 동기 엔드포인트와 비동기 작업 워커가 함께 쓴다.

    on_stage(stage)는 진행 단계를 알린다: Claude 호출 직전 "analyzing",
    픽셀 처리(세마포어 대기 포함) 직전 "rendering". 인증·인가는 호출하는 쪽에서 한다.
    예외는 올리지 않고 success=False 응답으로 돌려준다 (기존 동작 그대로).
    """

    def stage(name: str) -> None:
        if on_stage is None:
            return
        try:
            on_stage(name)
        except Exception:
            log.warning("analyze-and-transform: stage callback failed (%s)", name)

    try:
        # 1. Claude가 사진 분석 (Vision)
        log.info("analyze-and-transform: analyzing photo")
        stage("analyzing")
        analysis = transform_photo(
            style_profile=req.style_profile,
            image_base64=req.image_base64,
            media_type=req.media_type,
            user_id=req.user_id,
        )
        log.info("analyze-and-transform: analysis complete")
        stage("rendering")

        # 메모리 폭증 방지 — 픽셀 처리만 제한한다. Claude 호출(수십 초)까지
        # 함께 묶으면 슬롯이 LLM 대기로 차서, 슬라이더를 움직이는 다른 사용자의
        # 미리보기가 그만큼 밀린다. 대기 중에는 메모리를 쓰지 않는다.
        with _heavy_semaphore:
            # 2. 이미지 디코딩
            img = decode_base64_image(req.image_base64)

            # 3. 색 분석 중 측정 가능한 값은 실제 픽셀에서 계산해 덮어쓴다.
            #    모델이 hex를 눈대중하는 것보다 k-means가 정확하다.
            color_analysis = analysis.get("colorAnalysis")
            if not isinstance(color_analysis, dict):
                color_analysis = {}
                analysis["colorAnalysis"] = color_analysis
            color_analysis.update(measure_color_analysis(img))

            # 4. 목표값은 사용자의 대표 사진에서 직접 잰다. 스타일 프로필의
            #    5단계 카테고리는 모델의 눈대중 위에 상수를 얹은 구조라,
            #    실제로 그 사람이 올리는 사진과 어긋날 수 있다.
            reference = measure_reference_target(
                get_reference_image_paths(req.user_id) if req.user_id else []
            )
            # 적합도 '이전' 값은 손대기 전의 사진으로 잰다. 아래에서 img가 영역
            # 보정·기하 보정 결과로 바뀌므로 지금 재 두지 않으면 before에
            # 보정이 섞인다.
            before_stats = measure_image_stats(img) if reference else None

            # 설명문이 "피부를 정리했다"고 말하려면 질감 보정이 걸릴 얼굴이 실제로
            # 있어야 한다 (뒷모습·먼 전신이면 보정이 아무것도 하지 않는다).
            face_detected = None
            if req.skin_retouch_enabled and str(analysis.get("subjectType") or "").strip() == "인물":
                try:
                    with MediaPipeCache() as face_cache:
                        face_detected = has_retouchable_face(img, cache=face_cache)
                except Exception as exc:
                    log.warning("analyze-and-transform: face check failed: %s", exc)

            # 보정 파라미터: 히스토그램 측정 + 목표값으로 산출.
            # 왜 그 값이 나왔는지 설명도 함께 만든다 (모델 호출 없음).
            analysis["recommendedParams"], params_comment = build_params_with_comment(
                img, req.style_profile, analysis,
                reference=reference, reshape_enabled=req.reshape_enabled,
                skin_retouch_enabled=req.skin_retouch_enabled,
                face_detected=face_detected,
            )
            params = analysis_to_transform_params(analysis)
            log.info("analyze-and-transform: params=%s", params)
            log.info("analyze-and-transform: comment=%s", params_comment)

            # 5. 수평 보정 각도는 Hough 직선 검출로 잰다.
            #    지평선·건물 모서리가 기준이 있으면 눈대중보다 정확하고,
            #    기준선이 없거나 선들이 제각각이면 None을 돌려 손대지 않는다.
            auto_edits = analysis.get("autoEdits")
            if not isinstance(auto_edits, dict):
                auto_edits = {}
            # 수직 원근(키스톤)은 건축물용 변형이다. 한쪽 끝을 가로로 늘리므로
            # 인물에 적용하면 몸이 옆으로 퍼지고 다리 비율이 무너진다.
            # 사람이 주인공인 사진에서는 건드리지 않는다.
            # param_engine(build_params_with_comment)과 같은 기준 — 공백이 붙은
            # "인물 "을 한쪽은 인물로, 한쪽은 아니라고 보면 결정이 갈린다.
            subject = str(analysis.get("subjectType") or "").strip()
            keystone = 0.0 if subject == "인물" else estimate_keystone(img)
            if abs(keystone) >= 0.02:
                auto_edits["keystone"] = keystone
                log.info("analyze-and-transform: keystone %.3f", keystone)

            measured_tilt = detect_tilt_angle(img)
            if measured_tilt is not None and abs(measured_tilt) < _MIN_AUTO_STRAIGHTEN:
                # 직선이 확신 있게 '거의 수평'이라고 말한 경우다. 모델 예비값으로
                # 넘어가지 않고 그대로 둔다.
                log.info("analyze-and-transform: tilt %.2f° below %.1f° — skipped",
                         measured_tilt, _MIN_AUTO_STRAIGHTEN)
                auto_edits["straighten"] = None
                measured_tilt = None
            elif measured_tilt is not None:
                auto_edits["straighten"] = measured_tilt
                log.info("analyze-and-transform: measured tilt %.2f°", measured_tilt)
            elif auto_edits.get("straighten") is not None:
                # 측정이 확신하지 못하면 모델의 판단을 쓰되 안전 범위로 묶는다
                try:
                    llm_tilt = float(auto_edits["straighten"])
                    if not math.isfinite(llm_tilt):
                        # min/max는 NaN을 만나면 다른 쪽 인자를 돌려준다 — 8°가 된다
                        raise ValueError("non-finite tilt")
                    llm_tilt = max(-8.0, min(8.0, llm_tilt))
                    if abs(llm_tilt) < _MIN_AUTO_STRAIGHTEN:
                        raise ValueError("imperceptible tilt")
                    auto_edits["straighten"] = llm_tilt
                    measured_tilt = llm_tilt
                    log.info("analyze-and-transform: using model tilt %.2f°", llm_tilt)
                except (TypeError, ValueError):
                    auto_edits["straighten"] = None
            # 구도를 바꾸는 편집(크롭·비율)은 서버가 거른다. 인스타가 받지 않는
            # 비율만 맞추고, 모델 크롭은 프레임을 충분히 남길 때만 자동 적용한다
            # (앱에서 원래 구도로 되돌릴 수 있다). 과한 크롭·비율은 제안으로만.
            gate_auto_edits(auto_edits, img.size, allow_vertical_crop=subject != "인물")
            analysis["autoEdits"] = auto_edits
            params_comment = prefix_tilt_comment(params_comment, measured_tilt)

            # MediaPipe 캐시를 요청 스코프로 공유 — detect_regions / apply_regional_transforms / apply_all_transforms 간 중복 호출 제거
            with MediaPipeCache() as mp_cache:
                # 7. 영역별 스마트 보정 (regionParams가 있으면)
                region_params_raw = analysis.get("regionParams")
                if region_params_raw and isinstance(region_params_raw, dict):
                    # null이 아닌 영역만 필터링
                    valid_region_params = {
                        k: v for k, v in region_params_raw.items()
                        if v is not None and isinstance(v, dict)
                    }
                    if valid_region_params:
                        try:
                            regions = detect_regions(img, cache=mp_cache)
                            # 모델이 좌표로 짚은 국소 보정 영역(local_*)은 감지가 아니라
                            # 기하 정보로 만든다. detect_regions의 하늘/얼굴/배경 위에 얹힌다.
                            regions.update(build_local_regions(img.size, valid_region_params))
                            img = apply_regional_transforms(img, regions, valid_region_params, cache=mp_cache)
                            log.info("analyze-and-transform: applied regional transforms for regions=%s",
                                     list(valid_region_params.keys()))
                        except Exception as e:
                            log.warning("analyze-and-transform: regional transforms failed, falling back: %s", e)

                # 7. AI autoEdits 적용 (요소 제거 → 수평·원근 보정 → 크롭 → 비율)
                #
                # 영역 보정 다음에 온다. local_* 좌표는 모델이 원본 프레임을 보고
                # 짚은 것인데, 기하 보정이 먼저 돌면 프레임이 회전·크롭되어 그
                # 좌표가 엉뚱한 곳을 가리킨다 (실측: 날아간 창문 대신 평평한 벽에
                # 어두운 사각형이 찍혔다). 픽셀 편집을 원본 프레임에서 끝내고
                # 그 다음에 프레임을 다시 잡는다.
                if auto_edits:
                    # 인물은 위아래를 자르지 않는다 (머리·발이 잘려 다리가 짧아 보인다).
                    # 판단을 autoEdits에 적어 둔다 — 앱이 저장·미리보기에서 이
                    # 딕셔너리를 되돌려 보내므로 그 경로에도 같은 결정이 적용된다.
                    auto_edits["allow_vertical_crop"] = subject != "인물"
                    analysis["autoEdits"] = auto_edits
                    log.info("analyze-and-transform: applying autoEdits=%s", auto_edits)
                    img = apply_auto_edits(img, auto_edits)

                # 8. 슬라이더 변형 적용
                transformed = apply_all_transforms(img, cache=mp_cache, **params)
            result_b64 = encode_image_base64(transformed)

            # 피드 적합도: 모델의 추측이 아니라 대표 사진과의 실제 거리
            if reference:
                before = feed_compatibility(before_stats, reference)
                after = feed_compatibility(measure_image_stats(transformed), reference)
                analysis["feedCompatibility"] = after
                analysis["feedCompatibilityBefore"] = before
                log.info("analyze-and-transform: feed compatibility %s → %s", before, after)

        log.info("analyze-and-transform: success")
        return AnalyzeAndTransformResponse(
            success=True,
            analysis=analysis,
            image_base64=result_b64,
            params=params,
            params_comment=params_comment,
        )

    except Exception as e:
        log.exception("analyze-and-transform failed")
        return AnalyzeAndTransformResponse(success=False, error=str(e))


@app.post("/api/analyze-and-transform", response_model=AnalyzeAndTransformResponse)
def api_analyze_and_transform(
    req: AnalyzeAndTransformRequest,
    ctx: AuthContext = Depends(require_auth),
):
    """사진 분석 + 변형을 한 번에 수행. Claude가 사진을 분석하고, 결과를 바탕으로 즉시 변형."""
    _check_user(ctx, req.user_id)
    return _run_analyze_and_transform(req)


# ── 비동기 분석 작업 API (scratchpad/jobs_contract.md) ──
#
# 30~70초짜리 분석을 긴 연결 하나에 묶지 않는다. 앱이 백그라운드로 가서
# 소켓이 끊겨도 작업은 계속 돌고, 앱은 돌아와 job_id로 결과를 받는다.

_job_store = jobs.JobStore()


def _job_not_found() -> AuthError:
    # 인증 에러와 같은 본문 형식을 쓰려고 AuthError 핸들러를 빌린다
    return AuthError(404, "job_not_found", "작업이 없거나 만료되었습니다")


def _ctx_uid(ctx) -> str | None:
    return ctx.uid if isinstance(ctx, AuthContext) else None


def _analyze_job_fn(req: AnalyzeAndTransformRequest):
    def run(on_stage) -> dict:
        resp = _run_analyze_and_transform(req, on_stage=on_stage)
        if not resp.success:
            # 동기 엔드포인트의 error는 str(e) 그대로라 내부 정보가 섞일 수 있다.
            # 작업 조회에는 사람용 메시지만 싣는다 (원인은 위에서 이미 로그에 남았다).
            raise jobs.JobFailed()
        return resp.model_dump()

    return run


@app.post(
    "/api/jobs/analyze-and-transform",
    response_model=JobStartResponse,
    status_code=202,
)
def api_start_analyze_job(
    req: AnalyzeAndTransformRequest,
    ctx: AuthContext = Depends(require_auth),
):
    """analyze-and-transform을 백그라운드 작업으로 시작하고 job_id를 돌려준다."""
    _check_user(ctx, req.user_id)
    owner = _ctx_uid(ctx)
    # 세션 uid도 키에 넣는다 — user_id가 빈 요청이 다른 사용자의 작업으로
    # 합쳐지면 조회가 404가 되기 때문이다.
    key = jobs.dedupe_key(
        req.image_base64, req.style_profile, req.user_id,
        req.media_type, req.reshape_enabled, owner, req.skin_retouch_enabled,
    )
    try:
        snap = _job_store.submit(key, owner, _analyze_job_fn(req))
    except jobs.JobQueueFull:
        log.warning("jobs: 대기열이 가득 차 거절")
        raise AuthError(503, "busy", "요청이 많아 잠시 후 다시 시도해 주세요")
    log.info("jobs: analyze-and-transform job %s (%s)", snap.job_id[:8], snap.status)
    return JobStartResponse(success=True, job_id=snap.job_id, status=snap.status)


@app.get("/api/jobs/{job_id}", response_model=JobStatusResponse)
def api_get_job(
    job_id: str,
    ctx: AuthContext = Depends(require_auth),
):
    """작업 상태 조회 (폴링). 없거나 만료됐거나 남의 작업이면 404."""
    snap = _job_store.get(job_id)
    if snap is None:
        raise _job_not_found()
    # 세션 uid로 만든 작업은 같은 uid만 본다. 존재 여부도 드러내지 않는다.
    if snap.owner and _ctx_uid(ctx) != snap.owner:
        raise _job_not_found()
    return JobStatusResponse(
        success=True,
        job_id=snap.job_id,
        status=snap.status,
        stage=snap.stage,
        elapsed_sec=snap.elapsed_sec,
        result=snap.result if snap.status == jobs.DONE else None,
        error=snap.error if snap.status == jobs.ERROR else None,
    )


# ── 이미지 자동 변형 API ──


@app.post("/api/auto-transform", response_model=AutoTransformResponse)
def api_auto_transform(
    req: AutoTransformRequest,
    ctx: AuthContext = Depends(require_auth),
):
    """AI 분석 기반 자동 변형. 분석 결과와 스타일 프로필로 파라미터를 계산하여 변형."""
    try:
        log.info("auto-transform: computing params from analysis")

        # 이 엔드포인트도 같은 파이프라인을 같은 해상도로 돌린다 —
        # 제한 밖에 두면 여기로 들어온 요청이 메모리 상한을 넘길 수 있다.
        with _heavy_semaphore:
            # 1. 이미지 디코딩
            img = decode_base64_image(req.image_base64)

            # 2. 분석 → 파라미터. 분석 JSON에 recommendedParams가 있으면(옛 형식)
            #    그대로 쓰고, 없으면 측정 + 프로필로 계산한다.
            analysis = dict(req.analysis)
            params_comment = None
            if not isinstance(analysis.get("recommendedParams"), dict):
                analysis["recommendedParams"], params_comment = build_params_with_comment(
                    img, req.style_profile, analysis
                )
            params = analysis_to_transform_params(analysis)
            log.info("auto-transform: params=%s", params)

            # 3. AI autoEdits 적용 (크롭, 요소 제거, 인스타 비율)
            auto_edits = analysis.get("autoEdits", {})
            if auto_edits and isinstance(auto_edits, dict):
                log.info("auto-transform: applying autoEdits=%s", auto_edits)
                img = apply_auto_edits(img, auto_edits)

            # 4. 슬라이더 변형 적용
            transformed = apply_all_transforms(img, **params)
            result_b64 = encode_image_base64(transformed)

        log.info("auto-transform: success")
        return AutoTransformResponse(
            success=True,
            image_base64=result_b64,
            params=params,
            params_comment=params_comment,
        )

    except Exception as e:
        log.exception("auto-transform failed")
        return AutoTransformResponse(success=False, error=str(e))


@app.post("/api/apply-transform", response_model=ApplyTransformResponse)
def api_apply_transform(
    req: ApplyTransformRequest,
    ctx: AuthContext = Depends(require_auth),
):
    """슬라이더 값으로 수동 변형. 원본에서 항상 새로 적용 (누적 열화 방지)."""
    try:
        params = {
            "brightness": req.brightness,
            "contrast": req.contrast,
            "clarity": req.clarity,
            "dehaze": req.dehaze,
            "highlights": req.highlights,
            "shadows": req.shadows,
            "saturation": req.saturation,
            "temperature": req.temperature,
            "blemish_removal": req.blemish_removal,
            "skin_smoothing": req.skin_smoothing,
            "vignette": req.vignette,
            "sharpness": req.sharpness,
            "grain": req.grain,
            "auto_wb": req.auto_wb,
            "denoise": req.denoise,
            "background_blur": req.background_blur,
            "tone_curve_preset": req.tone_curve_preset,
            "tone_curve_strength": req.tone_curve_strength,
            # 분석 경로와 같은 정리를 거친다. 뒤섞인 점은 np.interp가 검사 없이
            # 엉뚱한 곡선을 만들고, 문자열 값은 변환 중 TypeError가 난다.
            "tone_curve_points": sanitize_tone_curve_points(req.tone_curve_points),
            "split_shadow_hue": req.split_shadow_hue,
            "split_shadow_strength": req.split_shadow_strength,
            "split_highlight_hue": req.split_highlight_hue,
            "split_highlight_strength": req.split_highlight_strength,
            "hsl_adjust": sanitize_hsl_adjust(req.hsl_adjust),
            "face_slim": req.face_slim,
            "jaw_sharpen": req.jaw_sharpen,
            "eye_enlarge": req.eye_enlarge,
            "leg_stretch": req.leg_stretch,
            "shoulder_width": req.shoulder_width,
            "waist_slim": req.waist_slim,
        }
        log.info("apply-transform: params=%s, preview=%s", params, req.preview)

        # analyze-and-transform과 같은 제한을 받아야 한다. 저장 경로도 같은
        # 해상도로 같은 파이프라인(reshape 포함)을 돌려 메모리 피크가 같고,
        # 슬라이더를 움직이는 동안 미리보기 요청이 연달아 들어온다.
        with _heavy_semaphore:
            img = decode_base64_image(req.image_base64)

            with MediaPipeCache() as mp_cache:
                valid_regions = {
                    k: v for k, v in (req.region_params or {}).items()
                    if isinstance(v, dict)
                }
                # 미리보기에서도 영역 보정을 적용한다. 건너뛰면 미리보기와 저장본의
                # 색이 갈린다 — 비싼 것은 잡티·스무딩뿐이고 그건 preview가 걸러낸다.
                if valid_regions:
                    try:
                        regions = detect_regions(img, cache=mp_cache)
                        regions.update(build_local_regions(img.size, valid_regions))
                        img = apply_regional_transforms(
                            img, regions, valid_regions, cache=mp_cache, preview=req.preview
                        )
                    except Exception as exc:
                        log.warning("apply-transform: regional transforms failed: %s", exc)

                # 기하 보정은 영역 보정 뒤에 — analyze-and-transform과 같은 순서다.
                # local_* 좌표가 원본 프레임 기준이므로 먼저 프레임을 바꾸면 안 된다.
                # 슬라이더만 적용하면 저장본에서 수평·크롭·영역 보정이 사라진다.
                if req.auto_edits:
                    img = apply_auto_edits(img, req.auto_edits)
                transformed = apply_all_transforms(
                    img, preview=req.preview, cache=mp_cache, **params
                )
            result_b64 = encode_image_base64(transformed)

        log.info("apply-transform: success")
        return ApplyTransformResponse(
            success=True,
            image_base64=result_b64,
            params_applied=params,
        )

    except Exception as e:
        log.exception("apply-transform failed")
        return ApplyTransformResponse(success=False, error=str(e))


# ── 대표 사진 조회 API ──


@app.get("/api/reference-images/{user_id}", response_model=ReferenceImagesResponse)
def api_reference_images(
    user_id: str,
    ctx: AuthContext = Depends(require_auth),
):
    """사용자의 대표 사진 3장을 base64로 반환합니다."""
    _check_user(ctx, user_id)

    try:
        paths = get_reference_image_paths(user_id)
        if not paths:
            return ReferenceImagesResponse(success=True, images=[])

        images_b64: list[str] = []
        for p in paths:
            with open(p, "rb") as f:
                images_b64.append(base64.b64encode(f.read()).decode())

        log.info("reference-images: returned %d images for user %s", len(images_b64), user_id)
        return ReferenceImagesResponse(success=True, images=images_b64)

    except Exception as e:
        log.exception("reference-images failed")
        return ReferenceImagesResponse(success=False, error=str(e))


# ── Instagram API ──

INSTAGRAM_TOKEN_URL = "https://api.instagram.com/oauth/access_token"
INSTAGRAM_GRAPH_URL = "https://graph.instagram.com"
INSTAGRAM_APP_REDIRECT = "gamdo://oauth/instagram"


@app.get("/api/instagram/callback")
def api_instagram_callback(code: str = Query(...), state: str = Query(default="")):
    """Instagram OAuth 콜백 → 앱 커스텀 스킴으로 리디렉션."""
    log.info("instagram/callback: received code, redirecting to app")
    # 값을 그대로 이어 붙이면 '&'·'#'이 든 값이 쿼리를 깨거나 파라미터를 끼워 넣는다
    query = {"code": code}
    if state:
        query["state"] = state
    return RedirectResponse(url=f"{INSTAGRAM_APP_REDIRECT}?{urlencode(query)}")


@app.post("/api/instagram/exchange-token", response_model=InstagramExchangeTokenResponse)
def api_instagram_exchange_token(req: InstagramExchangeTokenRequest):
    """Authorization code → short-lived access_token 교환 (client_secret 보호).

    인증 면제 — 앱은 아직 세션이 없는 상태로 부른다. 성공하면 감도 세션도 함께 발급한다.
    """
    if not INSTAGRAM_CLIENT_ID or not INSTAGRAM_CLIENT_SECRET:
        return InstagramExchangeTokenResponse(
            success=False,
            error="Instagram client credentials not configured on server",
        )

    try:
        log.info("instagram/exchange-token: exchanging code")

        with httpx.Client(timeout=30) as client:
            # Short-lived token 교환
            resp = client.post(
                INSTAGRAM_TOKEN_URL,
                data={
                    "client_id": INSTAGRAM_CLIENT_ID,
                    "client_secret": INSTAGRAM_CLIENT_SECRET,
                    "grant_type": "authorization_code",
                    "redirect_uri": req.redirect_uri,
                    "code": req.code,
                },
            )
            resp.raise_for_status()
            token_data = resp.json()

        access_token = token_data.get("access_token", "")
        user_id = token_data.get("user_id", "")

        if not access_token:
            return InstagramExchangeTokenResponse(
                success=False,
                error=f"No access_token in response: {_redact(token_data)}",
            )

        # Long-lived token 교환
        try:
            with httpx.Client(timeout=30) as client:
                ll_resp = client.get(
                    f"{INSTAGRAM_GRAPH_URL}/access_token",
                    params={
                        "grant_type": "ig_exchange_token",
                        "client_secret": INSTAGRAM_CLIENT_SECRET,
                        "access_token": access_token,
                    },
                )
                ll_resp.raise_for_status()
                ll_data = ll_resp.json()
                access_token = ll_data.get("access_token", access_token)
                log.info("instagram/exchange-token: upgraded to long-lived token")
        except Exception as e:
            log.warning("Long-lived token exchange failed, using short-lived: %s", _redact(e))

        log.info("instagram/exchange-token: success, user_id=%s", user_id)
        data = {"access_token": access_token, "user_id": str(user_id)}
        # user_id는 client_secret으로 서버가 직접 받은 값이라 믿을 수 있다.
        # 세션 미설정 서버면 두 필드를 null로 둔다 (앱은 예전처럼 동작).
        data["session_token"], data["session_expires_at"] = _try_issue_session(str(user_id))
        return InstagramExchangeTokenResponse(success=True, data=data)

    except httpx.HTTPStatusError as e:
        # 예외 메시지에 요청 URL(access_token·client_secret 쿼리 포함)이 들어가므로
        # 트레이스백 대신 가린 메시지만 남긴다
        log.error("instagram/exchange-token HTTP error: %s", _redact(e))
        body = _redact(e.response.text)
        return InstagramExchangeTokenResponse(
            success=False, error=f"Instagram API error: {body}"
        )
    except Exception as e:
        log.exception("instagram/exchange-token failed")
        return InstagramExchangeTokenResponse(success=False, error=_redact(e))


@app.post("/api/instagram/media", response_model=InstagramMediaResponse)
def api_instagram_media(req: InstagramMediaRequest):
    """Instagram 미디어 목록을 프록시 조회 (페이지네이션 포함).

    인증 면제 — 요청의 IG access_token 자체가 자격증명이다.
    """
    try:
        log.info("instagram/media: fetching media list")

        fields = "id,caption,media_type,media_url,thumbnail_url,timestamp,permalink"
        all_items: list[dict] = []
        max_pages = 5  # 최대 5페이지까지 조회

        with httpx.Client(timeout=30) as client:
            url = f"{INSTAGRAM_GRAPH_URL}/me/media"
            params = {
                "fields": fields,
                "limit": "50",
                "access_token": req.access_token,
            }

            for page in range(max_pages):
                resp = client.get(url, params=params)
                resp.raise_for_status()
                data = resp.json()

                items = data.get("data", [])
                all_items.extend(items)
                log.info("instagram/media: page %d fetched %d items", page + 1, len(items))

                # 다음 페이지 확인
                next_url = data.get("paging", {}).get("next")
                if not next_url:
                    break
                # 다음 페이지는 전체 URL이므로 직접 사용
                url = next_url
                params = {}  # next URL에 params가 포함되어 있음

        log.info("instagram/media: total %d items", len(all_items))
        return InstagramMediaResponse(success=True, data=all_items)

    except httpx.HTTPStatusError as e:
        # 예외 메시지에 요청 URL(access_token·client_secret 쿼리 포함)이 들어가므로
        # 트레이스백 대신 가린 메시지만 남긴다
        log.error("instagram/media HTTP error: %s", _redact(e))
        body = _redact(e.response.text)
        return InstagramMediaResponse(
            success=False, error=f"Instagram API error: {body}"
        )
    except Exception as e:
        log.exception("instagram/media failed")
        return InstagramMediaResponse(success=False, error=_redact(e))


@app.post("/api/instagram/stories", response_model=InstagramStoriesResponse)
def api_instagram_stories(req: InstagramStoriesRequest):
    """Instagram 스토리 목록을 프록시 조회. 인증 면제 (IG access_token이 자격증명)."""
    try:
        log.info("instagram/stories: fetching stories")

        fields = "id,caption,media_type,media_url,thumbnail_url,timestamp,permalink"

        with httpx.Client(timeout=30) as client:
            resp = client.get(
                f"{INSTAGRAM_GRAPH_URL}/me/stories",
                params={
                    "fields": fields,
                    "access_token": req.access_token,
                },
            )
            resp.raise_for_status()
            data = resp.json()

        story_items = data.get("data", [])
        log.info("instagram/stories: fetched %d items", len(story_items))

        return InstagramStoriesResponse(success=True, data=story_items)

    except httpx.HTTPStatusError as e:
        # 예외 메시지에 요청 URL(access_token·client_secret 쿼리 포함)이 들어가므로
        # 트레이스백 대신 가린 메시지만 남긴다
        log.error("instagram/stories HTTP error: %s", _redact(e))
        body = _redact(e.response.text)
        return InstagramStoriesResponse(
            success=False, error=f"Instagram API error: {body}"
        )
    except Exception as e:
        log.exception("instagram/stories failed")
        return InstagramStoriesResponse(success=False, error=_redact(e))


# ── 세션 API ──


def _try_issue_session(uid: str) -> tuple[str | None, int | None]:
    """세션 발급. 비밀키가 없거나 uid가 비었으면 (None, None)."""
    if not uid:
        return None, None
    try:
        return auth.issue_session(uid)
    except (auth.SessionNotConfigured, ValueError):
        return None, None


@app.post("/api/session", response_model=SessionResponse)
def api_session(req: SessionRequest):
    """Instagram long-lived 토큰을 확인하고 감도 세션을 발급한다. 인증 면제.

    앱은 로그인 복원 시(세션이 없거나 만료 7일 이내)와 API가 session_invalid로
    401을 줬을 때 부른다.
    """
    if not auth.session_configured():
        raise AuthError(503, "session_not_configured", "서버에 세션 비밀키가 설정되지 않았습니다")

    access_token = (req.access_token or "").strip()
    if not access_token:
        raise AuthError(401, "instagram_token_invalid", "Instagram 토큰이 유효하지 않습니다")

    try:
        with httpx.Client(timeout=15) as client:
            # 토큰은 쿼리 대신 헤더로 — URL은 프록시·예외 메시지에 그대로 남는다
            resp = client.get(
                f"{INSTAGRAM_GRAPH_URL}/me",
                params={"fields": "id,user_id,username"},
                headers={"Authorization": f"Bearer {access_token}"},
            )
    except httpx.HTTPError as e:
        log.error("session: Instagram 호출 실패: %s", _redact(e))
        raise AuthError(502, "instagram_unavailable", "Instagram에 연결할 수 없습니다")

    # Instagram은 무효·만료 토큰에 400(OAuthException code 190)을 준다
    if resp.status_code in (400, 401, 403):
        log.info("session: Instagram 토큰 거부 (HTTP %d)", resp.status_code)
        raise AuthError(401, "instagram_token_invalid", "Instagram 토큰이 유효하지 않습니다")
    if resp.status_code != 200:
        log.error("session: Instagram HTTP %d: %s", resp.status_code, _redact(resp.text[:300]))
        raise AuthError(502, "instagram_unavailable", "Instagram 응답이 올바르지 않습니다")

    try:
        me = resp.json()
    except ValueError:
        me = None
    # exchange-token이 준 user_id(토큰 교환 응답)와 /me의 user_id가 서로 다른 ID
    # 체계일 수 있다. 앱이 저장해 둔 값이 이 토큰의 id·user_id 중 하나와 맞으면
    # 그 값으로 발급해, 로그인 때 세션·RTDB 경로(users/{userId})와 uid를 일치시킨다.
    candidates: list[str] = []
    if isinstance(me, dict):
        for key in ("user_id", "id"):
            v = str(me.get(key) or "").strip()
            if v and v not in candidates:
                candidates.append(v)
    if not candidates:
        log.error("session: Instagram 응답에 user_id가 없음")
        raise AuthError(502, "instagram_unavailable", "Instagram 응답에 사용자 ID가 없습니다")

    claimed = (req.user_id or "").strip()
    if claimed:
        if claimed not in candidates:
            log.warning("session: 요청 user_id가 토큰 소유자와 다름")
            raise AuthError(403, "forbidden_user", "다른 사용자의 세션은 발급할 수 없습니다")
        uid = claimed
    else:
        uid = candidates[0]

    token, exp = auth.issue_session(uid)
    log.info("session: issued for user %s", uid)
    return SessionResponse(
        success=True,
        data={"session_token": token, "session_expires_at": exp, "user_id": uid},
    )


@app.post("/api/firebase-token", response_model=FirebaseTokenResponse)
def api_firebase_token(ctx: AuthContext = Depends(require_session)):
    """세션 uid로 Firebase 커스텀 토큰을 만든다 (RTDB 규칙의 auth.uid = Instagram user_id).

    서비스 계정이 없거나 firebase-admin이 없으면 503 — 앱은 Firebase Auth 없이 진행한다.
    """
    try:
        token = auth.create_firebase_token(ctx.uid)
    except Exception as e:
        log.error("firebase-token: 발급 실패: %s", type(e).__name__)
        token = None
    if not token:
        raise AuthError(503, "firebase_not_configured", "서버에 Firebase가 설정되지 않았습니다")
    log.info("firebase-token: issued for user %s", ctx.uid)
    return FirebaseTokenResponse(success=True, data={"firebase_token": token})


if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("PORT", "8000"))
    uvicorn.run("server:app", host="0.0.0.0", port=port, reload=True)
