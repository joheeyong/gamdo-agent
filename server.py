"""GAMDO Agent Server — FastAPI + Claude Code CLI."""

import base64
import logging
import os
import threading

from dotenv import load_dotenv
from fastapi import FastAPI, Header, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import RedirectResponse

import httpx

from models import (
    AnalyzeAndTransformRequest,
    AnalyzeAndTransformResponse,
    AnalyzeUserRequest,
    AnalyzeUserResponse,
    ApplyTransformRequest,
    ApplyTransformResponse,
    AutoTransformRequest,
    AutoTransformResponse,
    InstagramExchangeTokenRequest,
    InstagramExchangeTokenResponse,
    InstagramMediaRequest,
    InstagramMediaResponse,
    InstagramStoriesRequest,
    InstagramStoriesResponse,
    ReferenceImagesResponse,
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
    decode_base64_image,
    detect_regions,
    encode_image_base64,
)

load_dotenv()

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("gamdo-agent")

app = FastAPI(title="GAMDO Agent", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# GZip 압축: 500바이트 이상 응답을 자동 gzip 압축
# CORSMiddleware 뒤에 추가하여 CORS 헤더가 먼저 설정된 후 압축 적용
app.add_middleware(GZipMiddleware, minimum_size=500)

# 이미지 처리는 메모리를 많이 쓴다 (2560px 한 장에 피크 ~1.1GB).
# FastAPI는 동기 엔드포인트를 기본 40개 스레드까지 동시에 돌리므로,
# 제한이 없으면 동시 요청 몇 건에 프로세스가 죽는다.
_HEAVY_SLOTS = int(os.getenv("GAMDO_MAX_CONCURRENT", "3"))
_heavy_semaphore = threading.Semaphore(_HEAVY_SLOTS)

APP_TOKEN = os.getenv("APP_TOKEN", "")
INSTAGRAM_CLIENT_ID = os.getenv("INSTAGRAM_CLIENT_ID", "")
INSTAGRAM_CLIENT_SECRET = os.getenv("INSTAGRAM_CLIENT_SECRET", "")


def _verify_token(authorization: str | None = Header(None)):
    """Bearer 토큰 검증."""
    if not APP_TOKEN:
        return
    if not authorization:
        raise HTTPException(status_code=401, detail="Authorization header required")
    token = authorization.replace("Bearer ", "")
    if token != APP_TOKEN:
        raise HTTPException(status_code=401, detail="Invalid token")


@app.get("/health")
def health():
    return {"status": "ok", "service": "gamdo-agent"}


@app.post("/api/analyze-user", response_model=AnalyzeUserResponse)
def api_analyze_user(
    req: AnalyzeUserRequest,
    authorization: str | None = Header(None),
):
    """사용자의 게시글/피드/스토리를 분석하여 스타일 프로필을 반환합니다."""
    _verify_token(authorization)

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
    authorization: str | None = Header(None),
):
    """사용자 스타일 프로필에 맞춰 사진 보정 가이드를 반환합니다."""
    _verify_token(authorization)

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


@app.post("/api/analyze-and-transform", response_model=AnalyzeAndTransformResponse)
def api_analyze_and_transform(
    req: AnalyzeAndTransformRequest,
    authorization: str | None = Header(None),
):
    """사진 분석 + 변형을 한 번에 수행. Claude가 사진을 분석하고, 결과를 바탕으로 즉시 변형."""
    _verify_token(authorization)

    # 메모리 폭증 방지 — 동시에 도는 무거운 처리를 제한한다
    with _heavy_semaphore:
      try:
        # 1. Claude가 사진 분석 (Vision)
        log.info("analyze-and-transform: analyzing photo")
        analysis = transform_photo(
            style_profile=req.style_profile,
            image_base64=req.image_base64,
            media_type=req.media_type,
            user_id=req.user_id,
        )
        log.info("analyze-and-transform: analysis complete")

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

        # 보정 파라미터: 히스토그램 측정 + 목표값으로 산출.
        # 왜 그 값이 나왔는지 설명도 함께 만든다 (모델 호출 없음).
        analysis["recommendedParams"], params_comment = build_params_with_comment(
            img, req.style_profile, analysis,
            reference=reference, reshape_enabled=req.reshape_enabled,
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
        subject = str(analysis.get("subjectType") or "")
        keystone = 0.0 if subject == "인물" else estimate_keystone(img)
        if abs(keystone) >= 0.02:
            auto_edits["keystone"] = keystone
            log.info("analyze-and-transform: keystone %.3f", keystone)

        measured_tilt = detect_tilt_angle(img)
        if measured_tilt is not None:
            auto_edits["straighten"] = measured_tilt
            log.info("analyze-and-transform: measured tilt %.2f°", measured_tilt)
        elif auto_edits.get("straighten") is not None:
            # 측정이 확신하지 못하면 모델의 판단을 쓰되 안전 범위로 묶는다
            try:
                llm_tilt = max(-8.0, min(8.0, float(auto_edits["straighten"])))
                auto_edits["straighten"] = llm_tilt
                measured_tilt = llm_tilt
                log.info("analyze-and-transform: using model tilt %.2f°", llm_tilt)
            except (TypeError, ValueError):
                auto_edits["straighten"] = None
        analysis["autoEdits"] = auto_edits
        params_comment = prefix_tilt_comment(params_comment, measured_tilt)

        # 6. AI autoEdits 적용 (수평 보정, 크롭, 요소 제거, 인스타 비율)
        if auto_edits:
            log.info("analyze-and-transform: applying autoEdits=%s", auto_edits)
            # 인물은 위아래를 자르지 않는다 (머리·발이 잘려 다리가 짧아 보인다)
            img = apply_auto_edits(
                img, auto_edits, allow_vertical_crop=(subject != "인물")
            )

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
                        img = apply_regional_transforms(img, regions, valid_region_params, cache=mp_cache)
                        log.info("analyze-and-transform: applied regional transforms for regions=%s",
                                 list(valid_region_params.keys()))
                    except Exception as e:
                        log.warning("analyze-and-transform: regional transforms failed, falling back: %s", e)

            # 8. 슬라이더 변형 적용
            transformed = apply_all_transforms(img, cache=mp_cache, **params)
        result_b64 = encode_image_base64(transformed)

        # 피드 적합도: 모델의 추측이 아니라 대표 사진과의 실제 거리
        if reference:
            before = feed_compatibility(measure_image_stats(img), reference)
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


# ── 이미지 자동 변형 API ──


@app.post("/api/auto-transform", response_model=AutoTransformResponse)
def api_auto_transform(
    req: AutoTransformRequest,
    authorization: str | None = Header(None),
):
    """AI 분석 기반 자동 변형. 분석 결과와 스타일 프로필로 파라미터를 계산하여 변형."""
    _verify_token(authorization)

    try:
        log.info("auto-transform: computing params from analysis")

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
    authorization: str | None = Header(None),
):
    """슬라이더 값으로 수동 변형. 원본에서 항상 새로 적용 (누적 열화 방지)."""
    _verify_token(authorization)

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
            "tone_curve_points": req.tone_curve_points,
            "split_shadow_hue": req.split_shadow_hue,
            "split_shadow_strength": req.split_shadow_strength,
            "split_highlight_hue": req.split_highlight_hue,
            "split_highlight_strength": req.split_highlight_strength,
            "hsl_adjust": req.hsl_adjust,
            "face_slim": req.face_slim,
            "jaw_sharpen": req.jaw_sharpen,
            "eye_enlarge": req.eye_enlarge,
            "leg_stretch": req.leg_stretch,
            "shoulder_width": req.shoulder_width,
            "waist_slim": req.waist_slim,
        }
        log.info("apply-transform: params=%s, preview=%s", params, req.preview)

        img = decode_base64_image(req.image_base64)

        # analyze-and-transform과 같은 순서로 재현한다.
        # 슬라이더만 적용하면 저장본에서 수평·크롭·영역 보정이 사라진다.
        if req.auto_edits:
            img = apply_auto_edits(img, req.auto_edits)

        with MediaPipeCache() as mp_cache:
            valid_regions = {
                k: v for k, v in (req.region_params or {}).items()
                if isinstance(v, dict)
            }
            if valid_regions and not req.preview:
                try:
                    regions = detect_regions(img, cache=mp_cache)
                    img = apply_regional_transforms(
                        img, regions, valid_regions, cache=mp_cache
                    )
                except Exception as exc:
                    log.warning("apply-transform: regional transforms failed: %s", exc)
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
    authorization: str | None = Header(None),
):
    """사용자의 대표 사진 3장을 base64로 반환합니다."""
    _verify_token(authorization)

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
    redirect_url = f"{INSTAGRAM_APP_REDIRECT}?code={code}"
    if state:
        redirect_url += f"&state={state}"
    return RedirectResponse(url=redirect_url)


@app.post("/api/instagram/exchange-token", response_model=InstagramExchangeTokenResponse)
def api_instagram_exchange_token(
    req: InstagramExchangeTokenRequest,
    authorization: str | None = Header(None),
):
    """Authorization code → short-lived access_token 교환 (client_secret 보호)."""
    _verify_token(authorization)

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
                error=f"No access_token in response: {token_data}",
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
            log.warning("Long-lived token exchange failed, using short-lived: %s", e)

        log.info("instagram/exchange-token: success, user_id=%s", user_id)
        return InstagramExchangeTokenResponse(
            success=True,
            data={"access_token": access_token, "user_id": str(user_id)},
        )

    except httpx.HTTPStatusError as e:
        log.exception("instagram/exchange-token HTTP error")
        body = e.response.text
        return InstagramExchangeTokenResponse(
            success=False, error=f"Instagram API error: {body}"
        )
    except Exception as e:
        log.exception("instagram/exchange-token failed")
        return InstagramExchangeTokenResponse(success=False, error=str(e))


@app.post("/api/instagram/media", response_model=InstagramMediaResponse)
def api_instagram_media(
    req: InstagramMediaRequest,
    authorization: str | None = Header(None),
):
    """Instagram 미디어 목록을 프록시 조회 (페이지네이션 포함)."""
    _verify_token(authorization)

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
        log.exception("instagram/media HTTP error")
        body = e.response.text
        return InstagramMediaResponse(
            success=False, error=f"Instagram API error: {body}"
        )
    except Exception as e:
        log.exception("instagram/media failed")
        return InstagramMediaResponse(success=False, error=str(e))


@app.post("/api/instagram/stories", response_model=InstagramStoriesResponse)
def api_instagram_stories(
    req: InstagramStoriesRequest,
    authorization: str | None = Header(None),
):
    """Instagram 스토리 목록을 프록시 조회."""
    _verify_token(authorization)

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
        log.exception("instagram/stories HTTP error")
        body = e.response.text
        return InstagramStoriesResponse(
            success=False, error=f"Instagram API error: {body}"
        )
    except Exception as e:
        log.exception("instagram/stories failed")
        return InstagramStoriesResponse(success=False, error=str(e))


if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("PORT", "8000"))
    uvicorn.run("server:app", host="0.0.0.0", port=port, reload=True)
