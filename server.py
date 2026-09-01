"""GAMDO Agent Server — FastAPI + Claude Code CLI."""

import base64
import os
import logging

from dotenv import load_dotenv
from fastapi import FastAPI, Header, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
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
from image_processor import (
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

        # 2. 분석 결과에서 파라미터 추출
        params = analysis_to_transform_params(analysis)
        log.info("analyze-and-transform: params=%s", params)

        # 3. 이미지 디코딩
        img = decode_base64_image(req.image_base64)

        # 4. AI autoEdits 적용 (크롭, 요소 제거, 인스타 비율)
        auto_edits = analysis.get("autoEdits", {})
        if auto_edits and isinstance(auto_edits, dict):
            log.info("analyze-and-transform: applying autoEdits=%s", auto_edits)
            img = apply_auto_edits(img, auto_edits)

        # 5. 영역별 스마트 보정 (regionParams가 있으면)
        region_params_raw = analysis.get("regionParams")
        if region_params_raw and isinstance(region_params_raw, dict):
            # null이 아닌 영역만 필터링
            valid_region_params = {
                k: v for k, v in region_params_raw.items()
                if v is not None and isinstance(v, dict)
            }
            if valid_region_params:
                try:
                    regions = detect_regions(img)
                    img = apply_regional_transforms(img, regions, valid_region_params)
                    log.info("analyze-and-transform: applied regional transforms for regions=%s",
                             list(valid_region_params.keys()))
                except Exception as e:
                    log.warning("analyze-and-transform: regional transforms failed, falling back: %s", e)

        # 6. 슬라이더 변형 적용
        transformed = apply_all_transforms(img, **params)
        result_b64 = encode_image_base64(transformed)

        log.info("analyze-and-transform: success")
        return AnalyzeAndTransformResponse(
            success=True,
            analysis=analysis,
            image_base64=result_b64,
            params=params,
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

        # 1. 분석 → 파라미터 자동 계산
        params = analysis_to_transform_params(req.analysis)
        log.info("auto-transform: params=%s", params)

        # 2. 이미지 디코딩
        img = decode_base64_image(req.image_base64)

        # 3. AI autoEdits 적용 (크롭, 요소 제거, 인스타 비율)
        auto_edits = req.analysis.get("autoEdits", {})
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
            "tone_curve_preset": req.tone_curve_preset,
            "tone_curve_strength": req.tone_curve_strength,
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
        log.info("apply-transform: params=%s", params)

        img = decode_base64_image(req.image_base64)
        transformed = apply_all_transforms(img, **params)
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
