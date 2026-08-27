"""GAMDO Agent Server — FastAPI."""

import os
import logging

from dotenv import load_dotenv
from fastapi import FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from models import (
    AnalyzeUserRequest,
    AnalyzeUserResponse,
    TransformPhotoRequest,
    TransformPhotoResponse,
)
from claude_client import analyze_user, transform_photo

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

API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
APP_TOKEN = os.getenv("APP_TOKEN", "")


def _verify_token(authorization: str | None = Header(None)):
    """Bearer 토큰 검증."""
    if not APP_TOKEN:
        return  # 토큰 미설정 시 인증 스킵 (개발용)
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
    """
    사용자의 게시글/피드/스토리를 분석하여 스타일 프로필을 반환합니다.

    - posts: 게시글 목록 (텍스트, 이미지 URL 또는 base64)
    - feeds: 피드 목록
    - stories: 스토리 목록
    """
    _verify_token(authorization)

    if not API_KEY:
        return AnalyzeUserResponse(success=False, error="ANTHROPIC_API_KEY not configured")

    try:
        log.info(
            "analyze-user: posts=%d, feeds=%d, stories=%d",
            len(req.posts), len(req.feeds), len(req.stories),
        )

        result = analyze_user(
            api_key=API_KEY,
            posts=[p.model_dump() for p in req.posts],
            feeds=[f.model_dump() for f in req.feeds],
            stories=[s.model_dump() for s in req.stories],
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
    """
    사용자 스타일 프로필에 맞춰 사진 보정 가이드를 반환합니다.

    - style_profile: analyze-user에서 받은 styleProfile
    - image_base64: 변형할 사진 (base64 인코딩)
    - media_type: 이미지 MIME 타입 (기본 image/jpeg)
    """
    _verify_token(authorization)

    if not API_KEY:
        return TransformPhotoResponse(success=False, error="ANTHROPIC_API_KEY not configured")

    try:
        log.info("transform-photo: style=%s", req.style_profile.get("primaryStyle", "unknown"))

        result = transform_photo(
            api_key=API_KEY,
            style_profile=req.style_profile,
            image_base64=req.image_base64,
            media_type=req.media_type,
        )

        log.info("transform-photo: success")
        return TransformPhotoResponse(success=True, data=result)

    except Exception as e:
        log.exception("transform-photo failed")
        return TransformPhotoResponse(success=False, error=str(e))


if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("PORT", "8000"))
    uvicorn.run("server:app", host="0.0.0.0", port=port, reload=True)
