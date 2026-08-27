"""Pydantic request/response 모델."""

from pydantic import BaseModel, Field


# ── 사용자 분석 API ──

class PostItem(BaseModel):
    """게시글/피드/스토리 한 건."""
    text: str | None = None
    image_url: str | None = None       # 이미지 URL (외부 접근 가능한)
    image_base64: str | None = None    # 또는 base64 인코딩된 이미지
    media_type: str = "image/jpeg"
    timestamp: str | None = None


class AnalyzeUserRequest(BaseModel):
    posts: list[PostItem] = Field(default_factory=list, description="게시글 목록")
    feeds: list[PostItem] = Field(default_factory=list, description="피드 목록")
    stories: list[PostItem] = Field(default_factory=list, description="스토리 목록")


class AnalyzeUserResponse(BaseModel):
    success: bool
    data: dict | None = None
    error: str | None = None


# ── 사진 변형 API ──

class TransformPhotoRequest(BaseModel):
    style_profile: dict = Field(..., description="analyze-user에서 받은 styleProfile")
    image_base64: str = Field(..., description="변형할 사진 (base64)")
    media_type: str = "image/jpeg"


class TransformPhotoResponse(BaseModel):
    success: bool
    data: dict | None = None
    error: str | None = None
