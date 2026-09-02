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
    user_id: str = Field("", description="사용자 ID (대표 사진 저장용)")


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


# ── Instagram API ──

class InstagramExchangeTokenRequest(BaseModel):
    """OAuth authorization code → access_token 교환 요청."""
    code: str = Field(..., description="Instagram OAuth authorization code")
    redirect_uri: str = Field(..., description="OAuth redirect URI")


class InstagramExchangeTokenResponse(BaseModel):
    success: bool
    data: dict | None = None
    error: str | None = None


class InstagramMediaRequest(BaseModel):
    """Instagram 미디어 목록 조회 요청."""
    access_token: str = Field(..., description="Instagram access token")


class InstagramMediaResponse(BaseModel):
    success: bool
    data: list[dict] | None = None
    error: str | None = None


class InstagramStoriesRequest(BaseModel):
    """Instagram 스토리 목록 조회 요청."""
    access_token: str = Field(..., description="Instagram access token")


class InstagramStoriesResponse(BaseModel):
    success: bool
    data: list[dict] | None = None
    error: str | None = None


# ── 분석 + 변형 통합 API ──

class AnalyzeAndTransformRequest(BaseModel):
    """사진 분석 + 변형을 한 번에 수행."""
    image_base64: str = Field(..., description="원본 이미지 (base64)")
    style_profile: dict = Field(default_factory=dict, description="사용자 스타일 프로필")
    user_id: str = Field("", description="사용자 ID (대표 사진 레퍼런스 조회용)")
    media_type: str = "image/jpeg"


class AnalyzeAndTransformResponse(BaseModel):
    success: bool
    analysis: dict | None = None
    image_base64: str | None = None
    params: dict | None = None
    params_comment: str | None = Field(None, description="적용된 변형 요약 한 문장")
    error: str | None = None


# ── 자동 변형 API ──

class AutoTransformRequest(BaseModel):
    """AI 분석 기반 자동 변형 요청."""
    image_base64: str = Field(..., description="원본 이미지 (base64)")
    analysis: dict = Field(..., description="AI 분석 결과 JSON")
    style_profile: dict | None = Field(None, description="사용자 스타일 프로필")


class AutoTransformResponse(BaseModel):
    success: bool
    image_base64: str | None = None
    params: dict | None = None
    params_comment: str | None = Field(None, description="적용된 변형 요약 한 문장")
    error: str | None = None


class ApplyTransformRequest(BaseModel):
    """슬라이더 값으로 수동 변형 요청."""
    image_base64: str = Field(..., description="원본 이미지 (base64)")
    # 기하 보정과 영역별 보정은 슬라이더가 아니지만, 여기서 다시 적용하지 않으면
    # 저장본이 미리보기와 달라진다 (수평·크롭·영역 보정이 빠진 사진이 저장된다).
    auto_edits: dict | None = Field(None, description="수평 보정/크롭/요소 제거/비율")
    region_params: dict | None = Field(None, description="하늘/얼굴/배경 영역별 보정")
    preview: bool = Field(False, description="True이면 미리보기 모드 (색감 보정만, reshape/blemish 스킵)")
    brightness: float = Field(0.0, ge=-1.0, le=1.0)
    contrast: float = Field(0.0, ge=-1.0, le=1.0)
    clarity: float = Field(0.0, ge=-1.0, le=1.0, description="선명감 (미드톤 로컬 대비)")
    dehaze: float = Field(0.0, ge=-1.0, le=1.0, description="안개 제거 (Dark Channel Prior)")
    highlights: float = Field(0.0, ge=-1.0, le=1.0)
    shadows: float = Field(0.0, ge=-1.0, le=1.0)
    saturation: float = Field(0.0, ge=-1.0, le=1.0)
    temperature: float = Field(0.0, ge=-1.0, le=1.0)
    blemish_removal: float = Field(0.0, ge=0.0, le=1.0)
    skin_smoothing: float = Field(0.0, ge=0.0, le=1.0)
    vignette: float = Field(0.0, ge=-1.0, le=1.0)
    sharpness: float = Field(0.0, ge=-1.0, le=1.0)
    grain: float = Field(0.0, ge=0.0, le=1.0)
    auto_wb: float = Field(0.0, ge=0.0, le=1.0, description="자동 화이트밸런스 강도")
    denoise: float = Field(0.0, ge=0.0, le=1.0, description="노이즈 제거 강도")
    tone_curve_preset: str = Field("linear", description="톤 커브 프리셋 (linear|s_curve|film|fade|high_contrast|bright)")
    tone_curve_strength: float = Field(0.0, ge=0.0, le=1.0, description="톤 커브 강도")
    split_shadow_hue: float = Field(0.0, ge=0.0, le=360.0, description="스플릿 토닝 쉐도우 색상 (0~360)")
    split_shadow_strength: float = Field(0.0, ge=0.0, le=1.0, description="스플릿 토닝 쉐도우 강도")
    split_highlight_hue: float = Field(0.0, ge=0.0, le=360.0, description="스플릿 토닝 하이라이트 색상 (0~360)")
    split_highlight_strength: float = Field(0.0, ge=0.0, le=1.0, description="스플릿 토닝 하이라이트 강도")
    hsl_adjust: dict | None = Field(None, description="선택적 색상(HSL) 조절 {color: {hue, saturation, lightness}}")
    # ── 얼굴/체형 보정 ──
    face_slim: float = Field(0.0, ge=0.0, le=1.0, description="얼굴 축소 (양쪽→중심)")
    jaw_sharpen: float = Field(0.0, ge=0.0, le=1.0, description="턱선 V라인")
    eye_enlarge: float = Field(0.0, ge=0.0, le=1.0, description="눈 확대")
    leg_stretch: float = Field(0.0, ge=0.0, le=1.0, description="다리 늘리기")
    shoulder_width: float = Field(0.0, ge=-1.0, le=1.0, description="어깨 너비 (음수=좁게, 양수=넓게)")
    waist_slim: float = Field(0.0, ge=0.0, le=1.0, description="허리 라인 슬림")


class ApplyTransformResponse(BaseModel):
    success: bool
    image_base64: str | None = None
    params_applied: dict | None = None
    error: str | None = None


# ── 대표 사진 조회 API ──

class ReferenceImagesResponse(BaseModel):
    """사용자 대표 사진 base64 목록 응답."""
    success: bool
    images: list[str] = Field(default_factory=list, description="base64 인코딩된 대표 사진 목록")
    error: str | None = None
