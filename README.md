# GAMDO Agent Server

GAMDO 앱의 AI 분석 백엔드 서버입니다. Claude Vision API를 사용하여 사용자 스타일 분석 및 사진 변형 가이드를 제공합니다.

## API

### POST `/api/analyze-user`
사용자의 게시글/피드/스토리를 분석하여 스타일 프로필을 생성합니다.

**Request:**
```json
{
  "posts": [
    { "text": "오늘 카페에서", "image_base64": "...", "timestamp": "2025-01-01" }
  ],
  "feeds": [
    { "text": "일몰 풍경", "image_base64": "..." }
  ],
  "stories": [
    { "image_base64": "..." }
  ]
}
```

**Response:**
```json
{
  "success": true,
  "data": {
    "styleProfile": {
      "primaryStyle": "미니멀",
      "colorPreference": { "preferredTones": "cool", ... },
      "compositionPreference": { ... },
      ...
    },
    "summary": "이 사용자는 미니멀한 감성의...",
    "recommendations": ["...", "...", "..."]
  }
}
```

### POST `/api/transform-photo`
사용자 스타일 프로필에 맞춰 사진 보정 가이드를 제공합니다.

**Request:**
```json
{
  "style_profile": { "primaryStyle": "미니멀", ... },
  "image_base64": "...",
  "media_type": "image/jpeg"
}
```

**Response:**
```json
{
  "success": true,
  "data": {
    "currentAnalysis": { ... },
    "transformGuide": {
      "colorAdjustments": { ... },
      "editingSteps": [ ... ],
      ...
    },
    "overallScore": { "current": 65, "expectedAfter": 85 }
  }
}
```

## 실행

```bash
# 의존성 설치
pip install -e .

# 환경변수 설정
cp .env.example .env
# .env 파일에 ANTHROPIC_API_KEY 입력

# 서버 실행
python server.py
# → http://localhost:8000

# API 문서
# → http://localhost:8000/docs
```

## 인증

`APP_TOKEN` 환경변수를 설정하면 Bearer 토큰 인증이 활성화됩니다.

```
Authorization: Bearer your-app-secret-token
```
