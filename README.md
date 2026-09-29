# GAMDO Agent Server

감도(GAMDO) 앱의 AI 사진 코칭 백엔드입니다. FastAPI 서버이며, 스타일 분석·사진 분석은
서버 머신에 로그인된 **`claude` CLI(`claude -p`)** 로 실행합니다. Anthropic API 키
(`ANTHROPIC_API_KEY`)는 쓰지 않습니다 — 남아 있으면 CLI의 로그인 경로를 가로채지 않도록
자식 프로세스 환경에서 지웁니다. 보정 파라미터 계산과 픽셀 처리는 서버에서 직접 합니다
(OpenCV·MediaPipe).

## 실행

```bash
# 의존성 설치 (uv 권장)
uv sync --extra test              # Firebase 커스텀 토큰까지 쓰려면 --extra firebase 추가
# 또는: pip install -e '.[test]'   /   pip install -e '.[firebase]'

# claude CLI 로그인 (서버를 돌릴 계정으로 한 번)
claude                            # 대화형으로 실행해 /login. 헤드리스라면 `claude setup-token` → CLAUDE_CODE_OAUTH_TOKEN

# 환경변수
cp .env.example .env              # 아래 표 참고

# 서버 실행
python server.py                  # → http://localhost:8000 , 문서: /docs

# 테스트
.venv/bin/pytest -q
```

## 환경변수

| 이름 | 기본값 | 설명 |
| --- | --- | --- |
| `GAMDO_SESSION_SECRET` | (없음) | 세션 토큰 HMAC 서명 키. **32자 이상 랜덤.** 없거나 짧으면 세션 발급이 꺼진다 (`/api/session` 503, exchange-token의 세션 필드 null). 바꾸면 발급된 세션이 모두 무효가 된다. |
| `GAMDO_AUTH_REQUIRED` | 꺼짐 | `1`/`true`/`yes`면 세션 없는 요청을 401 `session_invalid`로 막는다. 꺼져 있으면 과도기 규칙(아래). |
| `APP_TOKEN` | (없음) | 서비스 토큰. `Authorization: Bearer <APP_TOKEN>`이면 항상 통과하고 uid 제한도 없다. 과도기에 설정돼 있으면 세션도 APP_TOKEN도 없는 요청은 401. |
| `FIREBASE_SERVICE_ACCOUNT` | (없음) | Firebase 서비스 계정 JSON 파일 경로. `firebase` extra 설치 필요. 없으면 `/api/firebase-token`은 503 `firebase_not_configured`. |
| `INSTAGRAM_CLIENT_ID` / `INSTAGRAM_CLIENT_SECRET` | (없음) | Instagram 로그인 OAuth 앱 자격증명 (code → token 교환). |
| `GAMDO_MAX_BODY_MB` | `50` | 요청 본문 상한(MB). 넘으면 413. |
| `GAMDO_MAX_CONCURRENT` | `3` | 이미지 처리 동시 실행 수 (메모리 보호). |
| `GAMDO_MAX_JOBS` | `4` | 비동기 분석 작업(`/api/jobs/...`) 워커 스레드 수. 픽셀 처리는 여전히 `GAMDO_MAX_CONCURRENT`로 묶인다. |
| `GAMDO_MAX_QUEUED_JOBS` | `20` | 대기 중인 작업 상한. 넘으면 새 작업은 503 `busy`. |
| `CLAUDE_CODE_OAUTH_TOKEN` | (선택) | 헤드리스 서버에서 `claude` CLI 로그인 대신 쓰는 토큰. |
| `PORT` | `8000` | `python server.py` 실행 포트. |

## 인증

앱은 모든 요청에 `Authorization: Bearer <session_token>`을 보낸다.

- **세션 토큰**: `v1.<base64url(payload)>.<base64url(HMAC-SHA256(secret, "v1."+payload_b64))>`,
  payload `{"uid": "<instagram user_id>", "iat": <unix초>, "exp": <unix초>}`, 유효기간 30일.
- **인증 면제**: `/health`, `/api/instagram/callback`, `/api/instagram/exchange-token`,
  `/api/session`, `/api/instagram/media`, `/api/instagram/stories` (뒤의 둘은 요청의 IG
  access_token 자체가 자격증명).
- **그 외 엔드포인트**
  - 유효한 세션 → uid 확보. 요청의 `user_id`(본문 또는 경로)가 비어 있지 않고 uid와 다르면
    403 `forbidden_user` (`analyze-user`, `analyze-and-transform`, `reference-images/{user_id}`).
  - `APP_TOKEN`과 일치 → 서비스 토큰, 항상 통과.
  - 세션 없음/무효: `GAMDO_AUTH_REQUIRED`가 켜져 있으면 401 `session_invalid`.
    꺼져 있으면(과도기) 예전 동작 — `APP_TOKEN`이 있으면 401, 없으면 통과(경고 로그).
- `/api/firebase-token`은 과도기에도 세션이 필수다 (uid가 있어야 토큰을 만든다).
- 인증 관련 에러 본문: `{"success": false, "error": "<메시지>", "error_code": "<코드>"}`
  (`session_invalid`, `forbidden_user`, `instagram_token_invalid`, `instagram_unavailable`,
  `session_not_configured`, `firebase_not_configured`).

### 배포 순서

1. 서버에 `GAMDO_SESSION_SECRET`을 넣고 배포한다 (`GAMDO_AUTH_REQUIRED`는 끈 채로).
   기존 앱 빌드는 그대로 동작한다.
2. 세션을 보내는 새 앱을 배포하고, 사용자 대부분이 업데이트할 때까지 기다린다.
3. `GAMDO_AUTH_REQUIRED=1`을 설정하고 서버를 재시작한다. 이때부터 옛 앱 빌드는 401을 받는다.
   옛 앱에 `APP_TOKEN`이 들어 있었다면 그 값은 여전히 서비스 토큰으로 통과하므로,
   함께 바꾸거나 비운다.
4. (선택) Firebase: 서비스 계정 JSON을 서버에 두고 `FIREBASE_SERVICE_ACCOUNT`를 설정,
   `firebase` extra를 설치한 뒤, 앱 저장소의 `database.rules.json`을
   `firebase deploy --only database`로 배포한다.

## API

모든 응답은 `{"success": bool, ..., "error": str | null}` 형식이다. 처리 중 오류는 대부분
HTTP 200 + `success: false`로, 인증 오류는 위의 HTTP 상태 코드로 온다. 자세한 스키마는 `/docs`.

| 메서드 | 경로 | 인증 | 설명 |
| --- | --- | --- | --- |
| GET | `/health` | 면제 | 상태 확인 |
| POST | `/api/analyze-user` | 필요 | 게시글/피드/스토리 → 스타일 프로필. `user_id`가 있으면 대표 사진 3장을 저장 |
| POST | `/api/transform-photo` | 필요 | 스타일 프로필 기준 사진 보정 가이드 (분석만) |
| POST | `/api/analyze-and-transform` | 필요 | 사진 분석 + 보정 이미지 생성. `user_id`의 대표 사진을 목표값으로 사용 (동기, 구버전 앱용) |
| POST | `/api/jobs/analyze-and-transform` | 필요 | 위와 같은 작업을 백그라운드로 시작 → `job_id` (202) |
| GET | `/api/jobs/{job_id}` | 필요 | 작업 상태·결과 조회 (폴링) |
| POST | `/api/auto-transform` | 필요 | 분석 결과 JSON으로 보정 이미지 생성 |
| POST | `/api/apply-transform` | 필요 | 슬라이더 값으로 보정 (미리보기/저장) |
| GET | `/api/reference-images/{user_id}` | 필요 | 대표 사진 base64 목록 |
| GET | `/api/instagram/callback` | 면제 | OAuth 콜백 → `gamdo://oauth/instagram` 리디렉션 |
| POST | `/api/instagram/exchange-token` | 면제 | code → long-lived 토큰 교환. 세션도 함께 발급 |
| POST | `/api/instagram/media` | 면제 | Instagram 미디어 목록 프록시 |
| POST | `/api/instagram/stories` | 면제 | Instagram 스토리 목록 프록시 |
| POST | `/api/session` | 면제 | IG long-lived 토큰 → 감도 세션 재발급 |
| POST | `/api/firebase-token` | 세션 필수 | Firebase 커스텀 토큰 (uid = Instagram user_id) |

### POST `/api/instagram/exchange-token`

```json
// 요청
{ "code": "<authorization code>", "redirect_uri": "https://.../api/instagram/callback" }
// 응답
{
  "success": true,
  "data": {
    "access_token": "<long-lived token>",
    "user_id": "17841400000000000",
    "session_token": "v1.xxx.yyy",       // 세션 미설정 서버면 null
    "session_expires_at": 1767225600     // unix초, 세션 미설정 서버면 null
  }
}
```

### POST `/api/session`

서버가 `https://graph.instagram.com/me?fields=user_id,username`을 토큰(Authorization 헤더)으로
불러 검증하고, 응답의 `user_id`(없으면 `id`)로 세션을 발급한다.

```json
// 요청
{ "access_token": "<instagram long-lived token>" }
// 응답
{ "success": true, "data": { "session_token": "v1...", "session_expires_at": 1767225600, "user_id": "..." } }
```

- IG 토큰 무효/만료 → 401 `instagram_token_invalid` (앱은 로그아웃 처리)
- Instagram 장애·응답 이상 → 502 `instagram_unavailable` (앱은 로그아웃하지 않는다)
- 세션 비밀키 미설정 → 503 `session_not_configured`

### POST `/api/firebase-token`

```json
// 응답
{ "success": true, "data": { "firebase_token": "<custom token>" } }
```

서비스 계정·패키지 미설정 → 503 `firebase_not_configured` (앱은 Firebase Auth 없이 진행).

### POST `/api/analyze-user`

```json
// 요청
{
  "posts":   [{ "text": "오늘 카페에서", "image_url": "https://scontent...cdninstagram.com/...", "timestamp": "2025-01-01" }],
  "feeds":   [{ "image_base64": "..." }],
  "stories": [],
  "user_id": "17841400000000000"
}
// 응답
{ "success": true, "data": { "styleProfile": { ... }, "summary": "...", "referenceImages": ["ref_0.jpg", ...] } }
```

### POST `/api/analyze-and-transform`

```json
// 요청
{ "image_base64": "...", "style_profile": { ... }, "user_id": "...", "media_type": "image/jpeg", "reshape_enabled": false }
// 응답
{ "success": true, "analysis": { ... }, "image_base64": "...", "params": { ... }, "params_comment": "..." }
```

같은 사진·프로필·사용자·대표 사진 조합의 분석 결과는 30분간 캐시된다 (대표 사진이 바뀌면
캐시를 쓰지 않는다).

### 비동기 분석 작업 — `POST /api/jobs/analyze-and-transform`, `GET /api/jobs/{job_id}`

분석은 30~70초 걸린다. 앱이 백그라운드로 가서 연결이 끊겨도 서버가 계속 처리하도록
작업으로 맡기고 폴링한다.

```json
// POST /api/jobs/analyze-and-transform — 요청 본문은 동기 엔드포인트와 같다. 응답 HTTP 202
{ "success": true, "job_id": "<32 hex>", "status": "queued" }
// 대기열이 가득 차면 HTTP 503
{ "success": false, "error_code": "busy", "error": "..." }

// GET /api/jobs/{job_id} — 응답 200
{
  "success": true, "job_id": "...",
  "status": "queued | running | done | error",
  "stage": "queued | analyzing | rendering | done | error",
  "elapsed_sec": 12.3,
  "result": { ... },   // status == done일 때만 — 동기 응답과 같은 모양 (아니면 null)
  "error": "..."       // status == error일 때만 — 사람용 메시지 (아니면 null)
}
// 없거나 만료됐거나 다른 사용자의 작업이면 HTTP 404
{ "success": false, "error_code": "job_not_found", "error": "..." }
```

- 인증·인가는 동기 엔드포인트와 같다 (`user_id`가 세션 uid와 다르면 403). 세션으로 만든
  작업은 같은 uid의 세션으로만 조회된다 (다르면 404 — 존재 여부를 드러내지 않는다).
- 같은 요청(사진·프로필·user_id·media_type·reshape_enabled + 세션 uid)이 대기·진행 중이거나
  성공 후 보관 중이면 새 작업을 만들지 않고 그 `job_id`를 돌려준다. 실패한 작업은 합치지 않는다.
- 분석이 실패하면(동기 응답의 `success: false`) `status: "error"`. 메시지에는 내부 정보를 싣지 않고
  원인은 서버 로그에 남긴다.
- 인메모리 저장이다. 완료/실패 후 30분 보관, 전체 200개를 넘으면 오래된 완료 작업부터 지운다.
  서버가 재시작되면(reload 포함) 작업이 사라진다 → 앱은 404를 받으면 다시 시작한다.
