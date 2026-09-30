"""Claude Code CLI(claude -p)를 사용한 분석 클라이언트 — 이미지 Vision 지원."""

import base64
import copy
import io
import hashlib
import json
import logging
import os
import re
import subprocess
import tempfile
import threading
import time
import uuid
from collections import OrderedDict
from typing import Any
from urllib.parse import urljoin, urlsplit

from PIL import Image

from prompts import (
    ANALYZE_USER_SYSTEM,
    ANALYZE_USER_PROMPT,
    TRANSFORM_PHOTO_SYSTEM,
    TRANSFORM_PHOTO_PROMPT,
)

log = logging.getLogger("gamdo-agent")

MODEL = "sonnet"

# Vision 분석용 이미지 최대 크기 (px). 해상도를 낮춰 전송량과 처리 시간을 줄인다.
_VISION_MAX_PX = 1024

# 변형할 사진을 메시지에 직접 넣을 때의 긴 변 상한. 모델이 보는 해상도의 상한과
# 같게 두어, Read로 원본 파일을 읽게 하던 때보다 화질을 잃지 않는다.
_INLINE_PHOTO_MAX_PX = 1568

# 이미지 전달 방식. inline(기본) = stream-json 메시지에 이미지를 직접 담는다.
# read = 임시 파일 경로를 주고 모델이 Read 도구로 읽게 한다 (예전 방식).
# Read는 "도구 호출 → 결과 → 답변"으로 모델 왕복이 최소 한 번 더 생긴다.
_VISION_INPUT_MODE = os.getenv("GAMDO_VISION_INPUT", "inline").strip().lower()

# 이미지 임시 저장 디렉토리
_TEMP_DIR = os.path.join(tempfile.gettempdir(), "gamdo-images")
os.makedirs(_TEMP_DIR, exist_ok=True)

# 대표 사진 영구 저장 디렉토리
_REF_DIR = os.path.join(os.path.dirname(__file__), "reference_images")
os.makedirs(_REF_DIR, exist_ok=True)


def _loads_strict(text: str) -> dict:
    """JSON을 읽되 NaN·Infinity 리터럴은 None으로 바꾼다.

    json.loads는 표준 밖의 NaN/Infinity/-Infinity를 그대로 float로 받아들인다.
    그 값이 파라미터로 흘러가면 min/max 비교가 상한을 돌려주기 때문에 "무시"가
    아니라 **최대 보정**이 됐다 (실측: straighten NaN → 8도 회전, 얼굴 워프
    전 항목 상한). None으로 바꿔 두면 뒤쪽 검증·기본값이 정상적으로 걸러낸다.
    """
    return json.loads(text, parse_constant=lambda _name: None)


# 코드블록(```json ... ``` 또는 태그 없는 ``` ... ```) 안쪽을 꺼낸다
_FENCE_RE = re.compile(r"```[ \t]*(?:json)?[ \t]*\r?\n?(.*?)```", re.DOTALL | re.IGNORECASE)


def _try_parse_object(text: str) -> dict | None:
    """text에서 JSON 객체 하나를 찾아 돌려준다. 못 찾으면 None."""
    try:
        value = _loads_strict(text)
        if isinstance(value, dict):
            return value
    except json.JSONDecodeError:
        pass

    start = text.find("{")
    if start == -1:
        return None

    # 첫 번째 { 에서 객체 하나만 읽는다 — 뒤에 붙은 설명문은 무시된다
    decoder = json.JSONDecoder(parse_constant=lambda _name: None)
    try:
        value, _ = decoder.raw_decode(text, start)
        if isinstance(value, dict):
            return value
    except json.JSONDecodeError:
        pass

    # 그래도 안 되면 첫 번째 { 부터 마지막 } 까지 추출
    end = text.rfind("}")
    if end > start:
        try:
            value = _loads_strict(text[start : end + 1])
            if isinstance(value, dict):
                return value
        except json.JSONDecodeError:
            pass
    return None


def _parse_json_response(text: str) -> dict:
    """응답 텍스트에서 JSON 객체를 추출한다.

    예전에는 ``` 로 split해서 [0]을 취했다. 태그 없는 ``` 앞에 설명문이
    있으면 JSON이 아니라 그 설명문을 파싱하다 실패했다. 코드블록 안쪽을
    먼저 보고, 없으면 전체 텍스트에서 찾는다. 호출부가 전부 .get()을 쓰므로
    객체(dict)가 아니면 실패로 본다.
    """
    text = text.strip()

    candidates = [m.group(1).strip() for m in _FENCE_RE.finditer(text)]
    candidates.append(text)
    for candidate in candidates:
        parsed = _try_parse_object(candidate)
        if parsed is not None:
            return parsed
    raise ValueError(f"응답에서 JSON 객체를 찾지 못함: {text[:200]!r}")


# Flutter의 PhotoAnalysisResponse가 파싱에 반드시 필요로 하는 필드.
# 하나라도 빠지면 앱이 "이전 형식의 분석 데이터입니다"를 띄운다.
_REQUIRED_SHAPE: dict[str, tuple[type, ...] | dict[str, tuple[type, ...]]] = {
    "colorAnalysis": {
        "colorHarmony": (str,),
        "paletteDescription": (str,),
    },
    "compositionAnalysis": {
        "primaryTechnique": (str,),
        "balanceScore": (int, float),
    },
    "toneReport": {
        "overallMood": (str,),
        "styleCategory": (str,),
        "narrative": (str,),
    },
}

# 복구도 실패했을 때 채워 넣는 값. 사진 변형은 서버가 계산하므로 영향이 없고,
# 분석 화면만 비어 보인다 — 응답의 warnings로 그 사실을 알린다.
_ANALYSIS_FALLBACKS: dict[str, Any] = {
    "colorAnalysis": {"colorHarmony": "분석 불가", "paletteDescription": ""},
    "compositionAnalysis": {"primaryTechnique": "분석 불가", "balanceScore": 0.5},
    "toneReport": {"overallMood": "", "styleCategory": "", "narrative": ""},
}


def _validate_analysis(parsed: dict) -> list[str]:
    """앱이 요구하는 필드가 갖춰졌는지 검사하고, 어긋난 경로 목록을 반환한다."""
    problems: list[str] = []
    for key, spec in _REQUIRED_SHAPE.items():
        value = parsed.get(key)
        if isinstance(spec, dict):
            if not isinstance(value, dict):
                problems.append(f"{key} (객체가 아님)")
                continue
            for sub, types in spec.items():
                if not isinstance(value.get(sub), types):
                    problems.append(f"{key}.{sub}")
        elif not isinstance(value, spec):
            problems.append(key)
    return problems


def _apply_analysis_fallbacks(parsed: dict, problems: list[str]) -> None:
    """검증에 실패한 필드만 기본값으로 메운다 (제자리 수정)."""
    for path in problems:
        key = path.split(" ")[0].split(".")[0]
        fallback = _ANALYSIS_FALLBACKS.get(key)
        if fallback is None:
            continue
        if isinstance(fallback, dict):
            target = parsed.get(key)
            if not isinstance(target, dict):
                parsed[key] = dict(fallback)
            else:
                for sub, default in fallback.items():
                    if sub not in target:
                        target[sub] = default
        else:
            parsed[key] = fallback


# 스타일 프로필은 한 번 저장되면 이후 모든 보정을 좌우한다.
# 잘못된 값이 들어가도 param_engine이 조용히 기본값으로 폴백해서
# "왜 보정이 밋밋하지"의 원인을 추적하기 어렵다. 저장 전에 잡는다.
_PROFILE_ENUMS: dict[str, tuple[str, ...]] = {
    "colorPreference.preferredTones":
        ("cool", "slightly_cool", "neutral", "slightly_warm", "warm", "mixed"),
    "colorPreference.saturationTendency":
        ("very_low", "low", "medium", "high", "very_high"),
    "colorPreference.brightnessTendency":
        ("very_low", "low", "medium", "high", "very_high"),
    "colorPreference.contrast":
        ("very_low", "low", "medium", "high", "very_high"),
    "editingStyle.filterTendency":
        ("none", "minimal", "moderate", "strong", "very_strong"),
    "editingStyle.grainPreference":
        ("none", "subtle", "moderate", "heavy", "film"),
    "editingStyle.vignettePreference":
        ("none", "subtle", "moderate", "strong"),
    "editingStyle.skinRetouchLevel":
        ("none", "light", "moderate", "heavy"),
    "trendCategory": (
        "warm_film", "korean_gamsung", "cinematic_moody",
        "bright_airy", "golden_hour", "clean_minimal",
        "flash_digicam", "soft_pastel", "bw_grain", "custom",
    ),
}

# 프로필에서 빠져도 되는 항목. param_engine이 "auto" 센티널로 알아서 정하므로
# 여기서 값을 지어내면 안 된다.
#
# 예전에는 이 세 개도 _PROFILE_DEFAULTS를 타고 "none"으로 채워졌다. 그러면
# param_engine의 `editing.get("grainPreference") or "auto"`가 "none"을 읽어
# 그레인·비네팅·피부보정이 전부 0이 됐다. 실측: warm_film + strong 인물
# 프로필에서 grain 0.20→0, vignette 0.12→0, 잡티 0.35→0, 스무딩 0.28→0.
_PROFILE_OPTIONAL = (
    "editingStyle.grainPreference",
    "editingStyle.vignettePreference",
    "editingStyle.skinRetouchLevel",
)

# 값이 어긋났을 때 쓰는 중립값 — 방향을 지어내기보다 중간으로 둔다
_PROFILE_DEFAULTS = {
    "colorPreference.preferredTones": "neutral",
    "colorPreference.saturationTendency": "medium",
    "colorPreference.brightnessTendency": "medium",
    "colorPreference.contrast": "medium",
    "editingStyle.filterTendency": "moderate",
    "editingStyle.grainPreference": "none",
    "editingStyle.vignettePreference": "none",
    "editingStyle.skinRetouchLevel": "none",
    "trendCategory": "custom",
}


def _validate_style_profile(profile: dict) -> list[str]:
    """열거형 값이 약속된 범위 안에 있는지 검사하고, 어긋난 경로를 돌려준다.

    선택 항목([_PROFILE_OPTIONAL])이 아예 없는 것은 문제가 아니다 —
    param_engine이 "auto"로 알아서 정한다. 값이 있으면서 어휘를 벗어난 경우만
    문제로 본다.
    """
    problems: list[str] = []
    for path, allowed in _PROFILE_ENUMS.items():
        parent, _, leaf = path.rpartition(".")
        node = profile.get(parent) if parent else profile
        if parent and not isinstance(node, dict):
            problems.append(f"{parent} (객체가 아님)")
            continue
        value = (node or {}).get(leaf)
        if value is None:
            if path not in _PROFILE_OPTIONAL:
                problems.append(f"{path} (없음)")
        elif value not in allowed:
            problems.append(f"{path}={value!r}")
    return problems


def _repair_style_profile(profile: dict, problems: list[str]) -> None:
    """어긋난 열거형만 중립값으로 되돌린다 (제자리 수정)."""
    for problem in problems:
        path = problem.split(" ")[0].split("=")[0]
        default = _PROFILE_DEFAULTS.get(path)
        if default is None:
            continue
        parent, _, leaf = path.rpartition(".")
        if parent:
            node = profile.setdefault(parent, {})
            if isinstance(node, dict):
                node[leaf] = default
        else:
            profile[leaf] = default


def _format_items(items: list[dict]) -> str:
    """PostItem 목록을 텍스트로 변환."""
    if not items:
        return "(없음)"
    parts = []
    for i, item in enumerate(items, 1):
        line = f"[{i}]"
        if item.get("text"):
            line += f" {item['text']}"
        if item.get("image_url"):
            line += f" (이미지: {item['image_url']})"
        if item.get("timestamp"):
            line += f" — {item['timestamp']}"
        parts.append(line)
    return "\n".join(parts)


# image_url은 클라이언트가 보낸 값이라 서버가 대신 요청을 날리는 통로가 된다
# (내부망·메타데이터 주소 접근, 거대 파일로 메모리 고갈). 앱이 보내는 것은
# Instagram Graph API의 media_url/thumbnail_url뿐이므로 그 CDN만 허용한다.
_ALLOWED_IMAGE_HOST_SUFFIXES = ("cdninstagram.com", "fbcdn.net")
_DOWNLOAD_MAX_BYTES = 15 * 1024 * 1024
_DOWNLOAD_MAX_REDIRECTS = 3


def _is_allowed_image_url(url: str) -> bool:
    """https + Instagram/Facebook CDN 호스트인지 검사한다."""
    try:
        parts = urlsplit(url)
        host = (parts.hostname or "").lower().rstrip(".")
        port = parts.port
    except ValueError:
        return False
    if parts.scheme != "https" or not host or port not in (None, 443):
        return False
    return any(host == s or host.endswith("." + s) for s in _ALLOWED_IMAGE_HOST_SUFFIXES)


def _download_temp_image(url: str) -> str | None:
    """이미지 URL을 받아 임시 파일로 저장한다. 이미지가 아니거나 실패하면 None.

    실제로 열리는지까지 확인한다. Instagram의 VIDEO 게시물은 media_url이
    .mp4라, 확인 없이 받으면 "다운로드 성공"으로 세어 분석 장수 한도를
    까먹고는 나중에 조용히 버려진다. 게다가 그 빈자리 때문에 뒤따르는
    사진들의 인덱스가 밀린다.
    """
    path: str | None = None
    try:
        import httpx

        if not _is_allowed_image_url(url):
            log.warning("Image URL not allowed (Instagram CDN https만 허용): %s", url[:80])
            return None

        # 리다이렉트는 직접 따라가며 매 hop마다 호스트를 다시 검사한다.
        # follow_redirects=True면 허용 도메인이 내부 주소로 튕겨도 그대로 따라간다.
        with httpx.Client(timeout=15, follow_redirects=False) as client:
            current = url
            for _ in range(_DOWNLOAD_MAX_REDIRECTS + 1):
                with client.stream("GET", current) as resp:
                    if resp.is_redirect:
                        location = resp.headers.get("location", "")
                        current = urljoin(current, location)
                        if not _is_allowed_image_url(current):
                            log.warning("Redirect target not allowed: %s", current[:80])
                            return None
                        continue

                    resp.raise_for_status()

                    content_type = resp.headers.get("content-type", "image/jpeg")
                    if not content_type.startswith("image/"):
                        log.warning("Not an image (%s): %s", content_type, url[:80])
                        return None

                    declared = resp.headers.get("content-length")
                    if declared and declared.isdigit() and int(declared) > _DOWNLOAD_MAX_BYTES:
                        log.warning("Image too large (%s bytes): %s", declared, url[:80])
                        return None

                    ext = ".jpg"
                    if "png" in content_type:
                        ext = ".png"
                    elif "webp" in content_type:
                        ext = ".webp"

                    # 헤더의 길이를 믿지 않고 받으면서 센다
                    size = 0
                    fd, path = tempfile.mkstemp(suffix=ext, dir=_TEMP_DIR)
                    with os.fdopen(fd, "wb") as f:
                        for chunk in resp.iter_bytes():
                            size += len(chunk)
                            if size > _DOWNLOAD_MAX_BYTES:
                                raise ValueError(f"image exceeds {_DOWNLOAD_MAX_BYTES} bytes")
                            f.write(chunk)
                    break
            else:
                log.warning("Too many redirects: %s", url[:80])
                return None

        # 헤더를 믿지 않고 실제로 열어 본다
        try:
            with Image.open(path) as probe:
                probe.verify()
        except Exception as exc:
            log.warning("Downloaded file is not a readable image (%s): %s", exc, url[:80])
            os.unlink(path)
            return None

        log.info("Downloaded image from %s → %s (%d bytes)", url[:80], path, size)
        return path
    except Exception as exc:
        log.warning("Failed to download image %s: %s", url[:80], exc)
        if path:
            try:
                os.unlink(path)
            except OSError:
                pass
        return None


def _save_temp_image(b64_data: str, media_type: str = "image/jpeg") -> str:
    """base64 이미지를 임시 파일로 저장하고 경로를 반환한다."""
    ext = ".jpg"
    if "png" in media_type:
        ext = ".png"
    elif "webp" in media_type:
        ext = ".webp"

    fd, path = tempfile.mkstemp(suffix=ext, dir=_TEMP_DIR)
    with os.fdopen(fd, "wb") as f:
        f.write(base64.b64decode(b64_data))
    return path


def _resize_for_vision(img: Image.Image, max_px: int = _VISION_MAX_PX) -> Image.Image:
    """Vision 분석용으로 이미지를 리사이즈한다. 긴 변이 max_px 이하면 원본 반환."""
    w, h = img.size
    if max(w, h) <= max_px:
        return img
    ratio = max_px / max(w, h)
    new_w, new_h = int(w * ratio), int(h * ratio)
    return img.resize((new_w, new_h), Image.LANCZOS)


def _create_reference_strip(ref_paths: list[str]) -> str | None:
    """대표 사진들을 세로로 이어 붙인 참고용 이미지를 만든다. 없으면 None.

    예전에는 변형할 사진과 대표 사진을 한 장으로 합성해서 보냈다. 그런데 모델은
    "0~1 정규화 좌표"를 자기가 보고 있는 이미지 기준으로 답한다. 합성 이미지에서
    변형할 사진은 왼쪽 2/3뿐이었으므로, 모델이 준 x·width가 모두 1.5배 어긋났다
    (실측: 크롭 결과에 피사체가 5.4%만 들어옴, 의도한 박스는 69.4%).
    좌표를 주는 사진은 반드시 그 사진 하나만 담긴 이미지여야 한다.
    """
    ref_imgs: list[Image.Image] = []
    for rp in ref_paths[:3]:
        try:
            ref_imgs.append(_resize_for_vision(Image.open(rp).convert("RGB"), max_px=512))
        except Exception as exc:
            log.warning("Failed to load reference image %s: %s", rp, exc)
    if not ref_imgs:
        return None

    cell_w = max(i.width for i in ref_imgs)
    cell_h = max(i.height for i in ref_imgs)
    canvas = Image.new("RGB", (cell_w, cell_h * len(ref_imgs)), (40, 40, 40))
    for i, ref in enumerate(ref_imgs):
        scale = max(cell_w / ref.width, cell_h / ref.height)
        scaled = ref.resize((int(ref.width * scale), int(ref.height * scale)), Image.LANCZOS)
        left = (scaled.width - cell_w) // 2
        top = (scaled.height - cell_h) // 2
        canvas.paste(scaled.crop((left, top, left + cell_w, top + cell_h)), (0, i * cell_h))

    fd, path = tempfile.mkstemp(suffix=".jpg", dir=_TEMP_DIR)
    os.close(fd)
    canvas.save(path, "JPEG", quality=85)
    log.info("Created reference strip: %s (%dx%d, %d refs)",
             path, canvas.width, canvas.height, len(ref_imgs))
    return path


def _encode_image_for_vision(img: Image.Image, max_px: int) -> str:
    """모델에 직접 넣을 JPEG base64. 긴 변을 max_px 이하로 줄인다."""
    img = _resize_for_vision(img.convert("RGB"), max_px=max_px)
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=90)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _parse_stream_result(stdout: str) -> dict:
    """stream-json 출력에서 마지막 result 이벤트를 꺼낸다. 없으면 RuntimeError."""
    final: dict | None = None
    for line in stdout.splitlines():
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            event = json.loads(line)
        except ValueError:
            continue
        if isinstance(event, dict) and event.get("type") == "result":
            final = event
    if final is None:
        raise RuntimeError("claude CLI inline: result 이벤트가 없음")
    if final.get("is_error"):
        raise RuntimeError(f"claude CLI inline error: {str(final.get('result'))[:200]}")
    text = final.get("result")
    if not isinstance(text, str) or not text.strip():
        raise RuntimeError("claude CLI inline returned empty result")
    return final


def _call_claude_inline(
    prompt: str,
    system_prompt: str,
    images_b64: list[str],
    timeout: int = 300,
) -> str:
    """이미지를 stream-json 사용자 메시지에 직접 담아 claude -p를 호출한다.

    도구를 하나도 열지 않는다(--tools ""). 모델이 파일을 읽으러 도구를 부를
    필요가 없어 한 번의 응답으로 끝나고, 프롬프트 인젝션이 닿을 도구도 없다.
    이미지는 JPEG base64이고, 메시지 안의 순서가 곧 프롬프트가 말하는 순서다.
    """
    content: list[dict] = [{"type": "text", "text": prompt}]
    for b64 in images_b64:
        content.append({
            "type": "image",
            "source": {"type": "base64", "media_type": "image/jpeg", "data": b64},
        })
    message = {"type": "user", "message": {"role": "user", "content": content}}

    cmd = [
        "claude", "-p",
        "--input-format", "stream-json",
        # print 모드의 stream-json 출력은 --verbose가 있어야 한다
        "--output-format", "stream-json", "--verbose",
        "--model", MODEL,
        "--dangerously-skip-permissions",
        "--strict-mcp-config",
        "--tools", "",
    ]
    if system_prompt:
        cmd.extend(["--system-prompt", system_prompt])

    log.info("Calling claude CLI inline (model=%s, prompt length=%d, images=%d)",
             MODEL, len(prompt), len(images_b64))

    started = time.monotonic()
    result = subprocess.run(
        cmd,
        input=json.dumps(message, ensure_ascii=False) + "\n",
        capture_output=True,
        text=True,
        timeout=timeout,
        env=child_env(os.environ),
    )
    if result.returncode != 0:
        log.error("claude CLI inline returncode: %d", result.returncode)
        log.error("claude CLI inline stderr: %s", result.stderr[:500] if result.stderr else "(empty)")
        raise RuntimeError(
            f"claude CLI inline failed (code={result.returncode}): "
            f"{(result.stderr or result.stdout)[:200]}"
        )

    final = _parse_stream_result(result.stdout)
    usage = final.get("usage") or {}
    log.info(
        "claude CLI inline done: %.1fs (api %.1fs), turns=%s, in=%s cache_read=%s out=%s",
        time.monotonic() - started,
        (final.get("duration_api_ms") or 0) / 1000,
        final.get("num_turns"),
        usage.get("input_tokens"),
        usage.get("cache_read_input_tokens"),
        usage.get("output_tokens"),
    )
    return final["result"]


def _call_claude(
    prompt: str,
    system_prompt: str,
    image_paths: list[str] | None = None,
    timeout: int = 300,
    tools: str | None = None,
) -> str:
    """claude -p 를 subprocess로 호출한다. 이미지 파일은 --add-dir로 접근 허용."""
    cmd = [
        "claude", "-p",
        "--output-format", "text",
        "--model", MODEL,
        "--dangerously-skip-permissions",
        # 사용자 설정의 MCP 서버를 불러오지 않는다 — 분석에 필요 없고,
        # 권한 확인을 건너뛰는 상태라 그대로 쓸 수 있게 된다.
        "--strict-mcp-config",
    ]

    if system_prompt:
        cmd.extend(["--system-prompt", system_prompt])

    # 도구 제한. --allowedTools는 "묻지 않고 허용"일 뿐 나머지 도구를 없애지
    # 않는다. --dangerously-skip-permissions와 함께면 Bash·Write·WebFetch까지
    # 전부 쓸 수 있어, 게시글 텍스트나 이미지에 심은 지시(프롬프트 인젝션)가
    # 명령 실행으로 이어질 수 있었다. --tools가 실제로 쓸 수 있는 도구 목록을
    # 정한다 ("" = 도구 없음).
    if tools is not None:
        cmd.extend(["--tools", tools])
        cmd.extend(["--allowedTools", tools])

    # 이미지 파일이 있으면 해당 디렉토리에 접근 허용
    if image_paths:
        cmd.extend(["--add-dir", _TEMP_DIR])
        # 대표 사진 디렉토리도 추가
        if os.path.isdir(_REF_DIR):
            cmd.extend(["--add-dir", _REF_DIR])

    log.info("Calling claude CLI (model=%s, prompt length=%d, images=%d, tools=%s)",
             MODEL, len(prompt), len(image_paths or []), tools or "all")

    env = child_env(os.environ)

    result = subprocess.run(
        cmd,
        input=prompt,
        capture_output=True,
        text=True,
        timeout=timeout,
        env=env,
    )

    if result.returncode != 0:
        log.error("claude CLI returncode: %d", result.returncode)
        log.error("claude CLI stderr: %s", result.stderr[:500] if result.stderr else "(empty)")
        log.error("claude CLI stdout: %s", result.stdout[:500] if result.stdout else "(empty)")
        raise RuntimeError(
            f"claude CLI failed (code={result.returncode}): "
            f"{result.stderr[:200] or result.stdout[:200]}"
        )

    log.info("claude CLI response length: %d", len(result.stdout))
    if not result.stdout or not result.stdout.strip():
        log.error("claude CLI returned empty response. stderr: %s", result.stderr)
        raise RuntimeError("claude CLI returned empty response")
    return result.stdout


# 자식 CLI에 넘길 환경변수. 중첩 세션 흔적은 지우되 인증 수단은 남긴다.
#
# CLAUDE_CODE_OAUTH_TOKEN이 곧 로그인 자격증명이다. "CLAUDE_CODE_ 접두어를
# 전부 지운다"로 바꿨다가 이걸 같이 날려 서버 전체가 인증 실패했다.
# 접두어 차단은 오타를 막아 주지만, 지우면 안 되는 것을 명시해야 안전하다.
_ENV_KEEP = frozenset({"CLAUDE_CODE_OAUTH_TOKEN"})

# 중첩 세션 표시 + 만료된 API 키. 후자는 .env에 남아 있는 옛 키가
# CLI의 OAuth 경로를 가로채지 않도록 지운다.
_ENV_BLOCK_EXACT = frozenset({"CLAUDECODE", "ANTHROPIC_API_KEY"})
_ENV_BLOCK_PREFIX = ("CLAUDE_CODE_", "CLAUDE_")


def child_env(source: "os._Environ[str] | dict[str, str]") -> dict[str, str]:
    """claude CLI 자식 프로세스에 넘길 환경을 만든다."""
    return {
        k: v
        for k, v in source.items()
        if k in _ENV_KEEP
        or (k not in _ENV_BLOCK_EXACT and not k.startswith(_ENV_BLOCK_PREFIX))
    }


def _cleanup_files(paths: list[str]) -> None:
    """임시 파일 정리."""
    for p in paths:
        try:
            os.unlink(p)
        except OSError:
            pass


def _create_grid_image(
    image_paths: list[str], max_cols: int = 5
) -> tuple[str, str, list[int]]:
    """여러 이미지를 하나의 그리드로 합성한다.

    반환: (합성 이미지 경로, 레이아웃 설명, 그리드 위치 → image_paths 인덱스)

    세 번째 값이 중요하다. 열리지 않는 파일을 건너뛰면 그리드 위치와
    원본 배열의 인덱스가 어긋나는데, 모델은 그리드 위치로 답한다.
    이 매핑 없이 모델의 답을 원본 배열에 그대로 넣으면 다른 사진이
    대표 사진으로 저장된다.
    """
    imgs: list[Image.Image] = []
    valid_indices: list[int] = []
    for i, p in enumerate(image_paths):
        try:
            img = _resize_for_vision(Image.open(p).convert("RGB"), max_px=512)
            imgs.append(img)
            valid_indices.append(i)
        except Exception as exc:
            log.warning("Failed to load image %s: %s", p, exc)

    if not imgs:
        raise RuntimeError("No valid images to create grid")

    # 그리드 레이아웃 계산
    n = len(imgs)
    cols = min(n, max_cols)
    rows = (n + cols - 1) // cols

    # 셀 크기: 모든 셀을 동일 크기로 (가장 큰 이미지 기준)
    cell_w = max(img.size[0] for img in imgs)
    cell_h = max(img.size[1] for img in imgs)
    # 셀 최대 크기 제한
    cell_w = min(cell_w, 400)
    cell_h = min(cell_h, 400)

    canvas_w = cols * cell_w
    canvas_h = rows * cell_h
    canvas = Image.new("RGB", (canvas_w, canvas_h), (40, 40, 40))

    for idx, img in enumerate(imgs):
        row, col = divmod(idx, cols)
        # 셀에 맞게 리사이즈 (비율 유지, 중앙 배치)
        iw, ih = img.size
        scale = min(cell_w / iw, cell_h / ih)
        nw, nh = int(iw * scale), int(ih * scale)
        resized = img.resize((nw, nh), Image.LANCZOS)
        x = col * cell_w + (cell_w - nw) // 2
        y = row * cell_h + (cell_h - nh) // 2
        canvas.paste(resized, (x, y))

    fd, path = tempfile.mkstemp(suffix=".jpg", dir=_TEMP_DIR)
    os.close(fd)
    canvas.save(path, "JPEG", quality=85)

    layout_desc = (
        f"{cols}열 x {rows}행 그리드, 총 {n}장. "
        f"왼쪽 위부터 오른쪽으로 인덱스 0~{n-1} 순서."
    )
    log.info("Created grid image: %s (%dx%d, %d images)", path, canvas_w, canvas_h, n)
    return path, layout_desc, valid_indices


def analyze_user(
    posts: list[dict],
    feeds: list[dict],
    stories: list[dict],
    user_id: str = "",
    **_kwargs,
) -> dict:
    """사용자의 게시글/피드/스토리를 분석하여 스타일 프로필을 반환한다.

    이미지를 하나의 그리드로 합성하여 Read 1회로 분석한다.
    대표 사진 3장을 서버에 영구 저장하여 이후 변형 시 레퍼런스로 사용한다.
    """
    temp_files: list[str] = []
    downloaded_paths: list[str] = []  # 다운로드된 이미지 경로 (인덱스 순서 보존)

    try:
        # 이미지를 임시 파일로 저장 (base64 또는 URL 다운로드)
        max_images = 10
        image_count = 0

        for label, items in [("게시글", posts), ("피드", feeds), ("스토리", stories)]:
            for i, item in enumerate(items):
                if image_count >= max_images:
                    break

                path: str | None = None
                if item.get("image_base64"):
                    media = item.get("media_type", "image/jpeg")
                    path = _save_temp_image(item["image_base64"], media)
                elif item.get("image_url"):
                    path = _download_temp_image(item["image_url"])

                if path:
                    temp_files.append(path)
                    downloaded_paths.append(path)
                    image_count += 1

        prompt_text = ANALYZE_USER_PROMPT.format(
            posts=_format_items(posts),
            feeds=_format_items(feeds),
            stories=_format_items(stories),
        )

        grid_to_source: list[int] = []
        if downloaded_paths:
            # 모든 이미지를 하나의 그리드로 합성
            grid_path, grid_desc, grid_to_source = _create_grid_image(downloaded_paths)
            temp_files.append(grid_path)

            prompt_text += (
                "\n\n=== 첨부 이미지 (그리드) ===\n"
                f"아래 이미지를 읽고(Read) 실제 색감, 밝기, 채도, 톤을 관찰하여 분석에 반영하세요.\n"
                f"그리드 레이아웃: {grid_desc}\n"
                f"그리고 referenceImageIndices에 스타일을 가장 잘 대표하는 사진 3장의 인덱스를 선택하세요.\n"
                f"이미지 경로: {grid_path}\n"
            )

            result = _call_claude(
                prompt_text, ANALYZE_USER_SYSTEM,
                image_paths=[grid_path],
                tools="Read",
            )
        else:
            prompt_text += (
                "\n\n[참고: 이미지를 확인할 수 없습니다. 텍스트 정보만으로 분석하세요.]"
            )
            result = _call_claude(prompt_text, ANALYZE_USER_SYSTEM, tools="Read")

        parsed = _parse_json_response(result)

        # 대표 사진 3장을 영구 저장
        ref_indices = parsed.get("referenceImageIndices", [])
        # 모델이 [2] 대신 2나 null을 주는 경우 — 순회하다 TypeError가 난다
        if isinstance(ref_indices, int) and not isinstance(ref_indices, bool):
            ref_indices = [ref_indices]
        elif not isinstance(ref_indices, list):
            ref_indices = []
        # 모델은 그리드 위치로 답한다 — 원본 인덱스로 되돌려서 넘긴다
        source_indices = [
            grid_to_source[i]
            for i in ref_indices
            if isinstance(i, int) and not isinstance(i, bool)
            and 0 <= i < len(grid_to_source)
        ]
        # 열거형 검증 — 어긋난 값은 중립값으로 되돌리고 기록한다
        style_profile = parsed.get("styleProfile")
        if isinstance(style_profile, dict):
            problems = _validate_style_profile(style_profile)
            if problems:
                log.error("style profile invalid: %s", problems)
                _repair_style_profile(style_profile, problems)
                parsed["warnings"] = [f"프로필 값 보정: {', '.join(problems)}"]
        else:
            log.error("style profile missing from analyze_user response")

        saved_refs = _save_reference_images(user_id, source_indices, downloaded_paths)
        if saved_refs:
            parsed["referenceImages"] = saved_refs
            log.info("Saved %d reference images for user %s", len(saved_refs), user_id)

        return parsed
    finally:
        _cleanup_files(temp_files)


# user_id는 요청 본문·URL 경로에서 검증 없이 들어온다. 그대로 경로에 붙이면
# "../.." 같은 값으로 _REF_DIR 밖을 읽거나 rmtree로 지울 수 있다.
# 실제 값은 Instagram 숫자 ID다 — 넉넉히 영숫자·_.- 만 허용한다.
_USER_ID_RE = re.compile(r"[A-Za-z0-9_.-]{1,64}")


def _user_ref_dir(user_id: str) -> str | None:
    """user_id의 대표 사진 디렉토리. 안전하지 않은 값이면 None."""
    if not isinstance(user_id, str) or not _USER_ID_RE.fullmatch(user_id):
        return None
    if user_id in (".", ".."):
        return None
    base = os.path.realpath(_REF_DIR)
    user_dir = os.path.join(base, user_id)
    # 심볼릭 링크로 밖을 가리키는 경우까지 막는다
    real = os.path.realpath(user_dir)
    if os.path.dirname(real) != base:
        return None
    return user_dir


def _save_reference_images(
    user_id: str,
    indices: list[int],
    downloaded_paths: list[str],
) -> list[str]:
    """대표 사진을 영구 저장 디렉토리에 복사하고 파일명 목록을 반환한다."""
    import shutil

    if not user_id or not indices or not downloaded_paths:
        return []

    user_dir = _user_ref_dir(user_id)
    if user_dir is None:
        log.warning("Invalid user_id for reference images: %r", user_id[:80])
        return []

    # 새 사진은 옆 임시 디렉토리에 먼저 쓰고, 다 되면 바꿔 끼운다.
    # 예전에는 기존 디렉토리를 지운 뒤 복사해, 복사가 실패하면 대표 사진이
    # 하나도 남지 않았다. '~'는 _USER_ID_RE가 허용하지 않는 문자라 임시·백업
    # 디렉토리를 user_id로 가리킬 수 없다.
    base = os.path.dirname(user_dir)
    os.makedirs(base, exist_ok=True)
    tmp_dir = tempfile.mkdtemp(prefix=f"{user_id}~tmp-", dir=base)

    saved: list[str] = []
    try:
        for i, idx in enumerate(indices[:3]):
            if not isinstance(idx, int) or idx < 0 or idx >= len(downloaded_paths):
                continue
            src = downloaded_paths[idx]
            ext = os.path.splitext(src)[1] or ".jpg"
            name = f"ref_{i}{ext}"
            try:
                shutil.copy2(src, os.path.join(tmp_dir, name))
                saved.append(name)
            except Exception as exc:
                log.warning("Failed to save reference image: %s", exc)

        if not saved:
            # 하나도 못 썼으면 기존 대표 사진을 그대로 둔다
            log.warning("No reference images saved for user %s — keeping previous", user_id)
            return []

        # 디렉토리끼리는 비어 있지 않은 대상에 os.replace가 안 된다 —
        # 기존 것을 옆으로 치우고, 새 것을 제자리에 놓고, 치운 것을 지운다.
        old_dir = None
        if os.path.lexists(user_dir):
            old_dir = f"{user_dir}~old-{uuid.uuid4().hex[:8]}"
            os.rename(user_dir, old_dir)
        try:
            os.rename(tmp_dir, user_dir)
        except Exception:
            if old_dir is not None:
                os.rename(old_dir, user_dir)  # 원상 복구
            raise
        tmp_dir = None
        if old_dir is not None:
            shutil.rmtree(old_dir, ignore_errors=True)
        for name in saved:
            log.info("Saved reference image: %s", os.path.join(user_dir, name))
        return saved
    except Exception as exc:
        log.warning("Failed to replace reference images for user %s: %s", user_id, exc)
        return []
    finally:
        if tmp_dir is not None:
            shutil.rmtree(tmp_dir, ignore_errors=True)


def get_reference_image_paths(user_id: str) -> list[str]:
    """사용자의 대표 사진 파일 경로 목록을 반환한다."""
    user_dir = _user_ref_dir(user_id)
    if user_dir is None or not os.path.isdir(user_dir):
        return []
    paths = sorted(
        os.path.join(user_dir, f)
        for f in os.listdir(user_dir)
        if f.startswith("ref_") and os.path.isfile(os.path.join(user_dir, f))
    )
    return paths


# ── 분석 결과 캐시 ──
#
# 분석 한 번이 30~70초, $0.07이다. 그런데 앱이 백그라운드로 가면 소켓이
# 끊기고, 사용자는 같은 사진으로 다시 시도한다. 캐시가 없으면 서버는 이미
# 끝낸 일을 처음부터 다시 하고 요금도 두 번 낸다 (FastAPI 동기 엔드포인트는
# 클라이언트가 끊겨도 끝까지 실행되므로, 첫 요청의 결과는 이미 나와 있다).
_ANALYSIS_CACHE_MAX = 32
_ANALYSIS_CACHE_TTL = 60 * 30  # 30분
_analysis_cache: "OrderedDict[str, tuple[float, dict]]" = OrderedDict()
# FastAPI 동기 엔드포인트는 스레드풀에서 동시에 돈다. 잠금 없이 get/put이
# 겹치면 pop 직후 move_to_end가 KeyError를 내거나 순서가 깨진다.
_analysis_cache_lock = threading.Lock()


def _reference_fingerprint(user_id: str) -> list:
    """대표 사진 파일들의 지문 — 재분석으로 사진이 바뀌면 달라진다.

    copy2는 원본의 mtime을 그대로 옮기므로 mtime만으로는 부족할 수 있다.
    ctime·inode는 복사할 때마다 새로 정해진다.
    """
    if not user_id:
        return []
    fp = []
    for path in get_reference_image_paths(user_id):
        try:
            st = os.stat(path)
        except OSError:
            continue
        fp.append([os.path.basename(path), st.st_mtime_ns, st.st_ctime_ns, st.st_size, st.st_ino])
    return fp


def _analysis_cache_key(image_base64: str, style_profile: dict, user_id: str) -> str:
    """같은 사진 + 같은 프로필 + 같은 사용자 + 같은 대표 사진이면 같은 결과가 나온다.

    대표 사진은 분석 프롬프트에 함께 들어간다. 키에서 빼면 재분석으로 대표
    사진이 바뀐 뒤에도 30분 동안 옛 사진 기준 결과가 나간다.
    """
    h = hashlib.sha256()
    h.update(image_base64.encode())
    h.update(json.dumps(style_profile, sort_keys=True, ensure_ascii=False).encode())
    h.update(user_id.encode())
    h.update(json.dumps(_reference_fingerprint(user_id)).encode())
    return h.hexdigest()


def _analysis_cache_get(key: str) -> dict | None:
    with _analysis_cache_lock:
        entry = _analysis_cache.get(key)
        if entry is None:
            return None
        stored_at, value = entry
        if time.time() - stored_at > _ANALYSIS_CACHE_TTL:
            _analysis_cache.pop(key, None)
            return None
        _analysis_cache.move_to_end(key)
    # 저장된 값은 수정되지 않으므로 복사는 잠금 밖에서 한다
    return copy.deepcopy(value)


def _analysis_cache_put(key: str, value: dict) -> None:
    stored = copy.deepcopy(value)
    with _analysis_cache_lock:
        _analysis_cache[key] = (time.time(), stored)
        _analysis_cache.move_to_end(key)
        while len(_analysis_cache) > _ANALYSIS_CACHE_MAX:
            _analysis_cache.popitem(last=False)


_PHOTO_COORD_RULE = (
    "crop·remove_areas·local_* 의 정규화 좌표(0~1)는 모두 "
    "**이 사진 한 장**을 기준으로 답하세요.\n"
)
_REF_STRIP_NOTE = (
    "이 사용자가 평소 올리는 사진들을 위에서 아래로 이어 붙인 것입니다. "
    "톤·색감의 방향만 참고하세요. "
    "좌표의 기준이 아니며, 이 이미지를 변형하는 것도 아닙니다.\n"
)


def _transform_photo_inline(prompt_text: str, image_base64: str, ref_strip: str | None) -> str:
    """변형할 사진(과 대표 사진 모음)을 메시지에 직접 담아 분석한다."""
    with Image.open(io.BytesIO(base64.b64decode(image_base64))) as src:
        images = [_encode_image_for_vision(src, _INLINE_PHOTO_MAX_PX)]
    full_prompt = prompt_text + "\n\n=== 분석할 이미지 ===\n"
    full_prompt += "첨부한 첫 번째 이미지가 변형할 사진입니다.\n" + _PHOTO_COORD_RULE
    if ref_strip:
        with Image.open(ref_strip) as strip:
            images.append(_encode_image_for_vision(strip, _INLINE_PHOTO_MAX_PX))
        full_prompt += "\n두 번째 이미지는 대표 사진 모음(참고용)입니다. " + _REF_STRIP_NOTE
    return _call_claude_inline(full_prompt, TRANSFORM_PHOTO_SYSTEM, images)


def _transform_photo_read(
    prompt_text: str,
    image_base64: str,
    media_type: str,
    ref_strip: str | None,
    temp_files: list[str],
) -> str:
    """예전 방식: 임시 파일 경로를 주고 모델이 Read 도구로 읽게 한다."""
    image_path = _save_temp_image(image_base64, media_type)
    temp_files.append(image_path)
    full_prompt = prompt_text + "\n\n" + (
        "=== 분석할 이미지 ===\n"
        f"변형할 사진: {image_path}\n"
        "이 사진을 Read로 읽고 분석하세요.\n"
    ) + _PHOTO_COORD_RULE
    image_paths = [image_path]
    if ref_strip:
        image_paths.append(ref_strip)
        full_prompt += (
            f"\n대표 사진 모음(참고용): {ref_strip}\n"
            "Read로 읽으세요. "
        ) + _REF_STRIP_NOTE
    # Read 도구만 허용하여 불필요한 도구 사용 방지
    return _call_claude(
        full_prompt, TRANSFORM_PHOTO_SYSTEM,
        image_paths=image_paths,
        tools="Read",
    )


def transform_photo(
    style_profile: dict,
    image_base64: str,
    media_type: str = "image/jpeg",
    user_id: str = "",
    **_kwargs,
) -> dict:
    """사용자 스타일 프로필에 맞춰 사진 보정 가이드를 반환한다.

    사진(과 대표 사진 모음)을 메시지에 직접 담아 모델 응답 한 번으로 분석한다.
    실패하면 예전 방식(Read 도구)으로 한 번 더 시도한다.
    """
    cache_key = _analysis_cache_key(image_base64, style_profile or {}, user_id)
    cached = _analysis_cache_get(cache_key)
    if cached is not None:
        log.info("transform_photo: 캐시 적중 — 모델 호출 건너뜀 (%s)", cache_key[:12])
        return cached

    temp_files: list[str] = []

    try:
        # 대표 사진 경로 조회
        ref_paths = get_reference_image_paths(user_id) if user_id else []

        # 대표 사진은 별도 이미지로 보낸다 — 변형할 사진과 합성하면 모델이
        # 좌표를 합성 이미지 기준으로 답해 x·width가 1.5배 어긋난다.
        ref_strip = _create_reference_strip(ref_paths)
        if ref_strip:
            temp_files.append(ref_strip)

        prompt_text = TRANSFORM_PHOTO_PROMPT.format(
            style_profile=json.dumps(style_profile, ensure_ascii=False, indent=2),
        )

        result: str | None = None
        if _VISION_INPUT_MODE != "read":
            try:
                result = _transform_photo_inline(prompt_text, image_base64, ref_strip)
            except Exception as exc:
                # 새 경로가 실패해도 분석은 계속돼야 한다 — 예전 방식으로 한 번 더
                log.warning("transform_photo inline failed, falling back to Read: %s", exc)
        if result is None:
            result = _transform_photo_read(
                prompt_text, image_base64, media_type, ref_strip, temp_files,
            )
        parsed = _parse_json_response(result)

        # 스키마 검증 — 어긋나면 부족한 필드만 짚어 한 번 다시 받는다.
        # 이미지를 다시 읽을 필요가 없어 복구 호출은 빠르고 저렴하다.
        problems = _validate_analysis(parsed)
        if problems:
            log.warning("analysis schema invalid: %s — retrying", problems)
            repair_prompt = (
                "직전 응답의 JSON에서 아래 필드가 빠졌거나 타입이 잘못됐습니다:\n"
                + "\n".join(f"- {p}" for p in problems)
                + "\n\n직전 응답 JSON:\n"
                + json.dumps(parsed, ensure_ascii=False)
                + "\n\n같은 분석 내용을 유지한 채 위 필드를 채워 완전한 JSON "
                "하나만 출력하세요. 설명이나 코드블록 없이 JSON만."
            )
            try:
                repaired = _parse_json_response(
                    _call_claude(repair_prompt, TRANSFORM_PHOTO_SYSTEM, tools="")
                )
                remaining = _validate_analysis(repaired)
                if len(remaining) < len(problems):
                    parsed, problems = repaired, remaining
            except Exception as exc:
                log.warning("analysis repair call failed: %s", exc)

        if problems:
            # 여기까지 왔으면 분석 화면 일부가 비지만, 사진 변형은 서버가
            # 측정으로 계산하므로 정상 동작한다. 조용히 넘기지 않고 표시한다.
            log.error("analysis still invalid after repair: %s", problems)
            _apply_analysis_fallbacks(parsed, problems)
            parsed["warnings"] = [f"분석 필드 누락: {', '.join(problems)}"]

        _analysis_cache_put(cache_key, parsed)
        return parsed
    finally:
        _cleanup_files(temp_files)
