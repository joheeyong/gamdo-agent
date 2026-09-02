"""Claude Code CLI(claude -p)를 사용한 분석 클라이언트 — 이미지 Vision 지원."""

import base64
import json
import os
import subprocess
import tempfile
import logging
from typing import Any

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

# 이미지 임시 저장 디렉토리
_TEMP_DIR = os.path.join(tempfile.gettempdir(), "gamdo-images")
os.makedirs(_TEMP_DIR, exist_ok=True)

# 대표 사진 영구 저장 디렉토리
_REF_DIR = os.path.join(os.path.dirname(__file__), "reference_images")
os.makedirs(_REF_DIR, exist_ok=True)


def _parse_json_response(text: str) -> dict:
    """응답 텍스트에서 JSON을 추출한다."""
    text = text.strip()

    # 코드블록 제거
    if "```json" in text:
        text = text.split("```json", 1)[1]
    if "```" in text:
        text = text.split("```")[0]
    text = text.strip()

    # 그래도 안 되면 첫 번째 { 부터 마지막 } 까지 추출
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            return json.loads(text[start : end + 1])
        raise


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
        "strengths": (list,),
        "improvements": (list,),
    },
    "toneReport": {
        "overallMood": (str,),
        "styleCategory": (str,),
        "narrative": (str,),
    },
    "shootingTips": (list,),
    "editingTips": (list,),
    "overallScore": (int, float),
}

# 복구도 실패했을 때 채워 넣는 값. 사진 변형은 서버가 계산하므로 영향이 없고,
# 분석 화면만 비어 보인다 — 응답의 warnings로 그 사실을 알린다.
_ANALYSIS_FALLBACKS: dict[str, Any] = {
    "colorAnalysis": {"colorHarmony": "분석 불가", "paletteDescription": ""},
    "compositionAnalysis": {
        "primaryTechnique": "분석 불가", "balanceScore": 0.5,
        "strengths": [], "improvements": [],
    },
    "toneReport": {"overallMood": "", "styleCategory": "", "narrative": ""},
    "shootingTips": [],
    "editingTips": [],
    "overallScore": 0,
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


def _download_temp_image(url: str) -> str | None:
    """이미지 URL을 다운로드하여 임시 파일로 저장한다. 실패 시 None."""
    try:
        import httpx

        resp = httpx.get(url, timeout=15, follow_redirects=True)
        resp.raise_for_status()

        content_type = resp.headers.get("content-type", "image/jpeg")
        ext = ".jpg"
        if "png" in content_type:
            ext = ".png"
        elif "webp" in content_type:
            ext = ".webp"

        fd, path = tempfile.mkstemp(suffix=ext, dir=_TEMP_DIR)
        with os.fdopen(fd, "wb") as f:
            f.write(resp.content)

        log.info("Downloaded image from %s → %s (%d bytes)", url[:80], path, len(resp.content))
        return path
    except Exception as exc:
        log.warning("Failed to download image %s: %s", url[:80], exc)
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


def _create_composite_image(
    main_path: str,
    ref_paths: list[str],
) -> tuple[str, str]:
    """새 사진 + 대표 사진을 하나의 합성 이미지로 만든다.

    레이아웃: [새 사진(크게)] | [대표1] [대표2] [대표3]
    왼쪽 절반에 새 사진, 오른쪽에 대표 사진들을 세로로 배치.
    대표 사진이 없으면 새 사진만 리사이즈하여 반환.

    반환: (합성 이미지 경로, 레이아웃 설명 텍스트)
    """
    main_img = _resize_for_vision(Image.open(main_path).convert("RGB"))

    if not ref_paths:
        # 대표 사진 없으면 새 사진만 저장
        fd, path = tempfile.mkstemp(suffix=".jpg", dir=_TEMP_DIR)
        os.close(fd)
        main_img.save(path, "JPEG", quality=85)
        return path, "이미지 전체가 변형할 새 사진입니다."

    # 대표 사진 로드 및 리사이즈
    ref_imgs: list[Image.Image] = []
    for rp in ref_paths[:3]:
        try:
            ref_imgs.append(
                _resize_for_vision(Image.open(rp).convert("RGB"), max_px=512)
            )
        except Exception as exc:
            log.warning("Failed to load reference image %s: %s", rp, exc)

    if not ref_imgs:
        fd, path = tempfile.mkstemp(suffix=".jpg", dir=_TEMP_DIR)
        os.close(fd)
        main_img.save(path, "JPEG", quality=85)
        return path, "이미지 전체가 변형할 새 사진입니다."

    # 캔버스 크기 계산: 왼쪽 절반 = 새 사진, 오른쪽 절반 = 대표 사진 세로 스택
    main_w, main_h = main_img.size
    ref_cell_w = main_w // 2  # 오른쪽 영역 너비
    ref_cell_h = main_h // len(ref_imgs)

    canvas_w = main_w + ref_cell_w
    canvas_h = main_h
    canvas = Image.new("RGB", (canvas_w, canvas_h), (40, 40, 40))

    # 왼쪽: 새 사진
    canvas.paste(main_img, (0, 0))

    # 오른쪽: 대표 사진들을 세로로 배치
    for i, ref in enumerate(ref_imgs):
        # 셀에 맞게 리사이즈 (비율 유지, 중앙 크롭)
        rw, rh = ref.size
        scale = max(ref_cell_w / rw, ref_cell_h / rh)
        scaled = ref.resize((int(rw * scale), int(rh * scale)), Image.LANCZOS)
        sw, sh = scaled.size
        left = (sw - ref_cell_w) // 2
        top = (sh - ref_cell_h) // 2
        cropped = scaled.crop((left, top, left + ref_cell_w, top + ref_cell_h))
        canvas.paste(cropped, (main_w, i * ref_cell_h))

    fd, path = tempfile.mkstemp(suffix=".jpg", dir=_TEMP_DIR)
    os.close(fd)
    canvas.save(path, "JPEG", quality=85)

    ref_count = len(ref_imgs)
    layout_desc = (
        f"합성 이미지 레이아웃: 왼쪽 2/3가 변형할 새 사진, "
        f"오른쪽 1/3에 대표 사진 {ref_count}장이 위에서 아래로 배치되어 있습니다. "
        f"새 사진의 색감·구도·피사체를 분석하고, 오른쪽 대표 사진들의 톤·색감을 참고하세요."
    )
    log.info("Created composite image: %s (%dx%d, %d refs)",
             path, canvas_w, canvas_h, ref_count)
    return path, layout_desc


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
    ]

    if system_prompt:
        cmd.extend(["--system-prompt", system_prompt])

    # 도구 제한 (Read만 허용하면 불필요한 도구 사용 방지)
    if tools is not None:
        cmd.extend(["--allowedTools", tools])

    # 이미지 파일이 있으면 해당 디렉토리에 접근 허용
    if image_paths:
        cmd.extend(["--add-dir", _TEMP_DIR])
        # 대표 사진 디렉토리도 추가
        if os.path.isdir(_REF_DIR):
            cmd.extend(["--add-dir", _REF_DIR])

    log.info("Calling claude CLI (model=%s, prompt length=%d, images=%d, tools=%s)",
             MODEL, len(prompt), len(image_paths or []), tools or "all")

    # 중첩 세션 환경변수 + 만료된 API 키 제거 → CLI 자체 인증(OAuth) 사용
    blocked = {"CLAUDECODE", "CLAUDE_CODE_ENTRY_POINT", "CLAUDE_CODE_SESSION", "ANTHROPIC_API_KEY"}
    env = {k: v for k, v in os.environ.items() if k not in blocked}

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


def _cleanup_files(paths: list[str]) -> None:
    """임시 파일 정리."""
    for p in paths:
        try:
            os.unlink(p)
        except OSError:
            pass


def _create_grid_image(image_paths: list[str], max_cols: int = 5) -> tuple[str, str]:
    """여러 이미지를 하나의 그리드로 합성한다.

    반환: (합성 이미지 경로, 레이아웃 설명)
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
    return path, layout_desc


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

        if downloaded_paths:
            # 모든 이미지를 하나의 그리드로 합성
            grid_path, grid_desc = _create_grid_image(downloaded_paths)
            temp_files.append(grid_path)

            prompt_text += (
                "\n\n=== 첨부 이미지 (그리드) ===\n"
                f"아래 이미지를 읽고(Read) 실제 색감, 밝기, 채도, 톤을 관찰하여 분석에 반영하세요.\n"
                f"그리드 레이아웃: {grid_desc}\n"
                f"특히 targetParams는 이 사진들의 공통된 보정 특성을 수치로 정확히 표현해야 합니다.\n"
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
        saved_refs = _save_reference_images(user_id, ref_indices, downloaded_paths)
        if saved_refs:
            parsed["referenceImages"] = saved_refs
            log.info("Saved %d reference images for user %s", len(saved_refs), user_id)

        return parsed
    finally:
        _cleanup_files(temp_files)


def _save_reference_images(
    user_id: str,
    indices: list[int],
    downloaded_paths: list[str],
) -> list[str]:
    """대표 사진을 영구 저장 디렉토리에 복사하고 파일명 목록을 반환한다."""
    import shutil

    if not user_id or not indices or not downloaded_paths:
        return []

    user_dir = os.path.join(_REF_DIR, user_id)
    # 기존 대표 사진 삭제 후 새로 저장
    if os.path.exists(user_dir):
        shutil.rmtree(user_dir)
    os.makedirs(user_dir, exist_ok=True)

    saved: list[str] = []
    for i, idx in enumerate(indices[:3]):
        if not isinstance(idx, int) or idx < 0 or idx >= len(downloaded_paths):
            continue
        src = downloaded_paths[idx]
        ext = os.path.splitext(src)[1] or ".jpg"
        dst = os.path.join(user_dir, f"ref_{i}{ext}")
        try:
            shutil.copy2(src, dst)
            saved.append(f"ref_{i}{ext}")
            log.info("Saved reference image: %s", dst)
        except Exception as exc:
            log.warning("Failed to save reference image: %s", exc)

    return saved


def get_reference_image_paths(user_id: str) -> list[str]:
    """사용자의 대표 사진 파일 경로 목록을 반환한다."""
    user_dir = os.path.join(_REF_DIR, user_id)
    if not os.path.isdir(user_dir):
        return []
    paths = sorted(
        os.path.join(user_dir, f)
        for f in os.listdir(user_dir)
        if f.startswith("ref_") and os.path.isfile(os.path.join(user_dir, f))
    )
    return paths


def transform_photo(
    style_profile: dict,
    image_base64: str,
    media_type: str = "image/jpeg",
    user_id: str = "",
    **_kwargs,
) -> dict:
    """사용자 스타일 프로필에 맞춰 사진 보정 가이드를 반환한다.

    새 사진 + 대표 사진을 하나의 합성 이미지로 만들어 Read 1회로 분석한다.
    (기존: 파일 4개를 각각 Read → API 왕복 4~5회 → 합성 이미지 1회로 단축)
    """
    temp_files: list[str] = []

    try:
        # 새 사진을 임시 파일로 저장
        image_path = _save_temp_image(image_base64, media_type)
        temp_files.append(image_path)

        # 대표 사진 경로 조회
        ref_paths = get_reference_image_paths(user_id) if user_id else []

        # 합성 이미지 생성 (새 사진 + 대표 사진 → 1장)
        composite_path, layout_desc = _create_composite_image(image_path, ref_paths)
        if composite_path != image_path:
            temp_files.append(composite_path)

        prompt_text = TRANSFORM_PHOTO_PROMPT.format(
            style_profile=json.dumps(style_profile, ensure_ascii=False, indent=2),
        )

        # 프롬프트 조립 — 합성 이미지 1장만 Read하도록 안내
        full_prompt = prompt_text + "\n\n"
        full_prompt += (
            "=== 분석할 이미지 ===\n"
            f"{layout_desc}\n"
            f"이미지를 읽고(Read) 분석하세요: {composite_path}\n"
        )

        # Read 도구만 허용하여 불필요한 도구 사용 방지
        result = _call_claude(
            full_prompt, TRANSFORM_PHOTO_SYSTEM,
            image_paths=[composite_path],
            tools="Read",
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

        return parsed
    finally:
        _cleanup_files(temp_files)
