"""이미지를 메시지에 직접 담는 분석 경로 (stream-json, 도구 없음) 검증.

Read 도구로 파일을 읽게 하면 "도구 호출 → 결과 → 답변"으로 모델 왕복이
한 번 더 생긴다. 인라인 경로는 도구를 하나도 열지 않고 한 번에 끝낸다.
Read 경로는 없앴다 — 실패를 다시 시도하면 최악 대기만 두 배가 됐다.
"""

import base64
import io
import json
import subprocess

import pytest
from PIL import Image

import claude_client as cc

_ANALYSIS = json.dumps({
    "colorAnalysis": {"colorHarmony": "x", "paletteDescription": ""},
    "compositionAnalysis": {"primaryTechnique": "x", "balanceScore": 0.5},
    "toneReport": {"overallMood": "x", "styleCategory": "x", "narrative": "x"},
    "subjectType": "풍경",
})


def _jpeg_b64(w=3413, h=2560) -> str:
    buf = io.BytesIO()
    Image.new("RGB", (w, h), (120, 160, 200)).save(buf, "JPEG")
    return base64.b64encode(buf.getvalue()).decode()


def _stream_stdout(text: str, **extra) -> str:
    events = [
        {"type": "system", "subtype": "init"},
        {"type": "assistant", "message": {"content": [{"type": "text", "text": text}]}},
        {"type": "result", "subtype": "success", "is_error": False, "result": text,
         "num_turns": 1, "duration_api_ms": 4200,
         "usage": {"input_tokens": 10, "cache_read_input_tokens": 20000, "output_tokens": 300},
         **extra},
    ]
    return "\n".join(json.dumps(e) for e in events) + "\n"


@pytest.fixture(autouse=True)
def _no_cache(monkeypatch, tmp_path):
    monkeypatch.setattr(cc, "_analysis_cache_get", lambda key: None)
    monkeypatch.setattr(cc, "_analysis_cache_put", lambda key, value, user_id="": None)
    monkeypatch.setattr(cc, "_TEMP_DIR", str(tmp_path))


def _capture_run(monkeypatch, stdout: str, returncode: int = 0):
    calls = []

    def fake_run(cmd, **kw):
        calls.append((cmd, kw))
        return subprocess.CompletedProcess(cmd, returncode, stdout=stdout, stderr="")

    monkeypatch.setattr(cc.subprocess, "run", fake_run)
    return calls


def test_inline_call_uses_stream_json_and_no_tools(monkeypatch):
    calls = _capture_run(monkeypatch, _stream_stdout(_ANALYSIS))
    out = cc._call_claude_inline("프롬프트", "시스템", ["QUJD"])
    assert out == _ANALYSIS

    cmd, kw = calls[0]
    assert cmd[cmd.index("--input-format") + 1] == "stream-json"
    assert cmd[cmd.index("--output-format") + 1] == "stream-json"
    assert "--verbose" in cmd
    # 도구를 하나도 열지 않는다 — Read도, Bash도 없음
    assert cmd[cmd.index("--tools") + 1] == ""
    assert "--allowedTools" not in cmd and "--add-dir" not in cmd
    assert "--strict-mcp-config" in cmd

    msg = json.loads(kw["input"])
    content = msg["message"]["content"]
    assert msg["type"] == "user" and msg["message"]["role"] == "user"
    assert content[0] == {"type": "text", "text": "프롬프트"}
    assert content[1]["type"] == "image"
    assert content[1]["source"] == {"type": "base64", "media_type": "image/jpeg", "data": "QUJD"}


@pytest.mark.parametrize("stdout, match", [
    ("", "result 이벤트"),
    ('{"type": "assistant"}\n', "result 이벤트"),
    ('{"type": "result", "is_error": true, "result": "overloaded"}\n', "overloaded"),
    ('{"type": "result", "is_error": false, "result": "  "}\n', "empty"),
    ("not json\n", "result 이벤트"),
])
def test_parse_stream_result_errors(stdout, match):
    with pytest.raises(RuntimeError, match=match):
        cc._parse_stream_result(stdout)


def test_inline_nonzero_exit_raises(monkeypatch):
    _capture_run(monkeypatch, "", returncode=1)
    with pytest.raises(RuntimeError, match="inline failed"):
        cc._call_claude_inline("p", "s", ["QUJD"])


def test_transform_photo_inline_skips_read_and_downsizes(monkeypatch):
    calls = _capture_run(monkeypatch, _stream_stdout(_ANALYSIS))

    result = cc.transform_photo({}, _jpeg_b64())
    assert result["subjectType"] == "풍경"

    msg = json.loads(calls[0][1]["input"])
    images = [c for c in msg["message"]["content"] if c["type"] == "image"]
    assert len(images) == 1
    with Image.open(io.BytesIO(base64.b64decode(images[0]["source"]["data"]))) as sent:
        assert max(sent.size) == cc._INLINE_PHOTO_MAX_PX
    text = msg["message"]["content"][0]["text"]
    assert "첫 번째 이미지" in text and "Read" not in text.split("=== 분석할 이미지 ===")[1]


def test_transform_photo_sends_reference_strip_second(monkeypatch, tmp_path):
    strip = tmp_path / "strip.jpg"
    Image.new("RGB", (512, 1536), (10, 20, 30)).save(strip, "JPEG")
    monkeypatch.setattr(cc, "get_reference_image_paths", lambda uid: ["/unused"])
    monkeypatch.setattr(cc, "_create_reference_strip", lambda paths: str(strip))
    calls = _capture_run(monkeypatch, _stream_stdout(_ANALYSIS))

    cc.transform_photo({}, _jpeg_b64(800, 600), user_id="123")
    msg = json.loads(calls[0][1]["input"])
    images = [c for c in msg["message"]["content"] if c["type"] == "image"]
    assert len(images) == 2
    assert "두 번째 이미지는 대표 사진 모음" in msg["message"]["content"][0]["text"]


def test_transform_photo_does_not_retry_when_inline_fails(monkeypatch):
    calls = _capture_run(monkeypatch, "", returncode=1)
    with pytest.raises(RuntimeError):
        cc.transform_photo({}, _jpeg_b64(800, 600))
    assert len(calls) == 1


def _model_returns(monkeypatch, *responses):
    seq = list(responses)
    calls = []

    def fake_inline(prompt, system, images, **k):
        calls.append((prompt, images, k))
        return seq.pop(0)

    monkeypatch.setattr(cc, "_call_claude_inline", fake_inline)
    return calls


def test_repair_keeps_unchecked_keys(monkeypatch):
    """복구 응답이 autoEdits 등을 빠뜨려도 원래 값이 남는다."""
    first = json.loads(_ANALYSIS)
    del first["toneReport"]
    first["autoEdits"] = {"crop": {"x": 0.1, "y": 0, "width": 0.8, "height": 1}}
    repaired = json.loads(_ANALYSIS)  # autoEdits 없음
    calls = _model_returns(monkeypatch, json.dumps(first), json.dumps(repaired))
    result = cc.transform_photo({}, _jpeg_b64(80, 60))
    assert result["autoEdits"] == first["autoEdits"]
    assert result["toneReport"]["styleCategory"] == "x"
    assert "warnings" not in result
    # 복구 호출은 이미지 없이, 짧은 시간 상한으로
    assert calls[1][1] == [] and calls[1][2]["timeout"] == cc._REPAIR_TIMEOUT


def test_unknown_subject_type_falls_back_and_is_not_cached(monkeypatch):
    bad = json.loads(_ANALYSIS)
    bad["subjectType"] = "인물 사진"
    _model_returns(monkeypatch, json.dumps(bad), json.dumps(bad))
    stored = []
    monkeypatch.setattr(cc, "_analysis_cache_put", lambda *a, **k: stored.append(a))
    result = cc.transform_photo({}, _jpeg_b64(80, 60))
    assert result["subjectType"] == "혼합"
    assert result["warnings"]
    assert stored == []  # 기본값으로 메운 결과는 캐시하지 않는다


def test_subject_type_whitespace_is_normalized(monkeypatch):
    ok = json.loads(_ANALYSIS)
    ok["subjectType"] = " 인물 "
    _model_returns(monkeypatch, json.dumps(ok))
    assert cc.transform_photo({}, _jpeg_b64(80, 60))["subjectType"] == "인물"


def test_fallback_replaces_null_values():
    parsed = {"toneReport": {"overallMood": None, "styleCategory": "x", "narrative": "y"}}
    problems = cc._validate_analysis(parsed)
    cc._apply_analysis_fallbacks(parsed, problems)
    assert parsed["toneReport"]["overallMood"] == ""
    assert parsed["toneReport"]["styleCategory"] == "x"


def test_disabled_toggles_tell_model_to_skip(monkeypatch):
    calls = _model_returns(monkeypatch, _ANALYSIS)
    cc.transform_photo({}, _jpeg_b64(80, 60), reshape_enabled=False, skin_retouch_enabled=False)
    prompt = calls[0][0]
    assert "reshapeParams를 생략" in prompt and "blemish_removal·skin_smoothing을 생략" in prompt


def test_toggles_change_cache_key():
    on = cc._analysis_cache_key("img", {}, "", (True, True))
    off = cc._analysis_cache_key("img", {}, "", (False, True))
    assert on != off
