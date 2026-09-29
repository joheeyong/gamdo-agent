"""이미지를 메시지에 직접 담는 분석 경로 (stream-json, 도구 없음) 검증.

Read 도구로 파일을 읽게 하면 "도구 호출 → 결과 → 답변"으로 모델 왕복이
한 번 더 생긴다. 인라인 경로는 도구를 하나도 열지 않고 한 번에 끝낸다.
실패하면 예전 Read 경로로 넘어가야 분석이 멈추지 않는다.
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
    monkeypatch.setattr(cc, "_analysis_cache_put", lambda key, value: None)
    monkeypatch.setattr(cc, "_TEMP_DIR", str(tmp_path))
    monkeypatch.setattr(cc, "_VISION_INPUT_MODE", "inline")


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
    monkeypatch.setattr(cc, "_call_claude", lambda *a, **k: pytest.fail("Read 경로를 타면 안 됨"))

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


def test_transform_photo_falls_back_to_read_when_inline_fails(monkeypatch):
    _capture_run(monkeypatch, "", returncode=1)
    read_calls = []

    def fake_call_claude(prompt, system, image_paths=None, tools=None, **k):
        read_calls.append((prompt, image_paths, tools))
        return _ANALYSIS

    monkeypatch.setattr(cc, "_call_claude", fake_call_claude)
    result = cc.transform_photo({}, _jpeg_b64(800, 600))
    assert result["subjectType"] == "풍경"
    prompt, image_paths, tools = read_calls[0]
    assert tools == "Read" and len(image_paths) == 1
    assert "Read로 읽고" in prompt


def test_read_mode_env_skips_inline(monkeypatch):
    monkeypatch.setattr(cc, "_VISION_INPUT_MODE", "read")
    monkeypatch.setattr(cc.subprocess, "run", lambda *a, **k: pytest.fail("인라인 경로를 타면 안 됨"))
    monkeypatch.setattr(cc, "_call_claude", lambda *a, **k: _ANALYSIS)
    assert cc.transform_photo({}, _jpeg_b64(800, 600))["subjectType"] == "풍경"
