"""claude_client의 입력 검증·파싱·동시성 회귀 테스트.

- user_id 경로 탈출 (rmtree로 _REF_DIR 밖을 지울 수 있었다)
- claude CLI 도구 제한 (--allowedTools만으로는 Bash 등이 막히지 않았다)
- image_url SSRF / 무제한 다운로드
- 태그 없는 ``` 앞에 설명문이 있으면 JSON 파싱 실패
- referenceImageIndices가 정수 하나일 때 TypeError
- 분석 캐시 동시 접근
"""

import io
import os
import threading

import httpx
import pytest
from PIL import Image

import claude_client as cc


# ── user_id 경로 ──


@pytest.fixture
def ref_dir(tmp_path, monkeypatch):
    d = tmp_path / "reference_images"
    d.mkdir()
    monkeypatch.setattr(cc, "_REF_DIR", str(d))
    return d


@pytest.mark.parametrize("uid", ["17841400000000000", "abc_DEF-1.2", "a" * 64])
def test_legit_user_ids_accepted(ref_dir, uid):
    assert cc._user_ref_dir(uid) == os.path.join(os.path.realpath(ref_dir), uid)


@pytest.mark.parametrize(
    "uid", ["", ".", "..", "../x", "a/b", "/etc", "..\\x", "a" * 65, "a\x00b", "한글"]
)
def test_bad_user_ids_rejected(ref_dir, uid):
    assert cc._user_ref_dir(uid) is None
    assert cc.get_reference_image_paths(uid) == []


def test_traversal_does_not_delete_outside(tmp_path, ref_dir):
    victim = tmp_path / "victim"
    victim.mkdir()
    (victim / "keep.txt").write_text("x")
    src = tmp_path / "src.jpg"
    src.write_bytes(b"img")

    assert cc._save_reference_images("../victim", [0], [str(src)]) == []
    assert (victim / "keep.txt").exists()


def test_symlink_escape_rejected(tmp_path, ref_dir):
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "ref_0.jpg").write_bytes(b"secret")
    os.symlink(outside, ref_dir / "evil")

    assert cc.get_reference_image_paths("evil") == []
    src = tmp_path / "src.jpg"
    src.write_bytes(b"img")
    assert cc._save_reference_images("evil", [0], [str(src)]) == []
    assert (outside / "ref_0.jpg").exists()


def test_save_and_get_roundtrip(tmp_path, ref_dir):
    src = tmp_path / "src.jpg"
    src.write_bytes(b"img")
    assert cc._save_reference_images("12345", [0], [str(src)]) == ["ref_0.jpg"]
    assert cc.get_reference_image_paths("12345") == [str(ref_dir / "12345" / "ref_0.jpg")]


# ── claude CLI 도구 제한 ──


class _FakeResult:
    returncode = 0
    stdout = "{}"
    stderr = ""


def _capture_cmd(monkeypatch):
    seen = {}

    def fake_run(cmd, **kwargs):
        seen["cmd"] = cmd
        return _FakeResult()

    monkeypatch.setattr(cc.subprocess, "run", fake_run)
    return seen


def _flag_value(cmd, flag):
    return cmd[cmd.index(flag) + 1]


def test_call_claude_restricts_available_tools(monkeypatch):
    seen = _capture_cmd(monkeypatch)
    cc._call_claude("p", "s", image_paths=["/x.jpg"], tools="Read")
    cmd = seen["cmd"]
    # --allowedTools만으로는 다른 도구가 사라지지 않는다 — --tools가 있어야 한다
    assert _flag_value(cmd, "--tools") == "Read"
    assert "--strict-mcp-config" in cmd


def test_call_claude_no_tools(monkeypatch):
    seen = _capture_cmd(monkeypatch)
    cc._call_claude("p", "s", tools="")
    assert _flag_value(seen["cmd"], "--tools") == ""


# ── image_url 허용 목록 ──


@pytest.mark.parametrize("url", [
    "https://scontent-ssn1-1.cdninstagram.com/v/t51.29350-15/abc.jpg?stp=dst-jpg",
    "https://scontent.xx.fbcdn.net/v/t51/abc.jpg",
    "https://video.cdninstagram.com/o1/v/t16/x.jpg",
])
def test_instagram_cdn_allowed(url):
    assert cc._is_allowed_image_url(url)


@pytest.mark.parametrize("url", [
    "http://scontent.cdninstagram.com/a.jpg",       # https만
    "https://169.254.169.254/latest/meta-data/",
    "https://localhost/a.jpg",
    "https://evilcdninstagram.com/a.jpg",            # 접미사 흉내
    "https://cdninstagram.com.evil.com/a.jpg",
    "https://scontent.cdninstagram.com:8080/a.jpg",  # 다른 포트
    "file:///etc/passwd",
    "not a url",
])
def test_other_urls_rejected(url):
    assert not cc._is_allowed_image_url(url)


def _jpeg_bytes():
    buf = io.BytesIO()
    Image.new("RGB", (8, 8), (200, 10, 10)).save(buf, "JPEG")
    return buf.getvalue()


@pytest.fixture
def fake_http(monkeypatch, tmp_path):
    """httpx.Client를 MockTransport로 바꾼다. handler는 테스트가 채운다."""
    monkeypatch.setattr(cc, "_TEMP_DIR", str(tmp_path))
    state = {"handler": None, "requested": []}
    real_client = httpx.Client

    def handler(request):
        state["requested"].append(str(request.url))
        return state["handler"](request)

    monkeypatch.setattr(
        httpx, "Client",
        lambda **kw: real_client(transport=httpx.MockTransport(handler), **kw),
    )
    return state


def test_download_normal_flow(fake_http):
    fake_http["handler"] = lambda r: httpx.Response(
        200, headers={"content-type": "image/jpeg"}, content=_jpeg_bytes())
    path = cc._download_temp_image("https://scontent.cdninstagram.com/a.jpg")
    assert path and os.path.exists(path)


def test_download_disallowed_host_not_requested(fake_http):
    fake_http["handler"] = lambda r: httpx.Response(200, content=_jpeg_bytes())
    assert cc._download_temp_image("http://127.0.0.1:8000/a.jpg") is None
    assert fake_http["requested"] == []


def test_download_redirect_to_internal_blocked(fake_http):
    fake_http["handler"] = lambda r: httpx.Response(
        302, headers={"location": "http://169.254.169.254/latest/"})
    assert cc._download_temp_image("https://scontent.cdninstagram.com/a.jpg") is None
    assert fake_http["requested"] == ["https://scontent.cdninstagram.com/a.jpg"]


def test_download_redirect_within_cdn_followed(fake_http):
    def handler(r):
        if r.url.host == "scontent.cdninstagram.com":
            return httpx.Response(302, headers={"location": "https://scontent.xx.fbcdn.net/b.jpg"})
        return httpx.Response(200, headers={"content-type": "image/jpeg"}, content=_jpeg_bytes())

    fake_http["handler"] = handler
    assert cc._download_temp_image("https://scontent.cdninstagram.com/a.jpg")


def test_download_size_cap(fake_http, monkeypatch, tmp_path):
    monkeypatch.setattr(cc, "_DOWNLOAD_MAX_BYTES", 1000)
    # content-length 없이 흘려보내도 받는 도중 끊어야 한다
    fake_http["handler"] = lambda r: httpx.Response(
        200, headers={"content-type": "image/jpeg"},
        stream=httpx.ByteStream(b"\xff" * 5000))
    assert cc._download_temp_image("https://scontent.cdninstagram.com/a.jpg") is None
    assert list(tmp_path.iterdir()) == []  # 반쯤 받은 파일도 지운다


# ── JSON 파싱 ──


def test_bare_fence_after_prose():
    text = "분석 결과입니다:\n```\n{\"a\": 1}\n```\n끝."
    assert cc._parse_json_response(text) == {"a": 1}


def test_json_fence_and_plain():
    assert cc._parse_json_response("```json\n{\"a\": 1}\n```") == {"a": 1}
    assert cc._parse_json_response("{\"a\": 1}") == {"a": 1}


def test_trailing_prose_with_braces():
    text = "{\"a\": 1}\n\n참고: {이건 JSON 아님}"
    assert cc._parse_json_response(text) == {"a": 1}


def test_nan_still_becomes_none():
    assert cc._parse_json_response("```\n{\"x\": NaN}\n```") == {"x": None}


def test_non_object_rejected():
    with pytest.raises(ValueError):
        cc._parse_json_response("[1, 2, 3]")
    with pytest.raises(ValueError):
        cc._parse_json_response("JSON 없음")


# ── referenceImageIndices ──


@pytest.mark.parametrize("indices", [1, None, "1", True])
def test_reference_indices_odd_types(monkeypatch, tmp_path, indices):
    monkeypatch.setattr(cc, "_TEMP_DIR", str(tmp_path))
    captured = {}
    monkeypatch.setattr(
        cc, "_call_claude",
        lambda *a, **k: '{"styleProfile": {}, "referenceImageIndices": %s}'
        % ("null" if indices is None else ('"1"' if indices == "1" else str(indices).lower())),
    )
    monkeypatch.setattr(
        cc, "_save_reference_images",
        lambda uid, idx, paths: captured.setdefault("idx", idx) and [],
    )
    buf = io.BytesIO()
    Image.new("RGB", (8, 8)).save(buf, "JPEG")
    import base64
    b64 = base64.b64encode(buf.getvalue()).decode()
    posts = [{"image_base64": b64}, {"image_base64": b64}]

    cc.analyze_user(posts, [], [], user_id="1")
    assert captured["idx"] == ([1] if type(indices) is int else [])


# ── 분석 캐시 동시성 ──


def test_cache_concurrent_access(monkeypatch):
    monkeypatch.setattr(cc, "_analysis_cache", cc.OrderedDict())
    monkeypatch.setattr(cc, "_ANALYSIS_CACHE_MAX", 4)
    errors = []

    def worker(n):
        try:
            for i in range(500):
                key = f"k{(n + i) % 8}"
                cc._analysis_cache_put(key, {"v": i})
                cc._analysis_cache_get(key)
                cc._analysis_cache_get(f"k{i % 8}")
        except Exception as exc:  # noqa: BLE001
            errors.append(exc)

    threads = [threading.Thread(target=worker, args=(n,)) for n in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert errors == []
    assert len(cc._analysis_cache) <= 4
