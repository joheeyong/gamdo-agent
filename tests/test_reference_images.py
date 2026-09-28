"""대표 사진 저장 교체의 원자성 + 분석 캐시 키의 대표 사진 지문."""

import os
import shutil

import pytest

import claude_client as cc


@pytest.fixture
def ref_dir(tmp_path, monkeypatch):
    base = tmp_path / "refs"
    base.mkdir()
    monkeypatch.setattr(cc, "_REF_DIR", str(base))
    return base


def _src(tmp_path, name: str, content: bytes) -> str:
    p = tmp_path / name
    p.write_bytes(content)
    return str(p)


def _names(base, uid="123"):
    return sorted(os.listdir(base / uid))


# ── C. 교체 ──


def test_save_creates_and_replaces(ref_dir, tmp_path):
    a = _src(tmp_path, "a.jpg", b"A")
    b = _src(tmp_path, "b.png", b"B")
    assert cc._save_reference_images("123", [0, 1], [a, b]) == ["ref_0.jpg", "ref_1.png"]
    assert _names(ref_dir) == ["ref_0.jpg", "ref_1.png"]

    c = _src(tmp_path, "c.jpg", b"C")
    assert cc._save_reference_images("123", [0], [c]) == ["ref_0.jpg"]
    # 옛 ref_1.png는 사라지고 새 것만 남는다
    assert _names(ref_dir) == ["ref_0.jpg"]
    assert (ref_dir / "123" / "ref_0.jpg").read_bytes() == b"C"
    # 임시·백업 디렉토리가 남지 않는다
    assert os.listdir(ref_dir) == ["123"]


def test_failed_copies_keep_previous_images(ref_dir, tmp_path, monkeypatch):
    a = _src(tmp_path, "a.jpg", b"A")
    cc._save_reference_images("123", [0], [a])

    def broken_copy(src, dst):
        raise OSError("disk full")

    monkeypatch.setattr(shutil, "copy2", broken_copy)
    assert cc._save_reference_images("123", [0], [_src(tmp_path, "n.jpg", b"N")]) == []
    # 예전에는 여기서 대표 사진이 모두 사라졌다
    assert _names(ref_dir) == ["ref_0.jpg"]
    assert (ref_dir / "123" / "ref_0.jpg").read_bytes() == b"A"
    assert os.listdir(ref_dir) == ["123"]


def test_invalid_indices_keep_previous_images(ref_dir, tmp_path):
    a = _src(tmp_path, "a.jpg", b"A")
    cc._save_reference_images("123", [0], [a])
    assert cc._save_reference_images("123", [5, -1, "x"], [a]) == []
    assert _names(ref_dir) == ["ref_0.jpg"]


def test_swap_failure_restores_previous(ref_dir, tmp_path, monkeypatch):
    cc._save_reference_images("123", [0], [_src(tmp_path, "a.jpg", b"A")])
    real_rename = os.rename

    def flaky_rename(src, dst):
        if "~tmp-" in str(src):
            raise OSError("rename failed")
        return real_rename(src, dst)

    monkeypatch.setattr(cc.os, "rename", flaky_rename)
    assert cc._save_reference_images("123", [0], [_src(tmp_path, "b.jpg", b"B")]) == []
    assert (ref_dir / "123" / "ref_0.jpg").read_bytes() == b"A"
    assert os.listdir(ref_dir) == ["123"]


@pytest.mark.parametrize("bad", ["..", ".", "../x", "a/b", "", "a~tmp-x", "x" * 65])
def test_path_traversal_still_blocked(ref_dir, tmp_path, bad):
    a = _src(tmp_path, "a.jpg", b"A")
    assert cc._save_reference_images(bad, [0], [a]) == []
    assert os.listdir(ref_dir) == []
    assert cc.get_reference_image_paths(bad) == []


def test_symlink_escape_still_blocked(ref_dir, tmp_path):
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "keep.txt").write_text("x")
    os.symlink(outside, ref_dir / "evil")
    assert cc._save_reference_images("evil", [0], [_src(tmp_path, "a.jpg", b"A")]) == []
    assert (outside / "keep.txt").exists()


def test_leftover_temp_dirs_are_not_readable_as_user(ref_dir):
    (ref_dir / "123~tmp-abc").mkdir()
    (ref_dir / "123~tmp-abc" / "ref_0.jpg").write_bytes(b"X")
    assert cc.get_reference_image_paths("123~tmp-abc") == []
    assert cc.get_reference_image_paths("123") == []


# ── B. 캐시 키 ──


def test_cache_key_changes_when_reference_images_change(ref_dir, tmp_path):
    k0 = cc._analysis_cache_key("img", {"a": 1}, "123")
    cc._save_reference_images("123", [0], [_src(tmp_path, "a.jpg", b"AAAA")])
    k1 = cc._analysis_cache_key("img", {"a": 1}, "123")
    assert k1 != k0
    # 바뀌지 않았으면 같은 키
    assert cc._analysis_cache_key("img", {"a": 1}, "123") == k1
    # 같은 크기·같은 원본 mtime(copy2가 보존)이어도 교체되면 달라진다
    src = _src(tmp_path, "b.jpg", b"BBBB")
    os.utime(src, ns=(os.stat(tmp_path / "a.jpg").st_atime_ns, os.stat(tmp_path / "a.jpg").st_mtime_ns))
    cc._save_reference_images("123", [0], [src])
    k2 = cc._analysis_cache_key("img", {"a": 1}, "123")
    assert k2 != k1


def test_cache_key_without_user_is_stable(ref_dir):
    assert cc._analysis_cache_key("img", {}, "") == cc._analysis_cache_key("img", {}, "")


def test_transform_photo_misses_cache_after_reference_change(ref_dir, tmp_path, monkeypatch):
    """재분석으로 대표 사진이 바뀌면 옛 분석 결과를 돌려주지 않는다."""
    calls = []

    def fake_call(*args, **kwargs):
        calls.append(1)
        raise RuntimeError("model called")

    monkeypatch.setattr(cc, "_call_claude", fake_call)
    cc._analysis_cache.clear()
    cc._save_reference_images("123", [0], [_src(tmp_path, "a.jpg", b"A")])
    key = cc._analysis_cache_key("aW1n", {}, "123")
    cc._analysis_cache_put(key, {"cached": True})
    assert cc.transform_photo({}, "aW1n", user_id="123") == {"cached": True}
    assert calls == []

    cc._save_reference_images("123", [0], [_src(tmp_path, "b.jpg", b"BB")])
    with pytest.raises(Exception):
        cc.transform_photo({}, "aW1n", user_id="123")
    assert calls  # 캐시를 건너뛰고 모델까지 갔다
    cc._analysis_cache.clear()
