"""MediaPipe 모델 인스턴스 풀 회귀 테스트.

요청마다 모델을 만들고 닫던 것을, 다 쓴 인스턴스를 풀에 돌려 두었다가
재사용하도록 바꿨다 (close() 하나에 0.25~0.35초). 지켜야 할 것:
  1) 한 인스턴스를 두 요청이 동시에 쓰지 않는다 (MediaPipe 태스크는 스레드 안전하지 않다)
  2) 예외를 낸 인스턴스는 풀에 돌려주지 않는다 (에러 상태로 남아 다음 호출에서 멈춘다)
  3) 유휴 인스턴스 수에 상한이 있다
  4) 재사용해도 결과가 같다 (실제 모델이 로컬에 있을 때만)
"""

import os
import threading
import time

import cv2
import numpy as np
import pytest

import image_processor as ip


class _FakeTask:
    """close 호출 여부만 기록하는 가짜 태스크."""

    def __init__(self, fail: bool = False):
        self.closed = False
        self.fail = fail
        self.calls = 0

    def close(self):
        self.closed = True

    def detect(self, image):
        self.calls += 1
        if self.fail:
            raise RuntimeError("graph error")
        return type("R", (), {"face_landmarks": [], "pose_landmarks": []})()

    def segment(self, image):
        self.calls += 1
        raise RuntimeError("graph error")


@pytest.fixture(autouse=True)
def clean_pool():
    ip._mp_pool_clear()
    yield
    ip._mp_pool_clear()


def test_닫으면_풀로_돌아가고_다음_요청이_재사용한다():
    fake = _FakeTask()
    ip._mp_pool_release("face", fake)

    with ip.MediaPipeCache() as cache:
        assert cache._get_face_landmarker() is fake
        assert ip._mp_pool["face"] == []      # 빌려 간 동안은 풀에 없다
    assert fake.closed is False               # 닫지 않고
    assert ip._mp_pool["face"] == [fake]      # 풀에 돌려준다

    with ip.MediaPipeCache() as cache:
        assert cache._get_face_landmarker() is fake


def test_한_인스턴스를_두_요청이_동시에_빌리지_않는다(monkeypatch):
    fake = _FakeTask()
    ip._mp_pool_release("pose", fake)
    monkeypatch.setattr(ip, "pose_model_path", lambda: None)  # 새로 만들 수 없게

    a, b = ip.MediaPipeCache(), ip.MediaPipeCache()
    try:
        assert a._get_pose_landmarker() is fake
        assert b._get_pose_landmarker() is None  # 풀이 비어 있다 — 공유하지 않는다
    finally:
        a.close()
        b.close()


def test_여러_스레드가_동시에_빌려도_겹치지_않는다():
    fakes = [_FakeTask() for _ in range(3)]
    for f in fakes:
        ip._mp_pool_release("face", f)

    in_use: set[int] = set()
    guard = threading.Lock()
    errors: list[str] = []

    def worker():
        for _ in range(200):
            inst = ip._mp_pool_acquire("face")
            if inst is None:
                continue
            with guard:
                if id(inst) in in_use:
                    errors.append("같은 인스턴스를 두 스레드가 동시에 빌렸다")
                in_use.add(id(inst))
            time.sleep(0)
            with guard:
                in_use.discard(id(inst))
            ip._mp_pool_release("face", inst)

    threads = [threading.Thread(target=worker) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert errors == []
    assert sorted(map(id, ip._mp_pool["face"])) == sorted(map(id, fakes))


def test_유휴_인스턴스가_상한을_넘으면_닫는다():
    fakes = [_FakeTask() for _ in range(ip._MP_POOL_MAX_IDLE + 2)]
    for f in fakes:
        ip._mp_pool_release("person", f)
    assert len(ip._mp_pool["person"]) == ip._MP_POOL_MAX_IDLE
    assert [f.closed for f in fakes] == [False] * ip._MP_POOL_MAX_IDLE + [True, True]


@pytest.mark.skipif(not ip._MP_AVAILABLE, reason="mediapipe 없음")
def test_감지에_실패한_랜드마커는_풀에_돌려주지_않는다():
    bad = _FakeTask(fail=True)
    ip._mp_pool_release("face", bad)
    with ip.MediaPipeCache() as cache:
        with pytest.raises(RuntimeError):
            cache.get_face_landmarks(np.zeros((64, 64, 3), np.uint8))
    assert bad.closed is True
    assert bad not in ip._mp_pool["face"]


@pytest.mark.skipif(not ip._MP_AVAILABLE, reason="mediapipe 없음")
def test_분할에_실패한_분할기는_풀에_돌려주지_않는다():
    bad = _FakeTask()
    ip._mp_pool_release("person", bad)
    with ip.MediaPipeCache() as cache:
        assert cache.get_person_mask(np.zeros((64, 64, 3), np.uint8)) is None
    assert bad.closed is True
    assert bad not in ip._mp_pool["person"]


def _local_model(kind: str) -> bool:
    """네트워크 없이 쓸 수 있는 모델 파일이 있는지 (내려받기를 유발하지 않는다)."""
    filename, _ = ip._MODEL_SPECS[kind]
    return any(os.path.exists(p) for p in ip._candidate_model_paths(filename))


def _textured(h=480, w=640) -> np.ndarray:
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    base = 120 + 50 * np.sin(xx / 60) + 30 * np.cos(yy / 45)
    fine = cv2.GaussianBlur(np.random.default_rng(7).normal(0, 40, (h, w)).astype(np.float32),
                            (0, 0), 1.2)
    base = base + fine
    return np.clip(np.stack([base, base * 0.95, base * 0.9], -1), 0, 255).astype(np.uint8)


@pytest.mark.skipif(not (ip._MP_AVAILABLE and _local_model("person")),
                    reason="인물 분할 모델이 로컬에 없음")
def test_재사용한_분할기도_같은_결과를_낸다():
    a = _textured()
    b = np.ascontiguousarray(_textured()[::-1, ::-1])
    with ip.MediaPipeCache() as cache:
        first = cache.get_person_mask(a)
        seg = cache._person_segmenter
    with ip.MediaPipeCache() as cache:
        cache.get_person_mask(b)          # 다른 이미지를 한 번 거친 뒤
    with ip.MediaPipeCache() as cache:
        again = cache.get_person_mask(a)
        assert cache._person_segmenter is seg   # 같은 인스턴스를 재사용했고
    assert np.array_equal(first, again)          # 결과도 같다
