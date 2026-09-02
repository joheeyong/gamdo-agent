"""무거운 처리 제한(_heavy_semaphore)이 걸리는 구간을 검증한다.

메모리 피크는 픽셀 처리에서 나온다 (2560px 한 장에 ~1.1GB). Claude 호출은
수십 초 기다리지만 메모리를 쓰지 않는다. 예전에는 세마포어가 Claude 호출까지
함께 묶고 있어서, 슬롯 3개가 LLM 대기로 차면 슬라이더를 움직이는 다른
사용자의 미리보기가 그만큼 밀렸다.
"""

import base64
import io
import threading

import numpy as np
import pytest
from PIL import Image

import server
from models import AnalyzeAndTransformRequest, ApplyTransformRequest


def _b64_photo(size=(320, 240)) -> str:
    arr = np.random.default_rng(0).integers(0, 255, (size[1], size[0], 3), dtype=np.uint8)
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="JPEG", quality=80)
    return base64.b64encode(buf.getvalue()).decode()


@pytest.fixture
def one_slot(monkeypatch):
    """슬롯 1개짜리 세마포어로 바꿔 점유 여부를 관측할 수 있게 한다."""
    sem = threading.Semaphore(1)
    monkeypatch.setattr(server, "_heavy_semaphore", sem)
    return sem


def _slot_free(sem) -> bool:
    """지금 슬롯이 남아 있나 (관측만 하고 즉시 돌려준다)."""
    if sem.acquire(blocking=False):
        sem.release()
        return True
    return False


def test_semaphore_is_not_held_during_the_claude_call(one_slot, monkeypatch):
    observed = {}

    def fake_transform_photo(**kwargs):
        observed["free_during_llm"] = _slot_free(one_slot)
        return {"subjectType": "풍경"}

    real_apply = server.apply_all_transforms

    def probing_apply(*args, **kwargs):
        observed["free_during_pixels"] = _slot_free(one_slot)
        return real_apply(*args, **kwargs)

    monkeypatch.setattr(server, "transform_photo", fake_transform_photo)
    monkeypatch.setattr(server, "apply_all_transforms", probing_apply)

    resp = server.api_analyze_and_transform(
        AnalyzeAndTransformRequest(image_base64=_b64_photo()), None
    )

    assert resp.success, resp.error
    assert observed["free_during_llm"] is True, "Claude 대기 중에 슬롯이 잡혀 있다"
    assert observed["free_during_pixels"] is False, "픽셀 처리가 제한 밖에서 돈다"


def test_apply_transform_is_also_limited(one_slot, monkeypatch):
    observed = {}
    real_apply = server.apply_all_transforms

    def probing_apply(*args, **kwargs):
        observed["free_during_pixels"] = _slot_free(one_slot)
        return real_apply(*args, **kwargs)

    monkeypatch.setattr(server, "apply_all_transforms", probing_apply)

    resp = server.api_apply_transform(
        ApplyTransformRequest(image_base64=_b64_photo(), brightness=0.1), None
    )

    assert resp.success, resp.error
    assert observed["free_during_pixels"] is False, "저장 경로가 제한 밖에서 돈다"


def test_slot_is_released_after_the_request(one_slot, monkeypatch):
    monkeypatch.setattr(server, "transform_photo", lambda **kw: {"subjectType": "풍경"})
    server.api_analyze_and_transform(
        AnalyzeAndTransformRequest(image_base64=_b64_photo()), None
    )
    assert _slot_free(one_slot), "요청이 끝났는데 슬롯이 반납되지 않았다"
