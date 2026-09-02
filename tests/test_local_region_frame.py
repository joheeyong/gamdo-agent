"""국소 보정 좌표가 기하 보정과 무관하게 짚은 곳에 걸리는지 검증한다.

모델은 원본 사진을 보고 0~1 좌표를 짚는다. 예전에는 서버가 기하 보정(수평·원근·
크롭)을 먼저 돌리고 그 뒤에 원본 좌표로 마스크를 만들어서, 회전·크롭으로 프레임이
바뀐 만큼 좌표가 밀렸다 — 날아간 창문 대신 평평한 벽에 어두운 사각형이 찍혔다.
"""

import base64
import io

import numpy as np
import pytest
from PIL import Image

import server
from models import AnalyzeAndTransformRequest

W, H = 1200, 900
WINDOW = {"x": 0.70, "y": 0.15, "width": 0.18, "height": 0.30}


def _photo() -> tuple[str, Image.Image]:
    """어두운 실내에 한 군데만 하얗게 날아간 창문."""
    arr = np.full((H, W, 3), 90, np.uint8)
    l, t = int(WINDOW["x"] * W), int(WINDOW["y"] * H)
    r = int((WINDOW["x"] + WINDOW["width"]) * W)
    b = int((WINDOW["y"] + WINDOW["height"]) * H)
    arr[t:b, l:r] = 252
    img = Image.fromarray(arr)
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=95)
    return base64.b64encode(buf.getvalue()).decode(), img


def _analysis(with_geometry: bool) -> dict:
    auto = {"straighten": 3.0, "keystone": 0.15} if with_geometry else {}
    return {
        "subjectType": "카페/일상",
        "colorAnalysis": {"colorHarmony": "x", "paletteDescription": "x"},
        "compositionAnalysis": {"primaryTechnique": "x", "balanceScore": 0.5,
                                "strengths": [], "improvements": []},
        "toneReport": {"overallMood": "x", "styleCategory": "x", "narrative": "x"},
        "shootingTips": [], "editingTips": [], "overallScore": 50,
        "autoEdits": auto,
        "regionParams": {
            "local_0": {"area": dict(WINDOW), "shape": "rect", "feather": 0.3,
                        "reason": "창문이 날아감",
                        "brightness": -0.5, "highlights": -0.6},
        },
        "hslAdjust": None, "reshapeParams": None,
    }


@pytest.mark.parametrize("with_geometry", [False, True])
def test_local_region_hits_the_window(monkeypatch, with_geometry):
    """국소 보정을 켰을 때와 껐을 때의 차이가 창문 위에 몰려 있어야 한다.

    전역 보정도 같이 걸리므로 절대 밝기로는 판단할 수 없다. 두 번 렌더해
    차이 영상을 보면 국소 보정만 분리된다 — 기하 보정은 두 렌더에 똑같이
    적용되므로 프레임도 일치한다.
    """
    data, _ = _photo()

    def render(with_local: bool) -> np.ndarray:
        analysis = _analysis(with_geometry)
        if not with_local:
            analysis["regionParams"] = None
        monkeypatch.setattr(server, "transform_photo", lambda **kw: analysis)
        resp = server.api_analyze_and_transform(
            AnalyzeAndTransformRequest(image_base64=data), None)
        assert resp.success, resp.error
        return np.asarray(
            Image.open(io.BytesIO(base64.b64decode(resp.image_base64))).convert("RGB"),
            np.float32)

    without = render(False)
    with_local = render(True)
    assert without.shape == with_local.shape

    diff = np.abs(with_local - without).mean(axis=2)
    assert diff.max() > 8, f"국소 보정이 아무 일도 하지 않았다 (최대 차이 {diff.max():.1f})"

    # 밝은 곳(창문)과 어두운 곳(실내)을 결과 프레임에서 직접 가른다
    luma = without.mean(axis=2)
    bright = luma > 200
    dark = luma < 140
    assert bright.sum() > 100 and dark.sum() > 100

    on_window = float(diff[bright].mean())
    on_wall = float(diff[dark].mean())
    assert on_window > on_wall * 3, (
        f"보정이 창문(평균 차이 {on_window:.1f})보다 벽({on_wall:.1f})에 "
        "더 걸렸다 — 좌표가 밀렸다는 뜻")
