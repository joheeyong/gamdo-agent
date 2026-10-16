"""체형 보정 세기를 측정으로 정하고, 보정 결과를 스스로 검사한다.

모델은 사진을 보고 체형 보정값(leg_stretch / waist_slim / shoulder_width)을
0.2~0.35 같은 눈대중으로 제안한다. 색 보정이 눈대중 대신 픽셀 측정으로
바뀐 것처럼, 체형도 관절 좌표와 인물 실루엣으로 실제 비율을 재서

  1. 세기: 모델 제안은 '이 부위를 손봐도 된다'는 허락(상한)으로만 쓰고,
     실제 세기는 목표 비율까지 모자란 만큼만 준다. 이미 비율이 좋은 사람은
     건드리지 않고, 같은 사진에는 늘 같은 값이 나온다.
  2. 자체 검사: 보정했을 때 인물 옆 배경 직선(문틀·벽 모서리)이 휘거나
     함께 찍힌 다른 사람이 변형되면 세기를 낮추거나 그 보정을 끈다.

자동 보정(분석 직후)에만 쓴다. 사용자가 슬라이더로 직접 고른 값은 그대로 둔다.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any

import cv2
import numpy as np
from PIL import Image

import image_processor as ip
from param_engine import _RESHAPE_MAX

log = logging.getLogger("gamdo-agent.body_check")

# ── 세기 상한 ──

# 모델 제안과 측정 결과 모두 _RESHAPE_MAX(param_engine) 안에 둔다.

# 측정할 수 없을 때(실루엣 없음·머리나 발이 잘림·팔이 허리를 가림) 모델 제안을
# 이 값까지만 받는다. 근거 없이 큰 변형을 하지 않는다.
_UNMEASURED_CAP = 0.2

# ── 목표 비율 ──
#
# 아래 값들은 '어색해 보이는 비율을 자연스러운 쪽으로 당긴다'는 기준이다.
# 성별·체형을 추정하지 않으므로 이상형이 아니라 하한/상한으로만 쓴다.

# 다리 비율 = (발끝 - 힙 관절) / (발끝 - 정수리). 서 있는 성인의 힙 관절 높이는
# 키의 0.50~0.53 정도다. 카메라를 높이 들고 내려 찍으면 0.45 안팎으로 줄어
# 다리가 짧아 보인다. 목표는 그 범위보다 살짝 위인 0.54다 — 자연 범위 안으로만
# 당기면(0.52) 실제 사진 대부분이 0이 되어 효과가 없었다(2026-10). 그래서 이미
# 평범한 비율(0.52~0.53)도 조금 늘어날 수 있다. 정수리 위 머리카락 여백은
# 키에 넣지 않는다.
_TARGET_LEG_RATIO = 0.54

# 허리 비율 = 허리 실루엣 폭 / 힙 높이 실루엣 폭. 이보다 넓을 때만, 이 비율까지
# 모자란 만큼만 줄인다.
_TARGET_WAIST_RATIO = 0.82

# 어깨 비율 = 어깨 관절 폭 / 힙 높이 실루엣 폭. 넓히기는 이보다 좁을 때만,
# 좁히기는 이보다 넓을 때만 허락한다.
_SHOULDER_WIDEN_BELOW = 0.85
_SHOULDER_NARROW_ABOVE = 1.15

# 몸통·다리가 수직에서 이만큼 넘게 기울면(눕거나 크게 기댄 자세) 비율을 믿지 않는다.
_UPRIGHT_MAX_DEG = 25.0

# ── 자체 검사 ──

# 배경 직선이 이 이상 휘면 실패로 본다 (짧은 변 대비, 최소 px).
_BEND_LIMIT_FRAC = 0.0015
_BEND_LIMIT_MIN_PX = 1.5
# 다른 사람의 관절이 이만큼 넘게 움직이면 실패로 본다 (px).
_OTHER_PERSON_MOVE_PX = 1.0
# 직선 검사에 쓰는 최소 선분 길이 (짧은 변 대비).
_MIN_LINE_FRAC = 0.05


@dataclass
class BodyMeasure:
    """한 사람의 체형 비율. 잴 수 없는 항목은 None."""

    leg_ratio: float | None = None
    waist_ratio: float | None = None
    shoulder_ratio: float | None = None
    upright: bool = False


@dataclass
class BodyCalibration:
    leg_stretch: float
    shoulder_width: float
    waist_slim: float
    measure: BodyMeasure | None = None
    notes: list[str] = field(default_factory=list)


# ── 순수 계산 (테스트 대상) ──


def leg_need(top_y: float, hip_y: float, ankle_y: float, foot_y: float,
             target: float = _TARGET_LEG_RATIO) -> float | None:
    """다리 비율을 target까지 채우는 leg_stretch 값. 이미 충분하면 0.

    다리 늘리기는 허벅지 중간~발목 구간에 extra = 값 × 게인 × (발목 - 힙) px을
    더한다. 늘어난 뒤 비율 (F + e) / (H + e)가 target이 되는 e를 풀어 값으로 바꾼다.
    """
    height = foot_y - top_y
    floor_to_hip = foot_y - hip_y
    leg = ankle_y - hip_y
    if height <= 0 or floor_to_hip <= 0 or leg <= 0 or not 0.0 < target < 1.0:
        return None
    if floor_to_hip / height >= target:
        return 0.0
    extra = (target * height - floor_to_hip) / (1.0 - target)
    return float(np.clip(extra / (ip._LEG_STRETCH_GAIN * leg), 0.0, _RESHAPE_MAX))


def waist_need(ratio: float, target: float = _TARGET_WAIST_RATIO) -> float:
    """허리 폭 비율을 target까지 줄이는 waist_slim 값. 이미 좁으면 0.

    waist_slim은 허리 반폭을 값 × 게인만큼(비율) 안쪽으로 민다.
    """
    if ratio <= target:
        return 0.0
    frac = 1.0 - target / ratio
    return float(np.clip(frac / ip._WAIST_SLIM_GAIN, 0.0, _RESHAPE_MAX))


def shoulder_allowed(requested: float, ratio: float) -> float:
    """모델이 제안한 어깨 값 중 측정 비율이 허락하는 만큼. 방향이 틀리면 0."""
    if requested > 0.0:
        if ratio >= _SHOULDER_WIDEN_BELOW:
            return 0.0
        frac = _SHOULDER_WIDEN_BELOW / ratio - 1.0
        return float(min(requested, frac / ip._SHOULDER_GAIN, _RESHAPE_MAX))
    if requested < 0.0:
        if ratio <= _SHOULDER_NARROW_ABOVE:
            return 0.0
        frac = 1.0 - _SHOULDER_NARROW_ABOVE / ratio
        return -float(min(-requested, frac / ip._SHOULDER_GAIN, _RESHAPE_MAX))
    return 0.0


def line_bend(points: np.ndarray, normal: np.ndarray, dx: np.ndarray, dy: np.ndarray,
              origin: tuple[int, int]) -> float:
    """선분 위 표본점들의 변위가 직선을 얼마나 휘게 하는지 (px).

    선 전체가 같이 움직이거나(평행 이동) 고르게 기울면(회전) 직선은 그대로다.
    그래서 선에 수직인 변위 성분을 선 위치 t에 대해 1차식으로 맞춘 뒤 남는
    잔차의 최댓값을 휨으로 본다.
    """
    x0, y0 = origin
    hh, ww = dx.shape
    xs = np.clip(np.round(points[:, 0] - x0).astype(int), 0, ww - 1)
    ys = np.clip(np.round(points[:, 1] - y0).astype(int), 0, hh - 1)
    perp = dx[ys, xs] * normal[0] + dy[ys, xs] * normal[1]
    t = np.linspace(0.0, 1.0, len(points))
    if len(points) < 3:
        return 0.0
    coef = np.polyfit(t, perp, 1)
    return float(np.max(np.abs(perp - np.polyval(coef, t))))


# ── 측정 ──


def _mask_edge(person: np.ndarray, center: np.ndarray, direction: np.ndarray,
               max_dist: float) -> float | None:
    """center에서 direction으로 실루엣이 처음 끝나는 거리. 못 찾으면 None."""
    h, w = person.shape[:2]
    ss = np.arange(1.0, max(2.0, max_dist), 1.0)
    xs = np.round(center[0] + direction[0] * ss).astype(int)
    ys = np.round(center[1] + direction[1] * ss).astype(int)
    inside = (xs >= 0) & (xs < w) & (ys >= 0) & (ys < h)
    if not inside.any():
        return None
    vals = np.zeros(len(ss), np.float32)
    vals[inside] = person[ys[inside], xs[inside]]
    vals[~inside] = 0.0
    off = np.flatnonzero(vals < 0.5)
    if off.size == 0 or not inside[off[0]]:
        return None  # 실루엣이 프레임 밖까지 이어짐
    return float(ss[off[0]])


def _head_top(person: np.ndarray, nose: tuple[float, float], half_width: float) -> float | None:
    """코 위쪽에서 실루엣의 가장 위 행 (머리카락 포함). 머리가 프레임 위에서 잘렸으면 None."""
    h, w = person.shape[:2]
    x0 = int(max(0, nose[0] - half_width))
    x1 = int(min(w, nose[0] + half_width + 1))
    y1 = int(min(h, max(1, nose[1])))
    if x1 <= x0 or y1 <= 0:
        return None
    rows = np.flatnonzero((person[:y1, x0:x1] > 0.5).any(axis=1))
    if rows.size == 0:
        return None
    top = int(rows[0])
    return None if top <= 0 else float(top)


def _angle_from_vertical(v: np.ndarray) -> float:
    return math.degrees(math.atan2(abs(float(v[0])), max(float(v[1]), 1e-6)))


def measure_body(pt, vis, person: np.ndarray | None, h: int, w: int) -> BodyMeasure:
    """관절 좌표와 실루엣으로 다리·허리·어깨 비율을 잰다."""
    m = BodyMeasure()
    if min(vis(11), vis(12), vis(23), vis(24)) < 0.5:
        return m
    P = lambda i: np.array(pt(i), dtype=np.float64)  # noqa: E731
    ls, rs, lh, rh = P(11), P(12), P(23), P(24)
    ms, mh = (ls + rs) / 2.0, (lh + rh) / 2.0
    torso = mh - ms
    t_len = float(np.linalg.norm(torso))
    sh_half = float(np.linalg.norm(ls - rs)) / 2.0
    if t_len < 4.0 or sh_half < 2.0:
        return m

    span = ip._leg_span(pt, vis)
    m.upright = _angle_from_vertical(torso) <= _UPRIGHT_MAX_DEG
    if span is not None:
        hip_y, ankle_y, foot_y = span
        leg_vec = np.array([0.0, ankle_y - hip_y])
        ankles = [P(i) for i in (27, 28) if vis(i) >= 0.5]
        if ankles:
            leg_vec = np.mean(ankles, axis=0) - mh
        # 서 있는 자세: 다리가 수직에 가깝고 몸통보다 길게 보인다 (앉으면 짧아진다)
        m.upright = (m.upright and _angle_from_vertical(leg_vec) <= _UPRIGHT_MAX_DEG
                     and (ankle_y - hip_y) >= 1.0 * t_len)

    if person is None:
        return m

    # 다리 비율 — 정수리(실루엣)와 발끝(관절)이 모두 프레임 안에 있어야 한다
    if span is not None and m.upright and vis(0) >= 0.5:
        hip_y, ankle_y, foot_y = span
        top = _head_top(person, pt(0), max(0.6 * sh_half, 0.02 * w))
        if top is not None and foot_y < h - 1:
            height = foot_y - top
            if height > 0:
                m.leg_ratio = (foot_y - hip_y) / height

    # 허리·힙 폭 — 몸통 중심선에서 좌우로 실루엣 끝까지
    down = torso / t_len
    lateral = np.array([down[1], -down[0]])
    reach = 2.5 * max(sh_half, float(np.linalg.norm(lh - rh)) / 2.0)

    def width_at(tl: float) -> tuple[float, float] | None:
        c = ms + torso * tl
        left = _mask_edge(person, c, -lateral, reach)
        right = _mask_edge(person, c, lateral, reach)
        if left is None or right is None:
            return None
        return left, right

    waist = width_at(0.62)
    hips = width_at(1.0)
    if hips is not None:
        hip_half = (hips[0] + hips[1]) / 2.0
        if hip_half > 1.0:
            m.shoulder_ratio = sh_half / hip_half
            if waist is not None and not _arm_covers_waist(P, vis, ms, torso, lateral, waist, sh_half):
                m.waist_ratio = (waist[0] + waist[1]) / (hips[0] + hips[1])
    return m


def _arm_covers_waist(P, vis, ms, torso, lateral, waist, sh_half) -> bool:
    """팔꿈치·손목이 허리 옆에 붙어 있으면 실루엣 폭에 팔이 섞인다 → 허리를 잴 수 없다."""
    t_len2 = float(np.dot(torso, torso))
    for i in (13, 14, 15, 16):
        if vis(i) < 0.5:
            continue
        rel = P(i) - ms
        tl = float(np.dot(rel, torso)) / t_len2
        if abs(tl - 0.62) > 0.25:
            continue
        side = float(np.dot(rel - torso * tl, lateral))
        edge = waist[1] if side > 0 else waist[0]
        if abs(side) <= edge + 0.15 * sh_half:
            return True
    return False


# ── 자체 검사 ──


def background_bend(arr: np.ndarray, person: np.ndarray, roi: tuple, fld) -> float:
    """보정 변위장이 인물 옆 배경 직선을 휘게 하는 최대량 (px)."""
    x0, y0, x1, y1 = roi
    dx, dy = fld
    h, w = arr.shape[:2]
    gray = cv2.cvtColor(arr[y0:y1, x0:x1], cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 60, 160)
    min_len = max(20, int(_MIN_LINE_FRAC * min(h, w)))
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=40,
                            minLineLength=min_len, maxLineGap=4)
    if lines is None:
        return 0.0
    # 인물 위의 선(옷 무늬·팔)은 의도한 변형이라 제외한다. 실루엣을 조금 키워 경계선도 뺀다.
    body = cv2.dilate((person[y0:y1, x0:x1] > 0.5).astype(np.uint8), np.ones((7, 7), np.uint8))
    worst = 0.0
    for x_a, y_a, x_b, y_b in np.asarray(lines).reshape(-1, 4):
        n = 24
        t = np.linspace(0.0, 1.0, n)
        pts_local = np.stack([x_a + (x_b - x_a) * t, y_a + (y_b - y_a) * t], axis=1)
        xi = np.clip(np.round(pts_local[:, 0]).astype(int), 0, body.shape[1] - 1)
        yi = np.clip(np.round(pts_local[:, 1]).astype(int), 0, body.shape[0] - 1)
        if body[yi, xi].mean() > 0.2:
            continue
        direction = np.array([x_b - x_a, y_b - y_a], dtype=np.float64)
        length = float(np.linalg.norm(direction))
        if length < 1.0:
            continue
        normal = np.array([-direction[1], direction[0]]) / length
        pts = pts_local + np.array([x0, y0], dtype=np.float64)
        worst = max(worst, line_bend(pts, normal, dx, dy, (x0, y0)))
    return worst


def _others_moved(others_pts: list[tuple[float, float]], roi: tuple, fld) -> float:
    """다른 사람 관절 위치에서의 최대 변위 (px)."""
    if not others_pts:
        return 0.0
    x0, y0, x1, y1 = roi
    dx, dy = fld
    worst = 0.0
    for x, y in others_pts:
        if x0 <= x < x1 and y0 <= y < y1:
            xi, yi = int(x - x0), int(y - y0)
            worst = max(worst, float(math.hypot(dx[yi, xi], dy[yi, xi])))
    return worst


def _visible_points(pose, w: int, h: int, min_vis: float = 0.5) -> list[tuple[float, float]]:
    pts = []
    for lm in pose:
        v = getattr(lm, "visibility", None)
        if v is None or float(v) >= min_vis:
            pts.append((lm.x * w, lm.y * h))
    return pts


# ── 진입점 ──


def calibrate_body_reshape(
    img: Image.Image,
    leg_stretch: float,
    shoulder_width: float,
    waist_slim: float,
    cache: Any = None,
) -> BodyCalibration:
    """모델이 제안한 체형 값을 측정과 자체 검사로 다듬는다.

    반환 값은 apply_body_reshape에 그대로 넘기면 된다. 사람을 찾지 못하면 체형
    보정을 모두 끈다 (보정할 대상이 없다).
    """
    leg = float(np.clip(leg_stretch, 0.0, _RESHAPE_MAX))
    waist = float(np.clip(waist_slim, 0.0, _RESHAPE_MAX))
    shoulder = float(np.clip(shoulder_width, -_RESHAPE_MAX, _RESHAPE_MAX))
    out = BodyCalibration(leg, shoulder, waist)
    if leg < 0.01 and waist < 0.01 and abs(shoulder) < 0.01:
        return out

    arr = np.array(img.convert("RGB"), dtype=np.uint8)
    h, w = arr.shape[:2]
    results = cache.get_pose_landmarks(arr) if cache is not None else None
    main, others = ip._select_main_pose(results) if results is not None else (None, [])
    if main is None:
        out.leg_stretch = out.waist_slim = out.shoulder_width = 0.0
        out.notes.append("사람을 찾지 못해 체형 보정을 끔")
        return out

    pt, vis = ip._pose_accessors(main, w, h)
    person = ip._cached_person_mask(cache, arr)
    m = measure_body(pt, vis, person, h, w)
    out.measure = m

    # 1. 세기: 모델 제안은 상한, 실제 세기는 측정으로
    if leg >= 0.01:
        span = ip._leg_span(pt, vis)
        if span is None:
            out.leg_stretch = 0.0
            out.notes.append("발목이 보이지 않아 다리 보정을 끔")
        elif m.leg_ratio is not None:
            hip_y, ankle_y, foot_y = span
            top = foot_y - (foot_y - hip_y) / m.leg_ratio
            need = leg_need(top, hip_y, ankle_y, foot_y) or 0.0
            out.leg_stretch = min(leg, need)
            out.notes.append(f"다리 비율 {m.leg_ratio:.3f} → 다리 {leg:.2f}→{out.leg_stretch:.2f}")
        else:
            out.leg_stretch = min(leg, _UNMEASURED_CAP)
            out.notes.append("다리 비율을 잴 수 없어 다리 보정을 낮춤")

    if waist >= 0.01:
        if m.waist_ratio is not None:
            out.waist_slim = min(waist, waist_need(m.waist_ratio))
            out.notes.append(f"허리 비율 {m.waist_ratio:.3f} → 허리 {waist:.2f}→{out.waist_slim:.2f}")
        else:
            out.waist_slim = min(waist, _UNMEASURED_CAP)
            out.notes.append("허리를 잴 수 없어 허리 보정을 낮춤")

    if abs(shoulder) >= 0.01:
        if m.shoulder_ratio is not None:
            out.shoulder_width = shoulder_allowed(shoulder, m.shoulder_ratio)
            out.notes.append(
                f"어깨 비율 {m.shoulder_ratio:.3f} → 어깨 {shoulder:.2f}→{out.shoulder_width:.2f}")
        else:
            out.shoulder_width = float(np.clip(shoulder, -_UNMEASURED_CAP, _UNMEASURED_CAP))
            out.notes.append("어깨를 잴 수 없어 어깨 보정을 낮춤")

    others_pts = [p for o in others for p in _visible_points(o, w, h)]

    # 2-a. 자체 검사: 다리 늘리기 구간에 다른 사람이 있으면 그 사람까지 늘어난다
    if out.leg_stretch >= 0.01:
        span = ip._leg_span(pt, vis)
        mapping = ip._leg_row_map(h, *span, out.leg_stretch) if span else None
        if mapping is not None:
            _, first_row = mapping
            if any(y >= first_row for _, y in others_pts):
                out.leg_stretch = 0.0
                out.notes.append("늘어나는 구간에 다른 사람이 있어 다리 보정을 끔")

    # 2-b. 자체 검사: 허리·어깨 변형이 배경 직선을 휘게 하거나 다른 사람을 밀면 줄인다
    if out.waist_slim >= 0.01 or abs(out.shoulder_width) >= 0.01:
        limit = max(_BEND_LIMIT_MIN_PX, _BEND_LIMIT_FRAC * min(h, w))
        for attempt in range(2):
            _, roi, fld = ip._torso_reshape(arr, pt, vis, person, out.shoulder_width, out.waist_slim)
            if fld is None:
                break
            bend = background_bend(arr, person, roi, fld) if person is not None else 0.0
            moved = _others_moved(others_pts, roi, fld)
            if bend <= limit and moved <= _OTHER_PERSON_MOVE_PX:
                if attempt:
                    out.notes.append(f"배경 휨 {bend:.1f}px로 기준 안 — 줄인 값 사용")
                break
            if attempt == 0 and moved <= _OTHER_PERSON_MOVE_PX:
                scale = 0.9 * limit / bend
                out.notes.append(f"배경 직선이 {bend:.1f}px 휨 (기준 {limit:.1f}px) — 허리·어깨를 {scale:.2f}배로")
                out.waist_slim *= scale
                out.shoulder_width *= scale
                continue
            reason = "다른 사람이 함께 변형됨" if moved > _OTHER_PERSON_MOVE_PX else f"배경 휨 {bend:.1f}px"
            out.notes.append(f"{reason} — 허리·어깨 보정을 끔")
            out.waist_slim = 0.0
            out.shoulder_width = 0.0
            break

    for key in ("leg_stretch", "waist_slim"):
        if getattr(out, key) < 0.01:
            setattr(out, key, 0.0)
    if abs(out.shoulder_width) < 0.01:
        out.shoulder_width = 0.0
    return out
