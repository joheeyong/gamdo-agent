"""비동기 분석 작업(Job) 저장소 + 실행기.

analyze-and-transform은 30~70초 걸린다. 긴 HTTP 연결 하나에 묶어 두면 앱이
백그라운드로 가서 소켓이 끊길 때 결과를 잃는다. 작업을 여기에 맡기고 앱은
job_id로 폴링한다 (계약: scratchpad/jobs_contract.md).

- 인메모리·단일 프로세스. 서버 재시작(reload 포함) 시 작업은 사라진다.
- 같은 요청(dedupe key)이 진행 중이거나 완료 후 보관 중이면 그 작업을 돌려준다.
- 완료/실패 후 TTL(30분) 보관, 전체 작업 수 상한을 넘으면 오래된 완료 작업부터 제거.
- 전용 스레드풀에서 돈다. 워커의 예외는 작업 상태로만 남고 풀을 죽이지 않는다.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import secrets
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Callable

log = logging.getLogger("gamdo-agent.jobs")

QUEUED = "queued"
RUNNING = "running"
DONE = "done"
ERROR = "error"

# 사용자에게 보여 줄 기본 실패 메시지. 예외 문자열에는 CLI 출력·경로 같은
# 내부 정보가 섞일 수 있어 그대로 내보내지 않는다 (자세한 내용은 서버 로그에).
DEFAULT_ERROR_MESSAGE = "사진 분석에 실패했습니다. 잠시 후 다시 시도해 주세요."

JOB_TTL_SECONDS = 30 * 60
MAX_TOTAL_JOBS = 200


def _now() -> float:
    """시계. 테스트가 TTL을 확인하려고 바꿔 끼운다."""
    return time.monotonic()


def _env_int(name: str, default: int, minimum: int) -> int:
    try:
        return max(minimum, int(os.getenv(name, str(default))))
    except ValueError:
        return default


def dedupe_key(*parts: Any) -> str:
    """요청 필드들로 만든 sha256. 필드 경계가 섞이지 않게 JSON 배열로 직렬화한다."""
    raw = json.dumps(parts, sort_keys=True, ensure_ascii=False, separators=(",", ":"), default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


class JobQueueFull(Exception):
    """대기열이 가득 찼다 → 503 busy."""


class JobFailed(Exception):
    """작업 함수가 사용자에게 보여도 되는 메시지로 실패를 알릴 때 쓴다."""

    def __init__(self, message: str = DEFAULT_ERROR_MESSAGE):
        super().__init__(message)
        self.message = message


@dataclass
class Job:
    job_id: str
    key: str
    owner: str | None
    created_at: float
    status: str = QUEUED
    stage: str = QUEUED
    started_at: float | None = None
    finished_at: float | None = None
    result: dict | None = None
    error: str | None = None
    # 실행 전까지만 들고 있는다 (요청 본문의 사진이 크다)
    fn: Callable[[Callable[[str], None]], dict] | None = field(default=None, repr=False)


@dataclass(frozen=True)
class JobSnapshot:
    """락 밖으로 내보내는 읽기 전용 사본."""
    job_id: str
    owner: str | None
    status: str
    stage: str
    elapsed_sec: float
    result: dict | None
    error: str | None


class JobStore:
    def __init__(
        self,
        max_workers: int | None = None,
        max_queued: int | None = None,
        ttl_seconds: float = JOB_TTL_SECONDS,
        max_total: int = MAX_TOTAL_JOBS,
    ):
        self.max_workers = max_workers or _env_int("GAMDO_MAX_JOBS", 4, 1)
        self.max_queued = (
            max_queued if max_queued is not None else _env_int("GAMDO_MAX_QUEUED_JOBS", 20, 0)
        )
        self.ttl_seconds = ttl_seconds
        self.max_total = max_total
        self._jobs: dict[str, Job] = {}
        self._by_key: dict[str, str] = {}
        self._lock = threading.Lock()
        self._executor: ThreadPoolExecutor | None = None

    # ── 내부 (락을 잡은 상태에서만 호출) ──

    def _remove(self, job: Job) -> None:
        self._jobs.pop(job.job_id, None)
        if self._by_key.get(job.key) == job.job_id:
            del self._by_key[job.key]

    def _cleanup(self, now: float) -> None:
        expired = [
            j for j in self._jobs.values()
            if j.finished_at is not None and now - j.finished_at >= self.ttl_seconds
        ]
        for j in expired:
            self._remove(j)
        # 상한 초과: 오래 전에 끝난 작업부터 제거. 진행 중 작업은 건드리지 않는다
        # (진행 중 작업 수는 워커 수 + 대기열 상한으로 이미 묶여 있다).
        overflow = len(self._jobs) - self.max_total
        if overflow > 0:
            finished = sorted(
                (j for j in self._jobs.values() if j.finished_at is not None),
                key=lambda j: j.finished_at,
            )
            for j in finished[:overflow]:
                self._remove(j)

    def _snapshot(self, job: Job, now: float) -> JobSnapshot:
        end = job.finished_at if job.finished_at is not None else now
        return JobSnapshot(
            job_id=job.job_id,
            owner=job.owner,
            status=job.status,
            stage=job.stage,
            elapsed_sec=round(max(0.0, end - job.created_at), 1),
            result=job.result,
            error=job.error,
        )

    def _get_executor(self) -> ThreadPoolExecutor:
        if self._executor is None:
            self._executor = ThreadPoolExecutor(
                max_workers=self.max_workers, thread_name_prefix="gamdo-job"
            )
        return self._executor

    # ── 공개 API ──

    def submit(
        self,
        key: str,
        owner: str | None,
        fn: Callable[[Callable[[str], None]], dict],
    ) -> JobSnapshot:
        """작업을 등록하고 스냅샷을 돌려준다.

        fn(on_stage)는 결과 dict를 돌려주거나 예외를 올린다. 같은 key의 작업이
        대기·진행 중이거나 성공 후 보관 중이면 새로 만들지 않고 그것을 돌려준다.
        실패한 작업은 합치지 않는다 — 재시도가 30분 동안 같은 실패를 받지 않도록.
        대기열이 가득 차면 JobQueueFull.
        """
        with self._lock:
            now = _now()
            self._cleanup(now)

            existing_id = self._by_key.get(key)
            existing = self._jobs.get(existing_id) if existing_id else None
            if existing is not None and existing.status != ERROR:
                return self._snapshot(existing, now)

            queued = sum(1 for j in self._jobs.values() if j.status == QUEUED)
            if queued >= self.max_queued:
                raise JobQueueFull()

            job = Job(
                job_id=secrets.token_hex(16),  # 128비트 — 추측 불가
                key=key,
                owner=owner,
                created_at=now,
                fn=fn,
            )
            self._jobs[job.job_id] = job
            self._by_key[key] = job.job_id
            snap = self._snapshot(job, now)
            executor = self._get_executor()

        try:
            executor.submit(self._run, job.job_id)
        except RuntimeError:
            # 인터프리터 종료 중 등으로 풀이 닫혔다
            log.error("jobs: 실행기에 작업을 넣지 못함 (%s)", job.job_id[:8])
            self._finish(job.job_id, error=DEFAULT_ERROR_MESSAGE)
            with self._lock:
                snap = self._snapshot(job, _now())
        return snap

    def get(self, job_id: str) -> JobSnapshot | None:
        with self._lock:
            now = _now()
            self._cleanup(now)
            job = self._jobs.get(job_id)
            return self._snapshot(job, now) if job else None

    def set_stage(self, job_id: str, stage: str) -> None:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is not None and job.status == RUNNING:
                job.stage = stage

    def _finish(self, job_id: str, result: dict | None = None, error: str | None = None) -> None:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return
            job.fn = None
            job.finished_at = _now()
            if error is None:
                job.status = job.stage = DONE
                job.result = result
            else:
                job.status = job.stage = ERROR
                job.error = error

    def _run(self, job_id: str) -> None:
        """워커 스레드 본체. 어떤 예외도 밖으로 내보내지 않는다."""
        try:
            with self._lock:
                job = self._jobs.get(job_id)
                if job is None or job.fn is None:
                    return
                fn = job.fn
                job.fn = None
                job.status = RUNNING
                job.started_at = _now()

            def on_stage(stage: str) -> None:
                try:
                    self.set_stage(job_id, stage)
                except Exception:
                    pass

            result = fn(on_stage)
            self._finish(job_id, result=result)
        except JobFailed as e:
            self._finish(job_id, error=e.message or DEFAULT_ERROR_MESSAGE)
        except BaseException:  # noqa: BLE001 — 워커가 죽으면 작업이 영원히 running으로 남는다
            log.exception("jobs: 작업 실패 (%s)", job_id[:8])
            try:
                self._finish(job_id, error=DEFAULT_ERROR_MESSAGE)
            except Exception:
                pass

    def shutdown(self, wait: bool = False) -> None:
        with self._lock:
            executor, self._executor = self._executor, None
        if executor is not None:
            executor.shutdown(wait=wait, cancel_futures=True)
