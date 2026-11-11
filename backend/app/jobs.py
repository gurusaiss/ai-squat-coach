"""In-memory job tracking for background video processing.

A real production system would use a durable queue (Redis/Celery, etc.);
for a 4-6 hour demo-scope build an in-process dict + background task is the
smallest thing that proves the upload -> processing -> result flow end to
end, as the assignment asks for ("choose the smallest implementation that
proves the complete experience"). Documented as a known limitation in
BUILD_NOTES.md (jobs do not survive a server restart, no multi-worker
support).
"""
from __future__ import annotations

import threading
import time
import uuid
from dataclasses import dataclass, field


@dataclass
class Job:
    id: str
    status: str = "queued"  # queued | processing | done | error
    progress: int = 0
    message: str = "Queued"
    result: dict | None = None
    error: str | None = None
    created_at: float = field(default_factory=time.time)


_jobs: dict[str, Job] = {}
_lock = threading.Lock()


def create_job() -> Job:
    job = Job(id=str(uuid.uuid4()))
    with _lock:
        _jobs[job.id] = job
    return job


def get_job(job_id: str) -> Job | None:
    with _lock:
        return _jobs.get(job_id)


def update_job(job_id: str, **kwargs):
    with _lock:
        job = _jobs.get(job_id)
        if job:
            for k, v in kwargs.items():
                setattr(job, k, v)
