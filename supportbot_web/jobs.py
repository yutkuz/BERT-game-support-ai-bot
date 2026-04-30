from __future__ import annotations

import threading


class JobRegistry:
    def __init__(self) -> None:
        self._jobs: dict[str, dict[str, bool]] = {}
        self._lock = threading.Lock()

    def start(self, job_id: str) -> None:
        with self._lock:
            self._jobs[job_id] = {"cancelled": False}

    def cancel(self, job_id: str) -> None:
        with self._lock:
            if job_id in self._jobs:
                self._jobs[job_id]["cancelled"] = True

    def is_cancelled(self, job_id: str) -> bool:
        with self._lock:
            return bool(self._jobs.get(job_id, {}).get("cancelled"))

    def finish(self, job_id: str) -> None:
        with self._lock:
            self._jobs.pop(job_id, None)


jobs = JobRegistry()
