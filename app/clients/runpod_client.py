from __future__ import annotations

import logging
import time
from typing import Any

import requests

from app.core.metrics import (
    observe_cold_start_event,
    observe_estimated_cost_index,
    observe_external_call,
)

logger = logging.getLogger(__name__)


class RunpodClient:
    """
    Minimal RunPod serverless client for submit + poll.
    """

    def __init__(
        self,
        endpoint_id: str,
        api_key: str,
        poll_interval: int = 2,
        poll_timeout: int = 120,
        cold_start_queue_threshold: int = 10,
    ):
        self.endpoint_id = endpoint_id
        self.api_key = api_key
        self.poll_interval = poll_interval
        self.poll_timeout = poll_timeout
        self.cold_start_queue_threshold = cold_start_queue_threshold
        self.base_url = f"https://api.runpod.ai/v2/{endpoint_id}"

    def _headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self.api_key}"}

    def submit(self, input_payload: dict[str, Any]) -> str:
        """
        Submit a job and return the job id.
        """
        resp = requests.post(
            f"{self.base_url}/run",
            headers=self._headers(),
            json={"input": input_payload},
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json()
        job_id = data.get("id") or data.get("jobId")
        if not job_id:
            raise RuntimeError(f"RunPod did not return job id: {data}")
        return job_id

    def poll(self, job_id: str) -> tuple[dict[str, Any], bool]:
        """
        Poll until completion or timeout.
        Returns: (output, cold_start_hint)
        """
        deadline = time.time() + self.poll_timeout
        url = f"{self.base_url}/status/{job_id}"
        queued_at: float | None = None
        cold_start_hint = False
        while True:
            if time.time() > deadline:
                raise TimeoutError(
                    f"RunPod job timeout after {self.poll_timeout}s (job_id={job_id})"
                )

            resp = requests.get(url, headers=self._headers(), timeout=15)
            resp.raise_for_status()
            data = resp.json()
            status = data.get("status")
            if status in {"IN_QUEUE", "QUEUED", "PENDING"}:
                if queued_at is None:
                    queued_at = time.time()
            elif (
                status in {"IN_PROGRESS", "RUNNING", "COMPLETED"}
                and queued_at is not None
                and not cold_start_hint
                and (time.time() - queued_at) >= self.cold_start_queue_threshold
            ):
                cold_start_hint = True

            if status == "COMPLETED":
                return data.get("output") or {}, cold_start_hint
            if status in {"FAILED", "CANCELLED"}:
                raise RuntimeError(f"RunPod job {status.lower()}: {data}")

            time.sleep(self.poll_interval)

    def generate(self, input_payload: dict[str, Any]) -> tuple[str, dict[str, Any]]:
        started_at = time.perf_counter()
        model = "serverless"
        try:
            job_id = self.submit(input_payload)
            output, cold_start_hint = self.poll(job_id)
            observe_external_call(
                provider="runpod",
                model=model,
                result="success",
                elapsed_seconds=time.perf_counter() - started_at,
            )
            delta_index = 1.0
            if cold_start_hint:
                observe_cold_start_event("idle_resume")
                delta_index += 3.0
            observe_estimated_cost_index(
                provider="runpod",
                delta_index=delta_index,
                success_request=True,
            )
            return job_id, output
        except TimeoutError:
            observe_external_call(
                provider="runpod",
                model=model,
                result="timeout",
                elapsed_seconds=time.perf_counter() - started_at,
            )
            observe_estimated_cost_index(
                provider="runpod",
                delta_index=1.0,
                success_request=False,
            )
            raise
        except Exception:
            observe_external_call(
                provider="runpod",
                model=model,
                result="error",
                elapsed_seconds=time.perf_counter() - started_at,
            )
            observe_estimated_cost_index(
                provider="runpod",
                delta_index=1.0,
                success_request=False,
            )
            raise
