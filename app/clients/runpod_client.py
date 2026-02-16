from __future__ import annotations

import logging
import time
from typing import Any

import requests

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
    ):
        self.endpoint_id = endpoint_id
        self.api_key = api_key
        self.poll_interval = poll_interval
        self.poll_timeout = poll_timeout
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

    def poll(self, job_id: str) -> dict[str, Any]:
        """
        Poll until completion or timeout. Returns the `output` dict.
        """
        deadline = time.time() + self.poll_timeout
        url = f"{self.base_url}/status/{job_id}"
        while True:
            if time.time() > deadline:
                raise TimeoutError(f"RunPod job timeout after {self.poll_timeout}s (job_id={job_id})")

            resp = requests.get(url, headers=self._headers(), timeout=15)
            resp.raise_for_status()
            data = resp.json()
            status = data.get("status")
            if status == "COMPLETED":
                return data.get("output") or {}
            if status in {"FAILED", "CANCELLED"}:
                raise RuntimeError(f"RunPod job {status.lower()}: {data}")

            time.sleep(self.poll_interval)

    def generate(self, input_payload: dict[str, Any]) -> tuple[str, dict[str, Any]]:
        job_id = self.submit(input_payload)
        output = self.poll(job_id)
        return job_id, output
