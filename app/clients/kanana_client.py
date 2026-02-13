import logging
from dataclasses import dataclass
from typing import Any, Dict, Optional

import requests


class KananaClientError(Exception):
    """Raised when quiz generation fails."""


@dataclass
class KananaClient:
    base_url: str
    model: str
    api_key: Optional[str] = None
    timeout_seconds: int = 30

    def __post_init__(self):
        self.logger = logging.getLogger(__name__)

    def generate(self, prompt: str) -> str:
        """
        Call an OpenAI-chat 호환 엔드포인트로 퀴즈 JSON을 생성한다.
        """
        url = self.base_url.rstrip("/")
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": "너는 한국어 독서 퀴즈를 만드는 조교다."},
                {"role": "user", "content": prompt},
            ],
            "response_format": {"type": "json_object"},
            "temperature": 0.3,
        }

        try:
            resp = requests.post(url, json=payload, headers=headers, timeout=self.timeout_seconds)
        except Exception as exc:  # noqa: BLE001
            raise KananaClientError(f"Kanana API 호출 실패: {exc}") from exc

        if resp.status_code >= 300:
            raise KananaClientError(f"Kanana API 오류 status={resp.status_code} body={resp.text[:500]}")

        try:
            data = resp.json()
        except Exception as exc:  # noqa: BLE001
            raise KananaClientError(f"Kanana 응답 JSON 파싱 실패: {resp.text[:200]}") from exc

        # OpenAI 호환 구조에서 content 추출
        try:
            choice = data["choices"][0]["message"]["content"]
        except Exception as exc:  # noqa: BLE001
            raise KananaClientError(f"Kanana 응답 형식 오류: {data}") from exc

        return choice
