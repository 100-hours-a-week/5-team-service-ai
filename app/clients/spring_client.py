"""
Lightweight client to push recommendations to the Spring API.
"""

from __future__ import annotations

import logging
from typing import Iterable

logger = logging.getLogger(__name__)


def _http_client():
    try:
        import requests

        return requests
    except ImportError:
        try:
            import httpx

            return httpx
        except ImportError as exc:  # pragma: no cover - import guard
            raise ImportError("requests 또는 httpx가 필요합니다. 패키지를 설치하세요.") from exc


def post_recommendations(base_url: str, rows: Iterable[dict], timeout: int = 10) -> dict:
    """
    POST recommendation rows to `${base_url}/api/recommendations`.

    Parameters
    ----------
    base_url : str
        Service base URL (e.g., http://localhost:8080).
    rows : Iterable[dict]
        Recommendation rows. Caller ensures shape:
        {"user_id", "meeting_id", "week_start_date", "rank"}
    timeout : int
        Request timeout seconds.

    Returns
    -------
    dict
        {"status_code": int, "text": str, "ok": bool}
    """
    client = _http_client()
    url = base_url.rstrip("/")
    if not url.endswith("/api/recommendations"):
        url = f"{url}/api/recommendations"

    payload = {"rows": list(rows)}
    try:
        resp = client.post(url, json=payload, timeout=timeout)
    except Exception as exc:  # noqa: BLE001
        logger.error("Failed to POST recommendations: %s", exc)
        raise

    try:
        text = resp.text
    except Exception:  # pragma: no cover - best effort
        text = ""

    return {"status_code": getattr(resp, "status_code", None), "text": text, "ok": getattr(resp, "ok", False)}
