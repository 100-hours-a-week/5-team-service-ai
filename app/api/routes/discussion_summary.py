from __future__ import annotations

import logging
from functools import lru_cache
from fastapi import APIRouter, Depends, HTTPException

from app.core.security import require_api_key
from app.schemas.discussion_summary_schema import (
    DiscussionSummaryRequest,
    DiscussionSummaryResponse,
)
from app.core.config import get_settings
from app.clients.runpod_client import RunpodClient
from app.services.discussion_summary_service import DiscussionSummaryService

logger = logging.getLogger(__name__)


router = APIRouter(
    prefix="/chat-rooms",
    tags=["discussion-summary"],
    dependencies=[Depends(require_api_key)],
)


@lru_cache
def _summary_service() -> DiscussionSummaryService:
    settings = get_settings()
    endpoint_id = settings.ft_endpoint_id or settings.runpod_endpoint_id
    if not endpoint_id or not settings.runpod_api_key:
        raise HTTPException(
            status_code=503, detail="RUNPOD endpoint or API key not configured"
        )

    client = RunpodClient(
        endpoint_id=endpoint_id,
        api_key=settings.runpod_api_key,
        poll_interval=settings.runpod_poll_interval_seconds,
        poll_timeout=settings.runpod_poll_timeout_seconds,
    )
    return DiscussionSummaryService(runpod_client=client)


def get_summary_service() -> DiscussionSummaryService:
    return _summary_service()


@router.post(
    "/{roomId}/discussion-summary",
    response_model=DiscussionSummaryResponse,
    summary="채팅 메시지 기반 토론 요약 생성",
)
async def generate_discussion_summary(
    roomId: int,
    payload: DiscussionSummaryRequest,
    service: DiscussionSummaryService = Depends(get_summary_service),
) -> DiscussionSummaryResponse:
    try:
        if not payload.messages:
            raise ValueError("messages must contain at least one item")

        return service.summarize(
            room_id=roomId,
            topic=payload.topic,
            round_no=payload.round_no,
            messages=payload.messages,
        )
    except Exception as exc:  # noqa: BLE001
        logger.exception("discussion summary generation failed: %s", exc)
        raise HTTPException(status_code=503, detail="discussion summary failed") from exc
