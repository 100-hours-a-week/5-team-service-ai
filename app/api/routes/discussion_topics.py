from __future__ import annotations

import logging
from functools import lru_cache

from fastapi import APIRouter, Depends, HTTPException

from app.clients.runpod_client import RunpodClient
from app.core.config import get_settings
from app.core.security import require_api_key
from app.schemas.discussion_topic_schema import (
    DiscussionTopicGenerateRequest,
    DiscussionTopicGenerateResponse,
)
from app.services.discussion_topic_service import DiscussionTopicService

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/meeting-rounds",
    tags=["discussion-topics"],
    dependencies=[Depends(require_api_key)],
)


@lru_cache
def _discussion_service() -> DiscussionTopicService:
    settings = get_settings()
    # Prefer FT endpoint for topic generation as requested; fall back to existing values.
    endpoint_id = settings.ft_endpoint_id or settings.base_endpoint_id or settings.runpod_endpoint_id
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
    return DiscussionTopicService(runpod_client=client)


def get_discussion_topic_service() -> DiscussionTopicService:
    """Dependency provider to allow overrides in tests."""
    return _discussion_service()


@router.post(
    "/{meetingRoundId}/discussion-topics/generate",
    response_model=DiscussionTopicGenerateResponse,
    summary="meeting_round_id 독후감 기반 토론 주제 1개 생성",
)
async def generate_discussion_topics(
    meetingRoundId: int,
    payload: DiscussionTopicGenerateRequest,
    service: DiscussionTopicService = Depends(get_discussion_topic_service),
) -> DiscussionTopicGenerateResponse:
    try:
        return await service.generate_topics(
            meeting_round_id=meetingRoundId,
            reports=payload.reports,
        )
    except Exception as exc:  # noqa: BLE001
        logger.exception("discussion topic generation failed: %s", exc)
        raise HTTPException(
            status_code=503, detail="discussion topic generation failed"
        ) from exc
