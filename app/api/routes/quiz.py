from __future__ import annotations

import logging
from functools import lru_cache

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from app.clients.runpod_client import RunpodClient
from app.core.config import get_settings
from app.core.redis_client import get_redis_client
from app.core.security import require_api_key
from app.db.session import get_db
from app.schemas.quiz_schema import QuizGenerateResponse
from app.services.embedder import Embedder
from app.services.quiz_service import QuizService

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/quiz",
    tags=["quiz"],
    dependencies=[Depends(require_api_key)],
)


class QuizGenerateRequest(BaseModel):
    author: str = Field(..., description="저자명")
    title: str = Field(..., description="책 제목")


@lru_cache
def _quiz_service() -> QuizService:
    settings = get_settings()
    if not settings.runpod_endpoint_id or not settings.runpod_api_key:
        raise HTTPException(
            status_code=503, detail="RUNPOD endpoint or API key not configured"
        )

    client = RunpodClient(
        endpoint_id=settings.runpod_endpoint_id,
        api_key=settings.runpod_api_key,
        poll_interval=settings.runpod_poll_interval_seconds,
        poll_timeout=settings.runpod_poll_timeout_seconds,
    )
    embedder = Embedder()
    redis_client = get_redis_client(settings)
    return QuizService(
        runpod_client=client,
        embedder=embedder,
        redis_client=redis_client,
        cache_ttl_seconds=settings.quiz_cache_ttl_seconds,
        cache_key_version=settings.quiz_cache_key_version,
    )


def get_quiz_service() -> QuizService:
    """
    Dependency provider so tests can override.
    """
    try:
        return _quiz_service()
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001
        logger.exception("quiz service initialization failed: %s", exc)
        raise HTTPException(
            status_code=503, detail="quiz service initialization failed"
        ) from exc


@router.post(
    "/generate",
    response_model=QuizGenerateResponse,
    summary="책 정보를 바탕으로 객관식 퀴즈 1문항 생성",
)
def generate_quiz(
    body: QuizGenerateRequest,
    db: Session = Depends(get_db),
    quiz_service: QuizService = Depends(get_quiz_service),
) -> QuizGenerateResponse:
    try:
        return quiz_service.generate(
            title=body.title, author=body.author, db=db
        )
    except Exception as exc:  # noqa: BLE001
        logger.exception("quiz generation failed: %s", exc)
        raise HTTPException(status_code=503, detail="quiz generation failed") from exc
