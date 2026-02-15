from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from app.clients.runpod_client import RunpodClient
from app.core.config import get_settings
from app.core.security import require_api_key

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/ai/quiz",
    tags=["quiz"],
    dependencies=[Depends(require_api_key)],
)


class QuizGenerateRequest(BaseModel):
    author: str = Field(..., description="저자명")
    title: str = Field(..., description="책 제목")
    prompt: str | None = Field(None, description="직접 프롬프트를 넣고 싶을 때 (옵션)")
    max_new_tokens: int = Field(256, ge=16, le=2048)
    temperature: float = Field(0.7, ge=0.0, le=2.0)
    top_p: float = Field(0.9, ge=0.0, le=1.0)


class QuizGenerateResponse(BaseModel):
    job_id: str
    text: str


def _client() -> RunpodClient:
    settings = get_settings()
    if not settings.runpod_endpoint_id or not settings.runpod_api_key:
        raise HTTPException(status_code=503, detail="RUNPOD endpoint or API key not configured")
    return RunpodClient(
        endpoint_id=settings.runpod_endpoint_id,
        api_key=settings.runpod_api_key,
        poll_interval=settings.runpod_poll_interval_seconds,
        poll_timeout=settings.runpod_poll_timeout_seconds,
    )


@router.post(
    "/generate",
    response_model=QuizGenerateResponse,
    summary="RunPod serverless를 호출해 퀴즈 생성",
)
def generate_quiz(body: QuizGenerateRequest) -> QuizGenerateResponse:
    client = _client()

    prompt_text = body.prompt
    if not prompt_text:
        prompt_text = (
            f"저자 {body.author}의 책 '{body.title}' 내용을 바탕으로, "
            "한국어 객관식 퀴즈 5문항을 만들어줘. "
            "각 문항은 보기 4개와 정답 번호, 간단한 해설을 포함해 JSON 배열로 반환해."
        )

    payload = {
        "prompt": prompt_text,
        "max_new_tokens": body.max_new_tokens,
        "temperature": body.temperature,
        "top_p": body.top_p,
    }

    try:
        job_id, output = client.generate(payload)
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001
        logger.exception("RunPod quiz generation failed: %s", exc)
        raise HTTPException(status_code=503, detail="quiz generation failed") from exc

    text = ""
    if isinstance(output, dict):
        text = output.get("text") or ""
        if not text:
            text = output.get("output") or ""
    if not text:
        text = str(output)

    return QuizGenerateResponse(job_id=job_id, text=text)
