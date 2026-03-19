from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from app.core.security import require_api_key
from app.db.repositories.recommendation_repo import RecommendationRepo
from app.db.session import get_db
from app.services.recommendation_v2 import generate_recommendations_v2

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="",
    tags=["recommendations"],
    dependencies=[Depends(require_api_key)],
)


class RecommendationV2Request(BaseModel):
    user_id: int = Field(..., ge=1)
    top_k: int = Field(4, ge=1, le=10)
    search_k: int = Field(60, ge=10, le=200)


@router.post("/recommendations/v2", summary="Generate recommendations with behavior reranker")
def generate_recommendations_v2_endpoint(
    body: RecommendationV2Request, db: Session = Depends(get_db)
) -> dict:
    repo = RecommendationRepo()
    try:
        rec_ids = generate_recommendations_v2(
            user_id=body.user_id,
            top_k=body.top_k,
            search_k=body.search_k,
            repo=repo,
            db=db,
        )
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        logger.exception("Failed to generate v2 recommendations: %s", exc)
        raise HTTPException(status_code=503, detail="recommendation v2 failed") from exc

    return {"user_id": body.user_id, "recommendations": rec_ids}

