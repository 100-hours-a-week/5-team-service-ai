from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from app.batch.weekly_batch import generate_from_db
from app.core.security import require_api_key
from app.db.session import get_db

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/ai", tags=["recommendations"], dependencies=[Depends(require_api_key)])


class RecommendationRequest(BaseModel):
    top_k: int = Field(4, ge=1, le=10)
    search_k: int = Field(20, ge=1)


@router.post(
    "/recommendations",
    summary="Generate recommendations from DB",
)
def generate_recommendations_post(
    body: RecommendationRequest,
    db: Session = Depends(get_db),
) -> dict:
    try:
        result = generate_from_db(top_k=body.top_k, search_k=body.search_k, db=db, persist=True)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        logger.exception("Failed to generate recommendations: %s", exc)
        raise HTTPException(status_code=503, detail="recommendation generation failed") from exc

    rows = result.get("rows") or []
    return {
        "users": result.get("users"),
        "rows_count": len(rows),
        "inserted": result.get("inserted"),
    }
