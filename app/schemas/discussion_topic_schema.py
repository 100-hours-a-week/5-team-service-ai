from __future__ import annotations

from datetime import datetime
from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field


class DiscussionReport(BaseModel):
    model_config = ConfigDict(extra="ignore")

    id: int = Field(..., description="원본 리포트 ID")
    content: str = Field(..., description="리포트 내용")


class DiscussionTopicGenerateRequest(BaseModel):
    model_config = ConfigDict(populate_by_name=True)
    reports: list[DiscussionReport] = Field(..., description="토론 주제 생성에 사용할 리포트 목록")


class DiscussionTopic(BaseModel):
    id: int
    meeting_round_id: int
    topic_no: int
    topic: str
    source: Literal["AI", "HUMAN"] = "AI"
    created_at: datetime
    updated_at: datetime
    deleted_at: Optional[datetime] = None


class DiscussionTopicData(BaseModel):
    topics: list[DiscussionTopic]


class DiscussionTopicSingle(BaseModel):
    model_config = ConfigDict(populate_by_name=True)
    topic: str


class DiscussionTopicGenerateResponse(BaseModel):
    status: Literal["success"] = "success"
    data: DiscussionTopicSingle
