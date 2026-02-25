from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field


class DiscussionMessage(BaseModel):
    """단일 대화 메시지."""

    model_config = ConfigDict(populate_by_name=True, extra="ignore")

    sender_nickname: str = Field(..., alias="senderNickname", description="메시지 보낸 사용자 닉네임")
    text_message: str = Field(..., alias="textMessage", description="메시지 본문 텍스트")


class DiscussionSummaryRequest(BaseModel):
    """방 단위 토론 요약 요청."""

    model_config = ConfigDict(populate_by_name=True, extra="ignore")

    topic: str = Field("", description="토론 주제 또는 힌트")
    round_no: str = Field("", alias="roundNo", description="라운드 번호")
    messages: list[DiscussionMessage] = Field(..., min_length=1, description="대화 메시지 목록")


class DiscussionSummary(BaseModel):
    """요약 결과 블록."""

    pro: list[str] = Field(default_factory=list, description="긍정적/장점 요약")
    con: list[str] = Field(default_factory=list, description="우려/단점 요약")
    main_issues: list[str] = Field(default_factory=list, description="핵심 쟁점 목록")
    unresolved_issues: list[str] = Field(default_factory=list, description="미해결/추가 논의 필요 사항")


class DiscussionSummaryResponse(BaseModel):
    """응답 최상위 스키마."""

    status: Literal["pending", "final"] = "final"
    summary: Optional[DiscussionSummary] = None
