from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field


class BookReportValidationRequest(BaseModel):
    model_config = ConfigDict(extra="ignore")

    user_id: Optional[int] = Field(None, description="Optional user identifier")
    meeting_session_id: Optional[int] = Field(None, description="Optional session identifier")
    book_title: Optional[str] = Field(None, description="Title of the book being reviewed")
    content: Optional[str] = Field(
        None, description="Optional content override; falls back to DB content when omitted"
    )


class BookReportValidationResponse(BaseModel):
    id: int
    user_id: int
    meeting_session_id: int
    content: str
    status: Literal["SUBMITTED", "REJECTED"]
    rejection_reason: Optional[str] = None
