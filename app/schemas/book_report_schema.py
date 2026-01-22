from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field


class BookReportValidationRequest(BaseModel):
    model_config = ConfigDict(extra="ignore")

    user_id: Optional[int] = Field(None, description="User identifier")
    title: Optional[str] = Field(None, description="Title of the book being reviewed")
    content: Optional[str] = Field(None, description="Book report content text")


class BookReportValidationResponse(BaseModel):
    id: int
    user_id: int
    status: Literal["SUBMITTED", "REJECTED"]
    rejection_reason: Optional[str] = None
