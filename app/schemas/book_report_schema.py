from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field


class BookReportValidationRequest(BaseModel):
    model_config = ConfigDict(extra="ignore")

    title: Optional[str] = Field(None, description="Title of the book being reviewed")
    content: Optional[str] = Field(None, description="Book report content text")


class BookReportValidationResponse(BaseModel):
    status: Literal["SUBMITTED", "REJECTED"]
    rejection_reason: Optional[str] = None
