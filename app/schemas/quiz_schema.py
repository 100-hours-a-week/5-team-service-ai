"""Pydantic schemas for quiz generation API."""

from pydantic import BaseModel, Field

__all__ = ["Quiz", "QuizChoice", "QuizGenerateResponse"]


class Quiz(BaseModel):
    question: str = Field(..., description="객관식 문제 문장")
    correct_choice_number: int = Field(
        ..., ge=1, le=4, description="정답 보기 번호 (1-4)"
    )


class QuizChoice(BaseModel):
    choice_number: int = Field(..., ge=1, le=4, description="보기 번호 (1-4)")
    choice_text: str = Field(..., description="보기 내용")


class QuizGenerateResponse(BaseModel):
    quiz: Quiz
    quiz_choices: list[QuizChoice]
