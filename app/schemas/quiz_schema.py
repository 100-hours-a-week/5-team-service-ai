"""Pydantic schemas for quiz generation API."""

from pydantic import BaseModel, Field

__all__ = ["Quiz", "QuizChoice", "QuizGenerateResponse"]


class Quiz(BaseModel):
    room_id: int = Field(..., gt=0, description="퀴즈가 속한 방/세션 ID")
    question: str = Field(..., description="객관식 문제 문장")
    correct_choice_number: int = Field(..., ge=1, le=4, description="정답 보기 번호 (1-4)")


class QuizChoice(BaseModel):
    room_id: int = Field(..., gt=0, description="퀴즈가 속한 방/세션 ID")
    choice_number: int = Field(..., ge=1, le=4, description="보기 번호 (1-4)")
    choice_text: str = Field(..., description="보기 내용")


class QuizGenerateResponse(BaseModel):
    quiz: Quiz
    quiz_choices: list[QuizChoice]

