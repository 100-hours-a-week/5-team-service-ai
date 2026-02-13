from pydantic import BaseModel, Field


class QuizGenerateRequest(BaseModel):
    title: str = Field(..., min_length=1, description="Book title")
    author: str = Field(..., min_length=1, description="Book author")


class Quiz(BaseModel):
    room_id: int = Field(..., description="Quiz room identifier")
    question: str = Field(..., min_length=1)
    correct_choice_number: int = Field(..., ge=1, le=4)


class QuizChoice(BaseModel):
    room_id: int
    choice_number: int = Field(..., ge=1, le=4)
    choice_text: str


class QuizGenerateResponse(BaseModel):
    quiz: Quiz
    quiz_choices: list[QuizChoice]
