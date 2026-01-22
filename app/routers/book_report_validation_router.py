from fastapi import APIRouter

from app.clients.gemini_client import GeminiClient
from app.core.config import get_settings
from app.schemas.book_report_schema import (
    BookReportValidationRequest,
    BookReportValidationResponse,
)
from app.services.book_report_validation_service import (
    BookReportValidationService,
)

settings = get_settings()
gemini_client = GeminiClient(
    api_key=settings.gemini_api_key,
    model_name=settings.gemini_model,
    model_preferences=settings.gemini_model_preferences,
    timeout_seconds=settings.gemini_timeout_seconds,
    max_output_tokens=settings.gemini_max_output_tokens,
    log_models_on_start=settings.gemini_log_models_on_start,
)
validation_service = BookReportValidationService(gemini_client, settings)

router = APIRouter(prefix="/api/book-reports", tags=["book-reports"])


@router.post(
    "/{id}/validate",
    response_model=BookReportValidationResponse,
    status_code=200,
)
def validate_book_report(
    id: int,
    payload: BookReportValidationRequest,
) -> BookReportValidationResponse:
    """
    Validate whether a book report is genuine and meaningful.
    """
    return validation_service.validate_report(id, payload)
