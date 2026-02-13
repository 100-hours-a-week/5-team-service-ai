from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.clients.kanana_client import KananaClient
from app.core.config import get_settings
from app.core.security import require_api_key
from app.db.repositories.book_repo import BookRepository
from app.db.session import get_db
from app.schemas.quiz_schema import QuizGenerateRequest, QuizGenerateResponse
from app.services.quiz_generator_service import QuizGenerationError, QuizGeneratorService
from app.services.embedder import Embedder
from app.services.faiss_store import FaissStore

router = APIRouter(
    prefix="/ai/quiz",
    tags=["quiz"],
    dependencies=[Depends(require_api_key)],
)

settings = get_settings()
book_repo = BookRepository()
kanana_client = None
if settings.kanana_base_url:
    kanana_client = KananaClient(
        base_url=settings.kanana_base_url,
        api_key=settings.kanana_api_key,
        model=settings.kanana_model,
        timeout_seconds=settings.kanana_timeout_seconds,
    )
embedder = Embedder()
faiss_store = FaissStore()
try:
    faiss_store.load(settings.books_faiss_index_path, settings.books_faiss_meta_path)
except Exception:
    faiss_store = None

quiz_service = QuizGeneratorService(
    book_repo=book_repo,
    llm_client=kanana_client,
    embedder=embedder,
    faiss_store=faiss_store,
    settings=settings,
)


def get_quiz_service() -> QuizGeneratorService:
    return quiz_service


@router.post(
    "/generate",
    response_model=QuizGenerateResponse,
    status_code=status.HTTP_200_OK,
    summary="Generate a quiz for a book",
)
def generate_quiz(
    body: QuizGenerateRequest,
    db: Session = Depends(get_db),
    service: QuizGeneratorService = Depends(get_quiz_service),
) -> QuizGenerateResponse:
    try:
        return service.generate(title=body.title, author=body.author, db=db)
    except QuizGenerationError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail="quiz generation failed") from exc
