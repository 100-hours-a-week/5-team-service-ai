import json
import logging
from dataclasses import dataclass
from typing import Optional, Sequence

from sqlalchemy.orm import Session

from app.clients.kanana_client import KananaClient
from app.db.repositories.book_repo import BookRepository
from app.services.embedder import Embedder
from app.services.faiss_store import FaissStore
from app.schemas.quiz_schema import QuizGenerateResponse, Quiz, QuizChoice


class QuizGenerationError(Exception):
    """Raised when a quiz cannot be generated."""


@dataclass
class QuizGeneratorService:
    """
    책 요약(summary)을 RAG 컨텍스트로 사용해 퀴즈를 생성한다.
    """

    book_repo: BookRepository
    llm_client: Optional[KananaClient] = None
    embedder: Optional[Embedder] = None
    faiss_store: Optional[FaissStore] = None
    settings: Optional[object] = None
    logger: logging.Logger = logging.getLogger(__name__)

    def generate(self, *, title: str, author: str, db: Session) -> QuizGenerateResponse:
        # 1) 책 요약 조회 (실패해도 fallback)
        try:
            book = self.book_repo.get_by_title_author(db, title=title, author=author)
        except Exception as exc:  # noqa: BLE001
            self.logger.warning("Book lookup failed, falling back without summary: %s", exc)
            book = None

        # 2) 벡터 검색 시도 (인덱스와 임베더가 있을 때만)
        contexts: list[dict] = []
        if self.faiss_store and self.embedder:
            contexts = self._retrieve_contexts(title=title, author=author, top_k=self._ctx_k)

        # 3) 프롬프트 구성용 컨텍스트 결정
        if contexts:
            prompt = self._build_prompt_with_contexts(title=title, author=author, contexts=contexts)
        elif book and book.get("summary"):
            prompt = self._build_prompt(book_title=book["title"], author=book["author"], summary=book["summary"])
        else:
            self.logger.warning("Book summary missing, falling back with default quiz.")
            return self._fallback_quiz(book_title=title, author=author)

        if self.llm_client:
            try:
                raw = self.llm_client.generate(prompt)
                return self._parse_response(raw)
            except QuizGenerationError as exc:
                self.logger.warning("Quiz LLM response parse failed, falling back: %s | raw=%r", exc, raw if 'raw' in locals() else None)
            except Exception as exc:  # noqa: BLE001
                self.logger.warning("Quiz LLM call failed, falling back: %s", exc)

        # Fallback: LLM 없이도 동작하도록 간단 퀴즈 생성 (또는 LLM 실패 시)
        return self._fallback_quiz(book_title=book["title"] if book else title, author=book["author"] if book else author)

    @property
    def _ctx_k(self) -> int:
        if self.settings and getattr(self.settings, "books_context_k", None):
            return self.settings.books_context_k
        return 4

    def _build_prompt(self, book_title: str, author: str, summary: str) -> str:
        return (
            "다음 책 요약을 기반으로 한국어 독서 퀴즈를 만들어라.\n"
            "- 문제는 책의 핵심 내용/상징/인물/사실을 묻는 1개 객관식 질문\n"
            "- 선택지는 4개, 번호는 1~4, 중복 없음\n"
            "- correct_choice_number는 정답 번호 (1~4)\n"
            "- 출력은 순수 JSON 한 덩어리로만 반환. 추가 텍스트/코멘트 금지.\n"
            "출력 스키마: {\n"
            '  "quiz": {\n'
            '    "room_id": int,\n'
            '    "question": str,\n'
            '    "correct_choice_number": int\n'
            "  },\n"
            '  "quiz_choices": [\n'
            '    {"room_id": int, "choice_number": int, "choice_text": str}, ... 4개\n'
            "  ]\n"
            "}\n"
            f"책 제목: {book_title}\n"
            f"저자: {author}\n"
            f"요약:\n{summary}\n"
        )

    def _build_prompt_with_contexts(self, title: str, author: str, contexts: Sequence[dict]) -> str:
        ctx_lines = []
        for i, ctx in enumerate(contexts, start=1):
            ctx_text = ctx.get("chunk_text") or ctx.get("text") or ""
            ctx_title = ctx.get("title") or ""
            ctx_authors = ctx.get("authors") or ""
            ctx_lines.append(f"[{i}] 제목: {ctx_title} / 저자: {ctx_authors}\n{ctx_text}")
        context_block = "\n\n".join(ctx_lines)
        return (
            "아래 컨텍스트 안에서만 근거를 찾아 한국어 객관식 퀴즈 1문제를 만들어라.\n"
            "- 선택지는 4개, 번호는 1~4, 중복 없음\n"
            "- correct_choice_number는 정답 번호 (1~4)\n"
            "- 출력은 순수 JSON 한 덩어리만 반환 (주석/코드펜스 금지)\n"
            "출력 스키마: {\n"
            '  \"quiz\": {\"room_id\": int, \"question\": str, \"correct_choice_number\": int},\n'
            '  \"quiz_choices\": [ {\"room_id\": int, \"choice_number\": int, \"choice_text\": str}, ... 4개 ]\n'
            "}\n"
            f"요청 책: 제목={title}, 저자={author}\n"
            "컨텍스트:\n"
            f"{context_block}\n"
        )

    def _parse_response(self, response_text: str) -> QuizGenerateResponse:
        data = self._extract_json(response_text)
        if data is None:
            raise QuizGenerationError("LLM 응답이 JSON이 아닙니다.")

        try:
            quiz = Quiz(**data["quiz"])
            choices = [QuizChoice(**c) for c in data["quiz_choices"]]
        except Exception as exc:  # noqa: BLE001
            raise QuizGenerationError("LLM 응답 스키마가 올바르지 않습니다.") from exc

        numbers = {c.choice_number for c in choices}
        if numbers != {1, 2, 3, 4}:
            raise QuizGenerationError("choice_number가 1~4를 모두 포함해야 합니다.")
        if quiz.correct_choice_number not in numbers:
            raise QuizGenerationError("correct_choice_number가 선택지 범위 밖입니다.")

        return QuizGenerateResponse(quiz=quiz, quiz_choices=choices)

    def _extract_json(self, text: str) -> dict | None:
        """
        LLM 응답에서 JSON 덩어리를 관대하게 추출한다.
        """
        # 1) 그대로 시도
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        cleaned = text.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.strip("`")
            if cleaned.lower().startswith("json"):
                cleaned = cleaned[4:].strip()

        import re

        match = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                return None
        return None

    def _retrieve_contexts(self, title: str, author: str, top_k: int) -> list[dict]:
        try:
            query_text = f"제목: {title}, 저자: {author}. 이 책에 대한 핵심 내용/주제/인물 정보를 찾아라."
            q_vec = self.embedder.encode([query_text])[0]
            hits = self.faiss_store.search_with_meta(q_vec, top_k=top_k)
        except Exception as exc:  # noqa: BLE001
            self.logger.warning("FAISS search failed, skipping retrieval: %s", exc)
            return []

        # 재랭크: title/author 보너스
        def _normalize(text: str | None) -> str:
            return (text or "").strip().lower()

        req_title = _normalize(title)
        req_author = _normalize(author)
        bonus_title = getattr(self.settings, "books_title_bonus", 0.05) if self.settings else 0.05
        bonus_author = getattr(self.settings, "books_author_bonus", 0.05) if self.settings else 0.05

        reranked = []
        for hit in hits:
            meta = hit.get("meta", {}) or {}
            score = float(hit.get("score", 0.0))
            m_title = _normalize(meta.get("title") or meta.get("book_title") or meta.get("meeting_title"))
            m_author = _normalize(meta.get("authors") or meta.get("author") or meta.get("leader"))
            if m_title and req_title and m_title == req_title:
                score += bonus_title
            if m_author and req_author and m_author == req_author:
                score += bonus_author
            reranked.append(
                {
                    "score": score,
                    "chunk_text": meta.get("chunk_text") or meta.get("text"),
                    "title": meta.get("title") or meta.get("book_title"),
                    "authors": meta.get("authors") or meta.get("author"),
                }
            )

        reranked.sort(key=lambda x: x["score"], reverse=True)
        return reranked[:top_k]

    def _fallback_quiz(self, book_title: str, author: str) -> QuizGenerateResponse:
        # 간단한 고정형 4지선다 생성 (LLM 없이 동작)
        distractors = [
            "어니스트 헤밍웨이",
            "버지니아 울프",
            "제임스 조이스",
        ]
        quiz = Quiz(room_id=1, question=f"'{book_title}'의 저자는 누구인가?", correct_choice_number=1)
        choices = [
            QuizChoice(room_id=1, choice_number=1, choice_text=author),
            QuizChoice(room_id=1, choice_number=2, choice_text=distractors[0]),
            QuizChoice(room_id=1, choice_number=3, choice_text=distractors[1]),
            QuizChoice(room_id=1, choice_number=4, choice_text=distractors[2]),
        ]
        return QuizGenerateResponse(quiz=quiz, quiz_choices=choices)
