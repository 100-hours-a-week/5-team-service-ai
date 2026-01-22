import logging
import re
import time
from collections import Counter

from fastapi import HTTPException, status

from app.clients.gemini_client import GeminiClient, GeminiClientError
from app.core.config import Settings
from app.schemas.book_report_schema import (
    BookReportValidationRequest,
    BookReportValidationResponse,
)


class BookReportValidationService:
    def __init__(
        self,
        gemini_client: GeminiClient,
        settings: Settings,
    ) -> None:
        self.gemini_client = gemini_client
        self.settings = settings
        self.logger = logging.getLogger(__name__)

    def validate_report(
        self,
        book_report_id: int,
        payload: BookReportValidationRequest,
    ) -> BookReportValidationResponse:
        start_time = time.perf_counter()
        gemini_called = False
        rule_rejected = False

        content = (payload.content or "").strip()
        title = payload.title
        if not content:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Book report content is required",
            )

        rule_rejected, rule_reason = self._rule_based_filter(content)
        if rule_rejected:
            result_status = "REJECTED"
            rejection_reason = rule_reason
        else:
            gemini_called = True
            try:
                gemini_result = self.gemini_client.evaluate_book_report(title, content)
                result_status = gemini_result.status
                rejection_reason = gemini_result.rejection_reason
            except GeminiClientError as exc:
                self.logger.error("Gemini validation failed: %s", exc, exc_info=exc)
                code = status.HTTP_503_SERVICE_UNAVAILABLE
                if "파싱" in str(exc) or "응답" in str(exc):
                    code = status.HTTP_502_BAD_GATEWAY
                raise HTTPException(status_code=code, detail=f"Gemini validation failed: {exc}") from exc
            except Exception as exc:  # noqa: BLE001
                self.logger.exception("Unexpected Gemini error: %s", exc)
                raise HTTPException(
                    status_code=status.HTTP_502_BAD_GATEWAY,
                    detail=f"Gemini validation failed: {exc}",
                ) from exc

        response = BookReportValidationResponse(
            id=book_report_id,
            user_id=payload.user_id or 0,
            status=result_status,
            rejection_reason=rejection_reason,
        )

        elapsed_ms = int((time.perf_counter() - start_time) * 1000)
        self.logger.info(
            "book report validation completed",
            extra={
                "id": book_report_id,
                "rule_rejected": rule_rejected,
                "gemini_called": gemini_called,
                "status": result_status,
                "elapsed_ms": elapsed_ms,
            },
        )
        return response

    def _rule_based_filter(self, content: str) -> tuple[bool, str | None]:
        # Thresholds are read from Settings so they can be tuned without code changes.
        if len(content) < self.settings.min_content_length:
            return True, "독후감 길이가 최소 기준에 미달합니다."

        if len(content) > self.settings.max_content_length:
            return True, "독후감 길이가 최대 기준을 초과합니다."

        words = re.findall(r"[가-힣A-Za-z0-9']+", content.lower())
        if words:
            counts = Counter(words)
            most_common_word, max_count = counts.most_common(1)[0]
            if (max_count / len(words)) > self.settings.max_repeat_word_ratio:
                return True, f"동일 단어('{most_common_word}')가 과도하게 반복됩니다."

        sentences = [s.strip() for s in re.split(r"[.!?\n]+", content) if s.strip()]
        if sentences:
            sentence_counts = Counter(sentences)
            if any(
                count >= self.settings.max_repeated_sentences
                for count in sentence_counts.values()
            ):
                return True, "동일 문장이 반복되어 독후감으로 보기 어렵습니다."

        noise_chars = re.findall(r"[^가-힣a-zA-Z0-9\s.,!?'\"-]", content)
        if content and (len(noise_chars) / len(content)) > self.settings.max_noise_char_ratio:
            return True, "무의미한 문자/기호가 과도하게 포함되어 있습니다."

        link_or_tag_matches = re.findall(r"(https?://|www\.|#\w+)", content)
        if len(link_or_tag_matches) > self.settings.max_links_or_tags:
            return True, "광고/링크/해시태그가 과도하게 포함되어 있습니다."

        if self._has_repeated_sequence(content):
            return True, "의미 없는 구절이 반복됩니다."

        return False, None

    def _has_repeated_sequence(self, content: str) -> bool:
        """Detects simple copy-paste spam like '재미있었다. 재미있었다...'."""
        condensed = re.sub(r"\s+", " ", content)
        phrases = re.findall(r"([가-힣A-Za-z0-9]{3,20})\1{2,}", condensed)
        return len(phrases) > 0
