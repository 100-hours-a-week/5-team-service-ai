import json
import logging
import re
from dataclasses import dataclass

try:
    from google import genai
    from google.api_core.exceptions import NotFound
    from google.genai import types as genai_types
except ImportError as exc:  # pragma: no cover - import guard
    raise ImportError(
        "google-genai가 설치되어야 합니다. 'pip install -r requirements.txt'로 최신 의존성을 설치하세요."
    ) from exc
from tenacity import AsyncRetrying, retry_if_exception, stop_after_attempt, wait_fixed


class GeminiClientError(Exception):
    """Raised when Gemini could not return a usable response."""


@dataclass
class GeminiResult:
    status: str
    rejection_reason: str | None


class GeminiClient:
    def __init__(
        self,
        api_key: str,
        model_name: str,
        model_preferences: list[str],
        timeout_seconds: int,
        max_output_tokens: int,
        max_parse_attempts: int = 2,
        log_models_on_start: bool = True,
    ) -> None:
        self.logger = logging.getLogger(__name__)
        self.client = genai.Client(api_key=api_key)
        self.async_client = self.client.aio
        self.model_name = model_name
        self.model_preferences = model_preferences
        self.timeout_seconds = timeout_seconds
        self.max_output_tokens = max_output_tokens
        self.max_parse_attempts = max_parse_attempts
        self._resolved_model: str | None = None

        if log_models_on_start:
            self._log_available_models()

    def _log_available_models(self) -> None:
        try:
            models = self.client.models.list()
            available = [m.name for m in models]
            sample = ", ".join(available[:5])
            self.logger.info("Gemini available models (sample): %s", sample or "none")
        except Exception as exc:  # noqa: BLE001
            self.logger.warning("Failed to list Gemini models: %s", exc)

    async def _resolve_model(self) -> str:
        if self._resolved_model:
            return self._resolved_model

        try:
            # async list; falls back to sync client on failure
            models_iter = self.async_client.models.list()
            models = [m async for m in models_iter]
        except Exception as exc:  # noqa: BLE001
            self.logger.warning(
                "Gemini 모델 목록 조회에 실패했습니다. 설정된 모델/선호도에 따라 계속 시도합니다: %s",
                exc,
            )
            self._resolved_model = self._ensure_model_path(self.model_name or self.model_preferences[0])
            return self._resolved_model

        supported = []
        for m in models:
            methods = getattr(m, "supported_generation_methods", None) or getattr(
                m, "generation_methods", None
            )
            if methods is None:
                # 관측치가 없으면 후보로 포함 (권한이 부족해도 리스트는 반환될 수 있음)
                supported.append(m.name)
            elif "generateContent" in methods:
                supported.append(m.name)

        if not supported:
            self.logger.warning(
                "generateContent 지원 여부를 확인할 수 없습니다. 설정된 모델/선호도로 계속 시도합니다."
            )
            self._resolved_model = self._ensure_model_path(self.model_name or self.model_preferences[0])
            return self._resolved_model

        # Preference order: explicit model name, then preference list, then first supported.
        candidates = [self.model_name] + list(self.model_preferences)
        normalized_supported = {self._normalize_model_name(n): n for n in supported}

        for cand in candidates:
            normalized = self._normalize_model_name(cand)
            exact = normalized_supported.get(normalized)
            if exact:
                self._resolved_model = self._ensure_model_path(exact)
                break
            # prefix match (e.g., prefer "gemini-2.5-flash" to match "models/gemini-2.5-flash")
            for key, original in normalized_supported.items():
                if key.startswith(normalized):
                    self._resolved_model = self._ensure_model_path(original)
                    break
            if self._resolved_model:
                break

        if not self._resolved_model:
            self._resolved_model = self._ensure_model_path(supported[0])
            self.logger.warning(
                "Preferred Gemini 모델을 찾지 못했습니다. 첫 번째 지원 모델(%s)로 대체합니다.",
                self._resolved_model,
            )
        else:
            self.logger.info("Using Gemini model: %s", self._resolved_model)

        return self._resolved_model

    def _normalize_model_name(self, name: str) -> str:
        return name.replace("models/", "").strip()

    def _ensure_model_path(self, name: str) -> str:
        return name if name.startswith("models/") else f"models/{name}"

    def _build_prompt(self, book_title: str | None, content: str, force_json_only: bool) -> str:
        title_part = book_title or "제목 미상"
        strict_suffix = (
            "\n반드시 JSON만 출력하세요. JSON 이외의 텍스트, 설명, 주석, 코드펜스를 절대 포함하지 마세요."
            if force_json_only
            else ""
        )
        return (
            "너는 독후감 검증 모델이다. 아래 독후감이 실제로 책을 읽고 작성된 내용인지, "
            "그리고 의미 없는 반복/도배/스팸이 아닌지를 판정한다.\n"
            "출력은 JSON 하나만 반환하고, status는 SUBMITTED 또는 REJECTED 중 하나다.\n"
            "SUBMITTED일 때 rejection_reason은 null, REJECTED일 때는 반드시 한국어 한 문장으로 쓴다.\n"
            "rejection_reason 규칙: 1문장만 허용(마침표 1개 이하), 25자 이내로 핵심만 요약, "
            "모호하거나 장문의 이유는 잘못된 응답이다.\n"
            '{"status": "SUBMITTED" | "REJECTED", "rejection_reason": string | null}\n'
            f"책 제목: {title_part}\n"
            f"독후감 내용:\n{content}\n"
            "- 책 제목과 내용이 일치하는지(책을 읽은 흔적, 인물, 줄거리/주제/사건/개념 언급, 인상 등) 확인한다.\n"
            "- 의미 없는 반복/무작위 문자열/광고 링크 등이 있으면 REJECTED로 한다.\n"
            "- 독후감으로 볼 수 있는 최소한의 길이와 서술이 있으면 SUBMITTED로 한다.\n"
            "JSON 포맷 외에 어떠한 서술도 추가하지 말 것. 응답은 JSON 한 덩어리만 제공할 것."
            f"{strict_suffix}"
        )

    def _parse_response_text(self, response_text: str) -> GeminiResult:
        cleaned = self._clean_response_text(response_text)
        try:
            payload = json.loads(cleaned)
        except json.JSONDecodeError as exc:
            status_value, rejection_value = self._extract_fields_with_regex(cleaned)
            if status_value:
                rejection_reason = None if status_value == "SUBMITTED" else rejection_value
                return GeminiResult(status=status_value, rejection_reason=rejection_reason)
            raise GeminiClientError(f"Gemini 응답 파싱 실패(JSON 아님): {cleaned[:200]}") from exc

        status = payload.get("status")
        if status not in {"SUBMITTED", "REJECTED"}:
            raise GeminiClientError("Gemini 응답 status 필드가 올바르지 않습니다.")

        rejection_reason = payload.get("rejection_reason")
        if status == "SUBMITTED":
            rejection_reason = None
        elif rejection_reason is None:
            rejection_reason = "사유가 제공되지 않았습니다."

        return GeminiResult(status=status, rejection_reason=rejection_reason)

    async def _generate(self, prompt: str, model_name: str):
        async for attempt in AsyncRetrying(
            reraise=True,
            stop=stop_after_attempt(2),
            wait=wait_fixed(1),
            retry=retry_if_exception(lambda exc: not isinstance(exc, TypeError)),
        ):
            with attempt:
                return await self.async_client.models.generate_content(
                    model=model_name,
                    contents=prompt,
                    config=genai_types.GenerateContentConfig(
                        max_output_tokens=self.max_output_tokens,
                        temperature=0.1,
                        response_mime_type="application/json",
                    ),
                )

    async def evaluate_book_report(self, book_title: str | None, content: str) -> GeminiResult:
        model_name = await self._resolve_model()
        prompt = self._build_prompt(book_title, content, force_json_only=False)
        last_error: Exception | None = None
        last_response_text: str | None = None

        for attempt in range(1, self.max_parse_attempts + 1):
            try:
                response = await self._generate(prompt, model_name)
                response_text = self._extract_text(response)
                last_response_text = response_text
                return self._parse_response_text(response_text)
            except NotFound as exc:
                # Model unexpectedly missing; do not keep retrying the same.
                last_error = exc
                self.logger.error("Gemini model not found: %s", model_name)
                break
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                self.logger.warning(
                    "Gemini parse/generation failed on attempt %s/%s with model=%s: %s",
                    attempt,
                    self.max_parse_attempts,
                    model_name,
                    exc,
                )
                prompt = self._build_prompt(book_title, content, force_json_only=True)

        suffix = f" | last_response={last_response_text[:200]!r}" if last_response_text else ""
        raise GeminiClientError(f"Gemini 응답 처리에 실패했습니다.{suffix}") from last_error

    def _extract_text(self, response) -> str:
        if getattr(response, "text", None):
            return response.text
        candidates = getattr(response, "candidates", None)
        if candidates:
            parts = []
            for part in candidates[0].content.parts:
                text = getattr(part, "text", None)
                if text:
                    parts.append(text)
            if parts:
                return "\n".join(parts)
        return ""

    def _clean_response_text(self, text: str) -> str:
        cleaned = text.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.strip("`")
            cleaned = re.sub(r"^json", "", cleaned, flags=re.IGNORECASE).strip()
        json_candidate = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if json_candidate:
            return json_candidate.group(0)
        return cleaned

    def _extract_fields_with_regex(self, text: str) -> tuple[str | None, str | None]:
        status_match = re.search(r'"status"\s*:\s*"(?P<status>SUBMITTED|REJECTED)"', text)
        rejection_match = re.search(
            r'"rejection_reason"\s*:\s*(null|"(?P<reason>[^"]*)")',
            text,
        )
        status_value = status_match.group("status") if status_match else None
        rejection_value = None
        if rejection_match:
            if rejection_match.group(0).lower().endswith("null"):
                rejection_value = None
            else:
                rejection_value = rejection_match.group("reason")
        return status_value, rejection_value
