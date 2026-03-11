"""
LLM 기반 퀴즈 생성 서비스.

요약:
- MySQL `books` 테이블에서 제목/저자 기반으로 책 요약을 찾는다.
- 요약을 문장 단위로 쪼개 sentence-transformer 임베딩을 계산하고,
  제목+저자 쿼리 벡터와 내적이 높은 문장 Top-K만 추린다.
- 추린 문장을 컨텍스트로 RunPod 서버리스 LLM에 프롬프트를 보내
  퀴즈 JSON을 생성한다.
- LLM 출력이 어긋나더라도 JSON 블록을 파싱/정규화해
  `QuizGenerateResponse` 스키마로 반환한다.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
import hashlib
from random import randint, sample
from typing import Sequence
import ast

import numpy as np
from sqlalchemy.orm import Session

from app.clients.runpod_client import RunpodClient
from app.core.metrics import (
    observe_quiz_cache_error,
    observe_quiz_cache_hit,
    observe_quiz_cache_miss,
)
from app.core.redis_client import RedisClient
from app.db.repositories.book_repo import BookRepository
from app.schemas.quiz_schema import Quiz, QuizChoice, QuizGenerateResponse
from app.services.embedder import Embedder

logger = logging.getLogger(__name__)

JSON_BLOCK_RE = re.compile(r"\{.*\}", re.DOTALL)


def _split_sentences(text: str) -> list[str]:
    """
    Rough sentence splitter for Korean/English text.
    """
    if not text:
        return []
    # Split on sentence enders or newlines.
    parts = re.split(r"(?<=[.?!。])\s+|(?<=[다])\s+|\n+", text)
    return [p.strip() for p in parts if p.strip()]


@dataclass
class QuizService:
    runpod_client: RunpodClient
    embedder: Embedder
    redis_client: RedisClient | None = None
    cache_ttl_seconds: int = 604_800
    cache_key_version: str = "v1"
    top_k_sentences: int = 5
    max_tokens: int = 512  # vLLM sampling_params.max_tokens에 전달
    temperature: float = 0.5  # 0.2   # 노이즈 줄이기 위해 낮춤
    top_p: float = 1.0  # 0.9

    def generate(
        self, *, title: str, author: str, db: Session
    ) -> QuizGenerateResponse:
        book = BookRepository.search_one(db, title=title, author=author)
        if not book:
            raise ValueError("book not found")

        # 세션을 길게 붙잡지 않도록 필요한 데이터만 복사해 둔 뒤 세션 정리
        summary = book.summary or ""
        book_id = getattr(book, "id", None)
        title_val = book.title
        author_val = book.authors or author
        cache_key = (
            self._make_cache_key(book_id=book_id, summary=summary)
            if book_id is not None
            else None
        )
        cached_resp = self._cache_get(cache_key) if cache_key else None
        if cached_resp:
            return cached_resp
        try:
            db.expunge(book)  # 세션에서 분리해도 메모리 데이터는 유지
        except Exception:
            pass
        try:
            db.close()  # RunPod 호출 전 커넥션 반환
        except Exception:
            pass

        sentences = self._select_sentences(
            summary=summary, title=title_val, author=author_val
        )
        prompt = self._build_prompt(
            title=title_val, author=author_val, sentences=sentences
        )
        logger.info("RunPod prompt (truncated 400 chars): %s", prompt[:400])

        payload = {
            "prompt": prompt,
            "sampling_params": {
                "max_tokens": self.max_tokens,
                "temperature": self.temperature,
                "top_p": self.top_p,
                "seed": randint(1, 1_000_000),  # 문제 바꾸기 위해 추가
            },
        }

        job_id, output = self.runpod_client.generate(payload)
        logger.info("RunPod job completed: %s", job_id)
        try:
            logger.info(
                "RunPod output (json, truncated 800 chars): %s",
                json.dumps(output)[:800],
            )
        except Exception:  # pragma: no cover - best effort
            pass

        raw_text = self._extract_text(output)
        logger.debug("LLM raw output (truncated 400 chars): %s", raw_text[:400])
        llm_success = False
        try:
            quiz_response = self._parse_quiz_response(raw_text)
            llm_success = True
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "LLM parse failed (%s); falling back to deterministic quiz", exc
            )
            quiz_response = self._fallback_quiz(title=title, author=author)

        if llm_success and cache_key and not self._is_placeholder(quiz_response):
            self._cache_set(cache_key, quiz_response)
        return quiz_response

    # ---------------- internal helpers ---------------- #

    def _make_cache_key(self, *, book_id: int, summary: str) -> str:
        summary_hash = hashlib.sha256((summary or "").strip().encode("utf-8")).hexdigest()
        return f"quiz:{self.cache_key_version}:{book_id}:{summary_hash}"

    def _cache_get(self, cache_key: str) -> QuizGenerateResponse | None:
        if not self.redis_client:
            return None
        try:
            cached = self.redis_client.get_json(cache_key)
            if cached is None:
                observe_quiz_cache_miss()
                return None
            observe_quiz_cache_hit()
            return QuizGenerateResponse.model_validate(cached)
        except Exception as exc:  # noqa: BLE001
            observe_quiz_cache_error()
            logger.warning("Quiz cache get failed for key %s: %s", cache_key, exc)
            return None

    def _cache_set(self, cache_key: str, response: QuizGenerateResponse) -> None:
        if not self.redis_client:
            return
        try:
            payload = response.model_dump()
            self.redis_client.set_json(
                cache_key,
                payload,
                ttl_seconds=self.cache_ttl_seconds,
            )
        except Exception as exc:  # noqa: BLE001
            observe_quiz_cache_error()
            logger.warning("Quiz cache set failed for key %s: %s", cache_key, exc)

    def _select_sentences(self, summary: str, title: str, author: str) -> Sequence[str]:
        """
        Pick quiz-worthy sentences using embedding similarity to (title, author).
        """
        sentences = _split_sentences(summary)
        if not sentences:
            return [summary or f"{title} - {author}"]

        try:
            sent_embeddings = self.embedder.encode(sentences)
            query_vec = self.embedder.encode(
                [f"{title} {author} 주요 내용 퀴즈 포인트"]
            )[0]
            scores = np.dot(sent_embeddings, query_vec)
            top_idx = np.argsort(scores)[::-1][: self.top_k_sentences]
            selected = [sentences[i] for i in top_idx if i < len(sentences)]
            return [s for s in selected if s]
        except Exception as exc:  # noqa: BLE001
            logger.warning("Fallback to raw summary because embedding failed: %s", exc)
            return sentences[: self.top_k_sentences]

    def _build_prompt(
        self, *, title: str, author: str, sentences: Sequence[str]
    ) -> str:
        context = "\n".join(f"- {s}" for s in sentences)
        return (
            "너는 한국어 객관식 퀴즈를 한 문제만 생성하는 출제자다.\n"
            "아래 책 정보와 컨텍스트 문장만 사용해 문제를 만들고, 반드시 JSON 한 덩어리만 출력해라.\n"
            "코드펜스(````), 주석, 설명, 앞뒤 문구 금지. 무조건 순수 JSON 문자열만 반환.\n"
            "JSON은 꼭 더블쿼트만 사용하고, 트레일링 콤마 금지. 보기 4개 고정, 한 문장씩.\n"
            "질문은 세 문장, 각 보기는 한 문장으로 간결하게 써라.\n"
            "출력 형식 예시:\n"
            "{\n"
            '  "quiz": {"question": "질문", "correct_choice_number": 2},\n'
            '  "quiz_choices": [\n'
            '    {"choice_number": 1, "choice_text": "보기1"},\n'
            '    {"choice_number": 2, "choice_text": "보기2"},\n'
            '    {"choice_number": 3, "choice_text": "보기3"},\n'
            '    {"choice_number": 4, "choice_text": "보기4"}\n'
            "  ]\n"
            "}\n"
            "규칙: JSON 외 설명 금지, 보기 4개, 보기 번호는 1~4, "
            "correct_choice_number는 정답 보기 번호.\n"
            f"책 제목: {title}\n"
            f"저자: {author}\n"
            f"컨텍스트:\n{context}\n"
            "위 정보를 참고해 한 문제만 만들어라."
        )

    def _extract_text(self, output) -> str:
        """
        Try multiple known fields from RunPod/vLLM style responses.
        """
        # Top-level list (e.g., [{"choices": [...]}])
        if isinstance(output, list) and output:
            first = output[0]
            if isinstance(first, dict):
                # reuse dict handling below
                return self._extract_text(first)
            try:
                return json.dumps(output)
            except Exception:
                return str(output)

        if isinstance(output, dict):
            # direct text
            direct = output.get("text")
            if isinstance(direct, str):
                return direct

            # nested output
            inner = output.get("output")
            if isinstance(inner, str):
                return inner
            if isinstance(inner, dict):
                txt = inner.get("text") or inner.get("output")
                if isinstance(txt, str):
                    return txt
                # OpenAI/vLLM style
                choices = inner.get("choices")
                if isinstance(choices, list) and choices:
                    msg = choices[0]
                    # message.content or text/content
                    content = (
                        (msg.get("message") or {}).get("content")
                        if isinstance(msg, dict)
                        else None
                    )
                    if isinstance(content, str):
                        return content
                    for key in ("text", "content", "generated_text"):
                        val = msg.get(key) if isinstance(msg, dict) else None
                        if isinstance(val, str):
                            return val

            # OpenAI-like top-level choices
            choices = output.get("choices")
            if isinstance(choices, list) and choices:
                msg = choices[0]
                # tokens array (RunPod)
                tokens = msg.get("tokens") if isinstance(msg, dict) else None
                if isinstance(tokens, list):
                    token_text = "".join(tokens)
                    if token_text.strip():
                        return token_text

                content = (
                    (msg.get("message") or {}).get("content")
                    if isinstance(msg, dict)
                    else None
                )
                if isinstance(content, str):
                    return content
                for key in ("text", "content", "generated_text"):
                    val = msg.get(key) if isinstance(msg, dict) else None
                    if isinstance(val, str):
                        return val

            # fallback: stringify dict
            try:
                return json.dumps(output)
            except Exception:
                return str(output)
        return str(output)

    def _extract_json_block(self, text: str) -> str | None:
        match = JSON_BLOCK_RE.search(text)
        return match.group(0) if match else None

    def _parse_quiz_response(self, raw_text: str) -> QuizGenerateResponse:
        text = raw_text or ""
        json_block = self._extract_json_block(text) or text.strip()
        if not json_block:
            raise ValueError("LLM output did not contain JSON")

        try:
            payload = json.loads(json_block)
        except json.JSONDecodeError as exc:
            # try to parse the first JSON object only
            try:
                decoder = json.JSONDecoder()
                payload, _ = decoder.raw_decode(json_block)
            except Exception:
                # 관찰되는 패턴: 키/문자열에 작은따옴표 사용. literal_eval로 보정 시도.
                try:
                    payload = ast.literal_eval(json_block)
                except Exception as inner:  # noqa: BLE001
                    raise ValueError(f"Failed to parse LLM JSON: {inner}") from inner

        normalized = self._normalize_payload(payload)
        response = QuizGenerateResponse(**normalized)
        return response

    def _normalize_payload(self, payload: dict) -> dict:
        quiz = payload.get("quiz") or {}
        choices = payload.get("quiz_choices") or []

        # question이 없으면 기본 질문 생성
        if not quiz.get("question"):
            quiz["question"] = "다음 중 정답을 고르세요."

        normalized_choices: list[dict] = []
        for idx, choice in enumerate(choices, start=1):
            choice = dict(choice)
            choice.setdefault("choice_number", idx)
            if not choice.get("choice_text"):
                choice["choice_text"] = f"보기 {idx}"
            normalized_choices.append(choice)

        # 보기가 4개 미만이면 자리 채우기
        for idx in range(len(normalized_choices) + 1, 5):
            normalized_choices.append(
                {"choice_number": idx, "choice_text": f"보기 {idx}"}
            )

        # correct_choice_number 보정
        if "correct_choice_number" not in quiz or not (
            1 <= int(quiz["correct_choice_number"]) <= 4
        ):
            quiz["correct_choice_number"] = 1

        return {"quiz": quiz, "quiz_choices": normalized_choices[:4]}

    def _fallback_quiz(
        self, *, title: str, author: str
    ) -> QuizGenerateResponse:
        """
        Deterministic backup when LLM output is unusable.
        """
        question = f"'{title}'의 저자는 누구인가?"

        distractor_pool = [
            "빅터 위고",
            "제인 오스틴",
            "어니스트 헤밍웨이",
            "가브리엘 가르시아 마르케스",
            "알베르 카뮈",
            "아서 코난 도일",
            "헤르만 헤세",
            "표도르 도스토옙스키",
            "레이 브래드버리",
            "하퍼 리",
        ]
        distractors = sample([d for d in distractor_pool if d != author], 3)

        correct_slot = randint(1, 4)
        choices: list[QuizChoice] = []
        dist_iter = iter(distractors)
        for idx in range(1, 5):
            text = author if idx == correct_slot else next(dist_iter)
            choices.append(
                QuizChoice(choice_number=idx, choice_text=text)
            )

        quiz = Quiz(
            question=question, correct_choice_number=correct_slot
        )
        return QuizGenerateResponse(quiz=quiz, quiz_choices=choices)

    # placeholder 감지는 완화: 질문이 비어 있고 보기도 전부 '보기 '로만 시작할 때만 fallback.
    def _is_placeholder(self, resp: QuizGenerateResponse) -> bool:
        q = (resp.quiz.question or "").strip()
        all_placeholder_choices = resp.quiz_choices and all(
            (c.choice_text or "").strip().startswith("보기 ") for c in resp.quiz_choices
        )
        return (not q or q == "다음 중 정답을 고르세요.") and all_placeholder_choices
