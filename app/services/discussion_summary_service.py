from __future__ import annotations

import json
import logging
import re
import threading
from typing import Iterable, List, Dict, Any, AnyStr

from app.clients.runpod_client import RunpodClient
from app.schemas.discussion_summary_schema import (
    DiscussionMessage,
    DiscussionSummary,
    DiscussionSummaryResponse,
)

logger = logging.getLogger(__name__)

JSON_OBJ_RE = re.compile(r"\{.*\}", re.DOTALL)
CODE_FENCE_RE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL)
FENCE_ONLY_RE = re.compile(r"```.*?```", re.DOTALL)


class DiscussionSummaryService:
    """Round-based discussion summarization via RunPod LLM."""

    # 프로세스 내 캐시: room_id -> {"messages": list[DiscussionMessage], "summaries": dict[int, DiscussionSummary]}
    _cache: Dict[int, Dict[str, Any]] = {}
    _lock = threading.Lock()

    def __init__(
        self,
        runpod_client: RunpodClient,
        *,
        max_tokens: int = 900,
        temperature: float = 0.6,
        top_p: float = 0.9,
        max_context_chars: int = 2400,
    ) -> None:
        self.runpod_client = runpod_client
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.max_context_chars = max_context_chars

    def summarize(
        self,
        *,
        room_id: int,
        topic: str,
        round_no: int | str,
        messages: Iterable[DiscussionMessage],
    ) -> DiscussionSummaryResponse:
        msg_list = [m for m in messages if getattr(m, "text_message", None)]
        if not msg_list:
            raise ValueError("messages must contain at least one item")

        round_no_int = self._coerce_round(round_no)

        # --- 캐시 기반 누적 ---
        state = self._get_room_state(room_id)
        if round_no_int == 1 or not state["messages"]:
            state["messages"] = list(msg_list)
        else:
            state["messages"].extend(msg_list)

        context = self._build_context(state["messages"])
        prompt = self._build_prompt(topic=topic, round_no=round_no_int, context=context)
        logger.info("RunPod discussion-summary prompt (truncate 400): %s", prompt[:400])

        payload = {
            "prompt": prompt,
            "sampling_params": {
                "max_tokens": self.max_tokens,
                "temperature": self.temperature,
                "top_p": self.top_p,
                "stop": ["```", "\n```", "\n\n\n", "#"],
            },
        }

        job_id = None
        try:
            job_id, output = self.runpod_client.generate(payload)
            logger.info("RunPod discussion-summary job completed: %s", job_id)
            raw_text = self._extract_text(output)
            summary = self._parse_summary(
                raw_text,
                output_snapshot=self._truncate(json.dumps(output, ensure_ascii=False), 400, 200),
                room_id=room_id,
                round_no=round_no_int,
                job_id=job_id,
            )
            if not summary:
                logger.warning(
                    "LLM summary parse returned None; using empty summary room=%s round=%s job=%s",
                    room_id,
                    round_no_int,
                    job_id,
                )
                summary = self._empty_summary()
        except TimeoutError as exc:
            logger.warning("RunPod timeout for room_id=%s round=%s: %s", room_id, round_no_int, exc)
            raise
        except Exception:
            # 상위 라우터에서 503 처리
            raise

        state["summaries"][round_no_int] = summary

        status = "final" if round_no_int >= 3 else "pending"
        response = DiscussionSummaryResponse(status=status, summary=summary)

        # 라운드 3 이상이면 캐시 정리
        if round_no_int >= 3:
            self._clear_room_state(room_id)

        return response

    # Helpers --------------------------------------------------------------
    def _coerce_round(self, round_no: int | str) -> int:
        try:
            return max(1, min(3, int(round_no)))
        except Exception:  # noqa: BLE001
            return 1

    def _build_context(self, messages: List[DiscussionMessage]) -> str:
        lines: list[str] = []
        total_len = 0
        for msg in messages:
            line = f"- {msg.sender_nickname or '익명'}: {msg.text_message}".strip()
            total_len += len(line)
            if total_len > self.max_context_chars:
                break
            lines.append(line)
        return "\n".join(lines)

    def _build_prompt(self, *, topic: str, round_no: int, context: str) -> str:
        topic_part = topic.strip() or "토론 주제 미상"
        stage = "중간 요약" if round_no < 3 else "최종 요약"
        return (
            "너는 한국어 토론 요약 어시스턴트다.\n"
            "다음 채팅 메시지를 보고 긍정/우려, 핵심 쟁점, 미해결 쟁점을 간결히 정리해라.\n"
            "항상 JSON 한 덩어리만 반환하고, 아래 스키마를 지켜라:\n"
            "{\n"
            '  "summary": {\n'
            '    "pro": ["항목1", "항목2", ...],\n'
            '    "con": ["항목1", ...],\n'
            '    "main_issues": ["쟁점1", ...],\n'
            '    "unresolved_issues": ["미해결1", ...]\n'
            "  }\n"
            "}\n"
            f"라운드: {round_no}/3 ({stage}), 방 ID: {topic_part}\n"
            f"대화 로그:\n{context}\n"
            "규칙: JSON 외 텍스트/설명/코드펜스 금지. 리스트는 한국어 한 문장씩 1~3개로 요약.\n"
            "무의미한 내용이면 간결한 안전 문구로 정리하고, 비어 있는 리스트는 []로 둔다."
        )

    def _extract_text(self, output) -> str:
        if isinstance(output, dict):
            choices = output.get("choices")
            if isinstance(choices, list) and choices:
                first = choices[0]
                if isinstance(first, dict):
                    tokens = first.get("tokens")
                    if isinstance(tokens, list) and tokens:
                        text = "".join(tokens)
                        cut = text.find("}`")
                        if cut != -1:
                            text = text[: cut + 1]
                        fence = text.find("```")
                        if fence != -1:
                            text = text[:fence]
                        return text
                message = first.get("message") if isinstance(first, dict) else None
                if isinstance(message, dict):
                    content = message.get("content")
                    if isinstance(content, str):
                        return content
            for key in ("text", "output"):
                val = output.get(key)
                if isinstance(val, str):
                    return val
                if isinstance(val, dict):
                    inner = val.get("text") or val.get("output")
                    if isinstance(inner, str):
                        return inner
        if isinstance(output, list) and output:
            return self._extract_text(output[0])
        return str(output)

    def _parse_summary(self, raw_text: str) -> DiscussionSummary | None:
        text = raw_text.strip()
        if text.startswith("```"):
            text = text.strip("`").strip()

        block = None
        m = CODE_FENCE_RE.search(text)
        if m:
            block = m.group(1)
        if not block:
            m = JSON_OBJ_RE.search(text)
            block = m.group(0) if m else None
        if not block:
            return None

        try:
            data = json.loads(block)
        except Exception:
            return None

        summary = data.get("summary") if isinstance(data, dict) else None
        if not isinstance(summary, dict):
            return None

        def _as_list(key: str, *fallback_keys: str) -> list[str]:
            for k in (key, *fallback_keys):
                items = summary.get(k, [])
                if isinstance(items, list):
                    return [str(i).strip() for i in items if str(i).strip()]
                if isinstance(items, str):
                    return [items.strip()] if items.strip() else []
                if items is None:
                    continue
                # 기타 타입은 문자열화
                text = str(items).strip()
                if text:
                    return [text]
            return []

        return DiscussionSummary(
            pro=_as_list("pro"),
            con=_as_list("con"),
            main_issues=_as_list("main_issues", "issues"),
            unresolved_issues=_as_list("unresolved_issues", "open_questions"),
        )

    # Cache utils ----------------------------------------------------------
    def _get_room_state(self, room_id: int) -> Dict[str, Any]:
        with self._lock:
            if room_id not in self._cache:
                self._cache[room_id] = {"messages": [], "summaries": {}}
            return self._cache[room_id]

    def _clear_room_state(self, room_id: int) -> None:
        with self._lock:
            self._cache.pop(room_id, None)

    # -------- Robust text extraction & parsing helpers -------- #
    def _extract_text(self, output) -> str:
        """
        RunPod/vLLM response -> raw text.
        우선순위:
        1) choices[0].message.content
        2) choices[0].text
        3) choices[0].tokens (join)
        Fallback: str(output)
        """
        try:
            if isinstance(output, list) and output:
                output = output[0]

            if isinstance(output, dict):
                choices = output.get("choices")
                if isinstance(choices, list) and choices:
                    first = choices[0]
                    if isinstance(first, dict):
                        message = first.get("message")
                        if isinstance(message, dict):
                            content = message.get("content")
                            if isinstance(content, str) and content.strip():
                                return content
                        text = first.get("text")
                        if isinstance(text, str) and text.strip():
                            return text
                        tokens = first.get("tokens")
                        if isinstance(tokens, list) and tokens:
                            joined = "".join(str(t) for t in tokens)
                            if joined.strip():
                                return joined
                # openai-ish direct fields
                for key in ("text", "output"):
                    val = output.get(key)
                    if isinstance(val, str) and val.strip():
                        return val
            return str(output)
        except Exception as exc:  # pragma: no cover
            logger.warning("LLM text extraction failed; returning str(output): %s", exc)
            return str(output)

    def _normalize_to_json_candidate(self, text: str) -> str | None:
        """코드펜스/앞뒤 설명을 걷어내고 JSON 블록만 추출."""
        if not text:
            return None
        cleaned = text.strip()
        # 코드펜스 제거
        m = CODE_FENCE_RE.search(cleaned)
        if m:
            cleaned = m.group(1)
        else:
            cleaned = FENCE_ONLY_RE.sub("", cleaned).strip()

        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start == -1 or end == -1 or end <= start:
            return None
        return cleaned[start : end + 1]

    def _parse_summary(
        self,
        raw_text: str,
        *,
        output_snapshot: str,
        room_id: int,
        round_no: int,
        job_id: str | None,
    ) -> DiscussionSummary | None:
        candidate = self._normalize_to_json_candidate(raw_text)
        if not candidate:
            logger.warning(
                "summary parse failed (no braces) room=%s round=%s job=%s raw_head=%s raw_tail=%s output_head=%s output_tail=%s",
                room_id,
                round_no,
                job_id,
                self._truncate(raw_text, 200, 80),
                self._truncate(raw_text, 80, 200),
                self._truncate(output_snapshot, 200, 80),
                self._truncate(output_snapshot, 80, 200),
            )
            return None

        data = None
        try:
            data = json.loads(candidate)
        except json.JSONDecodeError as exc:
            # LLM이 여러 JSON 블록을 붙여 보낼 때 `Extra data`가 발생할 수 있으므로 첫 객체만 파싱 시도
            if exc.msg == "Extra data":
                try:
                    decoder = json.JSONDecoder()
                    data, _ = decoder.raw_decode(candidate)
                except Exception:
                    data = None
            if data is None:
                try:
                    # 후보가 단일따옴표 기반 파이썬 dict 표현일 수 있으므로 보정 시도
                    import ast

                    data = ast.literal_eval(candidate)
                except Exception:
                    logger.warning(
                        "summary json decode error room=%s round=%s job=%s pos=%s msg=%s candidate_head=%s candidate_tail=%s",
                        room_id,
                        round_no,
                        job_id,
                        exc.pos,
                        exc.msg,
                        self._truncate(candidate, 200, 80),
                        self._truncate(candidate, 80, 200),
                        exc_info=True,
                    )
                    return None
        except Exception:  # pragma: no cover
            logger.exception(
                "summary json decode unknown error room=%s round=%s job=%s candidate_head=%s",
                room_id,
                round_no,
                job_id,
                self._truncate(candidate, 200, 80),
            )
            return None

        try:
            summary = data.get("summary") if isinstance(data, dict) else None
            if not isinstance(summary, dict):
                logger.warning(
                    "summary key missing room=%s round=%s job=%s candidate_head=%s",
                    room_id,
                    round_no,
                    job_id,
                    self._truncate(candidate, 200, 80),
                )
                summary = {}

            def _as_list(key: str, *fallback_keys: str) -> list[str]:
                for k in (key, *fallback_keys):
                    items = summary.get(k, [])
                    if isinstance(items, list):
                        return [str(i).strip() for i in items if str(i).strip()]
                    if isinstance(items, str):
                        return [items.strip()] if items.strip() else []
                    if items is None:
                        continue
                    text = str(items).strip()
                    if text:
                        return [text]
                return []

            return DiscussionSummary(
                pro=_as_list("pro"),
                con=_as_list("con"),
                main_issues=_as_list("main_issues", "issues"),
                unresolved_issues=_as_list("unresolved_issues", "open_questions"),
            )
        except Exception:  # pragma: no cover
            logger.exception(
                "summary coercion failed room=%s round=%s job=%s candidate_head=%s",
                room_id,
                round_no,
                job_id,
                self._truncate(candidate, 200, 80),
            )
            return None

    @staticmethod
    def _empty_summary() -> DiscussionSummary:
        return DiscussionSummary(
            pro=[],
            con=[],
            main_issues=[],
            unresolved_issues=[],
        )

    @staticmethod
    def _truncate(text: AnyStr, head: int, tail: int) -> str:
        if text is None:
            return ""
        s = str(text)
        if len(s) <= head + tail:
            return s
        return s[:head] + " ... " + s[-tail:]
