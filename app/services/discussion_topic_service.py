from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import random
import re
from collections import deque
from pathlib import Path
from typing import Iterable

import faiss
import numpy as np

from app.clients.runpod_client import RunpodClient
from app.schemas.discussion_topic_schema import (
    DiscussionReport,
    DiscussionTopicGenerateResponse,
    DiscussionTopicSingle,
)
from app.services.embedder import Embedder

logger = logging.getLogger(__name__)

# Non-greedy JSON object finder
JSON_OBJ_RE = re.compile(r"\{.*?\}", re.DOTALL)
CODE_FENCE_RE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL)


class DiscussionTopicService:
    """Generate discussion topics from given reports via RunPod LLM with caching."""

    def __init__(
        self,
        runpod_client: RunpodClient,
        *,
        max_tokens: int = 512,
        temperature: float = 0.8,
        top_p: float = 0.9,
        top_k: int = 5,
        max_context_chars: int = 1200,
        index_path: str = "data/discussion_topics.faiss",
        meta_path: str = "data/discussion_topics_meta.json",
    ) -> None:
        self.runpod_client = runpod_client
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.max_context_chars = max_context_chars
        self.embedder = Embedder()
        self.index_path = Path(index_path)
        self.meta_path = Path(meta_path)
        self.index: faiss.Index | None = None
        self.metadatas: list[dict] = []
        self._load_index()
        self.topic_pool_size = 5
        self._topic_pools: dict[str, deque[str]] = {}
        self._locks: dict[str, asyncio.Lock] = {}

    # Public ---------------------------------------------------------------
    async def generate_topics(
        self,
        *,
        meeting_round_id: int,
        reports: Iterable[DiscussionReport],
    ) -> DiscussionTopicGenerateResponse:
        reports_list = list(reports or [])
        if not reports_list:
            raise ValueError("no book reports provided in request")

        contents = self._collect_contents(reports_list)
        cache_key = self._build_cache_key(meeting_round_id, reports_list)
        lock = self._locks.setdefault(cache_key, asyncio.Lock())

        async with lock:
            pool = self._topic_pools.get(cache_key)

            if not pool:
                pool = await self._generate_topic_pool(
                    meeting_round_id=meeting_round_id,
                    contents=contents,
                )
                self._topic_pools[cache_key] = pool

            topic = pool.popleft() if pool else self._fallback_topic(contents)

        return DiscussionTopicGenerateResponse(
            status="success",
            data=DiscussionTopicSingle(topic=topic),
        )

    # Helpers --------------------------------------------------------------
    def _collect_contents(self, reports) -> list[str]:
        contents = [r.content for r in reports if getattr(r, "content", None)]
        return [c[:800] for c in contents if c]

    def _build_prompt(
        self, meeting_round_id: int, *, topic_count: int, contents: list[str]
    ) -> str:
        context_lines = self._build_context(contents)
        context = "\n".join(f"- {c}" for c in context_lines)
        variation_token = random.randint(1, 1_000_000)
        return (
            "너는 한국어 토론 주제를 만드는 전문가다.\n"
            f"아래 독후감 내용을 보고 질문형 토론 주제 {topic_count}개를 JSON 배열로 만들어라.\n"
            "규칙: 질문형 문장, 중복/모호 표현 금지, 한국어로 작성. 이전과 다른 각도/주제를 사용해라.\n"
            f"반드시 배열 길이가 {topic_count}가 되도록 채워라(부족하면 추가 생성).\n"
            "반드시 JSON만 출력하고, 형식은 아래와 같다:\n"
            "[\n"
            '  { "topic": "질문 내용1" },\n'
            '  { "topic": "질문 내용2" },\n'
            "  ...\n"
            "]\n"
            f"meeting_round_id: {meeting_round_id}\n"
            f"독후감 개수: {len(contents)}\n"
            f"독후감 요약:\n{context}\n"
            f"variation_token: {variation_token}\n"
            "JSON 외 텍스트나 설명을 절대 넣지 말 것. 뉴스나 정답 같은 다른 내용을 쓰지 말 것."
        )

    def _extract_text(self, output) -> str:
        # Prefer tokens array if present
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
                        # also cut at first ``` if present
                        fence = text.find("```")
                        if fence != -1:
                            text = text[:fence]
                        return text
            for key in ("text", "output"):
                val = output.get(key)
                if isinstance(val, str):
                    return val
                if isinstance(val, dict):
                    inner = val.get("text") or val.get("output")
                    if isinstance(inner, str):
                        return inner
            if isinstance(choices, list) and choices:
                first = choices[0]
                if isinstance(first, dict):
                    message = first.get("message") or {}
                    content = message.get("content") if isinstance(message, dict) else None
                    if isinstance(content, str):
                        return content
                    for key in ("text", "content", "generated_text"):
                        val = first.get(key)
                        if isinstance(val, str):
                            return val
        if isinstance(output, list) and output:
            return self._extract_text(output[0])
        return str(output)

    def _parse_topics(self, raw_text: str, expected: int) -> list[str]:
        text = raw_text.strip()
        if text.startswith("```"):
            text = text.strip("`").strip()

        # 1) 코드펜스/배열 블록 우선 추출
        block = None
        m = CODE_FENCE_RE.search(text)
        if m:
            block = m.group(1)
        if not block and text.startswith("[") and text.endswith("]"):
            block = text
        if not block:
            m = JSON_OBJ_RE.search(text)
            block = m.group(0) if m else None

        topics: list[str] = []

        def _coerce_from_obj(obj) -> list[str]:
            if isinstance(obj, dict):
                if obj.get("topic"):
                    return [str(obj["topic"]).strip()]
                if obj.get("topics") and isinstance(obj["topics"], list):
                    return [
                        (str(it["topic"]).strip() if isinstance(it, dict) else str(it).strip())
                        for it in obj["topics"]
                        if (isinstance(it, dict) and it.get("topic")) or isinstance(it, str)
                    ]
            if isinstance(obj, list):
                out = []
                for it in obj:
                    if isinstance(it, dict) and it.get("topic"):
                        out.append(str(it["topic"]).strip())
                    elif isinstance(it, str):
                        out.append(it.strip())
                return out
            return []

        # 2) block이 있으면 먼저 시도
        if block:
            try:
                data = json.loads(block)
                topics.extend(_coerce_from_obj(data))
            except Exception:
                pass

        # 3) 여전히 부족하면 텍스트 내 모든 JSON 객체를 순회해 추출
        if len(topics) < expected:
            for m in JSON_OBJ_RE.finditer(text):
                try:
                    obj = json.loads(m.group(0))
                    topics.extend(_coerce_from_obj(obj))
                except Exception:
                    continue
                if len(topics) >= expected:
                    break

        # 4) 정리
        deduped = []
        seen = set()
        for t in topics:
            if not t:
                continue
            if t in seen:
                continue
            seen.add(t)
            deduped.append(t)
            if len(deduped) >= expected:
                break
        return deduped

    def _fallback_topic(self, contents: list[str]) -> str:
        if contents:
            return "독후감에서 가장 인상적인 장면이 던지는 질문은 무엇인가요?"
        return "이 작품이 오늘날 우리에게 던지는 핵심 질문은 무엇인가요?"

    # ------------- RAG helpers ---------------- #
    def _normalize_text(self, text: str) -> str:
        return " ".join(text.strip().split())

    def _build_context(self, contents: list[str]) -> list[str]:
        """
        Build prompt 컨텍스트.

        1) 요청에 들어온 리포트들에서 유사도 상위 문단 추출.
        2) FAISS 인덱스에 기존 데이터가 있으면 함께 검색해 RAG 컨텍스트로 섞음.
        3) 인덱스가 비어 있으면 기존 동작 그대로 (write-only) 진행.
        """

        if not contents:
            return []

        normalized = [self._normalize_text(c) for c in contents]
        vecs = self.embedder.encode(normalized)
        vecs = vecs / (np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-8)
        query_vec = np.mean(vecs, axis=0)
        query_vec = query_vec / (np.linalg.norm(query_vec) + 1e-8)

        # 1) 현재 요청 내에서 상위 문단
        sims = vecs @ query_vec
        top_idx = np.argsort(sims)[::-1][: self.top_k]
        current_hits = [contents[i][:300] for i in top_idx]

        # 2) 기존 FAISS 인덱스 RAG (있을 때만)
        retrieved_hits: list[str] = []
        if self.index is not None and getattr(self.index, "ntotal", 0) > 0:
            try:
                query = query_vec.astype(np.float32, copy=False)[None, :]
                k = min(self.top_k, self.index.ntotal)
                scores, idxs = self.index.search(query, k)
                for idx in idxs[0]:
                    if idx == -1:
                        continue
                    if idx < len(self.metadatas):
                        content = self.metadatas[idx].get("content")
                        if content:
                            retrieved_hits.append(content)
            except Exception as exc:  # pragma: no cover
                logger.warning("faiss search failed: %s", exc)

        # 3) 중복 제거 후 길이 제한 내에서 컨텍스트 구성
        context_lines: list[str] = []
        seen = set()
        for h in current_hits + retrieved_hits:
            if h in seen:
                continue
            seen.add(h)
            context_lines.append(h)
            if len("\n".join(context_lines)) > self.max_context_chars:
                break
        return context_lines

    def _load_index(self):
        if self.index_path.exists():
            self.index = faiss.read_index(str(self.index_path))
        else:
            self.index = None
        if self.meta_path.exists():
            try:
                self.metadatas = json.loads(self.meta_path.read_text())
            except Exception:
                self.metadatas = []
        else:
            self.metadatas = []

    def _save_index(self):
        if self.index is not None:
            self.index_path.parent.mkdir(parents=True, exist_ok=True)
            faiss.write_index(self.index, str(self.index_path))
        self.meta_path.parent.mkdir(parents=True, exist_ok=True)
        self.meta_path.write_text(json.dumps(self.metadatas, ensure_ascii=False))

    def _add_to_index(self, contents: list[str], meeting_round_id: int):
        if not contents:
            return
        normalized = [self._normalize_text(c) for c in contents]
        vecs = self.embedder.encode(normalized)
        faiss.normalize_L2(vecs)
        if self.index is None:
            dim = vecs.shape[1]
            self.index = faiss.IndexFlatIP(dim)
        self.index.add(vecs)
        for c in contents:
            self.metadatas.append(
                {"meeting_round_id": meeting_round_id, "content": c[:300]}
            )
        self._save_index()

    # ------------- Topic pool helpers ---------------- #
    def _build_cache_key(self, meeting_round_id: int, reports: list[DiscussionReport]) -> str:
        normalized = [
            {
                "id": getattr(r, "id", None),
                "content": (getattr(r, "content", "") or "")[:800],
            }
            for r in sorted(reports, key=lambda r: getattr(r, "id", 0))
        ]
        blob = json.dumps(normalized, ensure_ascii=False, separators=(",", ":"))
        sig = hashlib.sha256(blob.encode("utf-8")).hexdigest()
        return f"{meeting_round_id}:{sig}"

    async def _generate_topic_pool(
        self, *, meeting_round_id: int, contents: list[str]
    ) -> deque[str]:
        topics: list[str] = []
        attempts = 0
        while attempts < 2 and len(topics) < self.topic_pool_size:
            attempts += 1
            prompt = self._build_prompt(
                meeting_round_id, topic_count=self.topic_pool_size, contents=contents
            )
            logger.info(
                "RunPod discussion prompt (attempt %s, truncate 400): %s",
                attempts,
                prompt[:400],
            )

            payload = {
                "prompt": prompt,
                "sampling_params": {
                    "max_tokens": self.max_tokens,
                    "temperature": self.temperature,
                    "top_p": self.top_p,
                    "stop": ["```", "\n```", "\n\n\n", "#", "뉴스", "정답"],
                },
            }

            try:
                job_id, output = await asyncio.to_thread(
                    self.runpod_client.generate, payload
                )
                logger.info("RunPod discussion job completed: %s", job_id)
                raw_text = self._extract_text(output)
                logger.error("RAW OUTPUT >>> %s", raw_text)
                topics = self._parse_topics(raw_text, expected=self.topic_pool_size)
            except Exception as exc:  # noqa: BLE001
                logger.exception("discussion topic pool generation failed: %s", exc)
                topics = []

        if len(topics) < self.topic_pool_size:
            # 부족하면 기존 결과를 순환 복제하거나 전부 없으면 fallback으로 채움
            if topics:
                idx = 0
                while len(topics) < self.topic_pool_size:
                    topics.append(topics[idx % len(topics)])
                    idx += 1
            else:
                while len(topics) < self.topic_pool_size:
                    topics.append(self._fallback_topic(contents))

        # 풀 생성 시에만 인덱스 저장 1회
        try:
            self._add_to_index(contents, meeting_round_id)
        except Exception as exc:  # pragma: no cover
            logger.warning("faiss add failed: %s", exc)

        return deque(topics)
