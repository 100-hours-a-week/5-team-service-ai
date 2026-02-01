import time
from collections import Counter
from pathlib import Path

from app.services.embedder import Embedder
from app.services.faiss_store import FaissStore
from app.services.recommender import build_meeting_text, build_user_query, load_jsonl

FIXTURES = Path(__file__).parent / "fixtures"


def test_fixture_counts():
    users = load_jsonl(FIXTURES / "users.jsonl")
    meetings = load_jsonl(FIXTURES / "meetings.jsonl")
    logs = load_jsonl(FIXTURES / "logs.jsonl")

    assert len(users) >= 20
    assert len(meetings) >= 100
    assert len(logs) >= 1000


def test_meeting_status_mix():
    meetings = load_jsonl(FIXTURES / "meetings.jsonl")
    statuses = Counter(m["status"] for m in meetings)

    assert "RECRUITING" in statuses
    assert "FINISHED" in statuses
    assert "CANCELED" in statuses
    recruiting_ratio = statuses["RECRUITING"] / sum(statuses.values())
    assert recruiting_ratio > 0.4  # recruiting meetings should be plentiful


def test_click_join_bias():
    logs = load_jsonl(FIXTURES / "logs.jsonl")
    interaction_counts = Counter()
    for row in logs:
        if row["event_type"] in {"click", "join"}:
            interaction_counts[row["meeting_id"]] += 1

    assert sum(interaction_counts.values()) > 0
    top_10 = sum(count for _, count in interaction_counts.most_common(10))
    share = top_10 / sum(interaction_counts.values())
    assert share >= 0.55  # interactions should be skewed to a subset of meetings


def test_dwell_optional_but_present():
    logs = load_jsonl(FIXTURES / "logs.jsonl")
    impressions_with_dwell = [row for row in logs if row["event_type"] == "impression" and "dwell_sec" in row]
    assert not impressions_with_dwell

    clicks_joins = [row for row in logs if row["event_type"] in {"click", "join"}]
    with_dwell = [row for row in clicks_joins if "dwell_sec" in row]
    assert clicks_joins, "click/join events should exist"
    assert with_dwell, "some interactions should carry dwell time"
    assert len(with_dwell) / len(clicks_joins) >= 0.4


def test_semantic_retrieval_top20():
    meetings = load_jsonl(FIXTURES / "meetings.jsonl")
    users = load_jsonl(FIXTURES / "users.jsonl")

    embedder = Embedder()

    meeting_texts = [build_meeting_text(m) for m in meetings]
    t0 = time.perf_counter()
    meeting_vecs = embedder.encode(meeting_texts)
    t1 = time.perf_counter()

    store = FaissStore()
    store.build(meeting_vecs, [{"meeting_id": m["id"], "status": m["status"]} for m in meetings])

    user_queries = [build_user_query(u) for u in users]
    t2 = time.perf_counter()
    user_vecs = embedder.encode(user_queries)
    t3 = time.perf_counter()

    search_start = time.perf_counter()
    recruiting_hits_total = 0
    for vec in user_vecs:
        hits = store.search(vec, top_k=20)
        assert len(hits) <= 20
        recruiting_hits = [h for h in hits if store.get_metadata(h["meeting_id"]).get("status") == "RECRUITING"]
        recruiting_hits_total += len(recruiting_hits)
    search_elapsed = time.perf_counter() - search_start

    print(f"meeting_embed={t1 - t0:.2f}s user_embed={t3 - t2:.2f}s search={search_elapsed:.2f}s")
    assert recruiting_hits_total > 0
