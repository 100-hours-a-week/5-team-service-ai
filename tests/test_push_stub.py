import json
import threading
from datetime import date, timedelta
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from app.clients.spring_client import post_recommendations
from app.services.recommender import load_jsonl

FIXTURES = Path(__file__).parent / "fixtures"


def week_start() -> str:
    today = date.today()
    monday = today - timedelta(days=today.weekday())
    return monday.isoformat()


def start_mock_server():
    received: dict = {}

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):  # noqa: N802
            length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(length)
            try:
                parsed = json.loads(body.decode())
            except Exception:  # noqa: BLE001
                parsed = {}
            received["path"] = self.path
            received["body"] = parsed
            self.send_response(202)
            self.end_headers()
            self.wfile.write(b"ok")

        def log_message(self, format, *args):  # noqa: A003
            return

    server = ThreadingHTTPServer(("localhost", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, thread, received


def test_post_recommendations_payload_shape():
    users = load_jsonl(FIXTURES / "users.jsonl")
    meetings = load_jsonl(FIXTURES / "meetings.jsonl")
    recruiting_ids = [m["id"] for m in meetings if m["status"] == "RECRUITING"][:4]
    assert len(recruiting_ids) >= 4

    rows = []
    for user in users:
        for rank, mid in enumerate(recruiting_ids, start=1):
            rows.append(
                {
                    "user_id": user["user_id"],
                    "meeting_id": mid,
                    "week_start_date": week_start(),
                    "rank": rank,
                }
            )

    server, thread, received = start_mock_server()
    base_url = f"http://{server.server_address[0]}:{server.server_address[1]}"
    try:
        resp = post_recommendations(base_url, rows, timeout=5)
    finally:
        server.shutdown()
        thread.join()

    assert resp["status_code"] == 202
    assert received.get("path") == "/ai/recommendations"

    body = received.get("body", {})
    assert "rows" in body
    assert len(body["rows"]) == len(users) * 4
    for row in body["rows"]:
        assert set(row.keys()) == {"user_id", "meeting_id", "week_start_date", "rank"}
        assert row["meeting_id"] in recruiting_ids
