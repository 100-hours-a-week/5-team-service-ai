import logging

# ✅ 1. 가장 먼저 SSM 로드!
from app.core.ssm import load_ssm_parameters

load_ssm_parameters()

# ✅ 2. dotenv
from dotenv import load_dotenv

load_dotenv()

# ✅ 3. FastAPI
from fastapi import FastAPI, Response

# ✅ 4. 이제 router import (Settings가 이미 준비됨)
from app.routers.book_report_validation_router import (
    router as book_report_validation_router,
)
from app.api.routes.discussion_topics import router as discussion_topic_router
from app.api.routes.discussion_summary import router as discussion_summary_router
from app.api.routes.recommendation import router as recommendation_router
from app.api.routes.quiz import router as quiz_router
from app.core.metrics import instrument_http_requests, metrics_response
from app.core.scheduler import shutdown_scheduler, start_scheduler

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)

# Nginx serves the app under /ai, so set root_path to keep docs/OpenAPI paths correct
app = FastAPI(title="Book Report Validation API", root_path="/ai")
app.middleware("http")(instrument_http_requests)

_scheduler = None


@app.get("/health", tags=["health"])
def health_check():
    return {"status": "ok"}


@app.get("/metrics", tags=["observability"])
def metrics() -> Response:
    return metrics_response()


app.include_router(book_report_validation_router)
app.include_router(discussion_topic_router)
app.include_router(discussion_summary_router)
app.include_router(recommendation_router)
app.include_router(quiz_router)


@app.on_event("startup")
def _start_scheduler():
    global _scheduler
    _scheduler = start_scheduler()


@app.on_event("shutdown")
def _stop_scheduler():
    shutdown_scheduler(_scheduler)
