import logging

from dotenv import load_dotenv
from fastapi import FastAPI

from app.routers.book_report_validation_router import router as book_report_validation_router
from app.api.routes.recommendation import router as recommendation_router
from app.core.scheduler import shutdown_scheduler, start_scheduler

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)

app = FastAPI(title="Book Report Validation API")


_scheduler = None


@app.get("/health", tags=["health"])
def health_check():
    return {"status": "ok"}


app.include_router(book_report_validation_router)
app.include_router(recommendation_router)


@app.on_event("startup")
def _start_scheduler():
    global _scheduler
    _scheduler = start_scheduler()


@app.on_event("shutdown")
def _stop_scheduler():
    shutdown_scheduler(_scheduler)
