"""Lightweight cron-based scheduler for weekly recommendation job."""

from __future__ import annotations

import logging
from datetime import datetime

import pytz
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger

from app.batch.weekly_batch import generate_from_db
from app.core.config import get_settings

logger = logging.getLogger(__name__)


def _job(top_k: int, search_k: int) -> None:
    try:
        result = generate_from_db(top_k=top_k, search_k=search_k, persist=True)
        logger.info(
            "reco batch completed",
            extra={
                "rows": len(result.get("rows", [])),
                "users": result.get("users"),
                "inserted": result.get("inserted"),
                "timings": result.get("timings"),
            },
        )
    except Exception as exc:  # noqa: BLE001
        logger.exception("reco batch failed: %s", exc)


def start_scheduler() -> BackgroundScheduler | None:
    settings = get_settings()
    if not settings.enable_reco_scheduler:
        logger.info("reco scheduler disabled via ENABLE_RECO_SCHEDULER")
        return None

    scheduler = BackgroundScheduler(timezone=pytz.timezone(settings.reco_scheduler_timezone))
    trigger = CronTrigger.from_crontab(settings.reco_scheduler_cron, timezone=scheduler.timezone)

    job_kwargs = {
        "top_k": settings.reco_scheduler_top_k,
        "search_k": settings.reco_scheduler_search_k,
    }

    # Run once on startup (in the scheduler thread) to avoid waiting until the next cron tick.
    bootstrap_time = datetime.now(tz=scheduler.timezone)
    scheduler.add_job(
        _job,
        "date",
        run_date=bootstrap_time,
        kwargs=job_kwargs,
        id="weekly_recommendations_bootstrap",
        replace_existing=True,
    )

    # Keep the existing weekly cron schedule.
    scheduler.add_job(
        _job,
        trigger,
        kwargs=job_kwargs,
        id="weekly_recommendations",
        replace_existing=True,
    )

    scheduler.start()
    next_run_job = scheduler.get_job("weekly_recommendations")
    logger.info(
        "reco scheduler started",
        extra={
            "cron": settings.reco_scheduler_cron,
            "tz": str(scheduler.timezone),
            "top_k": settings.reco_scheduler_top_k,
            "search_k": settings.reco_scheduler_search_k,
            "next_run": next_run_job.next_run_time if next_run_job else None,
            "bootstrap_run": bootstrap_time,
        },
    )
    return scheduler


def shutdown_scheduler(scheduler: BackgroundScheduler | None) -> None:
    if scheduler and scheduler.running:
        scheduler.shutdown(wait=False)
