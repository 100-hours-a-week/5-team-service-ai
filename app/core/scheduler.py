"""Lightweight cron-based scheduler for weekly recommendation job."""

from __future__ import annotations

import logging

import pytz
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger

import time

from app.batch.weekly_batch import generate_from_db
from app.batch.reco_v2_batch import main as reco_v2_main
from app.core.config import get_settings

logger = logging.getLogger(__name__)


def _job(top_k: int, search_k: int, use_v2: bool) -> None:
    try:
        start_ts = time.perf_counter()
        logger.info(
            "reco batch start",
            extra={"top_k": top_k, "search_k": search_k, "use_v2": use_v2},
        )

        if use_v2:
            # Train + generate in one shot using the behavior-aware pipeline.
            reco_v2_main(["--train", "--generate", "--top-k", str(top_k), "--search-k", str(search_k)])
            elapsed_ms = int((time.perf_counter() - start_ts) * 1000)
            logger.info("reco v2 batch completed", extra={"elapsed_ms": elapsed_ms})
        else:
            result = generate_from_db(top_k=top_k, search_k=search_k, persist=True)
            elapsed_ms = int((time.perf_counter() - start_ts) * 1000)
            logger.info(
                "reco batch completed",
                extra={
                    "rows": result.get("row_count"),
                    "users": result.get("users"),
                    "inserted": result.get("inserted"),
                    "timings": result.get("timings"),
                    "elapsed_ms": elapsed_ms,
                },
            )
    except Exception as exc:  # noqa: BLE001
        logger.exception("reco batch failed: %s", exc)


def start_scheduler() -> BackgroundScheduler | None:
    settings = get_settings()
    if not settings.enable_reco_scheduler:
        logger.info("reco scheduler disabled via ENABLE_RECO_SCHEDULER")
        return None

    scheduler = BackgroundScheduler(
        timezone=pytz.timezone(settings.reco_scheduler_timezone)
    )
    trigger = CronTrigger.from_crontab(
        settings.reco_scheduler_cron, timezone=scheduler.timezone
    )

    job_kwargs = {
        "top_k": settings.reco_scheduler_top_k,
        "search_k": settings.reco_scheduler_search_k,
        "use_v2": settings.reco_scheduler_use_v2,
    }

    # Weekly cron schedule only.
    scheduler.add_job(
        _job,
        trigger,
        kwargs=job_kwargs,
        id="weekly_recommendations",
        replace_existing=True,
    )

    scheduler.start()
    logger.info(
        "reco scheduler started",
        extra={
            "cron": settings.reco_scheduler_cron,
            "tz": str(scheduler.timezone),
            "top_k": settings.reco_scheduler_top_k,
            "search_k": settings.reco_scheduler_search_k,
            "next_run": scheduler.get_job("weekly_recommendations").next_run_time,
        },
    )
    return scheduler


def shutdown_scheduler(scheduler: BackgroundScheduler | None) -> None:
    if scheduler and scheduler.running:
        scheduler.shutdown(wait=False)
