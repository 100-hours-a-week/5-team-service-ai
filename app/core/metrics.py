from __future__ import annotations

import time
from threading import Lock
from typing import Any

from fastapi import Request, Response
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
)

AI_REQUEST_TOTAL = Counter(
    "ai_request_total",
    "AI API request count",
    labelnames=("endpoint", "status_class"),
)

AI_REQUEST_DURATION_SECONDS = Histogram(
    "ai_request_duration_seconds",
    "AI API request latency",
    labelnames=("endpoint",),
    buckets=(0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10, 20),
)

AI_EXTERNAL_CALL_TOTAL = Counter(
    "ai_external_call_total",
    "External dependency call count",
    labelnames=("provider", "model", "result"),
)

AI_EXTERNAL_CALL_DURATION_SECONDS = Histogram(
    "ai_external_call_duration_seconds",
    "External dependency call latency",
    labelnames=("provider", "model"),
    buckets=(0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 30),
)

AI_EXTERNAL_RETRY_TOTAL = Counter(
    "ai_external_retry_total",
    "External dependency retry count",
    labelnames=("provider", "reason"),
)

AI_COLD_START_EVENT_TOTAL = Counter(
    "ai_cold_start_event_total",
    "Cold-start hint event count",
    labelnames=("type",),
)

AI_MODEL_LOAD_DURATION_SECONDS = Histogram(
    "ai_model_load_duration_seconds",
    "Model load latency",
    labelnames=("stage",),
    buckets=(0.1, 0.5, 1, 2, 5, 10, 30, 60, 120),
)

AI_MODEL_LOAD_FAIL_TOTAL = Counter(
    "ai_model_load_fail_total",
    "Model load failure count",
    labelnames=("stage",),
)

AI_FIRST_REQUEST_AFTER_BOOT_DURATION_SECONDS = Histogram(
    "ai_first_request_after_boot_duration_seconds",
    "First request latency after boot",
    labelnames=("endpoint",),
    buckets=(0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10, 20),
)

_first_request_observed: set[str] = set()
_first_request_lock = Lock()
_provider_cost_total: dict[str, float] = {}
_provider_success_count: dict[str, int] = {}
_provider_cost_lock = Lock()

AI_ESTIMATED_COST_INDEX_TOTAL = Counter(
    "ai_estimated_cost_index_total",
    "Estimated relative cost index total",
    labelnames=("provider",),
)

AI_COST_INDEX_PER_SUCCESS_REQUEST = Gauge(
    "ai_cost_index_per_success_request",
    "Estimated cost index per successful request",
    labelnames=("provider",),
)


def _endpoint_template(request: Request) -> str:
    route: Any = request.scope.get("route")
    if route is None:
        return "__unmatched__"
    return getattr(route, "path_format", None) or getattr(route, "path", "__unknown__")


def _status_class(status_code: int) -> str:
    if 400 <= status_code < 500:
        return "4xx"
    if status_code >= 500:
        return "5xx"
    return "2xx"


async def instrument_http_requests(request: Request, call_next) -> Response:
    started_at = time.perf_counter()
    status_code = 500
    try:
        response = await call_next(request)
        status_code = response.status_code
        return response
    finally:
        endpoint = _endpoint_template(request)
        if endpoint != "/metrics":
            _observe_first_request_after_boot(endpoint, time.perf_counter() - started_at)
            AI_REQUEST_TOTAL.labels(
                endpoint=endpoint, status_class=_status_class(status_code)
            ).inc()
            AI_REQUEST_DURATION_SECONDS.labels(endpoint=endpoint).observe(
                time.perf_counter() - started_at
            )


def metrics_response() -> Response:
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


def observe_external_call(
    provider: str, model: str, result: str, elapsed_seconds: float
) -> None:
    AI_EXTERNAL_CALL_TOTAL.labels(
        provider=provider, model=model, result=result
    ).inc()
    AI_EXTERNAL_CALL_DURATION_SECONDS.labels(provider=provider, model=model).observe(
        elapsed_seconds
    )


def observe_external_retry(provider: str, reason: str) -> None:
    AI_EXTERNAL_RETRY_TOTAL.labels(provider=provider, reason=reason).inc()


def observe_cold_start_event(event_type: str) -> None:
    AI_COLD_START_EVENT_TOTAL.labels(type=event_type).inc()


def observe_model_load(stage: str, elapsed_seconds: float) -> None:
    AI_MODEL_LOAD_DURATION_SECONDS.labels(stage=stage).observe(elapsed_seconds)


def observe_model_load_fail(stage: str) -> None:
    AI_MODEL_LOAD_FAIL_TOTAL.labels(stage=stage).inc()


def _observe_first_request_after_boot(endpoint: str, elapsed_seconds: float) -> None:
    with _first_request_lock:
        if endpoint in _first_request_observed:
            return
        _first_request_observed.add(endpoint)
    AI_FIRST_REQUEST_AFTER_BOOT_DURATION_SECONDS.labels(endpoint=endpoint).observe(
        elapsed_seconds
    )


def observe_estimated_cost_index(
    provider: str, delta_index: float, success_request: bool = False
) -> None:
    if delta_index <= 0:
        return

    AI_ESTIMATED_COST_INDEX_TOTAL.labels(provider=provider).inc(delta_index)

    with _provider_cost_lock:
        _provider_cost_total[provider] = (
            _provider_cost_total.get(provider, 0.0) + delta_index
        )
        if success_request:
            _provider_success_count[provider] = (
                _provider_success_count.get(provider, 0) + 1
            )

        success_count = _provider_success_count.get(provider, 0)
        if success_count > 0:
            ratio = _provider_cost_total[provider] / success_count
            AI_COST_INDEX_PER_SUCCESS_REQUEST.labels(provider=provider).set(ratio)
