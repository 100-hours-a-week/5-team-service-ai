import logging

from fastapi import Depends, Header, HTTPException, status

from app.core.config import Settings, get_settings

logger = logging.getLogger(__name__)


def require_api_key(
    x_api_key: str | None = Header(None),
    settings: Settings = Depends(get_settings),
):
    """
    Simple header-based API key guard.

    Rejects the request with 401 if `x-api-key` is missing or does not match
    the configured value (defaults to "ai").
    """
    if x_api_key != settings.api_key:
        logger.warning(
            "API key mismatch provided_len=%s expected_len=%s",
            len(x_api_key) if x_api_key is not None else 0,
            len(settings.api_key),
        )
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="invalid api key")

    return True
