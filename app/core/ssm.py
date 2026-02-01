"""
AWS Systems Manager Parameter Store loader.

prod 환경에서만 동작하며, Parameter Store 값을 os.environ에 주입하여
기존 Pydantic Settings가 그대로 읽을 수 있도록 합니다.
"""

import logging
import os

logger = logging.getLogger(__name__)

# SSM 파라미터 이름 → 환경변수 이름 매핑
_PARAM_MAP: dict[str, str] = {
    "AI_DB_URL": "DB_URL",
    "AI_API_KEY": "API_KEY",
    "GEMINI_API_KEY": "GEMINI_API_KEY",
    "GEMINI_MODEL": "GEMINI_MODEL",
    "ENABLE_RECO_SCHEDULER": "ENABLE_RECO_SCHEDULER",
    "RECO_SCHEDULER_CRON": "RECO_SCHEDULER_CRON",
    "RECO_SCHEDULER_TZ": "RECO_SCHEDULER_TZ",
    "RECO_SCHEDULER_TOP_K": "RECO_SCHEDULER_TOP_K",
    "RECO_SCHEDULER_SEARCH_K": "RECO_SCHEDULER_SEARCH_K",
}


def load_ssm_parameters() -> None:
    """Parameter Store에서 값을 읽어 os.environ에 주입한다.

    USE_PARAMETER_STORE 환경변수가 "true"일 때만 실행된다.
    SPRING_PROFILES_ACTIVE 값으로 prefix를 결정한다.
    """
    if os.getenv("USE_PARAMETER_STORE", "").lower() != "true":
        logger.info("USE_PARAMETER_STORE is not set; skipping SSM loading")
        return

    try:
        import boto3
    except ImportError:
        logger.warning("boto3 is not installed; skipping SSM loading")
        return

    env = os.getenv("SPRING_PROFILES_ACTIVE", "dev")
    prefix = f"/doktori/{env}"
    client = boto3.client("ssm", region_name="ap-northeast-2")

    logger.info("Loading parameters from SSM prefix=%s", prefix)

    loaded = 0
    for ssm_key, env_key in _PARAM_MAP.items():
        name = f"{prefix}/{ssm_key}"
        try:
            resp = client.get_parameter(Name=name, WithDecryption=True)
            value = resp["Parameter"]["Value"]
            os.environ[env_key] = value
            loaded += 1
        except client.exceptions.ParameterNotFound:
            logger.debug("SSM parameter not found: %s (skipped)", name)
        except Exception:
            logger.warning("Failed to get SSM parameter: %s", name, exc_info=True)

    logger.info("Loaded %d/%d parameters from SSM", loaded, len(_PARAM_MAP))
