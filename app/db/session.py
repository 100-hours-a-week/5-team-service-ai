from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, sessionmaker

from app.core.config import get_settings


class Base(DeclarativeBase):
    pass


settings = get_settings()
engine = create_engine(
    settings.db_url,
    pool_pre_ping=True,
    future=True,
    pool_size=10,       # 기본 5 → 10으로 확대
    max_overflow=20,    # 추가 버퍼
    pool_timeout=30,    # 대기 시간
    pool_recycle=1800,  # 30분마다 재연결해 연결 고갈 방지
)
SessionLocal = sessionmaker(
    bind=engine, autoflush=False, autocommit=False, expire_on_commit=False, future=True
)


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
