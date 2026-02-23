from sqlalchemy import Column, Integer, String, Text

from app.db.session import Base


class BookReport(Base):
    __tablename__ = "book_report"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, nullable=False)
    meeting_session_id = Column(Integer, nullable=False)
    # NOTE: 실제 MySQL 테이블에 존재하는 회차 식별자. 기존 컬럼은 그대로 두고 함께 유지.
    meeting_round_id = Column(Integer, nullable=True)
    content = Column(Text, nullable=False)
    book_title = Column(String(255), nullable=True)
