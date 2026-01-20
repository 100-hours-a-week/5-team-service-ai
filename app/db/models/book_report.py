from sqlalchemy import Column, Integer, String, Text

from app.db.session import Base


class BookReport(Base):
    __tablename__ = "book_report"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, nullable=False)
    meeting_session_id = Column(Integer, nullable=False)
    content = Column(Text, nullable=False)
    book_title = Column(String(255), nullable=True)
