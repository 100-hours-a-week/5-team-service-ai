from sqlalchemy import Column, Integer, String, Text, Date, DateTime

from app.db.session import Base


class Book(Base):
    __tablename__ = "books"

    id = Column(Integer, primary_key=True, index=True)
    isbn13 = Column(String(13), nullable=True)
    title = Column(String(255), nullable=False, index=True)
    authors = Column(String(255), nullable=True, index=True)
    publisher = Column(String(255), nullable=True)
    thumbnail_url = Column(String(512), nullable=True)
    published_at = Column(Date, nullable=True)
    summary = Column(Text, nullable=True)
    deleted_at = Column(DateTime, nullable=True)
    created_at = Column(DateTime, nullable=False)
    updated_at = Column(DateTime, nullable=False)
