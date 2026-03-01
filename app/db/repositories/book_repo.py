from __future__ import annotations

from typing import List, Optional

from sqlalchemy import or_
from sqlalchemy.orm import Session

from app.db.models.book import Book


class BookRepository:
    @staticmethod
    def search(
        db: Session, *, title: str | None, author: str | None, limit: int = 5
    ) -> List[Book]:
        query = db.query(Book)
        filters = []
        if title:
            filters.append(Book.title.ilike(f"%{title}%"))
        if author:
            filters.append(Book.authors.ilike(f"%{author}%"))
        if filters:
            query = query.filter(or_(*filters))
        return query.limit(limit).all()

    @staticmethod
    def search_one(
        db: Session, *, title: str | None, author: str | None
    ) -> Optional[Book]:
        rows = BookRepository.search(db, title=title, author=author, limit=1)
        return rows[0] if rows else None
