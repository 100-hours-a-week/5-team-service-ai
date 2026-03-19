from __future__ import annotations

from typing import List, Optional

from sqlalchemy.orm import Session

from app.db.models.book import Book


class BookRepository:
    @staticmethod
    def search(
        db: Session, *, title: str | None, author: str | None, limit: int = 5
    ) -> List[Book]:
        query = db.query(Book)

        # When both title and author are provided we should match **both** to
        # avoid returning a different book by the same author (e.g., 요청이
        # '싯다르타'인데 '데미안'이 반환되는 문제). Only fall back to single-field
        # filtering when one side is missing.
        if title and author:
            query = query.filter(
                Book.title.ilike(f"%{title}%"),
                Book.authors.ilike(f"%{author}%"),
            )
        elif title:
            query = query.filter(Book.title.ilike(f"%{title}%"))
        elif author:
            query = query.filter(Book.authors.ilike(f"%{author}%"))

        return query.limit(limit).all()

    @staticmethod
    def search_one(
        db: Session, *, title: str | None, author: str | None
    ) -> Optional[Book]:
        rows = BookRepository.search(db, title=title, author=author, limit=1)
        return rows[0] if rows else None
