from sqlalchemy import text
from sqlalchemy.orm import Session


class BookRepository:
    """
    단순 Book 조회용 리포지토리.
    """

    def get_by_title_author(self, db: Session, title: str, author: str) -> dict | None:
        """
        Try both `author` and `authors` columns; fall back to title-only.
        This tolerates schema differences without raising.
        """

        def _run(sql: str, params: dict):
            return db.execute(text(sql), params).mappings().first()

        # 1) author 컬럼 존재 가정
        try:
            row = _run(
                """
                SELECT id, title, author, summary
                FROM books
                WHERE title = :title AND author = :author
                LIMIT 1
                """,
                {"title": title, "author": author},
            )
            if row:
                return dict(row)
        except Exception as exc:  # noqa: BLE001
            if "Unknown column 'author'" not in str(exc) and "no such column: author" not in str(exc):
                raise

        # 2) authors 컬럼 존재 가정
        try:
            row = _run(
                """
                SELECT id, title, authors AS author, summary
                FROM books
                WHERE title = :title AND authors = :author
                LIMIT 1
                """,
                {"title": title, "author": author},
            )
            if row:
                return dict(row)
        except Exception as exc:  # noqa: BLE001
            if "Unknown column 'authors'" not in str(exc) and "no such column: authors" not in str(exc):
                raise

        # 3) title-only fallback (coalesce author/authors)
        try:
            row = _run(
                """
                SELECT id, title, COALESCE(author, authors) AS author, summary
                FROM books
                WHERE title = :title
                LIMIT 1
                """,
                {"title": title},
            )
        except Exception:
            row = None

        if not row:
            return None

        data = dict(row)
        data.setdefault("author", author)
        return data

    def fetch_all_with_summary(self, db: Session) -> list[dict]:
        """
        Fetch all books that have a non-empty summary.
        Returns list of dicts with id, title, authors, summary.
        """
        sql = text(
            """
            SELECT id, title, authors, summary
            FROM books
            WHERE summary IS NOT NULL AND summary != ''
            """
        )
        rows = db.execute(sql).mappings().all()
        return [dict(r) for r in rows]
