from sqlalchemy.orm import Session

from app.db.models.book_report import BookReport


class BookReportRepository:
    def get_by_id(self, db: Session, report_id: int) -> BookReport | None:
        return db.get(BookReport, report_id)
