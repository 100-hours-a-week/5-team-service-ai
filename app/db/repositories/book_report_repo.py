from sqlalchemy.orm import Session

from app.db.models.book_report import BookReport


class BookReportRepository:
    def get_by_id(self, db: Session, report_id: int) -> BookReport | None:
        return db.get(BookReport, report_id)

    def list_by_meeting_round_id(
        self, db: Session, meeting_round_id: int
    ) -> list[BookReport]:
        """
        Fetch all book reports belonging to the given meeting round.
        """
        return (
            db.query(BookReport)
            .filter(BookReport.meeting_round_id == meeting_round_id)
            .order_by(BookReport.id.asc())
            .all()
        )
