from __future__ import annotations

import json
from typing import Iterable, Mapping

from sqlalchemy import text
from sqlalchemy.orm import Session


class RecommendationRepo:
    """
    Repository for recommendation inputs and persistence.
    Assumes MySQL tables with code columns already populated.
    """

    def fetch_users(self, db: Session) -> list[dict]:
        sql = text(
            """
            SELECT
                u.id AS user_id,
                rv.code AS reading_volume_code,
                (
                    SELECT JSON_ARRAYAGG(rp.code)
                    FROM user_reading_purposes urp
                    JOIN reading_purposes rp ON urp.reading_purpose_id = rp.id
                    WHERE urp.user_id = u.id
                ) AS purpose_codes,
                (
                    SELECT JSON_ARRAYAGG(rg.code)
                    FROM user_reading_genres urg
                    JOIN reading_genres rg ON urg.reading_genre_id = rg.id
                    WHERE urg.user_id = u.id
                ) AS genre_codes
            FROM users u
            LEFT JOIN user_preferences up ON up.user_id = u.id
            LEFT JOIN reading_volumes rv ON rv.id = up.reading_volume_id
            WHERE u.deleted_at IS NULL
            """
        )
        rows = db.execute(sql).mappings().all()
        return [self._convert_json_fields(row) for row in rows]

    def fetch_meetings(self, db: Session) -> list[dict]:
        sql = text(
            """
            SELECT
                m.id,
                rg.code AS reading_genre_code,
                m.title,
                m.description,
                m.status,
                m.capacity,
                m.current_count,
                m.leader_intro
            FROM meeting m
            JOIN reading_genres rg ON rg.id = m.reading_genre_id
            WHERE m.deleted_at IS NULL
            """
        )
        rows = db.execute(sql).mappings().all()
        return [dict(row) for row in rows]

    def upsert_recommendations(self, db: Session, rows: Iterable[dict]) -> int:
        """
        Persist recommendation rows. Uses a unique key to avoid duplicates per week.
        """
        rows = list(rows)
        if not rows:
            return 0

        sql = text(
            """
            INSERT INTO user_meeting_recommendation (user_id, meeting_id, week_start_date, `rank`)
            VALUES (:user_id, :meeting_id, :week_start_date, :rank)
            ON DUPLICATE KEY UPDATE
                `rank` = VALUES(`rank`),
                created_at = CURRENT_TIMESTAMP(6);
            """
        )
        db.execute(sql, rows)
        db.commit()
        return len(rows)

    @staticmethod
    def _convert_json_fields(row: Mapping) -> dict:
        """
        Parse JSON/text fields into Python lists if needed.
        """
        def _parse(value):
            if value is None:
                return []
            if isinstance(value, (list, tuple)):
                return list(value)
            if isinstance(value, str):
                try:
                    parsed = json.loads(value)
                    if isinstance(parsed, list):
                        return parsed
                except json.JSONDecodeError:
                    pass
                return [value]
            return [value]

        data = dict(row)
        data["purpose_codes"] = _parse(data.get("purpose_codes"))
        data["genre_codes"] = _parse(data.get("genre_codes"))
        return data
