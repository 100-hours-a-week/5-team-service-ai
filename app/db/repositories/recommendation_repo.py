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

    def iter_users(self, db: Session, *, chunk_size: int = 500):
        """
        Stream users in chunks to reduce peak memory.

        Uses LIMIT/OFFSET paging; acceptable for batch workloads and avoids
        materializing the full result set.
        """

        base_sql = text(
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
            LIMIT :limit OFFSET :offset
            """
        )

        offset = 0
        while True:
            rows = (
                db.execute(base_sql, {"limit": chunk_size, "offset": offset})
                .mappings()
                .all()
            )
            if not rows:
                break
            yield [self._convert_json_fields(row) for row in rows]
            offset += chunk_size

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
                m.leader_intro,
                m.leader_user_id
            FROM meetings m
            JOIN reading_genres rg ON rg.id = m.reading_genre_id
            WHERE m.deleted_at IS NULL
              AND m.status = 'RECRUITING'
            """
        )
        rows = db.execute(sql).mappings().all()
        return [dict(row) for row in rows]

    def fetch_user(self, db: Session, user_id: int) -> dict | None:
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
            WHERE u.deleted_at IS NULL AND u.id = :user_id
            LIMIT 1
            """
        )
        row = db.execute(sql, {"user_id": user_id}).mappings().first()
        if not row:
            return None
        return self._convert_json_fields(row)

    def upsert_recommendations(self, db: Session, rows: Iterable[dict]) -> int:
        """
        Persist recommendation rows. Uses a unique key to avoid duplicates per week.
        """
        rows = list(rows)
        if not rows:
            return 0

        # NOTE: DB는 (user_id, meeting_id, week_start_date) 유니크 키를 가져야 주차별 적재가 정상 동작한다.
        # 만약 week_start_date가 유니크 키에 포함되지 않으면 과거 주차 행이 업데이트되어 날짜가 뒤틀린다.
        sql = text(
            """
            INSERT INTO user_meeting_recommendations (user_id, meeting_id, week_start_date, `rank`)
            VALUES (:user_id, :meeting_id, :week_start_date, :rank)
            ON DUPLICATE KEY UPDATE
                week_start_date = VALUES(week_start_date),
                `rank` = VALUES(`rank`),
                created_at = CURRENT_TIMESTAMP;
            """
        )
        result = db.execute(sql, rows)
        db.commit()
        return result.rowcount

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
