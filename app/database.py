"""
PostgreSQL persistence layer.

Stores two things (per project scope):
1. assessments -- history of every risk assessment run through the API
2. usability_reviews -- Objective 5 reviewer ratings (clarity/relevance/practicality)

Point DATABASE_URL at your Postgres instance (Railway's Postgres plugin sets
this automatically once the service is attached). Nothing else needs to
change -- every function here has the same name and signature as the old
SQLite version, so main.py does not need any changes beyond what already
calls these functions.
"""

import os
import json
import psycopg2
import psycopg2.extras
from contextlib import contextmanager

DATABASE_URL = os.environ.get("DATABASE_URL")


def init_db():
    if not DATABASE_URL:
        raise RuntimeError(
            "DATABASE_URL is not set. Attach a PostgreSQL database to this "
            "service (see README/deployment notes) and make sure the "
            "DATABASE_URL environment variable is available to it."
        )
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS assessments (
                    id SERIAL PRIMARY KEY,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    name TEXT,
                    profile_json TEXT NOT NULL,
                    preferences_json TEXT,
                    risk_probability REAL NOT NULL,
                    risk_label INTEGER NOT NULL,
                    top_factors_json TEXT NOT NULL,
                    guidance_json TEXT NOT NULL
                )
            """)
            # Safe to run every startup: Postgres no-ops if the column
            # already exists, unlike SQLite which raises.
            cur.execute("ALTER TABLE assessments ADD COLUMN IF NOT EXISTS name TEXT")
            cur.execute("""
                CREATE TABLE IF NOT EXISTS usability_reviews (
                    id SERIAL PRIMARY KEY,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    scenario_label TEXT NOT NULL,
                    clarity_rating INTEGER NOT NULL,
                    relevance_rating INTEGER NOT NULL,
                    practicality_rating INTEGER NOT NULL,
                    comments TEXT
                )
            """)
        conn.commit()


@contextmanager
def get_conn():
    conn = psycopg2.connect(DATABASE_URL, cursor_factory=psycopg2.extras.RealDictCursor)
    try:
        yield conn
    finally:
        conn.close()


def save_assessment(profile: dict, preferences: dict, risk_probability: float,
                     risk_label: int, top_factors: list, guidance: list,
                     name: str = None) -> int:
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """INSERT INTO assessments
                   (name, profile_json, preferences_json, risk_probability, risk_label, top_factors_json, guidance_json)
                   VALUES (%s, %s, %s, %s, %s, %s, %s)
                   RETURNING id""",
                (name, json.dumps(profile), json.dumps(preferences), risk_probability,
                 risk_label, json.dumps(top_factors), json.dumps(guidance)),
            )
            new_id = cur.fetchone()["id"]
        conn.commit()
        return new_id


def list_assessments(limit: int = 50) -> list:
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT * FROM assessments ORDER BY id DESC LIMIT %s", (limit,)
            )
            rows = cur.fetchall()
            return [dict(r) for r in rows]


def get_assessment(assessment_id: int):
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT * FROM assessments WHERE id = %s", (assessment_id,)
            )
            row = cur.fetchone()
            return dict(row) if row else None


def save_usability_review(scenario_label: str, clarity: int, relevance: int,
                           practicality: int, comments: str = "") -> int:
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """INSERT INTO usability_reviews
                   (scenario_label, clarity_rating, relevance_rating, practicality_rating, comments)
                   VALUES (%s, %s, %s, %s, %s)
                   RETURNING id""",
                (scenario_label, clarity, relevance, practicality, comments),
            )
            new_id = cur.fetchone()["id"]
        conn.commit()
        return new_id


def list_usability_reviews() -> list:
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT * FROM usability_reviews ORDER BY id DESC")
            rows = cur.fetchall()
            return [dict(r) for r in rows]


def usability_summary() -> dict:
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """SELECT COUNT(*) as n,
                          AVG(clarity_rating) as avg_clarity,
                          AVG(relevance_rating) as avg_relevance,
                          AVG(practicality_rating) as avg_practicality
                   FROM usability_reviews"""
            )
            row = cur.fetchone()
            return dict(row)
