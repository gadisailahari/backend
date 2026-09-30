"""
SQLite persistence layer.

Stores two things (per project scope):
1. assessments -- history of every risk assessment run through the API
2. usability_reviews -- Objective 5 reviewer ratings (clarity/relevance/practicality)

On Railway, point DB_PATH at a mounted persistent volume (e.g. /data/app.db)
via the DB_PATH environment variable so data survives redeploys.
"""

import sqlite3
import os
import json
from contextlib import contextmanager

DB_PATH = os.environ.get("DB_PATH", "app.db")


def init_db():
    with get_conn() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS assessments (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                profile_json TEXT NOT NULL,
                preferences_json TEXT,
                risk_probability REAL NOT NULL,
                risk_label INTEGER NOT NULL,
                top_factors_json TEXT NOT NULL,
                guidance_json TEXT NOT NULL
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS usability_reviews (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
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
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
    finally:
        conn.close()


def save_assessment(profile: dict, preferences: dict, risk_probability: float,
                     risk_label: int, top_factors: list, guidance: list) -> int:
    with get_conn() as conn:
        cur = conn.execute(
            """INSERT INTO assessments
               (profile_json, preferences_json, risk_probability, risk_label, top_factors_json, guidance_json)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (json.dumps(profile), json.dumps(preferences), risk_probability,
             risk_label, json.dumps(top_factors), json.dumps(guidance)),
        )
        conn.commit()
        return cur.lastrowid


def list_assessments(limit: int = 50) -> list:
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT * FROM assessments ORDER BY id DESC LIMIT ?", (limit,)
        ).fetchall()
        return [dict(r) for r in rows]


def get_assessment(assessment_id: int):
    with get_conn() as conn:
        row = conn.execute(
            "SELECT * FROM assessments WHERE id = ?", (assessment_id,)
        ).fetchone()
        return dict(row) if row else None


def save_usability_review(scenario_label: str, clarity: int, relevance: int,
                           practicality: int, comments: str = "") -> int:
    with get_conn() as conn:
        cur = conn.execute(
            """INSERT INTO usability_reviews
               (scenario_label, clarity_rating, relevance_rating, practicality_rating, comments)
               VALUES (?, ?, ?, ?, ?)""",
            (scenario_label, clarity, relevance, practicality, comments),
        )
        conn.commit()
        return cur.lastrowid


def list_usability_reviews() -> list:
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT * FROM usability_reviews ORDER BY id DESC"
        ).fetchall()
        return [dict(r) for r in rows]


def usability_summary() -> dict:
    with get_conn() as conn:
        row = conn.execute(
            """SELECT COUNT(*) as n,
                      AVG(clarity_rating) as avg_clarity,
                      AVG(relevance_rating) as avg_relevance,
                      AVG(practicality_rating) as avg_practicality
               FROM usability_reviews"""
        ).fetchone()
        return dict(row)
