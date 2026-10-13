"""Postgres tables for plans, feedback, clinician review and the 3-system rating study."""
import json
from app.database import get_conn


def init_tables():
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("""CREATE TABLE IF NOT EXISTS plans (
                id SERIAL PRIMARY KEY, created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                assessment_id INTEGER, parent_plan_id INTEGER, system TEXT DEFAULT 'rag_checker',
                status TEXT, structured_json TEXT, plan_json TEXT, text TEXT, totals_json TEXT, sources_json TEXT,
                attempts INTEGER, review_status TEXT DEFAULT 'pending', reviewed_text TEXT, reviewer_note TEXT,
                reviewed_at TIMESTAMP)""")
            cur.execute("""CREATE TABLE IF NOT EXISTS plan_feedback (
                id SERIAL PRIMARY KEY, created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP, plan_id INTEGER,
                kind TEXT, message TEXT, result_status TEXT, new_plan_id INTEGER,
                clarity INTEGER, relevance INTEGER, practicality INTEGER, helpful BOOLEAN)""")
            cur.execute("""CREATE TABLE IF NOT EXISTS study_items (
                id SERIAL PRIMARY KEY, created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP, plan_code TEXT UNIQUE,
                scenario_label TEXT, person_summary TEXT, system TEXT, text TEXT, totals_json TEXT,
                within_sodium_target BOOLEAN, status TEXT, attempts INTEGER)""")
            cur.execute("""CREATE TABLE IF NOT EXISTS plan_ratings (
                id SERIAL PRIMARY KEY, created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP, rater_id TEXT, plan_code TEXT,
                clarity INTEGER, relevance INTEGER, practicality INTEGER, adherence INTEGER, safety INTEGER, comments TEXT,
                UNIQUE (rater_id, plan_code))""")
        conn.commit()


def _one(cur, sql, args):
    cur.execute(sql, args); r = cur.fetchone(); return dict(r) if r else None


def save_plan(assessment_id, parent_plan_id, system, status, structured, plan, text, totals, sources, attempts):
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("""INSERT INTO plans (assessment_id,parent_plan_id,system,status,structured_json,plan_json,text,totals_json,sources_json,attempts)
                           VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s) RETURNING id""",
                        (assessment_id, parent_plan_id, system, status, json.dumps(structured), json.dumps(plan), text,
                         json.dumps(totals), json.dumps(sources), attempts))
            pid = cur.fetchone()["id"]
        conn.commit(); return pid


def get_plan(pid):
    with get_conn() as conn:
        with conn.cursor() as cur: return _one(cur, "SELECT * FROM plans WHERE id=%s", (pid,))


def list_pending(limit=50):
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT id,created_at,assessment_id,status,text,review_status FROM plans WHERE review_status='pending' AND status='approved_by_checker' ORDER BY id DESC LIMIT %s", (limit,))
            return [dict(r) for r in cur.fetchall()]


def set_review(pid, action, text, note):
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("UPDATE plans SET review_status=%s, reviewed_text=%s, reviewer_note=%s, reviewed_at=CURRENT_TIMESTAMP WHERE id=%s RETURNING id",
                        (action, text, note, pid))
            ok = cur.fetchone() is not None
        conn.commit(); return ok


def save_feedback(plan_id, kind, message=None, result_status=None, new_plan_id=None, clarity=None, relevance=None, practicality=None, helpful=None):
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("""INSERT INTO plan_feedback (plan_id,kind,message,result_status,new_plan_id,clarity,relevance,practicality,helpful)
                           VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s) RETURNING id""",
                        (plan_id, kind, message, result_status, new_plan_id, clarity, relevance, practicality, helpful))
            fid = cur.fetchone()["id"]
        conn.commit(); return fid


def save_study_item(code, label, person, system, text, totals, within, status, attempts):
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("""INSERT INTO study_items (plan_code,scenario_label,person_summary,system,text,totals_json,within_sodium_target,status,attempts)
                           VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s) ON CONFLICT (plan_code) DO NOTHING""",
                        (code, label, person, system, text, json.dumps(totals), within, status, attempts))
        conn.commit()


def clear_study_items():
    with get_conn() as conn:
        with conn.cursor() as cur: cur.execute("DELETE FROM study_items")
        conn.commit()


def list_study_items():
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT * FROM study_items ORDER BY id"); return [dict(r) for r in cur.fetchall()]


def upsert_rating(rater, code, c, r, p, a, s, comments):
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("""INSERT INTO plan_ratings (rater_id,plan_code,clarity,relevance,practicality,adherence,safety,comments)
                           VALUES (%s,%s,%s,%s,%s,%s,%s,%s)
                           ON CONFLICT (rater_id,plan_code) DO UPDATE SET clarity=EXCLUDED.clarity, relevance=EXCLUDED.relevance,
                           practicality=EXCLUDED.practicality, adherence=EXCLUDED.adherence, safety=EXCLUDED.safety, comments=EXCLUDED.comments
                           RETURNING id""", (rater, code, c, r, p, a, s, comments))
            rid = cur.fetchone()["id"]
        conn.commit(); return rid


def rated_codes(rater):
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT plan_code FROM plan_ratings WHERE rater_id=%s", (rater,)); return {r["plan_code"] for r in cur.fetchall()}


def study_summary():
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("""SELECT i.system, COUNT(*) n, COUNT(DISTINCT r.rater_id) raters,
                           ROUND(AVG(r.clarity)::numeric,2) clarity, ROUND(AVG(r.relevance)::numeric,2) relevance,
                           ROUND(AVG(r.practicality)::numeric,2) practicality, ROUND(AVG(r.adherence)::numeric,2) adherence,
                           ROUND(AVG(r.safety)::numeric,2) safety
                           FROM plan_ratings r JOIN study_items i ON i.plan_code=r.plan_code GROUP BY i.system ORDER BY i.system""")
            ratings = [dict(r) for r in cur.fetchall()]
            cur.execute("""SELECT system, COUNT(*) plans, COUNT(within_sodium_target) measured, SUM(CASE WHEN within_sodium_target THEN 1 ELSE 0 END) within_sodium,
                           ROUND(AVG(attempts)::numeric,2) avg_attempts FROM study_items GROUP BY system ORDER BY system""")
            objective = [dict(r) for r in cur.fetchall()]
            cur.execute("SELECT COUNT(DISTINCT rater_id) raters, COUNT(*) ratings FROM plan_ratings"); totals = dict(cur.fetchone())
            return {"ratings_by_system": ratings, "objective_by_system": objective, "totals": totals}


def feedback_summary():
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("""SELECT kind, COUNT(*) n, ROUND(AVG(clarity)::numeric,2) clarity, ROUND(AVG(relevance)::numeric,2) relevance,
                           ROUND(AVG(practicality)::numeric,2) practicality FROM plan_feedback GROUP BY kind""")
            return [dict(r) for r in cur.fetchall()]
