"""One-login care hub: Patients, Doctors, Database and Tasks behind a single ID + PIN.

Design rules
- Login is checked on the server. Set HUB_ID and HUB_PIN in Railway's Variables. If either is missing, login is switched off
  (503), so a forgotten setting never leaves the hub open.
- A successful login returns a signed token that expires (HUB_TOKEN_HOURS, default 8). The page keeps it in sessionStorage only.
- Wrong IDs/PINs are rate limited per IP address (5 misses in 10 minutes, then 429).
- The Database section is read-only and limited to a fixed list of table names (no free SQL).
- Tasks are worked out from the live data (open questions, plans waiting for review, red-flag check-ins). Only "done" marks are stored.
- The older pages (clinician.html, review.html, dashboard.html) and their ADMIN_KEY keep working.
"""
import os, io, csv, json, hmac, time, hashlib, base64, threading
from typing import Optional
from fastapi import APIRouter, HTTPException, Request, Depends, Header, Query
from fastapi.responses import Response
from pydantic import BaseModel, Field

from app import care, plan_db, plan_routes as PR, notify
from app.database import get_conn

router = APIRouter(prefix="/hub")

HUB_ID = os.environ.get("HUB_ID", "")
HUB_PIN = os.environ.get("HUB_PIN", "")
TOKEN_HOURS = float(os.environ.get("HUB_TOKEN_HOURS", "8"))
_SECRET = (os.environ.get("HUB_SECRET") or hashlib.sha256(f"{HUB_ID}|{HUB_PIN}|{os.environ.get('ADMIN_KEY','')}".encode()).hexdigest()).encode()

TABLES = ["assessments", "plans", "checkins", "clinician_requests", "plan_feedback", "usability_reviews",
          "study_items", "plan_ratings", "notifications", "hub_activity", "hub_task_state"]

_fails, _fails_lock = {}, threading.Lock()


def init_hub_tables():
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("""CREATE TABLE IF NOT EXISTS hub_task_state (
                task_key TEXT PRIMARY KEY, state TEXT NOT NULL, note TEXT, updated_by TEXT, updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)""")
            cur.execute("""CREATE TABLE IF NOT EXISTS hub_activity (
                id SERIAL PRIMARY KEY, created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP, doctor TEXT, role TEXT, action TEXT, target_kind TEXT, target_id INTEGER)""")
        conn.commit()


# ---------------- login and token ----------------
def _sign(payload: str) -> str:
    return hmac.new(_SECRET, payload.encode(), hashlib.sha256).hexdigest()


def make_token(user: str, now: Optional[float] = None) -> str:
    exp = int((now or time.time()) + TOKEN_HOURS * 3600)
    body = base64.urlsafe_b64encode(f"{user}|{exp}".encode()).decode().rstrip("=")
    return f"{body}.{_sign(body)}"


def check_token(tok: str, now: Optional[float] = None) -> Optional[str]:
    try:
        body, sig = tok.split(".", 1)
        if not hmac.compare_digest(sig, _sign(body)): return None
        user, exp = base64.urlsafe_b64decode(body + "=" * (-len(body) % 4)).decode().rsplit("|", 1)
        return user if int(exp) >= (now or time.time()) else None
    except Exception:
        return None


def require_hub(authorization: Optional[str] = Header(default=None)):
    tok = (authorization or "").removeprefix("Bearer ").strip()
    user = check_token(tok) if tok else None
    if not user: raise HTTPException(401, "Please sign in again")
    return user


class LoginRequest(BaseModel):
    id: str = Field(max_length=80)
    pin: str = Field(max_length=80)


def _client(req: Request) -> str:
    fwd = req.headers.get("x-forwarded-for", "")
    return (fwd.split(",")[0].strip() if fwd else (req.client.host if req.client else "?"))


@router.get("/status")
def status():
    return {"hub": "on" if (HUB_ID and HUB_PIN) else "login_not_configured"}


@router.post("/login")
def login(body: LoginRequest, req: Request):
    if not (HUB_ID and HUB_PIN): raise HTTPException(503, "Hub login is not set up yet. Set HUB_ID and HUB_PIN on the server.")
    ip, now = _client(req), time.time()
    with _fails_lock:
        recent = [t for t in _fails.get(ip, []) if now - t < 600]
        _fails[ip] = recent
        if len(recent) >= 5: raise HTTPException(429, "Too many wrong tries. Wait 10 minutes and try again.")
    ok_id = hmac.compare_digest(body.id.strip().encode(), HUB_ID.encode())
    ok_pin = hmac.compare_digest(body.pin.strip().encode(), HUB_PIN.encode())
    if not (ok_id and ok_pin):
        with _fails_lock: _fails[ip].append(now)
        raise HTTPException(401, "Wrong ID or PIN")
    with _fails_lock: _fails.pop(ip, None)
    return {"token": make_token(HUB_ID), "expires_in_hours": TOKEN_HOURS}


# ---------------- tasks (worked out from live data) ----------------
def _task_states():
    return {r["task_key"]: r for r in care._rows("SELECT task_key,state,note,updated_by,updated_at FROM hub_task_state")}


def build_tasks():
    st = _task_states(); out = []
    for r in care._rows("""SELECT r.id,r.created_at,r.assessment_id,r.question,r.urgent,a.name FROM clinician_requests r
                           LEFT JOIN assessments a ON a.id=r.assessment_id WHERE r.status='open' ORDER BY r.urgent DESC, r.id"""):
        out.append(dict(key=f"q:{r['id']}", kind="question", priority="urgent" if r["urgent"] else "normal", title=f"Answer question #{r['id']}",
                        detail=(r["question"] or "")[:200], patient=r["name"], assessment_id=r["assessment_id"], created_at=r["created_at"], auto=True))
    for p in care._rows("""SELECT p.id,p.created_at,p.assessment_id,a.name FROM plans p LEFT JOIN assessments a ON a.id=p.assessment_id
                           WHERE p.review_status='pending' AND p.status='approved_by_checker' ORDER BY p.id"""):
        out.append(dict(key=f"plan:{p['id']}", kind="plan_review", priority="normal", title=f"Review plan #{p['id']}",
                        detail="Plan passed the software checks and waits for a clinician.", patient=p["name"], assessment_id=p["assessment_id"], created_at=p["created_at"], auto=True))
    for c in care._rows("""SELECT c.id,c.created_at,c.assessment_id,c.note,a.name FROM checkins c LEFT JOIN assessments a ON a.id=c.assessment_id
                           WHERE c.red_flag=TRUE ORDER BY c.id"""):
        out.append(dict(key=f"flag:{c['id']}", kind="red_flag", priority="urgent", title=f"Follow up red-flag check-in #{c['id']}",
                        detail=(c["note"] or "Patient reported a red-flag symptom. No new plan was made.")[:200], patient=c["name"], assessment_id=c["assessment_id"],
                        created_at=c["created_at"], auto=False))
    for t in out:
        s = st.get(t["key"])
        t["state"] = "done" if (s and s["state"] == "done") else "open"
        t["note"] = s["note"] if s else None; t["done_by"] = s["updated_by"] if s and t["state"] == "done" else None
    order = {"urgent": 0, "normal": 1}
    return sorted(out, key=lambda t: (t["state"] == "done", order[t["priority"]], str(t["created_at"])), reverse=False)


class TaskMark(BaseModel):
    by: str = Field(default="", max_length=80)
    note: Optional[str] = Field(default="", max_length=500)


@router.get("/tasks")
def tasks(_u: str = Depends(require_hub)):
    t = build_tasks()
    return {"tasks": t, "open": sum(1 for x in t if x["state"] == "open"), "done": sum(1 for x in t if x["state"] == "done")}


@router.post("/tasks/{key}/done")
def task_done(key: str, body: TaskMark, _u: str = Depends(require_hub)):
    if not any(t["key"] == key for t in build_tasks()): raise HTTPException(404, "Task not found")
    care._exec("""INSERT INTO hub_task_state (task_key,state,note,updated_by) VALUES (%s,'done',%s,%s)
                  ON CONFLICT (task_key) DO UPDATE SET state='done',note=EXCLUDED.note,updated_by=EXCLUDED.updated_by,updated_at=CURRENT_TIMESTAMP""",
               (key, body.note, body.by))
    return {"status": "done"}


@router.post("/tasks/{key}/reopen")
def task_reopen(key: str, _u: str = Depends(require_hub)):
    care._exec("DELETE FROM hub_task_state WHERE task_key=%s", (key,))
    return {"status": "open"}


# ---------------- overview ----------------
@router.get("/summary")
def summary(_u: str = Depends(require_hub)):
    one = lambda sql: list(care._rows(sql)[0].values())[0]
    t = build_tasks()
    return {"patients": len(care.patients(1000, True)), "assessments": one("SELECT COUNT(*) FROM assessments"),
            "plans": one("SELECT COUNT(*) FROM plans"), "checkins": one("SELECT COUNT(*) FROM checkins"),
            "open_tasks": sum(1 for x in t if x["state"] == "open"), "urgent_tasks": sum(1 for x in t if x["state"] == "open" and x["priority"] == "urgent"),
            "open_questions": one("SELECT COUNT(*) FROM clinician_requests WHERE status='open'"),
            "plans_to_review": one("SELECT COUNT(*) FROM plans WHERE review_status='pending' AND status='approved_by_checker'")}


# ---------------- patients ----------------
@router.get("/patients")
def patients(_u: str = Depends(require_hub)):
    return care.patients(500, True)


@router.get("/patients/{patient_id}")
def patient(patient_id: int, _u: str = Depends(require_hub)):
    return care.patient_detail(patient_id, True)


# ---------------- doctors ----------------
class HubAnswer(BaseModel):
    answer: str = Field(min_length=2, max_length=3000)
    doctor: str = Field(min_length=1, max_length=80)
    role: str = Field(default="", max_length=60)
    revised_plan_text: Optional[str] = Field(default=None, max_length=8000)


class HubReview(BaseModel):
    action: str = Field(pattern="^(approved|edited|rejected)$")
    doctor: str = Field(min_length=1, max_length=80)
    role: str = Field(default="", max_length=60)
    text: Optional[str] = None
    note: Optional[str] = ""


def _log_activity(doctor, role, action, kind, target):
    care._exec("INSERT INTO hub_activity (doctor,role,action,target_kind,target_id) VALUES (%s,%s,%s,%s,%s)", (doctor, role, action, kind, target))


@router.get("/doctors")
def doctors(_u: str = Depends(require_hub)):
    """Doctor activity (from answers and reviews done here or on the older pages) plus the work waiting for them."""
    ans = care._rows("""SELECT answered_by AS doctor, COUNT(*) AS answered, MAX(answered_at) AS last_active FROM clinician_requests
                        WHERE status='answered' AND answered_by IS NOT NULL GROUP BY answered_by""")
    rev = care._rows("SELECT doctor, role, action, COUNT(*) n, MAX(created_at) last FROM hub_activity GROUP BY doctor, role, action")
    people = {}
    for a in ans:
        name, _, role = a["doctor"].partition(",")            # answers are stored as "Name, Role"
        name, role = name.strip(), role.strip()
        p = people.setdefault(name, dict(doctor=name, role=role, answered=0, reviewed=0, last_active=a["last_active"]))
        p["answered"] += a["answered"]
        if role and not p["role"]: p["role"] = role
        if a["last_active"] and (p["last_active"] is None or a["last_active"] > p["last_active"]): p["last_active"] = a["last_active"]
    for r in rev:
        p = people.setdefault(r["doctor"], dict(doctor=r["doctor"], role="", answered=0, reviewed=0, last_active=r["last"]))
        if r["role"]: p["role"] = r["role"]
        if r["action"] == "review": p["reviewed"] += r["n"]
        if r["last"] and (p["last_active"] is None or r["last"] > p["last_active"]): p["last_active"] = r["last"]
    questions = care._rows("""SELECT r.id,r.created_at,r.plan_id,r.assessment_id,r.question,r.urgent,a.name,a.risk_probability FROM clinician_requests r
                              LEFT JOIN assessments a ON a.id=r.assessment_id WHERE r.status='open' ORDER BY r.urgent DESC, r.id""")
    plans = care._rows("""SELECT p.id,p.created_at,p.assessment_id,p.text,a.name,a.risk_probability FROM plans p LEFT JOIN assessments a ON a.id=p.assessment_id
                          WHERE p.review_status='pending' AND p.status='approved_by_checker' ORDER BY p.id DESC LIMIT 50""")
    return {"doctors": sorted(people.values(), key=lambda p: str(p["last_active"]), reverse=True), "open_questions": questions, "plans_to_review": plans}


@router.post("/requests/{rid}/answer")
def answer(rid: int, body: HubAnswer, _u: str = Depends(require_hub)):
    who = f"{body.doctor}, {body.role}" if body.role else body.doctor
    res = care.answer_request(rid, care.AnswerRequest(answer=body.answer, answered_by=who, revised_plan_text=body.revised_plan_text), True)
    _log_activity(body.doctor, body.role, "answer", "request", rid)
    return res


@router.post("/plans/{plan_id}/review")
def review(plan_id: int, body: HubReview, _u: str = Depends(require_hub)):
    note = (body.note or "") + f" [by {body.doctor}{', ' + body.role if body.role else ''}]"
    res = PR.review(plan_id, PR.ReviewRequest(action=body.action, text=body.text, note=note.strip()), True)
    _log_activity(body.doctor, body.role, "review", "plan", plan_id)
    return res


# ---------------- database (read-only, fixed table list) ----------------
def _existing_tables():
    have = {r["table_name"] for r in care._rows("SELECT table_name FROM information_schema.tables WHERE table_schema='public'")}
    return [t for t in TABLES if t in have]


SECRET_COLUMNS = {"token"}      # patients' private reply tokens are never shown, even to staff


def _clean(v):
    if isinstance(v, (bytes, bytearray)): return "(binary)"
    if hasattr(v, "isoformat"): return v.isoformat(sep=" ", timespec="seconds")
    return v


@router.get("/db/tables")
def db_tables(_u: str = Depends(require_hub)):
    out = []
    for t in _existing_tables():
        n = list(care._rows(f'SELECT COUNT(*) FROM "{t}"')[0].values())[0]
        out.append({"table": t, "rows": n})
    return out


@router.get("/db/{table}")
def db_rows(table: str, limit: int = Query(50, ge=1, le=500), offset: int = Query(0, ge=0), _u: str = Depends(require_hub)):
    if table not in _existing_tables(): raise HTTPException(404, "Unknown table")
    rows = care._rows(f'SELECT * FROM "{table}" ORDER BY 1 DESC LIMIT %s OFFSET %s', (limit, offset))
    total = list(care._rows(f'SELECT COUNT(*) FROM "{table}"')[0].values())[0]
    cols = list(rows[0].keys()) if rows else [r["column_name"] for r in care._rows(
        "SELECT column_name FROM information_schema.columns WHERE table_name=%s ORDER BY ordinal_position", (table,))]
    return {"table": table, "columns": cols, "rows": [{k: ("(hidden)" if k in SECRET_COLUMNS else _clean(v)) for k, v in r.items()} for r in rows], "total": total, "limit": limit, "offset": offset}


@router.get("/db/{table}/csv")
def db_csv(table: str, _u: str = Depends(require_hub)):
    if table not in _existing_tables(): raise HTTPException(404, "Unknown table")
    rows = care._rows(f'SELECT * FROM "{table}" ORDER BY 1')
    buf = io.StringIO(); w = csv.writer(buf)
    if rows:
        w.writerow(rows[0].keys())
        for r in rows: w.writerow([("(hidden)" if k in SECRET_COLUMNS else _clean(v)) for k, v in r.items()])
    return Response(buf.getvalue(), media_type="text/csv", headers={"Content-Disposition": f'attachment; filename="{table}.csv"'})
