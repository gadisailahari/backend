"""Follow-up care: patient check-ins, clinician questions and answers, patient list for staff, CSV benchmark.

Design rules
- A check-in with a red-flag symptom NEVER produces a new plan: it tells the person to get urgent care and alerts staff.
- A new plan after a check-in goes through the same generate -> check -> (rules fallback) pipeline as the first plan.
- A clinician answer is stored, the patient is alerted (email if they gave one) and sees it in the app.
- Staff pages need ADMIN_KEY. Patients read their own reply only with the secret token they were given.
"""
import os, re, io, csv, json, math, secrets, threading, time, difflib
from typing import Optional, List
from fastapi import APIRouter, HTTPException, Query, Depends, Body
from pydantic import BaseModel, Field

from app import database, plan_db, plan_routes as PR, notify
from app.database import get_conn
from app.ml.risk_model import run_risk_prediction, run_explainability
from app.ml.dietary_guidance import generate_core_guidance, personalize_presentation

router = APIRouter()
FRONTEND_URL = os.environ.get("FRONTEND_URL", "").rstrip("/")
ESCALATE_MINUTES = int(os.environ.get("ESCALATE_MINUTES", "120"))

RED_FLAGS = {"chest_pain": "chest pain or pressure", "fainting": "fainting or passing out", "severe_breathless": "severe shortness of breath",
             "stroke_signs": "face drooping, arm weakness or trouble speaking", "severe_headache": "a sudden severe headache"}
AMBER_FLAGS = {"dizzy": "dizziness or light-headedness", "palpitations": "a racing or irregular heartbeat", "swelling": "swelling in legs or feet",
               "joint_pain": "joint or muscle pain", "very_tired": "unusual tiredness"}
_RED_TEXT = re.compile(r"chest (pain|pressure|tight)|faint|passed out|can'?t breathe|short(ness)? of breath|stroke|slurred|face droop|worst headache|suicid|want to die|kill myself", re.I)
EMERGENCY = ("If you think this is an emergency, call 911 (or your local emergency number) now. Do not wait for a reply from this app.")


def init_care_tables():
    notify.init_tables()
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("""CREATE TABLE IF NOT EXISTS checkins (
                id SERIAL PRIMARY KEY, created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP, plan_id INTEGER, assessment_id INTEGER,
                followed TEXT, energy INTEGER, comfort INTEGER, symptoms_json TEXT, note TEXT, updates_json TEXT,
                red_flag BOOLEAN DEFAULT FALSE, adaptations_json TEXT, new_assessment_id INTEGER, new_plan_id INTEGER,
                old_risk DOUBLE PRECISION, new_risk DOUBLE PRECISION)""")
            cur.execute("""CREATE TABLE IF NOT EXISTS clinician_requests (
                id SERIAL PRIMARY KEY, created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP, plan_id INTEGER, assessment_id INTEGER,
                kind TEXT DEFAULT 'question', question TEXT, token TEXT, patient_email TEXT, patient_phone TEXT, urgent BOOLEAN DEFAULT FALSE,
                status TEXT DEFAULT 'open', answer TEXT, answered_by TEXT, answered_at TIMESTAMP, plan_changed BOOLEAN DEFAULT FALSE,
                reminders INTEGER DEFAULT 0, last_alert TIMESTAMP DEFAULT CURRENT_TIMESTAMP)""")
        conn.commit()
    if os.environ.get("CARE_REMINDERS", "1") != "0":
        threading.Thread(target=_reminder_loop, daemon=True).start()


def _rows(sql, args=()):
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, args); return [dict(r) for r in cur.fetchall()]


def _exec(sql, args=(), ret=False):
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, args); r = cur.fetchone() if ret else None
        conn.commit(); return dict(r) if r else None


def _link(req_id, token):
    return f"{FRONTEND_URL}/?req={req_id}&token={token}" if FRONTEND_URL else "(open the app and press 'Check for a reply')"


def _alert_staff(subject, body):
    return notify.notify_clinician(subject, body + ("\n\nReview page: " + FRONTEND_URL + "/clinician.html" if FRONTEND_URL else ""))


def _reminder_loop():
    while True:
        time.sleep(600)
        try:
            for r in _rows("""SELECT id,urgent,reminders FROM clinician_requests WHERE status='open' AND reminders<3
                              AND last_alert < CURRENT_TIMESTAMP - (%s || ' minutes')::interval""", (str(ESCALATE_MINUTES),)):
                _alert_staff(f"REMINDER: clinician request #{r['id']} is still unanswered", "A patient question has been waiting for a reply.")
                _exec("UPDATE clinician_requests SET reminders=reminders+1, last_alert=CURRENT_TIMESTAMP WHERE id=%s", (r["id"],))
        except Exception as e:
            print("reminder loop:", e, flush=True)


def _plan_and_assessment(plan_id):
    p = plan_db.get_plan(plan_id)
    if not p: raise HTTPException(404, "Plan not found")
    a = database.get_assessment(p["assessment_id"]) if p["assessment_id"] else None
    return p, a


# ---------------- patient check-in ----------------
class Updates(BaseModel):
    BMXBMI: Optional[float] = Field(default=None, ge=10, le=80)
    BMXWAIST: Optional[float] = Field(default=None, ge=40, le=200)
    add_conditions: Optional[List[str]] = None
    remove_conditions: Optional[List[str]] = None


class CheckinRequest(BaseModel):
    plan_id: int
    followed: str = Field(pattern="^(yes|mostly|no)$")
    energy: int = Field(ge=1, le=5)
    comfort: int = Field(ge=1, le=5)
    symptoms: List[str] = []
    note: Optional[str] = Field(default="", max_length=500)
    updates: Optional[Updates] = None
    email: Optional[str] = Field(default=None, max_length=120)


@router.post("/checkin")
def checkin(req: CheckinRequest):
    p, a = _plan_and_assessment(req.plan_id)
    if not a: raise HTTPException(404, "Assessment for this plan was not found")
    syms = [s for s in req.symptoms if s in RED_FLAGS or s in AMBER_FLAGS]
    red = [s for s in syms if s in RED_FLAGS] + (["note"] if _RED_TEXT.search(req.note or "") else [])
    amber = [s for s in syms if s in AMBER_FLAGS]
    upd = req.updates.dict(exclude_none=True) if req.updates else {}
    cid = _exec("""INSERT INTO checkins (plan_id,assessment_id,followed,energy,comfort,symptoms_json,note,updates_json,red_flag)
                   VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s) RETURNING id""",
                (req.plan_id, p["assessment_id"], req.followed, req.energy, req.comfort, json.dumps(syms), req.note, json.dumps(upd), bool(red)), ret=True)["id"]
    who = a.get("name") or f"assessment {a['id']}"
    if red:
        names = ", ".join(RED_FLAGS.get(s, "a serious symptom described in the note") for s in red)
        token = secrets.token_urlsafe(16)
        rid = _exec("""INSERT INTO clinician_requests (plan_id,assessment_id,kind,question,token,patient_email,urgent)
                       VALUES (%s,%s,'urgent_checkin',%s,%s,%s,TRUE) RETURNING id""",
                    (req.plan_id, p["assessment_id"], f"URGENT check-in: {names}. Note: {req.note or ''}", token, req.email), ret=True)["id"]
        _alert_staff(f"URGENT check-in #{rid}", f"{who} reported: {names}. Please contact them.")
        return {"urgent": True, "checkin_id": cid, "request_id": rid, "token": token,
                "message": "You reported a symptom that needs medical attention. Please get urgent care now. " + EMERGENCY +
                           " We did not make a new plan. A clinician has been alerted."}
    profile = json.loads(a["profile_json"]); prefs = json.loads(a["preferences_json"] or "{}")
    old_risk = a["risk_probability"]; adapt = []
    if "BMXBMI" in upd: profile["BMXBMI"] = upd["BMXBMI"]; adapt.append(f"Updated BMI to {upd['BMXBMI']}.")
    if "BMXWAIST" in upd: profile["BMXWAIST"] = upd["BMXWAIST"]; adapt.append(f"Updated waist to {upd['BMXWAIST']} cm.")
    conds = [c.lower().strip() for c in (prefs.get("conditions") or [])]
    for c in (upd.get("add_conditions") or []):
        c = c.lower().strip()
        if c and c not in conds: conds.append(c); adapt.append(f"Added health condition: {c}.")
    for c in (upd.get("remove_conditions") or []):
        c = c.lower().strip()
        if c in conds: conds.remove(c); adapt.append(f"Removed health condition: {c}.")
    prefs["conditions"] = conds
    acts = [x.lower() for x in (prefs.get("activities") or [])]
    gentle = req.comfort <= 2 or req.energy <= 2 or bool(amber)
    if gentle:
        prefs["fitness"] = "beginner"; adapt.append("Exercise set to a gentler (beginner) level because you felt uncomfortable, tired or had a symptom.")
        if req.comfort <= 2 and "zumba" in acts and len(acts) > 1:
            acts.remove("zumba"); adapt.append("Removed Zumba (higher impact) from your activities for now.")
        if not acts: acts = ["walking"]
        prefs["activities"] = acts
    fb = None
    if req.followed == "no": fb = "The person found the last plan hard to follow. Make it simpler: familiar foods, fewer different ingredients, easy meals."
    elif req.followed == "mostly": fb = "The person mostly followed the last plan. Keep what worked and keep it easy."
    if req.energy <= 2: fb = (fb or "") + " The person reports low energy: do not reduce calories below the plan's target."
    risk = run_risk_prediction(profile); expl = run_explainability(profile)
    guidance = personalize_presentation(generate_core_guidance(expl["top_factors"]), prefs)
    new_aid = database.save_assessment(profile=profile, preferences=prefs, risk_probability=risk["risk_probability"], risk_label=risk["risk_label"],
                                       top_factors=expl["top_factors"], guidance=guidance, name=a.get("name"))
    factors = [(f, s) for f, s in expl["top_factors"]]
    structured = PR.pipeline.build_structured_input(profile, risk["risk_probability"], factors, prefs)
    stored = json.loads(p["plan_json"]) if p["plan_json"] else None
    previous = stored.get("obj") if isinstance(stored, dict) else None
    res = PR.pipeline.generate_plan(structured, PR.get_retriever(), PR._llm, feedback=fb, previous=previous, allow_rules=True)
    new_pid, text = PR._store(structured, res, new_aid, parent=req.plan_id)
    _exec("UPDATE checkins SET adaptations_json=%s,new_assessment_id=%s,new_plan_id=%s,old_risk=%s,new_risk=%s WHERE id=%s",
          (json.dumps(adapt), new_aid, new_pid, old_risk, risk["risk_probability"], cid))
    if amber:
        names = ", ".join(AMBER_FLAGS[s] for s in amber)
        token = secrets.token_urlsafe(16)
        rid = _exec("""INSERT INTO clinician_requests (plan_id,assessment_id,kind,question,token,patient_email)
                       VALUES (%s,%s,'symptom_flag',%s,%s,%s) RETURNING id""",
                    (new_pid, new_aid, f"Check-in flag: {names}. Note: {req.note or ''}", token, req.email), ret=True)["id"]
        _alert_staff(f"Check-in symptom flag #{rid}", f"{who} reported: {names}. Their plan was made gentler.")
        adapt.append("A clinician has been told about your symptom.")
    return {"urgent": False, "checkin_id": cid, "adaptations": adapt, "old_risk": old_risk, "new_risk": risk["risk_probability"],
            "plan_id": new_pid, "status": res["status"], "text": text, "view": res.get("view"), "sources": res.get("sources"),
            "kidney_safe_mode": structured["kidney_safe_mode"],
            "message": "Thank you. Your new plan below uses your update." + (" If symptoms continue, please see a clinician." if amber else "")}


# ---------------- patient asks a clinician ----------------
class AskRequest(BaseModel):
    plan_id: int
    question: str = Field(min_length=5, max_length=1000)
    email: Optional[str] = Field(default=None, max_length=120)
    phone: Optional[str] = Field(default=None, max_length=30)


@router.post("/ask-clinician")
def ask_clinician(req: AskRequest):
    p, a = _plan_and_assessment(req.plan_id)
    urgent = bool(_RED_TEXT.search(req.question))
    token = secrets.token_urlsafe(16)
    rid = _exec("""INSERT INTO clinician_requests (plan_id,assessment_id,kind,question,token,patient_email,patient_phone,urgent)
                   VALUES (%s,%s,'question',%s,%s,%s,%s,%s) RETURNING id""",
                (req.plan_id, p["assessment_id"], req.question, token, req.email, req.phone, urgent), ret=True)["id"]
    who = (a or {}).get("name") or f"plan {req.plan_id}"
    _alert_staff(("URGENT " if urgent else "") + f"Clinician request #{rid}", f"{who} asks: {req.question[:200]}")
    msg = ("Your question was sent to the care team. A person (not this app) will answer. " +
           ("We will email you when it is answered. " if (req.email and os.environ.get("SMTP_HOST")) else "Email alerts are not switched on, so please come back and press 'Check for a reply'. ") +
           "Typical wait is not guaranteed.")
    if urgent: msg = "Your message mentions a possible emergency. " + EMERGENCY + " " + msg
    return {"request_id": rid, "token": token, "urgent": urgent, "message": msg, "emergency_note": EMERGENCY}


@router.get("/ask-clinician/{rid}")
def ask_status(rid: int, token: str = Query(min_length=8)):
    r = _rows("SELECT * FROM clinician_requests WHERE id=%s", (rid,))
    if not r or not secrets.compare_digest(r[0]["token"] or "", token): raise HTTPException(404, "Request not found")
    r = r[0]; out = {"request_id": rid, "status": r["status"], "question": r["question"], "created_at": r["created_at"]}
    if r["status"] == "answered":
        out.update(answer=r["answer"], answered_by=r["answered_by"], answered_at=r["answered_at"], plan_changed=r["plan_changed"], plan_id=r["plan_id"])
    return out


# ---------------- staff side ----------------
class AnswerRequest(BaseModel):
    answer: str = Field(min_length=2, max_length=3000)
    answered_by: str = Field(min_length=1, max_length=80)
    revised_plan_text: Optional[str] = Field(default=None, max_length=8000)


@router.get("/clinician/requests")
def list_requests(status: str = "open", _a: bool = Depends(PR.require_admin_key)):
    where = "" if status == "all" else "WHERE r.status=%s"
    rows = _rows(f"""SELECT r.id,r.created_at,r.plan_id,r.assessment_id,r.kind,r.question,r.urgent,r.status,r.answer,r.answered_by,r.answered_at,r.reminders,
                    a.name, a.risk_probability FROM clinician_requests r LEFT JOIN assessments a ON a.id=r.assessment_id {where}
                    ORDER BY r.urgent DESC, r.id DESC LIMIT 200""", () if status == "all" else (status,))
    return rows


@router.post("/clinician/requests/{rid}/answer")
def answer_request(rid: int, req: AnswerRequest, _a: bool = Depends(PR.require_admin_key)):
    r = _rows("SELECT * FROM clinician_requests WHERE id=%s", (rid,))
    if not r: raise HTTPException(404, "Request not found")
    r = r[0]; changed = False
    if req.revised_plan_text and r["plan_id"]:
        changed = plan_db.set_review(r["plan_id"], "edited", req.revised_plan_text, f"Edited while answering request #{rid} by {req.answered_by}")
    _exec("UPDATE clinician_requests SET status='answered',answer=%s,answered_by=%s,answered_at=CURRENT_TIMESTAMP,plan_changed=%s WHERE id=%s",
          (req.answer, req.answered_by, changed, rid))
    sent = notify.notify("patient", "Your care team replied",
                         "Your care team has replied to your question." + (" Your plan was also updated." if changed else "") +
                         f" Open: {_link(rid, r['token'])}", email=r["patient_email"], phone=r["patient_phone"])
    return {"status": "answered", "plan_changed": changed, "patient_alert": sent}


def _groups():
    """Group assessments into one patient thread: an assessment created by a check-in belongs to the thread of the one it came from."""
    parent = {c["new_assessment_id"]: c["assessment_id"] for c in _rows("SELECT assessment_id,new_assessment_id FROM checkins WHERE new_assessment_id IS NOT NULL")}
    def root(x):
        seen = set()
        while x in parent and x not in seen: seen.add(x); x = parent[x]
        return x
    return root


@router.get("/clinician/patients")
def patients(limit: int = 200, _a: bool = Depends(PR.require_admin_key)):
    root = _groups()
    ass = _rows("SELECT id,created_at,name,risk_probability,risk_label FROM assessments ORDER BY id DESC LIMIT %s", (limit,))
    plans = _rows("SELECT id,assessment_id,status,review_status,created_at FROM plans ORDER BY id")
    chk = _rows("SELECT assessment_id,red_flag,created_at FROM checkins")
    reqs = _rows("SELECT assessment_id,status,urgent FROM clinician_requests")
    g = {}
    for x in ass:
        r = root(x["id"]); e = g.setdefault(r, {"patient_id": r, "name": x["name"], "assessments": [], "latest_risk": None, "plans": 0, "latest_plan_status": None,
                                               "latest_review": None, "checkins": 0, "red_flags": 0, "open_questions": 0, "last_activity": x["created_at"]})
        e["assessments"].append(x["id"])
        if e["latest_risk"] is None: e["latest_risk"] = x["risk_probability"]
        if x["created_at"] > e["last_activity"]: e["last_activity"] = x["created_at"]
    where = {aid: r for r, e in g.items() for aid in e["assessments"]}
    for pl in plans:
        e = g.get(where.get(pl["assessment_id"]))
        if e: e["plans"] += 1; e["latest_plan_status"] = pl["status"]; e["latest_review"] = pl["review_status"]
    for c in chk:
        e = g.get(where.get(c["assessment_id"]))
        if e: e["checkins"] += 1; e["red_flags"] += 1 if c["red_flag"] else 0
    for q in reqs:
        e = g.get(where.get(q["assessment_id"]))
        if e and q["status"] == "open": e["open_questions"] += 1
    return sorted(g.values(), key=lambda e: (e["open_questions"] > 0, str(e["last_activity"])), reverse=True)


@router.get("/clinician/patient/{patient_id}")
def patient_detail(patient_id: int, _a: bool = Depends(PR.require_admin_key)):
    root = _groups()
    ids = [x["id"] for x in _rows("SELECT id FROM assessments") if root(x["id"]) == patient_id]
    if not ids: raise HTTPException(404, "Patient not found")
    ass = _rows("SELECT id,created_at,name,profile_json,preferences_json,risk_probability,risk_label,top_factors_json FROM assessments WHERE id = ANY(%s) ORDER BY id", (ids,))
    for x in ass:
        x["profile"] = json.loads(x.pop("profile_json")); x["preferences"] = json.loads(x.pop("preferences_json") or "{}"); x["top_factors"] = json.loads(x.pop("top_factors_json"))
    plans = _rows("SELECT id,created_at,assessment_id,parent_plan_id,status,review_status,reviewer_note,reviewed_at,text,reviewed_text FROM plans WHERE assessment_id = ANY(%s) ORDER BY id", (ids,))
    chk = _rows("SELECT * FROM checkins WHERE assessment_id = ANY(%s) ORDER BY id", (ids,))
    for c in chk:
        for k in ("symptoms_json", "updates_json", "adaptations_json"):
            c[k[:-5]] = json.loads(c.pop(k) or "null")
    reqs = _rows("SELECT id,created_at,plan_id,kind,question,urgent,status,answer,answered_by,answered_at,plan_changed FROM clinician_requests WHERE assessment_id = ANY(%s) ORDER BY id", (ids,))
    return {"patient_id": patient_id, "assessments": ass, "plans": plans, "checkins": chk, "requests": reqs}


# ---------------- CSV benchmark (known values vs the model) ----------------
PRED = ["RIAGENDR", "RIDAGEYR", "RIDRETH3", "INDFMPIR", "BMXBMI", "BMXWAIST", "DR1TKCAL", "DR1TSODI", "DR1TPOTA", "DR1TPROT", "DR1TCARB", "DR1TTFAT",
        "DR1TALCO", "LBXSCR", "SMQ020", "PAQ650", "PAQ665", "ALQ121"]
SYN = {"RIAGENDR": ["sex", "gender"], "RIDAGEYR": ["age", "ageyears", "ageinyears"], "RIDRETH3": ["race", "ethnicity", "raceethnicity"],
       "INDFMPIR": ["income", "pir", "povertyratio", "incometopovertyratio"], "BMXBMI": ["bmi", "bodymassindex"], "BMXWAIST": ["waist", "waistcm", "waistcircumference"],
       "DR1TKCAL": ["kcal", "calories", "energy", "energykcal"], "DR1TSODI": ["sodium", "sodiummg", "na"], "DR1TPOTA": ["potassium", "potassiummg", "k"],
       "DR1TPROT": ["protein", "proteing"], "DR1TCARB": ["carbs", "carbohydrate", "carbohydrates"], "DR1TTFAT": ["fat", "totalfat", "fatg"],
       "DR1TALCO": ["alcohol", "alcoholg"], "LBXSCR": ["creatinine", "serumcreatinine", "scr"], "SMQ020": ["smoker", "smoked", "smoking", "smoked100"],
       "PAQ650": ["vigorousrec", "vigorousactivity", "vigorous"], "PAQ665": ["moderaterec", "moderateactivity", "moderate"], "ALQ121": ["alcoholfreq", "alcfreq", "alcdays", "drinkingfrequency", "drinksfreq", "drinkingdays"]}
LABEL_NAMES = ["hypertension", "htn", "label", "target", "bprisk", "htnrisk", "knownrisk", "highbp", "outcome", "y"]
_norm = lambda s: re.sub(r"[^a-z0-9]", "", str(s).lower())


def map_columns(cols):
    n = {c: _norm(c) for c in cols}; mapping, how = {}, {}
    pool = {}
    for k in PRED: pool[_norm(k)] = k; [pool.setdefault(s, k) for s in SYN[k]]
    for c, nc in n.items():
        if nc in pool and pool[nc] not in mapping: mapping[pool[nc]] = c; how[pool[nc]] = "exact name"
    for c, nc in n.items():
        if c in mapping.values(): continue
        m = difflib.get_close_matches(nc, list(pool), n=1, cutoff=0.8)
        if m and pool[m[0]] not in mapping: mapping[pool[m[0]]] = c; how[pool[m[0]]] = f"close match to '{m[0]}'"
    return mapping, how


def _auc(y, s):
    pos = [(sc, i) for i, (yy, sc) in enumerate(zip(y, s)) if yy == 1]; neg = [sc for yy, sc in zip(y, s) if yy == 0]
    if not pos or not neg: return None
    allv = sorted([(sc, 0) for sc in neg] + [(sc, 1) for sc, _ in pos]); ranks, i = {}, 0
    while i < len(allv):
        j = i
        while j + 1 < len(allv) and allv[j + 1][0] == allv[i][0]: j += 1
        for k in range(i, j + 1): ranks[k] = (i + j) / 2 + 1
        i = j + 1
    rp = sum(ranks[k] for k, (_, lab) in enumerate(allv) if lab == 1)
    return (rp - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))


class BenchRequest(BaseModel):
    csv_text: str = Field(max_length=6_000_000)
    threshold: float = 0.5


@router.post("/benchmark")
def benchmark(req: BenchRequest, _a: bool = Depends(PR.require_admin_key)):
    rd = csv.DictReader(io.StringIO(req.csv_text)); cols = rd.fieldnames or []
    mapping, how = map_columns(cols); missing = [k for k in PRED if k not in mapping]
    if missing: raise HTTPException(400, {"error": "Could not find these model inputs in your file", "missing": missing, "your_columns": cols, "matched": mapping})
    lab = next((c for c in cols if _norm(c) in LABEL_NAMES), None)
    sbp = next((c for c in cols if _norm(c) in ("meansbp", "sbp", "systolic")), None); dbp = next((c for c in cols if _norm(c) in ("meandbp", "dbp", "diastolic")), None)
    ys, ss, skipped, n, missing_rows = [], [], 0, 0, 0
    for row in rd:
        n += 1
        if n > 6000: break
        try:
            prof = {k: (float(row[mapping[k]]) if str(row[mapping[k]]).strip() not in ("", "NA", "NaN", "nan", "None") else float("nan")) for k in PRED}
        except Exception: skipped += 1; continue
        if any(math.isnan(v) for v in prof.values()): missing_rows += 1
        for k in ("RIAGENDR", "RIDRETH3", "SMQ020", "PAQ650", "PAQ665"):
            if not math.isnan(prof[k]): prof[k] = int(prof[k])
        sc = run_risk_prediction(prof)["risk_probability"]
        if sc is None or (isinstance(sc, float) and math.isnan(sc)): skipped += 1; continue
        y = None
        try:
            if lab: v = str(row[lab]).strip().lower(); y = 1 if v in ("1", "1.0", "yes", "true", "high", "y") else 0 if v in ("0", "0.0", "no", "false", "low", "n") else None
            elif sbp and dbp: y = 1 if float(row[sbp]) >= 130 or float(row[dbp]) >= 80 else 0
        except Exception: y = None
        ss.append(sc); ys.append(y)
    out = {"rows_scored": len(ss), "rows_skipped_bad_values": skipped, "rows_with_blank_inputs_scored_anyway": missing_rows, "column_mapping": {k: {"your_column": mapping[k], "how": how[k]} for k in PRED},
           "known_value_column": lab or ("derived from SBP/DBP (>=130 or >=80, medication not known)" if sbp and dbp else None), "threshold": req.threshold}
    pairs = [(y, s) for y, s in zip(ys, ss) if y is not None]
    if pairs:
        tp = sum(1 for y, s in pairs if y == 1 and s >= req.threshold); fn = sum(1 for y, s in pairs if y == 1 and s < req.threshold)
        fp = sum(1 for y, s in pairs if y == 0 and s >= req.threshold); tn = sum(1 for y, s in pairs if y == 0 and s < req.threshold)
        out.update(true_positive=tp, false_negative=fn, false_positive=fp, true_negative=tn,
                   sensitivity=round(tp / (tp + fn), 3) if tp + fn else None, specificity=round(tn / (tn + fp), 3) if tn + fp else None,
                   accuracy=round((tp + tn) / len(pairs), 3), auc=(round(_auc([y for y, _ in pairs], [s for _, s in pairs]), 3) if _auc([y for y, _ in pairs], [s for _, s in pairs]) is not None else None),
                   note="If every known value is 1, only sensitivity (share correctly flagged) is meaningful; AUC needs both groups.")
    else:
        out["note"] = "No known-value column found, so only risk scores were produced."
    out["mean_risk"] = round(sum(ss) / len(ss), 3) if ss else None
    return out
