"""Patient accounts (with recovery): a system-made patient ID + PIN, and saved history (long-term memory).

Design rules
- The patient never picks an ID. The server makes a short code (for example PP-4K7M2Q) and shows it once.
- The PIN (4 to 8 digits) is stored only as a salted PBKDF2 hash. Wrong PINs are rate limited per patient code and per IP address.
- A successful sign-in returns a signed token. The browser sends it as "Authorization: Bearer ...".
- Accounts are optional. Without an account the app works exactly as before (each visit stored separately).
- A signed-in patient can read only their own records.
- Short-term memory (the visit in progress) lives in the browser; long-term memory (earlier visits) lives in these tables.
"""
import os, json, hmac, time, hashlib, base64, secrets, threading
from typing import Optional
from fastapi import APIRouter, HTTPException, Request, Header
from pydantic import BaseModel, Field

from app.database import get_conn

router = APIRouter(prefix="/account")

TOKEN_DAYS = float(os.environ.get("PATIENT_TOKEN_DAYS", "30"))
_SECRET = (os.environ.get("PATIENT_SECRET") or hashlib.sha256(("pp-patient|" + os.environ.get("DATABASE_URL", "") + "|" + os.environ.get("HUB_SECRET", "")).encode()).hexdigest()).encode()
ALPHABET = "23456789ABCDEFGHJKMNPQRSTUVWXYZ"   # no 0/O/1/I/L, easy to read aloud
MAX_FAILS, LOCK_SECONDS = 5, 600
_ip_fails, _lock = {}, threading.Lock()


def init_account_tables():
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("""CREATE TABLE IF NOT EXISTS patients (
                id SERIAL PRIMARY KEY, created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                patient_code TEXT UNIQUE NOT NULL, pin_salt TEXT NOT NULL, pin_hash TEXT NOT NULL,
                display_name TEXT, last_login TIMESTAMP, failed_attempts INTEGER DEFAULT 0, locked_until DOUBLE PRECISION DEFAULT 0)""")
            for col, typ in [("email", "TEXT"), ("recovery_salt", "TEXT"), ("recovery_hash", "TEXT"), ("reset_hash", "TEXT"),
                             ("reset_exp", "DOUBLE PRECISION DEFAULT 0"), ("reset_tries", "INTEGER DEFAULT 0")]:
                cur.execute(f"ALTER TABLE patients ADD COLUMN IF NOT EXISTS {col} {typ}")
            cur.execute("ALTER TABLE assessments ADD COLUMN IF NOT EXISTS patient_id INTEGER")
            cur.execute("ALTER TABLE assessments ADD COLUMN IF NOT EXISTS form_json TEXT")
        conn.commit()


# ---------- helpers ----------
def _hash(pin: str, salt: str) -> str:
    return hashlib.pbkdf2_hmac("sha256", pin.encode(), bytes.fromhex(salt), 200_000).hex()


def _sign(payload: str) -> str:
    return hmac.new(_SECRET, payload.encode(), hashlib.sha256).hexdigest()


def make_token(patient_id: int, now: Optional[float] = None) -> str:
    exp = int((now or time.time()) + TOKEN_DAYS * 86400)
    body = base64.urlsafe_b64encode(f"{patient_id}|{exp}".encode()).decode().rstrip("=")
    return f"{body}.{_sign(body)}"


def check_token(tok: str, now: Optional[float] = None) -> Optional[int]:
    try:
        body, sig = tok.split(".", 1)
        if not hmac.compare_digest(sig, _sign(body)): return None
        pid, exp = base64.urlsafe_b64decode(body + "=" * (-len(body) % 4)).decode().rsplit("|", 1)
        return int(pid) if int(exp) >= (now or time.time()) else None
    except Exception:
        return None


def patient_from_header(authorization: Optional[str]) -> Optional[int]:
    tok = (authorization or "").removeprefix("Bearer ").strip()
    return check_token(tok) if tok else None


def require_patient(authorization: Optional[str] = Header(default=None)) -> int:
    pid = patient_from_header(authorization)
    if not pid: raise HTTPException(401, "Please sign in again")
    return pid


def _name_stem(name) -> str:
    """First name as plain capital letters (accents removed), 2 to 12 letters, or '' if there is none."""
    import unicodedata, re
    first = (name or "").strip().split(" ")[0] if (name or "").strip() else ""
    ascii_ = unicodedata.normalize("NFKD", first).encode("ascii", "ignore").decode()
    letters = re.sub(r"[^A-Za-z]", "", ascii_).upper()[:12]
    return letters if len(letters) >= 2 else ""


def _norm_code(c: str) -> str:
    """Accepts PRIYA-482, priya482, priya 482, 482-193, 482193 (and older PP-XXXXXX codes) in any case."""
    import re
    c = (c or "").strip().upper().replace(" ", "")
    if c.isdigit() and len(c) == 6: return c[:3] + "-" + c[3:]
    m = re.fullmatch(r"([A-Z]{2,12})-?(\d{3,4})", c)
    return f"{m.group(1)}-{m.group(2)}" if m else c


def _client(req: Request) -> str:
    fwd = req.headers.get("x-forwarded-for", "")
    return fwd.split(",")[0].strip() if fwd else (req.client.host if req.client else "?")


def _q(sql, args=(), one=False, commit=False):
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, args)
            out = None
            if cur.description:
                rows = [dict(r) for r in cur.fetchall()]
                out = (rows[0] if rows else None) if one else rows
        if commit: conn.commit()
        return out


EMAIL_RE = r"^[^@\s]{1,64}@[^@\s]{1,200}\.[^@\s]{2,20}$"


class RegisterRequest(BaseModel):
    pin: str = Field(min_length=4, max_length=8, pattern=r"^[0-9]{4,8}$")
    display_name: Optional[str] = Field(default=None, max_length=60)
    email: Optional[str] = Field(default=None, max_length=254, pattern=EMAIL_RE)   # optional; used only for recovery


class LoginRequest(BaseModel):
    patient_code: str = Field(max_length=20)
    pin: str = Field(max_length=8)


def _email_on() -> bool:
    return bool(os.environ.get("SMTP_HOST"))


@router.get("/status")
def status():
    return {"accounts": "on", "email_recovery": _email_on()}


def _new_recovery_key():
    g = lambda: "".join(secrets.choice(ALPHABET) for _ in range(4))
    return f"RK-{g()}-{g()}-{g()}"


def _norm_key(k: str) -> str:
    return "".join(ch for ch in (k or "").upper() if ch.isalnum())   # ignores dashes, spaces and case


def _key_hash(key: str, salt: str) -> str:
    # the key is random (about 57 bits), so a fast salted hash is enough; the PIN uses PBKDF2 because it is short
    return hashlib.pbkdf2_hmac("sha256", _norm_key(key).encode(), bytes.fromhex(salt), 50_000).hex()


def _issue_recovery_key(pid: int) -> str:
    key, salt = _new_recovery_key(), secrets.token_hex(16)
    _q("UPDATE patients SET recovery_salt=%s, recovery_hash=%s WHERE id=%s", (salt, _key_hash(key, salt), pid), commit=True)
    return key


@router.post("/register")
def register(body: RegisterRequest, req: Request):
    ip, now = _client(req), time.time()
    with _lock:   # at most 10 new accounts per IP per hour, so the table cannot be flooded
        recent = [t for t in _ip_fails.get("reg:" + ip, []) if now - t < 3600]
        if len(recent) >= 10: raise HTTPException(429, "Too many new accounts from this connection. Try again later.")
        recent.append(now); _ip_fails["reg:" + ip] = recent
    salt = secrets.token_hex(16)
    stem = _name_stem(body.display_name)
    for i in range(40):
        if stem:   # easy to remember: first name + 3 digits, for example PRIYA-482 (4 digits if that name is very common)
            code = f"{stem}-{100 + secrets.randbelow(900)}" if i < 25 else f"{stem}-{1000 + secrets.randbelow(9000)}"
        else:      # no name given: six digits, for example 482-193
            n = 100000 + secrets.randbelow(900000); code = f"{n // 1000}-{n % 1000:03d}"
        try:
            row = _q("INSERT INTO patients (patient_code,pin_salt,pin_hash,display_name,email,last_login) VALUES (%s,%s,%s,%s,%s,CURRENT_TIMESTAMP) RETURNING id",
                     (code, salt, _hash(body.pin, salt), (body.display_name or "").strip() or None, (body.email or "").strip().lower() or None), one=True, commit=True)
            return {"patient_code": code, "token": make_token(row["id"]), "display_name": (body.display_name or "").strip() or None,
                    "recovery_key": _issue_recovery_key(row["id"])}
        except Exception:
            continue   # code clash: try another
    raise HTTPException(500, "Could not make an account. Please try again.")


@router.post("/login")
def login(body: LoginRequest, req: Request):
    ip, now = _client(req), time.time()
    with _lock:
        recent = [t for t in _ip_fails.get(ip, []) if now - t < LOCK_SECONDS]
        _ip_fails[ip] = recent
        if len(recent) >= MAX_FAILS * 2: raise HTTPException(429, "Too many wrong tries. Wait 10 minutes and try again.")
    code = _norm_code(body.patient_code)
    p = _q("SELECT * FROM patients WHERE patient_code=%s", (code,), one=True)
    if p and (p["locked_until"] or 0) > now:
        raise HTTPException(429, "This account is locked for a few minutes after too many wrong PINs.")
    ok = bool(p) and hmac.compare_digest(_hash(body.pin.strip(), p["pin_salt"]), p["pin_hash"])
    if not ok:
        with _lock: _ip_fails.setdefault(ip, []).append(now)
        if p:
            fails = (p["failed_attempts"] or 0) + 1
            _q("UPDATE patients SET failed_attempts=%s, locked_until=%s WHERE id=%s", (0 if fails >= MAX_FAILS else fails, now + LOCK_SECONDS if fails >= MAX_FAILS else 0, p["id"]), commit=True)
        raise HTTPException(401, "Wrong patient ID or PIN")
    _q("UPDATE patients SET failed_attempts=0, locked_until=0, last_login=CURRENT_TIMESTAMP WHERE id=%s", (p["id"],), commit=True)
    return {"token": make_token(p["id"]), "patient_code": p["patient_code"], "display_name": p["display_name"]}


def _tier(p):
    return "Lower" if p < 0.33 else ("Moderate" if p < 0.50 else "Higher")


@router.get("/me")
def me(authorization: Optional[str] = Header(default=None)):
    pid = require_patient(authorization)
    p = _q("SELECT patient_code, display_name, created_at, last_login FROM patients WHERE id=%s", (pid,), one=True)
    if not p: raise HTTPException(401, "Please sign in again")
    rows = _q("""SELECT a.id, a.created_at, a.risk_probability,
                        (SELECT id FROM plans WHERE assessment_id=a.id ORDER BY id DESC LIMIT 1) AS plan_id,
                        (SELECT review_status FROM plans WHERE assessment_id=a.id ORDER BY id DESC LIMIT 1) AS review_status
                 FROM assessments a WHERE a.patient_id=%s ORDER BY a.id DESC LIMIT 50""", (pid,))
    history = [{"assessment_id": r["id"], "date": str(r["created_at"])[:16], "risk_probability": r["risk_probability"],
                "tier": _tier(r["risk_probability"]), "plan_id": r["plan_id"], "review_status": r["review_status"]} for r in rows]
    return {"patient_code": p["patient_code"], "display_name": p["display_name"], "since": str(p["created_at"])[:10], "visits": len(history), "history": history}


@router.get("/assessment/{aid}")
def my_assessment(aid: int, authorization: Optional[str] = Header(default=None)):
    pid = require_patient(authorization)
    a = _q("SELECT id,created_at,name,profile_json,preferences_json,risk_probability,top_factors_json,form_json FROM assessments WHERE id=%s AND patient_id=%s", (aid, pid), one=True)
    if not a: raise HTTPException(404, "Not found")
    return {"assessment_id": a["id"], "date": str(a["created_at"])[:16], "name": a["name"], "profile": json.loads(a["profile_json"]),
            "preferences": json.loads(a["preferences_json"] or "{}"), "risk_probability": a["risk_probability"], "top_factors": json.loads(a["top_factors_json"]),
            "form_snapshot": json.loads(a["form_json"]) if a["form_json"] else None}


@router.get("/latest")
def latest(authorization: Optional[str] = Header(default=None)):
    """The most recent saved answers, used to pre-fill the form for a returning patient."""
    pid = require_patient(authorization)
    a = _q("SELECT id FROM assessments WHERE patient_id=%s ORDER BY id DESC LIMIT 1", (pid,), one=True)
    if not a: return {"assessment": None}
    return {"assessment": my_assessment(a["id"], authorization)}


@router.get("/plan/{plan_id}")
def my_plan(plan_id: int, authorization: Optional[str] = Header(default=None)):
    pid = require_patient(authorization)
    p = _q("""SELECT p.id, p.text, p.reviewed_text, p.review_status, p.reviewer_note FROM plans p
              JOIN assessments a ON a.id = p.assessment_id WHERE p.id=%s AND a.patient_id=%s""", (plan_id, pid), one=True)
    if not p: raise HTTPException(404, "Not found")
    final = p["reviewed_text"] if p["review_status"] == "edited" and p["reviewed_text"] else p["text"]
    return {"plan_id": p["id"], "text": final, "review_status": p["review_status"], "reviewer_note": p["reviewer_note"]}


@router.post("/change-pin")
def change_pin(body: dict, authorization: Optional[str] = Header(default=None)):
    pid = require_patient(authorization)
    old, new = str(body.get("old_pin", "")), str(body.get("new_pin", ""))
    if not (new.isdigit() and 4 <= len(new) <= 8): raise HTTPException(422, "The new PIN must be 4 to 8 digits")
    p = _q("SELECT pin_salt, pin_hash FROM patients WHERE id=%s", (pid,), one=True)
    if not p or not hmac.compare_digest(_hash(old, p["pin_salt"]), p["pin_hash"]): raise HTTPException(401, "Old PIN is wrong")
    salt = secrets.token_hex(16)
    _q("UPDATE patients SET pin_salt=%s, pin_hash=%s WHERE id=%s", (salt, _hash(new, salt), pid), commit=True)
    return {"status": "changed"}


def _snapshot(form) -> Optional[str]:
    """The form answers as typed (so a returning patient sees the same form). Only short strings, numbers and lists of short strings are kept."""
    if not isinstance(form, dict): return None
    clean = {}
    for k, v in list(form.items())[:60]:
        if not isinstance(k, str) or len(k) > 40: continue
        if isinstance(v, (int, float, bool)): clean[k] = v
        elif isinstance(v, str): clean[k] = v[:200]
        elif isinstance(v, list): clean[k] = [str(x)[:40] for x in v[:20]]
    out = json.dumps(clean)
    return out if len(out) <= 6000 else None


def claim_token(assessment_id: int) -> str:
    """Unguessable proof that the browser that ran /assess is the one claiming it (no table needed)."""
    return _sign(f"claim|{assessment_id}")[:24]


def link_assessment(assessment_id: int, authorization: Optional[str], form_snapshot=None):
    """Called by /assess: attach the new assessment to the signed-in patient (does nothing without a valid token)."""
    pid = patient_from_header(authorization)
    if pid:
        _q("UPDATE assessments SET patient_id=%s, form_json=%s WHERE id=%s", (pid, _snapshot(form_snapshot), assessment_id), commit=True)
    return pid


class ClaimRequest(BaseModel):
    assessment_id: int
    claim_token: str = Field(max_length=64)
    form_snapshot: Optional[dict] = None


@router.post("/claim")
def claim(body: ClaimRequest, authorization: Optional[str] = Header(default=None)):
    """Save a visit that was run before the patient had an account (works once, only with the claim token from that visit)."""
    pid = require_patient(authorization)
    if not hmac.compare_digest(body.claim_token, claim_token(body.assessment_id)): raise HTTPException(403, "This visit cannot be claimed")
    a = _q("SELECT id, patient_id FROM assessments WHERE id=%s", (body.assessment_id,), one=True)
    if not a: raise HTTPException(404, "Not found")
    if a["patient_id"] not in (None, pid): raise HTTPException(409, "This visit already belongs to another account")
    _q("UPDATE assessments SET patient_id=%s, form_json=COALESCE(%s, form_json) WHERE id=%s", (pid, _snapshot(body.form_snapshot), body.assessment_id), commit=True)
    return {"status": "saved"}



# ---------------- recovery ----------------
def _throttle(req: Request, bucket: str, limit: int, window: int):
    ip, now = _client(req), time.time()
    with _lock:
        recent = [t for t in _ip_fails.get(bucket + ip, []) if now - t < window]
        if len(recent) >= limit: raise HTTPException(429, "Too many tries. Please wait a while and try again.")
        recent.append(now); _ip_fails[bucket + ip] = recent


def _set_pin(pid: int, new_pin: str):
    if not (new_pin.isdigit() and 4 <= len(new_pin) <= 8): raise HTTPException(422, "The new PIN must be 4 to 8 digits")
    salt = secrets.token_hex(16)
    _q("UPDATE patients SET pin_salt=%s, pin_hash=%s, failed_attempts=0, locked_until=0, reset_hash=NULL, reset_exp=0, reset_tries=0 WHERE id=%s", (salt, _hash(new_pin, salt), pid), commit=True)


class KeyReset(BaseModel):
    patient_code: str = Field(max_length=20)
    recovery_key: str = Field(max_length=40)
    new_pin: str = Field(max_length=8)


@router.post("/reset-pin-with-key")
def reset_with_key(body: KeyReset, req: Request):
    """Forgot the PIN, but still have the recovery key shown at sign-up. The key is replaced after use."""
    _throttle(req, "key:", 10, 3600)
    p = _q("SELECT id, recovery_salt, recovery_hash FROM patients WHERE patient_code=%s", (_norm_code(body.patient_code),), one=True)
    if not (p and p["recovery_hash"] and hmac.compare_digest(_key_hash(body.recovery_key, p["recovery_salt"]), p["recovery_hash"])):
        raise HTTPException(401, "That patient ID and recovery key do not match")
    _set_pin(p["id"], body.new_pin.strip())
    return {"status": "pin_changed", "new_recovery_key": _issue_recovery_key(p["id"]), "token": make_token(p["id"])}


class EmailOnly(BaseModel):
    email: str = Field(max_length=254, pattern=EMAIL_RE)


class ForgotPin(BaseModel):
    patient_code: str = Field(max_length=20)


GENERIC = "If that matches an account with an email address on file, a message is on its way. It may take a minute."


def _send(to, subject, text):
    from app.notify import _email
    return _email(to, subject, text)


@router.post("/forgot-id")
def forgot_id(body: EmailOnly, req: Request):
    if not _email_on(): raise HTTPException(503, "Email recovery is not switched on. Use your recovery key, or ask the care team.")
    _throttle(req, "fid:", 5, 3600)
    rows = _q("SELECT patient_code FROM patients WHERE email=%s ORDER BY id", (body.email.strip().lower(),))
    if rows:
        _send(body.email.strip(), "Your Pulse & Plate patient ID", "Your patient ID" + ("s are: " if len(rows) > 1 else " is: ") + ", ".join(r["patient_code"] for r in rows) +
              "\n\nYour PIN is not included. If you forgot it too, use \"Forgot your PIN\" on the sign-in page.\nIf you did not ask for this, you can ignore this message.")
    return {"message": GENERIC}


@router.post("/forgot-pin")
def forgot_pin(body: ForgotPin, req: Request):
    if not _email_on(): raise HTTPException(503, "Email recovery is not switched on. Use your recovery key, or ask the care team.")
    _throttle(req, "fpin:", 5, 3600)
    p = _q("SELECT id, email FROM patients WHERE patient_code=%s", (_norm_code(body.patient_code),), one=True)
    if p and p["email"]:
        code = "".join(secrets.choice("0123456789") for _ in range(6)); salt = secrets.token_hex(8)
        _q("UPDATE patients SET reset_hash=%s, reset_exp=%s, reset_tries=0 WHERE id=%s", (salt + ":" + hashlib.sha256((salt + code).encode()).hexdigest(), time.time() + 900, p["id"]), commit=True)
        _send(p["email"], "Your Pulse & Plate PIN reset code", f"Your one-time code is {code}. It works for 15 minutes.\nIf you did not ask for this, ignore this message; your PIN has not changed.")
    return {"message": GENERIC}


class CodeReset(BaseModel):
    patient_code: str = Field(max_length=20)
    code: str = Field(max_length=10)
    new_pin: str = Field(max_length=8)


@router.post("/reset-pin")
def reset_with_code(body: CodeReset, req: Request):
    _throttle(req, "rcode:", 10, 3600)
    p = _q("SELECT id, reset_hash, reset_exp, reset_tries FROM patients WHERE patient_code=%s", (_norm_code(body.patient_code),), one=True)
    bad = HTTPException(401, "That code is wrong or has expired")
    if not (p and p["reset_hash"] and (p["reset_exp"] or 0) > time.time() and (p["reset_tries"] or 0) < 5): raise bad
    salt, h = p["reset_hash"].split(":", 1)
    if not hmac.compare_digest(hashlib.sha256((salt + body.code.strip()).encode()).hexdigest(), h):
        _q("UPDATE patients SET reset_tries=COALESCE(reset_tries,0)+1 WHERE id=%s", (p["id"],), commit=True); raise bad
    _set_pin(p["id"], body.new_pin.strip())
    return {"status": "pin_changed", "token": make_token(p["id"])}


# ---------------- signed-in account settings ----------------
class SetEmail(BaseModel):
    email: str = Field(max_length=254, pattern=EMAIL_RE)


@router.post("/set-email")
def set_email(body: SetEmail, authorization: Optional[str] = Header(default=None)):
    pid = require_patient(authorization)
    _q("UPDATE patients SET email=%s WHERE id=%s", (body.email.strip().lower(), pid), commit=True)
    return {"status": "saved"}


@router.post("/new-recovery-key")
def new_recovery_key(body: dict, authorization: Optional[str] = Header(default=None)):
    """Make a fresh recovery key (the old one stops working). Needs the PIN again."""
    pid = require_patient(authorization)
    p = _q("SELECT pin_salt, pin_hash FROM patients WHERE id=%s", (pid,), one=True)
    if not p or not hmac.compare_digest(_hash(str(body.get("pin", "")), p["pin_salt"]), p["pin_hash"]): raise HTTPException(401, "PIN is wrong")
    return {"recovery_key": _issue_recovery_key(pid)}
