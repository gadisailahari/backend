"""Email / SMS alerts. Every attempt is logged in the `notifications` table.

Email is sent only if SMTP_HOST (and SMTP_FROM) are set in the environment.
SMS is sent only if TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN and TWILIO_FROM are set.
If nothing is configured the alert is still logged (status "not_configured"), and the
clinician page still shows every open request, so nothing is lost.
"""
import os, smtplib, ssl, json, base64, urllib.request, urllib.parse
from email.message import EmailMessage
from app.database import get_conn


def init_tables():
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute("""CREATE TABLE IF NOT EXISTS notifications (
                id SERIAL PRIMARY KEY, created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP, to_kind TEXT, channel TEXT,
                to_addr TEXT, subject TEXT, body TEXT, status TEXT, error TEXT)""")
        conn.commit()


def _log(to_kind, channel, to_addr, subject, body, status, error=None):
    try:
        with get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute("INSERT INTO notifications (to_kind,channel,to_addr,subject,body,status,error) VALUES (%s,%s,%s,%s,%s,%s,%s)",
                            (to_kind, channel, to_addr, subject, body, status, error))
            conn.commit()
    except Exception as e:
        print("notification log failed:", e, flush=True)


def _email(to_addr, subject, body):
    host = os.environ.get("SMTP_HOST")
    if not host: return "not_configured", None
    try:
        msg = EmailMessage(); msg["Subject"] = subject; msg["To"] = to_addr
        msg["From"] = os.environ.get("SMTP_FROM") or os.environ.get("SMTP_USER") or "noreply@example.com"
        msg.set_content(body)
        port = int(os.environ.get("SMTP_PORT", "587"))
        if port == 465:
            s = smtplib.SMTP_SSL(host, port, timeout=10, context=ssl.create_default_context())
        else:
            s = smtplib.SMTP(host, port, timeout=10); s.starttls(context=ssl.create_default_context())
        with s:
            if os.environ.get("SMTP_USER"): s.login(os.environ["SMTP_USER"], os.environ.get("SMTP_PASSWORD", ""))
            s.send_message(msg)
        return "sent", None
    except Exception as e:
        return "error", f"{type(e).__name__}: {e}"[:300]


def _sms(to_phone, body):
    sid, tok, frm = os.environ.get("TWILIO_ACCOUNT_SID"), os.environ.get("TWILIO_AUTH_TOKEN"), os.environ.get("TWILIO_FROM")
    if not (sid and tok and frm): return "not_configured", None
    try:
        data = urllib.parse.urlencode({"To": to_phone, "From": frm, "Body": body[:300]}).encode()
        req = urllib.request.Request(f"https://api.twilio.com/2010-04-01/Accounts/{sid}/Messages.json", data=data)
        req.add_header("Authorization", "Basic " + base64.b64encode(f"{sid}:{tok}".encode()).decode())
        urllib.request.urlopen(req, timeout=10).read()
        return "sent", None
    except Exception as e:
        return "error", f"{type(e).__name__}: {e}"[:300]


def notify(to_kind, subject, body, email=None, phone=None):
    """to_kind: 'clinician' or 'patient'. Never raises. Returns a list of (channel, status)."""
    out = []
    try:
        if email:
            st, err = _email(email, subject, body); _log(to_kind, "email", email, subject, body, st, err); out.append(("email", st))
        if phone:
            st, err = _sms(phone, f"{subject}: {body}"); _log(to_kind, "sms", phone, subject, body, st, err); out.append(("sms", st))
        if not out:
            _log(to_kind, "none", None, subject, body, "no_address")
            out.append(("none", "no_address"))
    except Exception as e:
        print("notify failed:", e, flush=True)
    return out


def notify_clinician(subject, body):
    return notify("clinician", subject, body, email=os.environ.get("CLINICIAN_EMAIL"), phone=os.environ.get("CLINICIAN_PHONE"))
