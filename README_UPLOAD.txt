WHAT TO UPLOAD (everything new since the original project) - GitHub repo gadisailahari/backend
Keep the same folders. Replace or add these 8 files:

  app/accounts.py        NEW   (patient accounts, IDs, PINs, recovery)
  app/hub.py             NEW or replace   (Care Hub)
  app/main.py            replace
  app/care.py            replace
  frontend/index.html    replace   (patient app with sign-in)
  frontend/hub.html      replace   (Care Hub page)
  frontend/Dockerfile    replace   (so hub.html is served)
  README.md              optional

Then in Railway:
 1. Backend service > Variables (add if missing):
      HUB_ID, HUB_PIN, HUB_SECRET   (Care Hub login; HUB_SECRET = long random text)
      PATIENT_SECRET                (long random text, recommended)
      SMTP_HOST, SMTP_PORT, SMTP_USER, SMTP_PASSWORD, SMTP_FROM  (optional: email recovery)
 2. Redeploy the BACKEND and the FRONTEND service (variables are read only at start-up).
 3. Check: backend URL + /account/status  ->  {"accounts":"on",...}
          frontend URL + /hub.html  ->  Care Hub sign-in page.
New database tables and columns are created automatically.
