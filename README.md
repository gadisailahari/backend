# Pulse & Plate: hypertension risk assessment and personalized dietary guidance

Course project, TECH646 (Group 3), Purdue University Northwest. **Educational prototype. Not a medical device. Not medical advice.**

A web tool that (1) estimates hypertension risk from 18 survey-style inputs with an XGBoost model trained on NHANES 2017-2018, explains the main reasons with SHAP, (2) turns that into a meal and exercise plan drafted by a language model from a vetted menu and guideline passages, then **re-checked by code against USDA food tables and safety rules**, and (3) lets people type what they ate in their own words.

Design rule: **the AI proposes, code decides.** Only the XGBoost model is trained; the language model, embedding model, food tables and guideline index are used as they are.

## What is in this folder
| Path | What it is |
|---|---|
| `app/main.py` | FastAPI app: `/assess`, `/assessments`, `/scenarios`, `/health` |
| `app/ml/` | trained model (`xgb_hypertension_model_2017.json`), prediction + SHAP, fixed guidance messages |
| `app/rag/` | safety rules, nutrient checker, plan checker, vetted menu, rule-based builder, retrieval, AI pipeline, data files |
| `app/plan_routes.py`, `app/plan_db.py` | plan, chat, review, rating-study endpoints and tables |
| `app/care.py`, `app/notify.py` | check-ins, clinician questions, e-mail/SMS alerts |
| `app/diet_text.py` | free-text diet entry (sentence to grams to nutrients) |
| `frontend/` | six static pages: patient app, dashboard, clinician, review, rating, and `hub.html` (one sign-in for patients, doctors, database, tasks) |
| `app/hub.py` | Care Hub back end: ID + PIN login, patient/doctor/database/task endpoints |
| `training/` | how the risk model was built: data prep, training script, the 5,250-row training table |
| `Dockerfile`, `requirements.txt` | back-end container |

## Run the back end locally
```
pip install -r requirements.txt
export DATABASE_URL=postgresql://...      # needs a PostgreSQL database
export ADMIN_KEY=choose-something-long
export GROQ_API_KEY=...                   # optional; without it plans fall back to the rule-based builder
uvicorn app.main:app --reload
```
Open http://127.0.0.1:8000/docs for the interactive API. See `.env.example` for all settings.
The risk calculator needs no AI key. AI plans and the free-text diet reader use the Groq API.

## Front end
Static HTML. In each page, `BACKEND_URL` near the top of the script points to the back end; change it to your own server address. Serve with any static host (`frontend/Dockerfile` uses nginx).

## Deploy (what we did on Railway)
Three services: back end (root `/`, uses `Dockerfile`), front end (root directory `frontend`), and a PostgreSQL plugin. Set the variables in `.env.example` on the back end. Railway redeploys on every push to `main`.

## Reproduce the model
```
cd training
pip install -r requirements.txt
python train_model_2017_full.py        # uses data/processed_2017_full.csv (included)
```
Expected: 4,200 training rows, 1,050 test rows, held-out **AUC about 0.802**.
To rebuild the table from raw files: download the nine NHANES 2017-2018 `.xpt` files listed at the top of `data_prep_2017_full.py` into `training/data/raw_2017/`, then run `python data_prep_2017_full.py`.

How the data were prepared: 9,254 participants -> 3,398 under 18 removed -> 606 adults with no blood-pressure reading removed -> **5,250 used** (54.2% hypertensive). Rows with blank predictor cells (1,784 of 5,250) are kept; XGBoost handles blanks. Label: mean systolic >= 130, or mean diastolic >= 80, or taking BP medicine (2017 ACC/AHA).

## Known limits
- The blinded rating study is designed but not yet run; the diet reader has not been checked against a dietitian.
- Held-out AUC comes from one split of one survey cycle (95% bootstrap interval 0.775 to 0.828 on the test rows; no cross-validation, no tuning).
- Scope: hypertension only (nutritionist guidance). Kidney disease, a low eGFR, pregnancy and breastfeeding get a referral, not a plan. The old kidney-safe branches in `app/rag` are inactive and can be deleted later.
- CORS is open and the admin key is a shared secret: fine for a prototype, not for production.

## Data and sources
NHANES 2017-2018 (CDC/NCHS); USDA FNDDS 2017-2018 (USDA ARS); guideline passages from AHA/ACC, KDIGO, Dietary Guidelines for Americans, WHO, NIH ODS, ADA and others (see `app/rag/data/corpus.jsonl` for the source of each passage). Check each source's terms before redistributing the guideline text.
