"""
FastAPI backend for the Hypertension Risk Assessment system.

Endpoints:
  POST /assess              -- run a full risk assessment, saves to history
  GET  /assessments         -- list past assessments (requires admin key if ADMIN_KEY is set)
  GET  /assessments/{id}    -- get one past assessment (requires admin key if ADMIN_KEY is set)
  GET  /scenarios           -- the 5 standard usability-eval scenarios, pre-computed
  POST /usability-review    -- submit a reviewer's rating for a scenario
  GET  /usability-reviews   -- list all submitted reviews
  GET  /usability-summary   -- average ratings across all reviews
  GET  /health              -- simple health check (for Railway)

Run locally:
    uvicorn app.main:app --reload

Docker/Railway: see Dockerfile in this same backend/ folder.
"""

import os

from fastapi import FastAPI, HTTPException, Query, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional

from app.ml.risk_model import run_risk_prediction, run_explainability
from app.ml.dietary_guidance import generate_core_guidance, personalize_presentation, format_report
from app import database

app = FastAPI(title="Hypertension Risk Assessment API", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten this to your desktop app / frontend origin in production
    allow_methods=["*"],
    allow_headers=["*"],
)

# Set ADMIN_KEY in Railway's environment variables to require this key on
# the results endpoints (so random visitors can't browse everyone's name +
# results). Leave it unset only while you're first testing locally.
ADMIN_KEY = os.environ.get("ADMIN_KEY")


def require_admin_key(key: Optional[str] = Query(default=None)):
    if ADMIN_KEY and key != ADMIN_KEY:
        raise HTTPException(status_code=401, detail="Missing or invalid admin key")
    return True


@app.on_event("startup")
def startup():
    database.init_db()


class Profile(BaseModel):
    RIAGENDR: int
    RIDAGEYR: float
    RIDRETH3: int
    INDFMPIR: float
    BMXBMI: float
    BMXWAIST: float
    DR1TKCAL: float
    DR1TSODI: float
    DR1TPOTA: float
    DR1TPROT: float
    DR1TCARB: float
    DR1TTFAT: float
    DR1TALCO: float
    LBXSCR: float
    SMQ020: int
    PAQ650: int
    PAQ665: int
    ALQ121: float


class Preferences(BaseModel):
    cuisine: Optional[str] = None
    constraint: Optional[str] = None
    dislikes: Optional[list] = None


class AssessRequest(BaseModel):
    name: Optional[str] = None
    profile: Profile
    preferences: Optional[Preferences] = None


class UsabilityReviewRequest(BaseModel):
    scenario_label: str
    clarity_rating: int
    relevance_rating: int
    practicality_rating: int
    comments: Optional[str] = ""


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/assess")
def assess(req: AssessRequest):
    profile_dict = req.profile.dict()
    prefs_dict = req.preferences.dict(exclude_none=True) if req.preferences else {}

    risk_result = run_risk_prediction(profile_dict)
    explain_result = run_explainability(profile_dict)
    guidance = generate_core_guidance(explain_result["top_factors"])
    guidance = personalize_presentation(guidance, prefs_dict)

    assessment_id = database.save_assessment(
        profile=profile_dict,
        preferences=prefs_dict,
        risk_probability=risk_result["risk_probability"],
        risk_label=risk_result["risk_label"],
        top_factors=explain_result["top_factors"],
        guidance=guidance,
        name=req.name,
    )

    return {
        "id": assessment_id,
        "risk_probability": risk_result["risk_probability"],
        "risk_label": risk_result["risk_label"],
        "top_factors": explain_result["top_factors"],
        "guidance": guidance,
        "report_text": format_report(guidance),
    }


@app.get("/assessments")
def get_assessments(limit: int = 50, _auth: bool = Depends(require_admin_key)):
    return database.list_assessments(limit)


@app.get("/assessments/{assessment_id}")
def get_assessment(assessment_id: int, _auth: bool = Depends(require_admin_key)):
    result = database.get_assessment(assessment_id)
    if result is None:
        raise HTTPException(status_code=404, detail="Assessment not found")
    return result


# ---------- Usability evaluation (Objective 5) ----------

SCENARIOS = [
    {
        "label": "Scenario 1: Younger, lower-risk profile",
        "preferences": {},
        "profile": {
            "RIAGENDR": 1, "RIDAGEYR": 32, "RIDRETH3": 3, "INDFMPIR": 4.0,
            "BMXBMI": 23.5, "BMXWAIST": 82.0, "DR1TKCAL": 2200, "DR1TSODI": 2600,
            "DR1TPOTA": 3000, "DR1TPROT": 85, "DR1TCARB": 260, "DR1TTFAT": 65,
            "DR1TALCO": 2, "LBXSCR": 0.8, "SMQ020": 2, "PAQ650": 1, "PAQ665": 1,
            "ALQ121": 4,
        },
    },
    {
        "label": "Scenario 2: Middle-aged, sodium-driven risk",
        "preferences": {"cuisine": "Mexican"},
        "profile": {
            "RIAGENDR": 1, "RIDAGEYR": 55, "RIDRETH3": 4, "INDFMPIR": 1.8,
            "BMXBMI": 29.0, "BMXWAIST": 100.0, "DR1TKCAL": 2000, "DR1TSODI": 4800,
            "DR1TPOTA": 2300, "DR1TPROT": 75, "DR1TCARB": 250, "DR1TTFAT": 80,
            "DR1TALCO": 5, "LBXSCR": 1.0, "SMQ020": 2, "PAQ650": 2, "PAQ665": 2,
            "ALQ121": 3,
        },
    },
    {
        "label": "Scenario 3: Older adult, high overall risk",
        "preferences": {"constraint": "vegetarian"},
        "profile": {
            "RIAGENDR": 2, "RIDAGEYR": 68, "RIDRETH3": 3, "INDFMPIR": 2.5,
            "BMXBMI": 33.0, "BMXWAIST": 108.0, "DR1TKCAL": 1800, "DR1TSODI": 3900,
            "DR1TPOTA": 2000, "DR1TPROT": 60, "DR1TCARB": 220, "DR1TTFAT": 70,
            "DR1TALCO": 0, "LBXSCR": 1.1, "SMQ020": 1, "PAQ650": 2, "PAQ665": 2,
            "ALQ121": 1,
        },
    },
    {
        "label": "Scenario 4: Low income, diet-driven risk",
        "preferences": {"cuisine": "South Asian"},
        "profile": {
            "RIAGENDR": 2, "RIDAGEYR": 45, "RIDRETH3": 1, "INDFMPIR": 0.6,
            "BMXBMI": 27.5, "BMXWAIST": 95.0, "DR1TKCAL": 1700, "DR1TSODI": 4200,
            "DR1TPOTA": 1700, "DR1TPROT": 55, "DR1TCARB": 230, "DR1TTFAT": 60,
            "DR1TALCO": 0, "LBXSCR": 0.9, "SMQ020": 2, "PAQ650": 2, "PAQ665": 2,
            "ALQ121": 0,
        },
    },
    {
        "label": "Scenario 5: Active lifestyle, borderline risk",
        "preferences": {"dislikes": ["fish"]},
        "profile": {
            "RIAGENDR": 1, "RIDAGEYR": 41, "RIDRETH3": 6, "INDFMPIR": 3.2,
            "BMXBMI": 26.0, "BMXWAIST": 90.0, "DR1TKCAL": 2400, "DR1TSODI": 3400,
            "DR1TPOTA": 2600, "DR1TPROT": 95, "DR1TCARB": 270, "DR1TTFAT": 70,
            "DR1TALCO": 6, "LBXSCR": 1.0, "SMQ020": 2, "PAQ650": 1, "PAQ665": 1,
            "ALQ121": 5,
        },
    },
]


@app.get("/scenarios")
def get_scenarios():
    """Pre-computed usability-evaluation scenarios, ready for a reviewer to rate."""
    results = []
    for s in SCENARIOS:
        risk_result = run_risk_prediction(s["profile"])
        explain_result = run_explainability(s["profile"])
        guidance = generate_core_guidance(explain_result["top_factors"])
        guidance = personalize_presentation(guidance, s["preferences"])
        results.append({
            "label": s["label"],
            "risk_probability": risk_result["risk_probability"],
            "risk_label": risk_result["risk_label"],
            "guidance": guidance,
            "report_text": format_report(guidance),
        })
    return results


@app.post("/usability-review")
def submit_review(req: UsabilityReviewRequest):
    review_id = database.save_usability_review(
        scenario_label=req.scenario_label,
        clarity=req.clarity_rating,
        relevance=req.relevance_rating,
        practicality=req.practicality_rating,
        comments=req.comments or "",
    )
    return {"id": review_id, "status": "saved"}


@app.get("/usability-reviews")
def get_reviews():
    return database.list_usability_reviews()


@app.get("/usability-summary")
def get_summary():
    return database.usability_summary()
