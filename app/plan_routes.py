"""Grounded plan, feedback chat, clinician review and rating-study endpoints.
Included from main.py with:   from app.plan_routes import router, init_plan_tables ; app.include_router(router)
"""
import os, sys, re, json, random, string, threading, time
from pathlib import Path
from typing import Optional
from fastapi import APIRouter, HTTPException, Query, Depends
from pydantic import BaseModel, Field

RAG_DIR = Path(__file__).parent / "rag"
sys.path.insert(0, str(RAG_DIR))
import safety, checker, pipeline                    # noqa: E402
from retriever import Retriever                     # noqa: E402
from llm import groq_llm                            # noqa: E402
from app import database, plan_db                   # noqa: E402

router = APIRouter()
LLM_FN = groq_llm                                    # swapped for a fake in tests
ADMIN_KEY = os.environ.get("ADMIN_KEY")
STUDY_DELAY_S = float(os.environ.get("STUDY_DELAY_S", "6"))
_ret, _ret_lock = None, threading.Lock()
_study = {"state": "idle", "done": 0, "total": 0, "error": None}


def require_admin_key(key: Optional[str] = Query(default=None)):
    if ADMIN_KEY and key != ADMIN_KEY:
        raise HTTPException(status_code=401, detail="Missing or invalid admin key")
    return True


def init_plan_tables():
    plan_db.init_tables()


def get_retriever():
    global _ret
    with _ret_lock:
        if _ret is None: _ret = Retriever()
        return _ret


def _llm(messages, **kw):
    try:
        return LLM_FN(messages, **kw) if kw else LLM_FN(messages)
    except HTTPException:
        raise
    except Exception as e:                      # always answer with readable JSON, never a bare crash
        raise HTTPException(status_code=503, detail=str(e) if isinstance(e, RuntimeError) else f"{type(e).__name__}: {e}")


def fallback_text(structured):
    t = structured["TARGETS"]
    lines = ["We could not produce a personalised plan that passes our nutrient and safety checks, so here is general guidance only.",
             f"- Keep sodium under {t['sodium_max_mg']} mg per day: choose fresh foods, check labels and limit processed foods.",
             "- Follow the DASH eating pattern: plenty of vegetables, fruits, whole grains, low-fat dairy, beans and nuts.",
             "- Discuss any diet change with your doctor or a registered dietitian. This is education, not a diagnosis."]
    if structured["kidney_safe_mode"]:
        lines.append("- Your kidney function result is low or unknown, so please ask a kidney dietitian before changing potassium or protein intake.")
    return "\n".join(lines)


def _inputs(a):
    profile = json.loads(a["profile_json"]); prefs = json.loads(a["preferences_json"] or "{}")
    factors = [(f, s) for f, s in json.loads(a["top_factors_json"])]
    return profile, prefs, pipeline.build_structured_input(profile, a["risk_probability"], factors, prefs)


def _store(structured, res, assessment_id, parent=None):
    if res["status"] == "approved_by_checker":
        text = pipeline.render_plan_text(res["plan"])
        return plan_db.save_plan(assessment_id, parent, "rag_checker", res["status"], structured, res["plan"], text, res["totals"], res["sources"], res["attempts"]), text
    text = fallback_text(structured)
    return plan_db.save_plan(assessment_id, parent, "rag_checker", "rejected_fallback", structured, None, text, None, None, res["attempts"]), text


class PlanRequest(BaseModel):
    assessment_id: int


@router.get("/plan/status")
def plan_status():
    r = None
    try: r = get_retriever()
    except Exception as e: return {"retriever": f"error: {type(e).__name__}", "llm_key_present": bool(os.environ.get("GROQ_API_KEY"))}
    return {"retriever": r.mode, "passages": len(r.docs), "llm_key_present": bool(os.environ.get("GROQ_API_KEY")), "model": os.environ.get("LLM_MODEL", "openai/gpt-oss-120b")}


@router.post("/plan")
def make_plan(req: PlanRequest):
    a = database.get_assessment(req.assessment_id)
    if not a: raise HTTPException(404, "Assessment not found")
    try:
        _, _, structured = _inputs(a)
        res = pipeline.generate_plan(structured, get_retriever(), _llm)
        pid, text = _store(structured, res, req.assessment_id)
    except HTTPException:
        raise
    except Exception as e:
        import traceback; traceback.print_exc()          # full detail goes to the Railway logs
        raise HTTPException(500, f"Plan error: {type(e).__name__}: {str(e)[:200]}")
    print("PLAN", res["status"], "attempts", res["attempts"], "history", json.dumps(res.get("history"))[:1500], flush=True)   # visible in Railway logs
    return {"plan_id": pid, "status": res["status"], "attempts": res["attempts"], "text": text, "sources": res.get("sources"),
            "history": res.get("history"), "foods_matched": res.get("foods_matched"), "totals": res.get("totals"), "kidney_safe_mode": structured["kidney_safe_mode"], "kidney_note": structured["kidney_note"],
            "disclaimer": "Education only. Not a diagnosis. Review any change with a clinician or dietitian."}


class ChatRequest(BaseModel):
    plan_id: int
    message: str = Field(min_length=1, max_length=500)


_WHY = re.compile(r"\b(why|how come|explain|what does|what is|what's|reason)\b", re.I)


@router.post("/chat")
def chat(req: ChatRequest):
    p = plan_db.get_plan(req.plan_id)
    if not p: raise HTTPException(404, "Plan not found")
    structured = json.loads(p["structured_json"])
    if _WHY.search(req.message):
        ans = pipeline.answer_why(req.message, structured, get_retriever(), _llm)
        plan_db.save_feedback(req.plan_id, "why", req.message, ans["status"])
        return {"mode": "why", **ans}
    previous = json.loads(p["plan_json"]) if p["plan_json"] else None
    res = pipeline.generate_plan(structured, get_retriever(), _llm, feedback=req.message, previous=previous)
    pid, text = _store(structured, res, p["assessment_id"], parent=req.plan_id)
    plan_db.save_feedback(req.plan_id, "revise", req.message, res["status"], pid)
    return {"mode": "revise", "plan_id": pid, "status": res["status"], "attempts": res["attempts"], "text": text, "sources": res.get("sources"),
            "note": None if res["status"] == "approved_by_checker" else "Your request could not be applied safely; the standard guidance is shown."}


class FeedbackRequest(BaseModel):
    plan_id: int
    clarity: int = Field(ge=1, le=5)
    relevance: int = Field(ge=1, le=5)
    practicality: int = Field(ge=1, le=5)
    helpful: Optional[bool] = None
    comment: Optional[str] = Field(default="", max_length=1000)


@router.post("/feedback")
def feedback(req: FeedbackRequest):
    if not plan_db.get_plan(req.plan_id): raise HTTPException(404, "Plan not found")
    fid = plan_db.save_feedback(req.plan_id, "rating", req.comment, None, None, req.clarity, req.relevance, req.practicality, req.helpful)
    return {"id": fid, "status": "saved"}


# ---------- clinician / dietitian review ----------
class ReviewRequest(BaseModel):
    action: str = Field(pattern="^(approved|edited|rejected)$")
    text: Optional[str] = None
    note: Optional[str] = ""


@router.get("/review/pending")
def review_pending(_a: bool = Depends(require_admin_key)):
    return plan_db.list_pending()


@router.post("/review/{plan_id}")
def review(plan_id: int, req: ReviewRequest, _a: bool = Depends(require_admin_key)):
    if req.action == "edited" and not req.text: raise HTTPException(400, "Edited plans need text")
    if not plan_db.set_review(plan_id, req.action, req.text, req.note): raise HTTPException(404, "Plan not found")
    return {"status": "saved"}


@router.get("/plan/{plan_id}")
def get_plan(plan_id: int):
    p = plan_db.get_plan(plan_id)
    if not p: raise HTTPException(404, "Plan not found")
    final = p["reviewed_text"] if p["review_status"] == "edited" and p["reviewed_text"] else p["text"]
    return {"plan_id": plan_id, "text": final, "review_status": p["review_status"], "reviewer_note": p["reviewer_note"]}


# ---------- 3-system rating study ----------
def _code():
    return "P" + "".join(random.choices(string.ascii_uppercase + string.digits, k=6))


def _person(structured, label):
    f = ", ".join(x["factor"] + " (" + x["direction"] + ")" for x in structured["top_factors"])
    extra = " Kidney function is low, so kidney-safe rules apply." if structured["kidney_safe_mode"] else ""
    return f"{label}. Risk: {structured['risk_tier']}. Top factors: {f}. Cuisine: {structured['cuisine'] or 'any'}; diet type: {structured['diet_type'] or 'any'}.{extra}"


def _generate_study():
    from app.main import SCENARIOS
    from app.ml.risk_model import run_risk_prediction, run_explainability
    from app.ml.dietary_guidance import generate_core_guidance, personalize_presentation, format_report

    def paced(messages, **kw):
        time.sleep(STUDY_DELAY_S); return LLM_FN(messages, **kw) if kw else LLM_FN(messages)
    try:
        _study.update(state="running", done=0, total=len(SCENARIOS), error=None)
        plan_db.clear_study_items(); R = get_retriever()
        for s in SCENARIOS:
            prof, prefs = s["profile"], s["preferences"]
            rk = run_risk_prediction(prof); ex = run_explainability(prof)
            st = pipeline.build_structured_input(prof, rk["risk_probability"], ex["top_factors"], prefs)
            person = _person(st, s["label"]); cap = st["TARGETS"]["sodium_max_mg"]
            g = personalize_presentation(generate_core_guidance(ex["top_factors"]), prefs)
            plan_db.save_study_item(_code(), s["label"], person, "template", format_report(g), None, None, "template", 0)
            try:
                b = pipeline.llm_alone(st, paced)
                plan_db.save_study_item(_code(), s["label"], person, "llm_alone", pipeline.render_plan_text(b["plan"]), b["totals"],
                                        b["totals"]["sodium_mg"] <= cap, "unchecked", 1)
            except Exception as e:
                plan_db.save_study_item(_code(), s["label"], person, "llm_alone", f"[generation failed: {type(e).__name__}]", None, None, "failed", 0)
            res = pipeline.generate_plan(st, R, paced)
            if res["status"] == "approved_by_checker":
                plan_db.save_study_item(_code(), s["label"], person, "rag_checker", pipeline.render_plan_text(res["plan"]), res["totals"],
                                        res["totals"]["sodium_mg"] <= cap, res["status"], res["attempts"])
            else:
                plan_db.save_study_item(_code(), s["label"], person, "rag_checker", fallback_text(st), None, None, "rejected_fallback", res["attempts"])
            _study["done"] += 1
        _study["state"] = "done"
    except Exception as e:
        _study.update(state="error", error=f"{type(e).__name__}: {e}")


@router.post("/study/generate")
def study_generate(_a: bool = Depends(require_admin_key)):
    if _study["state"] == "running": raise HTTPException(409, "Already running")
    threading.Thread(target=_generate_study, daemon=True).start()
    return {"status": "started"}


@router.get("/study/status")
def study_status(_a: bool = Depends(require_admin_key)):
    return _study


@router.get("/study/task")
def study_task(rater: str = Query(min_length=1, max_length=60)):
    items = plan_db.list_study_items()
    done = plan_db.rated_codes(rater)
    random.Random(rater).shuffle(items)                    # each rater gets a stable, different order
    return {"rater": rater, "total": len(items), "rated": len(done & {i["plan_code"] for i in items}),
            "items": [{"plan_code": i["plan_code"], "person": i["person_summary"], "text": i["text"], "already_rated": i["plan_code"] in done} for i in items]}


class RatingRequest(BaseModel):
    rater_id: str = Field(min_length=1, max_length=60)
    plan_code: str
    clarity: int = Field(ge=1, le=5)
    relevance: int = Field(ge=1, le=5)
    practicality: int = Field(ge=1, le=5)
    adherence: int = Field(ge=1, le=5)
    safety: int = Field(ge=1, le=5)
    comments: Optional[str] = Field(default="", max_length=1000)


@router.post("/ratings")
def ratings(req: RatingRequest):
    if req.plan_code not in {i["plan_code"] for i in plan_db.list_study_items()}: raise HTTPException(404, "Unknown plan code")
    rid = plan_db.upsert_rating(req.rater_id, req.plan_code, req.clarity, req.relevance, req.practicality, req.adherence, req.safety, req.comments or "")
    return {"id": rid, "status": "saved"}


@router.get("/study/summary")
def study_summary(_a: bool = Depends(require_admin_key)):
    return {**plan_db.study_summary(), "feedback": plan_db.feedback_summary()}
