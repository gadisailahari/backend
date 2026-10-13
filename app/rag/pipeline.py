"""Grounded, checked dietary guidance pipeline.

profile + risk + SHAP factors  ->  structured input (risk tier, top-3 factors, intake gaps, diet type)
  -> retrieve guideline passages (RAG) + pinned cited facts
  -> LLM drafts JSON plan
  -> checks: JSON valid, citations real, text screen (no medication advice / kidney rules),
             foods resolved to USDA FNDDS, nutrients recomputed, targets enforced
  -> on failure the checker's reasons go back to the LLM (up to max_tries); if it still fails the plan is
     REJECTED and the caller must show the safe fallback, never the failed plan.
"""
import json, re
import safety, checker
from retriever import pinned_facts

LABEL = {"RIDAGEYR": "age", "BMXBMI": "body mass index", "BMXWAIST": "waist circumference", "DR1TSODI": "sodium intake",
         "DR1TPOTA": "potassium intake", "DR1TKCAL": "calorie intake", "DR1TALCO": "alcohol intake", "LBXSCR": "serum creatinine",
         "INDFMPIR": "income ratio", "RIAGENDR": "sex", "RIDRETH3": "ethnicity", "DR1TPROT": "protein intake",
         "DR1TCARB": "carbohydrate intake", "DR1TTFAT": "fat intake", "SMQ020": "smoking history", "PAQ650": "vigorous activity",
         "PAQ665": "moderate activity", "ALQ121": "drinking frequency"}
CUISINE_REGION = {"south asian": ["India", "Bangladesh"], "indian": ["India"], "bangladeshi": ["Bangladesh"], "japanese": ["Japan"],
                  "brazilian": ["Brazil"], "latin american": ["Brazil"], "malaysian": ["Malaysia"]}

SYSTEM = """You write dietary guidance for a hypertension education tool. You are not a doctor.
RULES
1. For any guideline claim or number, use ONLY the TARGETS and PASSAGES provided, and cite them like [P1] or [F2].
2. NEVER give medication advice. Never tell the user to start, stop or change any medicine or dose.
3. If kidney_safe_mode is true: do NOT recommend increasing potassium, potassium-rich foods, salt substitutes or potassium supplements; say to discuss diet with a kidney dietitian.
4. This is education, not a diagnosis.
5. Respect the user's cuisine, diet type and dislikes. Evidence stays the same; only the food examples change.
6. Reply with JSON only, in exactly this shape:
{"summary": "2-3 plain sentences on why risk is as it is, based on the top factors",
 "tips": [{"text": "...", "cite": ["P1"]}],
 "meal_plan": {"meals": [{"name": "Breakfast", "items": [{"food": "common food name", "grams": 150}]}]}}
Give 3-5 tips and a one-day plan (3 meals + 1 snack). Total sodium for the day must stay under TARGETS.sodium_max_mg."""


def risk_tier(p):
    return "high" if p >= 0.5 else "moderate" if p >= 0.33 else "lower"


def build_structured_input(profile, risk_probability, top_factors, prefs=None):
    prefs = prefs or {}
    k = safety.kidney_status(profile.get("LBXSCR"), profile.get("RIDAGEYR"), profile.get("RIAGENDR"))
    t = safety.nutrient_targets(k["ckd_mode"], profile["RIAGENDR"])
    gaps = []
    if profile.get("DR1TSODI") is not None:
        d = profile["DR1TSODI"] - t["sodium_max_mg"]
        gaps.append({"nutrient": "sodium", "intake_mg": round(profile["DR1TSODI"]), "limit_mg": t["sodium_max_mg"],
                     "status": f"{d:.0f} mg over the limit" if d > 0 else "within the limit"})
    if not k["ckd_mode"] and profile.get("DR1TPOTA") is not None and t["potassium_target_mg"]:
        d = t["potassium_target_mg"] - profile["DR1TPOTA"]
        gaps.append({"nutrient": "potassium", "intake_mg": round(profile["DR1TPOTA"]), "target_mg": t["potassium_target_mg"],
                     "status": f"{d:.0f} mg below the target" if d > 0 else "meets the target"})
    top3 = [{"factor": LABEL.get(f, f), "shap": round(s, 3), "direction": "raises risk" if s > 0 else "lowers risk"} for f, s in top_factors[:3]]
    return {"risk_tier": risk_tier(risk_probability), "risk_probability": round(risk_probability, 3), "top_factors": top3,
            "intake_gaps": gaps, "cuisine": prefs.get("cuisine"), "diet_type": prefs.get("constraint"), "dislikes": prefs.get("dislikes") or [],
            "kidney_safe_mode": k["ckd_mode"], "kidney_note": k["reason"], "egfr": k["egfr"], "TARGETS": t}


def retrieve(structured, retriever, k_per=2, max_passages=6):
    qs = ["daily sodium limit hypertension diet", "DASH eating plan servings food groups"]
    qs += [f"{f['factor']} blood pressure diet" for f in structured["top_factors"] if f["direction"] == "raises risk"]
    if structured["kidney_safe_mode"]:
        qs += ["chronic kidney disease sodium potassium diet", "kidney disease healthy eating"]
    roles = None
    regions = CUISINE_REGION.get((structured.get("cuisine") or "").lower())
    seen, out = set(), []
    for q in qs:
        for r in retriever.search(q, k=k_per):
            if r["id"] not in seen: seen.add(r["id"]); out.append(r)
    if regions:
        for r in retriever.search(f"{structured['cuisine']} traditional foods healthy meals", k=2, regions=regions):
            if r["id"] not in seen: seen.add(r["id"]); out.append(r)
    out = out[:max_passages]
    labelled = {f"P{i+1}": p for i, p in enumerate(out)}
    facts = pinned_facts(structured["kidney_safe_mode"])
    for i, f in enumerate(facts): labelled[f"F{i+1}"] = {**f, "text": f["text"]}
    return labelled


def _passage_block(labelled):
    lines = []
    for lab, p in labelled.items():
        txt = " ".join(p["text"].split()[:110])
        lines.append(f"[{lab}] ({p['title']}, p{p['page']}) {txt}")
    return "\n".join(lines)


def _parse_json(raw):
    m = re.search(r"\{.*\}", raw, re.S)
    if not m: raise ValueError("no JSON object found")
    obj = json.loads(m.group(0))
    for key in ("summary", "tips", "meal_plan"):
        if key not in obj: raise ValueError(f"missing key '{key}'")
    return obj


def _resolve_foods(obj):
    """Map each LLM food name to a USDA FNDDS code. Returns plan in checker format + resolution log."""
    meals, log = [], []
    for m in obj["meal_plan"].get("meals", []):
        items = []
        for it in m.get("items", []):
            name, g = str(it.get("food", "")), it.get("grams")
            hit = checker.search_foods(name, k=1)
            if hit:
                items.append({"food_code": hit[0]["food_code"], "grams": g}); log.append({"asked": name, "matched": hit[0]["desc"], "grams": g})
        meals.append({"name": m.get("name", "meal"), "items": items})
    return {"meals": meals}, log


def _text_of(obj):
    return " ".join([str(obj.get("summary", ""))] + [str(t.get("text", "")) for t in obj.get("tips", [])])


def check_draft(obj, labelled, structured, enforce=True):
    """All the gates. Returns (violations, plan_for_checker, resolution_log, check_result)."""
    v = []
    for t in obj.get("tips", []):
        cites = t.get("cite") or []
        if not cites: v.append(f"Tip without a citation: '{str(t.get('text',''))[:60]}'")
        for c in cites:
            if c not in labelled: v.append(f"Citation [{c}] does not exist; cite only the labelled passages.")
    for kind, snippet in safety.screen_text(_text_of(obj), structured["kidney_safe_mode"]):
        v.append(f"Not allowed ({kind}): '{snippet}'")
    plan, log = _resolve_foods(obj)
    res = checker.check_plan(plan, structured["TARGETS"], structured["kidney_safe_mode"])
    v += res["violations"]
    return v, plan, log, res


def generate_plan(structured, retriever, llm_fn, feedback=None, previous=None, max_tries=3):
    labelled = retrieve(structured, retriever)
    content = "INPUT:\n" + json.dumps({k: v for k, v in structured.items()}, indent=1) + "\n\nPASSAGES AND FACTS:\n" + _passage_block(labelled)
    if feedback:
        content += f"\n\nUSER FEEDBACK ON THE PREVIOUS PLAN (apply it only if it does not break the rules): {feedback}"
        if previous: content += "\nPREVIOUS SUMMARY: " + str(previous.get("summary", ""))[:400]
    messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": content}]
    history = []
    for attempt in range(1, max_tries + 1):
        raw = llm_fn(messages)
        try:
            obj = _parse_json(raw)
        except Exception as e:
            history.append({"attempt": attempt, "violations": [f"Invalid JSON: {e}"]})
            messages += [{"role": "assistant", "content": raw}, {"role": "user", "content": "Your reply was not valid JSON in the required shape. Return JSON only."}]
            continue
        viol, plan, log, res = check_draft(obj, labelled, structured)
        if not viol:
            return {"status": "approved_by_checker", "attempts": attempt, "plan": obj, "foods_matched": log, "totals": res["totals"],
                    "per_item": res["per_item"], "sources": {k: {"title": p["title"], "page": p["page"]} for k, p in labelled.items()},
                    "history": history}
        history.append({"attempt": attempt, "violations": viol})
        messages += [{"role": "assistant", "content": raw}, {"role": "user", "content": checker.retry_hint({"violations": viol}) +
                     "\nFoods were matched to USDA entries as: " + "; ".join(f"{l['asked']} -> {l['matched']}" for l in log[:12])}]
    return {"status": "rejected", "attempts": max_tries, "plan": None, "history": history,
            "message": "The system could not produce a plan that passes the nutrient and safety checks. Showing standard guidance instead."}


def answer_why(question, structured, retriever, llm_fn, max_tries=2):
    labelled = retrieve(structured, retriever)
    sys = ("You answer a user's 'why' question about their hypertension risk guidance. Use ONLY the INPUT and the cited PASSAGES. "
           "Cite like [P1]. Never give medication advice. Not a diagnosis. If kidney_safe_mode is true, never advise more potassium. "
           'Reply JSON: {"answer": "...", "cite": ["P1"]}')
    content = "INPUT:\n" + json.dumps(structured, indent=1) + "\n\nPASSAGES AND FACTS:\n" + _passage_block(labelled) + "\n\nQUESTION: " + question
    messages = [{"role": "system", "content": sys}, {"role": "user", "content": content}]
    for attempt in range(1, max_tries + 1):
        raw = llm_fn(messages)
        try:
            obj = json.loads(re.search(r"\{.*\}", raw, re.S).group(0)); ans = str(obj["answer"]); cites = obj.get("cite") or []
            v = [f"Citation [{c}] does not exist." for c in cites if c not in labelled]
            if not cites: v.append("Answer has no citation.")
            v += [f"Not allowed ({k}): '{s}'" for k, s in safety.screen_text(ans, structured["kidney_safe_mode"])]
        except Exception as e:
            v, ans = [f"Invalid JSON: {e}"], ""
        if not v:
            return {"status": "ok", "answer": ans, "sources": {c: {"title": labelled[c]["title"], "page": labelled[c]["page"]} for c in cites}}
        messages += [{"role": "assistant", "content": raw}, {"role": "user", "content": "Rejected: " + "; ".join(v) + ". Fix and return JSON only."}]
    return {"status": "refused", "answer": "I can't answer that safely from the approved guidance. Please ask a clinician or dietitian."}


def llm_alone(structured, llm_fn):
    """Study baseline: same LLM, no passages, no targets, no checker, no safety screen. Nutrients are still measured afterwards."""
    basic = {k: structured[k] for k in ("risk_tier", "top_factors", "cuisine", "diet_type", "dislikes")}
    msg = [{"role": "system", "content": "You write dietary advice for people at risk of hypertension. Reply with JSON only in this shape: "
            '{"summary": "...", "tips": [{"text": "..."}], "meal_plan": {"meals": [{"name": "Breakfast", "items": [{"food": "...", "grams": 150}]}]}}'},
           {"role": "user", "content": "Write dietary advice and a one-day meal plan for this person with hypertension risk:\n" + json.dumps(basic)}]
    obj = _parse_json(llm_fn(msg))
    plan, log = _resolve_foods(obj)
    res = checker.check_plan(plan, structured["TARGETS"], structured["kidney_safe_mode"])   # measured, not enforced
    return {"plan": obj, "foods_matched": log, "totals": res["totals"], "violations_if_checked": res["violations"]}


def render_plan_text(obj):
    lines = [obj.get("summary", ""), ""]
    for t in obj.get("tips", []): lines.append("- " + str(t.get("text", "")) + (" [" + ", ".join(t.get("cite", [])) + "]" if t.get("cite") else ""))
    lines.append("")
    for m in obj.get("meal_plan", {}).get("meals", []):
        lines.append(f"{m.get('name','Meal')}: " + "; ".join(f"{i.get('food')} ({i.get('grams')} g)" for i in m.get("items", [])))
    return "\n".join(lines).strip()
