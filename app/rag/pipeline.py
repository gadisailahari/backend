"""Grounded, checked eating-and-exercise plan pipeline.

profile + risk + SHAP factors + preferences  ->  structured input
  -> vetted FOOD MENU (filtered for diet, cuisine, conditions, dislikes, kidney rules) + EXERCISE MENU
  -> retrieve guideline passages (RAG) + pinned cited facts
  -> LLM drafts a JSON plan (meal options, exercise, tips)
  -> checks: valid JSON, only menu foods, amounts plausible, sodium worst case under the limit, protein and calorie ranges,
             exercise targets, citations real, no medication advice / kidney rules (text screen)
  -> on failure the reasons go back to the LLM (up to max_tries); if it still fails the plan is REJECTED and the caller shows safe fallback text.
Foods the AI picks are looked up by id in the menu, so the numbers shown to the user are data, not AI claims.
"""
import json, re
import safety, checker, menu, plan_check, rule_plan
from retriever import plan_facts

LABEL = {"RIDAGEYR": "age", "BMXBMI": "body mass index", "BMXWAIST": "waist circumference", "DR1TSODI": "sodium intake",
         "DR1TPOTA": "potassium intake", "DR1TKCAL": "calorie intake", "DR1TALCO": "alcohol intake", "LBXSCR": "serum creatinine (a kidney blood test)",
         "INDFMPIR": "income ratio", "RIAGENDR": "sex", "RIDRETH3": "ethnicity", "DR1TPROT": "protein intake",
         "DR1TCARB": "carbohydrate intake", "DR1TTFAT": "fat intake", "SMQ020": "smoking history", "PAQ650": "vigorous activity",
         "PAQ665": "moderate activity", "ALQ121": "drinking frequency"}
CUISINE_REGION = {"south asian": ["India", "Bangladesh"], "indian": ["India"], "bangladeshi": ["Bangladesh"], "japanese": ["Japan"],
                  "brazilian": ["Brazil"], "latin american": ["Brazil"], "malaysian": ["Malaysia"]}
CONDITIONS = {"pcos", "diabetes", "cholesterol", "pregnant"}
SODIUM_NOTE = ("Sodium numbers use USDA values for foods as usually prepared, which include normal cooking salt. "
               "Cooking without added salt makes them lower.")

SYSTEM = """You write a short, practical eating and exercise plan for a hypertension-education tool. You are not a doctor. Be brief: no stories, no filler.
RULES
1. Build meals ONLY from the FOOD MENU: use the id and grams. Each option has 2 to 6 items, amounts near the listed portion. Use foods marked * first (they fit the user's cuisine). The menu is already filtered for the user's diet, conditions and dislikes.
2. Give 2 options each for breakfast, lunch and dinner, and 1 or 2 for snack. Option names are short, like "Masala oats with banana".
3. Sodium: take the highest-sodium option of each meal and add them. The total must stay under TARGETS.sodium_max_mg. Budget per meal (mg): SLOT_BUDGET_MG. The Na column is mg per listed portion, so pick low-Na foods and keep salty items small.
4. Protein: aim for PROTEIN_TARGET_G per day if it is given (spread over meals). In kidney-safe mode PROTEIN_TARGET_G is a ceiling: stay at or below high_g and use small portions of protein foods. Vegetarian and vegan plans use the plant and dairy protein foods on the menu; non-vegetarian plans may use meat and fish on the menu.
5. Exercise: choose 3 to 6 items from the EXERCISE MENU with minutes and days per week. Meet EXERCISE_TARGETS (aerobic minutes per week = sum of minutes x days over aerobic items; strength days = days over resistance items). Put the user's preferred activities (*) first. Do not guess interests from sex.
6. Tips: 2 to 4 tips, each under 20 words, each with cite labels from the FACTS and PASSAGES (like F1 or P2). Cite only labels that exist.
7. NEVER give medication advice. If kidney_safe_mode is true: do not recommend potassium, potassium-rich foods, salt substitutes or supplements; say to ask a kidney dietitian.
8. This is education, not a diagnosis.
Reply with ONE JSON object only, nothing before or after it:
{"summary": "max 2 short sentences", "meals": {"breakfast": [{"name": "...", "items": [{"id": "oatmeal", "g": 240}]}], "lunch": [...], "dinner": [...], "snack": [...]},
 "exercise": [{"id": "brisk_walk", "minutes": 30, "days": 5}], "tips": [{"text": "...", "cite": ["F1"]}]}"""


def _weight(x):
    try: w = float(x)
    except (TypeError, ValueError): return None
    return round(w, 1) if 30 <= w <= 250 else None


def risk_tier(p):
    return "high" if p >= 0.5 else "moderate" if p >= 0.33 else "lower"


def _clean_list(x, allowed=None):
    if isinstance(x, str): x = re.split(r"[,;]", x)
    out = [str(t).strip().lower() for t in (x or []) if str(t).strip()]
    return [t for t in out if (allowed is None or t in allowed)]


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
    conds = _clean_list(prefs.get("conditions"), CONDITIONS)
    fit = str(prefs.get("fitness") or "").strip().lower()
    st = {"risk_tier": risk_tier(risk_probability), "risk_probability": round(risk_probability, 3), "top_factors": top3,
          "intake_gaps": gaps, "cuisine": prefs.get("cuisine"), "diet_type": prefs.get("constraint"), "dislikes": prefs.get("dislikes") or [],
          "conditions": conds, "activities": _clean_list(prefs.get("activities"), set(menu.ACTIVITY_KEYS)),
          "fitness": fit if fit in ("beginner", "active") else "beginner",
          "kcal_reference": profile.get("DR1TKCAL"), "weight_kg": _weight(prefs.get("weight_kg")),
          "kidney_safe_mode": k["ckd_mode"], "kidney_note": k["reason"], "egfr": k["egfr"], "TARGETS": t,
          "needs_clinician": "pregnant" in conds}
    st["PROTEIN_TARGET_G"] = menu.protein_target(st)
    st["EXERCISE_TARGETS"] = menu.exercise_targets(st)
    st["SLOT_BUDGET_MG"] = menu.SLOT_BUDGET_MG[bool(k["ckd_mode"])]
    return st


def retrieve(structured, retriever, k_per=2, max_passages=4):
    qs = ["daily sodium limit hypertension diet", "DASH eating plan servings food groups"]
    qs += [f"{f['factor']} blood pressure diet" for f in structured["top_factors"] if f["direction"] == "raises risk"]
    if structured["kidney_safe_mode"]:
        qs += ["chronic kidney disease sodium potassium diet", "kidney disease healthy eating"]
    regions = CUISINE_REGION.get((structured.get("cuisine") or "").lower())
    seen, out = set(), []
    for q in qs:
        for r in retriever.search(q, k=k_per):
            if r["id"] not in seen: seen.add(r["id"]); out.append(r)
    if regions:
        for r in retriever.search(f"{structured['cuisine']} traditional foods healthy meals", k=2, regions=regions):
            if r["id"] not in seen: seen.add(r["id"]); out.append(r)
    cond_src = {"pcos": (["pcos_2023"], "PCOS healthy eating and physical activity recommendations"),
                "cholesterol": (["dyslipidemia_2026", "aha_ldl_guide"], "dietary pattern and lifestyle to lower LDL cholesterol"),
                "diabetes": (["ada_2026_s5"], "nutrition therapy and physical activity for people with diabetes")}
    front = []
    for c in structured.get("conditions") or []:
        if c in cond_src:
            for r in retriever.search(cond_src[c][1], k=1, sources=cond_src[c][0]):
                if r["id"] not in seen: seen.add(r["id"]); front.append(r)
    out = (front + out)[:max_passages]
    labelled = {f"P{i+1}": p for i, p in enumerate(out)}
    for i, f in enumerate(plan_facts(structured)): labelled[f"F{i+1}"] = {**f, "text": f["text"]}
    return labelled


def _passage_block(labelled, words=70):
    return "\n".join(f"[{lab}] ({p['title']}, p{p['page']}) " + " ".join(p["text"].split()[:words]) for lab, p in labelled.items())


def _parse_json(raw):
    """Take the FIRST JSON object in the reply (models sometimes add text or a second object)."""
    s = str(raw).strip()
    s = re.sub(r"^```(?:json)?", "", s).strip()
    i = s.find("{")
    if i < 0: raise ValueError("no JSON object found")
    obj, _ = json.JSONDecoder().raw_decode(s[i:])
    for key in ("summary", "meals", "exercise", "tips"):
        if key not in obj: raise ValueError(f"missing key '{key}'")
    return obj


def _text_of(obj):
    names = [str(o.get("name", "")) for opts in (obj.get("meals") or {}).values() if isinstance(opts, list) for o in opts if isinstance(o, dict)]
    return " ".join([str(obj.get("summary", ""))] + [str(t.get("text", "")) for t in obj.get("tips", []) if isinstance(t, dict)] + names)


def check_draft(obj, labelled, structured, allowed):
    v = []
    tips = obj.get("tips") if isinstance(obj.get("tips"), list) else []
    if not (2 <= len(tips) <= 5): v.append("Give 2 to 4 tips.")
    for t in tips:
        cites = (t.get("cite") if isinstance(t, dict) else None) or []
        if not cites: v.append(f"Tip without a citation: '{str(t.get('text',''))[:60]}'")
        for c in cites:
            if c not in labelled: v.append(f"Citation [{c}] does not exist; cite only labels that were provided.")
    for kind, snippet in safety.screen_text(_text_of(obj), structured["kidney_safe_mode"]):
        v.append(f"Not allowed ({kind}): '{snippet}'")
    res = plan_check.check_plan_v2(obj, allowed, structured)
    return v + res["violations"], res


def _protein_view(pt, structured):
    if not pt: return {"text": ""}
    if not structured.get("kidney_safe_mode"):
        return {"low_g": pt["low_g"], "high_g": pt["high_g"],
                "text": f"Aim for about {pt['low_g']}-{pt['high_g']} g of protein a day (15-25% of calories, inside the 10-35% guideline range)."}
    tail = (" This follows KDIGO 2024 (Recommendation 3.3.1.1: 0.8 g of protein per kg of body weight per day for adults with chronic kidney disease stages 3-5 who are not on dialysis). "
            "Please have your doctor or kidney dietitian confirm your number. If you are on dialysis, your protein needs are higher, so follow your dialysis team.")
    if pt.get("basis") == "weight":
        return {"low_g": pt["low_g"], "high_g": pt["high_g"],
                "text": f"Kidney-safe protein: about {pt['target_g']} g a day (0.8 g x your weight of {pt['weight_kg']:g} kg). Staying between {pt['low_g']} and {pt['high_g']} g is fine." + tail}
    return {"low_g": pt["low_g"], "high_g": pt["high_g"],
            "text": f"Kidney-safe protein: keep it to about {pt['low_g']}-{pt['high_g']} g a day (a cautious cap of 10-15% of calories, because we do not have your weight). Add your weight on the form to get your exact number." + tail}


def build_view(obj, res, structured, labelled):
    """What the website shows. Numbers come from the menu data; sources come from the labels the AI cited."""
    pt = structured.get("PROTEIN_TARGET_G")
    meals = {s: [{"name": o["name"], "items": [{"label": menu.FOODS[i["id"]]["label"], "grams": round(i["g"])} for i in o["items"]],
                  "kcal": round(o["kcal"]), "protein_g": round(o["protein_g"]), "sodium_mg": round(o["sodium_mg"])} for o in opts]
             for s, opts in res["options"].items()}
    facts = {f["id"]: f for f in json.load(open(__import__("pathlib").Path(__file__).parent / "data/key_facts.json"))}
    avoid = [{"text": t, "source": {"title": facts[f]["title"], "page": facts[f]["page"]}} for t, f in menu.avoid_list(structured) if f in facts]
    cited = {c for t in obj.get("tips", []) for c in (t.get("cite") or []) if c in labelled}
    return {"summary": str(obj.get("summary", "")),
            "protein": _protein_view(pt, structured),
            "meals": meals, "avoid": avoid,
            "exercise": res["exercise"]["items"], "exercise_note": menu.SAFETY_EXERCISE,
            "exercise_summary": {"aerobic_min_per_week": res["exercise"]["aerobic_min"], "strength_days_per_week": res["exercise"]["resistance_days"]},
            "tips": [{"text": str(t.get("text", "")), "cite": [c for c in (t.get("cite") or []) if c in labelled]} for t in obj.get("tips", [])],
            "totals": res["totals"], "sodium_note": SODIUM_NOTE,
            "sources": {c: {"title": labelled[c]["title"], "page": labelled[c]["page"]} for c in sorted(cited)}}


def generate_plan(structured, retriever, llm_fn, feedback=None, previous=None, max_tries=3, allow_rules=False):
    if structured.get("needs_clinician"):
        return {"status": "needs_clinician", "attempts": 0, "plan": None, "history": [],
                "message": "Pregnant or breastfeeding: please get an eating plan from your doctor or a registered dietitian."}
    allowed = menu.allowed_foods(structured)
    if len(allowed) < 15:
        return {"status": "rejected", "attempts": 0, "plan": None, "history": [{"attempt": 0, "violations": ["Too few allowed foods for these choices."]}]}
    labelled = retrieve(structured, retriever)
    compact = {k: structured[k] for k in ("risk_tier", "top_factors", "intake_gaps", "cuisine", "diet_type", "dislikes", "conditions", "activities", "fitness",
                                          "kidney_safe_mode", "TARGETS", "PROTEIN_TARGET_G", "EXERCISE_TARGETS", "SLOT_BUDGET_MG")}
    content = ("INPUT:\n" + json.dumps(compact) + "\n\nFOOD MENU:\n" + menu.menu_for_prompt(allowed, structured.get("cuisine")) +
               "\n\nEXERCISE MENU:\n" + menu.exercise_for_prompt(structured.get("activities")) +
               "\n\nFACTS AND PASSAGES:\n" + _passage_block(labelled))
    if feedback:
        content += f"\n\nUSER FEEDBACK ON THE PREVIOUS PLAN (apply it only if it does not break the rules): {feedback}"
        if previous: content += "\nPREVIOUS PLAN: " + json.dumps({"meals": previous.get("meals"), "exercise": previous.get("exercise")})[:1800]
    messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": content}]
    history = []
    for attempt in range(1, max_tries + 1):
        try:
            raw = llm_fn(messages)
        except Exception as e:
            if not allow_rules: raise
            history.append({"attempt": attempt, "violations": [f"AI service error: {str(getattr(e, 'detail', e))[:160]}"]})
            break
        try:
            obj = _parse_json(raw)
        except Exception as e:
            history.append({"attempt": attempt, "violations": [f"Invalid JSON: {e}"]})
            messages += [{"role": "assistant", "content": str(raw)[:1500]}, {"role": "user", "content": "Your reply was not one valid JSON object in the required shape. Return ONE JSON object only."}]
            continue
        viol, res = check_draft(obj, labelled, structured, allowed)
        if not viol:
            return {"status": "approved_by_checker", "attempts": attempt, "plan": obj, "view": build_view(obj, res, structured, labelled),
                    "totals": res["totals"], "sources": {c: {"title": p["title"], "page": p["page"]} for c, p in labelled.items()}, "history": history}
        history.append({"attempt": attempt, "violations": viol[:8]})
        messages += [{"role": "assistant", "content": json.dumps(obj)[:2500]},
                     {"role": "user", "content": "The checker rejected your plan:\n- " + "\n- ".join(viol[:8]) + "\nFix ONLY these problems and return ONE JSON object."}]
    if allow_rules:
        obj = rule_plan.build_plan(structured, allowed, labelled)
        if obj:
            viol, res = check_draft(obj, labelled, structured, allowed)
            if not viol:
                view = build_view(obj, res, structured, labelled)
                view["built_by"] = "rules"
                down = any("AI service error" in v for h in history for v in h["violations"])
                view["built_note"] = ("The AI writer was busy, so this plan was put together by our rule-based builder. It uses only the vetted food list and passes the same sodium, protein and safety checks. You can press Create plan again for an AI-written version." if down else
                                      "The AI draft did not pass our safety checks, so we used our rule-based builder instead. It uses only the vetted food list and passes the same sodium, protein and safety checks.")
                history.append({"attempt": "rules", "violations": ["Software-built plan used (passed all checks)."]})
                return {"status": "approved_by_checker", "attempts": max_tries, "built_by": "rules", "plan": obj, "view": view, "totals": res["totals"],
                        "sources": {c: {"title": p["title"], "page": p["page"]} for c, p in labelled.items()}, "history": history}
            history.append({"attempt": "rules", "violations": ["Software-built plan failed: " + "; ".join(v[:100] for v in viol[:3])]})
    return {"status": "rejected", "attempts": max_tries, "plan": None, "history": history,
            "message": "The system could not produce a plan that passes the nutrient and safety checks. Showing standard guidance instead."}


def answer_why(question, structured, retriever, llm_fn, max_tries=2):
    labelled = retrieve(structured, retriever)
    sys_ = ("You answer a user's 'why' question about their hypertension risk guidance. Be brief (3 sentences at most). Use ONLY the INPUT and the cited PASSAGES. "
            "Cite like [P1]. Never give medication advice. Not a diagnosis. If kidney_safe_mode is true, never advise more potassium. "
            'Reply with ONE JSON object: {"answer": "...", "cite": ["P1"]}')
    compact = {k: structured[k] for k in ("risk_tier", "top_factors", "intake_gaps", "kidney_safe_mode", "TARGETS", "conditions", "diet_type")}
    content = "INPUT:\n" + json.dumps(compact) + "\n\nPASSAGES AND FACTS:\n" + _passage_block(labelled) + "\n\nQUESTION: " + question
    messages = [{"role": "system", "content": sys_}, {"role": "user", "content": content}]
    for attempt in range(1, max_tries + 1):
        raw = llm_fn(messages)
        try:
            i = str(raw).find("{"); obj, _ = json.JSONDecoder().raw_decode(str(raw)[i:])
            ans = str(obj["answer"]); cites = obj.get("cite") or []
            v = [f"Citation [{c}] does not exist." for c in cites if c not in labelled]
            if not cites: v.append("Answer has no citation.")
            v += [f"Not allowed ({k}): '{s}'" for k, s in safety.screen_text(ans, structured["kidney_safe_mode"])]
        except Exception as e:
            v, ans = [f"Invalid JSON: {e}"], ""
        if not v:
            return {"status": "ok", "answer": ans, "sources": {c: {"title": labelled[c]["title"], "page": labelled[c]["page"]} for c in cites}}
        messages += [{"role": "assistant", "content": str(raw)[:1500]}, {"role": "user", "content": "Rejected: " + "; ".join(v) + ". Fix and return ONE JSON object."}]
    return {"status": "refused", "answer": "I can't answer that safely from the approved guidance. Please ask a clinician or dietitian."}


# ---------------- study baseline: same AI, no menu, no passages, no checker ----------------
def llm_alone(structured, llm_fn):
    """Same LLM, same person, but no food menu, no guideline passages, no targets, no checker, no safety screen.
    Free-text foods are looked up in USDA FNDDS afterwards so sodium can be MEASURED (not enforced)."""
    basic = {k: structured[k] for k in ("risk_tier", "top_factors", "cuisine", "diet_type", "dislikes", "conditions", "activities", "fitness")}
    shape = ('{"summary": "...", "protein_note": "...", "meals": {"breakfast": [{"name": "...", "items": [{"food": "...", "grams": 150}]}], "lunch": [...], "dinner": [...], "snack": [...]}, '
             '"avoid": ["..."], "exercise": [{"activity": "...", "minutes": 30, "days": 5}], "tips": [{"text": "..."}]}')
    msg = [{"role": "system", "content": "You write short eating and exercise advice for people at risk of hypertension. Be brief. Give 2 options each for breakfast, lunch and dinner and 1 snack option. Reply with ONE JSON object only in this shape: " + shape},
           {"role": "user", "content": "Write the plan for this person:\n" + json.dumps(basic)}]
    raw = str(llm_fn(msg)); obj, _ = json.JSONDecoder().raw_decode(raw[raw.find("{"):])
    meals, worst, kcal, prot = {}, 0.0, 0.0, 0.0
    for slot in plan_check.SLOTS:
        rows, slot_worst, slot_k, slot_p = [], 0.0, [], []
        for o in (obj.get("meals") or {}).get(slot, [])[:3]:
            na = k_ = p_ = 0.0; its = []
            for it in o.get("items", []):
                hit = checker.search_foods(str(it.get("food", "")), k=1)
                g = it.get("grams") if isinstance(it.get("grams"), (int, float)) else 100
                its.append({"label": str(it.get("food", "")), "grams": round(g)})
                if hit:
                    n = checker._NUT[hit[0]["food_code"]]; f = g / 100.0
                    na += n["sodium_mg"] * f; k_ += n["kcal"] * f; p_ += n["protein_g"] * f
            rows.append({"name": str(o.get("name", "")), "items": its, "kcal": round(k_), "protein_g": round(p_), "sodium_mg": round(na)})
            slot_worst = max(slot_worst, na); slot_k.append(k_); slot_p.append(p_)
        meals[slot] = rows
        worst += slot_worst
        if slot_k: kcal += sum(slot_k) / len(slot_k); prot += sum(slot_p) / len(slot_p)
    view = {"summary": str(obj.get("summary", "")), "protein": {"text": str(obj.get("protein_note", ""))}, "meals": meals,
            "avoid": [{"text": str(a), "source": None} for a in obj.get("avoid", [])],
            "exercise": [{"label": str(e.get("activity", "")), "minutes": e.get("minutes"), "days": e.get("days")} for e in obj.get("exercise", []) if isinstance(e, dict)],
            "exercise_note": "", "tips": [{"text": str(t.get("text", "")), "cite": []} for t in obj.get("tips", []) if isinstance(t, dict)]}
    totals = {"sodium_mg_worst_case": round(worst), "kcal": round(kcal), "protein_g": round(prot)}
    return {"plan": view, "totals": totals}


def render_plan_text(view):
    """Plain-text version of a plan view (used for the rating study and for storage)."""
    L = [view.get("summary", ""), ""]
    if view.get("protein", {}).get("text"): L += ["PROTEIN", view["protein"]["text"], ""]
    L.append("MEALS (pick one option for each meal)")
    for slot in plan_check.SLOTS:
        opts = (view.get("meals") or {}).get(slot) or []
        if not opts: continue
        L.append(slot.capitalize())
        for n, o in enumerate(opts):
            L.append(f"  {chr(65+n)}) {o.get('name','')}: " + "; ".join(f"{i['label']} {i['grams']} g" for i in o["items"]) +
                     f"  (about {o['kcal']} kcal, {o['protein_g']} g protein)")
    if view.get("avoid"):
        L += ["", "FOODS TO AVOID OR LIMIT"] + ["- " + a["text"] for a in view["avoid"]]
    if view.get("exercise"):
        L += ["", "EXERCISE (per week)"]
        for e in view["exercise"]: L.append(f"- {e['label']}: {e.get('minutes')} min x {e.get('days')} days")
        if view.get("exercise_note"): L.append(view["exercise_note"])
    if view.get("tips"):
        L += ["", "TIPS"] + ["- " + t["text"] + (" [" + ", ".join(t["cite"]) + "]" if t.get("cite") else "") for t in view["tips"]]
    return "\n".join(L).strip()
