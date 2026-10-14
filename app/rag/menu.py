"""Vetted food and exercise menus. The AI may only choose items from these lists, so every number it is judged on comes from data, not from the AI.

Food nutrients: USDA FNDDS 2017-18 (foods as usually prepared, which includes normal cooking salt), plus two foods from USDA FoodData Central
that FNDDS does not list as plain items (tempeh, paneer). See data/menu.json for the source of each food.
"""
import json, pathlib, re

HERE = pathlib.Path(__file__).parent
_DATA = json.load(open(HERE / "data/menu.json"))
FOODS = {f["id"]: f for f in _DATA["foods"]}

CKD_HIGH_K_MG_PER_PORTION = 450          # PLACEHOLDER - not from a guideline; a dietitian must replace it before real use
REFINED_STARCH = {"white_rice", "couscous", "dosa", "upma"}   # left out for PCOS / diabetes (DGA: make at least half of grains whole)

SLOT_BUDGET_MG = {False: {"breakfast": 450, "lunch": 750, "dinner": 750, "snack": 350},    # sums to 2,300
                  True:  {"breakfast": 400, "lunch": 650, "dinner": 650, "snack": 300}}    # sums to 2,000 (kidney-safe)

# ---- exercise menu --------------------------------------------------------------------
EXERCISES = {
    "brisk_walk":    {"label": "Brisk walking (outdoors or treadmill)", "type": "aerobic", "where": "outdoors / gym", "keys": ["walking"]},
    "cycling":       {"label": "Cycling (outdoor or stationary bike)", "type": "aerobic", "where": "outdoors / gym / home", "keys": ["cycling", "gym"]},
    "swimming":      {"label": "Swimming or water aerobics", "type": "aerobic", "where": "pool", "keys": ["swimming"]},
    "zumba":         {"label": "Zumba or dance workout", "type": "aerobic", "where": "home / class", "keys": ["zumba"]},
    "home_cardio":   {"label": "Home cardio circuit (marching, step-ups, low-impact moves)", "type": "aerobic", "where": "home", "keys": ["home"]},
    "gym_cardio":    {"label": "Gym cardio (treadmill, bike or elliptical)", "type": "aerobic", "where": "gym", "keys": ["gym"]},
    "home_strength": {"label": "Home strength (squats, wall push-ups, lunges, resistance bands)", "type": "resistance", "where": "home", "keys": ["home"]},
    "gym_strength":  {"label": "Gym strength (machines or light weights; about 6 exercises, 3 sets of 10)", "type": "resistance", "where": "gym", "keys": ["gym"]},
    "yoga":          {"label": "Yoga", "type": "mind", "where": "home / class", "keys": ["yoga"]},
    "stretching":    {"label": "Stretching and mobility", "type": "mind", "where": "home", "keys": ["yoga", "home"]},
    "breathing":     {"label": "Slow breathing or meditation", "type": "mind", "where": "anywhere", "keys": ["yoga"]},
}
ACTIVITY_KEYS = ["yoga", "home", "gym", "zumba", "walking", "swimming", "cycling"]

SAFETY_EXERCISE = ("Start slowly if you are new to exercise and build up over weeks. Stop and get medical advice if you feel chest pain, "
                   "dizziness or unusual shortness of breath. Ask your doctor first if you have a heart condition or very high blood pressure.")


def _norm(s):
    return (s or "").strip().lower()


def diet_key(constraint):
    c = _norm(constraint)
    if "vegan" in c: return "vegan"
    if "vegetarian" in c and "non" not in c: return "vegetarian"
    return "non-vegetarian"


def _dislike_tokens(dislikes):
    if isinstance(dislikes, str): dislikes = re.split(r"[,;/]", dislikes)
    toks = [t.strip().lower() for t in (dislikes or []) if t and t.strip()]
    return [t[:-1] if len(t) > 3 and t.endswith("s") and not t.endswith("ss") else t for t in toks]   # "beans" -> "bean", "mushrooms" -> "mushroom"


def allowed_foods(structured):
    """Foods this person may be offered, after diet, cuisine rules, conditions, dislikes, gluten and kidney rules."""
    diet = diet_key(structured.get("diet_type"))
    kinds = {"vegan": {"plant"}, "vegetarian": {"plant", "dairy", "egg"}}.get(diet, {"plant", "dairy", "egg", "fish", "poultry", "beef"})
    if diet == "non-vegetarian" and _norm(structured.get("cuisine")) == "american": kinds = kinds | {"pork"}
    conds = set(structured.get("conditions") or [])
    gf = "gluten" in _norm(structured.get("diet_type"))
    toks = _dislike_tokens(structured.get("dislikes"))
    out = []
    for f in FOODS.values():
        if f["kind"] not in kinds: continue
        if gf and f["gluten"]: continue
        if conds & {"pcos", "diabetes"} and f["id"] in REFINED_STARCH: continue
        if any(t and (t in f["label"].lower() or t == f["kind"] or (t in ("meat",) and f["kind"] in ("beef", "pork", "poultry"))) for t in toks): continue
        if structured.get("kidney_safe_mode"):
            if f["per100"]["potassium_mg"] * f["portion_g"] / 100 >= CKD_HIGH_K_MG_PER_PORTION: continue
        out.append(f)
    return out


def per_portion(f):
    k = f["portion_g"] / 100.0
    return {n: f["per100"][n] * k for n in ("kcal", "protein_g", "sodium_mg", "potassium_mg")}


def menu_for_prompt(allowed, cuisine=None):
    """One compact line per food: id | name | meals | portion | sodium | protein | kcal  (all per listed portion). Cuisine matches first."""
    cu = _norm(cuisine)
    rows = sorted(allowed, key=lambda f: (0 if cu and cu in f["cuisines"] else 1, f["id"]))
    lines = ["id | food | meals(b=breakfast,m=lunch/dinner,s=snack) | portion g | Na mg | protein g | kcal   (* = fits the chosen cuisine)"]
    for f in rows:
        p = per_portion(f)
        star = "*" if cu and cu in f["cuisines"] else ""
        lines.append(f"{f['id']}{star} | {f['label']} | {f['slots']} | {f['portion_g']} | {p['sodium_mg']:.0f} | {p['protein_g']:.1f} | {p['kcal']:.0f}")
    return "\n".join(lines)


def exercise_for_prompt(prefs):
    want = set(prefs or [])
    lines = ["id | activity | type"]
    for i, e in EXERCISES.items():
        mark = "*" if want & set(e["keys"]) else ""
        lines.append(f"{i}{mark} | {e['label']} | {e['type']}")
    return "\n".join(lines) + ("\n(* = the user's preferred activity types)" if want else "")


def protein_target(structured):
    """Grams per day. None in kidney-safe mode (KDIGO ties protein to body weight; we do not know it, so no number)."""
    if structured.get("kidney_safe_mode"): return None
    kcal = min(max(structured.get("kcal_reference") or 2000, 1600), 2600)
    r5 = lambda x: int(round(x / 5.0) * 5)
    return {"low_g": r5(0.15 * kcal / 4), "high_g": r5(0.25 * kcal / 4), "kcal_reference": kcal}


def exercise_targets(structured):
    beginner = _norm(structured.get("fitness")) == "beginner"
    return {"aerobic_min_per_week": [90, 200] if beginner else [150, 300], "resistance_days_min": 2,
            "note": "Beginners: build up gradually toward 150 min/week." if beginner else ""}


# ---- "what to avoid" list: written by software from cited facts, not by the AI ---------
def avoid_list(structured):
    cu, conds = _norm(structured.get("cuisine")), set(structured.get("conditions") or [])
    A = [("Salty processed foods: bacon, sausage, ham, deli meat, chips, instant noodles and flavored rice or pasta mixes.", "salty_foods"),
         ("Pickles, olives, brined foods and salty sauces such as soy sauce, ketchup and barbecue sauce.", "salty_foods"),
         ("Adding salt at the table or in cooking. Use herbs, spices, garlic, lemon or vinegar instead.", "salty_foods"),
         ("Alcohol. If you drink, keep it to 1 drink a day (women) or 2 (men) at most.", "alcohol")]
    extra = {"south asian": "Papad, achaar (pickles), namkeen and other salted snacks.",
             "mexican": "Canned beans and soups, seasoning packets and salted chips or salsa jars.",
             "east asian": "Soy sauce, fish sauce, instant-noodle seasoning and pickled vegetables.",
             "mediterranean": "Olives, brined cheese (feta), cured meats and canned fish in brine.",
             "american": "Fast food, deli meats, canned soups and frozen dinners."}
    if cu in extra: A.insert(1, (extra[cu], "salty_foods"))
    if "cholesterol" in conds or diet_key(structured.get("diet_type")) == "non-vegetarian":
        A.append(("Fried foods and fatty cuts of meat. Choose baked, grilled or steamed.", "satfat"))
    if conds & {"pcos", "diabetes"}:
        A.append(("White rice and white bread. Choose whole grains, beans and lentils.", "whole_grains"))
    if structured.get("kidney_safe_mode"):
        A.append(("Salt substitutes and 'lite' salt. They contain potassium.", "salt_substitute_ckd"))
        A.append(("Starting new supplements or high-protein products without a kidney dietitian.", "ckd_dietitian"))
    return A
