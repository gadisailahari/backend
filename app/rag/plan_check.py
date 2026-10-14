"""Checks a structured meal + exercise plan against the vetted menus. Everything here is computed from data, not trusted from the AI.

Plan shape (what the AI returns):
 {"summary": "...", "meals": {"breakfast": [{"name": "...", "items": [{"id": "oatmeal", "g": 240}]}], "lunch": [...], "dinner": [...], "snack": [...]},
  "exercise": [{"id": "brisk_walk", "minutes": 30, "days": 5}], "tips": [{"text": "...", "cite": ["F1"]}]}
Sodium is judged on the worst case: for each meal we take the option with the MOST sodium and add them up, so any combination the user picks is within the limit.
"""
import menu

SLOTS = ["breakfast", "lunch", "dinner", "snack"]
MIN_OPTIONS = {"breakfast": 2, "lunch": 2, "dinner": 2, "snack": 1}
KCAL_RANGE = (1300, 2800)


def _num(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def option_nutrients(items, foods):
    t = dict(kcal=0.0, protein_g=0.0, fiber_g=0.0, satfat_g=0.0, potassium_mg=0.0, sodium_mg=0.0)
    for it in items:
        f, k = foods[it["id"]], it["g"] / 100.0
        for n in t: t[n] += f["per100"][n] * k
    return t


def check_plan_v2(obj, allowed, structured):
    """Returns {'violations': [...], 'totals': {...}, 'options': {slot: [..]}, 'exercise': {...}}"""
    viol = []
    allowed_ids = {f["id"]: f for f in allowed}
    ckd = structured["kidney_safe_mode"]
    cap = structured["TARGETS"]["sodium_max_mg"]
    budget = menu.SLOT_BUDGET_MG[bool(ckd)]
    meals = obj.get("meals")
    options, worst, mean = {}, {}, {}
    if not isinstance(meals, dict):
        return {"violations": ["'meals' must be an object with breakfast, lunch, dinner and snack lists."], "totals": None, "options": {}, "exercise": None}
    for slot in SLOTS:
        opts = meals.get(slot)
        if not isinstance(opts, list) or len(opts) < MIN_OPTIONS[slot]:
            viol.append(f"{slot}: give at least {MIN_OPTIONS[slot]} option(s)."); continue
        rows = []
        for n, o in enumerate(opts[:3], 1):
            name = str(o.get("name", f"option {n}"))[:60] if isinstance(o, dict) else ""
            items = o.get("items") if isinstance(o, dict) else None
            if not isinstance(items, list) or not (2 <= len(items) <= 6):
                viol.append(f"{slot} option {n} ('{name}') needs 2 to 6 items."); continue
            clean, bad = [], False
            for it in items:
                fid, g = (it.get("id"), it.get("g")) if isinstance(it, dict) else (None, None)
                if fid not in allowed_ids:
                    viol.append(f"{slot} option {n}: '{fid}' is not on the allowed FOOD MENU for this user (diet, dislikes or kidney rules)."); bad = True; continue
                p = allowed_ids[fid]["portion_g"]
                if not _num(g) or not (0.4 * p <= g <= 2.5 * p):
                    viol.append(f"{slot} option {n}: amount for {fid} must be between {0.4*p:.0f} and {2.5*p:.0f} g (portion is {p} g)."); bad = True; continue
                clean.append({"id": fid, "g": g})
            if bad or not clean: continue
            nut = option_nutrients(clean, allowed_ids)
            rows.append({"name": name, "items": clean, **{k: round(v, 1) for k, v in nut.items()}})
        if rows:
            options[slot] = rows
            worst[slot] = max(r["sodium_mg"] for r in rows)
            mean[slot] = {k: sum(r[k] for r in rows) / len(rows) for k in ("kcal", "protein_g", "fiber_g", "satfat_g", "potassium_mg", "sodium_mg")}
    totals = None
    if len(options) == len(SLOTS):
        totals = {k: round(sum(mean[s][k] for s in SLOTS), 1) for k in ("kcal", "protein_g", "fiber_g", "satfat_g", "potassium_mg")}
        totals["sodium_mg_worst_case"] = round(sum(worst.values()))
        totals["sodium_mg_average"] = round(sum(mean[s]["sodium_mg"] for s in SLOTS))
        if totals["sodium_mg_worst_case"] > cap:
            parts = ", ".join(f"{s} {worst[s]:.0f} (budget {budget[s]})" for s in SLOTS)
            over = [s for s in SLOTS if worst[s] > budget[s]]
            viol.append(f"Sodium: the highest-sodium option of each meal adds up to {totals['sodium_mg_worst_case']} mg, over the {cap} mg limit [{parts}]. "
                        f"Replace salty items in: {', '.join(over) or 'any meal'}. Low-sodium choices: tofu, eggs, fruit, raw vegetables, corn tortilla, nuts, yogurt.")
        if not (KCAL_RANGE[0] <= totals["kcal"] <= KCAL_RANGE[1]):
            viol.append(f"Energy {totals['kcal']:.0f} kcal per day is outside {KCAL_RANGE[0]}-{KCAL_RANGE[1]}; adjust amounts.")
        pct = totals["protein_g"] * 4 / max(totals["kcal"], 1)
        if pct < 0.10: viol.append(f"Protein is only {pct*100:.0f}% of calories; add protein foods (guideline range is 10-35%).")
        if not ckd and pct > 0.35: viol.append(f"Protein is {pct*100:.0f}% of calories, above the 35% guideline limit.")
        if "cholesterol" in (structured.get("conditions") or []) and totals["satfat_g"] * 9 / max(totals["kcal"], 1) > 0.10:
            viol.append("Saturated fat is above 10% of calories; choose leaner or plant protein and low-fat dairy.")
    ex = check_exercise(obj.get("exercise"), structured)
    viol += ex["violations"]
    return {"violations": viol, "totals": totals, "options": options, "exercise": ex}


def check_exercise(items, structured):
    tg = menu.exercise_targets(structured)
    lo, hi = tg["aerobic_min_per_week"]
    viol, rows = [], []
    aer, res = 0, 0
    if not isinstance(items, list) or not items:
        return {"violations": ["exercise: give 3 to 6 activities with minutes and days per week."], "items": [], "aerobic_min": 0, "resistance_days": 0}
    if len(items) > 6: viol.append("exercise: at most 6 activities.")
    prefs = set(structured.get("activities") or [])
    used_keys = set()
    for it in items[:6]:
        eid = it.get("id") if isinstance(it, dict) else None
        e = menu.EXERCISES.get(eid)
        m, d = (it.get("minutes"), it.get("days")) if isinstance(it, dict) else (None, None)
        if not e: viol.append(f"exercise: '{eid}' is not on the EXERCISE menu."); continue
        if not _num(m) or not (10 <= m <= 90) or not _num(d) or not (1 <= d <= 7):
            viol.append(f"exercise {eid}: minutes must be 10-90 and days 1-7."); continue
        used_keys |= set(e["keys"])
        if e["type"] == "aerobic": aer += m * d
        if e["type"] == "resistance": res += d
        rows.append({"id": eid, "label": e["label"], "type": e["type"], "where": e["where"], "minutes": m, "days": d})
    if aer < lo or aer > hi:
        viol.append(f"exercise: aerobic activity adds up to {aer} min/week; it must be {lo}-{hi} (AHA/ACC: at least 150 min/week moderate activity; beginners build up from 90).")
    if res < tg["resistance_days_min"] or res > 5:
        viol.append(f"exercise: strength sessions add up to {res} day(s)/week; they must be {tg['resistance_days_min']}-5 days.")
    if prefs and not (prefs & used_keys):
        viol.append(f"exercise: include at least one of the user's preferred activity types: {', '.join(sorted(prefs))}.")
    return {"violations": viol, "items": rows, "aerobic_min": aer, "resistance_days": res}
