"""Rule-based plan builder: composes a plan from the vetted menu with NO language model.
Used when the AI draft cannot pass the checks, so the user still gets a clear plan. Every number is guaranteed by the same checker.
Meal = one protein food + one starch + vegetables (+ fruit / fat), scaled to a calorie target, kept under the slot's sodium budget."""
import itertools
import menu

STARCH = {"oatmeal", "whole_wheat_cereal", "upma", "dosa", "roti", "whole_wheat_bread", "corn_tortilla", "brown_rice", "white_rice", "quinoa", "whole_pasta",
          "bulgur", "barley", "millet", "couscous", "sweet_potato"}
VEG = {"spinach", "broccoli", "cauliflower", "carrots", "cabbage", "green_beans", "bell_pepper", "tomato", "cucumber", "onion", "eggplant", "okra",
       "lettuce", "kale", "zucchini_squash", "peas", "bean_sprouts"}
FRUIT = {"apple", "banana", "orange", "berries", "blueberries", "papaya", "mango", "guava", "pear", "grapes", "watermelon"}
NUTS = {"almonds", "walnuts", "peanuts", "pumpkin_seeds", "chia", "peanut_butter", "popcorn"}
FATS = {"olive_oil"}
SHARE = {"breakfast": 0.24, "lunch": 0.32, "dinner": 0.32, "snack": 0.12}
MEAT = {"poultry", "fish", "beef", "pork"}


def _nut(f, g):
    k = g / 100.0
    return {n: f["per100"][n] * k for n in ("kcal", "protein_g", "sodium_mg")}


def _is_protein(f):
    return f["kind"] in MEAT | {"egg"} or f["id"] in {"lentils", "chickpeas", "kidney_beans", "black_beans", "pinto_beans", "tofu", "tempeh", "edamame", "paneer",
                                                      "greek_yogurt", "yogurt", "cottage_low_na", "soy_milk", "milk"}


def _slot_ok(f, slot):
    s = f["slots"]
    return ("b" in s) if slot == "breakfast" else ("s" in s) if slot == "snack" else ("m" in s)


MEAL_PROTEIN = {"lentils", "chickpeas", "kidney_beans", "black_beans", "pinto_beans", "tofu", "tempeh", "paneer", "egg"}


def _meal_protein(f):
    return f["kind"] in MEAT or f["id"] in MEAL_PROTEIN


def _pref(f, cuisine):
    return 0 if cuisine and cuisine in f["cuisines"] else 1


def _scaled(items, target_kcal):
    tot = sum(_nut(f, g)["kcal"] for f, g in items)
    k = max(0.7, min(1.35, target_kcal / max(tot, 1)))
    return [(f, g * k) for f, g in items]


def _option(slot, parts, target, budget):
    items = _scaled([(f, f["portion_g"] * m) for f, m in parts], target)
    items = [(f, max(0.45 * f["portion_g"], min(2.4 * f["portion_g"], g))) for f, g in items]
    na = sum(_nut(f, g)["sodium_mg"] for f, g in items)
    kcal = sum(_nut(f, g)["kcal"] for f, g in items)
    return (items, na, kcal) if na <= budget else None


def _to_obj(name, items):
    return {"name": name, "items": [{"id": f["id"], "g": int(round(g / 5.0) * 5) or 5} for f, g in items]}


def _name(items):
    main = [f["label"].split(",")[0] for f, _ in items][:3]
    return " with ".join([main[0], " and ".join(main[1:])]) if len(main) > 1 else main[0]


def build_meals(structured, allowed):
    cu = (structured.get("cuisine") or "").strip().lower()
    kcal = min(max(structured.get("kcal_reference") or 2000, 1600), 2600)
    budget = menu.SLOT_BUDGET_MG[bool(structured["kidney_safe_mode"])]
    A = sorted(allowed, key=lambda f: (_pref(f, cu), f["id"]))
    meals = {}
    USED_P = set()
    for slot in ("breakfast", "lunch", "dinner", "snack"):
        target = kcal * SHARE[slot]
        pool = [f for f in A if _slot_ok(f, slot)]
        prot = [f for f in pool if _is_protein(f) and (slot in ("lunch", "dinner") and _meal_protein(f) or slot == "breakfast" or slot == "snack" or f["id"] in {"greek_yogurt", "yogurt", "edamame", "cottage_low_na", "milk", "soy_milk"})]
        star = [f for f in pool if f["id"] in STARCH]
        veg = [f for f in pool if f["id"] in VEG]
        fruit = [f for f in pool if f["id"] in FRUIT]
        nuts = [f for f in pool if f["id"] in NUTS]
        fat = [f for f in pool if f["id"] in FATS]
        cands = []
        if slot == "snack":
            for p, fr in itertools.product(prot[:6] or nuts[:4], (fruit or veg)[:6]):
                parts = [(p, 1), (fr, 1)] + ([(nuts[0], 1)] if nuts and p["kind"] != "plant" else [])
                cands.append(parts)
            if not cands: cands = [[(fr, 1), (n, 1)] for fr, n in itertools.product(fruit[:5], nuts[:4])]
        elif slot == "breakfast":
            for s, p, fr in itertools.product(star[:8] or [None], prot[:8], (fruit or [None])[:5]):
                parts = [x for x in ((s, 1), (p, 1), (fr, 1)) if x[0]]
                if len(parts) >= 2: cands.append(parts)
        else:
            meat = [f for f in prot if f["kind"] in MEAT]
            fish = [f for f in meat if f["kind"] == "fish"]
            land = [f for f in meat if f["kind"] != "fish"]
            plant = [f for f in prot if f["id"] in MEAL_PROTEIN and f["id"] != "egg"] + [f for f in prot if f["id"] == "egg"]
            if slot == "lunch": order = (land[:3] + plant[:3] + meat[3:]) if land else plant
            else: order = (fish[:2] + land[3:5] + plant[3:] + plant[:3]) if meat else plant[3:] + plant[:3]
            order = [f for f in order if f in prot and _meal_protein(f)] or [f for f in prot if _meal_protein(f)]
            for p in order[:10]:
                for s_, v1, v2 in itertools.product(star[:8], veg[:8], veg[:8]):
                    if v1["id"] >= v2["id"]: continue
                    m = 0.8 if p["kind"] in MEAT else 1.0
                    cands.append([(s_, 1), (p, m), (v1, 1), (v2, 1)] + ([(fat[0], 1)] if fat else []))
        opts, used_p, used_s = [], set(USED_P) if slot in ('lunch', 'dinner') else set(), set()
        for parts in cands:
            r = _option(slot, parts, target, budget[slot])
            if not r: continue
            items = r[0]
            key_p = next((f["id"] for f, _ in items if _is_protein(f)), None)
            key_s = next((f["id"] for f, _ in items if f["id"] in STARCH), None)
            if key_p in used_p or (key_s and key_s in used_s and len(opts) == 1 and key_p in used_p): continue
            if key_p in used_p: continue
            used_p.add(key_p); used_s.add(key_s); USED_P.add(key_p) if slot == "lunch" else None
            opts.append(_to_obj(_name(items), items))
            if len(opts) == (1 if slot == "snack" else 2): break
        if len(opts) < (1 if slot == "snack" else 2): return None
        meals[slot] = opts
    return meals


def build_exercise(structured):
    tg = menu.exercise_targets(structured)
    lo, hi = tg["aerobic_min_per_week"]
    target = (lo + hi) // 2 if structured.get("fitness") == "active" else min(150, hi)
    prefs = set(structured.get("activities") or [])
    aer = []
    if "walking" in prefs: aer.append("brisk_walk")
    if "running" in prefs: aer.append("running")
    if "zumba" in prefs: aer.append("zumba")
    if "swimming" in prefs: aer.append("swimming")
    if "cycling" in prefs: aer.append("cycling")
    if "gym" in prefs: aer.append("gym_cardio")
    if "home" in prefs: aer.append("home_cardio")
    if not aer: aer = ["brisk_walk"]
    aer = aer[:3]
    minutes = 40 if structured.get("fitness") == "active" else 30
    sessions = max(len(aer), round(target / minutes))
    days = [sessions // len(aer) + (1 if i < sessions % len(aer) else 0) for i in range(len(aer))]
    items = [{"id": a, "minutes": minutes, "days": min(7, max(1, d))} for a, d in zip(aer, days)]
    # Running is hard on the body: at most 3 days a week for beginners and 4 for active people; the rest becomes brisk walking.
    cap = 4 if structured.get("fitness") == "active" else 3
    for it in list(items):
        if it["id"] == "running" and it["days"] > cap:
            extra = it["days"] - cap; it["days"] = cap
            walk = next((x for x in items if x["id"] == "brisk_walk"), None)
            if walk: walk["days"] = min(7, walk["days"] + extra)
            else: items.append({"id": "brisk_walk", "minutes": minutes, "days": min(7, extra)})
    items.append({"id": "gym_strength" if "gym" in prefs else "home_strength", "minutes": 40 if "gym" in prefs else 25, "days": 2})
    items.append({"id": "yoga", "minutes": 30, "days": 2} if "yoga" in prefs else {"id": "stretching", "minutes": 15, "days": 3})
    return items


def _mean_protein(meals):
    tot = 0.0
    for opts in meals.values():
        ps = [sum(menu.FOODS[i["id"]]["per100"]["protein_g"] * i["g"] / 100.0 for i in o["items"]) for o in opts]
        tot += sum(ps) / len(ps)
    return tot


def _trim_protein(meals, structured):
    """Kidney-safe mode: shrink protein foods until daily protein is inside the KDIGO-based limit."""
    pt = structured.get("PROTEIN_TARGET_G")
    if not (structured.get("kidney_safe_mode") and pt): return
    for _ in range(12):
        if _mean_protein(meals) <= pt["high_g"]: return
        for opts in meals.values():
            for o in opts:
                for i in o["items"]:
                    f = menu.FOODS[i["id"]]
                    floor = int(-(-0.45 * f["portion_g"] // 5) * 5)
                    if _is_protein(f) and i["g"] > floor:
                        i["g"] = max(floor, int(round(i["g"] * 0.85 / 5.0) * 5))


def _mean_kcal(meals):
    tot = 0.0
    for opts in meals.values():
        ks = [sum(menu.FOODS[i["id"]]["per100"]["kcal"] * i["g"] / 100.0 for i in o["items"]) for o in opts]
        tot += sum(ks) / len(ks)
    return tot


def _refill_energy(meals, structured):
    """After protein is trimmed, add energy back with grains / starch and fruit (low sodium) so the day is not under-fed."""
    ref = min(max(structured.get("kcal_reference") or 2000, 1600), 2600)
    for _ in range(8):
        deficit = ref - _mean_kcal(meals)
        if deficit < 0.05 * ref: return
        grow = [i for opts in meals.values() for o in opts for i in o["items"] if i["id"] in STARCH or i["id"] in FRUIT]
        base = sum(menu.FOODS[i["id"]]["per100"]["kcal"] * i["g"] / 100.0 for i in grow) / 4.0
        f = min(1.4, 1 + deficit / max(base, 1))
        for i in grow:
            cap = int(1.6 * menu.FOODS[i["id"]]["portion_g"] // 5 * 5)
            i["g"] = min(cap, int(round(i["g"] * f / 5.0) * 5))


def build_plan(structured, allowed, labelled):
    meals = build_meals(structured, allowed)
    if not meals: return None
    _trim_protein(meals, structured)
    if structured.get("kidney_safe_mode") and structured.get("PROTEIN_TARGET_G"):
        _refill_energy(meals, structured); _trim_protein(meals, structured)
    cites = {p.get("id"): lab for lab, p in labelled.items() if isinstance(p, dict) and p.get("id")}
    tips = []
    for fid, text in (("sodium_ckd", None), ("sodium_general", None), ("exercise_rec", None), ("whole_grains", None), ("protein_amdr", None)):
        if fid in cites and len(tips) < 3:
            tips.append({"text": labelled[cites[fid]]["text"], "cite": [cites[fid]]})
    tier = structured["risk_tier"]
    drivers = ", ".join(f["factor"] for f in structured["top_factors"] if f["direction"] == "raises risk")
    summary = (f"Your screening result: a {tier} chance of high blood pressure." +
               (f" The numbers that raise it most: {drivers}." if drivers else "") +
               " This is an estimate, not a diagnosis. The plan below keeps sodium within your limit.")
    return {"summary": summary, "meals": meals, "exercise": build_exercise(structured), "tips": tips}
