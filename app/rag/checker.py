"""Nutrient checker: recomputes the nutrients of an LLM-written plan from USDA FNDDS 2017-18
and rejects the plan if it breaks the targets or uses unknown/implausible items.

Plan format (what the LLM must return as JSON):
  {"meals":[{"name":"Breakfast","items":[{"food_code":"11100000","grams":240}, ...]}, ...]}
Nutrient values are per 100 g in the FNDDS table, so grams matter.
"""
import csv, re, pathlib
from rank_bm25 import BM25Okapi

DATA = pathlib.Path(__file__).parent / "data"
_NUT = {}
with open(DATA / "fndds_2017_18_per100g.csv", newline="") as f:
    for r in csv.DictReader(f):
        _NUT[r["code"]] = {k: (float(v) if v not in ("", None) and k not in ("code", "desc", "category") else v) for k, v in r.items()}
_PORT = {}
with open(DATA / "fndds_2017_18_portions.csv", newline="") as f:
    for r in csv.DictReader(f):
        _PORT.setdefault(r["code"], []).append((r["portion"], float(r["grams"])))

_tok = lambda s: re.findall(r"[a-z0-9]+", s.lower())
_codes = list(_NUT)
_bm = BM25Okapi([_tok(_NUT[c]["desc"] + " " + _NUT[c]["category"]) for c in _codes])

MAX_ITEM_G = 1000
MAX_DAY_KCAL = 5000


def search_foods(query, k=8):
    """Candidate foods for the LLM to choose from (so it uses real FNDDS codes)."""
    s = _bm.get_scores(_tok(query))
    top = sorted(range(len(_codes)), key=lambda i: -s[i])[:k]
    out = []
    for i in top:
        n = _NUT[_codes[i]]
        out.append({"food_code": n["code"], "desc": n["desc"], "kcal_100g": n["kcal"], "sodium_mg_100g": n["sodium_mg"],
                    "potassium_mg_100g": n["potassium_mg"], "portions": _PORT.get(n["code"], [])[:4]})
    return out


def _item_nutrients(code, grams):
    n = _NUT[code]; f = grams / 100.0
    return {k: n[k] * f for k in ("kcal", "protein_g", "fiber_g", "satfat_g", "potassium_mg", "sodium_mg")}


def check_plan(plan, targets, ckd_mode, kcal_target=None):
    """Returns {'ok': bool, 'totals': {...}, 'violations': [...], 'per_item': [...]}"""
    totals = dict(kcal=0.0, protein_g=0.0, fiber_g=0.0, satfat_g=0.0, potassium_mg=0.0, sodium_mg=0.0)
    viol, per_item = [], []
    for meal in plan.get("meals", []):
        for it in meal.get("items", []):
            code, g = str(it.get("food_code", "")), it.get("grams")
            if code not in _NUT:
                viol.append(f"Unknown food_code '{code}' in {meal.get('name','meal')} (not in FNDDS); the plan cannot be verified.")
                continue
            if not isinstance(g, (int, float)) or g <= 0 or g > MAX_ITEM_G:
                viol.append(f"Implausible amount for {_NUT[code]['desc']}: {g!r} g.")
                continue
            nn = _item_nutrients(code, g)
            for k in totals: totals[k] += nn[k]
            per_item.append({"meal": meal.get("name"), "food": _NUT[code]["desc"], "grams": g, **{k: round(v, 1) for k, v in nn.items()}})
            if ckd_mode and nn["potassium_mg"] > __import__("safety").CKD_HIGH_K_MG_PER_SERVING:
                viol.append(f"Kidney-safe mode: {_NUT[code]['desc']} ({g:.0f} g) has about {nn['potassium_mg']:.0f} mg potassium, above the per-item limit.")
    if not per_item and not viol:
        viol.append("Plan contains no verifiable foods.")
    if totals["sodium_mg"] > targets["sodium_max_mg"]:
        viol.append(f"Sodium {totals['sodium_mg']:.0f} mg exceeds the {targets['sodium_max_mg']} mg daily limit.")
    if totals["kcal"] > MAX_DAY_KCAL:
        viol.append(f"Energy {totals['kcal']:.0f} kcal is implausibly high for one day.")
    if kcal_target and not (0.8 * kcal_target <= totals["kcal"] <= 1.2 * kcal_target):
        viol.append(f"Energy {totals['kcal']:.0f} kcal is more than 20% away from the {kcal_target:.0f} kcal target.")
    return {"ok": not viol, "totals": {k: round(v, 1) for k, v in totals.items()}, "violations": viol, "per_item": per_item}


def retry_hint(result):
    """Text fed back to the LLM when the checker rejects its plan."""
    return "The checker rejected your plan:\n- " + "\n- ".join(result["violations"]) + "\nRevise the plan and return JSON only."
