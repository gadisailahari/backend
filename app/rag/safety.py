"""Safety rules for the dietary guidance pipeline.

Sources
- eGFR: CKD-EPI 2021 creatinine equation, race-free (Inker et al., NEJM 2021;385:1737-49, Table 2).
- CKD definition: eGFR < 60 ml/min/1.73 m2 (KDIGO 2024 CKD guideline, categories G3a-G5).
- Sodium in CKD: < 2 g sodium/day (KDIGO 2024, Rec 3.3.2.1, grade 2C).
- Potassium AI: 3,400 mg men / 2,600 mg women, adults 19+ (NIH ODS fact sheet citing NASEM 2019).
"""
import re

SODIUM_MAX_GENERAL_MG = 2300      # Dietary Guidelines for Americans; AHA/ACC
SODIUM_MAX_CKD_MG = 2000          # KDIGO 2024 Rec 3.3.2.1
POTASSIUM_AI_MG = {"M": 3400, "F": 2600}   # adults 19+ (NIH ODS / NASEM 2019)
CKD_EGFR_CUTOFF = 60.0            # KDIGO: G3a-G5

# PLACEHOLDER - not from a guideline. A dietitian must approve before any real use.
CKD_HIGH_K_MG_PER_SERVING = 300


def sex_code(sex):
    """Accept 'M'/'F', 'male'/'female', or NHANES RIAGENDR (1=male, 2=female)."""
    if sex in (1, "1"): return "M"
    if sex in (2, "2"): return "F"
    s = str(sex).strip().upper()[:1]
    if s in ("M", "F"): return s
    raise ValueError(f"unknown sex value: {sex!r}")


def egfr_ckdepi_2021(scr_mg_dl, age_years, sex):
    """CKD-EPI 2021 creatinine equation (race-free). Scr in mg/dL (NHANES LBXSCR)."""
    s = sex_code(sex)
    kappa, a1 = (0.7, -0.241) if s == "F" else (0.9, -0.302)
    ratio = scr_mg_dl / kappa
    val = 142 * (min(ratio, 1) ** a1) * (max(ratio, 1) ** -1.200) * (0.9938 ** age_years)
    return val * 1.012 if s == "F" else val


def kidney_status(scr_mg_dl=None, age_years=None, sex=None):
    """Returns dict: egfr, ckd_mode, reason. If creatinine is missing we do NOT assume normal."""
    if scr_mg_dl is None or age_years is None or sex is None:
        return {"egfr": None, "ckd_mode": True,
                "reason": "Kidney function unknown, so the cautious (kidney-safe) rules apply."}
    e = egfr_ckdepi_2021(scr_mg_dl, age_years, sex)
    return {"egfr": round(e, 1), "ckd_mode": e < CKD_EGFR_CUTOFF,
            "reason": f"eGFR {e:.0f} ml/min/1.73 m2 " + ("is below 60 (KDIGO CKD range)." if e < CKD_EGFR_CUTOFF else "is 60 or above.")}


def nutrient_targets(ckd_mode, sex):
    s = sex_code(sex)
    t = {"sodium_max_mg": SODIUM_MAX_CKD_MG if ckd_mode else SODIUM_MAX_GENERAL_MG}
    if ckd_mode:
        t["potassium_target_mg"] = None          # never push potassium; refer to clinician/dietitian
        t["potassium_note"] = "Kidney-safe mode: no potassium target and no advice to increase potassium. Refer to a kidney dietitian."
    else:
        t["potassium_target_mg"] = POTASSIUM_AI_MG[s]
    return t


# ---- text screening (for LLM output) -------------------------------------------------
_DRUGS = r"(lisinopril|enalapril|ramipril|benazepril|losartan|valsartan|olmesartan|amlodipine|metoprolol|atenolol|hydrochlorothiazide|hctz|furosemide|spironolactone|eplerenone|amiloride|triamterene|chlorthalidone|warfarin|statin|metformin|insulin)"
_MED_ACTION = [
    rf"\b(stop|quit|skip|discontinue|reduce|lower|increase|raise|double|halve|adjust|change|start|begin)\b[^.\n]{{0,40}}\b(medication|medicine|pill|prescription|dose|dosage|{_DRUGS})\b",
    rf"\btake\b[^.\n]{{0,25}}\b{_DRUGS}\b",
    r"\b(instead of|in place of|replace)\b[^.\n]{0,30}\b(medication|medicine|pills?)\b",
    r"\b(cure|reverse)\b[^.\n]{0,30}\bhypertension\b",
]
_K_PUSH = [r"\b(increase|boost|eat more|add more|get more|raise)\b[^.\n]{0,30}\bpotassium\b",
           r"\bpotassium[- ]rich\b", r"\bhigh[- ]potassium\b", r"\bsalt substitute", r"\bpotassium supplement",
           r"\blite salt\b", r"\bpotassium chloride\b"]


def screen_text(text, ckd_mode):
    """Return list of violations found in LLM-written text. Empty list = passed."""
    v = []
    for p in _MED_ACTION:
        m = re.search(p, text, re.I)
        if m: v.append(("medication_advice", m.group(0)))
    if ckd_mode:
        for p in _K_PUSH:
            m = re.search(p, text, re.I)
            if m: v.append(("potassium_in_kidney_mode", m.group(0)))
    return v
