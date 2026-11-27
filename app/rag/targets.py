from safety import POTASSIUM_AI_MG, SODIUM_MAX_GENERAL_MG
ACTIVITY_FACTOR = {'inactive': 1.2, 'active': 1.55}
FLOOR_KCAL = {'F': 1200, 'M': 1500}
CARB_PCT, PROTEIN_PCT, FAT_PCT, SATFAT_MAX_PCT, FIBRE_PER_1000 = (0.45, 0.2, 0.35, 0.1, 14.0)

def _num(x, lo, hi):
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if lo <= v <= hi else None

def is_active(profile):
    return str(profile.get('PAQ650')) in ('1', '1.0') or str(profile.get('PAQ665')) in ('1', '1.0')

def energy_targets(sex, age, height_cm, weight_kg, bmi, active):
    h, w, a = (_num(height_cm, 120, 230), _num(weight_kg, 30, 250), _num(age, 18, 100))
    s = 'M' if str(sex) in ('1', '1.0', 'M', 'm', 'male', 'Male') else 'F'
    if h is None or w is None or a is None:
        return None
    bmr = 10 * w + 6.25 * h - 5 * a + (5 if s == 'M' else -161)
    factor = ACTIVITY_FACTOR['active' if active else 'inactive']
    tdee = bmr * factor
    b = _num(bmi, 10, 80)
    if b is None:
        b = w / (h / 100.0) ** 2
    deficit = 500 if b >= 25 else 0
    kcal = max(tdee - deficit, FLOOR_KCAL[s])
    kcal = int(round(kcal))
    return {'energy_kcal': kcal, 'bmr_kcal': round(bmr), 'activity_factor': factor, 'activity_level': 'active' if active else 'inactive', 'deficit_kcal': deficit, 'floor_applied': tdee - deficit < FLOOR_KCAL[s], 'bmi_used': round(b, 1), 'carb_g': round(CARB_PCT * kcal / 4), 'protein_g': round(PROTEIN_PCT * kcal / 4), 'fat_g': round(FAT_PCT * kcal / 9), 'satfat_max_g': round(SATFAT_MAX_PCT * kcal / 9, 1), 'fibre_min_g': round(FIBRE_PER_1000 * kcal / 1000.0, 1), 'sodium_max_mg': SODIUM_MAX_GENERAL_MG, 'potassium_reference_mg': POTASSIUM_AI_MG[s], 'carb_pct': 45, 'protein_pct': 20, 'fat_pct': 35}
