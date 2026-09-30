"""
Dietary Guidance Translation Framework (Project Objective 3).

Takes the top SHAP-identified risk factors for one person and
translates them into evidence-based dietary guidance, grounded in the
DASH diet and 2017/2025 AHA/ACC hypertension guidelines.

Design principle (per the project proposal): the underlying evidence
stays fixed and consistent for everyone -- personalization (Objective 4)
only changes HOW the guidance is communicated, not what the evidence
itself says. This module produces the fixed, evidence-based core
recommendation; personalize_presentation() in this same file shows how
that core recommendation can be re-worded for a person's stated
preferences and constraints.
"""

# Evidence-based guidance for each modifiable dietary/lifestyle factor
# this model can identify. Sources: DASH diet, 2017 ACC/AHA and 2025
# AHA/ACC hypertension guidelines.
GUIDANCE_LIBRARY = {
    "DR1TSODI": {
        "label": "Sodium intake",
        "high_risk_direction": "high",
        "recommendation": (
            "Sodium intake is a leading contributor to your estimated risk. The DASH "
            "diet and AHA guidelines recommend limiting sodium to 2,300 mg/day, with "
            "1,500 mg/day as an ideal target for people managing hypertension. "
            "Practical steps: reduce processed and packaged foods (the largest source "
            "of dietary sodium for most people), check nutrition labels, and season "
            "food with herbs/spices instead of salt."
        ),
    },
    "DR1TPOTA": {
        "label": "Potassium intake",
        "high_risk_direction": "low",
        "recommendation": (
            "Low potassium intake is contributing to your estimated risk. The DASH "
            "diet recommends 3,500-4,700 mg/day from food sources such as leafy "
            "greens, bananas, potatoes (with skin), beans, and yogurt. "
            "IMPORTANT: if you take a potassium-sparing diuretic, an ACE inhibitor, "
            "an ARB, or have kidney disease, do not increase potassium intake "
            "without first consulting your doctor -- see the Trust and Safety note below."
        ),
    },
    "BMXBMI": {
        "label": "Body Mass Index (BMI)",
        "high_risk_direction": "high",
        "recommendation": (
            "Your BMI is contributing meaningfully to your estimated risk. The AHA "
            "guidelines note that even a modest weight reduction (5-10% of body "
            "weight) can produce a measurable drop in blood pressure. This is best "
            "approached gradually through combined dietary and activity changes "
            "rather than rapid weight loss."
        ),
    },
    "BMXWAIST": {
        "label": "Waist circumference",
        "high_risk_direction": "high",
        "recommendation": (
            "Waist circumference above 102 cm (men) or 88 cm (women) is associated "
            "with higher cardiometabolic risk independent of overall BMI. Reducing "
            "refined carbohydrates and added sugars, alongside regular physical "
            "activity, is the most evidence-supported approach to reducing central "
            "adiposity."
        ),
    },
    "DR1TALCO": {
        "label": "Alcohol intake",
        "high_risk_direction": "high",
        "recommendation": (
            "Alcohol intake is contributing to your estimated risk. AHA guidelines "
            "recommend limiting alcohol to no more than 1 drink/day for women or 2 "
            "drinks/day for men; reducing intake below this threshold has a "
            "well-documented blood-pressure-lowering effect."
        ),
    },
    "DR1TTFAT": {
        "label": "Total fat intake",
        "high_risk_direction": "high",
        "recommendation": (
            "Total fat intake is contributing to your estimated risk. The DASH diet "
            "emphasizes reducing saturated fat specifically (from red meat, full-fat "
            "dairy, and fried foods) in favor of unsaturated fats (olive oil, nuts, "
            "fish)."
        ),
    },
    "SMQ020": {
        "label": "Smoking history",
        "high_risk_direction": "high",
        "recommendation": (
            "A smoking history is contributing to your estimated risk profile. "
            "Smoking cessation is one of the highest-impact changes available for "
            "cardiovascular risk reduction generally; consider discussing cessation "
            "resources with a healthcare provider."
        ),
    },
    # Non-modifiable / contextual factors: explained, not "recommended against"
    "RIDAGEYR": {
        "label": "Age",
        "high_risk_direction": None,
        "recommendation": (
            "Age is a major contributor to your estimated risk. This is not a "
            "modifiable factor, but it means the modifiable factors below are "
            "worth prioritizing, since risk from non-modifiable factors compounds "
            "with modifiable ones."
        ),
    },
    "RIDRETH3": {
        "label": "Race/ethnicity",
        "high_risk_direction": None,
        "recommendation": (
            "Race/ethnicity shows a statistical association with hypertension risk "
            "in this population-level model. This reflects population patterns, not "
            "a personal cause, and should not be over-interpreted at the individual "
            "level -- see the Trust and Safety note below."
        ),
    },
    "INDFMPIR": {
        "label": "Income-to-poverty ratio",
        "high_risk_direction": None,
        "recommendation": (
            "Household income relative to the poverty line shows a statistical "
            "association with hypertension risk, likely reflecting access to "
            "healthcare and healthy food rather than a direct biological cause."
        ),
    },
}


def generate_core_guidance(top_factors: list) -> list:
    """
    top_factors: list of (feature_name, shap_value) tuples from the
    Explainability Agent, ordered by importance.

    Returns a list of guidance dicts for each recognized factor, each
    containing the FIXED, evidence-based recommendation (Objective 3).
    Personalization (Objective 4) happens in a separate step.
    """
    guidance = []
    for feat, shap_val in top_factors:
        entry = GUIDANCE_LIBRARY.get(feat)
        if entry is None:
            continue
        guidance.append({
            "factor": feat,
            "label": entry["label"],
            "shap_value": shap_val,
            "direction": "increases risk" if shap_val > 0 else "decreases risk",
            "is_modifiable": entry["high_risk_direction"] is not None,
            "recommendation": entry["recommendation"],
        })
    return guidance


def personalize_presentation(guidance: list, preferences: dict = None) -> list:
    """
    Objective 4: adapt HOW guidance is communicated to a person's stated
    preferences/constraints, without changing the underlying evidence.

    preferences: optional dict, e.g.
        {"cuisine": "South Asian", "dislikes": ["fish"], "constraint": "vegetarian"}

    This is a lightweight, rule-based personalization layer (per the
    proposal's stated scope -- not an LLM-based rewrite), appending a
    short contextual note rather than altering the evidence-based
    recommendation text itself.
    """
    if not preferences:
        return guidance

    personalized = []
    for item in guidance:
        note = None
        if item["factor"] == "DR1TSODI":
            cuisine = preferences.get("cuisine")
            if cuisine:
                note = (
                    f"For {cuisine} cooking specifically, watch for high-sodium "
                    "staples like soy sauce, fish sauce, bouillon, and pickled "
                    "condiments -- look for reduced-sodium versions where available."
                )
        elif item["factor"] == "DR1TPOTA":
            if preferences.get("constraint") == "vegetarian":
                note = (
                    "Good vegetarian potassium sources include lentils, beans, "
                    "potatoes, spinach, and bananas."
                )
            elif "fish" in (preferences.get("dislikes") or []):
                note = (
                    "Since fish is not an option for you, potassium-rich "
                    "alternatives include beans, potatoes, and leafy greens."
                )
        new_item = dict(item)
        if note:
            new_item["personalized_note"] = note
        personalized.append(new_item)
    return personalized


def format_report(guidance: list) -> str:
    """Human-readable text block combining evidence-based guidance with
    the Trust and Communication framing the proposal specifies:
    clearly separating model-derived findings from general guidance."""
    lines = ["DIETARY GUIDANCE (based on your top contributing risk factors)", ""]
    for item in guidance:
        modifiable_tag = "[Modifiable factor]" if item["is_modifiable"] else "[Contextual factor]"
        lines.append(f"- {item['label']} {modifiable_tag} -- {item['direction']}")
        lines.append(f"  {item['recommendation']}")
        if item.get("personalized_note"):
            lines.append(f"  Personalized note: {item['personalized_note']}")
        lines.append("")
    lines.append(
        "Trust and Safety note: The factors above are statistical associations "
        "identified from population data (SHAP-based model explanation), not a "
        "medical diagnosis. General dietary guidance above (from DASH/AHA "
        "guidelines) applies broadly; how it applies to YOUR specific situation "
        "should be confirmed with a doctor or registered dietitian, especially "
        "if you take blood pressure medication or have kidney disease."
    )
    return "\n".join(lines)
