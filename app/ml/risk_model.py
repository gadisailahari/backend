"""
Self-contained risk prediction + explainability, bundled for the
FastAPI backend / Docker image. Loads the pre-trained model shipped
alongside this file (ml/xgb_hypertension_model_2017.json).
"""

import xgboost as xgb
import shap
import pandas as pd
from pathlib import Path

MODEL_PATH = Path(__file__).parent / "xgb_hypertension_model_2017.json"

FEATURES = ["RIAGENDR", "RIDAGEYR", "RIDRETH3", "INDFMPIR", "BMXBMI", "BMXWAIST",
            "DR1TKCAL", "DR1TSODI", "DR1TPOTA", "DR1TPROT", "DR1TCARB",
            "DR1TTFAT", "DR1TALCO", "LBXSCR", "SMQ020", "PAQ650", "PAQ665", "ALQ121"]

_model = xgb.XGBClassifier()
_model.load_model(MODEL_PATH)
_explainer = shap.TreeExplainer(_model)


def run_risk_prediction(profile: dict) -> dict:
    row = pd.DataFrame([{f: profile.get(f) for f in FEATURES}])
    proba = float(_model.predict_proba(row)[0, 1])
    label = int(proba >= 0.5)
    return {"risk_probability": proba, "risk_label": label}


def run_explainability(profile: dict) -> dict:
    row = pd.DataFrame([{f: profile.get(f) for f in FEATURES}])
    shap_values = _explainer(row)
    values = shap_values.values[0]
    contributions = sorted(zip(FEATURES, values), key=lambda x: abs(x[1]), reverse=True)
    top = [(feat, float(val)) for feat, val in contributions[:5]]
    summary_lines = []
    for feat, val in top:
        direction = "increases risk" if val > 0 else "decreases risk"
        summary_lines.append(f"{feat} {direction} (SHAP: {val:+.3f})")
    return {"top_factors": top, "summary": "; ".join(summary_lines)}
