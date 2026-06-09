"""
app.py — Hydrometallurgical SX Section Prediction & Optimisation
Flask backend with two optimisation modes and revenue calculation.

Requirements:
    pip install flask scikit-learn scikit-optimize joblib numpy pandas
"""

import os
import traceback

import numpy as np
import pandas as pd
import joblib
from flask import Flask, render_template, request

# ── Optional: scikit-optimize ────────────────────────────────────────────────
try:
    from skopt import gp_minimize
    from skopt.space import Real
    SKOPT_AVAILABLE = True
except ImportError:
    SKOPT_AVAILABLE = False

app = Flask(__name__)

# ── Model bundle path ─────────────────────────────────────────────────────────
MODEL_PATH = os.path.join(os.path.dirname(__file__), "copper_optimizer.pkl")

# ── Load model once at startup ────────────────────────────────────────────────
bundle = None
model_load_error = None

if os.path.exists(MODEL_PATH):
    try:
        bundle = joblib.load(MODEL_PATH)
        print("✅  Model loaded:", MODEL_PATH)
        print("    Features:", bundle["features"])
    except Exception as exc:
        model_load_error = str(exc)
        print("❌  Model load error:", exc)
else:
    model_load_error = (
        f"Model file not found: '{MODEL_PATH}'. "
        "Please place copper_optimizer.pkl in the same directory as app.py."
    )
    print("⚠️ ", model_load_error)


# ── Helpers ───────────────────────────────────────────────────────────────────

def build_row(raw: dict, features: list) -> pd.Series:
    """Convert raw form dict into a feature Series, adding derived cols."""
    row = pd.Series({col: 0.0 for col in features})

    mapping = {
        "leach_solution_sec-1": float(raw.get("leach_solution_sec1", 0)),
        "leach_solution_sec-2": float(raw.get("leach_solution_sec2", 0)),
        "leach_acidity":        float(raw.get("leach_acidity", 0)),
        "leach_ph":             float(raw.get("leach_ph", 0)),
        "leach_fe":             float(raw.get("leach_fe", 0)),
        "leach_cu":             float(raw.get("leach_cu", 0)),
        "lean_cu_conc":         float(raw.get("lean_cu_conc", 0)),
        "lean_fe_conc":         float(raw.get("lean_fe_conc", 0)),
        "lean_h2so4":           float(raw.get("lean_h2so4", 0)),
        "lean_flow":            float(raw.get("lean_flow", 0)),
        "aqueous_flow":         float(raw.get("aqueous_flow", 0)),
        "organic_flow":         float(raw.get("organic_flow", 0)),
        "solvent_to_leach_flow_ratio": float(raw.get("solvent_to_leach_flow_ratio", 0)),
        "phase_engagement_time": float(raw.get("phase_engagement_time", 10)),
    }

    for col, val in mapping.items():
        if col in row.index:
            row[col] = val

    # Derived features (must match notebook)
    leach_cu = row.get("leach_cu", 0)
    leach_fe = row.get("leach_fe", 0)
    aq_flow  = row.get("aqueous_flow", 0)

    if "input_copper" in row.index:
        row["input_copper"] = (leach_cu * aq_flow * 24 * 60) / 1e6

    if "cu_fe_ratio" in row.index:
        row["cu_fe_ratio"] = leach_cu / (leach_fe + 1e-9)

    return row


def predict_output(row: pd.Series, mdl, scl, feats: list) -> float:
    df = pd.DataFrame([row])[feats]
    scaled = scl.transform(df)
    return float(mdl.predict(scaled)[0])


def optimize_flows(row: pd.Series, mdl, scl, feats: list,
                   copper_price: float, n_calls: int = 50) -> dict:
    """
    Mode 'full': Optimise BOTH aqueous_flow and organic_flow (±10%).
    Input copper changes with aqueous flow.
    """
    base_aq  = float(row["aqueous_flow"])
    base_org = float(row["organic_flow"])
    leach_cu = float(row.get("leach_cu", 0))

    # Current (baseline) values
    actual_input  = float(row.get("input_copper", (leach_cu * base_aq * 24 * 60) / 1e6))
    actual_output = predict_output(row, mdl, scl, feats)
    actual_extraction = (actual_output / actual_input * 100) if actual_input else 0.0
    actual_loss   = actual_input - actual_output

    if not SKOPT_AVAILABLE:
        # Fallback: 10×10 grid search
        best_ext  = actual_extraction
        best_aq   = base_aq
        best_org  = base_org

        for aq in np.linspace(base_aq * 0.9, base_aq * 1.1, 10):
            for org in np.linspace(base_org * 0.9, base_org * 1.1, 10):
                temp = row.copy()
                temp["aqueous_flow"] = aq
                temp["organic_flow"] = org
                if "input_copper" in temp.index:
                    temp["input_copper"] = (leach_cu * aq * 24 * 60) / 1e6
                inp = float(temp.get("input_copper", 1e-9))
                out = predict_output(temp, mdl, scl, feats)
                ext = (out / inp * 100) if inp else 0.0
                if ext > best_ext:
                    best_ext, best_aq, best_org = ext, aq, org
    else:
        def objective(params):
            aq_flow, org_flow = params
            temp = row.copy()
            temp["aqueous_flow"] = aq_flow
            temp["organic_flow"] = org_flow
            if "input_copper" in temp.index:
                temp["input_copper"] = (leach_cu * aq_flow * 24 * 60) / 1e6
            inp = float(temp.get("input_copper", 1e-9))
            out = predict_output(temp, mdl, scl, feats)
            extraction = (out / inp * 100) if inp else 0.0
            penalty = 0.0
            if aq_flow  > base_aq  * 1.1 or aq_flow  < base_aq  * 0.9:
                penalty += 0.2
            if org_flow > base_org * 1.1 or org_flow < base_org * 0.9:
                penalty += 0.2
            return -(extraction - penalty)

        space  = [Real(base_aq * 0.9, base_aq * 1.1),
                  Real(base_org * 0.9, base_org * 1.1)]
        result = gp_minimize(objective, space, n_calls=n_calls, random_state=42)
        best_aq, best_org = result.x

    # Final prediction at optimal flows
    best_row = row.copy()
    best_row["aqueous_flow"] = best_aq
    best_row["organic_flow"] = best_org
    if "input_copper" in best_row.index:
        best_row["input_copper"] = (leach_cu * best_aq * 24 * 60) / 1e6

    opt_input  = float(best_row.get("input_copper", (leach_cu * best_aq * 24 * 60) / 1e6))
    opt_output = predict_output(best_row, mdl, scl, feats)
    opt_extraction = (opt_output / opt_input * 100) if opt_input else 0.0
    opt_loss = opt_input - opt_output

    # Leach section splits (proportional to original)
    sec1 = float(row.get("leach_solution_sec-1", 0))
    sec2 = float(row.get("leach_solution_sec-2", 0))
    total_sec = sec1 + sec2 if (sec1 + sec2) != 0 else 1
    optimal_sec1 = best_aq * (sec1 / total_sec)
    optimal_sec2 = best_aq * (sec2 / total_sec)

    # Revenue
    current_revenue  = actual_output * copper_price
    optimised_revenue = opt_output  * copper_price
    revenue_change   = optimised_revenue - current_revenue

    return {
        "mode": "full",
        # Flows
        "current_aqueous_flow":  round(base_aq, 2),
        "current_organic_flow":  round(base_org, 2),
        "best_aqueous_flow":     round(best_aq, 2),
        "best_organic_flow":     round(best_org, 2),
        # Leach splits
        "optimal_sec1":          round(optimal_sec1, 2),
        "optimal_sec2":          round(optimal_sec2, 2),
        # Copper values
        "actual_input_copper":   round(actual_input, 4),
        "opt_input_copper":      round(opt_input, 4),
        "actual_output_copper":  round(actual_output, 4),
        "opt_output_copper":     round(opt_output, 4),
        # Extraction
        "actual_extraction_pct": round(actual_extraction, 4),
        "opt_extraction_pct":    round(opt_extraction, 4),
        # Loss
        "actual_loss":           round(actual_loss, 4),
        "opt_loss":              round(opt_loss, 4),
        # Revenue
        "copper_price":          copper_price,
        "current_revenue":       round(current_revenue, 2),
        "optimised_revenue":     round(optimised_revenue, 2),
        "revenue_change":        round(revenue_change, 2),
    }


def optimize_organic_only(row: pd.Series, mdl, scl, feats: list,
                           copper_price: float) -> dict:
    """
    Mode 'organic_only': Keep aqueous_flow fixed, optimise only organic_flow (±20%).
    Input copper stays constant.
    """
    base_aq  = float(row["aqueous_flow"])   # fixed
    base_org = float(row["organic_flow"])
    leach_cu = float(row.get("leach_cu", 0))

    # Current values (aqueous fixed so input_copper stays same)
    actual_input  = float(row.get("input_copper", (leach_cu * base_aq * 24 * 60) / 1e6))
    actual_output = predict_output(row, mdl, scl, feats)
    actual_extraction = (actual_output / actual_input * 100) if actual_input else 0.0
    actual_loss   = actual_input - actual_output

    if not SKOPT_AVAILABLE:
        best_ext = actual_extraction
        best_org = base_org
        for org in np.linspace(base_org * 0.8, base_org * 1.2, 20):
            temp = row.copy()
            temp["organic_flow"] = org
            out = predict_output(temp, mdl, scl, feats)
            ext = (out / actual_input * 100) if actual_input else 0.0
            if ext > best_ext:
                best_ext, best_org = ext, org
    else:
        def objective(params):
            org_flow = params[0]
            temp = row.copy()
            temp["aqueous_flow"] = base_aq   # always fixed
            temp["organic_flow"] = org_flow
            # input_copper unchanged
            out = predict_output(temp, mdl, scl, feats)
            extraction = (out / actual_input * 100) if actual_input else 0.0
            penalty = 0.0
            if org_flow > base_org * 1.2 or org_flow < base_org * 0.8:
                penalty += 0.2
            return -(extraction - penalty)

        space  = [Real(base_org * 0.8, base_org * 1.2)]
        result = gp_minimize(objective, space, n_calls=30, random_state=42)
        best_org = result.x[0]

    # Final prediction at optimal organic flow
    best_row = row.copy()
    best_row["organic_flow"] = best_org

    opt_output = predict_output(best_row, mdl, scl, feats)
    opt_extraction = (opt_output / actual_input * 100) if actual_input else 0.0
    opt_loss = actual_input - opt_output

    # Leach splits (aqueous fixed, so sections scale similarly)
    sec1 = float(row.get("leach_solution_sec-1", 0))
    sec2 = float(row.get("leach_solution_sec-2", 0))
    total_sec = sec1 + sec2 if (sec1 + sec2) != 0 else 1
    # sections remain the same since aqueous flow is fixed
    optimal_sec1 = base_aq * (sec1 / total_sec)
    optimal_sec2 = base_aq * (sec2 / total_sec)

    # Revenue
    current_revenue   = actual_output * copper_price
    optimised_revenue = opt_output    * copper_price
    revenue_change    = optimised_revenue - current_revenue

    return {
        "mode": "organic_only",
        # Flows
        "current_aqueous_flow":  round(base_aq, 2),
        "current_organic_flow":  round(base_org, 2),
        "best_aqueous_flow":     round(base_aq, 2),   # unchanged
        "best_organic_flow":     round(best_org, 2),
        # Leach splits
        "optimal_sec1":          round(optimal_sec1, 2),
        "optimal_sec2":          round(optimal_sec2, 2),
        # Copper values
        "actual_input_copper":   round(actual_input, 4),
        "opt_input_copper":      round(actual_input, 4),   # unchanged
        "actual_output_copper":  round(actual_output, 4),
        "opt_output_copper":     round(opt_output, 4),
        # Extraction
        "actual_extraction_pct": round(actual_extraction, 4),
        "opt_extraction_pct":    round(opt_extraction, 4),
        # Loss
        "actual_loss":           round(actual_loss, 4),
        "opt_loss":              round(opt_loss, 4),
        # Revenue
        "copper_price":          copper_price,
        "current_revenue":       round(current_revenue, 2),
        "optimised_revenue":     round(optimised_revenue, 2),
        "revenue_change":        round(revenue_change, 2),
    }


# ── Routes ────────────────────────────────────────────────────────────────────

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html",
                           show_results=False,
                           error=model_load_error,
                           skopt_available=SKOPT_AVAILABLE)


@app.route("/predict", methods=["POST"])
def predict():
    if bundle is None:
        return render_template("index.html",
                               show_results=False,
                               error=model_load_error or "Model not loaded.",
                               skopt_available=SKOPT_AVAILABLE)

    mdl   = bundle["model"]
    scl   = bundle["scaler"]
    feats = bundle["features"]

    raw = request.form.to_dict()

    try:
        row = build_row(raw, feats)

        # ── Copper price from form ────────────────────────────────────────────
        copper_price = float(raw.get("copper_price", 0) or 0)

        # ── Optimisation mode ─────────────────────────────────────────────────
        mode = raw.get("opt_mode", "full")   # "full" or "organic_only"

        # ── Initial baseline prediction ───────────────────────────────────────
        input_cu  = float(row.get("input_copper", 1e-9))
        output_cu = predict_output(row, mdl, scl, feats)
        init_extraction = (output_cu / input_cu * 100) if input_cu else 0.0
        init_loss = input_cu - output_cu

        initial_preds = {
            "input_copper":          round(input_cu, 4),
            "output_copper":         round(output_cu, 4),
            "extraction_percentage": round(init_extraction, 4),
            "copper_loss":           round(init_loss, 4),
        }

        # ── Run selected optimisation ─────────────────────────────────────────
        if mode == "organic_only":
            opt = optimize_organic_only(row, mdl, scl, feats, copper_price)
        else:
            opt = optimize_flows(row, mdl, scl, feats, copper_price)

        return render_template(
            "index.html",
            show_results=True,
            raw_input=raw,
            initial_preds=initial_preds,
            opt_results=opt,
            copper_price=copper_price,
            opt_mode=mode,
            error=None,
            skopt_available=SKOPT_AVAILABLE,
        )

    except Exception:
        err_msg = traceback.format_exc()
        print(err_msg)
        return render_template(
            "index.html",
            show_results=False,
            raw_input=raw,
            error="Prediction failed: " + traceback.format_exc(limit=3),
            skopt_available=SKOPT_AVAILABLE,
        )


if __name__ == "__main__":
    app.run(debug=True, port=5000)