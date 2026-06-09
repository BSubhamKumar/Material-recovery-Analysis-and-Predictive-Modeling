# Hydrometallurgical SX Section — Flask Prediction App

## Project Structure

```
sx_app/
├── app.py                   ← Flask application (this file)
├── copper_optimizer.pkl     ← Trained Ridge + Scaler bundle  ← YOU MUST PROVIDE THIS
├── requirements.txt
└── templates/
    └── index.html
```

---

## Step 1 — Generate the model file from your notebook

In `final.ipynb`, the last cell already saves the model:

```python
joblib.dump({
    "model":    ridge_search.best_estimator_,
    "scaler":   scaler,
    "features": X.columns.tolist()
}, "copper_optimizer.pkl")
```

Run that cell (or the full notebook with your Excel data) to produce
`copper_optimizer.pkl`, then copy it into `sx_app/`.

---

## Step 2 — Install dependencies

```bash
pip install -r requirements.txt
```

`scikit-optimize` is required for full Bayesian optimisation (50 trials).
Without it the app falls back to a 10×10 grid search automatically.

---

## Step 3 — Run the app

```bash
cd sx_app
python app.py
```

Open **http://127.0.0.1:5000** in your browser.

---

## Input Fields

| Module | Field | Example |
|--------|-------|---------|
| A — Leach Solution | Leach Solution Sec-1 | 30028.32 |
| A — Leach Solution | Leach Solution Sec-2 | 30079.88 |
| A — Leach Solution | Leach Acidity | 0 |
| A — Leach Solution | Leach pH | 1.427 |
| A — Leach Solution | Leach Fe | 34.805 |
| A — Leach Solution | Leach Cu | 0.702 |
| B — Lean Phase | Lean Cu Conc | 34.208 |
| B — Lean Phase | Lean Fe Conc | 8.03 |
| B — Lean Phase | Lean H₂SO₄ | 180.5 |
| B — Lean Phase | Lean Flow | 3048.45 |
| C — Flow | Aqueous Flow | 60108.2 |
| C — Flow | Organic Flow | 33087.87 |
| C — Flow | Solvent-to-Leach Flow Ratio | 1.1 |
| C — Flow | Phase Engagement Time | 10 |

---

## How it works

1. **Initial prediction** — user inputs are mapped to model features;
   `input_copper` and `cu_fe_ratio` are derived automatically.
2. **Bayesian optimisation** — `gp_minimize` (scikit-optimize) searches
   aqueous_flow and organic_flow in a ±10 % window around the input
   values to maximise extraction percentage.
3. **Results** — both initial and optimised predictions are displayed
   side-by-side with a Δ column and the optimal flow values.
4. **History** — previous runs are stored in `sessionStorage` and shown
   in the left panel with a live extraction trend chart.
