# ML Exclusion FVA Analyzer

Streamlit app that judges whether ML exclusions in a demand planning process
are justified, using Forecast Value Added (FVA) logic: **an exclusion is only
good if the user/consensus forecast beats the ML engine's own (shadow)
forecast against shipped actuals.**

## Input

Long-format CSV or Excel — one row per item per month — at **any level**
(SKU, SPU, CVC = SKU+SPU, country, ...). Required content:

| Content | Example column |
|---|---|
| One or more dimension columns | `SapCode`, `SPU Id` |
| ML exclusion flag (Y/N) | `ML Exclusion` |
| Month / period | `Month` |
| ML forecast (shadow / engine output) | `ML Forecast` |
| User / consensus forecast | `Global Lag 0 PBU Consensus FC` |
| Shipped units (actuals) | `Shipped Units` |

Columns are auto-detected by name and fully re-mappable in the sidebar.
Pick any subset of dimension columns as the analysis level — measures are
summed to that level before scoring.

## Verdict logic

Per item, over a user-selected trailing window:

- **WMAPE** for ML and user forecasts (volume-weighted, robust to zeros)
- **FVA** = WMAPE(ML) − WMAPE(User) — positive means the human adds value
- **Monthly win rate** — how often each side was closer to actuals
- **Bias** for both sides

Lights: 🟢 exclusion justified · 🔴 exclusion not justified (review for
removal) · 🟡 inconclusive · 🔵 non-excluded item where the human would beat
ML (consider excluding) · ⚠️ shadow-ML missing (ML = user on excluded rows,
so the comparison is blind).

A built-in **data-quality panel** flags the two silent killers: missing
shadow-ML on excluded items, and high intermittency.

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

Or deploy on Streamlit Community Cloud pointing at `app.py`.

`sample_data.csv` is an anonymized 12-month example at CVC (SapCode + SPU)
level.
