# Texas Housing Market Analysis

A Python project that analyses the key drivers of single-family home prices across the five major Texas metros — **Austin, Dallas, Fort Worth, Houston, and San Antonio** — and delivers actionable consumer insights for home buyers.

---

## Features

| Module | What it does |
|---|---|
| `housing_analysis/data_loader.py` | Generates a realistic synthetic dataset of Texas single-family homes with metro-calibrated pricing |
| `housing_analysis/analysis.py` | Runs Ridge Regression, Random Forest, and Gradient Boosting to identify price drivers; computes correlations, affordability, and price-per-sqft efficiency |
| `housing_analysis/insights.py` | Translates model results into plain-language consumer recommendations; includes a buyer-preference matching tool |
| `housing_analysis/visualizations.py` | Produces 8 publication-quality charts saved to `charts/` |
| `main.py` | Single entry-point that runs the full pipeline and prints a console report |

---

## Key Findings

### Top Price Drivers (all models combined)
1. **Square Footage** — strongest single predictor of price (r ≈ 0.63)
2. **School District Rating** — each 1-point improvement ≈ 2–3 % price premium
3. **Property Tax Rate** — higher rates negatively correlate with price
4. **Distance from City Center** — 5–10 mi closer ≈ 10–15 % more value
5. **Lot Size / Age of Home** — secondary but meaningful drivers

### City Market Tiers

| City | Median Price | $/sq ft | School Rating | Tier |
|---|---|---|---|---|
| Austin | ~$634 k | ~$324 | 8.1 | Premium |
| Dallas | ~$396 k | ~$194 | 7.2 | Mid-Range |
| Fort Worth | ~$356 k | ~$174 | 7.0 | Mid-Range |
| Houston | ~$346 k | ~$163 | 6.9 | Mid-Range |
| San Antonio | ~$277 k | ~$145 | 6.6 | Affordable |

### Affordability (household income $85 k/yr, 28 % rule)

| City | % of homes affordable |
|---|---|
| San Antonio | ~80 % |
| Houston | ~60 % |
| Fort Worth | ~59 % |
| Dallas | ~44 % |
| Austin | ~5 % |

---

## Consumer Tips

- 📐 **Size matters most**: the 1 800–2 500 sq ft range offers the best price-per-sqft value statewide.
- 🗺️ **Location premium**: every mile closer to the city center adds ~1–2 % to value.
- 🏫 **Schools pay dividends**: a top-rated school district protects resale value and quality of life.
- 🔑 **Newer = pricier**: homes built after 2000 carry a ~10 % premium — budget for renovation on older homes.
- 📊 **Tax-cost modelling**: at 2.1–2.3 % property tax, a $350 k home costs $7 350–$8 050/year in taxes alone.

---

## Quick Start

### 1 — Install dependencies

```bash
pip install -r requirements.txt
```

### 2 — Run the full analysis

```bash
python main.py
```

Charts are saved to `charts/`. Optional flags:

```
--samples 2000   # number of records to generate (default: 1000)
--seed 99        # random seed (default: 42)
--output my_dir  # output directory for charts (default: charts/)
```

### 3 — Run tests

```bash
python -m pytest tests/ -v
```

---

## Project Structure

```
housing_analysis/
├── housing_analysis/         # Main package
│   ├── __init__.py
│   ├── data_loader.py        # Synthetic Texas housing dataset
│   ├── analysis.py           # Price-driver models & statistics
│   ├── insights.py           # Consumer insights & buyer matching
│   └── visualizations.py     # Chart generation
├── tests/
│   ├── test_data_loader.py
│   ├── test_analysis.py
│   ├── test_insights.py
│   └── test_visualizations.py
├── charts/                   # Generated PNG charts (git-ignored)
├── main.py                   # CLI entry point
├── requirements.txt
└── README.md
```

---

## Charts

The pipeline generates 8 charts:

| Chart | Description |
|---|---|
| `price_distribution.png` | Box-plot of home prices by city |
| `price_vs_sqft.png` | Scatter: price vs. square footage, coloured by city |
| `feature_importance.png` | Ranked bar chart of price-driver importance |
| `correlation_heatmap.png` | Pearson correlation matrix |
| `affordability.png` | Median price vs. affordability threshold by city |
| `price_per_sqft.png` | Violin plot of $/sq ft by city |
| `school_vs_price.png` | School rating vs. price scatter with trend line |
| `age_vs_price.png` | Home age vs. price scatter with trend line |

---

## Data Notes

The dataset is **synthetically generated** to reflect realistic Texas market characteristics (metro-level median prices, school ratings, tax rates, and distributions) sourced from public market data. It is intended for educational and analytical purposes.
