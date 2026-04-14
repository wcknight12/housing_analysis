"""
Main entry point for the Texas Housing Market Analysis project.

Run this script to:
  1. Generate the synthetic Texas housing dataset
  2. Perform price-driver analysis (linear regression, random forest, gradient boosting)
  3. Produce consumer insights and recommendations
  4. Save all visualisation charts to the ./charts directory
  5. Print a summary report to the console

Usage
-----
    python main.py
    python main.py --samples 2000 --seed 99 --output my_charts
"""

import argparse
import os
import sys


def parse_args():
    parser = argparse.ArgumentParser(description="Texas Housing Market Analysis")
    parser.add_argument(
        "--samples",
        type=int,
        default=1_000,
        help="Number of synthetic housing records to generate (default: 1000).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="charts",
        help="Directory to save chart images (default: charts/).",
    )
    return parser.parse_args()


def print_section(title: str):
    print(f"\n{'=' * 65}")
    print(f"  {title}")
    print("=" * 65)


def main():
    args = parse_args()

    # ------------------------------------------------------------------ #
    # Imports here so --help works without heavy deps
    # ------------------------------------------------------------------ #
    from housing_analysis.data_loader import load_data
    from housing_analysis.analysis import run_full_analysis
    from housing_analysis.insights import run_insights
    from housing_analysis.visualizations import generate_all_charts

    # ------------------------------------------------------------------ #
    # 1. Load data
    # ------------------------------------------------------------------ #
    print_section("Step 1 / 4 — Loading Texas housing data")
    df = load_data(n_samples=args.samples, random_seed=args.seed)
    print(f"  Generated {len(df):,} synthetic Texas single-family home records.")
    print(f"  Cities covered : {', '.join(sorted(df['city'].unique()))}")
    print(f"  Price range    : ${df['price'].min():,.0f} – ${df['price'].max():,.0f}")
    print(f"  Median price   : ${df['price'].median():,.0f}")

    # ------------------------------------------------------------------ #
    # 2. Run analysis
    # ------------------------------------------------------------------ #
    print_section("Step 2 / 4 — Running price-driver analysis")
    results = run_full_analysis(df)

    print("\n  ── City Market Summary ──")
    print(results["descriptive"]["by_city"].to_string())

    print("\n  ── Correlations with Price ──")
    print(results["correlations"].to_string())

    print("\n  ── Combined Feature Importance ──")
    print(results["combined_importance"].to_string(index=False))

    print("\n  ── Model Performance ──")
    for model_name in ("linear_regression", "random_forest", "gradient_boosting"):
        m = results[model_name]["metrics"]
        label = model_name.replace("_", " ").title()
        print(
            f"  {label:25s}  R²={m['r2']:.4f}  CV-R²={m['cv_r2']:.4f}"
            f"  MAE=${m['mae']:,.0f}  RMSE={m['rmse']:.4f}"
        )

    print("\n  ── Affordability (household income $85k, 28% rule) ──")
    print(results["affordability"].to_string(index=False))

    # ------------------------------------------------------------------ #
    # 3. Consumer insights
    # ------------------------------------------------------------------ #
    print_section("Step 3 / 4 — Consumer insights & recommendations")
    insight_results = run_insights(results)

    print("\n  ── City Market Tiers ──")
    print(insight_results["city_summary"].to_string())

    print(f"\n  ── Best Value Cities (most affordable first) ──")
    for i, city in enumerate(insight_results["best_value_cities"], 1):
        print(f"    {i}. {city}")

    print("\n  ── Top Price Drivers ──")
    for i, driver in enumerate(insight_results["top_drivers"], 1):
        print(f"    {i}. {driver}")

    print("\n  ── Buyer Insights ──")
    for insight in insight_results["insights"]:
        print(f"  {insight}")

    # ------------------------------------------------------------------ #
    # 4. Charts
    # ------------------------------------------------------------------ #
    print_section("Step 4 / 4 — Generating visualizations")
    chart_paths = generate_all_charts(df, results, output_dir=args.output)
    for name, path in chart_paths.items():
        print(f"  ✔  {name:30s} → {path}")

    print(f"\n  All charts saved to: {os.path.abspath(args.output)}/")
    print_section("Analysis complete!")


if __name__ == "__main__":
    main()
