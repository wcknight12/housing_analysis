"""
Page 5 — Buy vs. Wait Simulator
================================
Financial model comparing the total cost of buying now vs. waiting N years,
accounting for price appreciation, rent, mortgage costs, and opportunity cost.
"""

import os
import sys

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

st.set_page_config(page_title="Buy vs. Wait", page_icon="⚖️", layout="wide")
st.title("⚖️ Buy vs. Wait Simulator")
st.caption("Compare the 10-year total cost of buying now versus waiting and renting first")

# ---------------------------------------------------------------------------
# Sidebar inputs
# ---------------------------------------------------------------------------

st.sidebar.header("🏡 Home & Loan Parameters")
home_price = st.sidebar.number_input("Current Home Price ($)", 100_000, 2_000_000, 380_000, step=10_000)
down_pct = st.sidebar.slider("Down Payment (%)", 3, 30, 20)
mortgage_rate = st.sidebar.slider("Mortgage Interest Rate (%)", 3.0, 10.0, 6.85, step=0.05)
loan_years = st.sidebar.selectbox("Loan Term (years)", [15, 30], index=1)
property_tax_pct = st.sidebar.slider("Property Tax Rate (%)", 1.0, 3.5, 2.2, step=0.05)
annual_insurance = st.sidebar.number_input("Annual Insurance ($)", 500, 5_000, 1_400, step=100)
annual_maintenance_pct = st.sidebar.slider("Annual Maintenance (% of value)", 0.5, 3.0, 1.0, step=0.1)

st.sidebar.header("📈 Market Assumptions")
price_appreciation = st.sidebar.slider("Annual Price Appreciation (%)", -5.0, 15.0, 3.5, step=0.5)
rent_per_month = st.sidebar.number_input("Current Monthly Rent ($)", 500, 10_000, 2_100, step=50)
rent_growth = st.sidebar.slider("Annual Rent Growth (%)", 0.0, 10.0, 3.0, step=0.5)
investment_return = st.sidebar.slider("Investment Return on Down Payment (%/yr)", 0.0, 12.0, 7.0, step=0.5)
max_wait_years = st.sidebar.slider("Simulate waiting up to (years)", 1, 10, 5)

# ---------------------------------------------------------------------------
# Calculations
# ---------------------------------------------------------------------------

def simulate(
    home_price, down_pct, mortgage_rate, loan_years,
    property_tax_pct, annual_insurance, annual_maintenance_pct,
    price_appreciation, rent_per_month, rent_growth, investment_return,
    horizon=10,
):
    """
    Returns a DataFrame with one row per scenario (buy now vs. wait 1..max years)
    showing total 10-year cost and net equity.
    """
    results = []

    r_m = mortgage_rate / 100 / 12
    n = loan_years * 12

    def monthly_payment(price, down_pct):
        loan = price * (1 - down_pct / 100)
        if r_m == 0:
            return loan / n
        return loan * (r_m * (1 + r_m) ** n) / ((1 + r_m) ** n - 1)

    for wait in range(0, max_wait_years + 1):
        label = "Buy Now" if wait == 0 else f"Wait {wait} yr{'s' if wait > 1 else ''}"

        # Future home price after waiting
        future_price = home_price * (1 + price_appreciation / 100) ** wait
        down_payment = future_price * down_pct / 100
        loan = future_price * (1 - down_pct / 100)
        monthly_pmt = monthly_payment(future_price, down_pct)
        own_years = horizon - wait

        if own_years <= 0:
            results.append({"wait_years": wait, "label": label,
                             "total_cost": 0, "ending_equity": 0,
                             "net_cost": 0})
            continue

        # Ownership costs over own_years
        total_mortgage = monthly_pmt * 12 * own_years

        # Property tax + insurance + maintenance (on appreciated value)
        annual_home_costs = 0
        for y in range(int(own_years)):
            val = future_price * (1 + price_appreciation / 100) ** y
            annual_home_costs += (
                val * property_tax_pct / 100
                + annual_insurance
                + val * annual_maintenance_pct / 100
            )

        # Rent cost during waiting period
        total_rent = 0
        monthly_r = rent_per_month
        for y in range(wait):
            total_rent += monthly_r * 12
            monthly_r *= 1 + rent_growth / 100

        # Opportunity cost: down payment invested
        opp_cost = down_payment * ((1 + investment_return / 100) ** wait - 1)

        total_cost = total_mortgage + annual_home_costs + total_rent + down_payment

        # Ending equity
        end_home_value = future_price * (1 + price_appreciation / 100) ** own_years
        remaining_balance = loan
        for _ in range(int(own_years * 12)):
            interest = remaining_balance * r_m
            principal = monthly_pmt - interest
            remaining_balance = max(0, remaining_balance - principal)
        equity = end_home_value - remaining_balance

        net_cost = total_cost - equity + opp_cost

        results.append({
            "wait_years": wait,
            "label": label,
            "total_cost": total_cost,
            "down_payment": down_payment,
            "total_mortgage": total_mortgage,
            "annual_home_costs": annual_home_costs,
            "total_rent": total_rent,
            "opp_cost": opp_cost,
            "ending_equity": equity,
            "net_cost": net_cost,
            "future_home_price": future_price,
        })

    return pd.DataFrame(results)


sim = simulate(
    home_price, down_pct, mortgage_rate, loan_years,
    property_tax_pct, annual_insurance, annual_maintenance_pct,
    price_appreciation, rent_per_month, rent_growth, investment_return,
    horizon=10,
)

# ---------------------------------------------------------------------------
# Key metrics
# ---------------------------------------------------------------------------

best_scenario = sim.loc[sim["net_cost"].idxmin()]
buy_now_cost = sim.loc[sim["wait_years"] == 0, "net_cost"].values[0]

st.subheader("Simulation Results")
m1, m2, m3, m4 = st.columns(4)
m1.metric("Best Scenario", best_scenario["label"])
m2.metric("Optimal Net 10-yr Cost", f"${best_scenario['net_cost']:,.0f}")
m3.metric("Buy-Now Net Cost", f"${buy_now_cost:,.0f}")
m4.metric(
    "Savings vs Buy Now",
    f"${buy_now_cost - best_scenario['net_cost']:,.0f}",
    delta=f"{(buy_now_cost - best_scenario['net_cost']) / abs(buy_now_cost) * 100:.1f}%",
)

# ---------------------------------------------------------------------------
# Chart 1: Net 10-year cost by wait scenario
# ---------------------------------------------------------------------------

fig1 = px.bar(
    sim, x="label", y="net_cost",
    color="net_cost",
    color_continuous_scale="RdYlGn_r",
    title="Net 10-Year Cost by Scenario (lower = better)",
    labels={"net_cost": "Net Cost ($)", "label": "Scenario"},
    text_auto=True,
)
fig1.update_traces(texttemplate="$%{y:,.0f}", textposition="outside")
fig1.update_layout(yaxis_tickformat="$,.0f", coloraxis_showscale=False, height=400)
st.plotly_chart(fig1, use_container_width=True)

# ---------------------------------------------------------------------------
# Chart 2: Cost breakdown stacked bar
# ---------------------------------------------------------------------------

col_a, col_b = st.columns(2)

with col_a:
    breakdown = sim[["label", "down_payment", "total_mortgage",
                      "annual_home_costs", "total_rent", "opp_cost"]].melt(
        id_vars="label", var_name="Component", value_name="Amount"
    )
    comp_labels = {
        "down_payment": "Down Payment",
        "total_mortgage": "Mortgage Payments",
        "annual_home_costs": "Tax/Insurance/Maintenance",
        "total_rent": "Rent (while waiting)",
        "opp_cost": "Opportunity Cost",
    }
    breakdown["Component"] = breakdown["Component"].map(comp_labels)
    fig2 = px.bar(
        breakdown, x="label", y="Amount", color="Component",
        title="Cost Breakdown by Scenario",
        labels={"Amount": "Cost ($)", "label": "Scenario"},
        barmode="stack",
    )
    fig2.update_layout(yaxis_tickformat="$,.0f", height=380)
    st.plotly_chart(fig2, use_container_width=True)

with col_b:
    fig3 = px.line(
        sim, x="wait_years", y=["total_cost", "net_cost", "ending_equity"],
        title="Total Cost & Equity over Wait Scenarios",
        labels={"value": "Amount ($)", "wait_years": "Years Waited"},
        color_discrete_sequence=["#d62728", "#ff7f0e", "#2ca02c"],
    )
    fig3.update_layout(
        yaxis_tickformat="$,.0f",
        legend_title_text="Metric",
        height=380,
    )
    fig3.for_each_trace(lambda t: t.update(name={
        "total_cost": "Total Cost",
        "net_cost": "Net Cost",
        "ending_equity": "Equity Built",
    }.get(t.name, t.name)))
    st.plotly_chart(fig3, use_container_width=True)

# ---------------------------------------------------------------------------
# Recommendation
# ---------------------------------------------------------------------------

st.divider()
wait = int(best_scenario["wait_years"])
if wait == 0:
    st.success(
        f"✅ **Recommendation: Buy Now.** Based on your inputs, buying immediately "
        f"results in the lowest 10-year net cost of **${best_scenario['net_cost']:,.0f}**."
    )
else:
    st.warning(
        f"⏳ **Recommendation: Wait {wait} year{'s' if wait > 1 else ''}.** "
        f"Waiting saves approximately **${buy_now_cost - best_scenario['net_cost']:,.0f}** "
        f"over 10 years compared to buying now."
    )
st.caption(
    "Disclaimer: This is a simplified financial model for educational purposes. "
    "Consult a financial advisor before making real estate decisions."
)
