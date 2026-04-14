"""
Data loader for Texas housing market analysis.

Generates a realistic synthetic dataset of Texas single-family home listings,
reflecting actual market characteristics for major Texas metros.
"""

import numpy as np
import pandas as pd


# Key Texas metros with median prices, price-per-sqft, and market characteristics
TEXAS_METROS = {
    "Austin": {
        "median_price": 530_000,
        "price_per_sqft": 295,
        "median_sqft": 1_900,
        "avg_school_rating": 8.1,
        "avg_distance_miles": 14.0,
        "price_growth_rate": 0.08,
        "property_tax_rate": 0.0215,
        "weight": 0.18,
    },
    "Houston": {
        "median_price": 320_000,
        "price_per_sqft": 155,
        "median_sqft": 2_100,
        "avg_school_rating": 6.8,
        "avg_distance_miles": 22.0,
        "price_growth_rate": 0.04,
        "property_tax_rate": 0.0230,
        "weight": 0.28,
    },
    "Dallas": {
        "median_price": 390_000,
        "price_per_sqft": 185,
        "median_sqft": 2_000,
        "avg_school_rating": 7.2,
        "avg_distance_miles": 18.0,
        "price_growth_rate": 0.06,
        "property_tax_rate": 0.0220,
        "weight": 0.22,
    },
    "San Antonio": {
        "median_price": 285_000,
        "price_per_sqft": 140,
        "median_sqft": 1_950,
        "avg_school_rating": 6.5,
        "avg_distance_miles": 16.0,
        "price_growth_rate": 0.05,
        "property_tax_rate": 0.0225,
        "weight": 0.18,
    },
    "Fort Worth": {
        "median_price": 345_000,
        "price_per_sqft": 168,
        "median_sqft": 1_980,
        "avg_school_rating": 7.0,
        "avg_distance_miles": 20.0,
        "price_growth_rate": 0.05,
        "property_tax_rate": 0.0218,
        "weight": 0.14,
    },
}

# Bedroom/bathroom distributions for single-family homes
BEDROOM_OPTIONS = [2, 3, 4, 5]
BEDROOM_WEIGHTS = [0.10, 0.50, 0.32, 0.08]

BATHROOM_MAP = {
    2: [1.0, 1.5, 2.0],
    3: [2.0, 2.5, 3.0],
    4: [2.5, 3.0, 3.5, 4.0],
    5: [3.0, 3.5, 4.0, 4.5, 5.0],
}


def generate_texas_housing_data(n_samples: int = 1_000, random_seed: int = 42) -> pd.DataFrame:
    """
    Generate a synthetic Texas single-family home dataset.

    Parameters
    ----------
    n_samples : int
        Number of property records to generate.
    random_seed : int
        Seed for reproducibility.

    Returns
    -------
    pd.DataFrame
        Dataset with columns:
        city, sqft, bedrooms, bathrooms, lot_size_acres, year_built,
        school_rating, distance_from_center_miles, property_tax_rate,
        price, price_per_sqft, age_years.
    """
    rng = np.random.default_rng(random_seed)

    metro_names = list(TEXAS_METROS.keys())
    metro_weights = [TEXAS_METROS[m]["weight"] for m in metro_names]

    city_indices = rng.choice(len(metro_names), size=n_samples, p=metro_weights)
    cities = [metro_names[i] for i in city_indices]

    records = []
    current_year = 2024

    for city in cities:
        metro = TEXAS_METROS[city]

        # Square footage: lognormal distribution around metro median
        sqft = int(
            np.clip(
                rng.lognormal(np.log(metro["median_sqft"]), 0.30),
                800,
                6_000,
            )
        )

        # Bedrooms and bathrooms
        bedrooms = int(rng.choice(BEDROOM_OPTIONS, p=BEDROOM_WEIGHTS))
        bath_choices = BATHROOM_MAP[bedrooms]
        bathrooms = float(rng.choice(bath_choices))

        # Lot size in acres (larger lots in smaller cities)
        lot_size_factor = 1.0 if city in ("Austin", "Dallas") else 1.3
        lot_size = round(
            float(np.clip(rng.lognormal(np.log(0.18 * lot_size_factor), 0.6), 0.05, 5.0)),
            3,
        )

        # Year built: uniform from 1960 to 2023
        year_built = int(rng.integers(1960, 2024))
        age_years = current_year - year_built

        # School rating (1–10): normal around metro average, clipped
        school_rating = round(
            float(np.clip(rng.normal(metro["avg_school_rating"], 1.2), 1.0, 10.0)),
            1,
        )

        # Distance from city center (miles)
        distance = round(
            float(np.clip(rng.lognormal(np.log(metro["avg_distance_miles"]), 0.5), 1.0, 60.0)),
            1,
        )

        # Property tax rate: slight variation around metro average
        tax_rate = round(
            float(np.clip(rng.normal(metro["property_tax_rate"], 0.001), 0.015, 0.030)),
            4,
        )

        # ---- Price model ----
        # Base: metro price-per-sqft × sqft
        base_price = metro["price_per_sqft"] * sqft

        # Adjustments (multiplicative factors)
        bedroom_adj = 1.0 + (bedrooms - 3) * 0.05
        bath_adj = 1.0 + (bathrooms - 2.0) * 0.04
        school_adj = 1.0 + (school_rating - 5.0) * 0.025
        distance_adj = 1.0 - (distance / metro["avg_distance_miles"] - 1.0) * 0.04
        age_adj = 1.0 - (age_years / 100.0) * 0.10
        lot_adj = 1.0 + (lot_size - 0.20) * 0.03

        # Random market noise (~8 % std dev)
        noise = rng.normal(1.0, 0.08)

        price = int(
            base_price
            * bedroom_adj
            * bath_adj
            * school_adj
            * distance_adj
            * age_adj
            * lot_adj
            * noise
        )
        price = max(price, 80_000)

        price_per_sqft = round(price / sqft, 2)

        records.append(
            {
                "city": city,
                "sqft": sqft,
                "bedrooms": bedrooms,
                "bathrooms": bathrooms,
                "lot_size_acres": lot_size,
                "year_built": year_built,
                "age_years": age_years,
                "school_rating": school_rating,
                "distance_from_center_miles": distance,
                "property_tax_rate": tax_rate,
                "price": price,
                "price_per_sqft": price_per_sqft,
            }
        )

    df = pd.DataFrame(records)
    return df


def load_data(n_samples: int = 1_000, random_seed: int = 42) -> pd.DataFrame:
    """
    Public entry-point: load (or generate) Texas housing data.

    Parameters
    ----------
    n_samples : int
        Number of records to generate.
    random_seed : int
        Seed for reproducibility.

    Returns
    -------
    pd.DataFrame
        Texas housing dataset.
    """
    return generate_texas_housing_data(n_samples=n_samples, random_seed=random_seed)
