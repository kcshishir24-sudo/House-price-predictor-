"""gen_data.py - simulate Nepalese residential listings (Report Section 4.2.3).

A verified real dataset was not available, so listings are simulated. Price =
(land value + building value) * noise, where land value is area x district rate
with step changes for road width and a hidden locality effect, and building
value is built-up area x a construction cost that depreciates with age.
Duplicates, missing values and some wrong prices are injected on purpose so the
cleaning steps in train.py can be exercised.

Replace data/listings.csv with real listings that have the same columns and run
`python3 train.py` to retrain on real data.
"""
import numpy as np
import pandas as pd
from unit_conversion import to_sqft

RNG = np.random.default_rng(42)

# district -> (share of listings, land rate NPR per sq ft, typical units)
DISTRICTS = {
    "Kathmandu": (0.30, 5400, ["aana", "ropani", "sqft"]),
    "Lalitpur":  (0.21, 4000, ["aana", "ropani", "sqft"]),
    "Pokhara":   (0.155, 2400, ["aana", "ropani", "sqft"]),
    "Bhaktapur": (0.15, 3000, ["aana", "ropani", "sqft"]),
    "Butwal":    (0.10, 1300, ["katha", "dhur", "sqft"]),
    "Chitwan":   (0.085, 1150, ["katha", "dhur", "sqft"]),
}
LOCALITY_SIGMA = 0.12
CONSTRUCTION_COST = 2000  # NPR per sq ft of built-up area (new)


def road_factor(w):
    if w < 8:
        return 0.75
    if w < 12:
        return 0.90
    if w < 20:
        return 1.00
    if w < 30:
        return 1.12
    return 1.25


def simulate(n=3045):
    names = list(DISTRICTS)
    probs = np.array([DISTRICTS[d][0] for d in names])
    probs = probs / probs.sum()
    locality = {d: RNG.normal(0, LOCALITY_SIGMA, 8) for d in names}
    rows = []
    for _ in range(n):
        d = RNG.choice(names, p=probs)
        _, rate, units = DISTRICTS[d]
        unit = RNG.choice(units, p=[0.55, 0.15, 0.30] if len(units) == 3 else [0.5, 0.2, 0.3])
        # plot area in sq ft (log-normal), then express in the seller's unit
        terai = d in ("Butwal", "Chitwan")
        area_sqft = float(np.clip(RNG.lognormal(np.log(2600 if not terai else 5500), 0.55), 500, 60000))
        if unit == "sqft":
            value = round(area_sqft)
        else:
            value = round(area_sqft / to_sqft(1, unit), 2)
            if value <= 0:
                value = 0.25
        area_sqft = to_sqft(value, unit)
        road = float(np.clip(np.round(RNG.gamma(4.5, 4.0)), 6, 45))
        floors = float(RNG.choice([1, 1.5, 2, 2.5, 3, 3.5, 4], p=[.08, .07, .25, .25, .2, .1, .05]))
        bedrooms = int(np.clip(round(floors * 1.6 + RNG.normal(0, 1)), 1, 9))
        bathrooms = int(np.clip(round(bedrooms * 0.75 + RNG.normal(0, 0.6)), 1, 8))
        age = int(np.clip(RNG.gamma(2.2, 6.0), 0, 50))

        loc = np.exp(RNG.choice(locality[d]))
        land = area_sqft * rate * road_factor(road) * loc
        footprint = min(area_sqft * 0.75, 3500)
        depreciation = max(0.35, 1 - 0.018 * age)
        building = footprint * floors * CONSTRUCTION_COST * depreciation * (1 + 0.02 * (bathrooms - 2))
        price = (land + building) * RNG.lognormal(0, 0.13)
        rows.append([d, value, unit, area_sqft, road, bedrooms, bathrooms, floors, age, round(price)])
    df = pd.DataFrame(rows, columns=["district", "area_value", "area_unit", "area_sqft",
                                     "road_width_ft", "bedrooms", "bathrooms", "floors",
                                     "building_age", "price_npr"])
    return df


def add_noise(df):
    df = df.copy()
    # ~262 missing numeric cells
    num_cols = ["road_width_ft", "bedrooms", "bathrooms", "floors", "building_age"]
    for _ in range(262):
        i = RNG.integers(0, len(df)); c = RNG.choice(num_cols)
        df.loc[i, c] = np.nan
    # ~30 wrong prices (x/÷ a big factor)
    for i in RNG.choice(len(df), 30, replace=False):
        df.loc[i, "price_npr"] = round(df.loc[i, "price_npr"] * RNG.choice([0.15, 0.2, 4.5, 6.0]))
    # 45 duplicates
    dup = df.sample(45, random_state=1)
    return pd.concat([df, dup], ignore_index=True)


if __name__ == "__main__":
    df = add_noise(simulate(3000))
    df.to_csv("data/listings.csv", index=False)
    import hashlib
    open("data/.simulated_sha256", "w").write(hashlib.sha256(open("data/listings.csv", "rb").read()).hexdigest())
    print(f"Wrote data/listings.csv with {len(df)} rows")
    print(df.district.value_counts().to_string())
