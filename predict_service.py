"""predict_service.py - prediction wrapper used by /predict and /api/predict (Report 4.2.3)."""
import json
import os
import threading

import joblib
import pandas as pd

from unit_conversion import to_sqft

BASE = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE, "models", "house_price_model.joblib")
META_PATH = os.path.join(BASE, "models", "meta.json")

_lock = threading.Lock()
model = None
META = {}


def load():
    """(Re)load the saved pipeline and its metadata. Safe to call at runtime."""
    global model, META
    with _lock:
        model = joblib.load(MODEL_PATH)
        with open(META_PATH) as f:
            META = json.load(f)


load()


def format_npr(amount):
    if amount >= 1e7:
        return f"Rs. {amount / 1e7:.2f} Crore"
    return f"Rs. {amount / 1e5:.2f} Lakh"


def outside_training_range(**values):
    """names of inputs that lie outside the range seen in training"""
    return [k for k, v in values.items()
            if not META["ranges"][k][0] <= v <= META["ranges"][k][1]]


def predict_price(district, area_value, unit, road_width_ft,
                  bedrooms, bathrooms, floors, building_age):
    area_sqft = to_sqft(area_value, unit)
    row = {"district": district, "area_sqft": area_sqft, "road_width_ft": road_width_ft,
           "bedrooms": bedrooms, "bathrooms": bathrooms, "floors": floors,
           "building_age": building_age}
    X = pd.DataFrame([row], columns=META["features"])   # training feature order
    price = float(model.predict(X)[0])                   # log-price converted back to NPR
    margin = META["range_margin"]
    low, high = price * (1 - margin), price * (1 + margin)
    outside = outside_training_range(
        area_sqft=area_sqft, road_width_ft=road_width_ft, bedrooms=bedrooms,
        bathrooms=bathrooms, floors=floors, building_age=building_age)
    return {"area_sqft": area_sqft, "predicted_npr": round(price),
            "price_min_npr": round(low), "price_max_npr": round(high),
            "display": format_npr(price),
            "range_display": f"{format_npr(low)} – {format_npr(high)}",
            "outside_range": outside}
