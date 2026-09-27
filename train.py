"""train.py - clean data, train four regressors, select the best by CV R^2 (Report 4.2.3, 4.3.2).

Usage:  python3 train.py [path/to/listings.csv]
Outputs: models/house_price_model.joblib, models/meta.json, data/listings_clean.csv
"""
import json
import sys
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (mean_absolute_error, mean_absolute_percentage_error,
                             mean_squared_error, r2_score)
from sklearn.model_selection import cross_val_predict, cross_val_score, train_test_split, KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeRegressor

from unit_conversion import to_sqft

SEED = 42
CATEGORICAL = ["district"]
NUMERIC = ["area_sqft", "road_width_ft", "bedrooms", "bathrooms", "floors", "building_age"]
FEATURES = CATEGORICAL + NUMERIC
TARGET = "price_npr"


# ------------------------------------------------------------------ cleaning
def clean(df):
    log = {"rows_raw": len(df)}
    df = df.drop_duplicates().copy()
    log["duplicates_removed"] = log["rows_raw"] - len(df)

    # recompute area in sq ft from the raw value and unit with the one converter
    df["area_sqft"] = [to_sqft(v, u) for v, u in zip(df["area_value"], df["area_unit"])]

    # median imputation of missing numeric cells
    cols = NUMERIC[1:]
    log["missing_cells_imputed"] = int(df[cols].isna().sum().sum())
    df[cols] = df[cols].fillna(df[cols].median())

    # outliers: price per sq ft of plot > 3 IQR from the district median
    ppsf = df[TARGET] / df["area_sqft"]
    keep = pd.Series(True, index=df.index)
    for _, idx in df.groupby("district").groups.items():
        s = ppsf.loc[idx]
        q1, q3 = s.quantile(.25), s.quantile(.75)
        iqr = q3 - q1
        med = s.median()
        keep.loc[idx] = ((s - med).abs() <= 3 * iqr)
    log["outliers_removed"] = int((~keep).sum())
    df = df[keep].reset_index(drop=True)
    log["rows_clean"] = len(df)
    return df, log


# ------------------------------------------------------------------ pipeline
def preprocessor():
    # plot area is right-skewed, so it is log-transformed before scaling
    area = Pipeline([("log", FunctionTransformer(np.log)), ("scale", StandardScaler())])
    return ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore"), ["district"]),
        ("area", area, ["area_sqft"]),
        ("num", StandardScaler(), NUMERIC[1:])])


def make_model(reg):
    # target is log(1 + price) during training; predictions come back in NPR
    pipe = Pipeline([("prep", preprocessor()), ("model", reg)])
    return TransformedTargetRegressor(regressor=pipe, func=np.log1p, inverse_func=np.expm1)


def candidates():
    return {
        "Linear Regression": make_model(LinearRegression()),
        "Decision Tree": make_model(DecisionTreeRegressor(max_depth=12, min_samples_leaf=3, random_state=SEED)),
        "Random Forest": make_model(RandomForestRegressor(n_estimators=300, min_samples_leaf=2,
                                                          random_state=SEED, n_jobs=-1)),
        "Gradient Boosting": make_model(GradientBoostingRegressor(n_estimators=300, learning_rate=0.05,
                                                                  max_depth=4, subsample=0.8, random_state=SEED)),
    }


def metrics(y, p):
    return {"r2": r2_score(y, p),
            "rmse_lakh": float(np.sqrt(mean_squared_error(y, p)) / 1e5),
            "mae_lakh": float(mean_absolute_error(y, p) / 1e5),
            "mape_pct": float(mean_absolute_percentage_error(y, p) * 100)}


# ---------------------------------------------------------------------- main
def main(path="data/listings.csv"):
    import hashlib, os
    raw = pd.read_csv(path)
    sha = hashlib.sha256(open(path, "rb").read()).hexdigest()
    marker = "data/.simulated_sha256"
    simulated = os.path.exists(marker) and open(marker).read().strip() == sha
    df, log = clean(raw)
    df.to_csv("data/listings_clean.csv", index=False)
    print("Cleaning:", log)
    print(f"Price range: NPR {df[TARGET].min()/1e5:.1f} Lakh - {df[TARGET].max()/1e7:.2f} Crore, "
          f"median {df[TARGET].median()/1e5:.1f} Lakh")

    X, y = df[FEATURES], df[TARGET]
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=SEED)
    print(f"Train {len(Xtr)} / Test {len(Xte)}")
    kf = KFold(5, shuffle=True, random_state=SEED)

    results, fitted = {}, {}
    for name, model in candidates().items():
        t0 = time.perf_counter(); model.fit(Xtr, ytr); train_s = time.perf_counter() - t0
        cv = cross_val_score(model, Xtr, ytr, cv=kf, scoring="r2")
        t0 = time.perf_counter()
        for _ in range(20):
            model.predict(Xte.iloc[[0]])
        pred_ms = (time.perf_counter() - t0) / 20 * 1000
        te = metrics(yte, model.predict(Xte))
        results[name] = {
            "train_r2": float(r2_score(ytr, model.predict(Xtr))),
            "test_r2": te["r2"], "cv_r2_mean": float(cv.mean()), "cv_r2_std": float(cv.std()),
            "rmse_lakh": te["rmse_lakh"], "mae_lakh": te["mae_lakh"], "mape_pct": te["mape_pct"],
            "pred_time_ms": pred_ms, "train_time_s": train_s}
        fitted[name] = model
        print(f"{name:18s} CV R2 {cv.mean():.3f}+/-{cv.std():.3f} | test R2 {te['r2']:.3f} | "
              f"MAE {te['mae_lakh']:.1f} L | MAPE {te['mape_pct']:.1f}%")

    best = max(results, key=lambda n: results[n]["cv_r2_mean"])
    print("Selected model:", best)
    model = fitted[best]

    # +/- band from cross-validated MAPE of the selected model on the training part
    oof = cross_val_predict(make_model(model.regressor.named_steps["model"].__class__(
        **model.regressor.named_steps["model"].get_params())), Xtr, ytr, cv=kf)
    margin = float(mean_absolute_percentage_error(ytr, oof))
    pred_te = model.predict(Xte)
    coverage = float(np.mean(np.abs(pred_te - yte) / yte <= margin) * 100)
    print(f"Range margin +/-{margin*100:.1f}% covers {coverage:.0f}% of test prices")

    imp = permutation_importance(model, Xte, yte, n_repeats=10, random_state=SEED, scoring="r2")
    importance = {f: float(v) for f, v in sorted(zip(FEATURES, imp.importances_mean), key=lambda t: -t[1])}
    print("Permutation importance:", {k: round(v, 3) for k, v in importance.items()})

    joblib.dump(model, "models/house_price_model.joblib")
    meta = {
        "features": FEATURES,
        "districts": sorted(df["district"].unique().tolist()),
        "data_source": "simulated" if simulated else "user-supplied",
        "selected_model": best,
        "range_margin": margin,
        "range_coverage_pct": coverage,
        "ranges": {c: [float(df[c].min()), float(df[c].max())] for c in NUMERIC},
        "cleaning": log,
        "results": results,
        "importance": importance,
        "split": {"train": len(Xtr), "test": len(Xte)},
    }
    json.dump(meta, open("models/meta.json", "w"), indent=2)
    np.save("models/test_actual.npy", yte.values); np.save("models/test_pred.npy", pred_te)
    print("Saved models/house_price_model.joblib and models/meta.json")
    return meta


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "data/listings.csv")
