"""Day-ahead (1-24 h) forecasts of spot price, building load and PV for the MPC.

Design
------
* Issue time t (every hour), horizons h = 1..24, target time t+h.
* One pooled LightGBM model per target with the horizon as a feature (direct
  multi-horizon strategy). Training issue times: 2019-2022; test: 2023.
* Only information available at issue time is used: observed values up to t,
  calendar of the target hour, and sun geometry / clear-sky irradiance of the
  target hour (deterministic). No numerical weather forecast is used, so the PV
  and load forecasts are deliberately conservative; see the paper's limitations.
* Spot prices are forecast as if unpublished. In reality Nord Pool publishes the
  next day's prices around 13:00 EET, so the MPC can replace forecasts with the
  known prices for published hours; results here are a lower bound on accuracy.
* Baselines: persistence (same hour yesterday, y[t+h-24]) and weekly (y[t+h-168]).

Run:  python -m src.forecasting.day_ahead
"""
from __future__ import annotations

from pathlib import Path

import holidays
import lightgbm as lgb
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data/processed/inputs_2019_2023_hourly.csv"
OUT = ROOT / "results/forecasting"
H = 24
TARGETS = {
    "price": "price_c_per_kwh",      # c/kWh spot
    "load": "building_load_kw",      # kW
    "pv": "pv_output_w_per_kwp",     # W/kWp
}
TEST_YEAR = 2023
YEARS = range(2019, 2024)


def load_data() -> pd.DataFrame:
    d = pd.read_csv(DATA, parse_dates=["time_utc"]).set_index("time_utc")
    d.index = d.index.tz_localize("UTC")
    d = d[~d.index.duplicated()].asfreq("h")
    d = d.drop(columns=["time_helsinki"], errors="ignore")
    d = d.interpolate(limit=3)
    local = d.index.tz_convert("Europe/Helsinki")
    fi = holidays.Finland(years=range(2018, 2025))
    d["hour"] = local.hour
    d["dow"] = local.dayofweek
    d["month"] = local.month
    d["doy_sin"] = np.sin(2 * np.pi * local.dayofyear / 365.25)
    d["doy_cos"] = np.cos(2 * np.pi * local.dayofyear / 365.25)
    d["holiday"] = np.array([dt.date() in fi for dt in local]).astype(int)
    d["offday"] = ((d["dow"] >= 5) | (d["holiday"] == 1)).astype(int)
    s = np.sin(np.radians(d["sun_height_deg"].clip(lower=0)))
    d["clearsky_ghi"] = np.where(s > 0.01, 1098 * s * np.exp(-0.057 / np.maximum(s, 0.01)), 0.0)
    return d


CAL = ["hour", "dow", "month", "doy_sin", "doy_cos", "holiday", "offday"]


def build_samples(d: pd.DataFrame, target: str) -> pd.DataFrame:
    """Rows = (issue time t, horizon h). All features known at t."""
    y = d[TARGETS[target]]
    base = pd.DataFrame(index=d.index)
    base["y_t"] = y
    base["y_t1"] = y.shift(1)
    base["y_t2"] = y.shift(2)
    base["y_mean24"] = y.rolling(24).mean()
    base["y_std24"] = y.rolling(24).std()
    base["y_mean168"] = y.rolling(168).mean()
    base["temp_t"] = d["temp_air_c"]
    if target == "price":
        base["natload_t"] = d["national_load_mw"]
        base["natload_mean24"] = d["national_load_mw"].rolling(24).mean()
    if target == "pv":
        base["ghi_t"] = d["ghi_wm2"]
        base["clear_index24"] = (d["ghi_wm2"].rolling(24).sum()
                                 / d["clearsky_ghi"].rolling(24).sum().replace(0, np.nan))
    frames = []
    for h in range(1, H + 1):
        f = base.copy()
        f["h"] = h
        tgt = d.shift(-h)  # values at target time t+h
        for c in CAL:
            f[f"tgt_{c}"] = tgt[c]
        f["tgt_sun"] = tgt["sun_height_deg"]
        f["tgt_clearsky"] = tgt["clearsky_ghi"]
        f["lag24_tgt"] = y.shift(24 - h)    # y at target-24 (known since h <= 24)
        f["lag48_tgt"] = y.shift(48 - h)
        f["lag168_tgt"] = y.shift(168 - h)
        f["y_target"] = y.shift(-h)
        f["target_time"] = d.index + pd.Timedelta(hours=h)
        frames.append(f)
    s = pd.concat(frames)
    s.index.name = "issue_time"
    return s.dropna()


def train_years(test_year: int) -> list[int]:
    """2023 is the main test year (trained on 2019-2022). The sensitivity years 2019 and 2022 use
    leave-one-year-out over 2019-2022, so 2023 never enters any training set."""
    return [y for y in range(2019, 2023) if y != test_year]


def split(s: pd.DataFrame, test_year: int = TEST_YEAR):
    tt = pd.DatetimeIndex(s["target_time"])
    test = (tt.year == test_year) & (s.index.year == test_year)
    ty = train_years(test_year)
    train = s.index.year.isin(ty) & tt.year.isin(ty)
    return s[train], s[test]


def fit_predict(train: pd.DataFrame, test: pd.DataFrame, target: str):
    feats = [c for c in train.columns if c not in ("y_target", "target_time")]
    params = dict(n_estimators=600, learning_rate=0.05, num_leaves=63, min_child_samples=50,
                  subsample=0.8, subsample_freq=1, colsample_bytree=0.8, verbose=-1, random_state=0, n_jobs=1)
    if target == "price":
        params["objective"] = "l1"  # robust to price spikes
    m = lgb.LGBMRegressor(**params)
    m.fit(train[feats], train["y_target"])
    pred = m.predict(test[feats])
    if target in ("pv", "load"):
        pred = np.clip(pred, 0, None)
    if target == "pv":
        pred = np.where(test["tgt_sun"] <= 0, 0.0, pred)
    return pred, m, feats


def metrics(y, p, scale):
    e = p - y
    return {"MAE": np.mean(np.abs(e)), "RMSE": np.sqrt(np.mean(e ** 2)),
            "nMAE_pct": 100 * np.mean(np.abs(e)) / scale}


def evaluate(target: str, d: pd.DataFrame, year: int = TEST_YEAR) -> tuple[pd.DataFrame, pd.DataFrame]:
    s = build_samples(d, target)
    train, test = split(s, year)
    pred, model, feats = fit_predict(train, test, target)
    y = test["y_target"].to_numpy()
    scale = np.mean(np.abs(y)) if target != "pv" else np.mean(y[y > 0])
    if target == "pv":
        scale = 1000.0  # W/kWp, i.e. nMAE relative to rated power
    rows = []
    for name, p in [("LightGBM", pred), ("persistence_24h", test["lag24_tgt"].to_numpy()),
                    ("weekly_168h", test["lag168_tgt"].to_numpy())]:
        r = metrics(y, p, scale)
        r.update(model=name, target=target)
        rows.append(r)
    t = pd.DataFrame(rows)
    pers = t.loc[t.model == "persistence_24h", "MAE"].item()
    t["skill_vs_persistence_pct"] = 100 * (1 - t["MAE"] / pers)
    byh = (pd.DataFrame({"h": test["h"].to_numpy(), "ae": np.abs(pred - y),
                         "ae_pers": np.abs(test["lag24_tgt"].to_numpy() - y)})
           .groupby("h").mean())
    byh["target"] = target
    preds = pd.DataFrame({"issue_time": test.index, "h": test["h"].to_numpy(),
                          "target_time": test["target_time"].to_numpy(),
                          "y": y, "yhat": pred})
    preds.to_parquet(OUT / f"pred_{target}_{year}.parquet", index=False) if _has_parquet() else \
        preds.to_csv(OUT / f"pred_{target}_{year}.csv.gz", index=False)
    imp = pd.Series(model.feature_importances_, index=feats).sort_values(ascending=False)
    imp.head(15).to_csv(OUT / f"importance_{target}{'' if year == TEST_YEAR else '_' + str(year)}.csv")
    return t, byh


def _has_parquet() -> bool:
    try:
        import pyarrow  # noqa: F401
        return True
    except ImportError:
        return False


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--year", type=int, default=TEST_YEAR)
    year = ap.parse_args().year
    OUT.mkdir(parents=True, exist_ok=True)
    d = load_data()
    tables, byhs = [], []
    for tgt in TARGETS:
        t, b = evaluate(tgt, d, year)
        t["test_year"] = year
        tables.append(t)
        byhs.append(b)
        print(t.round(3).to_string(index=False), flush=True)
    pd.concat(tables).to_csv(OUT / f"forecast_metrics_{year}.csv", index=False)
    pd.concat(byhs).to_csv(OUT / f"forecast_mae_by_horizon_{year}.csv")


if __name__ == "__main__":
    main()
