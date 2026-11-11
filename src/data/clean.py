"""Clean the raw PVGIS weather/PV series and Finnish spot prices into hourly UTC tables.

Inputs (data/raw/):
    pvgis_lappeenranta_2019_2023_hourly.csv  PVGIS v5.3 seriescalc, SARAH3 radiation + ERA5 weather,
                                             lat 61.059, lon 28.188. PV output is for 1 kWp, fixed at the
                                             optimal 44 deg slope, -5 deg azimuth, 14 % system loss.
    spot_price_FI_2019_2024_hourly.csv       Nord Pool day-ahead price, Finland bidding zone, EUR/MWh
                                             (downloaded from api.energy-charts.info; for publication,
                                             re-download from the ENTSO-E Transparency Platform).
    fingrid_124_consumption_FI_2019_2023.csv Fingrid open data, dataset 124 "Electricity consumption in
                                             Finland", MWh/h, CC BY 4.0. Hourly until 13.6.2023, then
                                             15-minute averages.

Outputs (data/processed/):
    weather_pv_hourly.csv      2019-2023, one row per UTC hour, satellite gaps filled and flagged
    spot_price_hourly.csv      2019-2024, one row per UTC hour, plus Helsinki local time and c/kWh
    load_hourly.csv            2019-2023, national load, its per-unit shape (mean 1 per year) and the
                               building load scaled to BUILDING_ANNUAL_MWH per year
    inputs_2019_2023_hourly.csv weather, PV, price and load joined on UTC hour
    data_summary.csv           annual figures used to check the data and choose calm/volatile years

Run from the repository root:  python src/data/clean.py
"""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RAW, OUT = ROOT / "data" / "raw", ROOT / "data" / "processed"
OUT.mkdir(parents=True, exist_ok=True)

# Stand-in building: the national load shape scaled to this annual consumption. Placeholder until a
# measured LUT building profile is available; 100 MWh/year is a mid-sized office or lab building
# (mean load about 11.4 kW).
BUILDING_ANNUAL_MWH = 100.0

IRR_COLS = ["ghi_wm2", "beam_horizontal_wm2", "diffuse_horizontal_wm2",
            "poa_44deg_wm2", "pv_output_w_per_kwp"]


def clean_pvgis() -> pd.DataFrame:
    df = pd.read_csv(RAW / "pvgis_lappeenranta_2019_2023_hourly.csv")
    # PVGIS stamps each hour at its centre-ish minute (hh:11); floor to the hour start, in UTC.
    df["time_utc"] = pd.to_datetime(df["time_utc"]).dt.floor("h").dt.tz_localize("UTC")
    df = df.set_index("time_utc").sort_index()
    assert df.index.is_unique, "duplicate hours in PVGIS data"

    # A satellite gap shows as a daytime hour with sun height 0 between two hours with the sun up.
    sun = df["sun_height_deg"]
    gap = (sun == 0) & (sun.shift(1) > 1) & (sun.shift(-1) > 1)
    df["gap_filled"] = gap.astype(int)
    df.loc[gap, IRR_COLS + ["sun_height_deg"]] = np.nan
    df[IRR_COLS + ["sun_height_deg"]] = df[IRR_COLS + ["sun_height_deg"]].interpolate(method="time")

    full = pd.date_range(df.index.min(), df.index.max(), freq="h", tz="UTC")
    assert len(full) == len(df), f"missing hours: {len(full) - len(df)}"
    return df


def clean_prices() -> pd.DataFrame:
    df = pd.read_csv(RAW / "spot_price_FI_2019_2024_hourly.csv")
    df["time_utc"] = pd.to_datetime(df["time_utc"]).dt.tz_localize("UTC")
    df = df.set_index("time_utc").sort_index()
    df = df[~df.index.duplicated(keep="first")]
    full = pd.date_range(df.index.min(), df.index.max(), freq="h", tz="UTC")
    n_missing = len(full) - len(df)
    df = df.reindex(full)
    df.index.name = "time_utc"
    df["price_filled"] = df["price_eur_mwh"].isna().astype(int)
    df["price_eur_mwh"] = df["price_eur_mwh"].interpolate(method="time")
    df["price_c_per_kwh"] = df["price_eur_mwh"] / 10.0
    df["time_helsinki"] = df.index.tz_convert("Europe/Helsinki").strftime("%Y-%m-%d %H:%M")
    print(f"prices: {n_missing} missing hours filled")
    return df


def clean_load() -> pd.DataFrame:
    df = pd.read_csv(RAW / "fingrid_124_consumption_FI_2019_2023.csv", sep=";")
    df.columns = ["start", "end", "mwh_h"]
    df["start"] = pd.to_datetime(df["start"], utc=True)
    # Rows are hourly before 13.6.2023 and 15-minute after; the value is the average power (MW) over
    # each row's period, so the hourly value is the mean of the rows that start inside the hour.
    hourly = df.groupby(df["start"].dt.floor("h"))["mwh_h"].mean()
    full = pd.date_range("2019-01-01 00:00", "2023-12-31 23:00", freq="h", tz="UTC")
    hourly = hourly.reindex(full)
    out = pd.DataFrame({"national_load_mw": hourly})
    out.index.name = "time_utc"
    out["load_filled"] = out["national_load_mw"].isna().astype(int)
    out["national_load_mw"] = out["national_load_mw"].interpolate(method="time").ffill()
    year_mean = out.groupby(out.index.year)["national_load_mw"].transform("mean")
    out["load_shape_pu"] = out["national_load_mw"] / year_mean
    hours_in_year = out.groupby(out.index.year)["national_load_mw"].transform("size")
    out["building_load_kw"] = out["load_shape_pu"] * BUILDING_ANNUAL_MWH * 1000 / hours_in_year
    print(f"load: {int(out['load_filled'].sum())} missing hours filled")
    return out


def summarise(w: pd.DataFrame, p: pd.DataFrame, l: pd.DataFrame) -> pd.DataFrame:
    wy = w.groupby(w.index.year).agg(
        ghi_kwh_m2=("ghi_wm2", lambda s: s.sum() / 1000),
        pv_yield_kwh_per_kwp=("pv_output_w_per_kwp", lambda s: s.sum() / 1000),
        mean_temp_c=("temp_air_c", "mean"),
        gap_hours_filled=("gap_filled", "sum"),
    )
    daily_spread = (p["price_eur_mwh"].resample("D").max() - p["price_eur_mwh"].resample("D").min())
    py = p.groupby(p.index.year).agg(
        price_mean_eur_mwh=("price_eur_mwh", "mean"),
        price_std_eur_mwh=("price_eur_mwh", "std"),
        price_min_eur_mwh=("price_eur_mwh", "min"),
        price_max_eur_mwh=("price_eur_mwh", "max"),
        negative_price_hours=("price_eur_mwh", lambda s: int((s < 0).sum())),
    )
    py["mean_daily_spread_eur_mwh"] = daily_spread.groupby(daily_spread.index.year).mean()
    ly = l.groupby(l.index.year).agg(
        national_load_twh=("national_load_mw", lambda s: s.sum() / 1e6),
        national_peak_mw=("national_load_mw", "max"),
        building_load_mwh=("building_load_kw", lambda s: s.sum() / 1000),
        building_peak_kw=("building_load_kw", "max"),
    )
    out = wy.join(py, how="outer").join(ly, how="outer").round(1)
    out.index.name = "year"
    return out


def main() -> None:
    w, p, l = clean_pvgis(), clean_prices(), clean_load()
    w.to_csv(OUT / "weather_pv_hourly.csv", date_format="%Y-%m-%d %H:%M")
    p.to_csv(OUT / "spot_price_hourly.csv", date_format="%Y-%m-%d %H:%M")
    l.to_csv(OUT / "load_hourly.csv", date_format="%Y-%m-%d %H:%M")
    joined = (w.join(p[["price_eur_mwh", "price_c_per_kwh", "time_helsinki"]], how="inner")
               .join(l[["national_load_mw", "load_shape_pu", "building_load_kw", "load_filled"]], how="inner"))
    joined.to_csv(OUT / "inputs_2019_2023_hourly.csv", date_format="%Y-%m-%d %H:%M")
    s = summarise(w, p, l)
    s.to_csv(OUT / "data_summary.csv")
    print(f"weather/PV rows: {len(w)}, price rows: {len(p)}, joined rows: {len(joined)}")
    print(s.to_string())


if __name__ == "__main__":
    main()
