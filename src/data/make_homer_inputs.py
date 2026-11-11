"""Write the 8760-hour input files HOMER Pro imports, for one study year.

HOMER works in the project's local standard time with no daylight saving, so the UTC data is
shifted to UTC+2 (Finnish standard time): local 1 Jan 00:00 = UTC 31 Dec 22:00 of the previous year.
Each file has one value per line, 8760 lines, no header (HOMER's single-column import format).

Electricity tariff (business customer, VAT excluded): buy = spot + retail margin + distribution charge +
electricity tax; sell = spot - selling fee, never below zero.
  * Electricity tax class I: 2.24 c/kWh energy tax + 0.013 c/kWh security-of-supply charge = 2.253 c/kWh
    for the study years (vero.fi tax tables); from 1.4.2026 the charge is 0.085, total 2.325 c/kWh.
  * Distribution: 2.50 c/kWh, representative of Finnish general distribution energy charges (Oulu Energia
    2.53, Kuopio Energia 2.56 c/kWh VAT 0 in 2026). Lappeenranta's own price list was not found; power-
    based tariffs (EUR/kW per month) are a sensitivity case.
  * Retail margin and selling fee: 0.30 c/kWh each, assumed (typical spot-product margins).

Run from the repository root:  python src/data/make_homer_inputs.py 2023
"""
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
PROC, OUT = ROOT / "data" / "processed", ROOT / "data" / "homer"

# --- tariff placeholders, c/kWh, VAT excluded -------------------------------------------------
RETAIL_MARGIN_C = 0.30      # retailer margin on spot price
DISTRIBUTION_C = 2.50       # network energy charge, representative Finnish DSO
ELECTRICITY_TAX_C = 2.253   # electricity tax class I incl. security-of-supply charge, study years
SELL_FEE_C = 0.30           # fee deducted from spot when selling surplus PV


def main(year: int) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(PROC / "inputs_2019_2023_hourly.csv", parse_dates=["time_utc"], index_col="time_utc")
    start = pd.Timestamp(f"{year - 1}-12-31 22:00")
    end = pd.Timestamp(f"{year}-12-31 21:00")
    d = df.loc[start:end].copy()
    # 2019 starts at UTC midnight, so local hours 00:00-01:59 on 1 Jan are missing: copy the first
    # available hour back (night, no sun; a two-hour approximation of load and price).
    full = pd.date_range(start, end, freq="h")
    if len(d) < len(full) and d.index.min() > start:
        n_pad = int((d.index.min() - start) / pd.Timedelta("1h"))
        d = d.reindex(full).bfill()
        print(f"  {year}: padded {n_pad} missing hours at the start of the year")
    if len(d) != 8760:
        raise ValueError(f"{year}: expected 8760 hours in local standard time, got {len(d)} "
                         "(leap years need HOMER's 8784-hour handling or dropping 29 Feb)")
    d.index = d.index + pd.Timedelta(hours=2)  # now local standard time, UTC+2
    d.index.name = "time_local_standard"

    buy = d["price_c_per_kwh"] + RETAIL_MARGIN_C + DISTRIBUTION_C + ELECTRICITY_TAX_C
    sell = (d["price_c_per_kwh"] - SELL_FEE_C).clip(lower=0)   # no paying to export

    files = {
        f"ghi_kw_m2_{year}.txt": d["ghi_wm2"] / 1000,
        f"temperature_c_{year}.txt": d["temp_air_c"],
        f"load_kw_{year}.txt": d["building_load_kw"],
        f"grid_buy_eur_kwh_{year}.txt": buy / 100,
        f"grid_sell_eur_kwh_{year}.txt": sell / 100,
        f"spot_eur_kwh_{year}.txt": d["price_c_per_kwh"] / 100,
    }
    for name, s in files.items():
        s.to_csv(OUT / name, index=False, header=False, float_format="%.5f")
    d.to_csv(OUT / f"homer_inputs_{year}_with_timestamps.csv", date_format="%Y-%m-%d %H:%M")

    print(f"{year}: wrote {len(files)} files to {OUT.relative_to(ROOT)}")
    print(f"  GHI {d['ghi_wm2'].sum()/1000:.0f} kWh/m2, load {d['building_load_kw'].sum()/1000:.1f} MWh, "
          f"peak {d['building_load_kw'].max():.1f} kW")
    print(f"  buy mean {buy.mean():.2f} c/kWh (min {buy.min():.2f}, max {buy.max():.2f}); "
          f"sell mean {sell.mean():.2f} c/kWh")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 2023)
