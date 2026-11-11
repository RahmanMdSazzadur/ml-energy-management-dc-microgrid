"""Plot monthly PV yield and the daily price spread per year, to check the inputs and pick study years."""
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
w = pd.read_csv(ROOT / "data/processed/weather_pv_hourly.csv", parse_dates=["time_utc"], index_col="time_utc")
p = pd.read_csv(ROOT / "data/processed/spot_price_hourly.csv", parse_dates=["time_utc"], index_col="time_utc")
l = pd.read_csv(ROOT / "data/processed/load_hourly.csv", parse_dates=["time_utc"], index_col="time_utc")

pv = w["pv_output_w_per_kwp"].resample("MS").sum() / 1000
spread = (p["price_eur_mwh"].resample("D").max() - p["price_eur_mwh"].resample("D").min()).resample("MS").mean()

fig, ax = plt.subplots(1, 3, figsize=(16, 4), constrained_layout=True)
months = ["J", "F", "M", "A", "M", "J", "J", "A", "S", "O", "N", "D"]
for y, s in pv.groupby(pv.index.year):
    ax[0].plot(range(1, 13), s.values, marker="o", ms=3, label=str(y))
ax[0].set(title="PV yield, Lappeenranta (1 kWp, 44° tilt)", ylabel="kWh per kWp per month", xticks=range(1, 13), xticklabels=months)
ax[0].legend(frameon=False, fontsize=8)
for y, s in spread.groupby(spread.index.year):
    ax[1].plot(range(1, 13), s.values, marker="o", ms=3, lw=2.2 if y in (2019, 2022) else 1, label=str(y))
ax[1].set(title="Finnish day-ahead price: mean daily max–min spread", ylabel="EUR/MWh", xticks=range(1, 13), xticklabels=months)
ax[1].legend(frameon=False, fontsize=8)
l23 = l.loc["2023"].copy()
l23.index = l23.index.tz_localize("UTC").tz_convert("Europe/Helsinki")
pv23 = w.loc["2023", "pv_output_w_per_kwp"].copy()
pv23.index = pv23.index.tz_localize("UTC").tz_convert("Europe/Helsinki")
for months_sel, name, c in [((1, 2, 12), "Winter (Dec-Feb)", "C0"), ((6, 7, 8), "Summer (Jun-Aug)", "C3")]:
    sel = l23[l23.index.month.isin(months_sel)]
    for wk, ls in [(True, "-"), (False, "--")]:
        d = sel[(sel.index.dayofweek < 5) == wk]
        ax[2].plot(d.groupby(d.index.hour)["building_load_kw"].mean(), ls=ls, color=c,
                   label=f"{name}, {'weekday' if wk else 'weekend'}")
    pvs = pv23[pv23.index.month.isin(months_sel)]
    ax[2].fill_between(range(24), pvs.groupby(pvs.index.hour).mean() * 10 / 1000, color=c, alpha=.12)
ax[2].set(title="Building load 2023 (lines) vs PV per 10 kWp (shaded)", xlabel="Hour, Helsinki time", ylabel="kW", xticks=range(0, 24, 3))
ax[2].legend(frameon=False, fontsize=8)
for a in ax: a.spines[["top", "right"]].set_visible(False); a.grid(alpha=.3)
out = ROOT / "results/figures/input_data_check.png"
fig.savefig(out, dpi=150)
print(out)
