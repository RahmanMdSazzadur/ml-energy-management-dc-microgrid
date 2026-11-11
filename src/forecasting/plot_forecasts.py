"""Figure: MAE by horizon (LightGBM vs persistence) and an example 2023 week of 24 h-ahead forecasts."""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.dates as mdates

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results/forecasting"
LAB = {"price": ("Spot price", "c/kWh"), "load": ("Building load", "kW"), "pv": ("PV output", "W/kWp")}


def _pred(t):
    p = OUT / f"pred_{t}_2023.parquet"
    return pd.read_parquet(p) if p.exists() else pd.read_csv(OUT / f"pred_{t}_2023.csv.gz")


def main():
    byh = pd.read_csv(OUT / "forecast_mae_by_horizon_2023.csv")
    fig, ax = plt.subplots(3, 2, figsize=(11, 9), gridspec_kw={"width_ratios": [1, 2.2]})
    for i, t in enumerate(["price", "load", "pv"]):
        name, unit = LAB[t]
        b = byh[byh.target == t]
        ax[i, 0].plot(b.h, b.ae, label="LightGBM", color="#1f6f8b", lw=2)
        ax[i, 0].plot(b.h, b.ae_pers, label="Persistence (24 h)", color="#999", ls="--")
        ax[i, 0].set_ylabel(f"MAE ({unit})"); ax[i, 0].set_title(f"{name}: error vs horizon", fontsize=10)
        ax[i, 0].set_xlim(1, 24)
        p = _pred(t)
        p["target_time"] = pd.to_datetime(p["target_time"], utc=True)
        w = p[(p.h == 24) & (p.target_time >= "2023-03-06") & (p.target_time < "2023-03-13")]
        ax[i, 1].plot(w.target_time, w.y, color="k", lw=1.2, label="Actual")
        ax[i, 1].plot(w.target_time, w.yhat, color="#1f6f8b", lw=1.2, label="24 h-ahead forecast")
        ax[i, 1].xaxis.set_major_formatter(mdates.DateFormatter("%a %d"))
        ax[i, 1].set_ylabel(unit); ax[i, 1].set_title(f"{name}: week of 6 Mar 2023 (test year)", fontsize=10)
    ax[2, 0].set_xlabel("Forecast horizon (h)")
    ax[0, 0].legend(fontsize=8); ax[0, 1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(ROOT / "results/figures/forecast_performance_2023.png", dpi=160)


if __name__ == "__main__":
    main()
