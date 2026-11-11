"""Figure: annual savings and strip compliance per strategy; example week of battery operation vs price."""
from pathlib import Path
import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results/mpc"
NAMES = {"rule": "Self-consumption\nrule", "homer": "HOMER Pro\n(cycle charging)",
         "mpc_fc": "MPC\n(ML forecasts)", "mpc_pf": "MPC\n(perfect foresight)"}
COL = {"rule": "#9aa5b1", "homer": "#d1495b", "mpc_fc": "#1f6f8b", "mpc_pf": "#66a182"}


def main(tag="N1_PV20"):
    nmod = int(tag[1:tag.index("_")])
    s = pd.read_csv(OUT / f"summary_{tag}.csv").set_index("strategy")
    order = [k for k in ["rule", "homer", "mpc_fc", "mpc_pf"] if k in s.index]
    fig = plt.figure(figsize=(12, 7.5))
    g = fig.add_gridspec(2, 2, height_ratios=[1, 1.15])
    ax1, ax2 = fig.add_subplot(g[0, 0]), fig.add_subplot(g[0, 1])
    b1 = ax1.bar([NAMES[k] for k in order], s.loc[order, "saving_vs_none_eur"], color=[COL[k] for k in order])
    ax1.bar_label(b1, labels=[f"{v:.0f} €" for v in s.loc[order, "saving_vs_none_eur"]], fontsize=9, padding=2)
    ax1.margins(y=0.15)
    ax1.set_ylabel("Saving vs no battery (€/yr)"); ax1.set_title(f"Annual electricity-cost saving, 2023 ({nmod} module{'s' if nmod > 1 else ''}, {11.44 * nmod:.1f} kWh)", fontsize=10)
    ov = 100 * s.loc[order, "strip_overdue_h"] / (8736 * nmod)
    b2 = ax2.bar([NAMES[k] for k in order], ov, color=[COL[k] for k in order])
    ax2.bar_label(b2, labels=[f"{v:.1f} %" for v in ov], fontsize=9, padding=2)
    ax2.margins(y=0.15)
    ax2.set_ylabel("Module-hours with strip overdue (%)"); ax2.set_title("ZBFB strip-cycle compliance (72 h rule)", fontsize=10)
    ax3 = fig.add_subplot(g[1, :])
    t0, t1 = "2023-04-17", "2023-04-24"
    pr = None
    for k in [x for x in ["homer", "mpc_fc"] if x in s.index]:
        h = pd.read_csv(OUT / f"hourly_{k}_{tag}.csv.gz", parse_dates=["time"]).set_index("time")
        w = h.loc[t0:t1]
        ax3.plot(w.index, 100 * w["E"] / (11.44 * int(tag[1:tag.index('_')])), color=COL[k], lw=1.6,
                 label=NAMES[k].replace("\n", " ") + " - state of charge")
        if pr is None:
            pr = (w["cost"] * 0 + 1)  # placeholder index
    sys.path.insert(0, str(ROOT))
    from src.mpc.ems import load_inputs, buy_price
    d = load_inputs().loc[t0:t1]
    ax4 = ax3.twinx()
    ax4.plot(d.index, 100 * buy_price(d["price_c_per_kwh"]), color="k", lw=0.9, alpha=0.6, label="Buy price")
    ax4.set_ylabel("Buy price (c/kWh)")
    ax3.set_ylabel("State of charge (%)"); ax3.set_ylim(-3, 103)
    ax3.xaxis.set_major_formatter(mdates.DateFormatter("%a %d %b"))
    ax3.set_title("Example week: battery operation vs electricity price", fontsize=10)
    l1, n1 = ax3.get_legend_handles_labels(); l2, n2 = ax4.get_legend_handles_labels()
    ax3.legend(l1 + l2, n1 + n2, fontsize=8, loc="upper left", ncol=3)
    fig.tight_layout()
    fig.savefig(ROOT / f"results/figures/ems_comparison_{tag}.png", dpi=160)


if __name__ == "__main__":
    main(*sys.argv[1:])
