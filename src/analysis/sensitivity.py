"""Phase 5: collect all scenario summaries, compute economics and draw the sensitivity figure.

Economics per module (real terms, 5 %, 25 y project):
  annualised cost(c) = c*E_mod*CRF*(1 + f_repl) + O&M(c),  O&M = 1 %/yr of capital
  f_repl = present value of the year-15 replacement net of salvage at year 25, as a share of capital.
Break-even capital cost is where the annual saving equals the annualised cost.

Run:  python -m src.analysis.sensitivity
"""
from __future__ import annotations

import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
MPC = ROOT / "results/mpc"
OUT = ROOT / "results/sensitivity"
FIG = ROOT / "results/figures"
E_MOD = 11.44
R, N_YR, OM = 0.05, 25, 0.01
CRF = R * (1 + R) ** N_YR / ((1 + R) ** N_YR - 1)
F_REPL = 1 / (1 + R) ** 15 - (5 / 15) / (1 + R) ** 25
LABEL = {"rule": "Self-consumption rule", "homer": "HOMER Pro", "mpc_fc": "MPC (ML forecasts)",
         "mpc_pf": "MPC (perfect foresight)"}


def annual_cost(capex_per_kwh: float, n: int) -> float:
    cap = capex_per_kwh * E_MOD * n
    return cap * CRF * (1 + F_REPL) + OM * cap


def breakeven(saving: float, n: int) -> float:
    return max(saving, 0.0) / (E_MOD * n * (CRF * (1 + F_REPL) + OM))


def collect() -> pd.DataFrame:
    rows = []
    for f in sorted(MPC.glob("summary_N*_PV*.csv")):
        m = re.match(r"summary_N(\d+)_PV(\d+)(?:_(\d{4}))?(?:_(.+))?\.csv", f.name)
        if not m or (m.group(4) and "none" not in m.group(4)):  # skip partial (per-strategy) summaries
            continue
        n, pv, yr = int(m.group(1)), int(m.group(2)), int(m.group(3) or 2023)
        s = pd.read_csv(f)
        s["modules"], s["pv_kwp"], s["year"] = n, pv, yr
        rows.append(s)
    t = pd.concat(rows, ignore_index=True)
    t["kwh"] = t["modules"] * E_MOD
    t["strip_overdue_pct"] = 100 * t["strip_overdue_h"] / (8736 * t["modules"])
    t["breakeven_eur_per_kwh"] = [breakeven(s, n) if k != "none" else np.nan
                                  for s, n, k in zip(t["saving_vs_none_eur"], t["modules"], t["strategy"])]
    t["net_benefit_at_650"] = t["saving_vs_none_eur"] - [annual_cost(650, n) for n in t["modules"]]
    t.loc[t.strategy == "none", ["net_benefit_at_650", "share_of_perfect_pct"]] = np.nan
    pf = t[t.strategy == "mpc_pf"].set_index(["modules", "pv_kwp", "year"])["saving_vs_none_eur"]
    t["share_of_perfect_pct"] = [100 * s / pf.get((n, p, y), np.nan)
                                 for s, n, p, y in zip(t["saving_vs_none_eur"], t["modules"], t["pv_kwp"], t["year"])]
    return t


def figure(t: pd.DataFrame) -> None:
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.6))
    col = {"rule": "#9aa5b1", "homer": "#d1495b", "mpc_fc": "#1f6f8b", "mpc_pf": "#66a182"}
    # (a) price year, N=1 PV20
    a = t[(t.modules == 1) & (t.pv_kwp == 20) & (t.strategy != "none")]
    years = sorted(a.year.unique())
    strats = [s for s in ["rule", "homer", "mpc_fc", "mpc_pf"] if s in a.strategy.unique()]
    w = 0.8 / len(strats)
    for j, s in enumerate(strats):
        v = [a[(a.year == y) & (a.strategy == s)]["saving_vs_none_eur"].sum() if
             len(a[(a.year == y) & (a.strategy == s)]) else np.nan for y in years]
        ax[0].bar(np.arange(len(years)) + (j - (len(strats) - 1) / 2) * w, v, w, color=col[s], label=LABEL[s])
    ax[0].set_xticks(range(len(years)), [f"{y}\n" + {2019: "calm", 2022: "volatile", 2023: "base"}.get(y, "") for y in years])
    ax[0].set_ylabel("Saving vs no battery (€/yr)"); ax[0].set_title("(a) Price year, 1 module, PV 20 kW", fontsize=10)
    ax[0].legend(fontsize=8)
    # (b) battery size and PV size, 2023, MPC forecasts
    b = t[(t.year == 2023) & (t.strategy == "mpc_fc")].sort_values("modules")
    for pv, mk in [(20, "o-"), (40, "s--")]:
        bb = b[b.pv_kwp == pv]
        if len(bb):
            ax[1].plot(bb["kwh"], bb["saving_vs_none_eur"], mk, color="#1f6f8b", label=f"MPC (forecasts), PV {pv} kW")
    r = t[(t.year == 2023) & (t.strategy == "rule") & (t.pv_kwp == 20)].sort_values("modules")
    ax[1].plot(r["kwh"], r["saving_vs_none_eur"], "o:", color="#9aa5b1", label="Rule, PV 20 kW")
    ax[1].set_xlabel("ZBFB capacity (kWh)"); ax[1].set_ylabel("Saving vs no battery (€/yr)")
    ax[1].set_title("(b) Battery and PV size, 2023", fontsize=10); ax[1].legend(fontsize=8)
    # (c) net annual benefit vs battery capital cost
    cap = np.linspace(50, 700, 66)
    for (n, pv, yr), ls in [((1, 20, 2023), "-"), ((4, 20, 2023), "--"), ((1, 20, 2022), ":")]:
        row = t[(t.modules == n) & (t.pv_kwp == pv) & (t.year == yr) & (t.strategy == "mpc_fc")]
        if len(row):
            s = row["saving_vs_none_eur"].item()
            ax[2].plot(cap, [s - annual_cost(c, n) for c in cap], ls, color="#1f6f8b",
                       label=f"{n} module{'s' if n > 1 else ''}, {yr}")
    ax[2].axhline(0, color="k", lw=0.8); ax[2].axvline(650, color="#d1495b", lw=0.8, ls="--")
    ax[2].text(655, ax[2].get_ylim()[0] * 0.9, "assumed\n650 €/kWh", color="#d1495b", fontsize=8)
    ax[2].set_xlabel("ZBFB capital cost (€/kWh)"); ax[2].set_ylabel("Net annual benefit (€/yr)")
    ax[2].set_title("(c) Economics with MPC (ML forecasts)", fontsize=10); ax[2].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "sensitivity.png", dpi=160)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    t = collect()
    cols = ["year", "pv_kwp", "modules", "kwh", "strategy", "net_cost_eur", "saving_vs_none_eur",
            "share_of_perfect_pct", "effective_rte", "strip_overdue_pct", "breakeven_eur_per_kwh", "net_benefit_at_650"]
    t = t[cols].sort_values(["year", "pv_kwp", "modules", "strategy"])
    t.to_csv(OUT / "all_scenarios.csv", index=False)
    pd.set_option("display.width", 200)
    print(t.round(1).to_string(index=False))
    figure(t)


if __name__ == "__main__":
    main()
