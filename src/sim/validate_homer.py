"""Dispatch-replay validation of our simulator against HOMER Pro.

HOMER's hourly battery decisions (charge / discharge power on the DC bus) are
replayed through our own component models:
  * idealized battery, one-way efficiency sqrt(RTE), current limits 50 A / 90 A at 52 V
  * bidirectional converter, 96 % each way, grid on the AC side
  * DC-bus energy balance and real-time grid prices
and every derived quantity (battery energy, grid purchases and sales, annual
energy cost) is compared with HOMER's own output, hour by hour and for the year.

Run:  python -m src.sim.validate_homer
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
HOURLY = ROOT / "results/homer/HOMER_hourly_2023_PV20_ZBFB1_CC.csv"
OUT_DIR = ROOT / "results/validation"
B = "ZBFB 11.44 kWh module ZBM2 ref "


@dataclass(frozen=True)
class IdealBattery:
    """HOMER 'idealized' storage model, as configured for the ZBM2 reference module."""

    v_nom: float = 52.0
    ah: float = 220.0
    rte: float = 0.78
    i_ch_max: float = 50.0
    i_dis_max: float = 90.0
    soc_min: float = 0.0
    e0_frac: float = 0.50

    @property
    def e_max(self) -> float:
        return self.v_nom * self.ah / 1000.0  # 11.44 kWh

    @property
    def eta(self) -> float:
        return float(np.sqrt(self.rte))  # one-way efficiency

    @property
    def p_ch_store_max(self) -> float:
        return self.v_nom * self.i_ch_max / 1000.0  # 2.60 kW into storage

    @property
    def p_dis_store_max(self) -> float:
        return self.v_nom * self.i_dis_max / 1000.0  # 4.68 kW out of storage


ETA_CONV = 0.96


def load_homer(path: Path = HOURLY) -> pd.DataFrame:
    h = pd.read_csv(path, skiprows=[0, 2])
    return h.rename(columns=lambda c: c.replace(B, "B_").replace("Generic flat plate PV ", "PV_"))


def replay(h: pd.DataFrame, bat: IdealBattery = IdealBattery()) -> pd.DataFrame:
    """Replay HOMER's bus-side battery decisions through our models."""
    n = len(h)
    pv = h["PV_Power Output"].to_numpy()
    load = h["DC Primary Load"].to_numpy()
    ch_req = h["B_Charge Power"].to_numpy()  # kW drawn from DC bus
    dis_req = h["B_Discharge Power"].to_numpy()  # kW delivered to DC bus

    e = np.empty(n)
    ch_bus = np.zeros(n)
    dis_bus = np.zeros(n)
    e_prev = bat.e0_frac * bat.e_max
    e_min = bat.soc_min * bat.e_max
    for t in range(n):
        # charging: stored = bus * eta, limited by current and headroom
        stored = min(ch_req[t] * bat.eta, bat.p_ch_store_max, bat.e_max - e_prev)
        # discharging: bus = drawn * eta, limited by current and available energy
        drawn = min(dis_req[t] / bat.eta, bat.p_dis_store_max, e_prev - e_min)
        ch_bus[t] = stored / bat.eta
        dis_bus[t] = drawn * bat.eta
        e_prev = e_prev + stored - drawn
        e[t] = e_prev

    net = pv + dis_bus - load - ch_bus  # DC-bus surplus (+) or deficit (-)
    sales = np.where(net > 0, net * ETA_CONV, 0.0)  # AC side
    purchases = np.where(net < 0, -net / ETA_CONV, 0.0)  # AC side
    price = h["Grid Power Price"].to_numpy()
    sell = h["Grid Sellback Rate"].to_numpy()
    return pd.DataFrame(
        {
            "E_kwh": e,
            "charge_bus_kw": ch_bus,
            "discharge_bus_kw": dis_bus,
            "grid_purchases_kw": purchases,
            "grid_sales_kw": sales,
            "energy_cost_eur": purchases * price - sales * sell,
        }
    )


def compare(h: pd.DataFrame, r: pd.DataFrame) -> pd.DataFrame:
    homer = {
        "E_kwh": h["B_Energy Content"],
        "charge_bus_kw": h["B_Charge Power"],
        "discharge_bus_kw": h["B_Discharge Power"],
        "grid_purchases_kw": h["Grid Purchases"],
        "grid_sales_kw": h["Grid Sales"],
        "energy_cost_eur": h["Grid Purchases"] * h["Grid Power Price"]
        - h["Grid Sales"] * h["Grid Sellback Rate"],
    }
    rows = []
    for k, hv in homer.items():
        hv = hv.to_numpy()
        ov = r[k].to_numpy()
        d = ov - hv
        annual_h = hv.sum() if k != "E_kwh" else hv.mean()
        annual_o = ov.sum() if k != "E_kwh" else ov.mean()
        rows.append(
            {
                "quantity": k,
                "HOMER_annual": annual_h,
                "ours_annual": annual_o,
                "annual_error_pct": 100 * (annual_o - annual_h) / annual_h if annual_h else np.nan,
                "hourly_MAE": np.abs(d).mean(),
                "hourly_max_abs": np.abs(d).max(),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    h = load_homer()
    r = replay(h)
    table = compare(h, r)
    table.to_csv(OUT_DIR / "homer_replay_validation_2023.csv", index=False)
    r.to_csv(OUT_DIR / "homer_replay_hourly_2023.csv", index=False)
    pd.set_option("display.width", 160)
    print(table.round(4).to_string(index=False))


if __name__ == "__main__":
    main()


def replay_zbfb_physics(h: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Same HOMER decisions, but through our physical ZBFB model (I^2R, pumps, strip clock)."""
    from src.battery.zbfb import ZBFBModule, ZBFBParams

    m = ZBFBModule(params=ZBFBParams())
    m.energy_kwh = 0.5 * m.params.e_max_kwh
    n = len(h)
    ch_req = h["B_Charge Power"].to_numpy()
    dis_req = h["B_Discharge Power"].to_numpy()
    p_out = np.zeros(n)
    e = np.zeros(n)
    overdue = 0
    for t in range(n):
        req = ch_req[t] if ch_req[t] > 1e-9 else (-dis_req[t] if dis_req[t] > 1e-9 else 0.0)
        p_out[t] = m.step(req)
        e[t] = m.energy_kwh
        overdue += m.strip_due_in_h < 0
    ch_bus = np.clip(p_out, 0, None)
    dis_bus = np.clip(-p_out, 0, None)
    pv = h["PV_Power Output"].to_numpy()
    load = h["DC Primary Load"].to_numpy()
    net = pv + dis_bus - load - ch_bus
    sales = np.where(net > 0, net * ETA_CONV, 0.0)
    purchases = np.where(net < 0, -net / ETA_CONV, 0.0)
    cost = purchases * h["Grid Power Price"].to_numpy() - sales * h["Grid Sellback Rate"].to_numpy()
    r = pd.DataFrame({"E_kwh": e, "charge_bus_kw": ch_bus, "discharge_bus_kw": dis_bus,
                      "grid_purchases_kw": purchases, "grid_sales_kw": sales, "energy_cost_eur": cost})
    info = {"strips_completed": m.strips_completed, "hours_strip_overdue": int(overdue)}
    return r, info


def physics_gap() -> pd.DataFrame:
    h = load_homer()
    ideal = replay(h)
    phys, info = replay_zbfb_physics(h)
    rows = []
    for k in ["charge_bus_kw", "discharge_bus_kw", "grid_purchases_kw", "grid_sales_kw", "energy_cost_eur"]:
        a, b = ideal[k].sum(), phys[k].sum()
        rows.append({"quantity": k, "HOMER_idealized": a, "ZBFB_physics": b,
                     "difference": b - a, "difference_pct": 100 * (b - a) / a})
    rte_ideal = ideal["discharge_bus_kw"].sum() / ideal["charge_bus_kw"].sum()
    rte_phys = phys["discharge_bus_kw"].sum() / phys["charge_bus_kw"].sum()
    rows.append({"quantity": "effective_RTE", "HOMER_idealized": rte_ideal, "ZBFB_physics": rte_phys,
                 "difference": rte_phys - rte_ideal, "difference_pct": 100 * (rte_phys - rte_ideal) / rte_ideal})
    t = pd.DataFrame(rows)
    t.attrs.update(info)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    t.to_csv(OUT_DIR / "homer_vs_zbfb_physics_2023.csv", index=False)
    return t
