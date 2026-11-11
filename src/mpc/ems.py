"""Energy management strategies for the PV + ZBFB DC microgrid, evaluated on 2023.

All strategies drive the same physical plant (src.battery.zbfb.ZBFBModule: I^2R loss,
pump parasitic when active, strip cycle every 72 h held 2 h at 0 %) and the same
DC bus / converter / grid model as the HOMER-validated simulator.

Strategies
  none        PV + grid only (no battery)
  rule        self-consumption rule: store PV surplus, cover deficits, drain + hold when a strip is due
  homer       HOMER Pro cycle-charging decisions replayed through the plant (N = 1 only)
  mpc_fc      MILP model-predictive control, 24 h horizon, realistic information:
              published day-ahead prices where available, ML forecasts otherwise; ML load/PV forecasts
  mpc_pf      same MPC with perfect foresight (actual prices, load, PV) - upper bound

Run:  python -m src.mpc.ems --modules 1
"""
from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.optimize import Bounds, LinearConstraint, milp

from src.battery.zbfb import ZBFBModule, ZBFBParams

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data/processed/inputs_2019_2023_hourly.csv"
FC = ROOT / "results/forecasting"
OUT = ROOT / "results/mpc"

# tariff (VAT 0), c/kWh -> EUR/kWh
ADDER_BUY_C = 0.30 + 2.50 + 2.253
SELL_FEE_C = 0.30


@dataclass
class System:
    pv_kwp: float = 20.0
    conv_kw: float = 20.0
    eta_conv: float = 0.96
    n_modules: int = 1
    horizon: int = 24


# --------------------------------------------------------------------------- data
def load_inputs() -> pd.DataFrame:
    d = pd.read_csv(DATA, parse_dates=["time_utc"]).set_index("time_utc")
    d.index = d.index.tz_localize("UTC")
    d = d[~d.index.duplicated()].asfreq("h")
    d = d[["price_c_per_kwh", "building_load_kw", "pv_output_w_per_kwp"]].interpolate(limit=3)
    return d


def load_forecast(name: str, year: int = 2023) -> pd.DataFrame:
    p = FC / f"pred_{name}_{year}.parquet"
    f = pd.read_parquet(p) if p.exists() else pd.read_csv(FC / f"pred_{name}_{year}.csv.gz")
    f["issue_time"] = pd.to_datetime(f["issue_time"], utc=True)
    return f.pivot(index="issue_time", columns="h", values="yhat")


def buy_price(spot_c):
    return (np.asarray(spot_c) + ADDER_BUY_C) / 100.0


def sell_price(spot_c):
    return np.maximum(np.asarray(spot_c) - SELL_FEE_C, 0.0) / 100.0


def published_horizon(t: pd.Timestamp, T: int) -> int:
    """Number of hours from t (inclusive) whose day-ahead price is already published.
    Nord Pool publishes the next CET delivery day around 12:45 CET; we use 13:00 CET."""
    cet = t.tz_convert("Europe/Berlin")
    end_day = cet.normalize() + pd.Timedelta(days=2 if cet.hour >= 13 else 1)
    return int(min(T, (end_day - cet) / pd.Timedelta(hours=1)))


# --------------------------------------------------------------------------- plant
class Plant:
    def __init__(self, sysm: System, params: ZBFBParams = ZBFBParams(), soc0: float = 0.5):
        self.s = sysm
        self.p = params
        self.mods = [ZBFBModule(params=params) for _ in range(sysm.n_modules)]
        for i, m in enumerate(self.mods):
            m.energy_kwh = soc0 * params.e_max_kwh
            m.hours_since_strip = i * params.strip_interval_h / max(1, sysm.n_modules)

    def step(self, p_req: np.ndarray, pv_kw: float, load_kw: float, pb: float, ps: float) -> dict:
        p_act = np.array([m.step(float(p)) for m, p in zip(self.mods, p_req)])
        s = self.s
        net = pv_kw - load_kw - p_act.sum()        # DC-bus surplus (+) / deficit (-)
        inv_in = min(max(net, 0.0), s.conv_kw)
        curtail = max(net, 0.0) - inv_in
        rect_out = min(max(-net, 0.0), s.conv_kw)
        unserved = max(-net, 0.0) - rect_out
        buy = rect_out / s.eta_conv
        sell = inv_in * s.eta_conv
        return dict(buy=buy, sell=sell, curtail=curtail, unserved=unserved,
                    p_batt=p_act.sum(), ch=np.clip(p_act, 0, None).sum(), dis=np.clip(-p_act, 0, None).sum(),
                    E=sum(m.energy_kwh for m in self.mods), cost=buy * pb - sell * ps)


# --------------------------------------------------------------------------- MPC
class MPC:
    """MILP over T hours; per module: charge/discharge in two loss segments, on/off binaries,
    energy; system: grid buy/sell with exclusivity binary, PV curtailment; strip window binaries."""

    def __init__(self, sysm: System, params: ZBFBParams = ZBFBParams()):
        self.s, self.p = sysm, params
        T, N = sysm.horizon, sysm.n_modules
        k = params.r_int * 1000.0 / params.v_nom ** 2          # loss = k P^2 (kW)
        Pc, Pd = params.p_charge_max_kw, params.p_discharge_max_kw
        self.Pc, self.Pd, self.Em = Pc, Pd, params.e_max_kwh
        self.a = (1 - k * Pc / 2, 1 - k * 3 * Pc / 2)            # stored per kW terminal (charge)
        self.b = (1 + k * Pd / 2, 1 + k * 3 * Pd / 2)            # drained per kW terminal (discharge)
        self.par = params.p_parasitic_kw
        self.T, self.N = T, N
        # solver settings: tight for one module, looser for banks (module symmetry slows branch-and-bound)
        self.gap = 1e-3 if N == 1 else 1e-2
        self.time_limit = 10 if N == 1 else 2
        # variable layout
        self.nm = 7                                               # c1 c2 d1 d2 uc ud E
        self.ns = 4                                               # buy sell cur y
        self.nw = T - 1                                           # strip start per module
        nv_t = N * self.nm + self.ns
        self.nv = T * nv_t + N * self.nw
        self.idx_mod = lambda t, i, j: t * nv_t + i * self.nm + j
        self.idx_sys = lambda t, j: t * nv_t + N * self.nm + j
        self.idx_w = lambda i, kk: T * nv_t + i * self.nw + kk
        self._build()

    def _build(self):
        T, N, s = self.T, self.N, self.s
        rows, cols, vals = [], [], []
        lb, ub = [], []
        r = 0
        M = s.conv_kw / s.eta_conv
        self.r_bal = []
        for t in range(T):                                        # DC balance (eq)
            for i in range(N):
                for j, v in ((0, -1), (1, -1), (2, 1), (3, 1)):
                    rows.append(r); cols.append(self.idx_mod(t, i, j)); vals.append(v)
            for j, v in ((0, s.eta_conv), (1, -1 / s.eta_conv), (2, -1)):
                rows.append(r); cols.append(self.idx_sys(t, j)); vals.append(v)
            self.r_bal.append(r); lb.append(0); ub.append(0); r += 1
        self.r_en0 = {}
        for i in range(N):                                        # energy (eq)
            for t in range(T):
                for j, v in ((6, 1), (0, -self.a[0]), (1, -self.a[1]), (2, self.b[0]), (3, self.b[1]),
                             (4, self.par), (5, self.par)):
                    rows.append(r); cols.append(self.idx_mod(t, i, j)); vals.append(v)
                if t > 0:
                    rows.append(r); cols.append(self.idx_mod(t - 1, i, 6)); vals.append(-1)
                else:
                    self.r_en0[i] = r
                lb.append(0); ub.append(0); r += 1
        for t in range(T):                                        # power vs on/off, exclusivity
            for i in range(N):
                for j, u, P in ((0, 4, self.Pc / 2), (1, 4, self.Pc / 2), (2, 5, self.Pd / 2), (3, 5, self.Pd / 2)):
                    rows += [r, r]; cols += [self.idx_mod(t, i, j), self.idx_mod(t, i, u)]; vals += [1, -P]
                    lb.append(-np.inf); ub.append(0); r += 1
                rows += [r, r]; cols += [self.idx_mod(t, i, 4), self.idx_mod(t, i, 5)]; vals += [1, 1]
                lb.append(-np.inf); ub.append(1); r += 1
            rows += [r, r]; cols += [self.idx_sys(t, 0), self.idx_sys(t, 3)]; vals += [1, -M]
            lb.append(-np.inf); ub.append(0); r += 1
            rows += [r, r]; cols += [self.idx_sys(t, 1), self.idx_sys(t, 3)]; vals += [1, M]
            lb.append(-np.inf); ub.append(M); r += 1
        self.r_wsum = {}
        for i in range(N):                                        # strip windows
            for kk in range(self.nw):
                for tt in (kk, kk + 1):
                    rows += [r, r]; cols += [self.idx_mod(tt, i, 6), self.idx_w(i, kk)]; vals += [1, self.Em]
                    lb.append(-np.inf); ub.append(self.Em); r += 1
            for kk in range(self.nw):
                rows.append(r); cols.append(self.idx_w(i, kk)); vals.append(1)
            self.r_wsum[i] = r; lb.append(0); ub.append(0); r += 1
        self.A = sp.csr_matrix((vals, (rows, cols)), shape=(r, self.nv))
        self.clb, self.cub = np.array(lb, float), np.array(ub, float)
        # variable bounds / integrality
        self.vlb = np.zeros(self.nv)
        self.vub = np.zeros(self.nv)
        self.integ = np.zeros(self.nv)
        for t in range(T):
            for i in range(N):
                self.vub[[self.idx_mod(t, i, 0), self.idx_mod(t, i, 1)]] = self.Pc / 2
                self.vub[[self.idx_mod(t, i, 2), self.idx_mod(t, i, 3)]] = self.Pd / 2
                self.vub[[self.idx_mod(t, i, 4), self.idx_mod(t, i, 5)]] = 1
                self.integ[[self.idx_mod(t, i, 4), self.idx_mod(t, i, 5)]] = 1
                self.vub[self.idx_mod(t, i, 6)] = self.Em
            self.vub[self.idx_sys(t, 0)] = M
            self.vub[self.idx_sys(t, 1)] = s.conv_kw * s.eta_conv
            self.vub[self.idx_sys(t, 3)] = 1
            self.integ[self.idx_sys(t, 3)] = 1
        for i in range(N):
            for kk in range(self.nw):
                self.integ[self.idx_w(i, kk)] = 1

    def solve(self, E0, due_in, pv, load, pb, ps, strip=True):
        T, N = self.T, self.N
        clb, cub = self.clb.copy(), self.cub.copy()
        vub = self.vub.copy()
        for t in range(T):
            clb[self.r_bal[t]] = cub[self.r_bal[t]] = load[t] - pv[t]
            vub[self.idx_sys(t, 2)] = max(pv[t], 0.0)
        for i in range(N):
            clb[self.r_en0[i]] = cub[self.r_en0[i]] = E0[i]
            need = strip and due_in[i] <= T
            clb[self.r_wsum[i]] = cub[self.r_wsum[i]] = 1.0 if need else 0.0
            # latest allowed start so that 2 h at 0 % finish by the due time (at least a few hours to drain)
            kmax = int(np.clip(np.floor(due_in[i]) - 2, 3, self.nw - 1)) if need else -1
            for kk in range(self.nw):
                vub[self.idx_w(i, kk)] = 1.0 if (need and kk <= kmax) else 0.0
        c = np.zeros(self.nv)
        for t in range(T):
            c[self.idx_sys(t, 0)] = pb[t]
            c[self.idx_sys(t, 1)] = -ps[t]
            c[self.idx_sys(t, 2)] = 1e-4                         # mild preference not to curtail
        lam = np.mean(pb) * (1 / self.b[0]) / self.s.eta_conv * 0.9   # terminal value of stored kWh
        for i in range(N):
            c[self.idx_mod(T - 1, i, 6)] = -lam
        res = milp(c, constraints=LinearConstraint(self.A, clb, cub), integrality=self.integ,
                   bounds=Bounds(self.vlb, vub), options={"time_limit": self.time_limit, "mip_rel_gap": self.gap})
        if res.x is None:
            return None
        x = res.x
        p = np.array([x[self.idx_mod(0, i, 0)] + x[self.idx_mod(0, i, 1)]
                      - x[self.idx_mod(0, i, 2)] - x[self.idx_mod(0, i, 3)] for i in range(N)])
        e_next = np.array([x[self.idx_mod(0, i, 6)] for i in range(N)])
        # plan says "empty": drain the residual left by the linearised loss model so the strip counts
        p = np.where((e_next < 1e-3) & (p <= 1e-6) & (np.asarray(E0) > 0), -self.Pd, p)
        return p


# --------------------------------------------------------------------------- strategies
def run(strategy: str, sysm: System, d: pd.DataFrame, fc: dict | None = None,
        homer_ch=None, homer_dis=None, hours: pd.DatetimeIndex | None = None) -> pd.DataFrame:
    hours = hours if hours is not None else d.index
    plant = Plant(sysm)
    mpc = MPC(sysm) if strategy.startswith("mpc") else None
    T = sysm.horizon
    spot = d["price_c_per_kwh"]
    load = d["building_load_kw"]
    pv = d["pv_output_w_per_kwp"] * sysm.pv_kwp / 1000.0
    pos = {t: k for k, t in enumerate(d.index)}
    rows = []
    fails = 0
    for n, t in enumerate(hours):
        k = pos[t]
        if strategy == "none":
            p_req = np.zeros(sysm.n_modules)
        elif strategy == "homer":
            p_req = np.array([homer_ch[n] if homer_ch[n] > 1e-9 else -homer_dis[n]])
        elif strategy == "rule":
            p_req = rule_dispatch(plant, pv.iloc[k] - load.iloc[k])
        else:
            sl = slice(k, k + T)
            a_spot, a_load, a_pv = spot.iloc[sl].to_numpy(), load.iloc[sl].to_numpy(), pv.iloc[sl].to_numpy()
            if strategy == "mpc_pf":
                f_spot, f_load, f_pv = a_spot, a_load, a_pv
            else:
                hrs = len(a_spot)
                f_spot = np.r_[a_spot[0], fc["price"].loc[t].to_numpy()[: hrs - 1]]
                npub = published_horizon(t, hrs)
                f_spot[:npub] = a_spot[:npub]
                f_load = np.r_[a_load[0], fc["load"].loc[t].to_numpy()[: hrs - 1]]
                f_pv = np.r_[a_pv[0], fc["pv"].loc[t].to_numpy()[: hrs - 1] * sysm.pv_kwp / 1000.0]
            E0 = [m.energy_kwh for m in plant.mods]
            due = [m.strip_due_in_h for m in plant.mods]
            p_req = mpc.solve(E0, due, f_pv, f_load, buy_price(f_spot), sell_price(f_spot))
            if p_req is None:
                fails += 1
                p_req = mpc.solve(E0, due, f_pv, f_load, buy_price(f_spot), sell_price(f_spot), strip=False)
                if p_req is None:
                    p_req = np.zeros(sysm.n_modules)
            p_req = np.where(np.abs(p_req) < 1e-4, 0.0, p_req)
        out = plant.step(p_req, pv.iloc[k], load.iloc[k],
                         float(buy_price(spot.iloc[k])), float(sell_price(spot.iloc[k])))
        out["time"] = t
        rows.append(out)
    r = pd.DataFrame(rows).set_index("time")
    r.attrs["strips"] = sum(m.strips_completed for m in plant.mods)
    r.attrs["overdue_h"] = sum(m.overdue_hours for m in plant.mods)
    r.attrs["mpc_infeasible"] = fails
    return r


def rule_dispatch(plant: Plant, net: float) -> np.ndarray:
    """Self-consumption with strip compliance. Modules whose strip is due are drained and held."""
    p = np.zeros(len(plant.mods))
    pp = plant.p
    strip = [m.strip_due_in_h <= 6 or (m.energy_kwh <= pp.empty_tol_kwh and m.hours_at_empty > 0)
             for m in plant.mods]
    for i, m in enumerate(plant.mods):                 # drain modules due for a strip
        if strip[i] and m.energy_kwh > pp.empty_tol_kwh:
            p[i] = -pp.p_discharge_max_kw
    rem = net - p.sum()                                # remaining surplus after draining (p<0 adds to bus)
    for i, m in enumerate(plant.mods):
        if strip[i]:
            continue
        if rem > 0 and m.energy_kwh < pp.e_max_kwh - 1e-6:
            p[i] = min(rem, pp.p_charge_max_kw); rem -= p[i]
        elif rem < 0 and m.energy_kwh > pp.empty_tol_kwh:
            p[i] = -min(-rem, pp.p_discharge_max_kw); rem -= p[i]
    return p


def summarize(name: str, r: pd.DataFrame) -> dict:
    ch, dis = r["ch"].sum(), r["dis"].sum()
    return dict(strategy=name, net_cost_eur=r["cost"].sum(), grid_buy_kwh=r["buy"].sum(),
                grid_sell_kwh=r["sell"].sum(), curtail_kwh=r["curtail"].sum(), unserved_kwh=r["unserved"].sum(),
                batt_in_kwh=ch, batt_out_kwh=dis, effective_rte=dis / ch if ch > 0 else np.nan,
                strip_overdue_h=(np.nan if name == "none" else r.attrs.get("overdue_h", 0)),
                mpc_infeasible=r.attrs.get("mpc_infeasible", 0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--modules", type=int, default=1)
    ap.add_argument("--pv", type=float, default=20.0)
    ap.add_argument("--strategies", default="none,rule,homer,mpc_pf,mpc_fc")
    ap.add_argument("--hours", type=int, default=0, help="limit hours for a quick test")
    ap.add_argument("--year", type=int, default=2023)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    sysm = System(pv_kwp=a.pv, n_modules=a.modules)
    d = load_inputs()
    fc = {k: load_forecast(k, a.year) for k in ("price", "load", "pv")}
    hours = d.index[(d.index >= f"{a.year}-01-01") & (d.index < f"{a.year}-12-31")]
    hours = hours[hours.isin(fc["price"].index)]
    if a.hours:
        hours = hours[: a.hours]
    homer_ch = homer_dis = None
    strategies = a.strategies.split(",")
    if "homer" in strategies:
        if a.modules != 1 or a.pv != 20.0 or a.year != 2023:
            strategies.remove("homer")
        else:
            from src.sim.validate_homer import load_homer
            h = load_homer()
            # HOMER export is local standard time (UTC+2) starting 2023-01-01 00:00 local
            h.index = pd.date_range("2022-12-31 22:00", periods=len(h), freq="h", tz="UTC")
            h = h.reindex(hours).fillna(0.0)
            homer_ch, homer_dis = h["B_Charge Power"].to_numpy(), h["B_Discharge Power"].to_numpy()
    res = []
    tag = f"N{a.modules}_PV{int(a.pv)}" + ("" if a.year == 2023 else f"_{a.year}")
    for s in strategies:
        t0 = time.time()
        r = run(s, sysm, d, fc, homer_ch, homer_dis, hours)
        r.to_csv(OUT / f"hourly_{s}_{tag}.csv.gz")
        row = summarize(s, r)
        row["runtime_s"] = round(time.time() - t0, 1)
        res.append(row)
        print(pd.DataFrame([row]).round(2).to_string(index=False), flush=True)
    t = pd.DataFrame(res)
    base = t.loc[t.strategy == "none", "net_cost_eur"]
    if len(base):
        t["saving_vs_none_eur"] = base.item() - t["net_cost_eur"]
    suffix = "" if a.strategies == ap.get_default("strategies") else "_" + a.strategies.replace(",", "-")
    t.to_csv(OUT / f"summary_{tag}{suffix}.csv", index=False)
    print(t.round(1).to_string(index=False))


if __name__ == "__main__":
    main()
