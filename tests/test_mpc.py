"""MPC sanity checks on a short window."""
import numpy as np
import pytest

from src.mpc.ems import DATA, FC, MPC, System, load_forecast, load_inputs, run

needs_prices = pytest.mark.skipif(not DATA.exists() or not any(FC.glob("pred_price_2023*")),
                                  reason="spot-price data not present (see data/README_data.md)")


@needs_prices
def test_mpc_short_run_respects_physics_and_strips():
    d = load_inputs()
    fc = {k: load_forecast(k) for k in ("price", "load", "pv")}
    hours = d.index[(d.index >= "2023-04-01") & (d.index < "2023-04-06")]
    s = System(n_modules=1)
    r = run("mpc_fc", s, d, fc, hours=hours)
    assert r["unserved"].sum() == 0
    assert (r["E"] >= -1e-9).all() and (r["E"] <= 11.44 + 1e-6).all()
    assert r.attrs["overdue_h"] <= 6
    none = run("none", s, d, fc, hours=hours)
    assert r["cost"].sum() <= none["cost"].sum() + 1.0


def test_milp_prefers_charging_when_cheap_then_discharging():
    s = System(n_modules=1)
    m = MPC(s)
    T = s.horizon
    pb = np.r_[np.full(6, 0.05), np.full(T - 6, 0.30)]
    p = m.solve([0.0], [100.0], np.zeros(T), np.full(T, 10.0), pb, np.zeros(T))
    assert p[0] > 0  # charge in the cheap first hour
