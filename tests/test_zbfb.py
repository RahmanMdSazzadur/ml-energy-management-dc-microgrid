"""Checks of the ZBFB model against figures from the literature. Run: python -m pytest -q"""
import math
import pytest
from src.battery.zbfb import ZBFBBank, ZBFBModule, ZBFBParams

P = ZBFBParams()


def cycle_efficiency(i_frac: float) -> float:
    """Charge an empty module to full, then discharge to empty, at a fraction of rated current."""
    m = ZBFBModule(P, energy_kwh=0.0)
    e_in = e_out = 0.0
    dt = 0.05
    while m.soc < 1 - 1e-9:
        e_in += m.step(i_frac * P.p_charge_max_kw, dt_h=dt) * dt
    while m.energy_kwh > 1e-9:
        e_out += -m.step(-i_frac * P.p_charge_max_kw, dt_h=dt) * dt
    return e_out / e_in


def test_module_size():
    assert P.e_max_kwh == pytest.approx(11.44)
    assert P.p_charge_max_kw == pytest.approx(2.6)
    assert P.p_discharge_max_kw == pytest.approx(4.68)


def test_round_trip_at_rated_current_near_80_percent():
    # Redflow guide: about 80 % typical; Roberts et al.: 80 % peak, 73 % for a well-run modular bank
    eff = cycle_efficiency(1.0)
    assert 0.72 <= eff <= 0.85, eff


def test_efficiency_peaks_at_mid_current_and_collapses_at_low_current():
    # resistive loss dominates at high current, the fixed parasitic loss at low current, so
    # efficiency peaks in between and falls steeply at low current (the reason to idle modules)
    e10, e50, e100 = cycle_efficiency(0.1), cycle_efficiency(0.5), cycle_efficiency(1.0)
    assert e50 > e100 and e50 > e10
    assert e10 < e100 - 0.15


def test_full_module_with_pumps_on_drains_in_about_7_days():
    # Roberts et al.: the manual says a full module drains in about 7 days (basis of the 1.3 A)
    m = ZBFBModule(P, energy_kwh=P.e_max_kwh)
    hours = 0
    while m.energy_kwh > 0:
        m.step(0.0, active=True)
        hours += 1
    assert 6.5 * 24 <= hours <= 7.5 * 24, hours / 24


def test_idle_module_keeps_charge():
    m = ZBFBModule(P, energy_kwh=5.0)
    for _ in range(48):
        m.step(0.0)
    assert m.energy_kwh == pytest.approx(5.0)


def test_energy_balance_charge():
    m = ZBFBModule(P, energy_kwh=2.0)
    p = m.step(2.0)
    assert m.energy_kwh - 2.0 == pytest.approx(p - m.loss_kw(p) - P.p_parasitic_kw)


def test_cannot_overfill_or_overdrain():
    m = ZBFBModule(P, energy_kwh=P.e_max_kwh - 0.5)
    m.step(2.6)
    assert m.energy_kwh == pytest.approx(P.e_max_kwh)
    m = ZBFBModule(P, energy_kwh=0.3)
    delivered = -m.step(-4.0)
    assert m.energy_kwh == pytest.approx(0.0, abs=1e-9)
    assert 0 < delivered < 0.3


def test_power_limits():
    m = ZBFBModule(P, energy_kwh=5.0)
    assert m.step(10.0) == pytest.approx(P.p_charge_max_kw)
    assert m.step(-10.0) == pytest.approx(-P.p_discharge_max_kw)


def test_strip_completes_after_two_hours_empty_and_flags_overdue():
    m = ZBFBModule(P, energy_kwh=0.0)
    m.step(0.0, active=True)
    m.step(0.0, active=True)
    assert m.strips_completed == 1 and m.hours_since_strip == 0
    m.energy_kwh = 5.0
    for _ in range(80):
        m.step(0.0)
    assert m.overdue_hours == pytest.approx(8.0)


def test_bank_uses_fewest_modules_at_part_load():
    bank = ZBFBBank(6, P, soc0=0.5)
    bank.step(-2.0)
    drained = [m for m in bank.modules if m.energy_kwh < 0.5 * P.e_max_kwh - 1e-9]
    assert len(drained) == 1
    bank.step(-10.0)   # needs ceil(10 / 4.68) = 3 modules
    assert sum(m.energy_kwh < 0.5 * P.e_max_kwh - 1e-9 for m in bank.modules) >= 3
