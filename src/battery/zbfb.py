"""Zinc-bromine flow battery (ZBFB) model for hourly energy-management studies.

Each module is modelled at its DC terminals with:
  * stored energy E (kWh) between 0 and E_max; 0-100 % depth of discharge is allowed
  * resistive (I^2 R) loss on charge and discharge, with I = P / V_nom
  * a fixed parasitic loss (pumps, fan) drawn from storage whenever the module is active,
    and none when it is idle (pumps off), following Roberts et al. (2024)
  * strip-cycle tracking: a module must be held at 0 % for STRIP_HOLD_H hours at least every
    STRIP_INTERVAL_H hours of clock time

Default parameters describe one Redflow ZBM2 module as reported by Roberts et al. (2024),
J. Energy Storage 81, 110280, doi:10.1016/j.est.2023.110280: 220 Ah, 52 V, 0.1 ohm,
50 A max charge, 90 A max discharge, 1.3 A parasitic, strip every 72 h held for 2 h.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field


@dataclass(frozen=True)
class ZBFBParams:
    v_nom: float = 52.0              # V, stack voltage
    capacity_ah: float = 220.0       # Ah
    r_int: float = 0.1               # ohm, internal resistance
    i_charge_max: float = 50.0       # A
    i_discharge_max: float = 90.0    # A
    i_parasitic: float = 1.3         # A drawn while the module is active
    self_discharge_idle_kw: float = 0.0   # kW lost while idle (pumps off); 0 per Roberts et al.
    strip_interval_h: float = 72.0   # h of clock time between strips
    strip_hold_h: float = 2.0        # h held at 0 % to complete a strip
    empty_tol_kwh: float = 1e-6

    @property
    def e_max_kwh(self) -> float:
        return self.capacity_ah * self.v_nom / 1000.0

    @property
    def p_charge_max_kw(self) -> float:
        return self.i_charge_max * self.v_nom / 1000.0

    @property
    def p_discharge_max_kw(self) -> float:
        return self.i_discharge_max * self.v_nom / 1000.0

    @property
    def p_parasitic_kw(self) -> float:
        return self.i_parasitic * self.v_nom / 1000.0


@dataclass
class ZBFBModule:
    params: ZBFBParams = field(default_factory=ZBFBParams)
    energy_kwh: float = 0.0
    hours_since_strip: float = 0.0
    hours_at_empty: float = 0.0
    strips_completed: int = 0
    overdue_hours: float = 0.0

    @property
    def soc(self) -> float:
        return self.energy_kwh / self.params.e_max_kwh

    def loss_kw(self, p_terminal_kw: float) -> float:
        """Resistive loss (kW) at a terminal power; positive p = charge, negative = discharge."""
        i = abs(p_terminal_kw) * 1000.0 / self.params.v_nom
        return i * i * self.params.r_int / 1000.0

    def step(self, p_request_kw: float, active: bool | None = None, dt_h: float = 1.0) -> float:
        """Advance one time step. Returns the terminal power actually delivered (kW).

        p_request_kw > 0 charges, < 0 discharges, 0 rests. `active` keeps the pumps running at
        zero power (for example while holding at 0 % during a strip); by default the module is
        active whenever it is charging or discharging.
        """
        p = self.params
        if active is None:
            active = p_request_kw != 0.0
        p_req = max(-p.p_discharge_max_kw, min(p.p_charge_max_kw, p_request_kw))
        e0 = self.energy_kwh
        parasitic = p.p_parasitic_kw if active else 0.0
        idle_loss = 0.0 if active else p.self_discharge_idle_kw

        if p_req > 0:  # charge: stored gain = P - I^2R - parasitic
            gain = (p_req - self.loss_kw(p_req) - parasitic) * dt_h
            room = p.e_max_kwh - e0
            if gain > room:  # scale terminal power down to just fill the module
                p_req = self._charge_power_for_gain(room / dt_h + parasitic)
                gain = room
            self.energy_kwh = e0 + gain
            p_out = p_req
        elif p_req < 0:  # discharge: stored drain = |P| + I^2R + parasitic
            drain = (-p_req + self.loss_kw(p_req) + parasitic) * dt_h
            if drain > e0:  # deliver only what is left after losses
                p_req = -self._discharge_power_for_drain(e0 / dt_h - parasitic)
                drain = e0
            self.energy_kwh = e0 - drain
            p_out = p_req
        else:
            self.energy_kwh = max(0.0, e0 - (parasitic + idle_loss) * dt_h)
            p_out = 0.0

        self._update_strip(dt_h)
        return p_out

    def _charge_power_for_gain(self, net_kw: float) -> float:
        # solve P - k P^2 = net for the smaller root, k = R / V^2 (kW units)
        k = self.params.r_int * 1000.0 / self.params.v_nom ** 2
        net_kw = max(0.0, net_kw)
        return (1 - math.sqrt(max(0.0, 1 - 4 * k * net_kw))) / (2 * k)

    def _discharge_power_for_drain(self, avail_kw: float) -> float:
        # solve P + k P^2 = avail for P >= 0
        k = self.params.r_int * 1000.0 / self.params.v_nom ** 2
        avail_kw = max(0.0, avail_kw)
        return (-1 + math.sqrt(1 + 4 * k * avail_kw)) / (2 * k)

    def _update_strip(self, dt_h: float) -> None:
        p = self.params
        self.hours_since_strip += dt_h
        if self.energy_kwh <= p.empty_tol_kwh:
            self.energy_kwh = 0.0
            self.hours_at_empty += dt_h
            if self.hours_at_empty >= p.strip_hold_h:
                self.strips_completed += 1
                self.hours_since_strip = 0.0
                self.hours_at_empty = 0.0
        else:
            self.hours_at_empty = 0.0
        if self.hours_since_strip > p.strip_interval_h:
            self.overdue_hours += dt_h

    @property
    def strip_due_in_h(self) -> float:
        return self.params.strip_interval_h - self.hours_since_strip


class ZBFBBank:
    """N independent modules. Dispatch uses the fewest modules that can meet the request, which keeps
    parasitic losses low at part load (the effect Roberts et al. found doubled efficiency)."""

    def __init__(self, n_modules: int, params: ZBFBParams | None = None, soc0: float = 0.5):
        params = params or ZBFBParams()
        self.modules = [ZBFBModule(params, energy_kwh=soc0 * params.e_max_kwh) for _ in range(n_modules)]
        # stagger strip clocks so strips do not all fall due at once
        for i, m in enumerate(self.modules):
            m.hours_since_strip = i * params.strip_interval_h / n_modules

    @property
    def energy_kwh(self) -> float:
        return sum(m.energy_kwh for m in self.modules)

    @property
    def e_max_kwh(self) -> float:
        return sum(m.params.e_max_kwh for m in self.modules)

    def step(self, p_request_kw: float, dt_h: float = 1.0) -> float:
        p = self.modules[0].params
        if p_request_kw == 0:
            return sum(m.step(0.0, dt_h=dt_h) for m in self.modules)
        charging = p_request_kw > 0
        p_mod_max = p.p_charge_max_kw if charging else p.p_discharge_max_kw
        # charge the emptiest first, discharge the fullest first
        order = sorted(self.modules, key=lambda m: m.energy_kwh, reverse=not charging)
        if charging:
            usable = [m for m in order if m.soc < 1 - 1e-9]
        else:
            usable = [m for m in order if m.energy_kwh > p.empty_tol_kwh]
        n_needed = min(len(usable), max(1, math.ceil(abs(p_request_kw) / p_mod_max - 1e-9)))
        chosen = usable[:n_needed]
        share = p_request_kw / n_needed if n_needed else 0.0
        delivered = 0.0
        for m in self.modules:
            delivered += m.step(share if m in chosen else 0.0, dt_h=dt_h)
        return delivered
