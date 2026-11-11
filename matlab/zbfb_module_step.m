function [m, p_out] = zbfb_module_step(m, p_request_kw, active, dt_h, p)
%ZBFB_MODULE_STEP Advance one ZBFB module by one time step.
%   [m, p_out] = zbfb_module_step(m, p_request_kw, active, dt_h, p)
%   m            struct with fields energy_kwh, hours_since_strip, hours_at_empty,
%                strips_completed, overdue_hours (see zbfb_module_init)
%   p_request_kw terminal power: > 0 charge, < 0 discharge, 0 rest
%   active       true keeps the pumps running at zero power; [] = active when power is nonzero
%   dt_h         time step, h
%   p            parameters from zbfb_params
%   p_out        terminal power actually delivered, kW
%
%   Mirrors src/battery/zbfb.py (ZBFBModule.step); keep the two in step.

if isempty(active)
    active = (p_request_kw ~= 0);
end
p_req = max(-p.p_discharge_max_kw, min(p.p_charge_max_kw, p_request_kw));
e0 = m.energy_kwh;
parasitic = p.p_parasitic_kw * active;
idle_loss = p.self_discharge_idle_kw * (~active);
k = p.k_loss;

if p_req > 0
    % charge: stored gain = P - k P^2 - parasitic
    gain = (p_req - k * p_req^2 - parasitic) * dt_h;
    room = p.e_max_kwh - e0;
    if gain > room
        net = max(0, room / dt_h + parasitic);
        p_req = (1 - sqrt(max(0, 1 - 4 * k * net))) / (2 * k);
        gain = room;
    end
    m.energy_kwh = e0 + gain;
    p_out = p_req;
elseif p_req < 0
    % discharge: stored drain = |P| + k P^2 + parasitic
    drain = (-p_req + k * p_req^2 + parasitic) * dt_h;
    if drain > e0
        avail = max(0, e0 / dt_h - parasitic);
        p_req = -(-1 + sqrt(1 + 4 * k * avail)) / (2 * k);
        drain = e0;
    end
    m.energy_kwh = e0 - drain;
    p_out = p_req;
else
    m.energy_kwh = max(0, e0 - (parasitic + idle_loss) * dt_h);
    p_out = 0;
end

% strip-cycle bookkeeping
m.hours_since_strip = m.hours_since_strip + dt_h;
if m.energy_kwh <= p.empty_tol_kwh
    m.energy_kwh = 0;
    m.hours_at_empty = m.hours_at_empty + dt_h;
    if m.hours_at_empty >= p.strip_hold_h
        m.strips_completed = m.strips_completed + 1;
        m.hours_since_strip = 0;
        m.hours_at_empty = 0;
    end
else
    m.hours_at_empty = 0;
end
if m.hours_since_strip > p.strip_interval_h
    m.overdue_hours = m.overdue_hours + dt_h;
end
end
