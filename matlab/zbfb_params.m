function p = zbfb_params()
%ZBFB_PARAMS Parameters of one zinc-bromine flow battery module (Redflow ZBM2).
%   Values from Roberts et al. (2024), J. Energy Storage 81, 110280,
%   doi:10.1016/j.est.2023.110280. Same values as src/battery/zbfb.py.

p.v_nom           = 52.0;   % V, stack voltage
p.capacity_ah     = 220.0;  % Ah
p.r_int           = 0.1;    % ohm, internal resistance
p.i_charge_max    = 50.0;   % A
p.i_discharge_max = 90.0;   % A
p.i_parasitic     = 1.3;    % A drawn while the module is active (pumps, fan)
p.self_discharge_idle_kw = 0.0;  % kW lost while idle, pumps off
p.strip_interval_h = 72.0;  % h of clock time between strips
p.strip_hold_h     = 2.0;   % h held at 0 % to complete a strip
p.empty_tol_kwh    = 1e-6;

% derived
p.e_max_kwh          = p.capacity_ah * p.v_nom / 1000;      % 11.44 kWh
p.p_charge_max_kw    = p.i_charge_max * p.v_nom / 1000;     % 2.60 kW
p.p_discharge_max_kw = p.i_discharge_max * p.v_nom / 1000;  % 4.68 kW
p.p_parasitic_kw     = p.i_parasitic * p.v_nom / 1000;      % 0.0676 kW
p.k_loss             = p.r_int * 1000 / p.v_nom^2;          % loss = k * P^2, kW
end
