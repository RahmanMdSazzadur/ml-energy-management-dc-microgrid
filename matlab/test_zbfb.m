% TEST_ZBFB  Same checks as tests/test_zbfb.py. Run in MATLAB from the matlab/ folder:  test_zbfb
% Expected (from the Python model): round-trip 0.782 at rated current, 0.818 at half, 0.575 at 10 %;
% a full module with pumps on drains in about 7 days.
p = zbfb_params();
fprintf('module: %.2f kWh, charge %.2f kW, discharge %.2f kW\n', p.e_max_kwh, p.p_charge_max_kw, p.p_discharge_max_kw);

fracs = [1.0 0.5 0.2 0.1];
for f = fracs
    m = zbfb_module_init(0, 0); dt = 0.05; e_in = 0; e_out = 0;
    while m.energy_kwh / p.e_max_kwh < 1 - 1e-9
        [m, po] = zbfb_module_step(m, f * p.p_charge_max_kw, [], dt, p); e_in = e_in + po * dt;
    end
    while m.energy_kwh > 1e-9
        [m, po] = zbfb_module_step(m, -f * p.p_charge_max_kw, [], dt, p); e_out = e_out - po * dt;
    end
    fprintf('round-trip at %3.0f %% current: %.3f\n', 100 * f, e_out / e_in);
end

m = zbfb_module_init(1, 0); h = 0;
while m.energy_kwh > 0
    m = zbfb_module_step(m, 0, true, 1, p); h = h + 1;
end
fprintf('full module, pumps on: empty after %.1f days\n', h / 24);
assert(h >= 6.5 * 24 && h <= 7.5 * 24, 'drain time out of range');
