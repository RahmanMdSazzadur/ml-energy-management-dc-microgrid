function m = zbfb_module_init(soc0, hours_since_strip0)
%ZBFB_MODULE_INIT Create one ZBFB module state.
%   m = zbfb_module_init(soc0, hours_since_strip0)
if nargin < 1, soc0 = 0.5; end
if nargin < 2, hours_since_strip0 = 0; end
p = zbfb_params();
m.energy_kwh = soc0 * p.e_max_kwh;
m.hours_since_strip = hours_since_strip0;
m.hours_at_empty = 0;
m.strips_completed = 0;
m.overdue_hours = 0;
end
