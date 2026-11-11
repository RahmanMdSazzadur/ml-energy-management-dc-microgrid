# ML-based energy management of a PV + zinc-bromine flow battery DC microgrid

Code, models and data pipeline for the paper *Machine Learning-Based Energy Management of a Solar PV System
Integrated with Zinc-Bromine Flow Battery Storage for DC Microgrid Applications*.

Author: Md Sazzadur Rahman, LUT University (Energy Technology)

## What this project does

A grid-connected DC microgrid on the LUT campus in Lappeenranta (61.06°N, 28.19°E) with rooftop PV and a
zinc-bromine flow battery (ZBFB). Machine learning forecasts PV and load for the next 24 hours; a model
predictive controller (MPC) plans battery power each hour, respecting ZBFB features that generic battery
models ignore: parasitic losses while a module is active, full 0–100 % depth of discharge and periodic strip
cycles. The controller runs inside HOMER Pro through its MATLAB Link and is compared with HOMER's Load
Following, Cycle Charging and Predictive (perfect foresight) dispatch.

## Folder layout

| Folder | Contents |
| --- | --- |
| `data/raw/` | Downloaded inputs as received (irradiance, temperature, load, spot prices). Not committed. |
| `data/processed/` | Cleaned hourly CSV files, one per input. |
| `notebooks/` | Exploration and plotting notebooks. |
| `src/data/` | Download and cleaning scripts. |
| `src/forecasting/` | Persistence, XGBoost and LSTM forecasting models. |
| `src/battery/` | Python version of the ZBFB model, for testing. |
| `matlab/homer_link/` | MATLAB Link dispatch scripts (MPC controller, ZBFB model). |
| `homer/` | HOMER Pro project files (`.homer`). |
| `models/` | Trained forecasting models. Not committed. |
| `results/` | Figures and tables generated for the paper. |
| `paper/` | Manuscript source and its figures. |
| `literature/` | `refs.bib` with the Phase 0 literature. |

## Tools

- HOMER Pro x64 3.18.4 with MATLAB Link, and MATLAB R2024b (LUT 2D Desktop, desktop.lut.fi)
- Python 3.11+, see `requirements.txt`

## Setup

```bash
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python src/data/clean.py              # raw -> data/processed
python src/data/make_homer_inputs.py 2023   # data/processed -> data/homer
```

## Tests

```bash
python -m pytest -q          # ZBFB model checks against literature values
```
In MATLAB, from `matlab/`: `test_zbfb` (should print the same efficiencies as the Python tests).

## Status

- [x] Phase 0: literature review and repository set-up
- [x] Phase 1: data collection (stand-in load from Fingrid; swap in measured LUT data if available)
- [x] Phase 2: ZBFB model; HOMER Pro base model and optimisation (2023); simulator validated against HOMER
- [x] Phase 3: day-ahead ML forecasting (price, load, PV)
- [x] Phase 4: MILP model-predictive controller with ZBFB physics, compared with HOMER and rule-based control
- [x] Phase 5: sensitivity analysis (battery size and cost, PV size, 2019 and 2022 price years)
- [ ] Phase 6: writing and submission

## How to reproduce the results

```bash
python -m src.sim.validate_homer          # simulator vs HOMER Pro (dispatch replay)
python -m src.forecasting.day_ahead       # train on 2019-2022, test on 2023
python -m src.forecasting.plot_forecasts
python -m src.mpc.ems --modules 1         # all strategies, 1 ZBFB module, PV 20 kW
python -m src.mpc.ems --modules 4 --strategies none,rule,mpc_pf,mpc_fc
python -m src.mpc.plot_results N1_PV20
python -m src.forecasting.day_ahead --year 2022   # leave-one-year-out forecasts (also 2019)
python -m src.mpc.ems --modules 1 --year 2022 --strategies none,rule,mpc_pf,mpc_fc
python -m src.analysis.sensitivity        # all scenarios, economics, sensitivity figure
```

## Method notes

* **HOMER Pro** (v3.18.4) is used for system sizing and as a baseline controller. Its storage model
  cannot represent ZBFB pump losses or the 72 h strip cycle, so control strategies are compared in our
  own simulator. Feeding HOMER's hourly battery decisions through our models reproduces HOMER's grid
  purchases, sales, battery energy and cost to rounding (results/validation/).
* HOMER Predictive does not support a grid connection, so it is not used as a baseline; the
  perfect-foresight MPC takes that role. The MPC is implemented in Python (SciPy/HiGHS MILP) instead of
  through the HOMER MATLAB Link, because the ZBFB physics live in our simulator, not in HOMER.
* **Forecasts**: pooled LightGBM, 1-24 h ahead, issued hourly; no numerical weather forecasts.
  The MPC uses published Nord Pool day-ahead prices for hours already published (13:00 CET rule).
* **MPC**: 24 h horizon, hourly receding horizon; per module two-segment resistive-loss model,
  pump on/off binaries, and binaries that choose when each 2 h strip at 0 % happens before it is due.
  Banks of several modules use a 1 % MIP gap and a 2 s time limit per solve.

