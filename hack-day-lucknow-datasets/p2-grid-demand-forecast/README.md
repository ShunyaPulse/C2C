# P2 · Day-Ahead Grid Demand — public data (participant-facing)

Hourly operations log of a South-Asian national power grid (source not disclosed).

Files
- `history.csv` — 79,283 hourly rows, 2015-04-19 00:00 to 2024-06-16 23:00. Columns: timestamp, generation_mw, demand_mw (TARGET), load_shedding, gas, liquid_fuel, coal, hydro, solar, wind, india_bheramara_hvdc, india_tripura, india_adani, nepal (all MW).
- `sample_forecast_one_origin.csv` — format of one 24-hour forecast: `timestamp,demand_mw,p10,p90`.

Warnings: the log is raw — there are missing hours, typo-like spikes (e.g. 121000 MW), and import columns that start mid-way. Cleaning is part of the task.
