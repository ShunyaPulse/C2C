# P1 · Chulha to Cylinder — public data (participant-facing)

Real household survey records from India (data source intentionally not disclosed). One row = one household.

Files
- `train.csv` — 11,523 households, 395 feature columns + `household_id` + target `target_solid_fuel` (1 = household still uses a solid fuel such as firewood, dung cake, crop residue or coal for cooking; 0 = it does not).
- `data_dictionary.csv` — column name, dtype, % missing, number of unique values, example value, and whether survey codes 88/99/999/9999 ("don't know / not applicable") occur.
- `sample_submission.csv` — required output format: `household_id,prob_solid_fuel` (probability in [0,1]).

Notes
- Column names follow the questionnaire numbering (`q2xx` household & housing, `q3xx` electricity, `q4xx` appliances, `q6xx` purchase behaviour & metering). All cooking-section questions were removed.
- Codes 88/99 (and 999/9999) usually mean "don't know / refused / not applicable" — treat them deliberately.
- The hidden evaluation set contains households from districts that are NOT present in this file.
