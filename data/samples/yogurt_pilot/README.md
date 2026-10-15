# Yogurt-style pilot (public sample data)

A real plant-based yogurt-style pilot (Feb–May 2026): 10 Rounds of 3 batches, a fixed budget. Rounds 1–2 were a random starting set drawn by the engine; Rounds 3–10 were chosen by EGBO, an earlier version of this engine.

- **Limits**: the pilot's own, per 100 g: each of the 7 ingredients' Lowest and Highest, and their total at 97.69051–98.69051 g. Soy protein isolate (1.00 %) and tricalcium phosphate (0.31 %) were kept constant in every batch, so they are not in the data.
- **Results**: each rating is the median of a trained panel of five, 0–15, one per attribute per batch; no batch was repeated.
- **Process**: mixed, cooked at 90 °C for 8 min, chilled. No culture or fermentation: the paper's method is mix, cook, chill, with no culture among its ingredients.

Rebuild the two shipped projects from the CSVs: `python data/samples/yogurt_pilot/build.py` (writes the two `.json` files here). `tests/test_sample_projects.py` checks they match.
