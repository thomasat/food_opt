# Example setup - plant-based burger (illustrative, no results)

How to set up a project: calculated rows (hydration water as 2.2 × the textured proteins, remaining water as "= rest"), pre-mixes, a process setting, records and a formulation total. Every value is illustrative, not measured, and the project has no results; it is the 0.7.x burger sample's set-up. Background reading: Peñaranda et al. (2025), Plant-Based Burgers with Reduced Texture Additives, https://doi.org/10.3390/foods14081373 (background only; it does not validate this example).

Rebuild the shipped project from `burger_ingredients.csv`: `python data/samples/burger_setup/build.py` (writes the `.json` here). `tests/test_sample_burger.py` checks it matches.
