"""The 0.8.0 usability demo projects as test fixtures (spec
2026-09-30-bo-views §7): Plant patty Lines A (clean panel), B (a new panel
from Round 4) and C (a noisy panel), and the 60-formulation Timing check,
copied from docs/usability/0.8.0-demo/projects (build_demo.py). Opened in a
fresh folder, never saved back."""
import json
import os
import pathlib
import tempfile

from food_bo import FoodOptimizer

HERE = pathlib.Path(__file__).parent / "fixtures" / "demo_projects"
LINE_A, LINE_B, LINE_C = (f"Plant patty - Line {x}" for x in "ABC")
TIMING = "Timing check - 60 formulations"
# The 0.7.2 upgrade file (docs/usability/0.8.0-demo/projects072,
# build_upgrade.py): 2 Round-ahead checks, so the A3 gate holds (polish
# round, read-through item 2).
UPGRADE = "Upgrade check - patty 0.7"
# Insights for a first-time user (spec 2026-10-01 rev. 2, review I17, R-I17):
# the Snack bar (3 scored, Round 1 only: every gate line), copied unchanged
# from docs/usability/0.8.0-demo/projects, and a study of process settings
# alone, built once by tests/process_only_fixture.py (Rounds 5, 4, 4, 4, 3
# with a Repeat of Best so far in Round 5; Round 6 open, 4 generated picks,
# with their generation records).
SNACK = "Snack bar - first round"
PROCESS_ONLY = "Oven trial - process settings"
# The six demo projects the first screen is pinned on, then the
# process-only one, in the order the review's tables list them.
SIX = (SNACK, LINE_A, LINE_B, LINE_C, UPGRADE, TIMING)
ALL = (*SIX, PROCESS_ONLY)


def load(name):
    """The demo project `name`, opened in a fresh temporary folder (the
    working folder is put back after, and the folder removed: T1 review
    M6), with save() switched off."""
    state = json.loads((HERE / f"{name}.json").read_text())
    here = os.getcwd()
    with tempfile.TemporaryDirectory(prefix="demo-") as folder:
        os.chdir(folder)
        try:
            opt = FoodOptimizer(state["project_name"])
            opt.save = lambda: None
            opt.import_json(state)
            opt.save = lambda: None
        finally:
            os.chdir(here)
    return opt
