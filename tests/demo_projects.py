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
