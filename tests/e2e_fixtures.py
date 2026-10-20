"""The 2026-10-01 end-to-end campaign's own projects as frozen test
fixtures (spec 2026-10-01-e2e-fixes-design.md rev. 2; spec-review I12):
the food scientist's pea burger patty (02) and the DoE expert's snack bar
in three panels, clean, drift and noisy (03). They are the files the
campaign left behind, exported with FoodOptimizer.export_json, never
regenerated: a replay of ask() would not reproduce runs that changed
acquisition mid-way and made Repeats by hand. tests/fixtures/e2e/README.md
names each file's source.

    load(name)                       the project as saved
    load(name, last_round=6)         only Rounds 1–6 (truncate_state)
    load(name, recommended=True)     bo_config None (Recommended settings)

Every project is opened in a fresh temporary folder (the working folder is
put back after), with save() switched off, like tests/demo_projects.py.
"""
import copy
import json
import os
import pathlib
import tempfile

from food_bo import ROUND_FIELD, FoodOptimizer

HERE = pathlib.Path(__file__).parent / "fixtures" / "e2e"

# 02: 8 Rounds × 4, Redness added before Round 4, F8 Not scored, Round 3
# made at 500 g, Repeats F16 (of F11) and F32 (of F26). Recommended settings.
PATTY = "02_patty_final"
# 02 after Round 3, before Redness was added: 12 formulations, three
# measurements at 40/35/25 %. The app's own "Before editing" copy. The 02
# Round 3 pins (A2, A4) use this, not a truncation of PATTY: truncating keeps
# Redness and the shares rebalanced for it (34/29.75/21.25/15), so every
# Overall score of Rounds 1–3 differs from what the screen showed then.
PATTY_ROUND3 = "02_patty_round3"
# 02 with Round 7 open (4 suggestions, none recorded): the copy the app
# saved before the restore (B3's warning).
PATTY_ROUND7_OPEN = "02_patty_round7_open"
# 03: the snack bar. Clean: 8 Rounds × 4, Custom matern52 with qLogNEI R1–4,
# qLogEI R5–6, qUCB R7–8 (the file is left at qLogNEI). Drift: 8 Rounds × 4,
# Custom rbf, +5 Liking from Round 5; F20 is a Repeat of F14. Noisy: 7
# Rounds × 4, Recommended during the run, left at Custom rbf / short /
# fixed_tiny; load it with recommended=True for the A3 pins.
BAR_CLEAN = "03_bar_clean"
BAR_DRIFT = "03_bar_drift"
BAR_NOISY = "03_bar_noisy"

NAMES = (PATTY, PATTY_ROUND3, PATTY_ROUND7_OPEN, BAR_CLEAN, BAR_DRIFT, BAR_NOISY)


def state(name):
    """The fixture's saved state, a fresh dict each call."""
    return json.loads((HERE / f"{name}.json").read_text())


def truncate_state(saved, last_round):
    """`saved` with only the formulations of Rounds up to `last_round` (rows
    with no Round, made earlier, are kept), as if the project had stopped
    there with no Round open. Measurements, shares and limits are left as
    they are: a measurement added later stays (see PATTY_ROUND3). The
    Overall scores are recalculated by import_json."""
    out = copy.deepcopy(saved)
    rounds = out.get("batch_history") or []
    keep = [i for i, r in enumerate(rounds) if r is None or int(r) <= last_round]
    for key in ("X_history", "Y_history", "recipe_history", "results_history",
                "timestamps_history", "formulation_ids", "batch_history", "notes_history"):
        if key in out and out[key] is not None:
            out[key] = [out[key][i] for i in keep]
    ids = {int(n) for n in out.get("formulation_ids") or []}
    out["set_aside"] = [n for n in out.get("set_aside") or [] if int(n) in ids]
    # Formulations never scored, filed under the Round they were issued in.
    out["skipped"] = [s for s in out.get("skipped") or []
                      if s.get(ROUND_FIELD) is None or int(s[ROUND_FIELD]) <= last_round]
    ids |= {int(s["formulation"]) for s in out["skipped"]}
    for key in ("batch_totals", "batch_generated_totals", "round_generation",
                "bench_records"):
        out[key] = {k: v for k, v in (out.get(key) or {}).items() if int(k) <= last_round}
    for premix in (out.get("premixes") or {}).values():
        premix["versions"] = {k: v for k, v in (premix.get("versions") or {}).items()
                              if int(k) <= last_round}
    out["next_formulation_no"] = max(ids, default=0) + 1
    out["next_batch_number"] = last_round + 1
    out.update(pending_batch=None, pending_batch_no=None, pending_batch_created=None,
               pending_batch_total=None, pending_batch_discarded=[], result_drafts={})
    return out


def load(name, last_round=None, recommended=False):
    """The fixture `name` as a FoodOptimizer, opened in a fresh temporary
    folder with save() switched off. `last_round`: truncate_state first.
    `recommended`: bo_config None, the Recommended model settings."""
    saved = state(name)
    if last_round is not None:
        saved = truncate_state(saved, last_round)
    if recommended:
        saved["bo_config"] = None
    here = os.getcwd()
    with tempfile.TemporaryDirectory(prefix="e2e-") as folder:
        os.chdir(folder)
        try:
            opt = FoodOptimizer(saved["project_name"])
            opt.save = lambda: None
            opt.import_json(saved)
            opt.save = lambda: None
        finally:
            os.chdir(here)
    return opt
