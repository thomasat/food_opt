"""The checks test_e2e.sh runs inside the packaged app (tests 5-5g), in one
file any Python can run: the Mac's bundled venv in test_e2e.sh, the Windows
build's bundled Python, or the dev Python against the source tree.

    python packaged_app_checks.py --resources DIR --projects DIR [--only NAME]...

DIR for --resources holds app.py and the modules it imports (the bundle's
Resources folder, or the repo). --projects is the folder the app keeps its
projects in; it is created and becomes the working directory. Each check
prints its sentinel(s) when it passes; a failed check prints "FAILED <name>"
and its traceback, and the run goes on to the next one.

Exit: 0 every selected check passed, 1 a check failed, 2 bad arguments.
"""
import argparse
import os
import sys
import traceback

sys.dont_write_bytecode = True  # nothing is written into the bundle


def check_smoke(app_path):
    from food_bo import FoodOptimizer
    opt = FoodOptimizer("E2E_Smoke")
    opt.add_ingredient("flour", 10.0, 100.0)
    opt.add_ingredient("sugar", 0.0, 50.0)
    opt.add_objective("taste", 1.0, goal="max")
    batch = opt.ask(n_suggestions=1)
    opt.tell(batch[0], {"taste": 7.5})
    # Full process continues: a parameter added mid-run needs a baseline (clear
    # error without one), history re-encodes, and everything survives a reload.
    try:
        opt.add_process_parameter("oven_temp", 150.0, 220.0)
        raise SystemExit("expected ValueError without baseline")
    except ValueError:
        pass
    opt.add_process_parameter("oven_temp", 150.0, 220.0, baseline=180.0)
    batch2 = opt.ask(n_suggestions=1)
    assert 150.0 <= batch2[0]["oven_temp"] <= 220.0
    opt.tell(batch2[0], {"taste": 8.0})
    opt2 = FoodOptimizer("E2E_Smoke")
    assert opt2.load_error is None and len(opt2.X_history) == 2
    assert all(len(x) == 3 for x in opt2.X_history)
    assert opt2.X_history[0][2] == 180.0  # first formulation encoded at baseline
    print("SMOKE_OK")


def check_ui_errors(app_path):
    import os
    import wording
    from streamlit.testing.v1 import AppTest
    from food_bo import FoodOptimizer

    opt = FoodOptimizer("UI_Check")
    opt.add_ingredient("water", 0.0, 100.0)
    opt.add_objective("taste", 1.0, goal="max")
    opt.tell({"water": 50.0}, {"taste": 7.0})

    at = AppTest.from_file(
        os.path.join(os.path.dirname(app_path), "app.py"), default_timeout=300)
    at.session_state["_loaded_project"] = "UI_Check"
    at.run()
    assert not at.exception, at.exception
    # One grid holds both types: Type is a cell, and a setting added mid-run
    # asks for the baseline the formulations already made were run at. Typing
    # into a data editor is injecting its own record of what was typed, which is
    # what the browser sends; it has to be injected before every run.
    EDITS = {"edited_rows": {}, "deleted_rows": [], "added_rows": [{
        wording.NAME_LABEL: "oven_temp", wording.TYPE_LABEL: wording.KIND_SETTING,
        wording.LOWEST_LABEL: 150.0, wording.HIGHEST_LABEL: 220.0,
        wording.UNIT_LABEL: "C",
        wording.BASELINE_LABEL: 100.0}]}          # outside [150, 220]
    at.session_state["ingredient_grid_0"] = dict(EDITS)
    at.run()
    next(b for b in at.button
         if b.key in ("save_ingredient_grid__save",
                      "save_ingredient_grid__btn")).click()
    at.session_state["ingredient_grid_0"] = dict(EDITS)
    at.run()
    assert not at.exception, at.exception   # a traceback here is the bug
    assert any("must be between" in str(e.value) for e in at.error), \
        [str(e.value) for e in at.error]
    print("UI_OK")


def check_wording_wave(app_path):
    import os
    import wording
    from streamlit.testing.v1 import AppTest
    from food_bo import FoodOptimizer

    opt = FoodOptimizer("Wave_Check")
    opt.add_ingredient("water", 0.0, 100.0)
    opt.add_objective("taste", 1.0, goal="max")
    opt.tell({"water": 50.0}, {"taste": 7.0})

    at = AppTest.from_file(
        os.path.join(os.path.dirname(app_path), "app.py"), default_timeout=300)
    at.session_state["_loaded_project"] = "Wave_Check"
    at.run()
    assert not at.exception, at.exception

    # Tab 1 - Set up: the measurements grid carries Share of score as the
    # column that is typed into, and the score line spells out the share each
    # measurement gets. There is no Importance column anywhere (0.5.0).
    grid = at.dataframe[1].value
    assert [c for c in grid.columns if c != "_id"] == [
        "Measurement", "Goal", "Target", wording.LOWEST_MEASURABLE_LABEL,
        wording.HIGHEST_MEASURABLE_LABEL, "Unit",
        wording.SHARE_COLUMN], list(grid.columns)
    assert list(grid["Share of score (%)"]) == [100.0]
    # The score line is the shares and nothing else now: the weight behind a
    # share left the screen with Importance, and the ceiling is always 100.
    assert any(c.value.startswith("Overall score = 100 % \u00d7 taste closeness")
               and "scores 100." in c.value
               for c in at.caption), [c.value for c in at.caption]

    # Tab 2 - Make a round: the retired Repeat checkbox never comes back, and
    # a formulation of your own has an expander to land in instead.
    at.session_state["main_tab"] = wording.TAB_BATCH
    at.run()
    assert not at.exception, at.exception
    expander_labels = [e.label for e in at.expander]
    assert wording.ADD_OWN_EXPANDER in expander_labels, expander_labels
    assert not any(c.label and c.label.startswith("Repeat") for c in at.checkbox), \
        [c.label for c in at.checkbox]

    # Tab 3 - Results: one "Correct, delete or add past formulations" fold
    # replaces the three retired controls; none of their old labels survive
    # anywhere on the tab.
    at.session_state["main_tab"] = wording.TAB_RESULTS
    at.run()
    assert not at.exception, at.exception
    expander_labels = [e.label for e in at.expander]
    assert wording.EDIT_PAST_FOLD in expander_labels, expander_labels
    retired = ("Correct a result", "Delete a batch or a formulation",
               "Delete the last batch", "Import past formulations from a CSV")
    seen = (expander_labels + [b.label for b in at.button]
            + [c.label for c in at.checkbox if c.label])
    assert not any(label in seen for label in retired), seen
    print("WAVE_OK")


def check_openpyxl(app_path):
    import openpyxl  # noqa: F401  (the workbook download needs it)
    print("OPENPYXL_OK")


def check_kitchen_trust(app_path):
    import os
    import wording
    from streamlit.testing.v1 import AppTest
    from food_bo import FoodOptimizer

    opt = FoodOptimizer("Kitchen_Check")
    opt.add_ingredient("water", 0.0, 100.0)
    opt.add_ingredient("flour", 0.0, 100.0)
    opt.add_objective("taste", 1.0, goal="max")
    opt.set_formulation_total(50.0)
    # Five scored rows: the Best-so-far column is drawn only once the cold
    # start is over (the first five formulations are spread out, not compared).
    for no, water in enumerate((20.0, 22.0, 24.0, 26.0, 28.0), start=1):
        opt.tell({"water": water, "flour": 50.0 - water}, {"taste": 5.0 + no * 0.4},
                 formulation_no=no, batch_no=1)
    opt.set_pending_batch([{"water": 25.0, "flour": 25.0}])


    def _unknown(node, kind, label):
        """AppTest has no accessor for st.download_button or st.file_uploader:
    both arrive as UnknownElement carrying the raw proto."""
        def walk(n):
            children = getattr(n, "children", None) or {}
            if hasattr(children, "values"):
                children = children.values()
            for child in children:
                if (type(child).__name__ == "UnknownElement"
                        and getattr(child, "type", None) == kind
                        and getattr(child.proto, "label", None) == label):
                    return child
                found = walk(child)
                if found is not None:
                    return found
            return None
        return walk(node)


    at = AppTest.from_file(
        os.path.join(os.path.dirname(app_path), "app.py"), default_timeout=300)
    at.session_state["_loaded_project"] = "Kitchen_Check"
    at.run()
    assert not at.exception, at.exception

    # Sidebar: Saved copies in plain words, not the retired backup language.
    assert any(m.value == wording.SAVED_COPIES_HEADING for m in at.sidebar.markdown), \
        [m.value for m in at.sidebar.markdown]
    assert _unknown(at.sidebar, "download_button", wording.SAVE_A_COPY) is not None
    assert _unknown(at.sidebar, "file_uploader", wording.OPEN_A_SAVED_COPY) is not None

    # Tab 1 - Set up: the default batch size box, inside More settings.
    assert wording.formulation_total_label("g") in [n.label for n in at.number_input], \
        [n.label for n in at.number_input]

    # Tab 2 - Make a round: one workbook download, and the round table (v3)
    # compares each formulation with the best so far, column by column.
    at.session_state["main_tab"] = wording.TAB_BATCH
    at.run()
    assert not at.exception, at.exception
    assert _unknown(at.main, "download_button",
                    wording.DOWNLOAD_BATCH_SHEETS) is not None
    best_cols = [str(c) for d in at.dataframe for c in d.value.columns
                 if str(c).startswith(wording.BEST_SO_FAR_COLUMN)]
    assert best_cols, [list(d.value.columns) for d in at.dataframe]
    print("KITCHEN_TRUST_OK")


def check_spreadsheet_feel(app_path):
    import io
    import os
    import wording
    from openpyxl import load_workbook
    from streamlit.testing.v1 import AppTest
    from food_bo import FoodOptimizer

    opt = FoodOptimizer("Grid_Check")
    opt.add_ingredient("water", 0.0, 100.0)
    opt.add_ingredient("flour", 0.0, 100.0)
    opt.add_objective("taste", 1.0, goal="max")
    opt.set_formulation_total(100.0)
    opt.set_pending_batch([{"water": 30.0, "flour": 70.0}])

    at = AppTest.from_file(
        os.path.join(os.path.dirname(app_path), "app.py"), default_timeout=300)
    at.session_state["_loaded_project"] = "Grid_Check"
    at.run()
    assert not at.exception, at.exception

    # Tab 1 - Set up: the ingredients list is one editable grid, with Vendor
    # and SKU beside the range and Calculation last (0.6.0). Baseline joins it
    # only once results exist. The headers come from wording, so this check
    # can never fall behind a rename the way it did in 0.7.1.
    assert [c for c in at.dataframe[0].value.columns if c != "_id"] == [
        wording.NAME_LABEL, wording.TYPE_LABEL, wording.MADE_AS_LABEL,
        wording.LOWEST_LABEL, wording.HIGHEST_LABEL, "Unit",
        wording.FORMULA_LABEL], list(at.dataframe[0].value.columns)

    # Three tiers on the tab: the grids, then More settings, then Advanced.
    labels = [e.label for e in at.expander]
    assert wording.MORE_SETTINGS_EXPANDER in labels, labels
    assert wording.ADVANCED_EXPANDER in labels, labels

    # Nothing is written while typing: an edit in hand lights Save changes and
    # puts Discard changes beside it. Typing into a data editor is injecting
    # its own record of what was typed, which is what the browser sends.
    at.session_state["ingredient_grid_0"] = {
        "edited_rows": {0: {wording.HIGHEST_LABEL: 80.0}},
        "deleted_rows": [], "added_rows": []}
    at.run()
    assert not at.exception, at.exception
    # Either key: a save that would take the open round away is drawn by
    # confirm_action (__btn) so it can ask first, and a save that would not is
    # the plain button (__save). Both say Save changes and both are the one lit
    # action on the tab.
    save = next(b for b in at.button
                if b.key in ("save_ingredient_grid__save",
                             "save_ingredient_grid__btn"))
    assert save.label == wording.SAVE_CHANGES_BUTTON, save.label
    assert save.proto.type == "primary", save.proto.type
    assert any(b.key == "save_ingredient_grid__discard"
               and b.label == wording.DISCARD_CHANGES_BUTTON for b in at.button), \
        [(b.key, b.label) for b in at.button]

    # Tab 2 - Make a round: the round screen's own Batch size box, and the
    # round named by number.
    assert wording.batch_size_label("g") in [n.label for n in at.number_input], \
        [n.label for n in at.number_input]
    assert wording.make_these(1, 1) in [m.value for m in at.main.markdown], \
        [m.value for m in at.main.markdown]

    # The workbook is protected, with a Lot cell per ingredient on the round's
    # summary page and an Actual (g) column on each formulation page.
    opt.set_records("lot", True)
    opt.set_records("actual", True)
    book = load_workbook(io.BytesIO(opt.workbook_bytes(opt.pending_batch, 100.0)))
    summary, page = book[book.sheetnames[0]], book[book.sheetnames[1]]


    def filled(sheet):
        return [str(c.value) for row in sheet.iter_rows()
                for c in row if c.value is not None]


    assert summary.protection.sheet and page.protection.sheet
    assert wording.LOT_COLUMN in filled(summary), filled(summary)
    assert opt._actual_column_head() in filled(page), filled(page)
    print("GRID_OK")


def check_rules_premix_sample_draft(app_path):
    import io
    import os
    import tempfile
    import pandas as pd
    import custom_records
    import sample_projects
    import wording
    from food_bo import FoodOptimizer
    from openpyxl import load_workbook
    from streamlit.testing.v1 import AppTest

    APP = os.path.join(os.path.dirname(app_path), "app.py")
    DATA = os.getcwd()

    # A project with what the rules and pre-mix checks need, built here: the
    # yogurt samples have no pre-mix, no calculated row and no "= rest" row.
    # Two calculated water rows, a fixed-ratio pre-mix with a 150 g floor, a
    # fixed-amount one, a weighed blend, a setting on a 5 s dial, a solids
    # limit, a default batch size, bench fields and a method.
    CSV = """Name,Part of,Preparation,Lowest,Highest,Composition (%),Calculation,Unit,Fat per 100 g
Textured pea protein,,,4,6,,,g,6.0
Textured soy protein,,,4,6,,,g,6.0
Hydration water,,,0,100,,= 2.2 * (Textured pea protein + Textured soy protein),g,0.0
Remaining water,,,0,100,,= rest,g,0.0
Dry blend,,Pre-mix: keep proportions fixed,15,18,,,g,
Pea protein isolate,Dry blend,,,,66,,g,8.0
Potato starch,Dry blend,,,,26,,g,0.1
Methylcellulose,Dry blend,,,,8,,g,0.0
Wheat gluten,,,3.5,6,,,g,1.9
Seasoning blend,,Pre-mix: keep proportions fixed,2.2,2.2,,,g,
Salt,Seasoning blend,,,,60,,g,0.0
Onion powder,Seasoning blend,,,,40,,g,1.0
Fats and oils,,Blend: vary each ingredient,,,,,g,
Coconut oil,Fats and oils,,6,10,,,g,99.0
Sunflower oil,Fats and oils,,2,5,,,g,100.0
"""
    NAMES = ["Textured pea protein", "Textured soy protein", "Hydration water",
             "Remaining water", "Dry blend", "Wheat gluten", "Seasoning blend",
             "Fats and oils", "Mixing time after fat"]
    built = FoodOptimizer("Premix_Check")
    built.set_amount_unit("g")
    built.load_ingredients_from_csv(pd.read_csv(io.StringIO(CSV)))
    built.add_process_parameter("Mixing time after fat", 45, 150, unit="s", step=5)
    built.add_objective("Firmness", 1, goal="target", target=6, min_val=0,
                        max_val=10, unit="/10")
    built.add_objective("Cook loss", 1, goal="min", min_val=0, max_val=40, unit="%")
    built.set_formulation_total(100)
    built.add_constraint("Fat per 100 g", max_val=16)
    built.add_chosen_quantity_constraint(
        ["Textured pea protein", "Textured soy protein", "Dry blend",
         "Wheat gluten", "Coconut oil", "Sunflower oil"], min_val=36, max_val=43)
    built.set_premix_smallest_quantity("Dry blend", 150)
    built.set_records("lot", True)
    built.set_records("actual", True)
    for field in ["Raw weight (g)", "Cooked weight (g)"]:
        custom_records.add_field(built, field, "formulation")
    built.set_method(
        "1. Hydrate the proteins.\n2. Mix the dry blend in cold.\n"
        "3. Griddle at 180 °C, 3 min per side, to 74 °C core.\n"
        "Contains soy and wheat (gluten).")

    at = AppTest.from_file(APP, default_timeout=300)
    at.session_state["_loaded_project"] = "Premix_Check"
    at.run()
    assert not at.exception, at.exception

    # Tab 1 - Set up: Rule is the last column on the ingredients grid.
    grid = at.dataframe[0].value
    columns = [c for c in grid.columns if c != "_id"]
    assert columns[-1] == wording.FORMULA_LABEL, columns

    # Remaining water takes the rest of the batch size, so its amount is not
    # typed: the Calculation cell says Fill to total, the two range cells are
    # blank beside it, and one caption under the grid says what it fills to
    # and what that comes to in numbers.
    water = grid[grid[wording.NAME_LABEL] == "Remaining water"]
    assert len(water) == 1, list(grid[wording.NAME_LABEL])
    assert water.iloc[0][wording.FORMULA_LABEL] == wording.FILL_TO_TOTAL, \
        water.iloc[0][wording.FORMULA_LABEL]
    assert water.iloc[0][wording.LOWEST_LABEL] == "", \
        water.iloc[0][wording.LOWEST_LABEL]
    assert water.iloc[0][wording.HIGHEST_LABEL] == "", \
        water.iloc[0][wording.HIGHEST_LABEL]
    captions = [c.value for c in at.caption]
    assert any(c.startswith("Remaining water fills to the 100 g ")
               for c in captions), captions
    # RULE_HINT is gone (audit V19): the one calculation cue left on the tab
    # is the Edit calculation line under the grid.
    assert wording.CALCULATION_EDIT_HINT in captions, captions

    # More settings - Limits: one Kind picker choosing the shape of the limit,
    # and because the project has a default batch size the "Write it as"
    # choice offers % of default batch size too.
    kind_select = next((s for s in at.selectbox if s.key == "qc_kind"), None)
    assert kind_select is not None, [s.key for s in at.selectbox]
    assert list(kind_select.options) == wording.LIMIT_KINDS, kind_select.options
    unit_select = next((s for s in at.selectbox if s.key == "qc_unit"), None)
    assert unit_select is not None, [s.key for s in at.selectbox]
    assert unit_select.label == wording.LIMIT_WRITTEN_AS_LABEL, unit_select.label
    assert wording.PERCENT_OF_BATCH_SIZE_UNIT in unit_select.options, \
        unit_select.options
    print("RULES_OK")

    # Both pre-mix modes and the preparation pages are in the shipped app.
    opt = at.session_state["optimizer"]
    assert list(grid[wording.NAME_LABEL]) == NAMES, list(grid[wording.NAME_LABEL])
    assert wording.MADE_AS_LABEL in columns
    assert opt.premixes["Fats and oils"]["mode"] == "weighed"
    assert len(opt.premixes["Dry blend"]["parts"]) == 3
    assert opt._by_name()["Seasoning blend"]["bounds"] == (2.2, 2.2)
    opt.ask(3)
    book = load_workbook(io.BytesIO(opt.workbook_bytes(opt.pending_batch, 100)))
    assert book.sheetnames[:2] == ["Pre-mix · Dry blend", "Pre-mix · Seasoning blend"]
    assert book.active["A1"].value.startswith("Dry blend · Prepare 150 g")
    assert "this Round needs" in book.active["A1"].value
    assert book.active.protection.sheet and not book.active["D4"].protection.locked
    summary = book[wording.batch_sheet_name(opt.pending_batch_no)]
    assert wording.MAKE_FOR_ROUND_HEADING in [c.value for row in summary for c in row]
    # The method the bench works from, with the cook in it, and the allergens.
    method = "\n".join(str(c.value) for row in summary for c in row if c.value)
    assert "180 °C" in method and "74 °C core" in method, method[:400]
    assert "Contains soy and wheat (gluten)." in method
    assert wording.SETTINGS_SHEET_HEADING in [c.value for row in summary for c in row]
    # The solids limit the project writes, and the design space it holds.
    limit = next(qc for qc in opt.quantity_constraints if qc.get("source") is None)
    assert (limit["min"], limit["max"]) == (36.0, 43.0), limit
    # A setting is issued on the step its dial has.
    for row in opt.pending_batch:
        assert row["recipe"]["Mixing time after fat"] % 5 == 0, row["recipe"]
    page = book[wording.formulation_sheet_name(opt.pending_batch[0]["formulation"])]
    assert any(c.value == "Coconut oil" and c.alignment.indent == 1 for row in page for c in row)
    compact = load_workbook(io.BytesIO(opt.workbook_bytes(opt.pending_batch, 100, print_pack=False)))
    # The three the reader sees. The metadata sheet is hidden, and openpyxl
    # lists a hidden sheet like any other.
    assert [s.title for s in compact if s.sheet_state == "visible"] == [
        "Round overview", "Preparation", "Results"], compact.sheetnames
    assert compact.active.title == "Round overview"
    assert any(str(c.value).startswith("Cook loss (%)") for row in compact["Results"] for c in row)
    # ...and on the print pack too, which is where it was missing.
    assert any(str(c.value).startswith("Cook loss (%)") for row in summary for c in row)
    print("PREMIX_OK")

    # The welcome Try the sample project (0.8.0): in a folder with no projects
    # the welcome panel offers it; one click writes both yogurt samples and
    # opens Round 6 to score on 4 · Review, with Round 6 (Formulations 16-18)
    # open and Rounds 1-5 scored.
    fresh_dir = tempfile.mkdtemp(prefix="sample_", dir=os.path.expanduser("~"))
    os.chdir(fresh_dir)
    at = AppTest.from_file(APP, default_timeout=300)
    at.run()
    assert not at.exception, at.exception
    assert "optimizer" not in at.session_state or at.session_state["optimizer"] is None
    assert wording.WELCOME_HEADER in [m.value for m in at.main.markdown]
    next(b for b in at.main.button if b.label == wording.OPEN_YOGURT_LABEL).click()
    at.run()
    assert not at.exception, at.exception
    for name in (sample_projects.ROUND6, sample_projects.ALL10):
        assert os.path.isfile(name + ".pkl"), (name, sorted(os.listdir(".")))
    sample = at.session_state["optimizer"]
    assert sample.project_name == sample_projects.ROUND6, sample.project_name
    assert at.session_state["main_tab"] == wording.TAB_INSIGHTS, \
        at.session_state["main_tab"]
    assert [r["formulation"] for r in sample.pending_batch] == [16, 17, 18], \
        sample.pending_batch
    assert len(sample.X_history) == 15, len(sample.X_history)
    print("SAMPLE_OK")

    # Committed manual fields survive a fresh application session: a score
    # typed for the first open formulation of the sample.
    number = sample.pending_batch[0]["formulation"]
    at.number_input(key=f"f{number}_Consistency").set_value(11).run()
    assert not at.exception, at.exception
    fresh = AppTest.from_file(APP, default_timeout=300)
    fresh.session_state["_loaded_project"] = sample.project_name
    fresh.run()
    assert not fresh.exception, fresh.exception
    assert fresh.number_input(key=f"f{number}_Consistency").value == 11
    assert not next(b for b in fresh.button if b.label == wording.SAVE_RESULTS).disabled
    os.chdir(DATA)
    print("DRAFT_OK")


# Run order is insertion order. Names may only be added (a pytest pins them).
CHECKS = {
    "smoke": check_smoke,                                          # test 5
    "ui_errors": check_ui_errors,                                  # test 5b
    "wording_wave": check_wording_wave,                            # test 5c
    "openpyxl": check_openpyxl,                                    # test 5d
    "kitchen_trust": check_kitchen_trust,                          # test 5e
    "spreadsheet_feel": check_spreadsheet_feel,                    # test 5f
    "rules_premix_sample_draft": check_rules_premix_sample_draft,  # test 5g
}

SENTINELS = {
    "smoke": ("SMOKE_OK",),
    "ui_errors": ("UI_OK",),
    "wording_wave": ("WAVE_OK",),
    "openpyxl": ("OPENPYXL_OK",),
    "kitchen_trust": ("KITCHEN_TRUST_OK",),
    "spreadsheet_feel": ("GRID_OK",),
    "rules_premix_sample_draft": ("RULES_OK", "PREMIX_OK", "SAMPLE_OK", "DRAFT_OK"),
}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--resources", required=True,
                        help="folder holding app.py and its modules")
    parser.add_argument("--projects", required=True,
                        help="projects folder; created, and the working directory")
    parser.add_argument("--only", action="append", default=[], metavar="NAME",
                        help="run only this check (repeatable)")
    try:
        args = parser.parse_args(argv)
    except SystemExit as exc:  # argparse exits 2 on bad arguments, 0 on --help
        return exc.code if isinstance(exc.code, int) else 2
    unknown = [name for name in args.only if name not in CHECKS]
    if unknown:
        print(f"unknown check(s): {', '.join(unknown)}; "
              f"known: {', '.join(CHECKS)}", file=sys.stderr)
        return 2
    selected = [name for name in CHECKS if not args.only or name in args.only]

    resources = os.path.abspath(args.resources)
    # The bundle's modules come first, where PYTHONPATH put them, in place of
    # this script's own folder.
    sys.path[0] = resources
    os.makedirs(args.projects, exist_ok=True)
    os.chdir(args.projects)
    app_path = os.path.join(resources, "app.py")

    failed = 0
    for name in selected:
        try:
            CHECKS[name](app_path)
        except (Exception, SystemExit):
            failed += 1
            print(f"FAILED {name}")
            traceback.print_exc(file=sys.stdout)
        sys.stdout.flush()
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
