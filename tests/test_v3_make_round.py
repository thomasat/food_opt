"""v3 · Make a round (final-v3/2-make-round.png, wording.md §2): the Round
as one table — the ingredients down, Best so far and each formulation
across, each cell the amount with its change from Best so far, grey shading
by that change as a share of the row's allowed range, limit tags and a
Total row — and no forecast anywhere on the screen."""
import pytest
from streamlit.testing.v1 import AppTest

import bo_views
import sample_projects as sp
import storage as storage_backend
import ui_batch
import wording
from food_bo import FoodOptimizer, uploaded_parts
from tests.test_app_ui import APP_PATH
from tests.test_insights import _burger


@pytest.fixture(autouse=True)
def _samples(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    sp.ensure(storage_backend.LocalStorage())
    with bo_views.fresh():
        yield


def _round6():
    return FoodOptimizer(sp.ROUND6, storage=storage_backend.LocalStorage())


def _cells(table):
    """{column: {row: CompareCell}}."""
    return {c: dict(zip(table.rows, col)) for c, col in zip(table.columns, table.cells)}


def _record_round6(opt):
    """Upload the pilot's Round 6 results file, as the screen does."""
    with open(sp.ROUND6_RESULTS, "rb") as fh:
        upload = opt.results_from_workbook(fh)
    sheet, weighed, lots = uploaded_parts(upload)
    parsed, _ = opt.parse_batch_results(sheet, opt.pending_batch, with_skipped=True,
                                        weighed=weighed)
    batch_no = opt.pending_batch_no
    by_number = {r["formulation"]: r["recipe"] for r in opt.pending_batch}
    for number, results, note in parsed:
        opt.tell(opt.amounts_as_weighed(by_number[number], weighed.get(number)),
                 results, formulation_no=number, batch_no=batch_no, note=note)
    opt.set_pending_batch(None)
    return opt


# ------------------------------------------------------------------ #
#  The table, worked out
# ------------------------------------------------------------------ #

def test_round6_table_values_signs_tags_and_total():
    table = ui_batch.round_compare(_round6())
    assert table.columns == ["Best so far (F10 · 95.0)", "F16", "F17", "F18"]
    assert table.rows[:7] == ["Coconut cream", "Tap water", "Lemon juice", "Salt",
                              "Nutritional yeast", "Tapioca starch", "Xanthan gum"]
    assert table.rows[-1] == "Total (allowed 97.69–98.69 g)"
    assert table.row_header == "Ingredient (g)"
    assert table.best_no == 10 and table.tied == [] and table.numbers == []
    c = _cells(table)
    best = c["Best so far (F10 · 95.0)"]
    assert best["Coconut cream"].text == "51.84" and best["Tap water"].text == "41.22"
    assert c["F16"]["Coconut cream"].text == "51.27 (−0.57)"
    assert c["F18"]["Coconut cream"].text == "52.27 (+0.43)"
    assert c["F17"]["Coconut cream"].text == "50.24 (−1.60)"
    # At the upper limit: the tag on a line of its own.
    assert c["F16"]["Tap water"].text == "42.00 (+0.78)\nat upper limit"
    assert c["F16"]["Tap water"].tag == wording.AT_UPPER_LIMIT
    assert all(c[f]["Tapioca starch"].tag == wording.AT_UPPER_LIMIT for f in ("F16", "F17", "F18"))
    assert c["F18"]["Xanthan gum"].text == "0.65 (+0.07)\nat upper limit"
    assert c["F18"]["Tap water"].tag == ""
    # Under 1 % of the allowed range (0.5 g for xanthan): no change written,
    # no shading.
    assert c["F17"]["Xanthan gum"].text == "0.58"
    assert c["F17"]["Xanthan gum"].share == 0
    # The change is the difference of the two numbers shown.
    assert c["F18"]["Lemon juice"].text == "0.45 (−0.15)"
    # The share is of the row's allowed range: salt 0–1.8 g.
    assert c["F18"]["Salt"].share == pytest.approx(abs(0.551 - 0.21) / 1.8, abs=0.01)
    totals = [c[k][table.rows[-1]].text for k in table.columns]
    assert totals == ["98.69", "98.69", "97.69", "98.69"]


def test_no_total_limit_reads_total_in_its_unit():
    opt = _burger(rounds=3, fill=False)
    opt.quantity_constraints = []
    opt.formulation_total = None
    ui_batch.ask_alone(opt, n_suggestions=2)
    table = ui_batch.round_compare(opt)
    assert table.rows[-1] == "Total (g)"


def test_tie_names_the_first_made_and_suggestions_are_numbered():
    """After the pilot's Round 6, F16 and F17 tie at 98.3: the column names
    F16 (first made) and the tie is said; an app-generated Round not yet
    made reads Suggestion 1-3, its numbers kept for the workbook."""
    opt = _record_round6(_round6())
    ui_batch.ask_alone(opt, n_suggestions=3)
    table = ui_batch.round_compare(opt)
    assert table.columns[0] == "Best so far (F16 · 98.3)"
    assert table.best_no == 16 and table.tied == [17]
    labels = [c.split(" · ")[0] for c in table.columns[1:]]
    assert labels == ["Suggestion 1", "Suggestion 2", "Suggestion 3"]
    numbers = [int(r["formulation"]) for r in opt.pending_batch]
    assert table.numbers == list(zip([1, 2, 3], numbers))
    assert numbers == [19, 20, 21]
    # The workbook and the uploader read the formulation numbers, never the
    # screen's labels.
    assert opt.batch_frame(opt.pending_batch)["Formulation"].tolist() == numbers
    assert "F16, F17 (tied)" in wording.improve_on_caption([16, 17])
    assert wording.best_tied_help(16, [17]).endswith("tied with F17.")


def test_own_rows_keep_their_numbers_beside_suggestions(monkeypatch):
    opt = _burger(rounds=3)
    ui_batch.ask_alone(opt, n_suggestions=2)
    rows = list(opt.pending_batch)
    rows[1] = dict(rows[1], note="My own")
    monkeypatch.setattr(opt, "pending_batch", rows)
    table = ui_batch.round_compare(opt)
    assert table.columns[1].startswith("Suggestion 1")
    assert table.columns[2].startswith(f"F{rows[1]['formulation']}")


def test_less_tried_label_only_for_explores(monkeypatch):
    opt = _burger(rounds=3)
    ui_batch.ask_alone(opt, n_suggestions=3)
    roles = iter([wording.SUGGESTION_CLOSE, wording.SUGGESTION_DIFFERENT,
                  wording.SUGGESTION_BALANCED])
    monkeypatch.setattr(opt, "suggestion_role", lambda recipe, before=None: next(roles))
    table = ui_batch.round_compare(opt)
    assert table.columns[1:] == ["Suggestion 1", "Suggestion 2 · Less-tried area",
                                 "Suggestion 3"]
    for c in table.columns:
        assert wording.SUGGESTION_CLOSE not in c and wording.SUGGESTION_BALANCED not in c


def test_cold_start_shows_amounts_only():
    opt = _burger(rounds=1)
    ui_batch.ask_alone(opt, n_suggestions=3)
    table = ui_batch.round_compare(opt)
    assert table.best_no is None
    assert not any(c.startswith("Best so far") for c in table.columns)
    assert all("(" not in cell.text for col in table.cells for cell in col)
    assert all(cell.share == 0 for col in table.cells for cell in col)
    styles = ui_batch.compare_styles(table, "light")
    assert not any("background" in css for col in styles for css in col)


# ------------------------------------------------------------------ #
#  Theme: light and dark tokens, shading from the plain cell colour
# ------------------------------------------------------------------ #

@pytest.mark.parametrize("which", ["light", "dark"])
def test_shading_uses_the_themes_tokens(which):
    table = ui_batch.round_compare(_round6())
    tokens = ui_batch._COMPARE_COLOURS[which]
    styles = ui_batch.compare_styles(table, which)
    c = {col: dict(zip(table.rows, css)) for col, css in zip(table.columns, styles)}
    assert f"background-color: {tokens['best']}" in c[table.columns[0]]["Salt"]
    # A hidden change: the plain cell (no background of its own).
    assert "background" not in c["F17"]["Xanthan gum"]
    # Salt +0.34 of 1.8 g (19 %, just under FULL_SHADE): nearly the darkest
    # grey, blended from the theme's plain cell colour.
    salt = _cells(table)["F18"]["Salt"].share
    want = ui_batch._blend(tokens["plain"], tokens["full"], min(1, salt / ui_batch.FULL_SHADE))
    assert f"background-color: {want}" in c["F18"]["Salt"]
    full = ui_batch._blend(tokens["plain"], tokens["full"], 1.0)
    assert full == tokens["full"]
    # A small change: part way from the plain cell colour.
    lemon = c["F16"]["Lemon juice"].split("background-color: ")[1][:7]
    assert lemon not in (tokens["plain"], tokens["full"])
    other = ui_batch._COMPARE_COLOURS["dark" if which == "light" else "light"]
    assert other["best"] not in " ".join(c[table.columns[0]].values())


def test_blend_runs_from_plain_to_full():
    assert ui_batch._blend("#ffffff", "#000000", 0) == "#ffffff"
    assert ui_batch._blend("#ffffff", "#000000", 1) == "#000000"
    assert ui_batch._blend("#0e1117", "#5a6170", 0.5) == "#343944"


# ------------------------------------------------------------------ #
#  The screen
# ------------------------------------------------------------------ #

def _app(name=sp.ROUND6):
    at = AppTest.from_file(APP_PATH, default_timeout=600)
    at.session_state["_loaded_project"] = name
    at.run()
    at.session_state["main_tab"] = wording.TAB_BATCH
    at.run()
    assert not at.exception, [e.message for e in at.exception]
    return at


_FORECAST_WORDS = ("Predicted", "predicted", "forecast", "Forecast", "95 %", "Predicts",
                   "Partly", "Not yet", "worth trying")


def test_round6_screen_has_the_table_captions_and_no_forecast():
    at = _app()
    tab = at.tabs[1]
    captions = [c.value for c in tab.caption]
    assert "Best bets to improve on F10." in captions
    assert wording.CHOSEN_BY_EGBO in captions
    assert ("(change from F10, g) · shading: change vs that ingredient's allowed range"
            in captions)
    assert "Sheets: Round overview · Results" in captions
    assert len(tab.table) == 1
    shown = tab.table[0].value
    assert list(shown.columns) == ["Best so far (F10 · 95.0)", "F16", "F17", "F18"]
    assert shown.loc["Tap water", "F16"] == "42.00 (+0.78)  \nat upper limit"
    assert shown.loc["**Total (allowed 97.69–98.69 g)**", "F17"] == "**97.69**"
    texts = captions + [e.label for e in tab.expander] + [m.value for m in tab.markdown]
    for word in _FORECAST_WORDS:
        assert not any(word in t for t in texts), word
    assert not any(e.label == wording.WHY_THESE for e in tab.expander)
    assert not tab.dataframe        # the old formulation-per-row table is gone
    # The key's "?" says how the shading reads.
    key = next(c for c in tab.caption if c.value.startswith("(change from"))
    assert "allowed range" in key.help


def test_app_generated_round_reads_suggestions_on_screen():
    opt = _record_round6(_round6())
    ui_batch.ask_alone(opt, n_suggestions=3)
    opt.save()
    at = _app()
    tab = at.tabs[1]
    shown = tab.table[0].value
    assert [c.split(" · ")[0] for c in shown.columns[1:]] == [
        "Suggestion 1", "Suggestion 2", "Suggestion 3"]
    assert shown.columns[0] == "Best so far (F16 · 98.3)"
    captions = [c.value for c in tab.caption]
    assert "Best bets to improve on F16, F17 (tied)." in captions
    assert wording.CHOSEN_BY_EGBO not in captions
    assert not any(c.startswith("F19") for c in shown.columns)
