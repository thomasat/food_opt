"""v3 lane C (final-v3, 3 · Results): "Round N results" — ratings against
their targets, no forecasts — and "How the score is worked out", whose
points are food_bo's own and add up to the stored Overall score; the
workbook's Preparation sheet only when a Round has pre-mixes; the shipped
Round 6 results file rebuilt from the current workbook."""
import importlib.util
import io

import pytest
from openpyxl import load_workbook
from streamlit.testing.v1 import AppTest

import sample_projects as sp
import storage as storage_backend
import ui_results
import wording
from food_bo import FoodOptimizer, uploaded_parts
from tests.test_app_ui import APP_PATH

NAMES = ("Consistency", "Creaminess", "Tangy", "Thickness uniformity")


def _builder():
    spec = importlib.util.spec_from_file_location("yogurt_build", sp.SAMPLE_DIR / "build.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture
def store(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    return storage_backend.LocalStorage()


def upload_round6(opt, source=None):
    """2 · Make a round > Or upload results from a file > Save uploaded
    results, with the shipped file (or `source`)."""
    with open(sp.ROUND6_RESULTS, "rb") if source is None else source as fh:
        upload = opt.results_from_workbook(fh)
    sheet, weighed, lots = uploaded_parts(upload)
    parsed, left_out = opt.parse_batch_results(
        sheet, opt.pending_batch, with_skipped=True, weighed=weighed)
    by_number = {r["formulation"]: r["recipe"] for r in opt.pending_batch}
    for number, results, note in parsed:
        opt.tell(opt.amounts_as_weighed(by_number[number], weighed.get(number)),
                 results, formulation_no=number, batch_no=opt.pending_batch_no, note=note)
    opt.store_lots(opt.pending_batch_no, lots)
    opt.set_pending_batch(None)
    return parsed, left_out


@pytest.fixture
def round6(store):
    sp.ensure(store)
    opt = FoodOptimizer(sp.ROUND6, storage=store)
    upload_round6(opt)
    return FoodOptimizer(sp.ROUND6, storage=store)


# ------------------------------------------------------------------ #
#  Round N results
# ------------------------------------------------------------------ #

def test_round6_results_table(round6):
    table = ui_results.round_results(round6)
    assert table["round"] == 6
    assert table["columns"] == [
        "Formulation", "Overall score", "Consistency (target 12)",
        "Creaminess (target 5)", "Tangy (target 5)", "Thickness uniformity (target 13)"]
    rows = table["frame"].values.tolist()
    assert rows == [
        ["F16 · new best (tied)", "98.3", "12", "5", "4", "13"],
        ["F17 · new best (tied)", "98.3", "12", "5", "4", "13"],
        ["F18", "91.7", "13", "7", "4", "12"]]
    assert table["best"] == [0, 1]
    tangy, cons, cream, thick = (table["columns"][k] for k in (4, 2, 3, 5))
    assert table["off"] == {(0, tangy), (1, tangy), (2, tangy), (2, cons),
                            (2, cream), (2, thick)}
    assert table["caption"] == ("Panel median (n = 5), 0–15; each made once, rated "
                                "once. Orange: off target.")


def test_the_new_best_is_measured_against_every_earlier_formulation(round6):
    """F16 and F17 beat F10's 95.0 (Round 4) by more than the tolerance;
    Round 5 raised nothing; Round 1's top is the first best."""
    assert [int(round6.formulation_ids[i]) for i in ui_results.new_best_rows(round6, 6)] \
        == [16, 17]
    assert ui_results.new_best_rows(round6, 5) == []
    assert len(ui_results.new_best_rows(round6, 1)) == 1


def test_a_single_new_best_is_not_called_tied(round6):
    i = round6.index_of_formulation(17)
    round6.edit_result(i, {**round6.results_history[i], "Consistency": 11.0})
    table = ui_results.round_results(round6)
    assert table["frame"].iloc[0, 0] == "F16 · new best"
    assert table["frame"].iloc[1, 0] == "F17"
    assert table["best"] == [0]


def test_a_tie_with_an_earlier_best_is_no_new_best(store):
    """Equal within 1e-6 to the best before the Round: not new."""
    sp.ensure(store)
    opt = FoodOptimizer(sp.ROUND6, storage=store)
    f10 = opt.results_history[opt.index_of_formulation(10)]
    for row in list(opt.pending_batch):
        opt.tell(row["recipe"], dict(f10), formulation_no=row["formulation"], batch_no=6)
    opt.set_pending_batch(None)
    table = ui_results.round_results(opt)
    assert table["best"] == []
    assert not any("new best" in v for v in table["frame"].iloc[:, 0])


def test_a_project_without_a_recorded_panel_claims_none(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    opt = FoodOptimizer("Mine", robust=False)
    opt.add_ingredient("Oil", 0, 10)
    opt.add_objective("Taste", 60, goal="max", min_val=0, max_val=10)
    opt.add_objective("Cook loss", 40, goal="target", target=20, min_val=0, max_val=100, unit="%")
    opt.tell({"Oil": 5}, {"Taste": 7, "Cook loss": 25}, batch_no=1)
    table = ui_results.round_results(opt)
    assert table["caption"] == "Orange: points lost."
    assert table["columns"][2:] == ["Taste (0–10, higher is better)",
                                    "Cook loss (0–100, %, target 20)"]
    assert sp.panel(opt) is None


def test_waiting_and_not_scored_rows_say_so(store):
    sp.ensure(store)
    opt = FoodOptimizer(sp.ROUND6, storage=store)
    a, b, c = list(opt.pending_batch)
    opt.tell(a["recipe"], {"Consistency": 12, "Creaminess": 5, "Tangy": 5,
                           "Thickness uniformity": 13}, formulation_no=a["formulation"], batch_no=6)
    opt.tell(b["recipe"], {"Consistency": 12}, formulation_no=b["formulation"], batch_no=6)
    opt.record_skipped(c["formulation"], 6, c["recipe"])
    opt.set_pending_batch(None)
    frame = ui_results.round_results(opt)["frame"]
    assert frame["Overall score"].tolist() == ["100.0", wording.MEASURED_WAITING,
                                               wording.NOT_SCORED]
    assert frame.iloc[0, 0] == "F16 · new best"
    assert frame.iloc[1, 2:].tolist() == ["12", "", "", ""]


def test_results_says_no_forecast(round6):
    """The Results tab after Round 6: the table and the fold, and nothing
    predicted, no range, no In/Above/Below, no stop."""
    at = AppTest.from_file(APP_PATH, default_timeout=300)
    at.session_state["_loaded_project"] = sp.ROUND6
    at.run()
    at.session_state["main_tab"] = wording.TAB_RESULTS
    at.run()
    assert not at.exception
    tab = at.tabs[2]
    said = " ".join([m.value for m in tab.markdown] + [c.value for c in tab.caption]
                    + [e.label for e in tab.expander])
    assert "Round 6 results" in said
    assert wording.SCORE_WORKED_OUT_FOLD in said
    for word in ("redict", "orecast", "95 %", "Above", " range", "measured against",
                 wording.REVIEW_SAYS_STOP, "noise", "Stop"):
        assert word not in said, word
    tables = [t.value for t in tab.table]
    assert any("F16 · new best (tied)" in str(t.index.tolist()) for t in tables)
    assert not [k for k in (b.key for b in tab.button) if k == "link_round_checked"]


# ------------------------------------------------------------------ #
#  How the score is worked out
# ------------------------------------------------------------------ #

def test_breakdown_equals_the_stored_score_for_every_pilot_formulation(store):
    sp.ensure(store)
    opt = FoodOptimizer(sp.ALL10, storage=store)
    assert len(opt.Y_history) == 30
    for i in range(30):
        parts = opt.score_breakdown(i)
        assert abs(sum(p["points"] for p in parts) - float(opt.Y_history[i])) < 1e-9
        shown = ui_results.score_breakdown_frame(opt, i)
        assert abs(shown["total"] - float(opt.Y_history[i])) < 1e-9
        assert shown["frame"].iloc[-1].tolist() == [
            "Overall score", "", "", "100", f"{float(opt.Y_history[i]):.1f}"]


def test_breakdown_of_f16(round6):
    shown = ui_results.score_breakdown_frame(round6, round6.index_of_formulation(16))
    assert shown["line"] == "Each point off costs 1/15 of its share."
    assert shown["frame"].values.tolist() == [
        ["Consistency", "12", "12", "25", "25.0"],
        ["Creaminess", "5", "5", "25", "25.0"],
        ["Tangy", "4", "5", "25", "23.3"],
        ["Thickness uniformity", "13", "13", "25", "25.0"],
        ["Overall score", "", "", "100", "98.3"]]
    assert shown["off"] == {(2, "Rating"), (2, "Points earned")}


@pytest.mark.parametrize("goal", ["max", "min", "target"])
def test_breakdown_equals_the_score_for_every_goal_and_a_partial_row(tmp_path, monkeypatch, goal):
    """Higher, lower and target goals, clipped ratings, and a formulation
    recorded before a measurement was added (its shares rescaled)."""
    monkeypatch.chdir(tmp_path)
    opt = FoodOptimizer("Goals", robust=False)
    opt.add_ingredient("Oil", 0, 10)
    opt.add_objective("Taste", 50, goal=goal, target=6, min_val=0, max_val=10)
    opt.add_objective("Bite", 30, goal="target", target=40, min_val=0, max_val=100)
    opt.tell({"Oil": 1}, {"Taste": 7.5, "Bite": 55}, batch_no=1)
    opt.tell({"Oil": 2}, {"Taste": 12, "Bite": -3}, batch_no=1)
    opt.add_objective("Crunch", 20, goal="min", min_val=0, max_val=5)
    opt.tell({"Oil": 3}, {"Taste": 2, "Bite": 40, "Crunch": 1.5}, batch_no=2)
    for i in range(len(opt.Y_history)):
        parts = opt.score_breakdown(i)
        assert abs(sum(p["points"] or 0.0 for p in parts) - float(opt.Y_history[i])) < 1e-9
        assert opt._compute_utility(opt.results_history[i], opt._number_at(i)) \
            == opt.Y_history[i]
    first = {p["name"]: p for p in opt.score_breakdown(0)}
    assert first["Crunch"]["points"] is None and not first["Crunch"]["applies"]
    assert abs(first["Taste"]["share_of_score"] + first["Bite"]["share_of_score"] - 100.0) \
        < 1e-9
    frame = ui_results.score_breakdown_frame(opt, 0)["frame"]
    assert frame.set_index("Attribute").loc["Crunch", "Rating"] == wording.DOES_NOT_APPLY


# ------------------------------------------------------------------ #
#  The workbook: Preparation only with pre-mixes; uploads either way
# ------------------------------------------------------------------ #

@pytest.fixture
def plain(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    opt = FoodOptimizer("plain", robust=False)
    opt.add_ingredient("Protein", 5, 20)
    opt.add_ingredient("Water", 0, 100)
    opt.add_objective("Taste", 1, goal="max", min_val=0, max_val=10)
    opt.set_pending_batch([{"Protein": 10, "Water": 90}], batch_no=1)
    return opt


def _with_premix(opt):
    opt.add_premix("Seasoning", "portioned")
    opt.set_premix_parts("Seasoning", [{"name": "Salt", "share": 100, "unit": "g"}])
    opt.add_ingredient("Seasoning", 1, 3)
    opt.set_pending_batch([{"Protein": 10, "Seasoning": 2, "Water": 88}], batch_no=1)
    return opt


def _book(opt):
    return load_workbook(io.BytesIO(opt.workbook_bytes(opt.pending_batch, 100, print_pack=False)))


def _save(book):
    out = io.BytesIO()
    book.save(out)
    out.seek(0)
    return out


def _texts(book):
    return [str(c.value) for s in book if s.sheet_state == "visible"
            for row in s for c in row if c.value is not None]


def _enter_taste(book, value=8):
    sheet = book["Results"]
    row = next(c.row for r in sheet for c in r
               if c.column == 1 and str(c.value or "").startswith("Taste"))
    sheet.cell(row, 2, value)


def test_without_premixes_no_preparation_sheet_or_mention(plain):
    book = _book(plain)
    assert [s.title for s in book if s.sheet_state == "visible"] == ["Round overview", "Results"]
    assert not [t for t in _texts(book) if "Preparation" in t]
    links = {c.hyperlink.location for s in book for r in s for c in r if c.hyperlink}
    assert links == {"'Round overview'!A1", "'Results'!A1"}
    _enter_taste(book)
    assert plain.results_from_workbook(_save(book)).frame["Taste"].tolist() == [8]


def test_with_premixes_the_preparation_sheet_is_there_and_named(plain):
    book = _book(_with_premix(plain))
    assert [s.title for s in book if s.sheet_state == "visible"] == [
        "Round overview", "Preparation", "Results"]
    assert book["Results"]["A2"].value.startswith(wording.sheet_navigation(True))
    assert "Round overview → Preparation → Results" in wording.sheet_navigation(True)
    _enter_taste(book)
    assert plain.results_from_workbook(_save(book)).frame["Taste"].tolist() == [8]


def test_a_premix_workbook_missing_its_preparation_sheet_is_refused_plainly(plain):
    """A pre-mix Round's workbook carries its Preparation sheet; one deleted
    by hand is refused with the Round-information message, never read
    without the pre-mix pages."""
    book = _book(_with_premix(plain))
    book.remove(book["Preparation"])
    _enter_taste(book, 6)
    with pytest.raises(ValueError, match="Round information"):
        plain.results_from_workbook(_save(book))


def test_an_older_workbook_with_an_empty_preparation_sheet_still_uploads(plain):
    """Downloaded before v3: an empty Preparation sheet beside the two."""
    book = _book(plain)
    sheet = book.create_sheet("Preparation", 1)
    sheet["A1"] = "Preparation — Pre-mixes for this round"
    sheet["A4"] = "No pre-mixes are required. Follow the method on Round overview."
    _enter_taste(book, 7)
    assert plain.results_from_workbook(_save(book)).frame["Taste"].tolist() == [7]


# ------------------------------------------------------------------ #
#  The shipped Round 6 results file
# ------------------------------------------------------------------ #

def _cells(book):
    return {s.title: [[c.value for c in row] for row in s.iter_rows()]
            for s in book if s.sheet_state == "visible"}


def test_the_shipped_round6_file_equals_a_rebuild(tmp_path):
    """build.py's round6_results() writes it from the current workbook:
    same sheets and cells, the pilot's values and notes."""
    out = tmp_path / "rebuilt.xlsx"
    _builder().write_round6_results(str(out), work_dir=str(tmp_path / "work"))
    shipped, rebuilt = load_workbook(sp.ROUND6_RESULTS), load_workbook(out)
    assert _cells(shipped) == _cells(rebuilt)
    assert [s.title for s in shipped if s.sheet_state == "visible"] == [
        "Round overview", "Results"]


def test_the_shipped_round6_file_records_the_pilots_values(store):
    sp.ensure(store)
    opt = FoodOptimizer(sp.ROUND6, storage=store)
    parsed, left_out = upload_round6(opt)
    assert left_out == []
    by_no = {no: (tuple(r[n] for n in NAMES), note) for no, r, note in parsed}
    assert by_no == {16: ((12, 5, 4, 13), "Pilot batch 6.1 · EGBO"),
                     17: ((12, 5, 4, 13), "Pilot batch 6.2 · EGBO"),
                     18: ((13, 7, 4, 12), "Pilot batch 6.3 · EGBO")}
    scores = {int(n): round(float(y), 1) for n, y in zip(opt.formulation_ids, opt.Y_history)}
    assert (scores[16], scores[17], scores[18], scores[10]) == (98.3, 98.3, 91.7, 95.0)
