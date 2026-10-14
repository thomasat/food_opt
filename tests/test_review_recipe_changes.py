"""What changed in the formulation (review, 2026-10-08): under Start vs best,
against target, Start (the first Round's best, as that chart) against Best
so far, one row per varied ingredient or setting — model-free, a fact about
amounts, never a cause. Synthetic projects only."""
import json
import itertools

import pytest
from streamlit.testing.v1 import AppTest

import ui_insights as U
import wording
from food_bo import FoodOptimizer
from tests.test_app_ui import APP_PATH
from tests.test_food_bo import _BANNED

_n = itertools.count(1)


@pytest.fixture(autouse=True)
def _isolate_cwd(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)


# Start F2 (Round 1's best) -> Best so far F4: Oil 10 -> 12 (+2 of 20, 0.10),
# Sugar 0 -> 0.6 (+0.6 of 2, 0.30), Starch 2 -> 4 (+2 of 4, 0.50).
ROWS = [  # (Round, number, Oil, Sugar, Starch, Firmness)
    (1, 1, 6.0, 1.0, 3.0, 3.0),
    (1, 2, 10.0, 0.0, 2.0, 5.0),
    (1, 3, 14.0, 1.5, 1.5, 4.0),
    (2, 4, 12.0, 0.6, 4.0, 6.0),
]


def _opt(rows=ROWS, fill=False, fixed=False, setting=None, starch=(1, 5)):
    opt = FoodOptimizer(f"changes_{next(_n)}")
    opt.set_amount_unit("g")
    opt.add_ingredient("Oil", 0, 20)
    opt.add_ingredient("Sugar", 0, 2)
    opt.add_ingredient("Starch", *starch)
    if fixed:
        opt.add_ingredient("Salt", 1, 1)
    if fill:
        opt.add_ingredient("Water", 0, 100)
        opt.set_formulation_total(100)
        opt.set_balance("Water")
    if setting:
        opt.add_process_parameter("Bake temperature", 150, 200, unit="°C")
    opt.add_objective("Firmness", 100, goal="target", target=6, min_val=0, max_val=10)
    for k, (r, no, oil, sugar, starch_g, firm) in enumerate(rows):
        recipe = {"Oil": oil, "Sugar": sugar, "Starch": starch_g}
        if fixed:
            recipe["Salt"] = 1.0
        if setting:
            recipe["Bake temperature"] = setting[k]
        opt.tell(recipe, {"Firmness": firm}, formulation_no=no, batch_no=r)
    return opt


def _row(ch, name):
    return next(r for r in ch.rows if r.name == name)


def _banned(text):
    return [p.pattern for p in _BANNED if p.search(text)]


def test_start_is_the_first_rounds_best_as_start_vs_best():
    opt = _opt()
    ch = U.what_changed(opt)
    s = U.story_start(opt)
    assert (ch.start_no, ch.best_no, ch.start_round) == (2, 4, 1)
    assert ch.start_no == int(opt.formulation_ids[s[0]])
    assert ch.best_no == opt.best_formulation_no()


def test_rows_ordered_by_share_of_range_largest_first():
    ch = U.what_changed(_opt())
    assert [r.name for r in ch.rows] == ["Starch", "Sugar", "Oil"]
    assert [round(r.share, 3) for r in ch.rows] == [0.5, 0.3, 0.1]
    # The picture (spec §10): by share of range, not by grams.
    assert [r["label"] for r in U.changes_picture(_opt(), ch)["rows"]] == [
        "+2 g", "+0.6 g", "+2.0 g"]


def test_signs_units_decimals_and_percent():
    ch = U.what_changed(_opt())
    oil, starch, sugar = _row(ch, "Oil"), _row(ch, "Starch"), _row(ch, "Sugar")
    assert (oil.start, oil.best, oil.change, oil.percent) == ("10.0 g", "12.0 g", "+2.0 g",
                                                              "(+20 %)")
    assert (starch.start, starch.best, starch.change, starch.percent) == ("2 g", "4 g", "+2 g",
                                                                          "(+100 %)")
    rows = [(1, 1, 6, 1, 3, 3), (1, 2, 10, 1.6, 4, 5), (2, 3, 3.6, 1.2, 3, 6)]
    down = U.what_changed(_opt(rows))
    assert _row(down, "Oil").change == "−6.4 g"          # a true minus sign
    assert _row(down, "Oil").percent == "(−64 %)"
    assert _row(down, "Sugar").change == "−0.4 g"


def test_percent_left_out_when_start_is_zero():
    sugar = _row(U.what_changed(_opt()), "Sugar")
    assert (sugar.start, sugar.change, sugar.percent) == ("0.0 g", "+0.6 g", "")


def test_the_picture_names_the_two_formulations_and_keeps_the_percent_in_the_tooltip():
    opt = _opt()
    pic = U.changes_picture(opt, U.what_changed(opt))
    assert (pic["start"], pic["best"]) == ("Start (F2)", "Best so far (F4)")
    tip = {r["name"]: (r["start_text"], r["best_text"], r["change_text"]) for r in pic["rows"]}
    assert tip["Starch"] == ("2 g", "4 g", "+2 g (+100 %)")
    assert tip["Sugar"] == ("0.0 g", "0.6 g", "+0.6 g")


def test_fixed_and_fill_to_total_rows_are_left_out_and_fill_is_said():
    opt = _opt(fill=True, fixed=True)
    ch = U.what_changed(opt)
    assert {r.name for r in ch.rows} == {"Oil", "Sugar", "Starch"}
    assert ch.fill == "Water" and ch.fixed
    help_ = wording.changes_picture_help("Start (F2)", "Best so far (F4)", ch.start_round,
                                         fill=ch.fill, fixed=ch.fixed)
    assert "Water fills to the total, so it follows the others and is left out." in help_


def test_process_settings_in_their_own_unit_without_percent():
    opt = _opt(setting=[160, 160, 170, 180])
    ch = U.what_changed(opt)
    bake = _row(ch, "Bake temperature")
    assert (bake.start, bake.best, bake.change, bake.percent) == ("160 °C", "180 °C", "+20 °C", "")
    assert bake.setting and round(bake.share, 2) == 0.4
    assert [r.name for r in ch.rows] == ["Starch", "Bake temperature", "Sugar", "Oil"]
    pic = U.changes_picture(opt, ch)
    assert next(r["label"] for r in pic["rows"] if r["name"] == "Bake temperature") == "+20 °C"


def test_a_choice_row_shows_its_option_names():
    opt = FoodOptimizer(f"changes_{next(_n)}")
    opt.add_process_parameter("Proof time", 30, 90, unit="min")
    opt.variables.append({'name': "Package", 'type': 'categorical',
                          'options': ["Tray", "Pouch"], 'category': 'process'})
    opt.add_objective("Volume", 100, goal="max", min_val=0, max_val=10)
    for no, (r, t, pack, v) in enumerate(((1, 30, "Tray", 3), (1, 60, "Tray", 4),
                                          (2, 66, "Pouch", 8)), 1):
        opt.tell({"Proof time": float(t), "Package": pack}, {"Volume": v},
                 formulation_no=no, batch_no=r)
    ch = U.what_changed(opt)
    pack = _row(ch, "Package")
    assert (pack.start, pack.best, pack.change) == ("Tray", "Pouch", wording.CHANGES_CHOICE_CHANGED)
    assert ch.rows[0].name == "Package"          # a changed choice: the whole of its options
    row = U.changes_picture(opt, ch)["rows"][0]
    assert (row["name"], row["label"], row["start"]) == ("Package", "Tray → Pouch", None)


def test_best_so_far_is_the_start():
    rows = [(1, 1, 6, 1, 3, 3), (1, 2, 10, 0, 2, 6), (2, 3, 12, 0.6, 4, 5)]
    opt = _opt(rows)
    ch = U.what_changed(opt)
    assert ch.same and ch.rows == [] and ch.limits_help is None
    assert wording.changes_same(ch.start_round) == "Best so far is still Round 1's best."


def test_limits_only_when_best_sits_at_a_limit():
    # Round panel (spec §10, re-pinned): the limits are in the title's "?".
    assert U.what_changed(_opt()).limits_help is None
    rows = ROWS[:3] + [(2, 4, 12.0, 0.6, 5.0, 6.0)]           # Starch at its Highest
    ch = U.what_changed(_opt(rows))
    assert ch.limits_help.startswith("Best so far sits at: Starch's Highest (5 g).")
    assert "lies beyond" not in ch.limits_help               # nothing tested beyond
    assert wording.CHANGES_WIDEN in ch.limits_help
    assert "should" not in ch.limits_help.lower()
    assert U.changes_picture(_opt(rows), ch)["rows"][0]["label"] == "+3 g, at Highest"


def test_lowest_and_a_limit_on_several_rows_are_named():
    rows = ROWS[:3] + [(2, 4, 12.0, 0.0, 1.0, 6.0)]           # Sugar and Starch at Lowest
    opt = _opt(rows)
    opt.add_quantity_constraint(["Oil", "Sugar"], max_val=12.0)
    ch = U.what_changed(opt)
    # F3 (14 + 1.5 g) lies beyond the Oil + Sugar limit; nothing below the
    # Lowests. The one-row limits first, then the limit on several rows.
    assert ch.limits_help.startswith("Best so far sits at: Sugar's Lowest (0 g); "
                                     "Starch's Lowest (1 g); the Oil + Sugar (≤ 12 g).")
    assert "Formulation 3 lies beyond the Oil + Sugar (≤ 12 g)." in ch.limits_help


def test_a_limit_on_several_rows_alone_is_named():
    """Follow-up 3: Best so far at no row's limit, only at a limit on
    several rows (the yogurt's state B): that limit in the "?"."""
    opt = _opt()
    opt.add_quantity_constraint(["Oil", "Sugar"], max_val=12.6)
    ch = U.what_changed(opt)
    assert ch.limits_help.startswith("Best so far sits at: the Oil + Sugar (≤ 12.6 g).")
    assert "Formulation 3 lies beyond the Oil + Sugar (≤ 12.6 g)." in ch.limits_help
    rows = [(1, 1, 6, 1, 3, 3), (1, 2, 10, 0, 2, 5), (1, 3, 9, 1.5, 1.5, 4), (2, 4, 12, 0.6, 4, 6)]
    clear = _opt(rows)
    clear.add_quantity_constraint(["Oil", "Sugar"], max_val=12.6)
    assert "lies beyond" not in U.what_changed(clear).limits_help


def test_change_is_the_shown_best_minus_the_shown_start():
    """Follow-up 1: 57.62 → 51.27 shows 57.6 → 51.3, so the change reads
    −6.3 (never −6.4, the unrounded −6.35), on the picture too."""
    rows = [(1, 1, 50.0, 1.0, 3.0, 3.0), (1, 2, 57.62, 1.0, 3.0, 5.0),
            (2, 3, 51.27, 1.0, 3.0, 6.0)]
    opt = FoodOptimizer(f"changes_{next(_n)}")
    opt.set_amount_unit("g")
    opt.add_ingredient("Coconut cream", 40, 70)
    opt.add_ingredient("Sugar", 0, 2)
    opt.add_ingredient("Starch", 1, 5)
    opt.add_objective("Firmness", 100, goal="target", target=6, min_val=0, max_val=10)
    for r, no, c, su, stch, f in rows:
        opt.tell({"Coconut cream": c, "Sugar": su, "Starch": stch}, {"Firmness": f},
                 formulation_no=no, batch_no=r)
    ch = U.what_changed(opt)
    row = _row(ch, "Coconut cream")
    assert (row.start, row.best, row.change) == ("57.6 g", "51.3 g", "−6.3 g")
    assert row.percent == "(−11 %)"                    # −6.3 of 57.6
    assert U.changes_picture(opt, ch)["rows"][0]["label"] == "−6.3 g"


def test_no_percent_from_a_start_near_zero():
    """Follow-up 2: a % only from a Start of at least 10 % of the row's
    range; the "?" says why the others have none."""
    rows = [(1, 1, 6, 1, 3, 3), (1, 2, 10, 0.1, 2, 5), (2, 3, 12, 1.2, 4, 6)]
    opt = _opt(rows)
    ch = U.what_changed(opt)
    sugar = _row(ch, "Sugar")                          # 0.1 g of a 2 g range: 5 %
    assert (sugar.start, sugar.change, sugar.percent) == ("0.1 g", "+1.1 g", "")
    assert sugar.near_zero
    assert _row(ch, "Oil").percent == "(+20 %)"        # 10 g of a 20 g range
    edge = _opt([(1, 1, 6, 1, 3, 3), (1, 2, 10, 0.2, 2, 5), (2, 3, 12, 1.2, 4, 6)])
    assert _row(U.what_changed(edge), "Sugar").percent == "(+500 %)"   # exactly 10 %


def test_nothing_tested_beyond_is_false_when_an_earlier_formulation_exceeds_the_limit():
    # Starch's Highest was 6 when Formulation 1 was made at 5.5 g; it is 5 now.
    rows = [(1, 1, 6.0, 1.0, 5.5, 3.0)] + ROWS[1:3] + [(2, 4, 12.0, 0.6, 5.0, 6.0)]
    opt = _opt(rows, starch=(1, 6))
    opt._var_by_name("Starch")['bounds'] = (1.0, 5.0)
    ch = U.what_changed(opt)
    assert ch.limits_help.startswith("Best so far sits at: Starch's Highest (5 g).")
    assert "Formulation 1 lies beyond Starch's Highest (5 g)." in ch.limits_help


def test_an_open_rounds_row_beyond_the_limit_counts_as_tested():
    rows = ROWS[:3] + [(2, 4, 12.0, 0.6, 5.0, 6.0)]
    opt = _opt(rows, starch=(1, 6))
    opt.pending_batch = [{'formulation': 5, 'recipe': {"Oil": 8.0, "Sugar": 1.0, "Starch": 5.8}}]
    opt._var_by_name("Starch")['bounds'] = (1.0, 5.0)
    assert "Formulation 5 lies beyond" in U.what_changed(opt).limits_help


def test_word_limits():
    opt = _opt(fill=True, fixed=True, setting=[160, 160, 170, 180])
    ch = U.what_changed(opt)
    assert len(wording.CHANGES_TITLE.split()) <= 5
    assert len(wording.CHANGES_CAPTION.split()) <= 8
    assert U.limit_words(wording.changes_same(1)) <= 12
    assert all(len(r["label"].split()) <= 5 for r in U.changes_picture(opt, ch)["rows"])
    # Many limits: every row says it on the face, the "?" names them.
    many = FoodOptimizer(f"changes_{next(_n)}")
    names = [f"Ingredient number {k}" for k in range(1, 7)]
    for n in names:
        many.add_ingredient(n, 0, 10)
    many.add_objective("Firmness", 100, goal="target", target=6, min_val=0, max_val=10)
    many.tell({n: 5.0 for n in names}, {"Firmness": 3}, formulation_no=1, batch_no=1)
    many.tell({n: 10.0 for n in names}, {"Firmness": 6}, formulation_no=2, batch_no=2)
    ch = U.what_changed(many)
    assert {r["label"] for r in U.changes_picture(many, ch)["rows"]} == {"+5.0 g, at Highest"}
    assert len(ch.limits_help.split()) <= wording.TOOLTIP_WORDS


def test_no_banned_word():
    opt = _opt(fill=True, fixed=True, setting=[160, 160, 170, 180])
    ch = U.what_changed(opt)
    pic = U.changes_picture(opt, ch)
    texts = [wording.CHANGES_TITLE, wording.CHANGES_CAPTION, wording.CHANGES_CHANGE,
             wording.CHANGES_CHOICE_CHANGED, wording.CHANGES_CHOICE_SAME, wording.CHANGES_WIDEN,
             wording.CHANGES_NONE, wording.changes_same(1),
             wording.changes_picture_help(pic["start"], pic["best"], 1, fill="Water", fixed=True),
             *[r["label"] for r in pic["rows"]]]
    rows = ROWS[:3] + [(2, 4, 12.0, 0.0, 1.0, 6.0)]
    lim = _opt(rows)
    lim.add_quantity_constraint(["Oil", "Sugar"], max_val=12.0)
    texts += [U.what_changed(lim).limits_help]
    assert [(t, _banned(t)) for t in texts if _banned(t)] == []


def test_the_review_draws_the_block_under_start_vs_best():
    opt = _opt()
    opt.save()
    at = AppTest.from_file(APP_PATH, default_timeout=600)
    at.session_state["_loaded_project"] = opt.project_name
    at.run()
    at.session_state["main_tab"] = wording.TAB_INSIGHTS
    at.run()
    assert not at.exception, [e.message for e in at.exception]
    texts = [m.value for m in at.tabs[3].markdown]
    # Slim Review (re-pinned): the section's Round slider (two Rounds) opens
    # at the latest; the title and the table name it.
    title = f"**{wording.changes_title(2)}**"
    assert title in texts
    k = texts.index(title)
    # Layout study (re-pinned): What changed sits beside the profile in the
    # Replay panel (left, under the score chart; the profile on the right),
    # its header wrapping only before the bracket (non-breaking spaces).
    assert texts.index(f"**{wording.STORY_TARGET_TITLE}**") > k
    # Round panel (spec §10, re-pinned): a picture, not a table — each row
    # its change at the right, the legend naming Start and the best after
    # the Round on view.
    assert not [t for t in texts if t.startswith("| ")]
    specs = [json.loads(e.proto.spec) for e in at.tabs[3].get("vega_lite_chart")]
    pictures = [s for s in specs if "start_text" in json.dumps(s)]
    assert len(pictures) == 1
    assert wording.changes_best_column(4, 2) in json.dumps(pictures[0])
    assert "Start (F2)" in json.dumps(pictures[0])
    pic = U.changes_picture(opt, U.what_changed(opt, 2))
    assert [(r["name"], r["label"]) for r in pic["rows"]][:2] == [
        ("Starch", "+2 g"), ("Sugar", "+0.6 g")]
    assert wording.CHANGES_CAPTION in [c.value for c in at.tabs[3].caption]
