"""Insights wave C on screen: the "More charts" expander after question 4 —
"Can you have both?", "What happens when three ingredients trade places?"
and "How far did each prediction miss?". AppTest, like
tests/test_insights_ui.py, plus the chart builders on their own."""
import json
import pickle
import re

import pytest
import torch
from streamlit.testing.v1 import AppTest

import insights
import ui_insights
import wording
from food_bo import FoodOptimizer
from tests.test_app_ui import APP_PATH, _submit_button, burger, scored  # noqa: F401
from tests.test_insights_ui import _open_insights, _tab_text, insights_scored  # noqa: F401


def _app():
    return AppTest.from_file(APP_PATH, default_timeout=180)


def _rerun_on(at, tab=wording.TAB_INSIGHTS):
    """Run again with `tab` open. AppTest does not replay the tab bar: a
    plain at.run() after a tab was chosen draws the first tab again."""
    at.session_state["main_tab"] = tab
    return at.run()


def _counting(monkeypatch, name):
    calls = []
    real = getattr(insights, name)
    monkeypatch.setattr(insights, name, lambda *a, **k: calls.append(1) or real(*a, **k))
    return calls


def _more(at):
    found = [e for e in at.tabs[3].expander if e.label == wording.MORE_CHARTS]
    return found[0] if found else None


def _charts(block):
    """The Altair charts drawn directly in `block`, as Vega-Lite dicts."""
    return [json.loads(c.proto.spec) for c in block.children.values()
            if getattr(c, "type", "") == "vega_lite_chart"]


def _state(opt):
    return pickle.dumps({k: v for k, v in opt.__dict__.items() if k != "_forms_cache"})


_WAVE_C_WORDS = re.compile(
    r"\b(held|hold|scales?|ranges|residuals?|posterior|probability|Round-ahead|optimum|"
    r"optimal|maximi[sz]e|confidence|the model)\b", re.I)
_CASED = re.compile(r"\b(Min|Max)\b")


def _assert_clean(*strings):
    """The repo-wide guard plus the words wave C must never put on screen."""
    from tests.test_food_bo import _BANNED
    for s in strings:
        assert not any(p.search(s) for p in _BANNED), s
        assert not _WAVE_C_WORDS.search(s), s
        assert not _CASED.search(s), s


# ------------------------------------------------------------------ #
#  Task 3: the expander and "Can you have both?"
# ------------------------------------------------------------------ #

def test_more_charts_is_collapsed_after_question_4_and_before_how(insights_scored):
    at = _open_insights(_app())
    assert not at.exception
    labels = [e.label for e in at.tabs[3].expander]
    assert labels.index(wording.MORE_CHARTS) == labels.index(wording.HOW_WORKED_OUT) - 1
    assert _more(at).proto.expanded is False
    md, _ = _tab_text(at)
    assert md.index(f"**{wording.Q_BEST}**") < md.index(f"**{wording.Q_PAIR}**")


def test_can_you_have_both_opens_on_the_two_largest_shares(insights_scored):
    at = _open_insights(_app())
    more = _more(at)
    assert [m.value for m in more.markdown][0] == f"**{wording.Q_PAIR}**"
    [box] = more.multiselect
    assert box.label == wording.PAIR_PICK
    assert box.value == ["Firmness", "Juiciness"]          # Share 1.5 before 1.0
    assert wording.PAIR_CAPTION in [c.value for c in more.caption]
    [spec] = _charts(more)
    titles = [layer["encoding"]["x"].get("title") for layer in spec["layer"]
              if "x" in layer.get("encoding", {})]
    assert wording.pair_axis("Firmness (N)", "target", 6) in titles


def test_one_measurement_picked_asks_for_two(insights_scored):
    at = _open_insights(_app())
    _more(at).multiselect[0].set_value(["Juiciness"])
    _rerun_on(at)
    assert not at.exception
    more = _more(at)
    assert wording.PICK_TWO in [c.value for c in more.caption]
    assert _charts(more) == []


def test_the_pick_is_kept_across_reruns(insights_scored):
    at = _open_insights(_app())
    _more(at).multiselect[0].set_value(["Juiciness", "Firmness"])
    _rerun_on(at)
    _rerun_on(at)
    assert _more(at).multiselect[0].value == ["Juiciness", "Firmness"]


def test_no_more_charts_before_five_formulations_have_all_their_results(scored):
    at = _open_insights(_app())
    assert not at.exception
    assert _more(at) is None


def test_no_more_charts_with_nothing_to_show(tmp_path, monkeypatch):
    """One measurement, no Formulation total and too few checks: every chart
    is past its gate, so the expander is not drawn empty."""
    monkeypatch.chdir(tmp_path)
    opt = FoodOptimizer("one")
    opt.set_amount_unit("g")
    opt.add_ingredient("Pea protein", 0, 25)
    opt.add_objective("Firmness", 1.0, goal="max", min_val=0, max_val=10, unit="N")
    for no in range(1, 7):
        opt.tell({"Pea protein": 3.0 * no}, {"Firmness": 0.3 * no}, formulation_no=no, batch_no=1)
    at = _open_insights(_app())
    assert not at.exception
    assert _more(at) is None


def test_more_charts_write_nothing_to_the_project(insights_scored):
    at = _open_insights(_app())
    opt = at.session_state["optimizer"]
    before = _state(opt)
    _more(at).multiselect[0].set_value(["Juiciness", "Firmness"])
    _rerun_on(at)
    assert _state(at.session_state["optimizer"]) == before


def test_a_project_switch_drops_the_pair_pick(insights_scored):
    at = _open_insights(_app())
    assert ui_insights.PAIR_KEY in at.session_state
    FoodOptimizer("second").add_ingredient("Flour", 0, 100)
    at.run()
    at.selectbox(key="project_select").set_value("second")
    at.run()
    _submit_button(at.sidebar, "Open").click()
    at.run()
    assert not at.exception
    assert ui_insights.PAIR_KEY not in at.session_state


def test_a_deleted_measurement_drops_out_of_the_pick(insights_scored):
    insights_scored.add_objective("Colour", 1.0, goal="max", min_val=0, max_val=10)
    at = _open_insights(_app())
    _more(at).multiselect[0].set_value(["Firmness", "Colour"])
    _rerun_on(at)
    at.session_state["optimizer"].remove_objective("Colour")
    _rerun_on(at)
    assert not at.exception
    more = _more(at)
    assert more.multiselect[0].value == ["Firmness"]
    assert wording.PICK_TWO in [c.value for c in more.caption]


def test_a_failed_chart_is_one_line_and_the_tab_stands(insights_scored, monkeypatch):
    def boom(*a):
        raise RuntimeError("no pair")
    monkeypatch.setattr(insights, "pair_points", boom)
    at = _open_insights(_app())
    assert not at.exception
    assert [c.value for c in _more(at).caption] == [wording.CHART_FAILED]
    assert wording.HOW_WORKED_OUT in [e.label for e in at.tabs[3].expander]


def _pair_opt(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    opt = FoodOptimizer("pair")
    opt.set_amount_unit("g")
    opt.add_ingredient("Pea protein", 0, 25)
    opt.add_objective("Firmness", 60, goal="target", target=6, min_val=0, max_val=10, unit="N")
    opt.add_objective("Cook loss", 40, goal="min", min_val=0, max_val=30, unit="%")
    for no, (f, c) in enumerate(((5.0, 20.0), (6.0, 25.0), (12.5, 10.0)), start=1):
        opt.tell({"Pea protein": 5.0 * no}, {"Firmness": f, "Cook loss": c},
                 formulation_no=no, batch_no=1)
    opt.add_objective("Juiciness", 20, goal="max", min_val=0, max_val=10)
    opt.tell({"Pea protein": 22.0}, {"Firmness": 7.0, "Cook loss": 15.0},
             formulation_no=4, batch_no=2)                 # waiting for Juiciness
    return opt


def _layers(spec, mark):
    return [l for l in spec["layer"]
            if (l["mark"]["type"] if isinstance(l["mark"], dict) else l["mark"]) == mark]


def test_pair_chart_axes_name_the_goal_and_span_range_plus_data(tmp_path, monkeypatch):
    opt = _pair_opt(tmp_path, monkeypatch)
    spec = ui_insights.pair_chart(opt, "Firmness", "Cook loss",
                                  insights.pair_points(opt, "Firmness", "Cook loss")).to_dict()
    [dots] = [l for l in _layers(spec, "circle")]
    assert dots["encoding"]["x"]["title"] == "Firmness (N) · target 6"
    assert dots["encoding"]["y"]["title"] == "Cook loss (%) · lower is better"
    assert dots["encoding"]["x"]["scale"]["domain"] == [0.0, 12.5]   # 12.5 is past the Range
    assert dots["encoding"]["y"]["scale"]["domain"] == [0.0, 30.0]


def test_pair_chart_target_lines_only_for_targets(tmp_path, monkeypatch):
    opt = _pair_opt(tmp_path, monkeypatch)
    spec = ui_insights.pair_chart(opt, "Firmness", "Cook loss",
                                  insights.pair_points(opt, "Firmness", "Cook loss")).to_dict()
    rules = _layers(spec, "rule")
    assert [list(r["encoding"]) for r in rules] == [["x"]]          # Firmness only
    [rows] = [v for k, v in spec["datasets"].items() if v and "t" in v[0]]
    assert rows == [{"t": 6.0}]


def test_pair_chart_colours_by_score_and_draws_waiting_rows_hollow(tmp_path, monkeypatch):
    opt = _pair_opt(tmp_path, monkeypatch)
    points = insights.pair_points(opt, "Firmness", "Cook loss")
    assert [p.score is None for p in points] == [False, False, False, True]
    spec = ui_insights.pair_chart(opt, "Firmness", "Cook loss", points).to_dict()
    [dots] = _layers(spec, "circle")
    colour = dots["encoding"]["color"]
    assert colour["field"] == "score"
    assert colour["scale"]["domain"] == [0, opt.utility_ceiling()]
    assert colour["scale"]["scheme"] == {"name": "blues", "extent": [0.25, 1]}
    hollow = [l for l in _layers(spec, "point") if l["mark"].get("filled") is False
              and l["mark"].get("color") == "gray" and l["mark"].get("size") == 60]
    assert hollow and hollow[0]["transform"] == [{"filter": {"field": "waiting", "equal": True}}]
    [label] = _layers(spec, "text")
    assert label["encoding"]["text"] == {"value": wording.PAIR_BEST}


def test_pair_axis_wording():
    assert wording.pair_axis("Cook loss (%)", "min") == "Cook loss (%) · lower is better"
    assert wording.pair_axis("Juiciness", "max") == "Juiciness · higher is better"
    assert wording.pair_axis("Firmness (N)", "target", 6) == "Firmness (N) · target 6"
    assert wording.pair_axis("Firmness (N)", "target", None) == "Firmness (N)"


def test_exact_wave_c_strings_for_the_expander_and_the_pair():
    assert wording.MORE_CHARTS == "More charts"
    assert wording.Q_PAIR == "Can you have both?"
    assert wording.PAIR_CAPTION == (
        "Each dot is a formulation you measured; lines mark your targets. If no dot sits near "
        "both targets, the two may pull against each other.")


def test_the_pair_strings_pass_the_word_guards():
    _assert_clean(wording.MORE_CHARTS, wording.Q_PAIR, wording.PAIR_PICK, wording.PICK_TWO,
                  wording.PAIR_CAPTION, wording.PAIR_NONE, wording.PAIR_BEST,
                  wording.CHART_FAILED,
                  wording.pair_axis("Cook loss (%)", "min"),
                  wording.pair_axis("Juiciness", "max"),
                  wording.pair_axis("Firmness (N)", "target", 6))


def test_no_formulation_has_both_values_says_so(insights_scored):
    insights_scored.add_objective("Colour", 1.0, goal="max", min_val=0, max_val=10)
    at = _open_insights(_app())
    _more(at).multiselect[0].set_value(["Firmness", "Colour"])
    _rerun_on(at)
    assert not at.exception
    more = _more(at)
    assert wording.PAIR_NONE in [c.value for c in more.caption]
    assert _charts(more) == []


# ------------------------------------------------------------------ #
#  Task 4: "What happens when three ingredients trade places?"
# ------------------------------------------------------------------ #

@pytest.fixture
def mixture_scored(tmp_path, monkeypatch):
    """Pea protein 10–20 g, Oil 2–6 g, Starch 1–5 g and Water filling to
    100 g; two measurements; twelve formulations over four Rounds, all with
    their results (enough for all three charts)."""
    monkeypatch.chdir(tmp_path)
    opt = FoodOptimizer("mix")
    opt.set_amount_unit("g")
    opt.add_ingredient("Pea protein", 10, 20)
    opt.add_ingredient("Oil", 2, 6)
    opt.add_ingredient("Starch", 1, 5)
    opt.add_ingredient("Water", 0, 100)
    opt.set_formulation_total(100)
    opt.set_formula("Water", "= rest")
    opt.add_objective("Firmness", 1.5, goal="target", target=6, min_val=0, max_val=10, unit="N")
    opt.add_objective("Juiciness", 1.0, goal="max", min_val=0, max_val=10, unit="/10")
    g = torch.Generator().manual_seed(4)
    for no in range(1, 13):
        u = torch.rand(3, generator=g).tolist()
        pea, oil, starch = 10 + 10 * u[0], 2 + 4 * u[1], 1 + 4 * u[2]
        opt.tell(opt.fill_formulas({"Pea protein": pea, "Oil": oil, "Starch": starch}),
                 {"Firmness": 0.4 * pea + 0.3 * starch, "Juiciness": 1 + 1.2 * oil},
                 formulation_no=no, batch_no=1 + (no - 1) // 3)
    return opt


def _triangles(block):
    return [c for c in _charts(block) if c.get("width") == ui_insights._MIX_SIZE]


def test_the_triangle_heading_picks_and_caption(mixture_scored):
    at = _open_insights(_app())
    assert not at.exception
    more = _more(at)
    md = [m.value for m in more.markdown]
    assert md.index(f"**{wording.Q_PAIR}**") < md.index(f"**{wording.Q_MIXTURE}**")
    boxes = {b.label: b for b in more.multiselect}
    three = boxes[wording.MIXTURE_PICK]
    assert three.options == ["Pea protein", "Oil", "Starch", "Water"]
    assert len(three.value) == 3
    best_no = mixture_scored.best_formulation_no()
    assert wording.mixture_caption(best_no) in [c.value for c in more.caption]
    [tri] = _triangles(more)
    assert tri["height"] == ui_insights._MIX_SIZE


def test_the_triangle_opens_on_the_largest_spreads(mixture_scored, monkeypatch):
    monkeypatch.setattr(insights, "spreads", lambda opt, fitted, start: [
        insights.Spread("Starch", 9.0), insights.Spread("Oil", 5.0),
        insights.Spread("Pea protein", 1.0)])
    at = _open_insights(_app())
    three = next(b for b in _more(at).multiselect if b.label == wording.MIXTURE_PICK)
    assert three.value == ["Starch", "Oil", "Pea protein"]


def test_without_spreads_the_triangle_opens_on_the_first_three(mixture_scored, monkeypatch):
    monkeypatch.setattr(insights, "spreads", lambda opt, fitted, start: [])
    at = _open_insights(_app())
    three = next(b for b in _more(at).multiselect if b.label == wording.MIXTURE_PICK)
    assert three.value == ["Pea protein", "Oil", "Starch"]


def test_two_picked_asks_for_three(mixture_scored):
    at = _open_insights(_app())
    next(b for b in _more(at).multiselect
         if b.label == wording.MIXTURE_PICK).set_value(["Oil", "Water"])
    _rerun_on(at)
    assert not at.exception
    more = _more(at)
    assert wording.PICK_THREE in [c.value for c in more.caption]
    assert _triangles(more) == []


def test_the_triangle_starts_from_section_2s_choice(mixture_scored):
    at = _open_insights(_app())
    at.tabs[3].selectbox(key=ui_insights.START_KEY).set_value(4)
    _rerun_on(at)
    assert wording.mixture_caption(4) in [c.value for c in _more(at).caption]


def test_the_triangle_is_worked_out_once(mixture_scored, monkeypatch):
    calls = _counting(monkeypatch, "mixture_slice")
    at = _open_insights(_app())
    _rerun_on(at)
    _rerun_on(at)
    assert calls == [1]
    next(b for b in _more(at).multiselect
         if b.label == wording.MIXTURE_PICK).set_value(["Oil", "Starch", "Water"])
    _rerun_on(at)
    assert calls == [1, 1]


def test_a_failed_triangle_is_one_line_and_not_retried(mixture_scored, monkeypatch):
    calls = []

    def boom(*a, **k):
        calls.append(1)
        raise RuntimeError("no triangle")
    monkeypatch.setattr(insights, "mixture_slice", boom)
    at = _open_insights(_app())
    assert not at.exception
    more = _more(at)
    assert wording.panel_failed(wording.OVERALL_SCORE_COLUMN) in [c.value for c in more.caption]
    assert wording.PAIR_CAPTION in [c.value for c in more.caption]      # the others still draw
    _rerun_on(at)
    assert calls == [1]


def _refused(monkeypatch, **kw):
    def fake(opt, names, start, steps=insights.GRID_STEPS):
        base = dict(names=list(names), units=["g"] * 3, start=dict(start), total=20.0,
                    lower=[0.0] * 3, upper=[1.0] * 3, span=0.0, points=[], dots=[],
                    refusal=None, start_breaks=None, follows=[], fill_moves=None, skipped=[])
        return insights.Mixture(**{**base, **kw})
    monkeypatch.setattr(insights, "mixture_slice", fake)


def test_no_room_is_one_line(mixture_scored, monkeypatch):
    _refused(monkeypatch, refusal=insights.NO_ROOM)
    at = _open_insights(_app())
    more = _more(at)
    best_no = mixture_scored.best_formulation_no()
    assert wording.mixture_no_room(best_no) in [c.value for c in more.caption]
    assert _triangles(more) == []


def test_a_start_that_breaks_a_limit_is_one_line(mixture_scored, monkeypatch):
    _refused(monkeypatch, start_breaks="Oil would go past its Highest")
    at = _open_insights(_app())
    best_no = mixture_scored.best_formulation_no()
    assert wording.path_start_breaks(best_no, "Oil would go past its Highest") in [
        c.value for c in _more(at).caption]


def test_the_fill_following_a_calculation_is_one_line(mixture_scored, monkeypatch):
    _refused(monkeypatch, refusal=insights.FILL_FOLLOWS, follows=["Hydration"])
    at = _open_insights(_app())
    assert wording.mixture_fill_follows("Hydration", "Water") in [
        c.value for c in _more(at).caption]


def test_calculation_and_fill_lines_under_the_caption(mixture_scored):
    opt = mixture_scored
    opt.add_ingredient("Hydration", 0, 50)
    opt.set_formula("Hydration", "= 0.5 * Oil")
    at = _open_insights(_app())
    next(b for b in _more(at).multiselect
         if b.label == wording.MIXTURE_PICK).set_value(["Pea protein", "Oil", "Starch"])
    _rerun_on(at)
    assert not at.exception
    caps = [c.value for c in _more(at).caption]
    i = caps.index(wording.mixture_caption(opt.best_formulation_no()))
    assert caps[i + 1:i + 3] == [wording.path_follows("Hydration"), wording.mixture_fills("Water")]


def test_a_project_switch_drops_the_wave_c_keys(mixture_scored):
    at = _open_insights(_app())
    keys = (ui_insights.PAIR_KEY, ui_insights.THREE_KEY, ui_insights.MIXTURE_KEY)
    assert all(k in at.session_state for k in keys)
    FoodOptimizer("second").add_ingredient("Flour", 0, 100)
    at.run()
    at.selectbox(key="project_select").set_value("second")
    at.run()
    _submit_button(at.sidebar, "Open").click()
    at.run()
    assert not at.exception
    assert not any(k in at.session_state for k in keys)


def test_the_triangle_is_drawn_at_its_own_size(mixture_scored, monkeypatch):
    """A layered chart stretches to the column by default, which would pull
    the triangle out of shape: it is drawn at its own width and height."""
    import streamlit as st
    drawn = []
    real = st.altair_chart

    def spy(chart, *a, **k):
        drawn.append((chart.to_dict().get("width"), k.get("width")))
        return real(chart, *a, **k)
    monkeypatch.setattr(st, "altair_chart", spy)
    _open_insights(_app())
    assert (ui_insights._MIX_SIZE, "content") in drawn


def test_the_triangle_writes_nothing_and_leaves_the_rng(mixture_scored):
    at = _open_insights(_app())
    before = _state(at.session_state["optimizer"])
    rng = torch.get_rng_state().clone()
    next(b for b in _more(at).multiselect
         if b.label == wording.MIXTURE_PICK).set_value(["Oil", "Starch", "Water"])
    _rerun_on(at)
    assert _state(at.session_state["optimizer"]) == before
    assert torch.equal(torch.get_rng_state(), rng)


def _mix_parts(opt):
    fitted = insights.fit_all(opt)
    start = insights.start_recipe(opt, opt.best_index())
    mix = insights.mixture_slice(opt, ["Pea protein", "Oil", "Water"], start)
    return mix, insights.mixture_predictions(opt, fitted, mix)


def test_triangle_corners_on_the_page():
    assert ui_insights._xy((1.0, 0.0, 0.0)) == (0.0, 0.0)
    assert ui_insights._xy((0.0, 1.0, 0.0)) == (1.0, 0.0)
    x, y = ui_insights._xy((0.0, 0.0, 1.0))
    assert (x, y) == pytest.approx((0.5, 3 ** 0.5 / 2))


def test_triangle_chart_keeps_its_shape_and_names_its_corners(mixture_scored):
    opt = mixture_scored
    mix, preds = _mix_parts(opt)
    spec = ui_insights.mixture_chart(opt, mix, preds, 7).to_dict()
    assert spec["width"] == spec["height"]
    enc = next(l for l in spec["layer"] if l["mark"]["type"] == "circle")["encoding"]
    xd, yd = enc["x"]["scale"]["domain"], enc["y"]["scale"]["domain"]
    assert xd[1] - xd[0] == pytest.approx(yd[1] - yd[0])
    assert enc["x"]["axis"] is None and enc["y"]["axis"] is None
    texts = [r["t"] for rows in spec["datasets"].values() for r in rows if "t" in r]
    for k, name in enumerate(mix.names):
        assert wording.mixture_corner(name, ui_insights._amount(mix.upper[k], "g")) in texts
    assert wording.start_formulation(7) in texts


def test_triangle_chart_grey_where_a_limit_is_broken_and_smaller_where_untested(mixture_scored):
    opt = mixture_scored
    mix, preds = _mix_parts(opt)
    spec = ui_insights.mixture_chart(opt, mix, preds, 7).to_dict()
    # opacity 1 excludes the invisible, legend-only circle layer.
    circles = [l for l in spec["layer"]
               if l["mark"]["type"] == "circle" and l["mark"].get("opacity") == 1]
    grey, coloured = circles
    assert grey["mark"]["color"] == "lightgray"
    assert [t["title"] for t in grey["encoding"]["tooltip"]][-1] == wording.MIXTURE_BLOCKED
    assert coloured["encoding"]["color"]["scale"]["scheme"] == {"name": "blues", "extent": [0.25, 1]}
    allowed = [q.mean for q in preds if q is not None]
    assert coloured["encoding"]["color"]["scale"]["domain"] == pytest.approx([min(allowed), max(allowed)])
    size = coloured["encoding"]["size"]["scale"]
    assert size["domain"] == [wording.MIXTURE_TESTED, wording.MIXTURE_UNTESTED, wording.MIXTURE_BLOCKED]
    assert size["range"][0] > size["range"][1]
    frame, _, _ = ui_insights.mixture_frame(opt, mix, preds)
    assert (frame["state"] == wording.MIXTURE_BLOCKED).sum() == sum(p.broken is not None for p in mix.points)
    assert (frame["state"] == wording.MIXTURE_TESTED).sum() == sum(
        p.tested and p.broken is None for p in mix.points)
    shown = frame[frame["state"] != wording.MIXTURE_BLOCKED]["pred"]
    assert all(" (" in t for t in shown)


# ------------------------------------------------------------------ #
#  Task 4 fix round 1: the triangle as Streamlit actually renders it
# ------------------------------------------------------------------ #

def _mark_type(layer):
    mark = layer["mark"]
    return mark["type"] if isinstance(mark, dict) else mark


def test_the_triangle_is_not_squashed_by_streamlits_default_autosize(mixture_scored):
    """Streamlit sets autosize: fit/padding when a spec declares none, which
    lets the legends eat into the 380 px width. Pinning autosize to "pad"
    keeps the plot itself at its full, undistorted size."""
    opt = mixture_scored
    mix, preds = _mix_parts(opt)
    spec = ui_insights.mixture_chart(opt, mix, preds, 7).to_dict()
    assert spec["width"] == spec["height"] == ui_insights._MIX_SIZE
    assert spec["autosize"]["type"] == "pad"


def test_the_corner_and_start_labels_are_grey_not_black(mixture_scored):
    """Streamlit's theme sets no default text colour, so an uncoloured
    mark_text renders black — invisible in dark mode."""
    opt = mixture_scored
    mix, preds = _mix_parts(opt)
    spec = ui_insights.mixture_chart(opt, mix, preds, 7).to_dict()
    texts = [l for l in spec["layer"] if _mark_type(l) == "text"]
    assert len(texts) == 4          # the three corners plus the start's own label
    assert all(l["mark"].get("color") == "gray" for l in texts)


def test_decorative_layers_carry_no_x_y_tooltip_and_the_ring_has_one(mixture_scored):
    """The outline and the text layers only draw the frame; without an
    explicit tooltip they would otherwise leak raw X/Y on hover. The start
    ring gets a real one: the formulation number and its three amounts, the
    same fields (and titles) as the grid's own tooltip."""
    opt = mixture_scored
    mix, preds = _mix_parts(opt)
    spec = ui_insights.mixture_chart(opt, mix, preds, 7).to_dict()
    for layer in spec["layer"]:
        if _mark_type(layer) in ("line", "text"):
            assert layer["mark"].get("tooltip") is False
    ring = next(l for l in spec["layer"]
                if _mark_type(l) == "point" and l["mark"].get("size") == 260)
    tips = ring["encoding"]["tooltip"]
    assert [t["field"] for t in tips] == ["no", "a0", "a1", "a2", "sum"]
    assert tips[0]["title"] == wording.FORMULATION_CAP


def test_the_state_legend_shows_three_differently_sized_and_coloured_marks(mixture_scored):
    """Tested near here: larger, blue. Not tested near here: small, blue.
    Not allowed by your limits: small, light grey. Vega-Lite merges a size
    legend with a fill/stroke legend that share the same field into one
    combined legend, so this comes from an invisible layer alongside the
    grey/grid ones, using fill+stroke (not colour, which the grid layer's
    own predicted-score gradient already claims)."""
    opt = mixture_scored
    mix, preds = _mix_parts(opt)
    spec = ui_insights.mixture_chart(opt, mix, preds, 7).to_dict()
    dummy = next(l for l in spec["layer"]
                 if l["mark"].get("type") == "circle" and l["mark"].get("opacity") == 0)
    kinds = [wording.MIXTURE_TESTED, wording.MIXTURE_UNTESTED, wording.MIXTURE_BLOCKED]
    size_scale = dummy["encoding"]["size"]["scale"]
    fill_scale = dummy["encoding"]["fill"]["scale"]
    stroke_scale = dummy["encoding"]["stroke"]["scale"]
    assert size_scale["domain"] == kinds
    assert size_scale["range"][0] > size_scale["range"][1] == size_scale["range"][2]
    # Vega-Lite compiles a nominal size scale with no explicit type to
    # "point", which only honours the range's two ends and interpolates
    # the middle one instead of using its own value: pin "ordinal" so the
    # untested entry draws at 16, not partway between 70 and 16.
    assert size_scale["type"] == "ordinal"
    assert fill_scale["domain"] == stroke_scale["domain"] == kinds
    assert fill_scale["range"] == stroke_scale["range"]
    assert fill_scale["range"][2] == "lightgray"
    assert fill_scale["range"][0] == fill_scale["range"][1] != "lightgray"
    assert dummy["encoding"]["fill"]["legend"]["labelLimit"] == 0


def test_the_invisible_legend_layer_is_clipped(mixture_scored):
    """Its rows sit at X=Y=-1, well outside the plotted domains, so they
    are never mistaken for real data — but under Streamlit's autosize
    Vega would otherwise still grow the canvas to include them even at
    opacity 0. clip=True keeps the canvas to the plot plus the legend."""
    opt = mixture_scored
    mix, preds = _mix_parts(opt)
    spec = ui_insights.mixture_chart(opt, mix, preds, 7).to_dict()
    dummy = next(l for l in spec["layer"]
                 if l["mark"].get("type") == "circle" and l["mark"].get("opacity") == 0)
    assert dummy["mark"]["clip"] is True
    x_domain = dummy["encoding"]["x"]["scale"]["domain"]
    y_domain = dummy["encoding"]["y"]["scale"]["domain"]
    assert all(not (x_domain[0] <= r["X"] <= x_domain[1]) for r in
               spec["datasets"][dummy["data"]["name"]])
    assert all(not (y_domain[0] <= r["Y"] <= y_domain[1]) for r in
               spec["datasets"][dummy["data"]["name"]])


def test_the_left_and_right_corner_labels_anchor_at_their_own_corner(mixture_scored):
    """The left corner's label grows rightward from x=0 with align="left";
    the right corner's grows leftward from x=1 with align="right"; the top
    label stays centred. This keeps the two side labels away from each
    other instead of drifting toward the middle."""
    opt = mixture_scored
    mix, preds = _mix_parts(opt)
    spec = ui_insights.mixture_chart(opt, mix, preds, 7).to_dict()
    left_text = wording.mixture_corner(mix.names[0], ui_insights._amount(mix.upper[0], mix.units[0]))
    right_text = wording.mixture_corner(mix.names[1], ui_insights._amount(mix.upper[1], mix.units[1]))
    top_text = wording.mixture_corner(mix.names[2], ui_insights._amount(mix.upper[2], mix.units[2]))

    def align_for(text):
        for l in spec["layer"]:
            if _mark_type(l) != "text":
                continue
            rows = spec["datasets"].get(l["data"]["name"], [])
            if any(r.get("t") == text for r in rows):
                return l["mark"]["align"]
        return None
    assert align_for(left_text) == "left"
    assert align_for(right_text) == "right"
    assert align_for(top_text) == "center"


def test_exact_wave_c_strings_for_the_triangle():
    assert wording.Q_MIXTURE == "What happens when three ingredients trade places?"
    assert wording.mixture_caption(14) == (
        "Colour = predicted Overall score as the three amounts trade places; everything else "
        "as in Formulation 14.")
    assert wording.mixture_corner("Coconut oil", "9.4 g") == "Coconut oil up to 9.4 g"


def test_the_triangle_strings_pass_the_word_guards():
    _assert_clean(wording.Q_MIXTURE, wording.MIXTURE_PICK, wording.PICK_THREE,
                  wording.mixture_caption(14), wording.mixture_corner("Coconut oil", "9.4 g"),
                  wording.mixture_fills("Water"), wording.mixture_no_room(14),
                  wording.mixture_fill_follows("Hydration water", "Remaining water"),
                  wording.mixture_together("g"), wording.MIXTURE_TESTED,
                  wording.MIXTURE_UNTESTED, wording.MIXTURE_BLOCKED)
