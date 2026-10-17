"""Insights wave C on screen: the "More charts" expander after question 4 —
"Can you have both?", "What happens when three ingredients trade places?"
and "How far did each prediction miss?". AppTest, like
tests/test_insights_ui.py, plus the chart builders on their own."""
import dataclasses
import itertools
import json
import pickle
import re

import pytest
from tests.demo_build_reason import REVIEW_FOUR
import torch
from streamlit.testing.v1 import AppTest

import insights
import ui_insights
import wording
from food_bo import FoodOptimizer
from tests.test_app_ui import APP_PATH, _submit_button, burger, scored  # noqa: F401
from tests.test_insights_ui import insights_scored  # noqa: F401


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
    """Charts proto: the two-measurement fold (Two measurements, was
    Trade-offs) and the triangle, which now sits in Explored and
    unexplored, read as one block (pair first, as these tests were
    written); None when neither is drawn."""
    folds = {e.label: e for e in at.tabs[3].expander}
    found = [folds[k] for k in (wording.PAIR_FOLD, wording.FOLD_EXPLORED) if k in folds]
    pair = wording.PAIR_FOLD in folds
    tri = wording.FOLD_EXPLORED in folds and any(
        m.value == f"##### {wording.Q_MIXTURE}" for m in folds[wording.FOLD_EXPLORED].markdown)
    if not pair and not tri:
        return None
    return _Folds(found if tri else found[:1])


class _Folds:
    """Several folds read as one: list attributes joined, in order."""

    def __init__(self, folds):
        self.folds = folds

    @property
    def proto(self):
        return self.folds[0].proto

    @property
    def label(self):
        return self.folds[0].label

    @property
    def children(self):
        out = {}
        for k, f in enumerate(self.folds):
            out.update({(k, key): c for key, c in f.children.items()})
        return out

    def __getattr__(self, name):
        out = []
        for f in self.folds:
            out += list(getattr(f, name))
        return out


def _open_insights(at):
    """test_insights_ui's, with Explored and unexplored open: the triangle
    is drawn there now (charts proto), and only while it is open."""
    at.run()
    at.session_state[ui_insights.EXPLORED_KEY] = True
    at.session_state["main_tab"] = wording.TAB_INSIGHTS
    at.run()
    return at


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


@pytest.mark.skip(reason=REVIEW_FOUR)
def test_can_you_have_both_opens_on_the_two_largest_shares(insights_scored):
    at = _open_insights(_app())
    more = _more(at)
    assert [m.value for m in more.markdown][0] == f"##### {wording.Q_PAIR}"
    [box] = more.multiselect
    assert box.label == wording.PAIR_PICK
    assert box.value == ["Firmness", "Juiciness"]          # Share 1.5 before 1.0
    assert wording.PAIR_CAPTION in [t for c in more.caption for t in (c.value, c.proto.help)]    # both have targets
    [spec] = _charts(more)
    titles = [layer["encoding"]["x"].get("title") for layer in spec["layer"]
              if "x" in layer.get("encoding", {})]
    assert wording.pair_axis("Firmness (N)", "target", 6) in titles


@pytest.mark.skip(reason=REVIEW_FOUR)
def test_one_measurement_picked_asks_for_two(insights_scored):
    at = _open_insights(_app())
    _more(at).multiselect[0].set_value(["Juiciness"])
    _rerun_on(at)
    assert not at.exception
    more = _more(at)
    assert wording.PICK_TWO in [t for c in more.caption for t in (c.value, c.proto.help)]
    assert _charts(more) == []


@pytest.mark.skip(reason=REVIEW_FOUR)
def test_the_pick_is_kept_across_reruns(insights_scored):
    at = _open_insights(_app())
    _more(at).multiselect[0].set_value(["Juiciness", "Firmness"])
    _rerun_on(at)
    _rerun_on(at)
    assert _more(at).multiselect[0].value == ["Juiciness", "Firmness"]


@pytest.mark.skip(reason=REVIEW_FOUR)
def test_more_charts_write_nothing_to_the_project(insights_scored):
    at = _open_insights(_app())
    opt = at.session_state["optimizer"]
    before = _state(opt)
    _more(at).multiselect[0].set_value(["Juiciness", "Firmness"])
    _rerun_on(at)
    assert _state(at.session_state["optimizer"]) == before


@pytest.mark.skip(reason=REVIEW_FOUR)
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


@pytest.mark.skip(reason=REVIEW_FOUR)
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
    assert wording.PICK_TWO in [t for c in more.caption for t in (c.value, c.proto.help)]


@pytest.mark.skip(reason=REVIEW_FOUR)
def test_a_failed_chart_is_one_line_and_the_tab_stands(insights_scored, monkeypatch):
    def boom(*a):
        raise RuntimeError("no pair")
    monkeypatch.setattr(insights, "pair_points", boom)
    at = _open_insights(_app())
    assert not at.exception
    assert [c.value for c in _more(at).caption] == [wording.CHART_FAILED]
    assert wording.FOLD_METHOD in [e.label for e in at.tabs[3].expander]


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


@pytest.mark.skip(reason=REVIEW_FOUR)
def test_pair_chart_axes_name_the_goal_and_span_range_plus_data(tmp_path, monkeypatch):
    opt = _pair_opt(tmp_path, monkeypatch)
    spec = ui_insights.pair_chart(opt, "Firmness", "Cook loss",
                                  insights.pair_points(opt, "Firmness", "Cook loss")).to_dict()
    [dots] = [l for l in _layers(spec, "circle")]
    assert dots["encoding"]["x"]["title"] == "Firmness (N) · target 6"
    assert dots["encoding"]["y"]["title"] == "Cook loss (%) · lower is better"
    assert dots["encoding"]["x"]["scale"]["domain"] == [0.0, 12.5]   # 12.5 is past the Range
    assert dots["encoding"]["y"]["scale"]["domain"] == [0.0, 30.0]


@pytest.mark.skip(reason=REVIEW_FOUR)
def test_pair_chart_target_lines_only_for_targets(tmp_path, monkeypatch):
    opt = _pair_opt(tmp_path, monkeypatch)
    spec = ui_insights.pair_chart(opt, "Firmness", "Cook loss",
                                  insights.pair_points(opt, "Firmness", "Cook loss")).to_dict()
    rules = _layers(spec, "rule")
    assert [list(r["encoding"]) for r in rules] == [["x"]]          # Firmness only
    [rows] = [v for k, v in spec["datasets"].items() if v and "t" in v[0]]
    assert rows == [{"t": 6.0}]


@pytest.mark.skip(reason=REVIEW_FOUR)
def test_pair_chart_colours_by_score_and_draws_waiting_rows_hollow(tmp_path, monkeypatch):
    opt = _pair_opt(tmp_path, monkeypatch)
    points = insights.pair_points(opt, "Firmness", "Cook loss")
    assert [p.score is None for p in points] == [False, False, False, True]
    spec = ui_insights.pair_chart(opt, "Firmness", "Cook loss", points).to_dict()
    [dots] = _layers(spec, "circle")
    colour = dots["encoding"]["color"]
    assert colour["field"] == "score"
    # Review plots: the measured score's ramp, as What you've tried.
    assert colour["scale"]["range"] == ui_insights.score_colours()
    hollow = [l for l in _layers(spec, "point") if l["mark"].get("filled") is False
              and l["mark"].get("color") == ui_insights.mark("neutral")[0]  # D1 token (integration)
              and l["mark"].get("size") == 60]
    assert hollow and hollow[0]["transform"] == [{"filter": {"field": "waiting", "equal": True}}]
    [label] = [l for l in _layers(spec, "text") if not l["mark"].get("stroke")]
    assert label["encoding"]["text"] == {"value": wording.PAIR_BEST}


def test_pair_axis_wording():
    assert wording.pair_axis("Cook loss (%)", "min") == "Cook loss (%) · lower is better"
    assert wording.pair_axis("Juiciness", "max") == "Juiciness · higher is better"
    assert wording.pair_axis("Firmness (N)", "target", 6) == "Firmness (N) · target 6"
    assert wording.pair_axis("Firmness (N)", "target", None) == "Firmness (N)"


def test_exact_wave_c_strings_for_the_expander_and_the_pair():
    # Concision round: fold labels at most 5 words.
    assert wording.trade_offs_fold() == "Trade-offs"
    assert wording.trade_offs_fold(False) == "Trade-offs"
    assert wording.Q_PAIR == "Two measurements against each other"
    assert wording.PAIR_CAPTION == (
        "Measured values, one dot per formulation; lines at the targets. No dot near both "
        "targets suggests a trade-off.")


def test_the_pair_strings_pass_the_word_guards():
    _assert_clean(wording.trade_offs_fold(), wording.Q_PAIR, wording.PAIR_PICK, wording.PICK_TWO,
                  wording.PAIR_CAPTION, wording.PAIR_NONE, wording.PAIR_BEST,
                  wording.CHART_FAILED,
                  wording.pair_axis("Cook loss (%)", "min"),
                  wording.pair_axis("Juiciness", "max"),
                  wording.pair_axis("Firmness (N)", "target", 6))


@pytest.mark.skip(reason=REVIEW_FOUR)
def test_no_formulation_has_both_values_says_so(insights_scored):
    insights_scored.add_objective("Colour", 1.0, goal="max", min_val=0, max_val=10)
    at = _open_insights(_app())
    _more(at).multiselect[0].set_value(["Firmness", "Colour"])
    _rerun_on(at)
    assert not at.exception
    more = _more(at)
    assert wording.PAIR_NONE in [t for c in more.caption for t in (c.value, c.proto.help)]
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
    # The one chart drawn without a view frame (configure_view(stroke=None)).
    return [c for c in _charts(block)
            if "stroke" in ((c.get("config") or {}).get("view") or {})
            and c["config"]["view"]["stroke"] is None]


def _corner_texts(mix):
    """The three labels as mixture_chart draws them: each one's largest
    amount inside the region, one shared number of decimals across all
    three (the most any of them needs, up to 2)."""
    amounts = mix.top
    decimals = ui_insights._shared_decimals(amounts)
    return [wording.mixture_corner(mix.names[k],
                                   ui_insights.join_unit(f"{amounts[k]:.{decimals}f}", mix.units[k]))
            for k in range(3)]


@pytest.mark.skip(reason=REVIEW_FOUR)
def test_the_triangle_heading_picks_and_caption(mixture_scored):
    at = _open_insights(_app())
    assert not at.exception
    more = _more(at)
    md = [m.value for m in more.markdown]
    assert md.index(f"##### {wording.Q_PAIR}") < md.index(f"##### {wording.Q_MIXTURE}")
    boxes = {b.label: b for b in more.multiselect}
    three = boxes[wording.MIXTURE_PICK]
    assert three.options == ["Pea protein", "Oil", "Starch", "Water"]
    assert len(three.value) == 3
    best_no = mixture_scored.best_formulation_no()
    assert wording.mixture_caption(best_no) in [t for c in more.caption for t in (c.value, c.proto.help)]
    [tri] = _triangles(more)
    assert max(tri["width"], tri["height"]) == ui_insights._MIX_SIZE


def _most_room(opt, order):
    """The first combination of three, in `order`, with the largest allowed
    share (the default rule, worked out here independently)."""
    start = insights.start_recipe(opt, opt.best_index())
    combos = list(itertools.combinations(order, 3))[:ui_insights._THREE_TRIES]
    shares = [insights.allowed_share(opt, list(c), start) for c in combos]
    return list(combos[shares.index(max(shares))])


@pytest.mark.skip(reason=REVIEW_FOUR)
def test_the_triangle_opens_on_the_most_room_in_spread_order(mixture_scored, monkeypatch):
    monkeypatch.setattr(insights, "spreads", lambda opt, fitted, start: insights.Spreads([
        insights.Spread("Starch", 9.0), insights.Spread("Oil", 5.0),
        insights.Spread("Pea protein", 1.0)], []))
    at = _open_insights(_app())
    three = next(b for b in _more(at).multiselect if b.label == wording.MIXTURE_PICK)
    assert three.value == _most_room(mixture_scored, ["Starch", "Oil", "Pea protein", "Water"])


@pytest.mark.skip(reason=REVIEW_FOUR)
def test_without_spreads_the_triangle_opens_on_the_most_room_in_set_up_order(
        mixture_scored, monkeypatch):
    monkeypatch.setattr(insights, "spreads", lambda opt, fitted, start: insights.Spreads([], []))
    at = _open_insights(_app())
    three = next(b for b in _more(at).multiselect if b.label == wording.MIXTURE_PICK)
    assert three.value == _most_room(mixture_scored, ["Pea protein", "Oil", "Starch", "Water"])


@pytest.mark.skip(reason=REVIEW_FOUR)
def test_two_picked_asks_for_three(mixture_scored):
    at = _open_insights(_app())
    next(b for b in _more(at).multiselect
         if b.label == wording.MIXTURE_PICK).set_value(["Oil", "Water"])
    _rerun_on(at)
    assert not at.exception
    more = _more(at)
    assert wording.PICK_THREE in [t for c in more.caption for t in (c.value, c.proto.help)]
    assert _triangles(more) == []


@pytest.mark.skip(reason="review profiler prototype: pins Try one's path and Compare all, which the profiler replaces; reconcile at the final merge")
def test_the_triangle_starts_from_section_2s_choice(mixture_scored):
    at = _open_insights(_app())
    at.tabs[3].selectbox(key=ui_insights.START_KEY).set_value(4)
    _rerun_on(at)
    assert wording.mixture_caption(4) in [t for c in _more(at).caption for t in (c.value, c.proto.help)]


def _full_slices(monkeypatch):
    """Counts the triangle's own slices: the default three's room check
    (steps=1, three points) is not one."""
    calls = []
    real = insights.mixture_slice

    def counted(*a, **k):
        if k.get("steps", insights.GRID_STEPS) != 1:
            calls.append(1)
        return real(*a, **k)
    monkeypatch.setattr(insights, "mixture_slice", counted)
    return calls


@pytest.mark.skip(reason=REVIEW_FOUR)
def test_the_triangle_is_worked_out_once(mixture_scored, monkeypatch):
    calls = _full_slices(monkeypatch)
    at = _open_insights(_app())
    _rerun_on(at)
    _rerun_on(at)
    assert calls == [1]
    box = next(b for b in _more(at).multiselect if b.label == wording.MIXTURE_PICK)
    other = next(list(c) for c in itertools.combinations(box.options, 3)
                 if list(c) != list(box.value))
    box.set_value(other)
    _rerun_on(at)
    assert calls == [1, 1]


@pytest.mark.skip(reason=REVIEW_FOUR)
def test_a_failed_triangle_is_one_line_and_not_retried(mixture_scored, monkeypatch):
    calls = []

    def boom(*a, **k):
        if k.get("steps", insights.GRID_STEPS) != 1:
            calls.append(1)
        raise RuntimeError("no triangle")
    monkeypatch.setattr(insights, "mixture_slice", boom)
    at = _open_insights(_app())
    assert not at.exception
    more = _more(at)
    assert wording.panel_failed(wording.OVERALL_SCORE_COLUMN) in [t for c in more.caption for t in (c.value, c.proto.help)]
    assert wording.pair_caption(["Firmness"]) in [t for c in more.caption for t in (c.value, c.proto.help)]  # the others still draw
    _rerun_on(at)
    assert calls == [1]


def _refused(monkeypatch, **kw):
    def fake(opt, names, start, steps=insights.GRID_STEPS):
        base = dict(names=list(names), units=["g"] * 3, start=dict(start), total=20.0,
                    lower=[0.0] * 3, upper=[1.0] * 3, span=0.0, points=[], dots=[],
                    refusal=None, start_breaks=None, follows=[], fill_moves=None, skipped=[])
        return insights.Mixture(**{**base, **kw})
    monkeypatch.setattr(insights, "mixture_slice", fake)


@pytest.mark.skip(reason=REVIEW_FOUR)
def test_no_room_is_one_line(mixture_scored, monkeypatch):
    _refused(monkeypatch, refusal=insights.NO_ROOM)
    at = _open_insights(_app())
    more = _more(at)
    best_no = mixture_scored.best_formulation_no()
    assert wording.mixture_no_room(best_no) in [t for c in more.caption for t in (c.value, c.proto.help)]
    assert _triangles(more) == []


@pytest.mark.skip(reason=REVIEW_FOUR)
def test_a_start_that_breaks_a_limit_is_one_line(mixture_scored, monkeypatch):
    _refused(monkeypatch, start_breaks="Oil would go past its Highest")
    at = _open_insights(_app())
    best_no = mixture_scored.best_formulation_no()
    assert wording.path_start_breaks(best_no, "Oil would go past its Highest") in [
        c.value for c in _more(at).caption]


@pytest.mark.skip(reason=REVIEW_FOUR)
def test_the_fill_following_a_calculation_is_one_line(mixture_scored, monkeypatch):
    _refused(monkeypatch, refusal=insights.FILL_FOLLOWS, follows=["Hydration"])
    at = _open_insights(_app())
    assert wording.mixture_fill_follows("Hydration", "Water") in [
        c.value for c in _more(at).caption]


@pytest.mark.skip(reason=REVIEW_FOUR)
def test_calculation_and_fill_lines_under_the_caption(mixture_scored):
    opt = mixture_scored
    opt.add_ingredient("Hydration", 0, 50)
    opt.set_formula("Hydration", "= 0.5 * Oil")
    at = _open_insights(_app())
    next(b for b in _more(at).multiselect
         if b.label == wording.MIXTURE_PICK).set_value(["Pea protein", "Oil", "Starch"])
    _rerun_on(at)
    assert not at.exception
    caps = [t for c in _more(at).caption for t in (c.value, c.proto.help) if t]
    i = caps.index(wording.mixture_caption(opt.best_formulation_no()))
    assert caps[i + 1:i + 3] == [wording.path_follows("Hydration"), wording.mixture_fills("Water")]


@pytest.mark.skip(reason=REVIEW_FOUR)
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


@pytest.mark.skip(reason=REVIEW_FOUR)
def test_the_triangle_is_drawn_at_its_own_size(mixture_scored, monkeypatch):
    """A layered chart stretches to the column by default, which would pull
    the triangle out of shape: it is drawn at its own width and height."""
    import streamlit as st
    drawn = []
    real = st.altair_chart

    def spy(chart, *a, **k):
        spec = chart.to_dict()
        drawn.append((max(spec.get("width") or 0, spec.get("height") or 0), k.get("width")))
        return real(chart, *a, **k)
    monkeypatch.setattr(st, "altair_chart", spy)
    _open_insights(_app())
    assert (ui_insights._MIX_SIZE, "content") in drawn


@pytest.mark.skip(reason=REVIEW_FOUR)
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


@pytest.mark.skip(reason=REVIEW_FOUR)
def test_triangle_corners_on_the_page():
    assert ui_insights._xy((1.0, 0.0, 0.0)) == (0.0, 0.0)
    assert ui_insights._xy((0.0, 1.0, 0.0)) == (1.0, 0.0)
    x, y = ui_insights._xy((0.0, 0.0, 1.0))
    assert (x, y) == pytest.approx((0.5, 3 ** 0.5 / 2))


@pytest.mark.skip(reason=REVIEW_FOUR)
def test_corner_labels_never_mix_decimal_counts(mixture_scored):
    """One corner's own amount can round to a whole number (10 g) by
    `_num`'s own rule (one decimal from 10) while another needs none (9 g):
    shown side by side that reads as two different precisions, so the three
    corners share one count instead — the most any of them needs, up to 2."""
    opt = mixture_scored
    mix, preds = _mix_parts(opt)
    mix = dataclasses.replace(mix, top=[10.0, 9.0, 5.0])
    spec = ui_insights.mixture_chart(opt, mix, preds, 7).to_dict()
    texts = [r["t"] for rows in spec["datasets"].values() for r in rows if "t" in r]
    assert wording.mixture_corner(mix.names[0], "10.0 g") in texts
    assert wording.mixture_corner(mix.names[1], "9.0 g") in texts
    assert wording.mixture_corner(mix.names[2], "5.0 g") in texts
    assert wording.mixture_corner(mix.names[0], "10 g") not in texts
    assert wording.mixture_corner(mix.names[1], "9 g") not in texts


@pytest.mark.skip(reason=REVIEW_FOUR)
def test_triangle_chart_grey_where_a_limit_is_broken_and_smaller_where_untested(mixture_scored):
    opt = mixture_scored
    mix, preds = _mix_parts(opt)
    spec = ui_insights.mixture_chart(opt, mix, preds, 7).to_dict()
    # opacity 1 excludes the invisible, legend-only circle layer.
    # Charts proto: not allowed is a hollow ring in the neutral token (3.2:1
    # on white; lightgray was 1.5:1), the score in the score colours.
    [grey] = [l for l in spec["layer"] if l["mark"]["type"] == "point"
              and l["mark"].get("filled") is False and "size" in l["encoding"]]
    [coloured] = [l for l in spec["layer"]
                  if l["mark"]["type"] == "circle" and l["mark"].get("opacity") == 1]
    assert grey["mark"]["color"] == ui_insights.mark("neutral")[0]
    assert [t["title"] for t in grey["encoding"]["tooltip"]][-1] == wording.MIXTURE_BLOCKED
    assert coloured["encoding"]["color"]["scale"]["range"] == ui_insights.score_colours()
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


@pytest.mark.skip(reason=REVIEW_FOUR)
def test_the_triangle_is_not_squashed_by_streamlits_default_autosize(mixture_scored):
    """Streamlit sets autosize: fit/padding when a spec declares none, which
    lets the legends eat into the 380 px width. Pinning autosize to "pad"
    keeps the plot itself at its full, undistorted size."""
    opt = mixture_scored
    mix, preds = _mix_parts(opt)
    spec = ui_insights.mixture_chart(opt, mix, preds, 7).to_dict()
    assert max(spec["width"], spec["height"]) == ui_insights._MIX_SIZE
    assert spec["autosize"]["type"] == "pad"


@pytest.mark.skip(reason=REVIEW_FOUR)
def test_the_corner_and_start_labels_are_grey_not_black(mixture_scored):
    """Streamlit's theme sets no default text colour, so an uncoloured
    mark_text renders black — invisible in dark mode."""
    opt = mixture_scored
    mix, preds = _mix_parts(opt)
    spec = ui_insights.mixture_chart(opt, mix, preds, 7).to_dict()
    texts = [l for l in spec["layer"] if _mark_type(l) == "text" and not l["mark"].get("stroke")]
    assert len(texts) == 4          # the three corners plus the start's own label
    assert all(l["mark"].get("color") == ui_insights.muted() for l in texts)


@pytest.mark.skip(reason=REVIEW_FOUR)
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
    assert fill_scale["range"][2] == ui_insights.mark("neutral")[0]     # charts proto
    assert fill_scale["range"][0] == fill_scale["range"][1] != fill_scale["range"][2]
    assert dummy["encoding"]["fill"]["legend"]["labelLimit"] == 0


@pytest.mark.skip(reason=REVIEW_FOUR)
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


@pytest.mark.skip(reason=REVIEW_FOUR)
def test_each_label_grows_away_from_the_region(mixture_scored):
    """A label right of the region's centre grows rightward (align left), one
    left of it grows leftward, one above or below it is centred: the three
    never grow into the grid or into each other."""
    opt = mixture_scored
    mix, preds = _mix_parts(opt)
    spec = ui_insights.mixture_chart(opt, mix, preds, 7).to_dict()
    cx = sum(ui_insights._xy(p.x)[0] for p in mix.points) / len(mix.points)
    for text in _corner_texts(mix):
        layer = next(l for l in spec["layer"] if _mark_type(l) == "text"
                     and any(r.get("t") == text for r in _layer_rows(spec, l)))
        [row] = _layer_rows(spec, layer)
        align = layer["mark"]["align"]
        if align == "left":
            assert row["X"] > cx
        elif align == "right":
            assert row["X"] < cx


def test_exact_wave_c_strings_for_the_triangle():
    assert wording.Q_MIXTURE == "Three ingredients at a fixed sum"
    assert wording.mixture_caption(14) == (
        "Colour: predicted Overall score (model mean) as the three amounts vary at a fixed "
        "sum; all other amounts as in Formulation 14.")
    assert wording.mixture_corner("Coconut oil", "9.4 g") == "Coconut oil 9.4 g"


def test_the_triangle_strings_pass_the_word_guards():
    _assert_clean(wording.Q_MIXTURE, wording.MIXTURE_PICK, wording.PICK_THREE,
                  wording.mixture_caption(14), wording.mixture_corner("Coconut oil", "9.4 g"),
                  wording.mixture_fills("Water"), wording.mixture_no_room(14),
                  wording.mixture_fill_follows("Hydration water", "Remaining water"),
                  wording.mixture_together("g"), wording.MIXTURE_TESTED,
                  wording.MIXTURE_UNTESTED, wording.MIXTURE_BLOCKED)


# ------------------------------------------------------------------ #
#  Task 5: "How far did each prediction miss?"
# ------------------------------------------------------------------ #

def _checks(opt):
    return insights.check_rounds(opt)


# ------------------------------------------------------------------ #
#  On wave B's final code: the triangle's default three read the moved
#  spreads, and every wave C line is in the project's own words.
# ------------------------------------------------------------------ #

_OFFERED = ["Pea protein", "Oil", "Starch", "Water"]


def _three_with(monkeypatch, found):
    def fake(opt, fitted, start):
        if isinstance(found, Exception):
            raise found
        return found
    monkeypatch.setattr(insights, "spreads", fake)
    return ui_insights._three_default(None, {}, _OFFERED, {})


@pytest.mark.skip(reason=REVIEW_FOUR)
def test_the_default_three_are_the_largest_moved_spreads(monkeypatch):
    assert _three_with(monkeypatch, insights.Spreads(
        [insights.Spread("Water", 9.0), insights.Spread("Starch", 5.0),
         insights.Spread("Oil", 2.0), insights.Spread("Pea protein", 1.0)], [])) == [
        "Water", "Starch", "Oil"]


@pytest.mark.skip(reason=REVIEW_FOUR)
def test_a_choice_the_limits_keep_in_place_is_not_ranked(monkeypatch):
    """A held choice has no spread: it is only a top-up, in offered order."""
    assert _three_with(monkeypatch, insights.Spreads(
        [insights.Spread("Water", 9.0), insights.Spread("Starch", 1.0)],
        ["Pea protein", "Oil"])) == ["Water", "Starch", "Pea protein"]


@pytest.mark.skip(reason=REVIEW_FOUR)
def test_a_moved_spread_the_triangle_does_not_offer_is_skipped(monkeypatch):
    assert _three_with(monkeypatch, insights.Spreads(
        [insights.Spread("Mixing time", 20.0), insights.Spread("Oil", 3.0)], [])) == [
        "Oil", "Pea protein", "Starch"]


@pytest.mark.skip(reason=REVIEW_FOUR)
def test_no_score_model_or_a_failure_opens_on_the_first_three(monkeypatch):
    assert _three_with(monkeypatch, insights.Spreads([], [])) == ["Pea protein", "Oil", "Starch"]
    assert _three_with(monkeypatch, RuntimeError("no paths")) == ["Pea protein", "Oil", "Starch"]


@pytest.mark.skip(reason=REVIEW_FOUR)
def test_the_triangle_opens_on_the_real_moved_spreads(mixture_scored):
    at = _open_insights(_app())
    assert not at.exception
    fitted = at.session_state[ui_insights.FITTED_KEY][1]
    best = mixture_scored.best_index()
    found = insights.spreads(mixture_scored, fitted,
                             insights.start_recipe(mixture_scored, best))
    offered = insights.mixture_choices(mixture_scored)
    ranked = [s.name for s in found.moved if s.name in offered]
    want = _most_room(mixture_scored, ranked + [n for n in offered if n not in ranked])
    three = next(b for b in _more(at).multiselect if b.label == wording.MIXTURE_PICK)
    assert three.value == want


@pytest.mark.skip(reason=REVIEW_FOUR)
def test_more_charts_lists_its_two_questions_in_order(mixture_scored):
    """The misses chart went (spec 2026-09-30 §2.5): measured against
    predicted, under "Show each prediction", says the same."""
    at = _open_insights(_app())
    md = [m.value for m in _more(at).markdown if m.value.startswith("#####")]
    assert md == [f"##### {q}" for q in (wording.Q_PAIR, wording.Q_MIXTURE)]


@pytest.mark.skip(reason=REVIEW_FOUR)
def test_more_charts_is_offered_when_any_one_chart_is(monkeypatch):
    """Charts proto: the fold holds the two-measurement chart alone (the
    triangle moved to Explored and unexplored), so two measurements open it."""
    opt = type("O", (), {"complete_count": lambda self: insights.MIN_ROWS})()
    fitted = {insights.SCORE: type("F", (), {"model": object()})()}
    for pair, three, offered in ((0, 0, False), (2, 0, True), (1, 3, False), (1, 2, False)):
        opt.objectives = [{"name": f"M{k}"} for k in range(pair)]
        monkeypatch.setattr(insights, "mixture_choices", lambda o, n=three: ["x"] * n)
        assert ui_insights.more_charts_offered(
            opt, {"fitted": fitted, "checks": []}) is offered, (pair, three)


@pytest.fixture
def ferment_scored(tmp_path, monkeypatch):
    """Process settings alone, two measurements, twelve trials over four
    Rounds with their results: "Can you have both?" passes its gate."""
    monkeypatch.chdir(tmp_path)
    opt = FoodOptimizer("ferment")
    opt.add_process_parameter("Incubation temperature", 30, 42, unit="°C")
    opt.add_process_parameter("Incubation time", 4, 16, unit="h")
    opt.add_objective("Acidity", 1.0, goal="target", target=4.5, min_val=3.0, max_val=7.0,
                      unit="pH")
    opt.add_objective("Thickness", 1.0, goal="max", min_val=0, max_val=10, unit="/10")
    g = torch.Generator().manual_seed(7)
    for no in range(1, 13):
        u = torch.rand(2, generator=g).tolist()
        temp, hours = 30 + 12 * u[0], 4 + 12 * u[1]
        opt.tell({"Incubation temperature": temp, "Incubation time": hours},
                 {"Acidity": 3.0 + 0.08 * temp, "Thickness": 1 + 0.5 * hours},
                 formulation_no=no, batch_no=1 + (no - 1) // 3)
    return opt


@pytest.mark.skip(reason=REVIEW_FOUR)
def test_more_charts_reads_trial_on_a_process_only_project(ferment_scored):
    at = _open_insights(_app())
    assert not at.exception
    more = _more(at)
    assert more is not None
    said = ([m.value for m in more.markdown] + [c.value for c in more.caption]
            + [more.label] + [b.label for b in more.multiselect]
            + [str(o) for b in more.multiselect for o in b.options]
            + [json.dumps(c) for c in _charts(more)])
    assert not [t for t in said if "ormulation" in t], [t for t in said if "ormulation" in t]
    assert not [t for t in said if "ngredient" in t], [t for t in said if "ngredient" in t]
    md = [m.value for m in more.markdown]
    assert md == [f"##### {wording.Q_PAIR}"]
    caps = [t for c in more.caption for t in (c.value, c.proto.help) if t]
    assert ("Measured values, one dot per trial; the line marks Acidity's target, and the "
            "other axis title gives the better direction.") in caps
    [pair] = _charts(more)
    assert {"field": "no", "title": wording.TRIAL_CAP, "type": "quantitative"} in [
        t for layer in pair["layer"] for t in layer["encoding"].get("tooltip", [])]


@pytest.mark.skip(reason=REVIEW_FOUR)
def test_no_trial_has_both_values_reads_trial(ferment_scored, monkeypatch):
    monkeypatch.setattr(insights, "pair_points", lambda opt, a, b: [])
    at = _open_insights(_app())
    assert "No trial has a value for both yet." in [t for c in _more(at).caption for t in (c.value, c.proto.help)]


@pytest.mark.skip(reason=REVIEW_FOUR)
def test_the_triangle_lines_go_through_the_projects_own_words(mixture_scored, monkeypatch):
    """The triangle needs ingredients, so it never shows on a project of
    settings alone; its lines still go through the same wording pass."""
    said = []
    real = wording.for_project
    monkeypatch.setattr(wording, "for_project", lambda opt, text: said.append(text)
                        or real(opt, text))
    _open_insights(_app())
    best_no = mixture_scored.best_formulation_no()
    for line in (wording.PAIR_FOLD, wording.Q_MIXTURE, wording.MIXTURE_PICK,
                 wording.mixture_caption(best_no), wording.MIXTURE_TESTED,
                 wording.MIXTURE_UNTESTED, wording.MIXTURE_BLOCKED,
                 wording.Q_PAIR, wording.PAIR_PICK,
                 wording.pair_caption(["Firmness"]), wording.PAIR_BEST):
        assert line in said, line


# ------------------------------------------------------------------ #
#  Final fix wave (waves B + C)
# ------------------------------------------------------------------ #

def _layer_rows(spec, layer):
    return spec["datasets"][layer["data"]["name"]]


@pytest.mark.skip(reason=REVIEW_FOUR)
def test_each_label_is_the_largest_amount_a_grid_point_shows(mixture_scored):
    """Review fixes §5.1: each label is that one's largest amount inside
    the drawn region, the most any grid point's tooltip shows for it, and
    never past its own Highest (the 0.8.0 build said "Beet juice powder
    7.2 g" with a Highest of 1.2 g)."""
    opt = mixture_scored
    mix, preds = _mix_parts(opt)
    spec = ui_insights.mixture_chart(opt, mix, preds, 7).to_dict()
    frame, _, _ = ui_insights.mixture_frame(opt, mix, preds)
    labels = [r["t"] for rows in spec["datasets"].values() for r in rows if "t" in r]
    corner_texts = _corner_texts(mix)
    highest = {v['name']: float(v['bounds'][1]) for v in opt.variables}
    for k, name in enumerate(mix.names):
        [label] = [t for t in labels if t.startswith(name + " ")]
        assert label == corner_texts[k]
        shown = float(label[len(name) + 1:].split()[0])
        # The grid reaches it to within one step of the grid.
        assert -0.051 <= shown - frame[f"a{k}"].max() <= mix.span / mix.steps + 0.051
        if name != opt.balance_row_name():
            assert shown <= highest[name] + 0.051
        assert "up to" not in label


@pytest.mark.skip(reason=REVIEW_FOUR)
def test_the_triangle_colour_legend_title_is_never_cut_short(mixture_scored):
    opt = mixture_scored
    mix, preds = _mix_parts(opt)
    spec = ui_insights.mixture_chart(opt, mix, preds, 7).to_dict()
    grid = next(l for l in spec["layer"]
                if _mark_type(l) == "circle" and "color" in l["encoding"])
    assert grid["encoding"]["color"]["legend"]["titleLimit"] == 0


def _assert_offset_not_haloed(spec, label_layer):
    """No second, white-stroked layer duplicates the label (a halo glows in
    dark mode and doubles the text for screen readers): a plain grey label,
    moved clear of the mark it names by a `dy` offset, stands alone."""
    assert not label_layer["mark"].get("stroke")
    assert label_layer["mark"].get("dy")
    same_data = [l for l in spec["layer"] if _mark_type(l) == "text"
                 and l["data"] == label_layer["data"]]
    assert same_data == [label_layer]


@pytest.mark.skip(reason=REVIEW_FOUR)
def test_the_best_so_far_label_is_offset_not_haloed(tmp_path, monkeypatch):
    opt = _pair_opt(tmp_path, monkeypatch)
    spec = ui_insights.pair_chart(opt, "Firmness", "Cook loss",
                                  insights.pair_points(opt, "Firmness", "Cook loss")).to_dict()
    [label] = _layers(spec, "text")
    assert label["mark"]["color"] == ui_insights.muted()
    assert label["encoding"]["text"] == {"value": wording.PAIR_BEST}
    _assert_offset_not_haloed(spec, label)


@pytest.mark.skip(reason=REVIEW_FOUR)
def test_the_start_ring_on_screen_carries_its_own_prediction(mixture_scored):
    """Streamlit sends a chart's rows apart from its spec, so the start's
    prediction is read where the triangle keeps it, beside its grid."""
    at = _open_insights(_app())
    assert not at.exception
    assert _triangles(_more(at))
    mix, preds, got, _ = at.session_state[ui_insights.MIXTURE_KEY][1]
    opt = mixture_scored
    fitted = at.session_state[ui_insights.FITTED_KEY][1]
    start = insights.start_recipe(opt, opt.best_index())
    [own] = insights.predict(opt, fitted[insights.SCORE], [opt._encode(start)])
    assert (got.mean, got.low, got.high) == (own.mean, own.low, own.high)
    spec = ui_insights.mixture_chart(opt, mix, preds, 7, got).to_dict()
    ring = next(l for l in spec["layer"]
                if _mark_type(l) == "point" and l["mark"].get("size") == 260)
    [row] = spec["datasets"][ring["data"]["name"]]
    assert row["pred"] == ui_insights.score_text(own)


@pytest.mark.skip(reason=REVIEW_FOUR)
def test_the_default_three_have_room_from_a_start_at_its_lowests(mixture_scored, monkeypatch):
    """Pea protein, Oil and Starch all at their Lowest: the three largest
    spreads cannot trade places from there, so the default is a
    combination that can, the one with the most room."""
    opt = mixture_scored
    monkeypatch.setattr(insights, "spreads", lambda o, fitted, start: insights.Spreads([
        insights.Spread("Pea protein", 9.0), insights.Spread("Oil", 5.0),
        insights.Spread("Starch", 2.0)], []))
    start = opt.fill_formulas({"Pea protein": 10.0, "Oil": 2.0, "Starch": 1.0})
    offered = insights.mixture_choices(opt)
    assert insights.mixture_slice(opt, ["Pea protein", "Oil", "Starch"], start).refusal == \
        insights.NO_ROOM
    got = ui_insights._three_default(opt, {}, offered, start)
    assert "Water" in got
    shares = {tuple(c): insights.allowed_share(opt, list(c), start)
              for c in itertools.combinations(offered, 3)}
    assert shares[tuple(got)] == max(shares.values())
    mix = insights.mixture_slice(opt, got, start)
    assert mix.refusal is None and not mix.start_breaks


@pytest.mark.skip(reason=REVIEW_FOUR)
def test_with_no_triple_with_room_the_default_is_as_before(mixture_scored, monkeypatch):
    opt = mixture_scored
    monkeypatch.setattr(insights, "spreads", lambda o, fitted, start: insights.Spreads([
        insights.Spread("Starch", 9.0)], []))
    real = insights.mixture_slice

    def no_room(o, names, start, steps=insights.GRID_STEPS):
        return insights.Mixture(**{**real(o, names, start, steps).__dict__,
                                   "refusal": insights.NO_ROOM})
    monkeypatch.setattr(insights, "mixture_slice", no_room)
    start = insights.start_recipe(opt, opt.best_index())
    got = ui_insights._three_default(opt, {}, insights.mixture_choices(opt), start)
    assert got == ["Starch", "Pea protein", "Oil"]


@pytest.mark.skip(reason=REVIEW_FOUR)
def test_the_default_three_search_is_capped(mixture_scored, monkeypatch):
    opt = mixture_scored
    calls = []
    monkeypatch.setattr(insights, "spreads", lambda o, fitted, start: insights.Spreads([], []))
    monkeypatch.setattr(insights, "mixture_slice",
                        lambda *a, **k: calls.append(1) or (_ for _ in ()).throw(RuntimeError()))
    offered = [f"R{k}" for k in range(10)]                 # 120 combinations
    got = ui_insights._three_default(opt, {}, offered, {})
    assert got == offered[:3]
    assert len(calls) == ui_insights._THREE_TRIES == 20


def _pair_caption_for(opt, a, b):
    return ui_insights.pair_caption(opt, a, b)


@pytest.mark.skip(reason=REVIEW_FOUR)
def test_the_pair_caption_says_which_lines_are_drawn(tmp_path, monkeypatch):
    opt = _pair_opt(tmp_path, monkeypatch)
    assert _pair_caption_for(opt, "Firmness", "Cook loss") == (
        "Measured values, one dot per formulation; the line marks Firmness's target, and the "
        "other axis title gives the better direction.")
    assert _pair_caption_for(opt, "Cook loss", "Juiciness") == (
        "Measured values, one dot per formulation; the axis titles give the better direction. "
        "No dot good on both suggests a trade-off.")
    opt.add_objective("Colour", 10, goal="target", target=4, min_val=0, max_val=10)
    assert _pair_caption_for(opt, "Firmness", "Colour") == wording.PAIR_CAPTION


def test_the_pair_caption_variants_pass_the_word_guards():
    _assert_clean(wording.pair_caption([]), wording.pair_caption(["Firmness"]),
                  wording.pair_caption(["Firmness", "Colour"]), wording.PAIR_WAITING)
    assert wording.pair_caption(["Firmness", "Colour"]) == wording.PAIR_CAPTION


@pytest.mark.skip(reason=REVIEW_FOUR)
def test_waiting_dots_are_named_in_the_legend_only_when_there_are_some(tmp_path, monkeypatch):
    opt = _pair_opt(tmp_path, monkeypatch)
    points = insights.pair_points(opt, "Firmness", "Cook loss")
    spec = ui_insights.pair_chart(opt, "Firmness", "Cook loss", points).to_dict()
    [hollow] = [l for l in _layers(spec, "point") if l["mark"].get("size") == 60]
    stroke = hollow["encoding"]["stroke"]
    assert stroke["scale"] == {"domain": [wording.PAIR_WAITING],
                               "range": [ui_insights.mark("neutral")[0]]}   # D1 token
    assert stroke["legend"]["title"] is None
    done = [p for p in points if p.score is not None]
    spec = ui_insights.pair_chart(opt, "Firmness", "Cook loss", done).to_dict()
    assert not [l for l in spec["layer"] if "stroke" in l.get("encoding", {})]   # no legend entry


@pytest.mark.skip(reason=REVIEW_FOUR)
def test_the_pair_tooltip_titles_go_through_the_projects_own_words(tmp_path, monkeypatch):
    opt = _pair_opt(tmp_path, monkeypatch)
    said = []
    real = wording.for_project
    monkeypatch.setattr(wording, "for_project", lambda o, text: said.append(text) or real(o, text))
    ui_insights.pair_chart(opt, "Firmness", "Cook loss",
                           insights.pair_points(opt, "Firmness", "Cook loss"))
    assert "Firmness (N)" in said and "Cook loss (%)" in said
