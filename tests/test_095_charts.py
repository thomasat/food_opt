"""0.8.0 BO views, Task 5 (spec 2026-09-30-bo-views §2.2, §3.2, §3.3;
review I4, I6, I8, M7, M8, M11, M14): the triangle and the one-ingredient
path gain the suggestion model's Uncertainty and Expected gain layers
(through a suggestion still to make, by default), the Predicted score layer
outlines what is predicted above Best so far and says when its colours are
within the model's own uncertainty, and the Progress chart fold shows the
expected improvement by Round."""
import json

import pytest
from streamlit.testing.v1 import AppTest

import bo_views
import insights
import ui_batch
import ui_insights
import wording
from tests.test_app_ui import APP_PATH
from tests.test_insights import _burger


@pytest.fixture(autouse=True)
def _a_model_with_skill(monkeypatch):
    """These tests are about the effect lines, so they stand in a model
    whose Round-ahead skill lets them be said: the 0.8.0 e2e fixes' A3 gate
    (review I4) withholds them below 10 checks or skill 0.20, which these
    small projects are. tests/test_102_w1_insights.py covers the gate."""
    monkeypatch.setattr(insights, "skill_gate", lambda opt, checks, fact=None: None)



@pytest.fixture(autouse=True)
def _isolate(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    with bo_views.fresh():
        yield


def _open_round():
    opt = _burger(rounds=4)
    ui_batch.ask_alone(opt, n_suggestions=3)
    return opt


def _insights(opt, **state):
    at = AppTest.from_file(APP_PATH, default_timeout=300)
    at.session_state["_loaded_project"] = opt.project_name
    at.run()
    for key, value in state.items():
        at.session_state[key] = value
    at.session_state["main_tab"] = wording.TAB_INSIGHTS
    at.run()
    assert not at.exception
    return at


def _specs(node):
    return [json.loads(c.proto.spec) for c in node.children.values()
            if getattr(c, "type", "") == "vega_lite_chart"]


def _more(at):
    return next(e for e in at.tabs[3].expander if e.label == wording.MORE_CHARTS)


def _try_one(at):
    return next(e for e in at.tabs[3].expander if e.label.startswith("Effect of one"))


def _all_layers(spec):
    out, todo = [], [spec]
    while todo:
        s = todo.pop()
        out.append(s)
        for key in ("layer", "vconcat", "hconcat", "concat"):
            todo.extend(s.get(key, []))
        if "spec" in s:
            todo.append(s["spec"])
    return out


# ------------------------------------------------------------------ #
#  The layers (review I6, I8)
# ------------------------------------------------------------------ #

def test_one_radio_offers_the_three_layers():
    opt = _open_round()
    assert ui_insights.layers_offered(opt, insights.fit_all(opt)) == (
        wording.LAYER_SCORE, wording.LAYER_SD, wording.LAYER_GAIN)
    at = _insights(opt, insights_try_one=True)
    radios = [r for r in at.tabs[3].radio if r.key in (ui_insights.LAYER_KEY,
                                                       ui_insights.PATH_LAYER_KEY)]
    assert {r.key for r in radios} == {ui_insights.LAYER_KEY, ui_insights.PATH_LAYER_KEY}
    assert all(list(r.options) == [wording.LAYER_SCORE, wording.LAYER_SD, wording.LAYER_GAIN]
               and r.value == wording.LAYER_SCORE for r in radios)


def test_the_guard_takes_the_model_layers_away(monkeypatch):
    """Review I4: the model-free parts stay."""
    opt = _open_round()
    monkeypatch.setattr(insights, "noise_guard", lambda *a, **k: insights.Guard(["floor"]))
    assert ui_insights.layers_offered(opt, insights.fit_all(opt)) == (wording.LAYER_SCORE,)
    at = _insights(opt, insights_try_one=True)
    assert not [r for r in at.tabs[3].radio if r.key in (ui_insights.LAYER_KEY,
                                                         ui_insights.PATH_LAYER_KEY)]


def test_an_unreliable_verdict_takes_the_expected_improvement_layer_away(monkeypatch):
    """Fix round 1: as Progress withholds its expected improvement. The
    Uncertainty layer stays, as the explored–unexplored map does (spec
    2026-10-01 rev. 2, review R-I16, which overrides e2e I4 for that layer
    only; test_107_w3_explored)."""
    opt = _open_round()
    fitted = insights.fit_all(opt)
    monkeypatch.setattr(ui_insights, "checks_for", lambda o, stage: insights.check_rounds(o))
    monkeypatch.setattr(insights, "verdict_state", lambda *a: "unreliable")
    assert ui_insights.layers_offered(opt, fitted) == (wording.LAYER_SCORE, wording.LAYER_SD)


def test_the_uncertainty_layer_goes_through_the_first_suggestion():
    """Review I8: its default anchor is the open Round's first row still to
    make, named in the caption; the Predicted score layer keeps Best so far."""
    opt = _open_round()
    first = ui_insights.suggested_numbers(opt)[0]
    at = _insights(opt, **{ui_insights.LAYER_KEY: wording.LAYER_SD})
    more = _more(at)
    titles = [l["encoding"]["color"]["title"] for s in _specs(more) for l in _all_layers(s)
              if l.get("encoding", {}).get("color", {}).get("field") == "score"
              and "scheme" in l["encoding"]["color"].get("scale", {})
              and l["encoding"]["color"]["title"] != wording.OVERALL_SCORE_COLUMN]
    assert titles == [wording.LAYER_SD_LEGEND]
    caps = [c.value for c in more.caption]
    assert wording.layer_caption(wording.LAYER_SD, wording.start_suggested(first)) in caps
    # The score layer's start is Best so far (T56 I6: was vacuous).
    assert ui_insights._start_number(opt) == int(opt.formulation_ids[opt.best_index()])


def test_the_layer_values_are_the_suggestion_models_own():
    opt = _open_round()
    fitted = insights.fit_all(opt)
    recipes = [insights.start_recipe(opt, i) for i in range(3)]
    sd = ui_insights.layer_values(opt, fitted, recipes, wording.LAYER_SD)
    gain = ui_insights.layer_values(opt, fitted, recipes, wording.LAYER_GAIN)
    model = ui_insights.layer_model(opt, fitted)
    want = bo_views.evaluate_here(model, bo_views.points(opt, recipes))
    assert sd == pytest.approx(want[1].tolist()) and gain == pytest.approx(want[2].tolist())


def test_the_path_layer_starts_from_a_suggestion_and_says_so():
    opt = _open_round()
    first = ui_insights.suggested_numbers(opt)[0]
    at = _insights(opt, insights_try_one=True,
                   **{ui_insights.PATH_LAYER_KEY: wording.LAYER_GAIN})
    fold = _try_one(at)
    start = next(s for s in fold.selectbox if s.key == ui_insights.LAYER_START_KEY)
    assert start.value == first
    assert start.format_func(first) == wording.start_suggested(first)
    [spec] = _specs(fold)
    ys = [l["encoding"]["y"]["title"] for l in _all_layers(spec)
          if l.get("encoding", {}).get("y", {}).get("field") == "v"]
    assert ys == [wording.LAYER_GAIN_LEGEND]
    assert wording.path_layer_caption(wording.LAYER_GAIN, wording.start_suggested(first)) in [
        c.value for c in fold.caption]


def test_the_score_path_names_its_prediction_as_the_current_models():
    opt = _burger(rounds=3)
    fitted = insights.fit_all(opt)
    path = insights.build_path(opt, "Oil", insights.start_recipe(opt, opt.best_index()))
    spec = ui_insights.path_chart(opt, path, insights.path_predictions(opt, fitted, path)).to_dict()
    titles = {t["title"] for l in _all_layers(spec) for t in l.get("encoding", {}).get("tooltip", [])
              if isinstance(t, dict)}
    assert wording.PREDICTED in titles and wording.TRACK_PREDICTED not in titles


# ------------------------------------------------------------------ #
#  The Predicted score layer: the outline and the colour span (I6, M7, M14)
# ------------------------------------------------------------------ #

@pytest.mark.parametrize("low,mean,ref,top,yard,outlined", [
    (91.0, 94.0, 90.0, 92.0, 1.0, True),      # whole interval above, clearly higher as shown
    (89.0, 94.0, 90.0, 92.0, 1.0, False),     # mean above, interval not (T56 I6)
    (90.5, 90.8, 90.0, 89.0, 1.0, False),     # interval above, gain under the threshold
    (91.6, 92.4, 90.0, 92.5, 1.0, False),     # as shown: 92 against 92.50 (Timing check)
    (91.0, 94.0, None, 92.0, 1.0, False),     # no Repeat of Best so far
])
def test_the_outline_rule_is_best_standings(low, mean, ref, top, yard, outlined):
    """T56 I1: the outline marks only what Progress would call higher."""
    assert insights.above_best(low, mean, ref, top, yard) is outlined


def test_the_outline_is_drawn_where_the_rule_holds(monkeypatch):
    opt = _open_round()
    fitted = insights.fit_all(opt)
    mix = insights.mixture_slice(opt, ["Pea", "Oil", "Water"],
                                 insights.start_recipe(opt, opt.best_index()))
    preds = insights.mixture_predictions(opt, fitted, mix)
    seen = []
    monkeypatch.setattr(insights, "above_best", lambda *a: seen.append(a) or True)
    extra = ui_insights._mixture_extra(opt, fitted, mix, preds, wording.LAYER_SCORE)
    allowed = [p.recipe is not None for p in mix.points]
    assert extra["outline"] == allowed
    # Called with the display model's own predicted score at each point.
    assert [a[1] for a in seen] == [p.mean for p in preds if p is not None]
    spec = ui_insights.mixture_chart(opt, mix, preds, 3, None, None, None,
                                     extra["outline"]).to_dict()
    assert any(l.get("mark", {}).get("stroke") == ui_insights.outline_colour()
               and l["mark"].get("filled") is False for l in _all_layers(spec))
    monkeypatch.setattr(insights, "above_best", lambda *a: False)
    extra = ui_insights._mixture_extra(opt, fitted, mix, preds, wording.LAYER_SCORE)
    assert not any(extra["outline"])


def test_the_timing_check_outlines_nothing_while_progress_says_none():
    """T56 I1's scenario: Coconut oil / Textured soy protein / Water from
    Best so far; Progress says no predicted gain beyond the noise
    threshold, so nothing is outlined."""
    from tests import demo_projects as demo
    opt = demo.load(demo.TIMING)
    fitted = insights.fit_all(opt)
    checks = insights.check_rounds(opt)
    best = insights.predicted_best(opt, fitted, checks)
    yard = insights.noise_yardstick(opt, checks).value
    assert not insights.best_standing(opt, best, yard).higher
    start = insights.start_recipe(opt, opt.best_index())
    mix = insights.mixture_slice(opt, ["Coconut oil", "Textured soy protein", "Water"], start)
    preds = insights.mixture_predictions(opt, fitted, mix)
    extra = ui_insights._mixture_extra(opt, fitted, mix, preds, wording.LAYER_SCORE)
    assert extra["outline"] and not any(extra["outline"])


def test_the_caption_says_when_colours_are_within_the_models_uncertainty():
    """Review M7: the span of the colours against the median 1.96 × the
    model's own sd over the allowed points; a caption, never a rescale."""
    opt = _open_round()
    fitted = insights.fit_all(opt)
    mix = insights.mixture_slice(opt, ["Pea", "Oil", "Water"],
                                 insights.start_recipe(opt, opt.best_index()))
    preds = insights.mixture_predictions(opt, fitted, mix)
    extra = ui_insights._mixture_extra(opt, fitted, mix, preds, wording.LAYER_SCORE)
    means = [p.mean for p in preds if p is not None]
    assert extra["span"] == pytest.approx(max(means) - min(means))
    model = ui_insights.layer_model(opt, fitted)
    got = bo_views.evaluate(opt, model, bo_views.points(
        opt, [p.recipe for p in mix.points if p.recipe is not None]))
    halves = sorted((insights.Z95 * got.sd).tolist())
    assert extra["half"] == pytest.approx(halves[len(halves) // 2])


@pytest.mark.parametrize("line,said", [("B", True), ("A", False)])
def test_the_within_noise_caption_on_the_demo_lines(line, said):
    """T56 I6: said on Line B (span about 4.1 < 6.8), not on Line A (10.7 >
    4.0), on each one's default three from Best so far."""
    from tests import demo_projects as demo
    opt = demo.load(getattr(demo, f"LINE_{line}"))
    fitted = insights.fit_all(opt)
    start = insights.start_recipe(opt, opt.best_index())
    names = ui_insights._three_default(opt, fitted, insights.mixture_choices(opt), start)
    mix = insights.mixture_slice(opt, names, start)
    preds = insights.mixture_predictions(opt, fitted, mix)
    extra = ui_insights._mixture_extra(opt, fitted, mix, preds, wording.LAYER_SCORE)
    assert (extra["span"] < extra["half"]) is said
    assert bool(ui_insights.within_noise_line(extra)) is said


# ------------------------------------------------------------------ #
#  §3.2 Expected improvement by Round in the Progress chart fold (M8, M11)
# ------------------------------------------------------------------ #

def _progress(at):
    return next(e for e in at.tabs[3].expander if e.label == wording.SHOW_PROGRESS)


def _again(at):
    """One more run, as opening the fold makes: the fold is drawn before the
    tab works its checks out, and draws from them once they are."""
    at.session_state["main_tab"] = wording.TAB_INSIGHTS
    at.run()
    assert not at.exception
    return at


def test_the_progress_fold_is_worked_out_only_when_open(monkeypatch):
    opt = _open_round()
    calls = []
    monkeypatch.setattr(bo_views, "round_history", lambda *a, **k: calls.append(1) or [])
    at = _insights(opt)
    assert calls == [] and not _specs(_progress(at))


def test_the_fold_shows_the_expected_improvement_by_round():
    opt = _open_round()
    at = _again(_insights(opt, **{ui_insights.PROGRESS_KEY: True}))
    fold = _progress(at)
    specs = _specs(fold)
    assert len(specs) == 2                           # Best so far, then by Round
    assert any(m.value.endswith(wording.GAIN_HEADING) or wording.GAIN_HEADING in m.value
               for m in fold.markdown)
    assert wording.GAIN_CHART_CAPTION in [c.value for c in fold.caption]
    bars = [l for l in _all_layers(specs[1]) if l.get("mark", {}).get("type") == "bar"
            or l.get("mark") == "bar"]
    assert bars
    # Rounds made after 5 rows (Round 2 had 3 before it), then the open one.
    rounds = [g.round for g in bo_views.round_history(opt)]
    assert rounds == [3, 4, opt.pending_batch_no]


def test_under_the_guard_the_fold_keeps_best_so_far_only(monkeypatch):
    """T56 I4: no heading, chart or caption for a withheld figure; Best so
    far is charted once, above."""
    opt = _open_round()
    monkeypatch.setattr(insights, "noise_guard", lambda *a, **k: insights.Guard(["noise"]))
    at = _again(_insights(opt, **{ui_insights.PROGRESS_KEY: True}))
    fold = _progress(at)
    assert len(_specs(fold)) == 1
    assert not any(wording.GAIN_HEADING in m.value for m in fold.markdown)
    assert wording.GAIN_CHART_CAPTION not in [c.value for c in fold.caption]


def test_the_two_panels_share_one_round_axis():
    """T56 m1: a Round with no bar keeps its place under its Round."""
    history = [bo_views.RoundGain(1, 80.0, None, None, False, "own"),
               bo_views.RoundGain(2, 80.0, 81.0, 0.4, True, None)]
    spec = ui_insights.gain_chart(_burger(rounds=2), history).to_dict()
    assert spec["resolve"]["scale"]["x"] == "shared"


def test_qucb_offers_no_expected_gain_layer():
    """T56 I2: qUCB's value is an optimistic score, not a gain."""
    opt = _burger(rounds=4)
    opt.set_bo_config({"acquisition": "qucb"})
    ui_batch.ask_alone(opt, n_suggestions=3)
    assert ui_insights.layers_offered(opt, insights.fit_all(opt)) == (
        wording.LAYER_SCORE, wording.LAYER_SD)


def test_try_one_has_one_anchor_under_a_layer():
    """01 O10 (over T56 I3): the path follows the layers' anchor, and so do
    the answer line and Compare all: one anchor per headline."""
    opt = _open_round()
    first = ui_insights.suggested_numbers(opt)[0]
    at = _insights(opt, insights_try_one=True, **{ui_insights.PATH_LAYER_KEY: wording.LAYER_SD})
    fold = _try_one(at)
    said = [m.value for m in fold.markdown]
    assert any(wording.start_suggested(first) in m for m in said), said
    assert not any("Best so far" in m for m in said), said
    assert wording.path_layer_caption(wording.LAYER_SD, wording.start_suggested(first)) in [
        c.value for c in fold.caption]


def test_a_round_without_an_acquisition_has_no_bar():
    history = [bo_views.RoundGain(1, None, None, None, False, "cold"),
               bo_views.RoundGain(2, 80.0, 81.0, 0.4, True, None)]
    opt = _burger(rounds=2)
    spec = ui_insights.gain_chart(opt, history).to_dict()
    bars = [l for l in _all_layers(spec) if l.get("mark", {}).get("type") == "bar"]
    [bar] = bars
    frame = ui_insights.gain_frame(history)
    assert list(frame.dropna(subset=["gain"])["round"]) == [2]
    assert set(frame["how"]) == {wording.GAIN_EXACT, wording.GAIN_RECOMPUTED}
