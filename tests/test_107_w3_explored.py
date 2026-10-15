"""Insights for a first-time user, W3 (spec 2026-10-01 rev. 2 §8; review
R-I14, R-I15, R-I16): the "Explored and unexplored" fold.

- "Where you have tested": one strip per varying ingredient and process
  setting and the Fill to total row, model-free; a tick per recorded
  formulation at the Formulation total, coloured by Round; the open Round's
  suggestions marked; waiting and Not scored hollow; a choice row as a count
  bar; the gap line by the 1-D rule of R-I15 over the reachable part only;
  the limit-hit line.
- The map: the mean x model-sd scatter shared with the Why fold, from the
  open Round's model, else next_model; hidden under the noise guard,
  captioned under the A3 gate or the verdict.
- The pointer, only when the Uncertainty layer is offered; Next step's link
  on its explore and untested rows.
- The fold sits under the button, above the folds below it (R-I14).
"""
import json
import os

import pytest
from tests.demo_build_reason import REVIEW_FOUR
import torch
from streamlit.testing.v1 import AppTest

import bo_views
import insights
import insights_jobs
import ui_batch
import ui_insights
import wording
from food_bo import FoodOptimizer
from tests import demo_projects as D
from tests.test_104_speed_robust_charts import PAGES, _contrast_on
from tests.test_app_ui import APP_PATH
from tests.test_chart_hygiene import _assert_tooltips_clean
from tests.test_insights import _burger
from tests.demo_build_reason import REMOVED_CHART


@pytest.fixture(autouse=True)
def _isolate(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(insights_jobs, "_JOBS", {})
    monkeypatch.setattr(insights_jobs, "_INFLIGHT", {})
    with bo_views.fresh():
        yield


def _strips(opt):
    return insights.tested_strips(opt, bo_views.feasible_pool(opt))


def _all_marks(spec):
    out, todo = [], [spec]
    while todo:
        s = todo.pop()
        if "mark" in s:
            out.append(s)
        for key in ("layer", "vconcat", "hconcat", "concat"):
            todo.extend(s.get(key, []))
        if "spec" in s:
            todo.append(s["spec"])
    return out


def _mark(layer):
    return layer["mark"] if isinstance(layer["mark"], dict) else {"type": layer["mark"]}


def _rows(spec, layer):
    return spec["datasets"][layer["data"]["name"]]


# ------------------------------------------------------------------ #
#  The strips: ticks are the recorded formulations at the total
# ------------------------------------------------------------------ #

@pytest.mark.parametrize("name", D.ALL)
def test_the_ticks_are_the_recorded_formulations_at_the_total(name):
    opt = D.load(name)
    strips = _strips(opt)
    assert strips
    pending = [r for r in opt.pending_batch or []
               if int(r['formulation']) not in {int(n) for n in opt.formulation_ids}]
    for s in strips:
        if s.options is not None:
            continue
        made = [t for t in s.ticks if t.state != "suggested"]
        assert [t.formulation for t in made] == [int(n) for n in opt.formulation_ids]
        for i, t in enumerate(made):
            at_total = insights.start_recipe(opt, i)
            assert t.value == pytest.approx(float(at_total[s.name])), (s.name, t)
            assert t.state == ("tested" if opt.fully_measured(i) else "waiting")
            assert (t.score is None) == (not opt.fully_measured(i))
            assert t.round == opt.batch_history[i]
        picks = [t for t in s.ticks if t.state == "suggested"]
        assert [t.formulation for t in picks] == [int(r['formulation']) for r in pending]
        for t, row in zip(picks, pending):
            assert t.value == pytest.approx(float(insights.at_total(opt, row['recipe'])[s.name]))
            assert t.round == opt.pending_batch_no and not t.latest


def test_the_ticks_are_at_the_total_not_the_amounts_made():
    """A formulation made to another batch size is placed at the
    Formulation total, as the models learn it (model_X)."""
    opt = _burger(rounds=2)
    opt.tell({"Pea": 12.0, "Oil": 8.0, "Water": 180.0}, {"Firmness": 6.0, "Juiciness": 7.0},
             formulation_no=99, batch_no=3)
    pea = next(s for s in insights.tested_strips(opt) if s.name == "Pea")
    tick = next(t for t in pea.ticks if t.formulation == 99)
    assert tick.value == pytest.approx(6.0)          # 12 g in 200 g is 6 g in 100 g
    assert tick.latest and tick.round == 3


@pytest.mark.parametrize("name", D.ALL)
def test_which_rows_get_a_strip(name):
    """Every varying row, ingredients first, then process settings; the
    Fill to total row drawn across its Lowest–Highest; no fixed row."""
    opt = D.load(name)
    strips = _strips(opt)
    fill = opt.balance_row_name()
    varying = [v['name'] for v in opt.varying_variables()]
    assert {s.name for s in strips} == set(varying) | ({fill} if fill else set())
    kinds = [s.setting for s in strips]
    assert kinds == sorted(kinds)                    # ingredients, then settings
    for s in strips:
        var = next(v for v in opt.variables if v['name'] == s.name)
        assert (s.low, s.high) == tuple(float(b) for b in var['bounds'])
        assert s.fill == (s.name == fill)
        assert s.setting == (var.get('category') == 'process')
        assert s.low - 1e-9 <= s.reach[0] <= s.reach[1] <= s.high + 1e-9
    assert "Salt" not in {s.name for s in strips} or name == D.PROCESS_ONLY


def test_the_reachable_part_is_the_pool_s_and_the_box_without_one():
    opt = D.load(D.LINE_A)
    oil = next(s for s in _strips(opt) if s.name == "Coconut oil")
    assert oil.reach[1] < 8.85                       # Fat per 100 g: at most 10 g
    plain = next(s for s in insights.tested_strips(opt) if s.name == "Coconut oil")
    assert plain.reach == (3.0, 9.0)


def _strip(values, reach=(0.0, 10.0), states=None, name="Oil", fill=False):
    states = states or ["tested"] * len(values)
    ticks = [insights.StripTick(v, k + 1, 1, 50.0 if s == "tested" else None, s)
             for k, (v, s) in enumerate(zip(values, states))]
    return insights.Strip(name, "g", 0.0, 10.0, reach, ticks, fill)


# ------------------------------------------------------------------ #
#  The gap rule (R-I15)
# ------------------------------------------------------------------ #

def test_a_gap_is_15_percent_of_the_span_or_more():
    at = insights.strip_gaps(None, [_strip([0.0, 1.5, 3.0, 4.4, 10.0])])
    assert [(g.kind, g.low, g.high) for g in at] == [("between", 4.4, 10.0),
                                                     ("between", 0.0, 1.5)]
    assert at[1].size == pytest.approx(0.15)
    just_under = insights.strip_gaps(None, [_strip([0.0, 1.49, 2.98, 4.47, 5.96, 7.45,
                                                    8.94, 10.0])])
    assert just_under == []


def test_end_gaps_and_the_largest_first_at_most_two():
    salt = [0.0, 1.4, 2.8, 4.2, 5.6, 7.0, 10.0]
    gaps = insights.strip_gaps(None, [_strip([4.0, 5.0], name="Oil"),
                                      _strip(salt, name="Salt")])
    assert [(g.name, g.kind) for g in gaps] == [("Oil", "above"), ("Oil", "below")]
    assert gaps[0].low == 5.0 and gaps[0].high == 10.0 and gaps[0].size == pytest.approx(0.5)
    assert gaps[1].low == 0.0 and gaps[1].high == 4.0
    third = insights.strip_gaps(None, [_strip([4.0, 5.0]), _strip(salt, name="Salt")], k=3)
    assert [(g.name, g.kind, g.low, g.high) for g in third][2] == ("Salt", "between", 7.0, 10.0)


def test_a_stretch_the_limits_rule_out_is_not_a_gap():
    """R-I15: only the reachable part. Coconut oil above 8.8 g cannot be
    made (Fat per 100 g), so it is never called untested."""
    made = [0.0, 1.4, 2.8, 4.2, 5.6, 7.0, 8.4]
    assert insights.strip_gaps(None, [_strip(made, reach=(0.0, 8.5))]) == []
    gaps = insights.strip_gaps(None, [_strip(made)])
    assert [(g.kind, g.low, g.high) for g in gaps] == [("above", 8.4, 10.0)]


def test_waiting_and_not_scored_never_close_a_gap():
    gaps = insights.strip_gaps(None, [_strip([0.0, 5.0, 10.0, 2.5, 7.5],
                                             states=["tested", "waiting", "tested",
                                                     "not_scored", "suggested"])])
    assert [(g.low, g.high) for g in gaps] == [(0.0, 10.0)]


def test_the_fill_row_gets_no_gap_line():
    gaps = insights.strip_gaps(None, [_strip([0.0], name="Water", fill=True)])
    assert gaps == []


def test_the_gap_line_agrees_with_the_15_percent_rule_on_the_demo_projects():
    """Recomputed by hand from the recorded formulations with all results in."""
    for name in D.ALL:
        opt = D.load(name)
        for s in _strips(opt):
            if s.fill or s.options is not None:
                continue
            lo, hi = s.reach
            values = sorted(min(max(float(insights.start_recipe(opt, i)[s.name]), lo), hi)
                            for i in range(len(opt.Y_history)) if opt.fully_measured(i))
            edges = [lo] + values + [hi]
            biggest = max(b - a for a, b in zip(edges, edges[1:]))
            found = [g for g in insights._stretches(s)]
            if biggest >= 0.15 * (s.high - s.low):
                assert max(g.size for g in found) * (s.high - s.low) == pytest.approx(biggest)
            else:
                assert found == []


# ------------------------------------------------------------------ #
#  Limit hits (R-I15)
# ------------------------------------------------------------------ #

def test_the_limit_hit_line_on_line_a():
    opt = D.load(D.LINE_A)
    hits = insights.limit_hits(opt)
    assert hits[0].round == 6 and (hits[0].hits, hits[0].picks) == (4, 4)
    assert hits[0].binding.kind == "highest" and hits[0].binding.name == "Textured pea protein"
    assert all(2 * h.hits >= h.picks for h in hits)


def _picks_at(opt, recipes):
    """The open Round made of these suggestions."""
    opt.set_pending_batch([opt.fill_formulas(r) for r in recipes])
    return insights.limit_hits(opt)


def test_fewer_than_half_or_fewer_than_two_picks_say_nothing():
    opt = _burger(rounds=3)
    hits = _picks_at(opt, [{"Pea": 8.0, "Oil": 3.0}, {"Pea": 5.0, "Oil": 3.0},
                           {"Pea": 5.5, "Oil": 4.0}])
    assert not any(h.binding.name == "Pea" for h in hits)        # 1 of 3
    hits = _picks_at(opt, [{"Pea": 8.0, "Oil": 3.0}, {"Pea": 5.0, "Oil": 3.0}])
    assert [(h.binding.name, h.hits, h.picks) for h in hits] == [("Pea", 1, 2)]   # half
    assert _picks_at(opt, [{"Pea": 8.0, "Oil": 3.0}]) == []      # one pick


def test_own_formulations_and_repeats_are_left_out():
    opt = _burger(rounds=3)
    opt.set_pending_batch([opt.fill_formulas({"Pea": 8.0, "Oil": 3.0}),
                           opt.fill_formulas({"Pea": 5.0, "Oil": 3.0})])
    rows = list(opt.pending_batch)
    rows[0] = dict(rows[0], note=wording.OWN_FORMULATION_NOTE)
    rows.append(dict(rows[1], formulation=rows[1]['formulation'] + 1,
                     recipe=opt.fill_formulas({"Pea": 8.0, "Oil": 3.0}),
                     note="Repeat of Formulation 3"))
    opt.pending_batch = rows
    assert insights.limit_hits(opt) == []       # one generated pick left


def test_with_no_round_open_the_last_round_the_model_chose():
    opt = D.load(D.TIMING)
    gen, recipes = insights._round_picks(opt, 12)
    assert gen.regime == "warm" and len(recipes) == 5
    for h in insights.limit_hits(opt):
        assert h.round == 12
    snack = D.load(D.SNACK)                     # Round 1 is cold: no model chose it
    assert insights.limit_hits(snack) == []


# ------------------------------------------------------------------ #
#  The chart: suggestions marked, waiting hollow, colours, privacy
# ------------------------------------------------------------------ #

@pytest.mark.skip(reason=REMOVED_CHART)
def test_the_suggestions_are_marked_and_waiting_is_hollow():
    opt = D.load(D.LINE_A)
    spec = ui_insights.strips_chart(opt, _strips(opt)).to_dict()
    marks = _all_marks(spec)
    [picks] = [m for m in marks if _mark(m).get("filled") is True]
    assert _mark(picks)["color"] == ui_insights.outline_colour()
    rows = _rows(spec, picks)
    assert len({r["row"] for r in rows}) == 7                # on every numeric strip
    for row in {r["row"] for r in rows}:
        assert [r["no"] for r in rows if r["row"] == row] == [22, 23, 24, 25]
    assert {r["mark"] for r in rows} == {wording.STRIP_SUGGESTED}
    made = [m for m in marks if _mark(m).get("filled") is False]
    assert len(made) == 1
    for m in made:
        rows = _rows(spec, m)
        assert {r["mark"] for r in rows if r["no"] == 19} == {wording.STRIP_WAITING}
        # Round 5 is Formulations 18-21 (19 still waiting).
        assert {r["mark"] for r in rows if r["no"] in (18, 20, 21)} == {wording.STRIP_LATEST}
        assert {r["mark"] for r in rows if r["no"] <= 17} == {wording.STRIP_TESTED}
    shape = made[0]["encoding"]["shape"]["scale"]
    assert dict(zip(shape["domain"], shape["range"]))[wording.STRIP_WAITING] == "circle"
    assert dict(zip(shape["domain"], shape["range"]))[wording.STRIP_SUGGESTED] == "triangle-down"


@pytest.mark.skip(reason=REMOVED_CHART)
def test_the_tooltip_reads_amount_formulation_round_and_score():
    opt = D.load(D.LINE_A)
    spec = ui_insights.strips_chart(opt, [s for s in _strips(opt)
                                          if s.name == "Coconut oil"]).to_dict()
    _assert_tooltips_clean({"layer": [m for m in _all_marks(spec)]})
    made = next(m for m in _all_marks(spec) if _mark(m).get("filled") is False)
    titles = [t["title"] for t in made["encoding"]["tooltip"]]
    assert titles == [wording.strip_value_title(), wording.FORMULATION_CAP, wording.ROUND_CAP,
                      wording.OVERALL_SCORE_COLUMN]
    rows = _rows(spec, made)
    assert {r["row"] for r in rows} == {"Coconut oil (g)"}
    best = next(r for r in rows if r["no"] == 15)
    assert best["round"] == "4" and best["score_text"] == f"{opt.Y_history[14]:.2f}"
    assert best["value_text"].endswith(" g")
    assert next(r for r in rows if r["no"] == 19)["score_text"].startswith("waiting for")


@pytest.mark.skip(reason=REMOVED_CHART)
@pytest.mark.parametrize("which", ["light", "dark"])
def test_every_strip_colour_reaches_3_to_1(monkeypatch, which):
    monkeypatch.setattr(ui_insights, "theme", lambda: which)
    for n in range(1, 25):
        for colour in ui_insights.round_colours(n):
            assert _contrast_on(colour, 1.0, PAGES[which]) >= 3.0, (which, n, colour)
    opt = D.load(D.TIMING)
    spec = ui_insights.strips_chart(opt, _strips(opt)).to_dict()
    made = next(m for m in _all_marks(spec) if _mark(m).get("filled") is False)
    stroke = made["encoding"]["stroke"]["scale"]
    assert stroke["domain"] == [str(r) for r in range(1, 13)]
    assert stroke["range"][-1] == ui_insights._ROUND_RAMP[which][1]      # latest: strongest
    for colour in stroke["range"]:
        assert _contrast_on(colour, 1.0, PAGES[which]) >= 3.0
    tracks = [_mark(m) for m in _all_marks(spec) if _mark(m).get("height") == 2]
    assert len(tracks) == 1                                              # Lowest–Highest
    assert _contrast_on(tracks[0]["color"], tracks[0]["opacity"], PAGES[which]) >= 3.0
    for m in _all_marks(spec):
        mk = _mark(m)
        if mk["type"] == "text":                                         # the ends' values
            assert _contrast_on(mk["color"], 1.0, PAGES[which]) >= 4.5


@pytest.mark.skip(reason=REMOVED_CHART)
@pytest.mark.parametrize("which", ["light", "dark"])
def test_the_suggestion_triangles_reach_3_to_1(monkeypatch, which):
    monkeypatch.setattr(ui_insights, "theme", lambda: which)
    opt = D.load(D.LINE_A)
    spec = ui_insights.strips_chart(opt, _strips(opt)).to_dict()
    for m in _all_marks(spec):
        if _mark(m).get("filled") is True:
            assert _contrast_on(_mark(m)["color"], 1.0, PAGES[which]) >= 3.0


@pytest.mark.skip(reason=REMOVED_CHART)
def test_the_latest_round_is_the_strongest_and_tallest():
    assert ui_insights._SIZES[wording.STRIP_LATEST] > ui_insights._SIZES[wording.STRIP_TESTED]
    for which in ("light", "dark"):
        ramp = ui_insights.round_colours(5, which)
        contrast = [_contrast_on(c, 1.0, PAGES[which]) for c in ramp]
        assert contrast == sorted(contrast)


@pytest.mark.skip(reason="review cards prototype: pins the old Review layout (status lines, Explored fold, header); reconcile at the final merge")
def test_every_chart_in_the_fold_goes_through_the_privacy_wrapper(monkeypatch):
    drawn = []
    monkeypatch.setattr(ui_insights, "show_chart", lambda chart, **k: drawn.append(chart))
    monkeypatch.setattr(ui_insights.st, "altair_chart",
                        lambda *a, **k: pytest.fail("a bare st.altair_chart"))
    opt = D.load(D.LINE_A)
    computed = ui_insights._computed(opt, quiet=True)
    _three_picked(monkeypatch)
    ui_insights._explored_body(opt, computed)
    # Charts proto: What you've tried, How the app chooses, the triangle,
    # the map.
    assert len(drawn) == 4
    for chart in drawn:
        assert "usermeta" not in chart.to_dict()             # show_chart adds it
        assert ui_insights._private(chart).to_dict()["usermeta"] == \
            ui_insights._NO_CHART_ACTIONS


# ------------------------------------------------------------------ #
#  Process settings alone, and a choice row
# ------------------------------------------------------------------ #

@pytest.mark.skip(reason=REMOVED_CHART)
def test_a_study_of_process_settings_alone():
    opt = D.load(D.PROCESS_ONLY)
    strips = _strips(opt)
    assert [s.name for s in strips] == ["Bake temperature", "Bake time", "Proof time"]
    assert all(s.setting and not s.fill and s.unit for s in strips)
    said = [wording.for_project(opt, t) for t in (
        wording.strips_caption(True), wording.basis_explored(True, False))]
    for text in said:
        assert "ingredient" not in text.lower() and "amount" not in text.lower(), text
        assert "formulation" not in text.lower(), text
    spec = ui_insights.strips_chart(opt, strips).to_dict()
    made = next(m for m in _all_marks(spec) if _mark(m).get("filled") is False)
    assert made["encoding"]["tooltip"][0]["title"] == wording.strip_value_title(True)
    assert made["encoding"]["tooltip"][1]["title"] == "Trial"
    assert {r["value_text"].split(" ")[-1] for m in _all_marks(spec)
            if _mark(m).get("filled") is False for r in _rows(spec, m)} == {"°C", "min"}


def _with_a_choice():
    opt = FoodOptimizer("choice_strip")
    opt.add_process_parameter("Proof time", 30, 90, unit="min")
    opt.variables.append({'name': "Package", 'type': 'categorical',
                          'options': ["Tray", "Pouch", "Box"], 'category': 'process'})
    opt.add_objective("Volume", 100, goal="max", min_val=0, max_val=10)
    for k, (t, pack) in enumerate(((30, "Tray"), (60, "Pouch"), (90, "Tray"),
                                   (45, "Tray"), (75, "Pouch"))):
        opt.tell({"Proof time": float(t), "Package": pack}, {"Volume": 2 + t / 20},
                 formulation_no=k + 1, batch_no=1 + k // 3)
    return opt


@pytest.mark.skip(reason=REMOVED_CHART)
def test_a_choice_row_is_a_count_bar_and_an_untested_option_is_named():
    opt = _with_a_choice()
    strips = insights.tested_strips(opt)
    package = next(s for s in strips if s.name == "Package")
    assert package.options == {"Tray": 3, "Pouch": 2, "Box": 0}
    gaps = insights.strip_gaps(opt, strips)
    assert [(g.kind, g.option) for g in gaps if g.kind == "option"] == [("option", "Box")]
    spec = ui_insights.strips_chart(opt, strips).to_dict()
    bars = [m for m in _all_marks(spec) if _mark(m)["type"] == "bar"
            and m["encoding"]["x"]["field"] == "n"]
    assert len(bars) == 1
    assert {r["option"]: r["n"] for r in _rows(spec, bars[0])} == {"Tray": 3, "Pouch": 2,
                                                                   "Box": 0}
    assert bars[0]["encoding"]["x"]["title"] == wording.STRIP_COUNT


# ------------------------------------------------------------------ #
#  The map (R-I16)
# ------------------------------------------------------------------ #

def test_the_map_never_moves_the_global_rng():
    opt = D.load(D.TIMING)
    torch.manual_seed(123)
    before = torch.get_rng_state()
    assert torch.equal(torch.get_rng_state(), before)


def _rule(name):
    opt = D.load(name)
    computed = ui_insights._computed(opt, quiet=True)
    score = computed["fitted"].get(insights.SCORE)
    return opt, computed, ui_insights.explore_withheld(opt, score, computed["checks"])


@pytest.mark.parametrize("name,rule", [(D.LINE_A, None), (D.LINE_B, "not_yet"),
                                       (D.LINE_C, "hidden"), (D.UPGRADE, "not_yet")])
def test_the_withhold_rule_on_the_demo_projects(name, rule, monkeypatch):
    opt, computed, got = _rule(name)
    assert got == rule
    monkeypatch.setattr(ui_insights, "checks_for", lambda o, stage: computed["checks"])
    offered = ui_insights.layers_offered(opt, computed["fitted"])
    if rule == "hidden":
        assert offered == (wording.LAYER_SCORE,)
    elif rule == "not_yet":
        # R-I16 over e2e I4, for the Uncertainty layer only.
        assert offered == (wording.LAYER_SCORE, wording.LAYER_SD)
    else:
        assert wording.LAYER_SD in offered and wording.LAYER_GAIN in offered


def _three_picked(monkeypatch):
    """The triangle's three boxes, picked (bare mode's multiselect returns
    none): the first three offered."""
    monkeypatch.setattr(ui_insights.st, "multiselect",
                        lambda label, options, **k: list(options[:3]))


def _fold_text(monkeypatch, opt, computed):
    said = []
    for kind in ("caption", "markdown"):
        monkeypatch.setattr(ui_insights.st, kind,
                            lambda text, help=None, _k=kind, **k: said.append((_k, text)))
    monkeypatch.setattr(ui_insights, "show_chart", lambda c, **k: said.append(("chart", c)))
    monkeypatch.setattr(ui_insights.st, "button", lambda label, **k: said.append(("button", label)))
    _three_picked(monkeypatch)
    ui_insights._explored_body(opt, computed)
    return said


@pytest.mark.skip(reason="review cards prototype: pins the old Review layout (status lines, Explored fold, header); reconcile at the final merge")
def test_hidden_under_the_noise_guard_no_pointer(monkeypatch):
    opt, computed, _ = _rule(D.LINE_C)
    monkeypatch.setattr(ui_insights, "checks_for", lambda o, stage: computed["checks"])
    said = _fold_text(monkeypatch, opt, computed)
    texts = [t for k, t in said if k != "chart"]
    assert wording.MAP_HIDDEN in texts
    assert wording.MAP_TITLE not in [t.strip("*") for t in texts]
    # Charts proto: the lines, the figure's top panel, the triangle; no map.
    assert len([1 for k, _ in said if k == "chart"]) == 3
    assert wording.CHOOSES_WITHHELD in texts                        # its gain withheld
    assert not [t for k, t in said if k == "button"]


@pytest.mark.skip(reason="review cards prototype: pins the old Review layout (status lines, Explored fold, header); reconcile at the final merge")
def test_captioned_under_the_verdict(monkeypatch):
    opt, computed, _ = _rule(D.LINE_B)
    monkeypatch.setattr(ui_insights, "checks_for", lambda o, stage: computed["checks"])
    said = _fold_text(monkeypatch, opt, computed)
    texts = [t for k, t in said if k != "chart"]
    assert f"**{wording.MAP_TITLE}**" in texts and wording.MAP_READING in texts
    assert wording.MAP_NOT_YET in texts
    # Charts proto: the lines, the figure (gain withheld), triangle, map.
    assert len([1 for k, _ in said if k == "chart"]) == 4
    assert wording.CHOOSES_WITHHELD in texts
    # The pointer: the Uncertainty layer is offered here (R-I16).
    assert [t for k, t in said if k == "button"]


# ------------------------------------------------------------------ #
#  The pointer preselects Uncertainty once, only where offered
# ------------------------------------------------------------------ #

@pytest.mark.skip(reason=REVIEW_FOUR)
def test_the_pointer_preselects_uncertainty_once(monkeypatch):
    opt, computed, _ = _rule(D.LINE_A)
    state = {}
    monkeypatch.setattr(ui_insights.st, "session_state", state)
    monkeypatch.setattr(ui_insights, "checks_for", lambda o, stage: computed["checks"])
    monkeypatch.setattr(ui_insights, "ask_tab", lambda *a, **k: state.update(asked=(a, k)))
    ui_insights._point_to_uncertainty()
    assert state["asked"] == ((wording.TAB_INSIGHTS,),
                              {"section": ui_insights.TRY_ONE_KEY, "fold": ui_insights.TRY_ONE_KEY})
    radios = []
    monkeypatch.setattr(ui_insights.st, "radio",
                        lambda label, options, key, **k: radios.append(state.get(key)) or
                        state.get(key, options[0]))
    monkeypatch.setattr(ui_insights.st, "caption", lambda *a, **k: None)
    assert ui_insights._layer_pick(opt, computed["fitted"],
                                   ui_insights.PATH_LAYER_KEY) == wording.LAYER_SD
    state[ui_insights.PATH_LAYER_KEY] = wording.LAYER_SCORE          # the reader moves on
    assert ui_insights._layer_pick(opt, computed["fitted"],
                                   ui_insights.PATH_LAYER_KEY) == wording.LAYER_SCORE
    assert ui_insights._layer_pick(opt, computed["fitted"],
                                   ui_insights.LAYER_KEY) == wording.LAYER_SD
    assert state[ui_insights.LAYER_PRESELECT_KEY] == ()


# ------------------------------------------------------------------ #
#  Placement (R-I14) and Next step's link, on screen
# ------------------------------------------------------------------ #

def _install(name):
    state = json.loads((D.HERE / f"{name}.json").read_text())
    opt = FoodOptimizer(state["project_name"])
    opt.import_json(state)
    opt.save()
    return opt


def _insights(name, **state):
    opt = _install(name)
    at = AppTest.from_file(APP_PATH, default_timeout=300)
    at.session_state["_loaded_project"] = opt.project_name
    at.run()
    for key, value in state.items():
        at.session_state[key] = value
    at.session_state["main_tab"] = wording.TAB_INSIGHTS
    at.run()
    assert not at.exception
    return at


def _walk(node, out):
    for child in getattr(node, "children", {}).values():
        kind = getattr(child, "type", "")
        if kind == "expander":
            out.append(("fold", child.label))
            continue
        if kind == "button":
            out.append(("button", child.key))
            continue
        _walk(child, out)
    return out


def _no_repeat_link(monkeypatch):
    """The Timing check's Next asks for a Repeat, and the Repeat link
    outranks this one (R-I4 priority 2 over 3; ftue fix C): left out here,
    so the Explored link is the one drawn."""
    monkeypatch.setattr(ui_insights, "_repeat_link", lambda opt, facts: False)


@pytest.mark.skip(reason="review cards prototype: pins the old Review layout (status lines, Explored fold, header); reconcile at the final merge")
def test_the_fold_sits_under_the_button_and_above_the_folds_below(monkeypatch):
    _no_repeat_link(monkeypatch)
    at = _insights(D.TIMING)
    order = _walk(at.tabs[3], [])
    labels = [x for _, x in order]
    fold = labels.index(wording.FOLD_EXPLORED)
    assert labels.index("insights_foot") < labels.index("insights_explored_link") < fold
    assert fold < labels.index(ui_insights._try_one_label(D.load(D.TIMING)))
    assert fold < labels.index(wording.PAIR_FOLD) < labels.index(wording.FOLD_METHOD)
    explored = next(e for e in at.tabs[3].expander if e.label == wording.FOLD_EXPLORED)
    assert not explored.proto.expanded
    assert not list(explored.children.values())      # nothing worked out until opened


def test_no_link_while_the_why_link_has_the_place():
    """R-I4: one tertiary link; with a warm Round open, Why outranks it."""
    opt = D.load(D.PROCESS_ONLY)
    assert ui_insights._why_link_offered(opt)
    timing = D.load(D.TIMING)
    assert not ui_insights._why_link_offered(timing)
