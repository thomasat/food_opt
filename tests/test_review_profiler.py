"""Review profiler (prototype, 2026-10-02): the JMP-style prediction
profiler that replaces Try one's path and Compare all. Pinned on the Plant
patty Line A demo (a Formulation total with Water as Fill to total, a
fixed Salt row, a process setting, three measurements, Round 6 open): the
grid's rows and columns, the targets, the reference line, the mixture path
agreeing with Try one's, the withheld Expected gain row, "+N more" and the
palette. One model fit for the module."""
import inspect
import re
import time

import pytest

import bo_views
import insights
import insights_jobs
import ui_insights as U
import wording
from tests import demo_projects as D
from tests.test_food_bo import _BANNED


@pytest.fixture(scope="module")
def line_a(tmp_path_factory):
    mp = pytest.MonkeyPatch()
    mp.chdir(tmp_path_factory.mktemp("review-profiler"))
    mp.setattr(insights_jobs, "_JOBS", {})
    mp.setattr(insights_jobs, "_INFLIGHT", {})
    mp.setattr(U.st, "session_state", {})
    with bo_views.fresh():
        opt = D.load(D.LINE_A)
        computed = U._computed(opt, quiet=True)
        mp.setattr(U, "checks_for", lambda o, stage: computed["checks"])
        yield opt, computed
    mp.undo()


def _profile(opt, computed, which=0, reason="ask"):
    fitted = computed["fitted"]
    label, number, recipe, is_pick = U.profile_refs(opt)[which]
    reason = U.chooses_reason(opt, fitted) if reason == "ask" else reason
    return U.profile_data(opt, fitted, recipe, label, number, is_pick, reason)


@pytest.fixture(scope="module")
def best(line_a):
    return _profile(*line_a)


@pytest.fixture(scope="module")
def spec(line_a, best):
    opt, _ = line_a
    return U.profile_chart(opt, best).to_dict()


def _blocks(spec):
    """Review readable: the profiler wraps at U.PROFILE_WRAP panels a row;
    more columns make a second block (a vconcat of its own rows). With one
    block, the rows are the top-level vconcat as before."""
    top = spec["vconcat"]
    return [b["vconcat"] for b in top] if "vconcat" in top[0] else [top]


def _rows(spec):
    """Each row's (facet spec in the first block, its data rows across every
    block)."""
    blocks = _blocks(spec)
    return [(r, [d for b in blocks for d in spec["datasets"][b[k]["data"]["name"]]])
            for k, r in enumerate(blocks[0])]


def _y_title(row):
    for layer in row["spec"]["layer"]:
        y = layer.get("encoding", {}).get("y")
        if y and y.get("axis", {}).get("title"):
            return y["axis"]["title"]
    return None


def test_the_grid_has_a_column_per_choice_and_a_row_per_result(line_a, best, spec):
    opt, _ = line_a
    # Fixed Salt and the Fill to total row (Water) have no column.
    assert sorted(c.name for c in best.columns) == sorted(insights.path_choices(opt))
    assert "Salt" not in [c.name for c in best.columns]
    assert "Water" not in [c.name for c in best.columns] and best.fill == "Water"
    # Ordered by their effect on the predicted Overall score near what has
    # been tested (Compare all's ranking), then along the whole line.
    keys = [(c.near_effect, c.effect) for c in best.columns]
    assert keys == sorted(keys, reverse=True)
    assert all(0 <= c.near_effect <= c.effect + 1e-12 for c in best.columns)
    rows = _rows(spec)
    gain = best.reason is None
    assert len(rows) == 1 + len(opt.objectives) + (1 if gain else 0)
    titles = [_y_title(r) for r, _ in rows]
    assert titles[0][0] == "Overall score" and titles[0][1].startswith("(of 100")
    assert [t[0] for t in titles[1:1 + len(opt.objectives)]] == [
        o['name'] for o in opt.measurements_by_importance()]
    if gain:
        assert titles[-1][0] == wording.PROFILER_GAIN_ROW
    # Every row has every column, in the same order; the top row names them.
    for r, data in rows:
        assert len({d["col"] for d in data}) == len(best.columns)
        assert r["facet"]["column"]["sort"] == rows[0][0]["facet"]["column"]["sort"]
    assert rows[0][0]["facet"]["column"]["header"].get("labels", True) is not False
    assert all(r["facet"]["column"]["header"].get("labels") is False for r, _ in rows[1:])
    # Each column's x runs over its own Lowest-Highest.
    for c in best.columns:
        lo, hi = (float(b) for b in insights._var(opt, c.name)['bounds'])
        ends = sorted(d["x"] for d in rows[0][1]
                      if d["mark"] == "ends" and d["col"].replace("\n", " ").startswith(c.name))
        assert ends == [pytest.approx(min(lo, c.path.start_amount)),
                        pytest.approx(max(hi, c.path.start_amount))]


def test_targets_are_drawn_on_their_rows(line_a, spec):
    opt, _ = line_a
    rows = _rows(spec)
    by_name = {_y_title(r)[0]: data for r, data in rows}
    for o in opt.objectives:
        targets = {d["target"] for d in by_name[o['name']] if d["mark"] == "target"}
        if o['goal'] == 'target':
            assert targets == {float(o['target'])}, o['name']
            assert _y_title_of(rows, o['name'])[-1].startswith("target ")
        else:
            assert not targets
    # Dashed and grey.
    rules = [l for r, _ in rows for l in r["spec"]["layer"]
             if l.get("transform", [{}])[0].get("filter", {}).get("equal") == "target"]
    assert rules and all(l["encoding"]["strokeDash"]["scale"]["range"][1] == [4, 3]
                         for l in rules)


def _y_title_of(rows, name):
    return next(_y_title(r) for r, _ in rows if _y_title(r)[0] == name)


def test_the_reference_line_is_drawn_in_every_panel(line_a, best, spec):
    opt, _ = line_a
    for r, data in _rows(spec):
        refs = {d["col"]: d["x"] for d in data if d["mark"] == "ref"}
        assert len(refs) == len(best.columns)
        assert all(d["role"] == wording.PROFILER_REF_BEST for d in data if d["mark"] == "ref")
    start = insights.start_recipe(opt, opt.best_index())
    for c in best.columns:
        assert c.path.start_amount == pytest.approx(float(start[c.name]))


def test_the_mixture_path_matches_try_one(line_a, best):
    """Each column is Try one's own path (Water fills to the total), and its
    predictions are Try one's to 1e-6, at the profiler's points and at the
    start Try one's 60-point path shares."""
    opt, computed = line_a
    fitted = computed["fitted"]
    start = insights.start_recipe(opt, opt.best_index())
    total = float(opt.formulation_total)
    ingredients = [v['name'] for v in opt.variables if v.get('category') == 'ingredient']
    for c in best.columns:
        path = insights.build_path(opt, c.name, start, points=U.PROFILE_POINTS)
        assert path.amounts == c.path.amounts
        try_one = insights.path_predictions(opt, fitted, path)
        for name, preds in c.preds.items():
            for a, b in zip(preds, try_one[name]):
                assert a.mean == pytest.approx(b.mean, abs=1e-6)
                assert a.low == pytest.approx(b.low, abs=1e-6)
                assert a.high == pytest.approx(b.high, abs=1e-6)
        full = insights.build_path(opt, c.name, start)
        k_full = full.amounts.index(full.start_amount)
        k_here = c.path.amounts.index(c.path.start_amount)
        full_preds = insights.path_predictions(opt, fitted, full)
        for name in c.preds:
            assert c.preds[name][k_here].mean == pytest.approx(full_preds[name][k_full].mean,
                                                                 abs=1e-6)
        if c.path.mode == "fill":
            for r in c.path.recipes:
                assert sum(float(r[n]) for n in ingredients) == pytest.approx(total, rel=1e-6)


def test_a_withheld_gain_is_greyed_and_says_why(line_a):
    """Review four, ruling 2 (was: no row): the gain row is drawn, greyed,
    with the reason under the chart; a cold Round still has none."""
    opt, computed = line_a
    reason = wording.CHOOSES_REASON_NOISE
    prof = _profile(opt, computed, reason=reason)
    assert prof.reason == reason and prof.greyed
    assert all(c.gain is not None for c in prof.columns)
    spec = U.profile_chart(opt, prof).to_dict()
    titles = [_y_title(r)[0] for r, _ in _rows(spec)]
    assert wording.PROFILER_GAIN_ROW in titles
    src = inspect.getsource(U._profiler)
    assert "wording.GAIN_GREYED[hero_reason_key(profile.reason)]" in src
    assert "wording.tip_gain_greyed(profile.reason)" in src
    assert "chooses_reason(opt, fitted)" in inspect.getsource(U.profile_cached)
    cold = _profile(opt, computed, reason=wording.CHOOSES_REASON_COLD)
    assert not cold.gain_drawn and all(c.gain is None for c in cold.columns)


def test_the_gain_row_is_the_heros_quantity(line_a, best):
    opt, computed = line_a
    if best.reason is not None:
        pytest.skip("gain withheld on this fixture")
    model = U.suggestion_model(opt, computed["fitted"])
    for c in best.columns[:2]:
        want = bo_views.evaluate(opt, model, bo_views.points(opt, c.path.recipes)).value.tolist()
        assert c.gain == pytest.approx(want, abs=1e-9)


def test_more_than_six_columns_or_four_rows_are_capped(line_a, best):
    opt, _ = line_a
    shown, extra, rows, extra_rows = U.profile_shown(best)
    # Review story: 4 columns by default (Line A has 6: two under "+N more").
    assert len(shown) == min(U.PROFILE_COLUMNS, len(best.columns))
    assert extra == [c.name for c in best.columns[U.PROFILE_COLUMNS:]]
    many = U.Profile(best.columns + best.columns[:3], best.rows + ["a", "b"], *(
        best.ref_label, best.ref_no, best.is_pick, best.reason, best.fill, best.fixed, []))
    shown, extra, rows, extra_rows = U.profile_shown(many)
    assert len(shown) == U.PROFILE_COLUMNS and len(extra) == len(many.columns) - U.PROFILE_COLUMNS
    assert len(rows) == U.PROFILE_ROWS and extra_rows == many.rows[1 + U.PROFILE_ROWS:]
    assert wording.profiler_more(3) == "+3 more"
    # Under the noise guard: Set up's order, no ranking.
    order = list(reversed(insights.path_choices(opt)))
    assert U.profile_shown(best, order)[0] == order[:U.PROFILE_COLUMNS]
    # A chart of a few columns draws only those.
    spec = U.profile_chart(opt, best, shown[:2]).to_dict()
    for _, data in _rows(spec):
        assert len({d["col"] for d in data}) == 2


def _colours(spec):
    found = set()

    def walk(x):
        if isinstance(x, dict):
            for k, v in x.items():
                if k in ("color", "stroke", "fill") and isinstance(v, str):
                    found.add(v.lower())
                elif k == "range" and isinstance(v, list):
                    found.update(c.lower() for c in v if isinstance(c, str))
                else:
                    walk(v)
        elif isinstance(x, list):
            for v in x:
                walk(v)
    walk(spec)
    return found


def test_the_palette(line_a, spec):
    opt, computed = line_a
    ink = U.mark("bar_latest")[0].lower()
    blue = U.model_blue().lower()
    band = U._BAND_FILL[U.theme()][0].lower()
    grey = U._FILLS[U.theme()]["untested"][0].lower()
    neutral = U.mark("neutral")[0].lower()
    orange = U.outline_colour().lower()
    found = _colours(spec)
    assert {ink, blue, band, grey, neutral} <= found
    assert orange not in found
    # Through Suggestion 1, the reference line is orange, not ink.
    pick = _profile(opt, computed, which=1)
    assert pick.is_pick and pick.ref_label == wording.PROFILER_REF_PICK
    found = _colours(U.profile_chart(opt, pick).to_dict())
    assert orange in found and ink not in found


def test_every_encoded_colour_has_a_legend_and_every_mark_a_bo_tooltip(spec):
    used, shown, tips = set(), set(), []
    for r in (r for b in _blocks(spec) for r in b):
        for layer in r["spec"]["layer"]:
            enc = layer.get("encoding", {})
            for ch in ("color", "fill", "stroke", "strokeDash", "shape"):
                e = enc.get(ch)
                if isinstance(e, dict) and "field" in e:
                    used.add(e["field"])
                    if e.get("legend", {}) is not None:
                        shown.add(e["field"])
            t = enc.get("tooltip")
            if t:
                tips += t if isinstance(t, list) else [t]
    assert used <= shown
    assert any(t.get("title") == wording.PLOT_BO for t in tips)
    data = [d for ds in spec["datasets"].values() for d in ds if d.get("mark") == "curve"]
    assert data and all(d["bo"].startswith(("GP posterior mean", "qLogNEI")) for d in data)
    assert all(re.search(r"\(.+\)", d["text"]) or d["bo"].startswith("qLogNEI") for d in data)


def test_the_words(line_a):
    words = lambda t: len([w for w in re.sub(r"[^\w%' ]", " ", t).split() if w])
    assert words(wording.PLOT_PROFILER) <= 6
    assert words(wording.PROFILER_CAPTION) <= 8
    assert words(wording.profiler_fill_note("Tap water")) <= 8
    for label in (wording.PROFILER_THROUGH, wording.PROFILER_REF_BEST, wording.PROFILER_REF_PICK,
                  wording.profiler_more(12), wording.PROFILER_GAIN_ROW):
        assert words(label) <= 5, label
    texts = [wording.PLOT_PROFILER, wording.PROFILER_CAPTION, wording.PROFILER_THROUGH,
             wording.PROFILER_TESTED, wording.PROFILER_GAIN_ROW, wording.PROFILER_GAIN_AXIS,
             wording.PROFILER_ROW_LOWER, wording.PROFILER_ROW_HIGHER, wording.PROFILER_STOPPED,
             wording.profiler_more_help(), wording.profiler_more_help(rows=True),
             wording.profiler_fill_note("Water"), wording.profiler_at_end("Highest"),
             wording.profiler_help("Best so far", 7, "Water", ("Salt",)),
             wording.profiler_help("Best so far", 7, settings_only=True, gain=False)]
    for t in texts:
        assert not any(p.search(t) for p in _BANNED), t
        assert "spread" not in t.lower(), t
    # A process-only study says Trial.
    opt = D.load(D.PROCESS_ONLY)
    assert "Trial 7" in U._say(opt, wording.profiler_help("Best so far", 7, settings_only=True))


def test_the_profile_is_fast(line_a):
    """Every column through every display model and the suggestion model,
    after the fit: well under a second on Line A."""
    opt, computed = line_a
    t = time.perf_counter()
    prof = _profile(opt, computed)
    U.profile_chart(opt, prof).to_dict()
    # Review readable: past PROFILE_WRAP columns a second block is built
    # (every column drawn here; the app draws 4 by default): was < 3.0 s.
    assert time.perf_counter() - t < 4.5
