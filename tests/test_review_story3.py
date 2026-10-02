"""Review story 3 (prototype, 2026-10-02; owner: demo the yogurt pilot only,
and fade the profiler's untested stretches so a model curve there does not
read as a finding).

- The profiler draws each line solid and its band as before over tested
  amounts, and thin, dashed and faint, the band lighter, over the "not
  tested near here" stretches (the same rule as the grey).
- Q3's finding and the column order use each row's effect within its
  tested stretch; with no clear leader there, it says so.
- Two rows whose tested amounts moved together (|r| >= 0.7) are noted
  under the chart, the exact r in the note's "?".
- Method and definitions defines Expected gain once.

Synthetic: the Plant patty Line A demo project (CI); no pilot data."""
import re

import pytest

import bo_views
import insights
import insights_jobs
import ui_insights as U
import wording
from tests import demo_projects as D
from tests.test_food_bo import _BANNED


def _words(text):
    plain = re.sub(r":\w+\[(.*?)\]", r"\1", text).replace("**", "")
    return len([w for w in plain.split() if re.search(r"\w", w)])


@pytest.fixture(scope="module")
def line_a(tmp_path_factory):
    mp = pytest.MonkeyPatch()
    mp.chdir(tmp_path_factory.mktemp("review-story3"))
    mp.setattr(insights_jobs, "_JOBS", {})
    mp.setattr(insights_jobs, "_INFLIGHT", {})
    mp.setattr(U.st, "session_state", {})
    with bo_views.fresh():
        opt = D.load(D.LINE_A)
        computed = U._computed(opt, quiet=True)
        mp.setattr(U, "checks_for", lambda o, stage: computed["checks"])
        label, number, recipe, is_pick = U.profile_refs(opt)[0]
        reason = U.chooses_reason(opt, computed["fitted"])
        profile = U.profile_data(opt, computed["fitted"], recipe, label, number, is_pick,
                                 reason)
        yield opt, computed, profile
    mp.undo()


# ------------------------------------------------------------------ #
#  Tested and untested stretches drawn differently
# ------------------------------------------------------------------ #

def test_split_tested_meets_where_the_grey_begins():
    amounts = [0.0, 1.0, 2.0, 3.0, 4.0]
    near = [False, True, True, False, False]
    pts = [{"v": float(v)} for v in (10, 20, 30, 40, 50)]
    runs = U.split_tested(amounts, near, pts)
    assert [ok for ok, _ in runs] == [False, True, False]
    assert [p["x"] for p in runs[0][1]] == [0.0, 0.5]
    assert [p["x"] for p in runs[1][1]] == [0.5, 1.0, 2.0, 2.5]
    assert [p["x"] for p in runs[2][1]] == [2.5, 3.0, 4.0]
    assert runs[1][1][0]["v"] == 15.0 and runs[1][1][-1]["v"] == 35.0
    # The pieces' joins are exactly the grey's edges.
    class _P:
        pass
    path = _P()
    path.amounts = amounts
    edges = {e for lo, hi in insights.untested_runs(path, near) for e in (lo, hi)}
    assert {0.5, 2.5} <= edges
    # One piece when all tested.
    assert [ok for ok, _ in U.split_tested(amounts, [True] * 5, pts)] == [True]


def _rows(spec):
    return [(r, spec["datasets"][r["data"]["name"]]) for r in spec["vconcat"]]


def _filtered(layer):
    t = layer.get("transform") or []
    return next((x["filter"]["equal"] for x in t if "filter" in x), None)


def test_untested_stretches_are_faded_on_every_row(line_a):
    opt, _, profile = line_a
    assert any(not all(c.near) for c in profile.columns)
    spec = U.profile_chart(opt, profile).to_dict()
    rows = _rows(spec)
    # Overall score, each measurement, Expected gain (when drawn).
    assert len(rows) == len(profile.rows) + (1 if profile.gain_drawn else 0)
    for row, data in rows:
        marks = {d["mark"] for d in data}
        assert {"seg_t", "seg_u", "grey"} <= marks
        lines = {_filtered(l): l["mark"] for l in row["spec"]["layer"]
                 if l["mark"]["type"] == "line"}
        solid, faded = lines["seg_t"], lines["seg_u"]
        assert faded["strokeDash"] == [3, 3] and faded["opacity"] < 0.6
        assert faded["strokeWidth"] < solid["strokeWidth"] == 2
        assert solid.get("opacity", 1) == 1
        areas = {_filtered(l): l["mark"] for l in row["spec"]["layer"]
                 if l["mark"]["type"] == "area"}
        assert areas["seg_u"]["opacity"] < 0.5 * areas["seg_t"]["opacity"]
        # Each piece is its own line (detail), never joined across a gap.
        layer = next(l for l in row["spec"]["layer"] if _filtered(l) == "seg_u"
                     and l["mark"]["type"] == "line")
        assert layer["encoding"]["detail"]["field"] == "run"
    # The legend says what the dashes mean, in the grey's own entry.
    assert wording.PROFILER_UNTESTED_DASHED in str(spec)
    assert _words(wording.PROFILER_UNTESTED_DASHED) <= 5


def test_the_faded_pieces_cover_the_untested_runs(line_a):
    opt, _, profile = line_a
    df = U._profile_row_frame(opt, profile.columns, "score", insights.SCORE,
                              {c.name: c.name for c in profile.columns},
                              ("r", "t", "u", "tested", "band"))
    for c in profile.columns:
        mine = df[df["col"] == c.name]
        for lo, hi in insights.untested_runs(c.path, c.near):
            faded = mine[(mine["mark"] == "seg_u") & (mine["x"] >= lo - 1e-9)
                         & (mine["x"] <= hi + 1e-9)]
            assert faded["x"].min() == pytest.approx(lo) and faded["x"].max() == pytest.approx(hi)
            inside = mine[(mine["mark"] == "seg_t") & (mine["x"] > lo + 1e-9)
                          & (mine["x"] < hi - 1e-9)]
            assert inside.empty, c.name


# ------------------------------------------------------------------ #
#  The finding: tested stretches only
# ------------------------------------------------------------------ #

class _Col:
    def __init__(self, name, near_effect, effect=None, end=None):
        self.name, self.near_effect, self.end = name, near_effect, end
        self.effect = near_effect if effect is None else effect


class _Prof:
    def __init__(self, cols, ref="Best so far"):
        self.columns, self.ref_label = cols, ref


def test_the_ranking_is_the_tested_stretch_effect(line_a):
    _, _, profile = line_a
    for c in profile.columns:
        score = [p.mean for p in c.preds[insights.SCORE]]
        runs = [m for ok, m in U._runs(c.near, score) if ok]
        assert c.near_effect == pytest.approx(max([max(m) - min(m) for m in runs] or [0.0]))
        assert c.near_effect <= c.effect + 1e-12
    keys = [(c.near_effect, c.effect) for c in profile.columns]
    assert keys == sorted(keys, reverse=True)


def test_the_finding_uses_tested_stretches_only():
    fine = insights.Guard([])
    no_leader = f"**{wording.story_no_leader()}.**"
    # A big swing that is all guesswork does not lead (profile_data puts the
    # tested-stretch effect first; the line reads that order).
    p = _Prof([_Col("Salt", 4.0, effect=4.5), _Col("Tap water", 1.0, effect=15.0)])
    assert U.driver_finding(None, p, fine) == "**Salt matters most.** Then Tap water."
    # A near tie over tested amounts: no clear leader.
    p = _Prof([_Col("Tap water", 4.0, effect=15.0), _Col("Salt", 3.5)])
    assert U.driver_finding(None, p, fine) == no_leader
    # Under one tasting's noise (1 point assumed): no clear leader.
    assert U.driver_finding(None, _Prof([_Col("Salt", 0.8, effect=9.0)]), fine) == no_leader
    # A clear lead, at a limit.
    p = _Prof([_Col("Tap water", 6.0, end="Highest"), _Col("Salt", 2.0)])
    assert U.driver_finding(None, p, fine) == \
        "**Tap water matters most.** Best so far uses its Highest."
    assert U.clear_leader(_Prof([_Col("A", 5.0), _Col("B", 3.99)]), 1.0)
    assert not U.clear_leader(_Prof([_Col("A", 5.0), _Col("B", 4.0)]), 1.0)
    assert _words(no_leader) <= 10
    for kind in ("ingredients", "settings", "both"):
        assert _words(wording.story_no_leader(kind)) <= 10


# ------------------------------------------------------------------ #
#  Amounts that moved together
# ------------------------------------------------------------------ #

def _together(monkeypatch, opt, pairs):
    """Rewrite the tested amounts: each (follower, leader, slope, wobble)
    makes `follower` move with `leader`."""
    real = insights.start_recipe

    def fake(o, i):
        r = dict(real(o, i))
        for follow, lead, slope, wobble in pairs:
            r[follow] = 2.0 + slope * float(r[lead]) + wobble * ((i * 7) % 5 - 2)
        return r
    monkeypatch.setattr(insights, "start_recipe", fake)


def test_no_note_without_a_pair(line_a):
    opt, _, profile = line_a
    assert all(abs(r) < 0.7 for _, _, r in insights.moved_pairs(opt, 0.0))
    assert U.moved_note(opt, [c.name for c in profile.columns]) is None


def test_the_note_names_the_strongest_shown_pair_with_r_in_its_help(line_a, monkeypatch):
    opt, _, _ = line_a
    _together(monkeypatch, opt, [("Coconut oil", "Textured pea protein", -0.3, 0.0),
                                 ("Beet juice powder", "Methylcellulose", 0.5, 0.03)])
    pairs = insights.moved_pairs(opt)
    first = pairs[0]
    assert {first[0], first[1]} == {"Coconut oil", "Textured pea protein"}
    assert first[2] == pytest.approx(-1.0)
    assert all(abs(r) >= 0.7 - 0.005 for _, _, r in pairs)
    shown = ["Textured pea protein", "Coconut oil", "Methylcellulose", "Beet juice powder"]
    note, help_ = U.moved_note(opt, shown)
    assert note == wording.moved_note("Textured pea protein", "Coconut oil")
    assert note.endswith("moved together: effects mixed.")
    assert "r = −1.00" in help_ and "|r| ≥ 0.7" in help_
    # The other pairs are listed in the "?", not on screen.
    beet = next(p for p in pairs if "Beet juice powder" in p[:2])
    assert f"({wording.moved_r(beet[2])})" in help_ and "Also moved together" in help_
    # Only pairs with a shown column count for the note.
    note, _ = U.moved_note(opt, ["Methylcellulose", "Mixing time after fat"])
    assert "Methylcellulose" in note and "Beet juice powder" in note
    assert U.moved_note(opt, ["Mixing time after fat"]) is None


def test_the_threshold_is_point_seven(line_a, monkeypatch):
    opt, _, _ = line_a
    _together(monkeypatch, opt, [("Coconut oil", "Textured pea protein", 0.3, 0.0)])
    assert insights.moved_pairs(opt, 0.7)
    real = insights.moved_pairs
    # Below 0.7 (to two decimals) nothing is noted.
    monkeypatch.setattr(insights, "moved_pairs", lambda o: [p for p in real(o, 0.0)
                                                            if round(abs(p[2]), 2) >= 1.01])
    assert U.moved_note(opt, ["Coconut oil"]) is None


# ------------------------------------------------------------------ #
#  Method and definitions; wording
# ------------------------------------------------------------------ #

def test_method_defines_expected_gain(line_a):
    opt, _, _ = line_a
    facts = insights.model_summary(opt)
    lines = wording.model_summary(facts)
    if facts.acquisition == "qlognei":
        assert wording.METHOD_GAIN in lines
        # Beside the acquisition line it explains.
        assert lines.index(wording.METHOD_GAIN) == \
            next(i for i, t in enumerate(lines) if t.startswith("Acquisition:")) + 1
    assert wording.METHOD_GAIN.startswith("Expected gain: expected improvement in the Overall "
                                          "score (qLogNEI), in points")
    assert wording.METHOD_GAIN.endswith("Not information gain.")


def _new_strings():
    return [wording.PROFILER_UNTESTED_DASHED, wording.story_no_leader(),
            wording.story_no_leader("settings"), wording.story_no_leader("both"),
            wording.story_leader_rule("1"), wording.moved_note("Oat flour", "Salt"),
            wording.moved_help("Oat flour", "Salt", -0.71, 15, [("A", "B", 0.8)]),
            wording.METHOD_GAIN]


def test_new_strings_pass_the_vocabulary_guard():
    for text in _new_strings():
        assert not any(p.search(text) for p in _BANNED), text
        assert "spread" not in text.lower(), text
    assert _words(wording.moved_note("Tap water", "Coconut cream")) <= 10
    src = open(wording.__file__).read()
    assert "# --- review story 3 ---" in src
    assert src.index("# --- review story 3 ---") < src.index("PROFILER_UNTESTED_DASHED =")
