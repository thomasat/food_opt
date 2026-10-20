"""Final walk, the 4 · Review "Best so far" chart: (1) after a discarded
Round its numbers retire (F1–F3 recorded, Round 2 discarded, the next made
F7–F9), so the chart plots by place in the order made — no empty slots 4–6
— while its labels and tooltips keep the real numbers; (2) the score axis
labels a tick at or above the highest dot (88.4 sat over the top label, 85),
the 100 line when that tick would reach it."""
import pytest

import ui_insights as U
import wording
from food_bo import FoodOptimizer


@pytest.fixture(autouse=True)
def _isolate_cwd(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)


def _project(rounds):
    """One 0–10 'max' measurement, so a rating of 8.84 scores 88.4.
    `rounds`: [(Round, [(formulation number, rating)])]."""
    opt = FoodOptimizer("walk")
    opt.set_amount_unit("g")
    opt.add_ingredient("Oil", 0, 20)
    opt.add_ingredient("Starch", 0, 5)
    opt.add_objective("Taste", 100, goal="max", min_val=0, max_val=10)
    for r, made in rounds:
        for no, rating in made:
            opt.tell({"Oil": 1.0 + no, "Starch": 1.0}, {"Taste": rating},
                     formulation_no=no, batch_no=r)
    return opt


ROUND1 = (1, [(1, 8.74), (2, 8.84), (3, 8.05)])
ROUND2_AFTER_DISCARD = (2, [(7, 9.25), (8, 8.76), (9, 8.83)])


def _units(spec):
    """Every unit layer of a spec with its rows (from the named datasets)."""
    out = []

    def walk(node):
        for child in node.get("layer", []):
            walk(child)
        if "mark" in node:
            data = node.get("data", {})
            out.append((node, spec.get("datasets", {}).get(data.get("name"),
                                                           data.get("values", []))))
    walk(spec)
    return out


def _kind(node):
    return node["mark"] if isinstance(node["mark"], str) else node["mark"]["type"]


def _chart(opt, upto=None):
    return U.improving_chart(opt, height=U.PANEL_CHART_H, upto=upto).to_dict()


def _dots(spec):
    return [(node, rows) for node, rows in _units(spec) if _kind(node) == "point" and rows
            and wording.OVERALL_SCORE_COLUMN in rows[0]]


def _bottom_x_axis(spec):
    return next(node["encoding"]["x"]["axis"] for node, rows in _dots(spec))


# ------------------------------------------------------------------ #
#  1 · places, not numbers, after a discard
# ------------------------------------------------------------------ #

def test_after_a_discard_the_dots_sit_side_by_side_labelled_with_their_numbers():
    opt = _project([ROUND1, ROUND2_AFTER_DISCARD])
    spec = _chart(opt)
    formulation = wording.FORMULATION_CAP
    placed = sorted((row["place"], row[formulation]) for _node, rows in _dots(spec)
                    for row in rows)
    assert placed == [(1, 1), (2, 2), (3, 3), (4, 7), (5, 8), (6, 9)]
    for node, _rows in _dots(spec):
        assert node["encoding"]["x"]["field"] == "place"
        tips = [t["field"] for t in node["encoding"]["tooltip"]]
        assert formulation in tips and "place" not in tips       # the real number
    axis = _bottom_x_axis(spec)
    assert axis["values"] == [1, 2, 3, 4, 5, 6]
    expr = axis["labelExpr"]
    for place, no in ((1, 1), (3, 3), (4, 7), (6, 9)):
        assert f"datum.value == {place} ? '{no}'" in expr
    domain = next(node["encoding"]["x"]["scale"]["domain"] for node, _ in _dots(spec))
    assert domain == pytest.approx([0.3, 6.7])                   # no empty slots 4–6
    assert _bottom_x_axis(spec) and wording.Q2_X_TITLE.endswith("(in order made)")


def test_after_a_discard_the_round_line_averages_and_best_step_follow_the_places():
    opt = _project([ROUND1, ROUND2_AFTER_DISCARD])
    units = _units(_chart(opt))
    rules = [row for node, rows in units if _kind(node) == "rule"
             and node["encoding"].get("x", {}).get("axis", {}).get("orient") == "top"
             for row in rows]
    assert [r["place"] for r in rules] == [3.5]                # between F3 and F7
    top = next(node["encoding"]["x"]["axis"] for node, _ in units
               if node["encoding"].get("x", {}).get("axis", {}).get("orient") == "top")
    assert top["values"] == [2.0, 5.0]                            # R1, R2 over their middles
    bars = sorted((row["x0"], row["x1"]) for node, rows in units if _kind(node) == "rule"
                  and "x2" in node["encoding"] for row in rows)
    assert bars == [pytest.approx((0.6, 3.4)), pytest.approx((3.6, 6.4))]
    line = next((node, rows) for node, rows in units if _kind(node) == "line")
    assert [row["x"] for row in line[1]] == [1, 2, 3, 4, 5, 6]
    assert [row["no"] for row in line[1]] == [1, 2, 3, 7, 8, 9]
    assert line[0]["encoding"]["tooltip"][0]["field"] == "no"


def test_an_earlier_round_on_view_keeps_the_places_and_the_axes():
    opt = _project([ROUND1, ROUND2_AFTER_DISCARD])
    now, then = _chart(opt), _chart(opt, upto=1)
    assert _bottom_x_axis(then) == _bottom_x_axis(now)
    shown = sorted(row["place"] for _n, rows in _dots(then) for row in rows
                   if row[wording.OVERALL_SCORE_COLUMN] is not None)
    assert shown == [1, 2, 3]


def test_numbers_in_order_keep_plain_tick_labels():
    """The yogurt pilot's case: numbers 1, 2, 3 … are their places, so the
    axis needs no relabelling and looks as before."""
    opt = _project([ROUND1, (2, [(4, 9.25), (5, 8.76), (6, 8.83)])])
    spec = _chart(opt)
    assert "labelExpr" not in _bottom_x_axis(spec)
    assert all(row["place"] == row[wording.FORMULATION_CAP]
               for _n, rows in _dots(spec) for row in rows)


# ------------------------------------------------------------------ #
#  2 · a labelled tick at or above the highest dot
# ------------------------------------------------------------------ #

def _y(spec):
    node = next(node for node, _rows in _dots(spec))
    return node["encoding"]["y"]["axis"]["values"], node["encoding"]["y"]["scale"]["domain"]


@pytest.mark.parametrize("rounds", [[ROUND1], [ROUND1, ROUND2_AFTER_DISCARD],
                                    [(1, [(1, 6.1), (2, 7.33), (3, 4.2)])],
                                    [(1, [(1, 0.2), (2, 1.1), (3, 2.0)])],
                                    [(1, [(1, 9.55), (2, 9.52)])]])
def test_the_top_tick_is_at_or_above_the_highest_dot(rounds):
    opt = _project(rounds)
    scores = [float(v) for v in opt.Y_history]
    ticks, (bottom, top) = _y(_chart(opt))
    assert max(ticks) >= max(scores) - 1e-9, (ticks, scores)
    assert top > max(ticks)                                  # its label is not clipped
    assert bottom < min(scores)                              # N12 still holds
    assert all(t <= float(opt.utility_ceiling()) + 1e-9 for t in ticks)


def test_round1_at_88_labels_90_and_round2_keeps_the_100_line():
    one = _chart(_project([ROUND1]))
    ticks, _ = _y(one)
    assert max(ticks) == 90
    assert "100 = " not in str(one)
    both = _chart(_project([ROUND1, ROUND2_AFTER_DISCARD]))
    ticks, _ = _y(both)
    assert max(ticks) == 100
    assert wording.review_top_line("100") in str(both)


def test_a_top_tick_that_would_reach_100_shows_the_100_line():
    spec = _chart(_project([(1, [(1, 9.55), (2, 9.52)])]))
    ticks, _ = _y(spec)
    assert max(ticks) == 100
    assert wording.review_top_line("100") in str(spec)
