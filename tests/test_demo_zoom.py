"""Presentation zoom check (2026-10-07): at 125 % zoom a Round's average
label sat on a formulation's dot. Each label now sits under its bar unless
a mark is there, then over it, then the smaller nudge clear.

Synthetic projects only; no pilot data."""
import json

import pytest

import ui_insights as U
from food_bo import FoodOptimizer

H, W, F = U.IMPROVING_HEIGHT, U._MIN_PLOT_W, U.FONT_SMALL


def _px(y_domain, x_domain):
    (y0, y1), (x0, x1) = y_domain, x_domain
    return H / (y1 - y0), W / (x1 - x0)


def _clears(label_y, avg, mid, text, marks, y_domain, x_domain):
    """Whether a label centred at label_y keeps off every (x, y, r) mark."""
    ypx, xpx = _px(y_domain, x_domain)
    half_w, half_h = 0.31 * F * len(text), F / 2
    for x, y, r in marks:
        dx, dy = abs(x - mid) * xpx, abs(y - label_y) * ypx
        if dx < half_w + r and dy < half_h + r:
            return False
    return True


Y, X = (70.0, 105.0), (0.3, 9.7)


def test_with_no_mark_near_the_label_stays_under_its_bar():
    avgs = [(1, 1, 3, 80.0, 3)]
    (y,) = U.average_label_y(avgs, [], [], [], Y, X)
    ypx, _ = _px(Y, X)
    assert y == pytest.approx(80.0 - (U._AVG_GAP + F / 2) / ypx)


def test_a_dot_under_the_bar_moves_the_label_over_it():
    # The middle dot just under the average, where the label used to go.
    avgs = [(1, 1, 3, 80.0, 3)]
    dots = [(1, 72.0), (2, 79.0), (3, 89.0)]
    (y,) = U.average_label_y(avgs, dots, [], [], Y, X)
    assert y > 80.0
    assert _clears(y, 80.0, 2, "80.0", [(x, v, U._DOT_R) for x, v in dots], Y, X)


def test_dots_both_sides_nudge_the_label_clear():
    avgs = [(1, 1, 3, 90.0, 3)]
    ypx, _ = _px(Y, X)
    # One dot just under, one just over: neither side is free at the gap.
    dots = [(2, 90.0 - 6 / ypx), (2, 90.0 + 8 / ypx)]
    (y,) = U.average_label_y(avgs, dots, [], [], Y, X)
    assert _clears(y, 90.0, 2, "90.0", [(x, v, U._DOT_R) for x, v in dots], Y, X)
    assert abs(y - 90.0) * ypx <= U._AVG_GAP + U._AVG_MAX_SHIFT + F


def test_the_best_line_over_the_bar_keeps_the_label_under():
    # A dot under the bar and Best so far just over it: the label is nudged
    # under the dot rather than drawn on the line.
    avgs = [(1, 1, 3, 95.0, 3)]
    ypx, _ = _px(Y, X)
    dots = [(2, 95.0 - 7 / ypx)]
    steps = [(0.5, 3.5, 95.0 + 10 / ypx, 1)]
    (y,) = U.average_label_y(avgs, dots, steps, [], Y, X)
    assert y < 95.0
    assert _clears(y, 95.0, 2, "95.0", [(x, v, U._DOT_R) for x, v in dots]
                   + [(k / 10, steps[0][2], 1.5) for k in range(5, 36)], Y, X)


def test_the_label_never_leaves_the_plot():
    # The bar at the very bottom: a label under it would be cut off.
    avgs = [(1, 1, 3, 70.5, 3)]
    (y,) = U.average_label_y(avgs, [], [], [], Y, X)
    assert y > 70.5


def _project(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    opt = FoodOptimizer("zoom")
    opt.set_amount_unit("g")
    opt.add_ingredient("Oat", 0, 100)
    opt.add_ingredient("Water", 0, 100)
    opt.add_objective("Consistency", 100.0, goal="target", target=12, min_val=0, max_val=15)
    # Score 100 − 100·|v − 12| / 15; each Round's middle dot just under its
    # average.
    rounds = [[4.5, 7.8, 12.0], [7.5, 9.45, 11.7], [9.0, 10.35, 11.85]]
    no = 1
    for r, row in enumerate(rounds, 1):
        for v in row:
            opt.tell({"Oat": 5.0 * no, "Water": 50.0}, {"Consistency": v},
                     formulation_no=no, batch_no=r)
            no += 1
    return opt


def test_the_improving_chart_keeps_every_average_label_off_the_dots(tmp_path, monkeypatch):
    opt = _project(tmp_path, monkeypatch)
    spec = U.improving_chart(opt).to_dict()
    rows = [v for v in spec["datasets"].values() if v and "label_y" in v[0]]
    assert rows, "the Round averages carry their label position"
    labels = rows[0]
    text = json.dumps(spec)
    assert '"field": "label_y"' in text
    # The y domain the chart draws, read off the spec.
    domain = None
    for layer in json.loads(text).get("layer", []):
        for sub in [layer] + layer.get("layer", []):
            d = sub.get("encoding", {}).get("y", {}).get("scale", {}).get("domain")
            if d:
                domain = tuple(d)
    assert domain
    dots = [(int(n), float(s)) for n, s in
            zip(opt.formulation_ids, opt.Y_history)]
    for row in labels:
        assert _clears(row["label_y"], row["avg"], row["mid"], row["text"],
                       [(x, v, U._DOT_R) for x, v in dots], domain,
                       tuple(U._x_domain(opt))), row
    # Round 1's middle dot sits under its bar, so its label is over it.
    first = min(labels, key=lambda r: r["mid"])
    assert first["label_y"] > first["avg"]
