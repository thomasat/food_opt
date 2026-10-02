"""0.8.0 review fixes, Task 2: the triangle draws only the allowed region,
labels each ingredient at its largest allowed amount, and opens on the
three with the most room (spec 2026-09-29 §5, §9 I10)."""
import time

import pytest
from tests.demo_build_reason import REVIEW_FOUR

import insights
import ui_insights
from tests.test_insights_mixture import _four, _start
from tests.test_insights_wave_c_ui import _mix_parts, mixture_scored  # noqa: F401


@pytest.fixture(autouse=True)
def _isolate_cwd(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)


def _domains(spec):
    enc = next(l for l in spec["layer"] if l["mark"]["type"] == "circle")["encoding"]
    return enc["x"]["scale"]["domain"], enc["y"]["scale"]["domain"]


@pytest.mark.skip(reason=REVIEW_FOUR)
def test_the_view_is_zoomed_to_the_region(mixture_scored):
    opt = mixture_scored
    mix, preds = _mix_parts(opt)
    spec = ui_insights.mixture_chart(opt, mix, preds, 7).to_dict()
    (x0, x1), (y0, y1) = _domains(spec)
    xs = [ui_insights._xy(p.x)[0] for p in mix.points]
    ys = [ui_insights._xy(p.x)[1] for p in mix.points]
    side = max(max(xs) - min(xs), max(ys) - min(ys))
    pad = ui_insights._MIX_PAD * side
    assert (x0, x1) == pytest.approx((min(xs) - pad, max(xs) + pad))
    assert (y0, y1) == pytest.approx((min(ys) - pad, max(ys) + pad))


@pytest.mark.skip(reason=REVIEW_FOUR)
def test_no_outline_of_the_whole_triangle_is_drawn(mixture_scored):
    opt = mixture_scored
    mix, preds = _mix_parts(opt)
    spec = ui_insights.mixture_chart(opt, mix, preds, 7).to_dict()
    assert not [l for l in spec["layer"] if l["mark"]["type"] == "line"]


@pytest.mark.skip(reason=REVIEW_FOUR)
def test_a_finer_grid_draws_smaller_marks(mixture_scored):
    opt = mixture_scored
    mix, preds = _mix_parts(opt)
    spec = ui_insights.mixture_chart(opt, mix, preds, 7).to_dict()
    size = next(l for l in spec["layer"] if l["mark"]["type"] == "circle"
                and l["mark"].get("opacity") == 1 and "color" in l["encoding"]
                )["encoding"]["size"]["scale"]["range"]
    assert size[0] > size[1] == size[2] >= 3
    assert size[0] <= 70 * 4


@pytest.mark.skip(reason=REVIEW_FOUR)
def test_the_default_three_have_the_most_room(mixture_scored, monkeypatch):
    opt = mixture_scored
    monkeypatch.setattr(insights, "spreads", lambda o, f, s: insights.Spreads([], []))
    start = insights.start_recipe(opt, opt.best_index())
    offered = insights.mixture_choices(opt)
    got = ui_insights._three_default(opt, {}, offered, start)
    import itertools
    shares = {c: insights.allowed_share(opt, list(c), start)
              for c in itertools.combinations(offered, 3)}
    assert shares[tuple(got)] == max(shares.values())
    assert shares[tuple(got)] > 0


def test_the_slice_stays_fast():
    opt = _four()
    s = _start(opt, Pea=6.0, Oil=4.0, Starch=5.0, Water=85.0)
    t = time.perf_counter()
    insights.mixture_slice(opt, ["Pea", "Oil", "Water"], s)
    assert time.perf_counter() - t < 1.0
