"""Insights wave B, computation: predictions for the open Round, the spread
of each ingredient or process setting, and the formulation predicted to
score highest. Nothing here may write to the project or move the RNG."""
import pytest

import insights
from tests.test_insights import _burger


@pytest.fixture(autouse=True)
def _isolate_cwd(tmp_path, monkeypatch):
    # tell() saves "<project_name>.pkl" into the cwd.
    monkeypatch.chdir(tmp_path)


# ------------------------------------------------------------------ #
#  Task 1: predictions for the open Round
# ------------------------------------------------------------------ #


def test_at_total_leaves_a_formulation_at_the_total_alone():
    opt = _burger(rounds=2)
    r = opt.fill_formulas({"Pea": 6.0, "Oil": 4.0})
    assert insights.at_total(opt, r) == pytest.approx(r)


def test_at_total_brings_a_resized_formulation_back():
    opt = _burger(rounds=2)
    big = {k: v * 1.5 for k, v in opt.fill_formulas({"Pea": 6.0, "Oil": 4.0}).items()}
    back = insights.at_total(opt, big)
    assert back["Pea"] == pytest.approx(6.0) and back["Oil"] == pytest.approx(4.0)
    assert back["Water"] == pytest.approx(90.0)


# ------------------------------------------------------------------ #
#  Task 3: which ingredients matter most
# ------------------------------------------------------------------ #


#  Task 4: the formulation predicted to score highest
# ------------------------------------------------------------------ #


def test_distance_is_the_largest_share_of_each_span():
    opt = _burger(rounds=2)                          # Pea 4–8, Oil 2–6: spans 4
    a = opt.fill_formulas({"Pea": 6.0, "Oil": 4.0})
    assert insights.distance(opt, a, opt.fill_formulas({"Pea": 6.6, "Oil": 4.2})) == pytest.approx(0.15)
    assert insights.distance(opt, a, opt.fill_formulas({"Pea": 6.0, "Oil": 3.0})) == pytest.approx(0.25)


def _categorical(opt):
    # No public API makes a categorical row any more (old files only), so it
    # is appended the way tests/test_insights.py's path-choice test does.
    opt.variables.append({'name': "Package", 'type': 'categorical',
                          'options': ["Tray", "Pouch"], 'category': 'process'})
    return opt


def test_a_choice_that_differs_is_never_close():
    opt = _categorical(_burger(rounds=0))
    a = dict(opt.fill_formulas({"Pea": 6.0, "Oil": 4.0}), Package="Tray")
    assert insights.distance(opt, a, dict(a, Package="Pouch")) is None
    assert insights.distance(opt, a, dict(a)) == 0.0


def test_setup_key_follows_every_limit_but_not_a_rerun():
    opt = _burger(rounds=2)
    k = insights.setup_key(opt)
    assert insights.setup_key(opt) == k
    opt.add_quantity_constraint(["Pea", "Oil"], max_val=11.0)
    k2 = insights.setup_key(opt)
    assert k2 != k
    opt.add_property("Fat")
    opt.set_property_value("Oil", "Fat", 100.0)
    k3 = insights.setup_key(opt)
    assert k3 != k2
    opt.set_formulation_total(120)
    assert insights.setup_key(opt) != k3
