"""0.8.0 speed and simplicity, Plan B, Task 7: the "Try one ingredient or
process setting" fold ranks near what has been tested and says so against
the yardstick, Compare all draws the tested-near change, the path has two
captions, and "More charts" has no misses chart (spec 2026-09-30 §2.5;
review I8, M3, M15, M19)."""
import pytest
from streamlit.testing.v1 import AppTest

import insights
import ui_insights
import wording
from food_bo import FoodOptimizer
from tests.test_app_ui import APP_PATH, burger  # noqa: F401
from tests.test_insights_ui import _open_insights, insights_scored  # noqa: F401
from tests.test_insights_wave_b_ui import ferment, mixed_scored, warm_ferment  # noqa: F401
from tests.test_insights_wave_c_ui import mixture_scored  # noqa: F401


def _app():
    return AppTest.from_file(APP_PATH, default_timeout=300)


def _changes(monkeypatch, *rows, held=(), start_tested=True, yard=1.0):
    """tested_near_changes stood in: rows of (name, near, whole)."""
    monkeypatch.setattr(insights, "tested_near_changes",
                        lambda opt, fitted, start, name=insights.SCORE: insights.NearChanges(
                            [insights.NearChange(n, a, b) for n, a, b in rows], list(held),
                            start_tested))
    monkeypatch.setattr(insights, "noise_yardstick",
                        lambda opt, checks=(), name=insights.SCORE:
                        insights.Yardstick(yard, None, 0, False))


def _fold(at, label=None):
    label = wording.try_one(False) if label is None else label
    return next(e for e in at.tabs[3].expander if e.label == label)


def _said(fold):
    return [m.value for m in fold.markdown]


def test_the_top_change_near_what_was_tested_is_named(insights_scored, monkeypatch):
    _changes(monkeypatch, ("Methylcellulose", 1.4, 1.45), ("Pea protein", 0.3, 4.0))
    at = _open_insights(_app())
    assert not at.exception
    said = _said(_fold(at))
    assert said[0] == wording.near_line("Methylcellulose", "1.4", "Best so far")
    assert wording.near_bigger() not in said                      # 1.45 is not 0.1 more


def test_bigger_changes_away_from_the_tested_are_said(insights_scored, monkeypatch):
    _changes(monkeypatch, ("Pea protein", 1.4, 5.0), ("Methylcellulose", 0.3, 0.4))
    at = _open_insights(_app())
    said = _said(_fold(at))
    assert said[:2] == [wording.near_line("Pea protein", "1.4", "Best so far"),
                        wording.near_bigger()]


def test_two_nearly_tied_are_named_together(insights_scored, monkeypatch):
    _changes(monkeypatch, ("Pea protein", 1.5, 1.5), ("Methylcellulose", 1.3, 1.3))
    at = _open_insights(_app())
    assert _said(_fold(at))[0] == wording.near_two("Pea protein", "Methylcellulose", "1.5",
                                                   "1.3", "Best so far")


def test_a_tie_is_read_on_the_tested_near_change(insights_scored, monkeypatch):
    """Only the tested-near changes tie (1.3 of 1.5): named together."""
    _changes(monkeypatch, ("Pea protein", 1.5, 6.0), ("Methylcellulose", 1.3, 1.3))
    at = _open_insights(_app())
    assert _said(_fold(at))[0] == wording.near_two("Pea protein", "Methylcellulose", "1.5",
                                                   "1.3", "Best so far")


def test_whole_path_changes_that_tie_are_no_tie(insights_scored, monkeypatch):
    """The whole-path changes tie (1.4 of 1.5), the tested-near ones do not."""
    _changes(monkeypatch, ("Pea protein", 1.5, 1.5), ("Methylcellulose", 0.5, 1.4))
    at = _open_insights(_app())
    assert _said(_fold(at))[0] == wording.near_line("Pea protein", "1.5", "Best so far")


def test_stands_out_is_judged_by_the_yardstick(insights_scored, monkeypatch):
    """Repeats differ by 2 points: a 1.4-point change does not stand out."""
    _changes(monkeypatch, ("Pea protein", 1.4, 1.5), ("Methylcellulose", 0.3, 0.4), yard=2.0)
    at = _open_insights(_app())
    assert _said(_fold(at))[0] == wording.near_none(False, False)


def test_nothing_beyond_the_yardstick_stands_out(insights_scored, monkeypatch):
    """The Line C-like case (review M15): changes smaller than 1 point."""
    _changes(monkeypatch, ("Pea protein", 0.6, 3.0), ("Methylcellulose", 0.2, 0.9))
    at = _open_insights(_app())
    said = _said(_fold(at))
    assert said[0] == wording.near_none(False, False)
    assert "Near what you have tested, no ingredient stands out yet." == said[0]


def test_a_start_not_tested_near_ranks_nothing(insights_scored, monkeypatch):
    _changes(monkeypatch, ("Pea protein", None, 3.0), ("Methylcellulose", None, 0.9),
             start_tested=False)
    at = _open_insights(_app())
    assert _said(_fold(at))[0] == wording.near_start_untested("Best so far")


def test_the_box_opens_on_the_top_change_near_what_was_tested(insights_scored, monkeypatch):
    _changes(monkeypatch, ("Methylcellulose", 1.4, 1.5), ("Pea protein", 0.3, 9.0))
    at = _open_insights(_app())
    assert at.tabs[3].selectbox(key=ui_insights.CHOICE_KEY).value == "Methylcellulose"


def test_compare_all_draws_the_tested_near_change(insights_scored, monkeypatch):
    _changes(monkeypatch, ("Methylcellulose", 1.4, 1.5), ("Pea protein", 0.3, 9.0))
    at = _open_insights(_app())
    compare = next(e for e in at.tabs[3].expander if e.label == wording.compare_all(False))
    assert [c.value for c in compare.caption] == [wording.change_caption()]


def test_the_path_has_two_captions(insights_scored):
    at = _open_insights(_app())
    caps = [c.value for c in _fold(at).caption]
    opt = FoodOptimizer("burger")
    name = at.tabs[3].selectbox(key=ui_insights.CHOICE_KEY).value
    best = opt.best_formulation_no()
    first = next(i for i, c in enumerate(caps) if c.startswith(f"More {name}"))
    assert caps[first].endswith(".") and caps[-1].endswith(f"Dotted: Formulation {best}; dashed: target.")
    assert not any(c.startswith("Stops at ") for c in caps)        # merged into the first
    assert not any(c.startswith("Dotted line: ") for c in caps)     # the old key line


def test_the_whole_answer_needs_no_spread(insights_scored):
    at = _open_insights(_app())
    import re
    assert not [m.value for m in at.tabs[3].markdown if re.search(r"\bspreads?\b", m.value)]


def test_more_charts_has_no_misses_chart(mixture_scored):
    at = _open_insights(_app())
    assert not at.exception
    more = next(e for e in at.tabs[3].expander if e.label == wording.MORE_CHARTS)
    assert [m.value for m in more.markdown] == [f"**{wording.Q_PAIR}**",
                                                 f"**{wording.Q_MIXTURE}**"]
    assert not hasattr(insights, "misses") and not hasattr(ui_insights, "misses_chart")


def test_the_limits_held_in_place_are_named_under_the_line(insights_scored, monkeypatch):
    _changes(monkeypatch, ("Pea protein", 1.4, 1.5), held=["Methylcellulose"])
    at = _open_insights(_app())
    fold = _fold(at)
    assert _said(fold)[0] == wording.near_line("Pea protein", "1.4", "Best so far", only=True)
    assert wording.held_line("Best so far", ["Methylcellulose"]) in [c.value for c in fold.caption]
    # One choice: nothing to compare.
    assert wording.compare_all(False) not in [e.label for e in at.tabs[3].expander]


def test_everything_held_says_so(insights_scored, monkeypatch):
    _changes(monkeypatch, held=["Pea protein", "Methylcellulose"])
    at = _open_insights(_app())
    assert _said(_fold(at))[0] == wording.held_all("Best so far")


def test_another_start_is_named(insights_scored, monkeypatch):
    _changes(monkeypatch, ("Pea protein", 1.4, 1.5), ("Methylcellulose", 0.2, 0.3))
    at = _open_insights(_app())
    box = at.tabs[3].selectbox(key=ui_insights.START_KEY)
    other = box.options[1]
    box.set_value(int(other.split()[-1]) if isinstance(other, str) else other)
    at.session_state["main_tab"] = wording.TAB_INSIGHTS
    at.run()
    no = at.tabs[3].selectbox(key=ui_insights.START_KEY).value
    assert _said(_fold(at))[0] == wording.near_line("Pea protein", "1.4", f"Formulation {no}")


# ------------------------------------------------------------------ #
#  A process-only study and a mixed one (review I14, Task 7 review I-2)
# ------------------------------------------------------------------ #

_SETTINGS = ("Incubation temperature", "Incubation time")


def _ferment_fold(at, opt):
    return _fold(at, wording.for_project(opt, wording.try_one(True, False)))


def test_a_process_only_project_says_no_process_setting_stands_out(warm_ferment, monkeypatch):
    _changes(monkeypatch, (_SETTINGS[0], 0.6, 3.0), (_SETTINGS[1], 0.2, 0.9))
    at = _open_insights(_app())
    assert not at.exception
    fold = _ferment_fold(at, warm_ferment)
    assert fold.label == "Try one process setting"
    assert _said(fold)[0] == wording.for_project(warm_ferment, wording.near_none(True, True))
    assert _said(fold)[0] == "Near what you have tested, no process setting stands out yet."


def test_a_process_only_project_names_the_setting_and_its_values(warm_ferment, monkeypatch):
    _changes(monkeypatch, (_SETTINGS[0], 1.4, 5.0), (_SETTINGS[1], 0.3, 0.4))
    at = _open_insights(_app())
    assert not at.exception
    said = _said(_ferment_fold(at, warm_ferment))
    assert said[:2] == [wording.near_line(_SETTINGS[0], "1.4", "Best so far"),
                        wording.near_bigger(values=True)]
    assert "values not tested yet" in said[1]
    compare = next(e for e in at.tabs[3].expander
                   if e.label == wording.for_project(warm_ferment, wording.compare_all(True, False)))
    assert compare.label == "Compare all process settings"
    assert [c.value for c in compare.caption] == [
        wording.for_project(warm_ferment, wording.change_caption(True))]
    assert "allowed values" in compare.caption[0].value


def test_a_process_only_project_with_every_setting_held(warm_ferment, monkeypatch):
    _changes(monkeypatch, held=list(_SETTINGS))
    at = _open_insights(_app())
    assert not at.exception
    assert _said(_ferment_fold(at, warm_ferment))[0] == wording.for_project(
        warm_ferment, wording.held_all("Best so far", True))


@pytest.mark.parametrize("top,values", [("Mixing time", True), ("Pea protein", False)])
def test_values_for_a_setting_and_amounts_for_an_ingredient(mixed_scored, monkeypatch,
                                                            top, values):
    other = "Methylcellulose"
    _changes(monkeypatch, (top, 1.4, 5.0), (other, 0.3, 0.4))
    at = _open_insights(_app())
    assert not at.exception
    said = _said(_fold(at, wording.try_one(True, True)))
    assert said[:2] == [wording.near_line(top, "1.4", "Best so far"),
                        wording.near_bigger(values=values)]
    assert ("values" if values else "amounts") in said[1]
