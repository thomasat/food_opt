"""Complete results only (0.8.0) on screen: Results, Make a round, Insights.
Spec: docs/superpowers/specs/2026-09-28-complete-results-design.md (rev. 2)."""
import pytest
from streamlit.testing.v1 import AppTest

import wording
from food_bo import FoodOptimizer
from tests.test_app_ui import (  # noqa: F401  (fixtures)
    APP_PATH, _submit_button, burger, open_batch, scored)


def _app(tab=None):
    at = AppTest.from_file(APP_PATH, default_timeout=180)
    at.run()
    if tab is not None:
        at.session_state["main_tab"] = tab
        at.run()
    return at


def _firmness_only(opt, numbers, round_no=1):
    """Formulations with Firmness measured and Juiciness still to come."""
    for no in numbers:
        opt.tell({"Pea protein": 8.0 + no, "Methylcellulose": 0.5 + 0.2 * no},
                 {"Firmness": 3.0 + 0.3 * no}, formulation_no=no,
                 batch_no=round_no)


@pytest.fixture
def all_waiting(burger):
    """Two formulations, neither with Juiciness yet: no best at all."""
    _firmness_only(burger, (1, 2))
    return burger


@pytest.fixture
def one_waiting(scored):
    """`scored` (1 and 2 complete, 3 Not scored) plus 4, waiting for
    Juiciness, in Round 2."""
    _firmness_only(scored, (4,), round_no=2)
    return scored


@pytest.fixture
def six_waiting(burger):
    """Six formulations over two Rounds, all waiting: Insights opens."""
    _firmness_only(burger, (1, 2, 3), 1)
    _firmness_only(burger, (4, 5, 6), 2)
    return burger


# ---------------------------------------------------------------- Task 4


def test_results_with_no_complete_formulation_names_no_best(all_waiting):
    at = _app(wording.TAB_RESULTS)
    assert not at.exception
    assert not any(h.value.startswith("Best so far:") for h in at.subheader), \
        [h.value for h in at.subheader]


def test_make_a_round_with_no_complete_formulation_offers_no_best(all_waiting):
    at = _app(wording.TAB_BATCH)
    assert not at.exception
    assert wording.START_FROM_BEST not in [b.label for b in at.button]


def test_insights_with_no_complete_formulation(six_waiting):
    at = _app(wording.TAB_INSIGHTS)
    assert not at.exception
    start = at.tabs[3].selectbox(key="insights_start")
    assert start.options[0] == wording.start_formulation(6)


# ---------------------------------------------------------------- Task 6


def _overall(at):
    table = next(d.value for d in at.dataframe
                 if wording.OVERALL_SCORE_COLUMN in d.value.columns)
    return dict(zip(table["Formulation"], table[wording.OVERALL_SCORE_COLUMN]))


def test_the_table_says_what_a_waiting_formulation_waits_for(one_waiting):
    at = _app(wording.TAB_RESULTS)
    assert not at.exception
    assert _overall(at)[4] == "waiting for Juiciness"
    captions = [c.value for c in at.caption]
    assert sum(c == wording.COMPLETE_RESULTS_CAPTION for c in captions) == 1, captions
    assert wording.RESULT_BEST_COMPLETE_ONLY in captions
    assert wording.RESULT_BEST_COMPLETE_ONLY == \
        "Only formulations with all their results are compared."
    assert not any("contribute zero" in c for c in captions)


def test_no_complete_formulation_says_so(all_waiting):
    at = _app(wording.TAB_RESULTS)
    assert any(c.value == "No formulation has all its results yet."
               for c in at.caption), [c.value for c in at.caption]


def test_a_selected_waiting_formulation_offers_score_without(one_waiting):
    at = _app(wording.TAB_RESULTS)
    at.selectbox(key="result_view_burger").set_value(4)
    at.run()
    captions = [c.value for c in at.caption]
    assert "Overall score waiting for Juiciness." in captions, captions
    button = _submit_button(at, "Score without Juiciness")
    assert button.proto.type == "secondary"
    button.click()
    at.run()
    assert not at.exception
    assert any(s.value == "Formulation 4 scored without Juiciness."
               for s in at.success), [s.value for s in at.success]
    reloaded = FoodOptimizer("burger")
    assert reloaded.fully_measured(reloaded.index_of_formulation(4))


def test_a_measurement_added_later_shows_a_dash(scored):
    scored.add_objective("Cook loss", 20, goal="min", min_val=0, max_val=40)
    at = _app(wording.TAB_RESULTS)
    assert not at.exception
    table = next(d.value for d in at.dataframe
                 if wording.OVERALL_SCORE_COLUMN in d.value.columns)
    assert list(table.loc[table["Formulation"] == 1, "Cook loss"]) == ["—"]
    assert not any("waiting" in str(v) for v in table[wording.OVERALL_SCORE_COLUMN])


def test_correcting_a_waiting_formulation_says_what_it_still_waits_for(one_waiting):
    at = _app(wording.TAB_RESULTS)
    at.selectbox(key="correct_formulation").set_value(4)
    at.run()
    at.number_input(key="correct_4_Firmness").set_value(6.0)
    at.run()
    _submit_button(at, "Save correction").click()
    at.run()
    assert any(s.value.startswith("Formulation 4 corrected · waiting for Juiciness.")
               for s in at.success), [s.value for s in at.success]


def test_saving_a_round_with_a_blank_names_what_it_waits_for(open_batch):
    at = _app(wording.TAB_BATCH)
    at.number_input(key="f1_Firmness").set_value(6.0)
    at.number_input(key="f2_Firmness").set_value(5.0)
    at.number_input(key="f2_Juiciness").set_value(7.0)
    at.run()
    _submit_button(at, "Save results").click()
    at.run()
    assert not at.exception
    assert any(i.value == "Formulation 1 saved · waiting for Juiciness."
               for i in at.info), [i.value for i in at.info]
    assert not any("Formulation 2 saved" in i.value for i in at.info)


# ---------------------------------------------------------------- Task 7


def _generate_captions(at):
    return [c.value for c in at.tabs[1].caption]


def test_make_a_round_says_nothing_while_nothing_waits(scored):
    at = _app(wording.TAB_BATCH)
    assert not any("waiting for" in c for c in _generate_captions(at))


def test_make_a_round_names_one_waiting_formulation(one_waiting):
    at = _app(wording.TAB_BATCH)
    assert ("1 formulation is waiting for Juiciness and is not used for "
            "suggestions yet. Enter its results on Results.") in _generate_captions(at)


def test_make_a_round_counts_several_and_says_why_it_is_still_spread_out(six_waiting):
    at = _app(wording.TAB_BATCH)
    assert ("6 formulations are waiting for Juiciness and are not used for "
            "suggestions yet. Enter their results on Results. Suggestions stay "
            "spread out until 5 formulations have all their results."
            ) in _generate_captions(at), _generate_captions(at)


def test_a_row_recorded_before_a_measurement_was_added_is_not_missing_it(open_batch):
    open_batch.tell({"Pea protein": 10.0, "Methylcellulose": 1.0},
                    {"Juiciness": 7.0, "Firmness": 6.0}, formulation_no=1,
                    batch_no=open_batch.pending_batch_no)
    open_batch.add_objective("Cook loss", 20, goal="min", min_val=0, max_val=40)
    at = _app(wording.TAB_BATCH)
    assert not at.exception
    assert not any("Cook loss not measured" in c.value for c in at.caption), \
        [c.value for c in at.caption]
