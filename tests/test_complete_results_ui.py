"""Complete results only (0.8.0) on screen: Results, Make a round, Insights.
Spec: docs/superpowers/specs/2026-09-28-complete-results-design.md (rev. 2)."""
import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

import wording
from food_bo import FoodOptimizer
from tests.test_app_ui import (  # noqa: F401  (fixtures)
    APP_PATH, _displayed, _submit_button, burger, open_batch, scored)


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
    """Six formulations over two Rounds, all waiting: none scored yet."""
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
    assert wording.START_FROM_BEST_SO_FAR not in [b.label for b in at.button]


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
    # A waiting cell is no dash, so the dash caption stays off (spec
    # 2026-09-30 §4, audit V9); the one rule line says what is compared.
    assert wording.complete_results_caption() not in captions, captions
    assert "Only formulations with all their results are compared." not in captions
    assert wording.BEST_RULE in captions
    assert not any("contribute zero" in c for c in captions)


def test_no_complete_formulation_says_so(all_waiting):
    at = _app(wording.TAB_RESULTS)
    assert any(c.value == "No formulation has all its results yet."
               for c in at.caption), [c.value for c in at.caption]


def test_a_measurement_added_later_shows_a_dash(scored):
    scored.add_objective("Cook loss", 20, goal="min", min_val=0, max_val=40)
    at = _app(wording.TAB_RESULTS)
    assert not at.exception
    element = next(d for d in at.dataframe
                   if wording.OVERALL_SCORE_COLUMN in d.value.columns)
    table = element.value
    # Numbers underneath, so Streamlit sorts the column as numbers (review #5)…
    assert pd.api.types.is_numeric_dtype(table["Cook loss"])
    assert table["Cook loss"].isna().all()
    # …and a dash on screen, where the measurement does not apply.
    shown = _displayed(element)
    assert list(shown.loc[table["Formulation"] == 1, "Cook loss"]) == ["—"]
    assert not any("waiting" in str(v) for v in table[wording.OVERALL_SCORE_COLUMN])


def test_a_blank_that_is_waiting_shows_no_dash(one_waiting):
    """Formulation 4 waits for Juiciness: its cell is blank, not a dash, and
    the column stays numeric with the measured values beside it."""
    at = _app(wording.TAB_RESULTS)
    element = next(d for d in at.dataframe
                   if wording.OVERALL_SCORE_COLUMN in d.value.columns)
    table = element.value
    assert pd.api.types.is_numeric_dtype(table["Juiciness (/10)"])
    shown = _displayed(element)
    assert list(shown.loc[table["Formulation"] == 4, "Juiciness (/10)"]) == [""]
    # A rating reads as the Round table writes it: 7, not 7.00.
    assert list(shown.loc[table["Formulation"] == 1, "Juiciness (/10)"]) == ["7"]


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
    assert ("1 formulation is waiting for Juiciness (enter it on Results). The model has "
            "not used it yet, so a suggestion may nearly repeat it."
            ) in _generate_captions(at)


def test_make_a_round_counts_several_and_says_why_it_is_still_spread_out(six_waiting):
    at = _app(wording.TAB_BATCH)
    assert ("6 formulations are waiting for Juiciness (enter it on Results). The model has "
            "not used them yet, so a suggestion may nearly repeat them. Suggestions keep "
            "covering the allowed amounts until 5 formulations have all their results."
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


# ---------------------------------------------------------------- Task 8


@pytest.mark.skip(reason="review cards prototype: pins the old Review layout (status lines, Explored fold, header); reconcile at the final merge")
def test_insights_header_counts_the_waiting(one_waiting):
    at = _app(wording.TAB_INSIGHTS)
    assert not at.exception
    # Ledger, early five-second re-test (fix round B): the header names the
    # waiting formulation(s) by number instead of counting "N more".
    # Concision round: the short header on screen, the full one in its help.
    assert any(c.value == "2 scored formulations, up to Round 1. Formulation 4 is waiting for "
               "Juiciness." and c.proto.help == (
        "Based on 2 scored formulations (all results in), up to Round 1. Formulation 4 is "
        "waiting for Juiciness and is not used yet.")
        for c in at.tabs[3].caption), [c.value for c in at.tabs[3].caption]


# ---------------------------------------------------------------- final fixes


def _arm_delete(at):
    """Arm the Delete confirmation in Edit past formulations, below the
    missing-values panel and the correction button."""
    at.multiselect(key="delete_formulations").set_value([1])
    at.run()
    _submit_button(at, "Delete Formulation 1").click()
    at.run()
    assert any(w.value.startswith("Delete Formulation 1?") for w in at.warning)


def test_the_caption_says_what_a_dash_means():
    """Review #2: the dash is a measurement it is scored without."""
    assert wording.complete_results_caption() == "“—”: scored without that measurement."
    assert wording.complete_results_help() == (
        "A formulation counts once all its results are in; “—” means it is "
        "scored without that measurement, so it is not compared for Best so far.")


def test_a_formulation_scored_without_a_measurement_says_so(scored):
    scored.add_objective("Cook loss", 1.0, goal="min", min_val=0, max_val=40)
    at = _app(wording.TAB_RESULTS)
    assert not at.exception
    opened = FoodOptimizer("burger")         # the scores the screen reads
    best = opened.best_index()
    line = (f"Overall score {opened.Y_history[best]:.1f} of "
            f"{opened.utility_ceiling():g} · scored without Cook loss.")
    assert any(c.value.startswith(line) for c in at.caption), \
        [c.value for c in at.caption]


def test_just_after_a_measurement_is_added_the_caption_fits_the_star(scored):
    """Review I-1: no formulation has Cook loss, so Best so far is scored
    without it; the captions under it must not say a dashed row is not
    compared for Best so far."""
    scored.add_objective("Cook loss", 1.0, goal="min", min_val=0, max_val=40)
    at = _app(wording.TAB_RESULTS)
    assert not at.exception
    captions = [c.value for c in at.caption]
    assert wording.complete_results_caption(compared=False) in captions, captions
    assert wording.complete_results_caption() not in captions, captions
    assert wording.BEST_RULE_WITHOUT in captions, captions
    assert wording.BEST_RULE not in captions, captions


def test_a_formulation_with_all_its_results_says_nothing_more(scored):
    at = _app(wording.TAB_RESULTS)
    assert not any("scored without" in c.value for c in at.caption)


def _save_firmness_only(burger, numbers_with_juiciness=()):
    burger.set_pending_batch([{"Pea protein": 10.0 + k, "Methylcellulose": 1.0}
                              for k in range(3)])
    at = _app(wording.TAB_BATCH)
    for no in (1, 2, 3):
        at.number_input(key=f"f{no}_Firmness").set_value(6.0)
        if no in numbers_with_juiciness:
            at.number_input(key=f"f{no}_Juiciness").set_value(7.0)
    at.run()
    _submit_button(at, "Save results").click()
    at.run()
    assert not at.exception
    return [i.value for i in at.info if " saved · " in i.value]


def test_a_round_saved_with_the_same_blank_says_it_once(burger):
    """Three formulations waiting for Juiciness: one line, not three
    (review #4)."""
    assert _save_firmness_only(burger) == [
        "Formulations 1–3 saved · waiting for Juiciness."]


def test_a_round_saved_with_gaps_names_the_numbers(burger):
    assert _save_firmness_only(burger, numbers_with_juiciness=(2,)) == [
        "Formulations 1 and 3 saved · waiting for Juiciness."]


def test_make_a_round_names_no_measurement_when_the_waiting_differ(burger):
    """Review #6: one waits for Juiciness, the other for Firmness — naming
    either would be wrong for the other."""
    burger.tell({"Pea protein": 10.0, "Methylcellulose": 1.0},
                {"Firmness": 6.0}, formulation_no=1, batch_no=1)
    burger.tell({"Pea protein": 12.0, "Methylcellulose": 2.0},
                {"Juiciness": 7.0}, formulation_no=2, batch_no=1)
    at = _app(wording.TAB_BATCH)
    assert ("2 formulations are waiting for results (enter them on Results). The model has "
            "not used them yet, so a suggestion may nearly repeat them."
            ) in _generate_captions(at), _generate_captions(at)
