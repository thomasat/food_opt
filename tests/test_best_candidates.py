"""0.8.0 review fixes, Task 1: Best so far is chosen only among formulations
with a value for every current measurement (spec 2026-09-29 §1.2, §9 B3/I4).
A formulation scored without a measurement keeps its rescaled score and
still teaches the suggestions; it is never Best so far while another
formulation has every measurement."""
import pytest

import insights
import wording
from tests.test_ask_unchanged import exempt_and_waiting, fixture, run, same
from tests.test_complete_results import _add_cook_loss, _patty, _tell


@pytest.fixture(autouse=True)
def _isolate_cwd(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)


def _upgrade_like():
    """Rows 1-3 recorded before Cook loss existed (scored without it, and
    higher for it: Cook loss never reaches full closeness), rows 4-5 with it."""
    opt = _patty(rows=3, cook_loss=False)
    _add_cook_loss(opt)
    _tell(opt, 4, 2)
    _tell(opt, 5, 2)
    return opt


def test_a_formulation_scored_without_a_measurement_is_never_best():
    opt = _upgrade_like()
    exempt = [0, 1, 2]
    assert all(opt.fully_measured(i) and opt.not_expected(i) for i in exempt)
    assert max(opt.Y_history[i] for i in exempt) > max(opt.Y_history[3:])
    assert opt.best_candidates() == [3, 4]
    assert opt.best_index() == max((3, 4), key=lambda i: opt.Y_history[i])
    assert opt.best_scored_without() == []


def test_with_none_measured_on_everything_the_best_is_scored_without_it():
    """Adding a measurement exempts every recorded row: the project keeps a
    Best so far (never the cold-start wording) and says what it lacks."""
    opt = _patty(rows=6, cook_loss=False)
    before = opt.best_index()
    _add_cook_loss(opt)
    assert opt.best_candidates() == list(range(6))
    assert opt.best_index() == before
    assert opt.best_scored_without() == ["Cook loss"]
    assert opt.compared_with_column() != wording.COMPARED_WITH_ALLOWED
    # The first formulation measured on it takes the star (owner to judge).
    _tell(opt, 7, 3)
    assert opt.best_index() == 6
    assert opt.best_scored_without() == []


def test_best_so_far_runs_over_the_same_formulations():
    opt = _upgrade_like()
    run_max = opt.best_so_far()
    assert run_max[:3] == [None, None, None]
    assert run_max[3] == pytest.approx(opt.Y_history[3])
    assert run_max[4] == pytest.approx(max(opt.Y_history[3], opt.Y_history[4]))


def test_best_first_sorts_scored_without_after_every_measurement():
    opt = _upgrade_like()
    _tell(opt, 6, 2, drop=("Cook loss",))           # waiting
    frame = opt.history_frame()
    order = frame["Formulation"].tolist()
    assert order[:2] == sorted((4, 5), key=lambda n: -opt.Y_history[n - 1])
    assert set(order[2:5]) == {1, 2, 3}
    assert order[5] == 6
    assert frame[wording.BEST_SO_FAR_COLUMN].tolist()[0] == "★"
    # The score cell stays a number: the tail lives in sentences only.
    cell = frame.loc[frame["Formulation"] == 1, wording.OVERALL_SCORE_COLUMN].item()
    assert cell == f"{opt.Y_history[0]:.1f}"


def test_repeats_and_nearest_compare_only_the_same_formulations():
    opt = _upgrade_like()
    # Formulation 4 made again as a Repeat of exempt Formulation 1.
    opt.recipe_history[3] = dict(opt.recipe_history[0])
    opt.X_history[3] = list(opt.X_history[0])
    assert insights.repeat_sd(opt, insights.SCORE).sd is None
    i, _ = insights.nearest(opt, insights.start_recipe(opt, 0))
    assert i in opt.best_candidates()


def test_the_insights_best_line_names_what_it_is_scored_without():
    opt = _patty(rows=6, cook_loss=False)
    _add_cook_loss(opt)
    facts = insights.best_facts(opt)
    assert facts.without == ["Cook loss"]
    line = wording.stand_best("81.00", facts.formulation, facts.round, "Cook loss")
    assert line == (f"Best so far: Overall score 81.00, Formulation {facts.formulation} "
                    f"(Round {facts.round}) · scored without Cook loss.")
    assert wording.stand_best("81.00", 14, "4") == (
        "Best so far: Overall score 81.00, Formulation 14 (Round 4).")


def test_complete_results_caption_says_dashes_are_not_compared():
    assert wording.COMPLETE_RESULTS_CAPTION == "“—”: scored without that measurement."


def test_suggestions_are_unchanged_by_the_new_best():
    """ask() never reads best_index: the same suggestions and model rows."""
    opt = exempt_and_waiting()
    assert opt.best_index() != max(
        (i for i in range(len(opt.Y_history)) if opt.fully_measured(i)),
        key=lambda i: opt.Y_history[i])     # the rule moved the star here
    same(run(opt), fixture()["exempt_and_waiting"])


def test_the_fallback_caption_does_not_contradict_the_star():
    """Review I-1: while no formulation has every measurement, Best so far
    has a dash itself, so the caption must not say dashed rows are never
    compared."""
    assert wording.complete_results_caption() == wording.COMPLETE_RESULTS_CAPTION
    assert wording.complete_results_caption(compared=False) == (
        "“—”: scored without that measurement, as is Best so far.")


def test_every_new_string_passes_the_vocabulary_guard():
    from tests.test_food_bo import _BANNED
    strings = [wording.COMPLETE_RESULTS_CAPTION,
               wording.complete_results_caption(compared=False)]
    for s in strings:
        for pattern in _BANNED:
            assert not pattern.search(s), (pattern.pattern, s)


# Written from ce205c9 (before the review fixes) and never regenerated.
_TRAJECTORY_AT_CE205C9 = """\
In play (2): Pea, Oil

  Formulation 1: score 77.576 | Pea=4.37, Oil=2.61 | results: Firmness=5.93, Juiciness=4.13
  Formulation 2: score 79.077 | Pea=4.74, Oil=3.22 | results: Firmness=6.27, Juiciness=4.86
* Formulation 3: score 79.741 | Pea=5.11, Oil=3.83 | results: Firmness=6.6, Juiciness=5.6
  Formulation 4: score 77.544 | Pea=5.48, Oil=4.44 | results: Firmness=6.93, Juiciness=6.33, Cook loss=13.6
  Formulation 5: score 78.630 | Pea=5.85, Oil=5.05 | results: Firmness=7.26, Juiciness=7.06, Cook loss=12.5"""


def test_the_expert_trajectory_keeps_the_old_star():
    """Review M-3: export_trajectory feeds the adaptive expert (the
    suggestion side), so it keeps the pre-0.8.0 rule: exempt Formulation 3
    outscores Formulation 5 and keeps the '*', though the screens star 5."""
    opt = _upgrade_like()
    assert opt.best_formulation_no() == 5
    assert opt.export_trajectory() == _TRAJECTORY_AT_CE205C9
