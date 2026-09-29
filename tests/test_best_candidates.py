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
    assert cell == f"{opt.Y_history[0]:.2f}"


def test_repeats_and_nearest_compare_only_the_same_formulations():
    opt = _upgrade_like()
    # Formulation 4 made again as a Repeat of exempt Formulation 1.
    opt.recipe_history[3] = dict(opt.recipe_history[0])
    opt.X_history[3] = list(opt.X_history[0])
    assert insights.repeat_spread(opt) is None
    i, _ = insights.nearest(opt, insights.start_recipe(opt, 0))
    assert i in opt.best_candidates()


def test_the_insights_best_line_names_what_it_is_scored_without():
    opt = _patty(rows=6, cook_loss=False)
    _add_cook_loss(opt)
    facts = insights.best_facts(opt)
    assert facts.without == ["Cook loss"]
    line = wording.best_line("81.0", facts.formulation, facts.round, "Cook loss")
    assert line == (f"Best so far: 81.0, Formulation {facts.formulation} in Round "
                    f"{facts.round} · scored without Cook loss.")
    assert wording.best_line("81.0", 14, "4") == "Best so far: 81.0, Formulation 14 in Round 4."


def test_complete_results_caption_says_dashes_are_not_compared():
    assert wording.COMPLETE_RESULTS_CAPTION == (
        "A formulation counts once all its results are in; “—” means it is "
        "scored without that measurement, so it is not compared for Best so far.")


def test_suggestions_are_unchanged_by_the_new_best():
    """ask() never reads best_index: the same suggestions and model rows."""
    opt = exempt_and_waiting()
    assert opt.best_index() != max(
        (i for i in range(len(opt.Y_history)) if opt.fully_measured(i)),
        key=lambda i: opt.Y_history[i])     # the rule moved the star here
    same(run(opt), fixture()["exempt_and_waiting"])
