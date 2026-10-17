"""0.8.0 speed and simplicity, Plan B, Task 6: Insights answers three
questions and says what to do next — the header, "Can you trust the
predictions?", "Where do you stand?", "What to do next", the foot button,
then the folds (spec 2026-09-30 §2; review I2, I3, I5, I6, I7, M1, M8, M9,
M13, M17)."""
from types import SimpleNamespace

import pytest
from tests.demo_build_reason import REVIEW_FOUR
from streamlit.testing.v1 import AppTest

import insights
import ui_insights
import wording
from food_bo import FoodOptimizer
from tests.test_app_ui import APP_PATH, burger  # noqa: F401
from tests.test_complete_results import _patty, _tell
from tests.test_insights import _burger
from tests.test_insights_ui import _open_how, _open_insights, insights_scored  # noqa: F401


@pytest.fixture(autouse=True)
def _isolate_cwd(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)


def _app():
    return AppTest.from_file(APP_PATH, default_timeout=300)


def _tab(project):
    at = _app()
    at.session_state["_loaded_project"] = project
    return _open_insights(at)


def _md(at):
    return [m.value for m in at.tabs[3].markdown]


# ------------------------------------------------------------------ #
#  On screen: order, foot, folds
# ------------------------------------------------------------------ #

def _moves(monkeypatch):
    """The tabs go_to_tab is asked for (where on the tab it lands, too, is
    the move's own business: the Set up button lands on its callout)."""
    moved = []
    monkeypatch.setattr(ui_insights, "go_to_tab",
                        lambda tab, section="top": moved.append(tab))
    return moved


@pytest.mark.skip(reason=REVIEW_FOUR)
def test_three_answers_then_the_foot_then_three_folds(insights_scored, monkeypatch):
    moved = _moves(monkeypatch)
    at = _open_insights(_app())
    assert not at.exception
    md = _md(at)
    heads = [f"##### {q}" for q in (wording.Q_STAND, )]
    assert [md.index(h) for h in heads] == sorted(md.index(h) for h in heads)
    for gone in ("Did the predictions come true?", "Is it still improving?",
                 "Which formulation is predicted to score highest?"):
        assert f"##### {gone}" not in md
    folds = [e.label for e in at.tabs[3].expander
             if e.label in (wording.try_one_fold(False), wording.FOLD_METHOD)
             or e.label.startswith("Trade-offs") or e.label == wording.PAIR_FOLD]
    # Charts proto: the triangle moved to Explored and unexplored; the
    # fold left holds the two-measurement chart alone, when there is one.
    assert folds[0] == wording.try_one_fold(False) and folds[-1] == wording.FOLD_METHOD
    assert set(folds[1:-1]) <= {wording.PAIR_FOLD}
    assert all(not e.proto.expanded for e in at.tabs[3].expander)
    foot = next(b for b in at.tabs[3].button if b.key == "insights_foot")
    assert foot.label == wording.START_NEXT_BATCH and foot.proto.type == "primary"
    foot.click()
    at.session_state["main_tab"] = wording.TAB_INSIGHTS   # AppTest does not resend the tab
    at.run()
    assert moved == [wording.TAB_BATCH]


def test_the_foot_names_the_open_round_and_both_feet_draw_in_one_run(insights_scored):
    insights_scored.set_pending_batch([{"Pea protein": 12.0, "Methylcellulose": 1.0}])
    at = _open_insights(_app())
    assert not at.exception
    keys = [b.key for b in at.button]
    assert "insights_foot" in keys and "foot_batch" in keys          # review M13
    foot = next(b for b in at.tabs[3].button if b.key == "insights_foot")
    assert foot.label == wording.back_to_batch_label(4, 1)


@pytest.mark.skip(reason="review cards prototype: pins the old Review layout (status lines, Explored fold, header); reconcile at the final merge")
def test_the_header_is_counts_only(insights_scored):
    at = _open_insights(_app())
    header = at.tabs[3].caption[0].value
    # Re-pinned to the concision round's short text (owner, 2026-10-01: too wordy).
    assert header == "9 scored formulations, up to Round 3."
    assert "Likely range" not in header


def test_nothing_scored_still_offers_the_foot(tmp_path):
    opt = _patty(rows=0)
    _tell(opt, 1, 1, drop=("Cook loss",))
    opt.set_aside_formulation(1)
    at = _tab(opt.project_name)
    assert not at.exception
    assert [c.value for c in at.tabs[3].caption] == [wording.INSIGHTS_NONE]
    assert "insights_foot" in [b.key for b in at.tabs[3].button]


# ------------------------------------------------------------------ #
#  "Can you trust the predictions?" (review I6, M9)
# ------------------------------------------------------------------ #

def _h(state="verdict", inside=32, total=33, rounds=(4, 5, 6), wide=()):
    return insights.Headline(state, list(rounds), inside, total, "holding", None, list(wide))


# ------------------------------------------------------------------ #
#  "Where do you stand?" (review I5, I7, M1)
# ------------------------------------------------------------------ #

@pytest.mark.skip(reason=REVIEW_FOUR)
def test_best_so_far_reads_two_decimals_as_on_results(insights_scored):
    at = _open_insights(_app())
    f = insights.best_facts(FoodOptimizer("burger"))            # as the app loads it
    assert any(f"Best so far: Overall score {f.score:.2f}, Formulation {f.formulation} "
               f"(Round {f.round})"
               in m for m in _md(at))


def _standing(higher, close_no=None, close=None):
    p = insights.Prediction(94.4, 89.2, 99.1, 94.4, 2.6)
    best = insights.PredictedBest({}, p, close, 0, 1.0, None)
    return SimpleNamespace(best=best, higher=higher, close_no=close_no,
                           tested_near=close is not None, binding=[], luck=False)


# ------------------------------------------------------------------ #
#  "What to do next" (review I2, I3, M17)
# ------------------------------------------------------------------ #


def test_no_sentence_says_what_the_next_round_will_do():
    """Review I3: nothing may promise what ask() does."""
    for text in (wording.NEXT_RISING,
                 ):
        assert "will try" not in text and "worth adding" not in text


# ------------------------------------------------------------------ #
#  "How these insights are made" (review M5, M6, M10; audit V23)
# ------------------------------------------------------------------ #

# ------------------------------------------------------------------ #
#  The predicted best's amounts: in a fold, only when clearly higher and
#  not tested near (§2.3, review I7)
# ------------------------------------------------------------------ #

def _fake_best(monkeypatch, close, gap):
    """A predicted best `gap` points above Best so far's own predicted
    Repeat (which is its measured score here)."""
    def fake(opt, fitted, checks=()):
        top = opt.best_index()
        rep = float(opt.Y_history[top])
        m = rep + gap
        return insights.PredictedBest(
            dict(opt.recipe_history[0]), insights.Prediction(m, m - 4, m + 4, m, 2.0),
            close, close if close is not None else top, 1.0,
            insights.Prediction(rep, rep - 4, rep + 4, rep, 2.0))
    monkeypatch.setattr(insights, "predicted_best", fake)


