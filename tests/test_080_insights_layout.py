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
    heads = [f"##### {q}" for q in (wording.Q_TRUST, wording.Q_STAND, wording.Q_NEXT)]
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


@pytest.mark.skip(reason="review cards prototype: pins the old Review layout (status lines, Explored fold, header); reconcile at the final merge")
def test_the_gate_is_header_best_so_far_the_gate_line_and_the_foot(tmp_path):
    """Review M8; since ftue W1 (R-B1 row 3, T0 §5) the gate line is the
    Predictions status line's, and Best so far sits in Progress figures."""
    opt = _patty(rows=3)
    at = _tab(opt.project_name)
    assert not at.exception
    md = _md(at)
    # Re-pinned to the concision round's short line; pred_gate's full
    # sentence opens its tooltip.
    assert md[0] == wording.status_line(wording.STATUS_PREDICTIONS, wording.short_gate(3),
                                        wording.PREDICTIONS_WORDS["not_yet"])
    assert md[1].startswith("Progress: **") and md[2].startswith("Next: ")
    assert md[3] == f"##### {wording.Q_STAND}" and md[4].startswith("Best so far: ")
    captions = [c.value for c in at.tabs[3].caption]
    assert captions[0] == ui_insights.header_text(opt)
    assert wording.insights_gate(3) not in captions
    assert "insights_foot" in [b.key for b in at.tabs[3].button]


def test_nothing_scored_still_offers_the_foot(tmp_path):
    opt = _patty(rows=0)
    _tell(opt, 1, 1, drop=("Cook loss",))
    opt.set_aside_formulation(1)
    at = _tab(opt.project_name)
    assert not at.exception
    assert [c.value for c in at.tabs[3].caption] == [wording.INSIGHTS_NONE]
    assert "insights_foot" in [b.key for b in at.tabs[3].button]


@pytest.mark.skip(reason="review cards prototype: pins the old Review layout (status lines, Explored fold, header); reconcile at the final merge")
def test_a_failed_fit_says_so_once_and_keeps_best_so_far_and_the_foot(
        insights_scored, monkeypatch):
    monkeypatch.setattr(insights, "fit_all", lambda opt: (_ for _ in ()).throw(RuntimeError()))
    at = _open_insights(_app())
    assert not at.exception
    md = " ".join(_md(at))
    assert wording.Q_TRUST not in md and wording.Q_NEXT not in md
    assert wording.Q_STAND in md
    assert [c.value for c in at.tabs[3].caption].count(wording.INSIGHTS_FAILED) == 1
    assert "insights_foot" in [b.key for b in at.tabs[3].button]


def test_the_set_up_button_goes_to_set_up(insights_scored, monkeypatch):
    moved = _moves(monkeypatch)
    monkeypatch.setattr(insights, "next_step",
                        lambda opt, checks, standing=None, gain=None: insights.NextStep(
                            "close", binding=insights.Binding("highest", "Pea protein",
                                                              "Pea protein")))
    at = _open_insights(_app())
    assert not at.exception
    button = next(b for b in at.tabs[3].button if b.key == "insights_change_setup")
    # Review polish: the label at most 5 words; the change it names, in its "?".
    assert button.label == wording.SETUP_BUTTON_SHORT
    assert button.proto.help == "Change Pea protein's Highest on Set up"
    assert button.proto.type == "secondary"
    button.click()
    at.session_state["main_tab"] = wording.TAB_INSIGHTS
    at.run()
    assert moved == [wording.TAB_SETUP]


# ------------------------------------------------------------------ #
#  "Can you trust the predictions?" (review I6, M9)
# ------------------------------------------------------------------ #

def _h(state="verdict", inside=32, total=33, rounds=(4, 5, 6), wide=()):
    return insights.Headline(state, list(rounds), inside, total, "holding", None, list(wide))


@pytest.mark.parametrize("state,text", [
    ("none", wording.TRUST_NONE),
    ("early", wording.trust_early(33)),
    ("wide", wording.TRUST_WIDE),
    ("within", wording.trust_within(32, 33, [4, 5, 6], allowed=3)),
    ("over_wide", wording.trust_over_wide(33, [4, 5, 6])),
    ("caution", wording.trust_misses(3, [4, 5, 6], 32, 33, allowed=3)),
    ("unreliable", wording.trust_misses(3, [4, 5, 6], 32, 33, many=True, allowed=3)),
])
def test_one_verdict_per_state(state, text):
    opt = _patty(rows=6)
    assert ui_insights.verdict_text(opt, _h(), state) == text


def test_some_outputs_wide_adds_the_wide_line():
    opt = _patty(rows=6)
    text = ui_insights.verdict_text(opt, _h(wide=["Cook loss"]), "within")
    assert text == (wording.trust_within(32, 33, [4, 5, 6], allowed=3) + " "
                    + wording.wide_line("Cook loss"))


def test_the_score_gap_says_what_to_do_only_beyond_the_yardstick():
    one = insights.Yardstick(1.0, None, 0, False)
    repeats = insights.Yardstick(1.2, 1.2, 4, True)
    assert ui_insights._beyond(-0.8, one) == ""
    assert ui_insights._beyond(-1.5, one) == "1 point"
    assert ui_insights._beyond(-1.5, repeats) == "Repeats differ"
    assert ui_insights._beyond(1.0, repeats) == ""


@pytest.mark.skip(reason=REVIEW_FOUR)
def test_a_disagreeing_repeat_is_said_once(insights_scored, monkeypatch):
    d = insights.Disagreement(9, 3, -10.6, 8)
    monkeypatch.setattr(insights, "disagreeing_repeat", lambda opt, checks, m=None: d)
    at = _open_insights(_app())
    assert not at.exception
    line = ui_insights.disagreement_text(d)
    assert line == ("Formulation 9, a Repeat of Formulation 3, scored 10.6 points lower. "
                    "If something changed (panel, supplier, equipment), check it before "
                    "relying on the next suggestions.")
    assert _md(at).count(line) == 1


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


def test_the_repeats_line_only_after_a_rise():
    assert ui_insights.stand_text(insights.Improvement("not_lately", None, None, 3)) == (
        "Flat: no higher score in 3 Rounds.")
    # Fix A follow-up (decision 4): the threshold and its make-up, q = 1.
    assert ui_insights.stand_text(insights.Improvement("barely", 0.2, 1.2, 0, 2.79, 1)) == (
        "Within noise: up 0.2 points in the last Round, under the 2.8 points needed: √2 × 1.64 "
        "× the noise threshold (Repeat SD 1.2 points).")


def _standing(higher, close_no=None, close=None):
    p = insights.Prediction(94.4, 89.2, 99.1, 94.4, 2.6)
    best = insights.PredictedBest({}, p, close, 0, 1.0, None)
    return SimpleNamespace(best=best, higher=higher, close_no=close_no,
                           tested_near=close is not None, binding=[], luck=False)


def test_the_four_texts_of_line_two():
    opt = _patty(rows=6)
    one = insights.Yardstick(1.0, None, 0, False)
    repeats = insights.Yardstick(1.2, 1.2, 4, True)
    assert ui_insights.standing_text(opt, _standing(False, 3, 2), repeats) == (
        "Predicted best: 94 (95 % interval 89–99), near Formulation 3. Predicted gain over "
        "Best so far: none beyond the noise threshold (Repeat SD 1.2 points).")
    assert ui_insights.standing_text(opt, _standing(False), one) == (
        "Predicted best: 94 (95 % interval 89–99), in untested amounts. Predicted gain over "
        "Best so far: none beyond the 1-point noise threshold.")
    gain = wording.signed(94.4 - opt.Y_history[opt.best_index()])
    assert ui_insights.standing_text(opt, _standing(True), one) == (
        "Predicted best: 94 (95 % interval 89–99), in untested amounts. Predicted gain over a "
        f"Repeat of Best so far: {gain} points, above the 1-point noise threshold.")
    assert ui_insights.standing_text(opt, _standing(True, 3, 2), one) == (
        f"Predicted best: 94 (95 % interval 89–99), near Formulation 3 "
        f"(measured {opt.Y_history[2]:.2f}). Predicted gain over a Repeat of Best so far: "
        f"{gain} points, above the 1-point noise threshold.")


def test_line_three_names_what_the_predicted_best_sits_at():
    items = [insights.Binding("highest", "Pea", "Pea"), insights.Binding("lowest", "Salt", "Salt"),
             insights.Binding("limit", "Fat", "Fat: at most 20 g")]
    assert wording.stand_limits([ui_insights.binding_text(b) for b in items]) == (
        "Predicted best is at the Highest of Pea, the Lowest of Salt and the limit "
        "Fat: at most 20 g (active bounds).")


# ------------------------------------------------------------------ #
#  "What to do next" (review I2, I3, M17)
# ------------------------------------------------------------------ #

@pytest.mark.parametrize("step,text,button", [
    # 0.8.0 e2e fixes (review I2): the steady bias rides on the misses row.
    (insights.NextStep("misses", gap=-5.0),
     wording.next_misses(bias_text=wording.next_bias("5.0", True)), None),
    (insights.NextStep("misses"), wording.NEXT_MISSES, None),
    (insights.NextStep("early"), wording.NEXT_TOO_EARLY, None),
    (insights.NextStep("untested"), wording.next_untested(open_round=False), None),
    (insights.NextStep("close"), wording.next_close_to_top(), None),
    (insights.NextStep("close", binding=insights.Binding("highest", "Oil", "Oil")),
     wording.next_close_to_top(wording.raise_highest("Oil")), "Change Oil's Highest on Set up"),
    (insights.NextStep("close", binding=insights.Binding("lowest", "Salt", "Salt")),
     wording.next_close_to_top(wording.lower_lowest("Salt")), "Change Salt's Lowest on Set up"),
    (insights.NextStep("close", binding=insights.Binding("limit", "Fat", "Fat: at most 20 g")),
     wording.next_close_to_top(wording.widen_limit("Fat: at most 20 g")),
     wording.CHANGE_LIMITS_ON_SETUP),
    (insights.NextStep("rising"), wording.NEXT_RISING, None),
    (insights.NextStep("see"), wording.NEXT_SEE, None),
])
def test_each_next_step_has_its_sentence(step, text, button):
    opt = _patty(rows=6)
    assert ui_insights.next_step_text(opt, step) == (text, button)


def test_no_sentence_says_what_the_next_round_will_do():
    """Review I3: nothing may promise what ask() does."""
    for text in (wording.next_untested(), wording.next_untested(True), wording.NEXT_RISING,
                 wording.NEXT_SEE, wording.stand_higher("94", "89", "99")):
        assert "will try" not in text and "worth adding" not in text


@pytest.mark.skip(reason=REVIEW_FOUR)
def test_a_process_only_project_names_settings(tmp_path):
    opt = FoodOptimizer("ferment_x")
    opt.add_process_parameter("Mixing time", 1, 10)
    opt.add_objective("Acidity", 1.0, goal="max", min_val=0, max_val=10, unit="pH")
    assert ui_insights._try_one_label(opt) == "Try one process setting"
    text, _ = ui_insights.next_step_text(opt, insights.NextStep("untested"))
    assert "values" in text and "amounts" not in text


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


@pytest.mark.skip(reason=REVIEW_FOUR)
def test_the_amounts_fold_only_when_clearly_higher_and_untested(insights_scored, monkeypatch):
    label = wording.show_its_amounts()
    # The fold's own rule, with the model's gains not withheld: this small
    # fixture is below the A3 gate, which holds the fold back too (polish
    # round; tests/test_106_last_fixes.py).
    monkeypatch.setattr(ui_insights, "claims_withheld_reason", lambda *a, **k: None)
    _fake_best(monkeypatch, None, 5.0)
    at = _open_insights(_app())
    assert not at.exception
    fold = next(e for e in at.tabs[3].expander if e.label == label)
    assert list(fold.dataframe[0].value[wording.FORMULATION_CAP])[0] == wording.BEST_ROW
    for close, gap in ((2, 5.0), (None, 0.5)):
        _fake_best(monkeypatch, close, gap)
        at = _open_insights(_app())
        assert label not in [e.label for e in at.tabs[3].expander]


@pytest.mark.skip(reason=REVIEW_FOUR)
def test_a_failed_amounts_table_is_one_line(insights_scored, monkeypatch):
    monkeypatch.setattr(ui_insights, "claims_withheld_reason", lambda *a, **k: None)
    _fake_best(monkeypatch, None, 5.0)
    monkeypatch.setattr(ui_insights, "best_frame",
                        lambda opt, best, fitted: (_ for _ in ()).throw(RuntimeError()))
    at = _open_insights(_app())
    assert not at.exception
    fold = next(e for e in at.tabs[3].expander if e.label == wording.show_its_amounts())
    assert [c.value for c in fold.caption] == [wording.BEST_FAILED]
