"""Insights for a first-time user, W1 (spec 2026-10-01 rev. 2 §1, §2, §4;
review R-B1, R-I4, R-I5, R-I7, R-I8, R-I9, M2, M7): the first screen of
each demo project and of the process-only study, as AppTest sees it.

The first screen is everything before the first detail heading: the
header line, the three status lines, Next's primary button (with its Set
up secondary when there is one) and at most one tertiary link. Pinned
here: its order, its exact lines (the review's R-B1 table, built from
wording in tests/test_107_ftue.py), the word budget, no statistic name in
a status sentence, no withheld number, and today's exact lines moved into
their figures folds. The 1280 x 800 position of the button is pinned in
tests/browser/test_107_first_screen.py.

Each project is opened once (the checks take seconds; the Timing check
about 25 s)."""
import json
import os
import re

import pytest
from streamlit.testing.v1 import AppTest

import bo_views
import insights
import ui_batch
import ui_insights
import wording
from food_bo import FoodOptimizer
from tests import demo_projects as D
from tests.test_107_ftue import FIRST_SCREENS, statistic_names_in
from tests.test_app_ui import APP_PATH
from ui_helpers import SCROLL_KEY

_SHORT = {D.SNACK: "Snack bar", D.LINE_A: "Line A", D.LINE_B: "Line B", D.LINE_C: "Line C",
          D.UPGRADE: "Upgrade check", D.TIMING: "Timing check"}

# The process-only study (T0's fixture): Not yet (skill −0.03, n = 16),
# the Overall score's intervals wide, Within noise and Next's see row with
# the Repeat sentence, in Trial forms. Ftue fix A: the rise of 1.9 from a
# Round of 4 is under 1 × √2 × z_4 = 3.16 (cold read I6; it read Rising),
# and the Round-grouped folds widen the Overall score's intervals (I9).
_PROCESS_ONLY = (
    "not_yet", wording.pred_unclear(clauses=[wording.clause_wide(["the Overall score"])]),
    "within_noise", wording.progress_within_noise(5, "1.9", "87.92", 100, 21),
    wording.next_plain_see(6, 4, repeat_no=21, values=True))

# R-I8: at most 130 words above the fold (header + 3 lines + button +
# link), pinned per project as they stand.
_BUDGET = 130
# Ftue fix A: Line B 113 -> 105, Line C 119 -> 113, process-only 109 -> 114.
_WORDS = {D.SNACK: 65, D.LINE_A: 118, D.LINE_B: 105, D.LINE_C: 113, D.UPGRADE: 79,
          D.TIMING: 126, D.PROCESS_ONLY: 114}


def _install(name):
    state = json.loads((D.HERE / f"{name}.json").read_text())
    opt = FoodOptimizer(state["project_name"])
    opt.import_json(state)
    opt.save()


def _open(name, state=None):
    at = AppTest.from_file(APP_PATH, default_timeout=600)
    at.session_state["_loaded_project"] = name
    at.run()
    at.session_state["main_tab"] = wording.TAB_INSIGHTS
    for k, v in (state or {}).items():
        at.session_state[k] = v
    at.run()
    assert not at.exception, [e.message for e in at.exception]
    return at


@pytest.fixture(scope="module")
def screens(tmp_path_factory):
    here = os.getcwd()
    os.chdir(tmp_path_factory.mktemp("ftue-w1"))
    try:
        with bo_views.fresh():
            out = {}
            for name in D.ALL:
                _install(name)
                out[name] = _open(name)
            yield out
    finally:
        os.chdir(here)


def _walk(node, folds=()):
    """Every leaf under `node`, in screen order, with the fold labels it
    sits in."""
    for child in node.children.values():
        if child.type == "expander":
            yield child, folds
            yield from _walk(child, folds + (child.label,))
        elif getattr(child, "children", None):
            yield from _walk(child, folds)
        else:
            yield child, folds


def _text(el):
    if el.type in ("markdown", "caption"):
        return el.value
    if el.type == "button":
        return el.label
    if el.type == "expander":
        return el.label
    return ""


def first_screen(at):
    """[(type, text)] before the first detail heading or fold: header,
    status lines, buttons (type "button:<kind>")."""
    out = []
    for el, _ in _walk(at.tabs[3]):
        text = _text(el)
        if el.type == "expander" or text.startswith("##### "):
            break
        if el.type == "button":
            out.append((f"button:{el.proto.type}", text))
        elif text:
            out.append((el.type, text))
    return out


def _status(at):
    return [t for k, t in first_screen(at) if k == "markdown"]


def _expected(name):
    word_p, pred, word_g, prog, nxt = (_PROCESS_ONLY if name == D.PROCESS_ONLY
                                       else FIRST_SCREENS[_SHORT[name]])
    opt = D.load(name)
    return [wording.for_project(opt, wording.status_line(
                wording.STATUS_PREDICTIONS, pred, wording.PREDICTIONS_WORDS[word_p])),
            wording.for_project(opt, wording.status_line(
                wording.STATUS_PROGRESS, prog, wording.PROGRESS_WORDS[word_g])),
            wording.for_project(opt, wording.status_line(wording.STATUS_NEXT, nxt))]


def _sentence(line):
    """A status line's sentence: its label and bold word taken off."""
    return re.sub(r"^\w+: (\*\*[^*]+\*\* )?", "", line)


def _words(text):
    return len(text.replace("**", "").split())


# ------------------------------------------------------------------ #
#  The lines, in order (§1, R-B1, M7)
# ------------------------------------------------------------------ #

@pytest.mark.parametrize("name", D.ALL)
def test_the_three_status_lines_are_the_rulings_sentences(screens, name):
    assert _status(screens[name]) == _expected(name)


@pytest.mark.parametrize("name", D.ALL)
def test_the_first_screen_reads_header_lines_button_link(screens, name):
    """§1's order: the header, the three status lines, Next's primary
    button, then at most one tertiary link (R-I4); nothing else."""
    kinds = [k for k, _ in first_screen(screens[name])]
    assert kinds[0] == "caption"
    assert kinds[1:4] == ["markdown"] * 3
    assert kinds[4] == "button:primary"
    rest = kinds[5:]
    assert rest in ([], ["button:tertiary"], ["button:secondary"],
                    ["button:secondary", "button:tertiary"]), rest


@pytest.mark.parametrize("name", D.ALL)
def test_one_primary_button_on_the_tab(screens, name):
    buttons = [el for el, _ in _walk(screens[name].tabs[3]) if el.type == "button"]
    assert [b.label for b in buttons if b.proto.type == "primary"] == [
        first_screen(screens[name])[4][1]]


def test_next_names_its_status_word_only_for_predictions_and_progress(screens):
    """M7: "Next" is a label with no status word."""
    for name in D.ALL:
        p, g, n = _status(screens[name])
        assert p.startswith("Predictions: **") and g.startswith("Progress: **")
        assert n.startswith("Next: ") and "**" not in n


# ------------------------------------------------------------------ #
#  The budget (R-I8) and no statistic names (§1, R-I17)
# ------------------------------------------------------------------ #

@pytest.mark.parametrize("name", D.ALL)
def test_the_word_budget(screens, name):
    words = sum(_words(t) for _, t in first_screen(screens[name]))
    assert words <= _BUDGET
    assert words == _WORDS[name]


@pytest.mark.parametrize("name", D.ALL)
def test_no_statistic_name_in_a_status_sentence(screens, name):
    for line in _status(screens[name]):
        assert statistic_names_in(_sentence(line)) == [], line


@pytest.mark.parametrize("name", D.ALL)
def test_each_status_sentence_is_at_most_two_body_lines(screens, name):
    """R-I8: about 220 characters at the 980 px main block; Next may run to
    three (the review's own Line B and Timing check sentences do)."""
    p, g, n = (_sentence(line) for line in _status(screens[name]))
    assert len(p) <= 220 and len(g) <= 220 and len(n) <= 330


# ------------------------------------------------------------------ #
#  Withheld numbers (§1, §4, R-I7, R-I9)
# ------------------------------------------------------------------ #

@pytest.mark.parametrize("name", D.ALL)
def test_no_withheld_number_on_the_first_screen(screens, name):
    """The predicted best never stands at level 0; the luck clause's
    predicted Repeat only under Reliable; the expected gain only on the
    explore row, which the withhold rule already gates."""
    text = " ".join(t for _, t in first_screen(screens[name]))
    assert "Predicted best" not in text and "predicted best" not in text
    p = _status(screens[name])[0]
    if "**Reliable.**" not in p:
        assert "predicted near" not in text


def test_the_upgrade_files_predicted_best_is_only_in_its_fold(screens):
    """R-I7: the withhold rule binds level 0; the fold keeps the exact line,
    which already says the gain is withheld."""
    at = screens[D.UPGRADE]
    inside = [(t, f) for el, f in _walk(at.tabs[3]) if (t := _text(el)).startswith(
        "Predicted best: 100")]
    assert inside and all(f[:1] == (wording.FOLD_PROGRESS_FIGURES,) for _, f in inside)


def test_luck_at_level_0_on_the_timing_check_only(screens):
    with_luck = [n for n in D.ALL if "probably helped by luck" in _status(screens[n])[1]]
    assert with_luck == [D.TIMING]


# ------------------------------------------------------------------ #
#  The one link (R-I3, R-I4, R-I14)
# ------------------------------------------------------------------ #

def _links(at):
    return [t for k, t in first_screen(at) if k == "button:tertiary"]


@pytest.mark.parametrize("name,link", [
    (D.SNACK, []), (D.UPGRADE, []),
    (D.LINE_A, [wording.why_link(6)]), (D.LINE_B, [wording.why_link(6)]),
    (D.LINE_C, [wording.why_link(6)]), (D.PROCESS_ONLY, [wording.why_link(6)]),
    # No Round open, the explore row: priority 3.
    (D.TIMING, [wording.explored_link()]),
])
def test_the_one_link(screens, name, link):
    assert _links(screens[name]) == link


def test_no_repeat_link_is_drawn(screens):
    """R-I3's default (binding e2e B2): the Repeat is a sentence."""
    for name in D.ALL:
        assert not any(t.startswith("Make a Repeat") for _, t in first_screen(screens[name]))


def test_the_why_link_lands_on_make_a_rounds_why_fold():
    assert ui_insights.WHY_FOLD_KEY == ui_batch.WHY_KEY


# ------------------------------------------------------------------ #
#  The detail sections (§2, R-I8): noun headings, one figures fold each,
#  today's exact lines inside
# ------------------------------------------------------------------ #

def _folds_of(at, text):
    return [f for el, f in _walk(at.tabs[3]) if _text(el) == text]


def test_the_noun_headings_stay_after_the_first_screen(screens):
    at = screens[D.LINE_A]
    heads = [t for el, _ in _walk(at.tabs[3]) if (t := _text(el)).startswith("##### ")]
    assert heads[:3] == [f"##### {q}" for q in (wording.Q_TRUST, wording.Q_STAND,
                                                wording.Q_NEXT)]


def test_todays_exact_lines_sit_in_their_figures_folds(screens):
    """The line → fold map (spec rev. 2 T0 addendum), sampled on Line A."""
    at = screens[D.LINE_A]
    opt = D.load(D.LINE_A)
    checks = insights.check_rounds(opt)
    h = insights.headline(opt, checks)
    verdict = wording.trust_within(h.inside, h.total, h.rounds,
                                   allowed=insights.allowed_misses(h.total))
    pred = (wording.FOLD_PREDICTION_FIGURES,)
    prog = (wording.FOLD_PROGRESS_FIGURES,)
    assert _folds_of(at, verdict) == [pred]
    skill = ("Overall score, Rounds 2–5: skill 0.36 (n = 15; 0 = as good as the average, "
             "1 = perfect), better than the average of earlier results.")
    assert _folds_of(at, skill) == [pred]
    assert _folds_of(at, wording.TRUST_BEFOREHAND) == [pred]
    assert _folds_of(at, wording.BASIS_PREDICTIONS) == [pred]
    line1 = wording.stand_best("90.45", 15, 4) + " " + wording.stand_since(1)
    assert _folds_of(at, line1) == [prog]
    assert _folds_of(at, wording.BASIS_PROGRESS) == [prog]
    tie = wording.tie_runner_up(18, "90.12", wording.beat_by())
    assert _folds_of(at, tie) == [prog]
    reasoning = ui_insights.next_step_text(
        opt, insights.next_step(opt, checks, ui_insights.standing_for(
            opt, insights.predicted_best(opt, insights.fit_all(opt), checks), checks,
            insights.noise_yardstick(opt, checks))), h.rounds)[0]
    assert _folds_of(at, reasoning) == [(wording.FOLD_NEXT_REASONING,)]


def test_the_chart_folds_sit_inside_their_figures_folds(screens):
    at = screens[D.LINE_A]
    folds = {el.label: f for el, f in _walk(at.tabs[3]) if el.type == "expander"}
    assert folds[wording.FOLD_MEASURED_CHART] == (wording.FOLD_PREDICTION_FIGURES,)
    assert folds[wording.FOLD_PROGRESS_CHART] == (wording.FOLD_PROGRESS_FIGURES,)
    for label in (wording.FOLD_PREDICTION_FIGURES, wording.FOLD_PROGRESS_FIGURES,
                  wording.FOLD_NEXT_REASONING):
        assert folds[label] == ()


def test_the_disagreeing_repeat_is_said_once_at_each_level(screens):
    """Line B: Next carries the Repeat (level 0, and verbatim in Next-step
    reasoning), so Prediction figures does not say it again."""
    at = screens[D.LINE_B]
    said = [f for el, f in _walk(at.tabs[3])
            if "Formulation 21, a Repeat of Formulation 11, scored" in _text(el)]
    assert said == [(), (wording.FOLD_NEXT_REASONING,)]


def test_one_round_does_not_say_it_came_from_the_latest_round(screens):
    """§4: the Snack bar has one Round."""
    texts = [_text(el) for el, _ in _walk(screens[D.SNACK].tabs[3])]
    assert not any(wording.STAND_LATEST in t for t in texts)
    # Its tie is named at level 0 (equal scores) and in Progress figures.
    assert any(t == wording.tie_runner_up(3, "96.00", wording.beat_by()) for t in texts)


def test_the_gate_line_is_gone_from_the_gate(screens):
    """T0 §5: pred_gate replaces insights_gate at level 0."""
    texts = [_text(el) for el, _ in _walk(screens[D.SNACK].tabs[3])]
    assert wording.insights_gate(3) not in texts


# ------------------------------------------------------------------ #
#  Filling in (R-I8): captions hold the places, the button under them
# ------------------------------------------------------------------ #

def test_while_the_workers_fit_the_places_hold_captions_above_the_button(screens, monkeypatch):
    """The workers still fitting everything: the three places show their
    captions, the button is under them, and the progress line under the
    button, so nothing above the button grows by more than a line each."""
    monkeypatch.setattr(ui_insights, "waiting", lambda opt, stage: True)
    at = _open(D.LINE_A)
    screen = first_screen(at)
    assert [t for _, t in screen[1:4]] == [wording.CHECKING_PREDICTIONS,
                                           wording.CHECKING_PROGRESS,
                                           wording.PREPARING_NEXT_STEP]
    assert screen[4][0] == "button:primary"
    # The progress line is drawn after the button and its link (Line A's
    # Why link needs no model, so it is there while the workers fit).
    order = [(el.type, _text(el)) for el, _ in _walk(at.tabs[3])]
    progress = next(i for i, (k, _) in enumerate(order) if k == "progress")
    assert progress > order.index(("button", wording.why_link(6)))


def test_a_pending_line_keeps_its_caption():
    """status_texts leaves out the lines facts.pending names; _status_line
    then draws their captions."""
    opt = D.load(D.SNACK)
    facts = insights.status_facts(opt, ())
    assert set(ui_insights.status_texts(opt, facts)) == {"predictions", "progress", "next"}
    facts.pending = ("progress", "next")
    assert set(ui_insights.status_texts(opt, facts)) == {"predictions"}


# ------------------------------------------------------------------ #
#  Landing (M2): a link to a section opens the fold its lines sit in
# ------------------------------------------------------------------ #

def test_e1_lands_on_prediction_check_with_both_folds_open(screens):
    at = _open(D.LINE_A, {SCROLL_KEY: ui_insights.TRUST_SECTION, ui_insights.EACH_KEY: True})
    assert at.session_state[ui_insights.PREDICTION_FIGURES_KEY] is True
    assert at.session_state[ui_insights.EACH_KEY] is True


def test_a_landing_on_progress_opens_progress_figures(screens):
    at = _open(D.LINE_A, {SCROLL_KEY: ui_insights.STAND_SECTION})
    assert at.session_state[ui_insights.PROGRESS_FIGURES_KEY] is True


def test_results_e1_link_lands_on_the_section():
    import inspect
    import ui_results
    src = inspect.getsource(ui_results)
    assert "section=ui_insights.TRUST_SECTION, fold=ui_insights.EACH_KEY" in src


# ------------------------------------------------------------------ #
#  A failed fit: no model lines, the model-free Progress line kept
# ------------------------------------------------------------------ #

def test_a_failed_fit_keeps_the_progress_line_alone(screens, monkeypatch):
    def broken(opt):
        raise RuntimeError("no fit")
    monkeypatch.setattr(insights, "fit_all", broken)
    at = _open(D.LINE_B)
    screen = first_screen(at)
    texts = [t for _, t in screen]
    assert wording.INSIGHTS_FAILED in texts
    assert not any(t.startswith("Predictions:") or t.startswith("Next:") for t in texts)
    assert [t for t in texts if t.startswith("Progress: **")] == [_expected(D.LINE_B)[1]]
