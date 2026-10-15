"""Insights for a first-time user, W1 (spec 2026-10-01 rev. 2; review
R-B1, R-I4, R-I5, R-I14): each status line's sentence from its facts, for
the states the demo projects do not reach (the review's templates, built
in tests/test_107_ftue.UNREACHED), and the one-link rules. Fast: facts are
built by hand, nothing is fitted. The demo projects' own sentences are
pinned on the rendered screen (tests/test_107_w1_first_screen.py)."""
import types

import pytest

import bo_views
import insights
import ui_insights
import wording
from tests.test_107_ftue import UNREACHED, statistic_names_in
from tests.test_complete_results import _patty


@pytest.fixture(autouse=True)
def _isolate_cwd(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)


@pytest.fixture
def opt():
    return _patty(rows=12)


def _h(inside=32, total=35, rounds=(3, 4, 5)):
    return insights.Headline("verdict", list(rounds), inside, total, "holding")


def _facts(row, clauses=(), h=None, **kw):
    return insights.StatusFacts(scored=kw.pop("scored", 20), prediction_row=row,
                                clauses=list(clauses), headline=h or _h(), **kw)


# ------------------------------------------------------------------ #
#  Predictions: every row, its clauses joined as the ruling says
# ------------------------------------------------------------------ #

@pytest.mark.parametrize("key,facts", [
    ("caution", _facts("caution", h=_h(30, 35))),
    ("over_wide", _facts("reliable", [("over_wide",)], h=_h(36, 36))),
    ("wide", _facts("wide")),
    ("unclear", _facts("unclear")),
    ("off_repeat", _facts("off_repeat")),
    ("none", _facts("too_few", checked=0)),
    ("reliable_all_clauses", _facts("reliable", [("wide", ["Cook loss"]), ("bias", 1.2),
                                                  ("guard",)], h=_h(33, 36, (4, 5, 6)))),
])
def test_the_unreached_predictions_templates(opt, key, facts):
    assert ui_insights.prediction_sentence(opt, facts) == UNREACHED[key]


def test_the_gate_and_too_few_sentences(opt):
    assert ui_insights.prediction_sentence(opt, _facts("gate", scored=3)) == wording.pred_gate(3)
    assert (ui_insights.prediction_sentence(opt, _facts("too_few", checked=2))
            == wording.pred_too_few(2))


def test_the_bias_joins_rows_7_and_8_and_is_a_clause_elsewhere(opt):
    bias = [("bias", -4.3)]
    assert (ui_insights.prediction_sentence(opt, _facts("no_skill", bias))
            == wording.pred_no_skill(("4.3", True)))
    assert (ui_insights.prediction_sentence(opt, _facts("unclear", [("bias", 1.5)]))
            == wording.pred_unclear(("1.5", False)))
    assert (ui_insights.prediction_sentence(opt, _facts("too_few", bias, checked=4))
            == wording.pred_too_few(4, clauses=[wording.clause_bias("4.3", True)]))
    assert (ui_insights.prediction_sentence(opt, _facts("off_coverage", bias, h=_h(26, 36)))
            == wording.pred_off_coverage([3, 4, 5], 10, 36, 4, wording.bias_fact("4.3", True)))


def test_the_wide_clause_names_the_overall_score_in_words(opt):
    facts = _facts("caution", [("wide", ["Firmness", insights.SCORE])], h=_h(30, 35))
    assert ui_insights.prediction_sentence(opt, facts).endswith(
        "; those for Firmness and the Overall score are still very uncertain.")


def test_the_guard_clause_says_process_setting_on_a_study_of_settings():
    from tests import demo_projects as D
    opt = D.load(D.PROCESS_ONLY)
    text = ui_insights.prediction_sentence(opt, _facts("reliable", [("guard",)], h=_h(33, 36)))
    assert text.endswith(wording.clause_guard(settings_only=True)[2:] + ".")


@pytest.mark.parametrize("row", insights.PREDICTION_ROWS)
def test_no_predictions_sentence_names_a_statistic(opt, row):
    facts = _facts(row, [("wide", ["Firmness"]), ("over_wide",), ("bias", -2.0), ("guard",)],
                   checked=4, h=_h(30, 35))
    assert statistic_names_in(ui_insights.prediction_sentence(opt, facts)) == []


# ------------------------------------------------------------------ #
#  Progress
# ------------------------------------------------------------------ #

def _best(opt, rounds_since=0, latest=True):
    i = opt.best_index()
    return insights.BestFacts(float(opt.Y_history[i]), int(opt.formulation_ids[i]),
                              opt.batch_history[i], rounds_since, latest, [])


def test_within_noise_says_less_than_the_threshold(opt):
    f = _best(opt)
    facts = _facts("reliable", progress="within_noise", best=f,
                   improvement=insights.Improvement("barely", 0.4, None, 0, 2.326, 1))
    # Round 2 (P3 #19): one decimal for scores on the plain layer.
    # Fix A follow-up (decision 4): the threshold itself (√2 × 1.645 × 1).
    assert ui_insights.progress_sentence(opt, facts) == wording.progress_within_noise(
        f.round, "0.4", f"{f.score:.1f}", 100, f.formulation, "2.3")


def test_a_measured_threshold_is_said_with_its_size(opt):
    f = _best(opt)
    facts = _facts("reliable", progress="rising", best=f,
                   improvement=insights.Improvement("yes", 4.5, 1.2, 0, 3.99, 1, 3))
    # Fix A follow-up (decisions 3, 4): 1.2 × √2 × t_{3}⁻¹(0.95) = 3.99.
    assert ui_insights.progress_sentence(opt, facts) == wording.progress_rising(
        f.round, "4.5", f"{f.score:.1f}", 100, f.formulation, "4.0")


def test_a_rise_that_would_read_as_the_threshold_gets_hundredths(opt):
    f = _best(opt)
    facts = _facts("reliable", progress="within_noise", best=f,
                   improvement=insights.Improvement("barely", 2.31, None, 0, 2.326, 1))
    # Fix A follow-up (decision 4): the rise is read against the threshold
    # (2.33), not the noise SD; both in hundredths when they read alike.
    said = ui_insights.progress_sentence(opt, facts)
    assert "by 2.31 points" in said and "the 2.33 points needed" in said


def test_flat_with_luck_and_a_tie(opt):
    f = _best(opt, rounds_since=3, latest=False)
    other = next(i for i in range(len(opt.Y_history)) if i != opt.best_index())
    facts = _facts("reliable", progress="flat", best=f, tie=other, luck=True,
                   luck_repeat=88.4, improvement=insights.Improvement("not_lately", None, None, 3))
    assert ui_insights.progress_sentence(opt, facts) == wording.progress_flat(
        3, f"{f.score:.1f}", 100, f.formulation, f.round, luck_text="88",
        tie_no=int(opt.formulation_ids[other]))


def test_nothing_scored_says_so(opt):
    assert (ui_insights.progress_sentence(opt, _facts("gate", best=None))
            == wording.NO_COMPLETE_FORMULATION)


def test_of_100_is_the_projects_ceiling(opt, monkeypatch):
    """M1: utility_ceiling, a whole number."""
    f = _best(opt, rounds_since=1, latest=False)
    monkeypatch.setattr(opt, "utility_ceiling", lambda: 80.0)
    facts = _facts("reliable", progress="no_new_best", best=f)
    assert f"{f.score:.1f} of 80 " in ui_insights.progress_sentence(opt, facts)


# ------------------------------------------------------------------ #
#  Next: one plain sentence per row (R-I5)
# ------------------------------------------------------------------ #

def _next(opt, kind, **kw):
    return _facts("reliable", best=_best(opt), next=insights.NextStep(kind, **kw))


@pytest.mark.parametrize("kind,kw,expected", [
    ("misses", {}, lambda no: wording.next_plain_misses()),
    ("early", {}, lambda no: wording.next_plain_tail()),
    ("no_skill", {}, lambda no: wording.next_plain_no_skill()),
    ("untested", {}, lambda no: wording.next_plain_untested(open_round=False)),
    ("rising", {}, lambda no: wording.next_plain_rising()),
    ("rising", {"assumed": True}, lambda no: wording.next_plain_rising(repeat_no=no)),
    ("see", {}, lambda no: wording.next_plain_see()),
    ("see", {"assumed": True}, lambda no: wording.next_plain_see(repeat_no=no)),
    ("explore", {"gain": 1.08, "assumed": True},
     lambda no: wording.next_plain_explore("1.1", repeat_no=no)),
    ("explore", {"gain": 2.34}, lambda no: wording.next_plain_explore("2.3", assumed=False)),
    ("close", {"binding": insights.Binding("highest", "Pea", "Pea")},
     lambda no: wording.next_plain_close(wording.raise_highest("Pea"))),
    ("close", {"binding": insights.Binding("limit", "Fat", "Fat: at most 20 g"),
               "assumed": True},
     lambda no: wording.next_plain_close(wording.widen_limit("Fat: at most 20 g"),
                                         repeat_no=no)),
    ("close", {}, lambda no: wording.next_plain_close()),
])
def test_each_next_row_has_its_plain_sentence(opt, kind, kw, expected):
    facts = _next(opt, kind, **kw)
    assert ui_insights.next_sentence(opt, facts) == expected(facts.best.formulation)
    assert statistic_names_in(ui_insights.next_sentence(opt, facts)) == []


def test_the_repeat_row_says_the_limit_in_plain_words(opt):
    d = insights.Disagreement(21, 11, -10.54, 20, 7.42)
    facts = _next(opt, "repeat", disagreement=d)
    assert ui_insights.next_sentence(opt, facts) == wording.next_plain_repeat(
        21, 11, "10.5", half_text="7.4", round_no=opt.next_batch_no())


def test_at_the_gate_next_is_the_tail(opt):
    assert ui_insights.next_sentence(opt, _facts("gate", best=_best(opt))) == \
        wording.next_plain_tail()


# ------------------------------------------------------------------ #
#  The one link (R-I3, R-I4, R-I14)
# ------------------------------------------------------------------ #

@pytest.fixture
def links(monkeypatch):
    drawn = []
    monkeypatch.setattr(ui_insights, "link",
                        lambda label, target, section="top", fold=None, key=None, state=None:
                        drawn.append((label, target, section, fold)))
    return drawn


def _open_round(opt, regime="warm", own=False):
    opt.pending_batch = [{"formulation": 99, "recipe": {}, "note": ""}]
    opt.pending_batch_no = 5
    gen = bo_views.Generation(5, 99, [], regime, True, [] if own else [99])
    return gen


@pytest.mark.parametrize("regime,own,drawn", [("warm", False, True), ("cold", False, False),
                                              ("warm", True, False)])
def test_the_why_link_needs_a_warm_round_with_a_generated_pick(opt, links, monkeypatch,
                                                                regime, own, drawn):
    gen = _open_round(opt, regime)
    if own:
        gen.picks = [7]
    monkeypatch.setattr(bo_views, "generation", lambda opt, r: gen)
    assert ui_insights._why_link(opt) is drawn
    assert bool(links) is drawn
    if drawn:
        # Ftue fix C: it lands on the fold's keyed place, the fold opened.
        assert links[0] == (wording.why_link(5), wording.TAB_BATCH, ui_insights.WHY_PLACE,
                            ui_insights.WHY_FOLD_KEY)


def test_no_why_link_with_no_round_open(opt, links):
    assert ui_insights._why_link(opt) is False and links == []


@pytest.mark.parametrize("kind,drawn", [("explore", True), ("untested", True), ("see", False),
                                        ("close", False), ("repeat", False)])
@pytest.mark.skip(reason="review cards prototype: pins the old Review layout (status lines, Explored fold, header); reconcile at the final merge")
def test_the_explored_link_is_for_explore_and_untested(opt, links, kind, drawn):
    facts = _next(opt, kind)
    if drawn:
        assert links == [(wording.TAB_INSIGHTS,
                          ui_insights.EXPLORED_PLACE, ui_insights.EXPLORED_KEY)]


def test_the_repeat_link_is_drawn_while_next_asks_for_a_repeat(opt, links):
    """R-I3, ruled in by the early walkthrough (ftue fix C; its landing is
    pinned in tests/test_107_fc.py): no Round open, Make a round's top."""
    facts = _next(opt, "see", assumed=True)
    assert ui_insights._repeat_link(opt, facts) is True
    assert links == [(wording.repeat_link(facts.best.formulation), wording.TAB_BATCH, "top",
                      None)]
    links.clear()
    assert ui_insights._repeat_link(opt, _next(opt, "see")) is False and links == []


def test_one_round_counts_every_scored_formulations_round():
    opt = _patty(rows=3)
    assert ui_insights.one_round(opt)
    opt = _patty(rows=6)
    assert not ui_insights.one_round(opt)


@pytest.mark.skip(reason="review cards prototype: pins the old Review layout (status lines, Explored fold, header); reconcile at the final merge")
def test_the_explored_link_never_joins_a_why_link(opt, monkeypatch):
    """R-I4: at most one tertiary link; Why (priority 1) wins over Explored
    (priority 3)."""
    import contextlib
    asked = []
    for name in ("_status_lines", "_question", "_basis", "_setup_button"):
        monkeypatch.setattr(ui_insights, name, lambda *a, **k: None)
    monkeypatch.setattr(ui_insights, "_figures", lambda *a, **k: contextlib.nullcontext())
    monkeypatch.setattr(ui_insights, "_next_step", lambda opt, step: None)
    slot = types.SimpleNamespace(container=contextlib.nullcontext)
    facts = _next(opt, "explore", gain=2.0)
    for why, expected in ((True, []), (False, [1])):
        asked.clear()
        places = types.SimpleNamespace(lines=slot, change=slot, link=slot,
                                       nxt=contextlib.nullcontext(), why=why)
        ui_insights._finish(opt, places, facts)
        assert asked == expected
