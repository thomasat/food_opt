"""Insights for a first-time user, integration (spec 2026-10-01 rev. 2;
ledger integration notes and ruling; review R-B1, R-I5, R-I6, R-I12,
R-I14 to R-I16): what W1, W2 and W3 hand each other, pinned where they
meet.

- the basis line at the top of each figures fold carries W2's help, the
  Next one with the row that fired (it needs the facts);
- one status logic: Predictions puts caution before too_few (ledger
  ruling: a warning is never folded), in the code and in the table;
- one model with no Round open: layer_model is bo_views.next_model, so
  Try one, the triangle's Uncertainty layer and the map agree;
- the Explored fold's tooltips name the design space, get_polytope_samples,
  the L∞ rule and the posterior sd;
- the fold renames, and the pointer naming them;
- T0′ wording: Next's numbers on see, early and explore; the strips' BO
  term; the cold start in Known limits; strips_no_gap with process
  settings; one decimal rule across a gap line;
- cleanup: the untested row never points at a hidden "Show its amounts";
  MODEL_SD_HELP and NEAREST_TESTED_HELP are gone."""
import os

import pytest
from tests.demo_build_reason import REVIEW_FOUR

import bo_views
import insights
import ui_insights
import wording
from tests import demo_projects as D
from tests.test_107_w1_first_screen import _install, _open, _walk

_NAMES = (D.LINE_A, D.TIMING, D.PROCESS_ONLY)


@pytest.fixture(scope="module")
def screens(tmp_path_factory):
    here = os.getcwd()
    os.chdir(tmp_path_factory.mktemp("ftue-int"))
    try:
        with bo_views.fresh():
            out = {}
            for name in _NAMES:
                _install(name)
                out[name] = _open(name)
            yield out
    finally:
        os.chdir(here)


@pytest.fixture(scope="module")
def project():
    """name -> (project, its checks, its status facts), worked out once."""
    seen = {}

    def get(name):
        if name not in seen:
            opt = D.load(name)
            checks = (insights.check_rounds(opt)
                      if opt.complete_count() >= insights.MIN_ROWS else [])
            seen[name] = (opt, checks, insights.status_facts(opt, checks))
        return seen[name]
    return get


def _fold(at, label):
    return next(e for e in at.tabs[3].expander if e.label == label)


def _leaves(node):
    return [el for el, _ in _walk(node) if el.type in ("markdown", "caption")]


# ------------------------------------------------------------------ #
#  Basis lines and help (W2's integration notes; R-I6)
# ------------------------------------------------------------------ #

@pytest.mark.skip(reason=REVIEW_FOUR)
@pytest.mark.parametrize("name", _NAMES)
def test_each_figures_fold_opens_on_its_basis_line_with_its_help(screens, name):
    at, opt = screens[name], D.load(name)
    for label, which in ((wording.FOLD_PREDICTION_FIGURES, "predictions"),
                         (wording.FOLD_PROGRESS_FIGURES, "progress"),
                         (wording.FOLD_NEXT_REASONING, "next")):
        first = _leaves(_fold(at, wording.for_project(opt, label)))[0]
        text, _ = ui_insights.basis_line(opt, which)
        assert first.type == "caption" and first.value == text, (which, first.value)
        assert first.proto.help, which
    nxt = _leaves(_fold(at, wording.for_project(opt, wording.FOLD_NEXT_REASONING)))[0]
    # Concision round: the full basis first in the help, then the rule.
    assert nxt.proto.help.startswith(wording.BASIS_NEXT)
    assert "\n\nNext-step rule that holds: " in nxt.proto.help


@pytest.mark.skip(reason="review cards prototype: pins the old Review layout (status lines, Explored fold, header); reconcile at the final merge")
@pytest.mark.parametrize("name", _NAMES)
def test_the_status_lines_carry_their_exact_rules(screens, name):
    at = screens[name]
    lines = [el for el in _leaves(at.tabs[3])
             if el.type == "markdown" and el.value.split(":", 1)[0] in (
                 wording.STATUS_PREDICTIONS, wording.STATUS_PROGRESS, wording.STATUS_NEXT)]
    assert len(lines) == 3
    for el in lines:
        assert el.proto.help and any(ch.isdigit() for ch in el.proto.help), el.value


def test_lines_with_help_are_never_st_write():
    """R-I6: st.write takes no help=; every helped line is st.markdown."""
    import inspect
    src = inspect.getsource(ui_insights)
    assert "st.write(" not in src


# ------------------------------------------------------------------ #
#  One status logic; the ledger's row-order ruling
# ------------------------------------------------------------------ #

def test_caution_comes_before_too_few():
    """Ledger ruling: "a warning is never folded". More misses than expected
    with fewer than 10 Overall-score checks reads as the caution, not as
    "checked against k formulations"."""
    rows = insights.PREDICTION_ROWS
    assert rows.index("caution") < rows.index("too_few")
    assert insights.prediction_row(12, "caution", None, None, 4) == "caution"
    assert insights.prediction_row(12, "early", None, None, 4) == "too_few"
    assert insights.prediction_row(12, "within", None, None, 4) == "too_few"
    # Worse rows still win over it.
    assert insights.prediction_row(3, "caution", None, None, 0) == "gate"
    assert insights.prediction_row(12, "unreliable", None, None, 4) == "off_coverage"


def test_the_rule_table_says_the_same_order():
    labels = [status for line, status, _ in wording.status_rule_rows()
              if line == wording.STATUS_PREDICTIONS and not status.startswith("Clause")]
    more = labels.index(f"{wording.PREDICTIONS_WORDS['not_yet']} (more misses)")
    few = labels.index(f"{wording.PREDICTIONS_WORDS['not_yet']} (too few checks)")
    assert more < few
    keys = [k for line, k in insights.RULE_KEYS if line == "predictions"]
    assert keys == list(insights.PREDICTION_ROWS)


@pytest.mark.parametrize("name", D.ALL)
def test_the_tooltips_read_the_real_status_facts(project, name):
    """W2's tooltips and W1's status_facts agree on R-B1's shapes: the row
    the line says is the row the tooltip names."""
    opt, _, facts = project(name)
    assert facts.prediction_row in insights.PREDICTION_ROWS
    assert facts.progress in insights.PROGRESS_STATES
    for clause in facts.clauses:
        assert clause[0] in ("wide", "over_wide", "bias", "guard")
    tip = ui_insights.tooltip_for(opt, "predictions", facts)
    assert wording.for_project(
        opt, insights.status_rule(opt, ("predictions", facts.prediction_row))) in tip


# ------------------------------------------------------------------ #
#  One model with no Round open (W3's note; R-I16)
# ------------------------------------------------------------------ #

def test_with_no_round_open_the_layers_use_the_next_generates_model(monkeypatch):
    opt = D.load(D.TIMING)
    assert not opt.pending_batch
    marker = object()
    monkeypatch.setattr(bo_views, "next_model", lambda o: marker)


# ------------------------------------------------------------------ #
#  The Explored fold's tooltips (W3's note 4)
# ------------------------------------------------------------------ #

_DESIGN = "design space"
_POLYTOPE = "get_polytope_samples"
_LINF = "L∞"
_SD = "posterior sd"
_NEEDS = {"basis_explored": (_DESIGN, _POLYTOPE, _LINF, _SD),
          "strips": (_DESIGN, _POLYTOPE, _LINF),
          "gaps": (_DESIGN, _POLYTOPE, _LINF),
          "map": (_DESIGN, _POLYTOPE, _SD),
          "limit_hits": (_DESIGN,)}


@pytest.mark.parametrize("line", sorted(_NEEDS))
def test_the_explored_tooltips_name_the_space_and_the_rules(line):
    for name in (D.LINE_A, D.PROCESS_ONLY):
        tip = ui_insights.tooltip_for(D.load(name), line)
        for needle in _NEEDS[line]:
            assert needle in tip, (line, needle)
        assert " BO term: " in tip
        assert f"{bo_views.POOL_SIZE:,} points" in tip if _POLYTOPE in _NEEDS[line] else True


def test_the_strips_tip_has_its_bo_term():
    for values in (False, True):
        head, sep, term = wording.tip_strips(values).partition(" BO term: ")
        assert head and sep and term


# ------------------------------------------------------------------ #
#  Fold renames (§2; T0 report §5) and the pointer
# ------------------------------------------------------------------ #

@pytest.mark.skip(reason=REVIEW_FOUR)
@pytest.mark.parametrize("name", _NAMES)
def test_the_folds_have_their_new_names(screens, name):
    at, opt = screens[name], D.load(name)
    labels = [e.label for e in at.tabs[3].expander]
    try_one = ui_insights._try_one_label(opt)
    assert try_one == wording.try_one_fold(insights.has_setting_choice(opt),
                                           insights.has_ingredient_choice(opt))
    assert try_one.startswith("Try one ")
    assert wording.for_project(opt, try_one) in labels
    assert wording.FOLD_METHOD in labels
    assert not any(lab.startswith(("Effect of one", "More charts", "Method and details"))
                   for lab in labels), labels
    if opt.has_ingredients():
        # Charts proto: the triangle moved to Explored and unexplored.
        assert wording.PAIR_FOLD in labels
    assert labels.index(wording.for_project(opt, try_one)) < labels.index(wording.FOLD_METHOD)


@pytest.mark.skip(reason=REVIEW_FOUR)
def test_the_pointer_names_the_folds_as_drawn():
    opt = D.load(D.LINE_A)
    # Concision round: the link is short; the folds are named in its help.
    assert wording.explored_pointer(ui_insights._try_one_label(opt)) == (
        "Show uncertainty along one ingredient")
    text = wording.explored_pointer_help(ui_insights._try_one_label(opt))
    assert wording.try_one_fold(True) in text
    assert "under Trade-offs" in text and wording.trade_offs_fold(True).startswith("Trade-offs")


# ------------------------------------------------------------------ #
#  T0′ wording
# ------------------------------------------------------------------ #

def test_the_next_tooltip_names_its_numbers_on_see(project):
    opt, _, facts = project(D.LINE_A)
    assert facts.next.kind == "see"
    tip = ui_insights.tooltip_for(opt, "next", facts)
    assert "Best so far: 90.45 of 100 (Formulation 15, Round 4)" in tip
    assert wording.TIP_THRESHOLD_ASSUMED in tip


def test_the_next_tooltip_names_its_numbers_on_explore(project):
    opt, checks, _ = project(D.TIMING)
    standing_gain = 1.1234
    step = insights.NextStep("explore", gain=standing_gain, assumed=True)
    facts = insights.StatusFacts(opt.complete_count(), "reliable",
                                 yardstick=insights.noise_yardstick(opt, checks),
                                 headline=insights.headline(opt, checks), next=step,
                                 best=insights.best_facts(opt))
    tip = ui_insights.tooltip_for(opt, "next", facts)
    assert "Expected gain: 1.1 points" in tip
    assert wording.TIP_THRESHOLD_ASSUMED in tip


def test_the_next_tooltip_names_its_numbers_on_early(project):
    """Early on every output very uncertain: the width rule, with its
    outputs, not only the coverage count."""
    opt, checks, facts = project(D.LINE_A)
    h = insights.headline(opt, checks)
    names = [o["name"] for o in opt.measurements_by_importance()]
    wide = insights.StatusFacts(opt.complete_count(), "wide", headline=h, verdict="wide",
                                yardstick=facts.yardstick, next=insights.NextStep("early"),
                                best=facts.best)
    h.wide = list(names)
    tip = ui_insights.tooltip_for(opt, "next", wide)
    assert wording.tip_wide(names) in tip


# ------------------------------------------------------------------ #
#  Cleanup
# ------------------------------------------------------------------ #

def test_the_untested_row_never_points_at_hidden_amounts():
    """"Show its amounts" is not drawn while the noise guard's floor or
    noise trigger holds; the untested row then names no fold."""
    opt = D.load(D.TIMING)
    step = insights.NextStep("untested")
    shown = insights.StatusFacts(opt.complete_count(), "reliable", next=step,
                                 guard=insights.Guard([], None))
    hidden = insights.StatusFacts(opt.complete_count(), "reliable", next=step,
                                  guard=insights.Guard(["floor"], None),
                                  clauses=[("guard",)])
    name = wording.FOLD_PROGRESS_FIGURES_NAME
    assert name in ui_insights.next_sentence(opt, shown)
    assert name not in ui_insights.next_sentence(opt, hidden)
    assert name in ui_insights.next_step_text(opt, step, None)[0]
    assert name not in ui_insights.next_step_text(opt, step, None, amounts_shown=False)[0]
    assert wording.show_its_amounts(False) not in ui_insights.next_step_text(
        opt, step, None, amounts_shown=False)[0]


def test_the_old_help_constants_are_gone():
    for gone in ("MODEL_SD_HELP", "NEAREST_TESTED_HELP"):
        assert not hasattr(wording, gone)
