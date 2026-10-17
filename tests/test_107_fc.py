"""Insights for a first-time user, fix round C (ledger, "Early walkthrough"
rulings; 05-walkthrough-retest-early.md §3, §5 items 2, 3, 5, 9;
06-statistician-cold-read.md I1):

1. "Why the model suggested Round N" lands on Make a round's Why fold,
   opened and scrolled into view, and the fold's first line is the role key.
2. The Why fold's labels are plain, the BO term in each tooltip: Expected
   gain over your best, the "·" picks named, "Figures recalculated from
   today's results.", Closest formulation already made, and a note that
   Compared with is always Best so far.
3. "Make a Repeat of Formulation N" under Next while Next asks for a
   Repeat: Make a round's "Add a formulation to this round" opens with
   Start from Best so far applied (with a Round open now, or after the
   next Generate); nothing is saved.
4. "See the amounts you have not tried" lands on the Explored fold, opened.

Suggestions are untouched: only screens, links and landings change."""
import re
import types

import pytest
from streamlit.testing.v1 import AppTest

import bo_views
import insights
import ui_batch
import ui_helpers
import ui_insights
import wording
from tests.test_app_ui import APP_PATH
from tests.test_complete_results import _patty
from tests.test_food_bo import _BANNED
from tests.test_insights import _burger


@pytest.fixture(autouse=True)
def _isolate(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    with bo_views.fresh():
        yield


class _Settings:
    """A study of process settings alone, for wording.for_project."""
    def has_ingredients(self):
        return False

    def _process_settings(self):
        return [{"name": "Time"}]


# ------------------------------------------------------------------ #
#  The new strings
# ------------------------------------------------------------------ #

def test_the_role_key_is_the_rulings_line():
    # Concision round (2026-10-01).
    # Review polish (2026-10-02): the screen's key at most 8 words; the
    # rule in its "?" (role_help). WHY_ROLE_KEY stays for the wording round.
    assert wording.WHY_ROLE_KEY_SHORT == "Close to the best: near. Explores: far. Balanced: between."
    # One vocabulary with the Role column and the Compared with cell.
    for word in (wording.ROLE_REFINES, wording.ROLE_EXPLORES, wording.ROLE_BALANCED):
        assert f"{word.capitalize()}: " in wording.WHY_ROLE_KEY_SHORT


def test_the_why_folds_plain_labels():
    assert wording.EXPECTED_GAIN == "Expected gain over your best (points)"
    assert wording.CLOSEST_MADE == "Closest formulation already made"
    assert wording.WHY_RECALCULATED == "Figures recalculated from today's results."
    assert wording.why_picked_after([23, 24, 25]) == (
        "Formulations 23–25, marked ·: what each adds to earlier picks.")
    assert wording.why_picked_after([23]) == (
        "Formulation 23, marked ·: what it adds to earlier picks.")
    assert wording.why_compared_with(15) == "Compared with: always Best so far, Formulation 15."
    assert wording.why_gain_short("verdict") == (
        "Expected gain not shown: predictions miss too often.")
    assert wording.why_gain_said(wording.WHY_CHECKING).startswith("Expected gain is not shown")
    assert wording.why_gain_said(wording.why_guard_line(verdict=True)).startswith(
        "Expected gain is not shown: ")


@pytest.mark.parametrize("kind", ["qlognei", "qlogei"])
def test_the_gain_tooltip_reads_plain_first_then_the_bo_term(kind):
    tip = wording.tip_expected_gain(kind)
    plain, _, term = tip.partition("BO term: ")
    assert "an average that counts no gain as zero" in plain
    assert "an estimate, not a promise" in plain
    assert term.startswith("expected improvement (the acquisition value")
    assert "approximate" in term and "512 quasi-Monte Carlo draws" in term
    if kind == "qlognei":
        # Statistician I1: no single incumbent; per draw, the noisy one.
        assert "qLogNEI" in term and "noisy incumbent" in term and "each posterior draw" in term
    else:
        assert "qLogEI" in term and "Best so far's measured score" in term


def test_the_512_draws_are_the_samplers_own():
    import pathlib
    source = (pathlib.Path(__file__).resolve().parent.parent / "food_bo.py").read_text()
    assert "SobolQMCNormalSampler(sample_shape=torch.Size([512]))" in source


def test_the_repeat_lines():
    assert wording.repeat_link(24) == "Repeat Formulation 24"
    assert wording.repeat_after_generate(24) == (
        "Click Generate first. The Repeat of Formulation 24 then opens below.")
    assert wording.repeat_after_generate_help(24) == (
        "Click Generate first: the Repeat of Formulation 24 then opens, filled in, under "
        "Add a formulation to this round.")


def _new_strings():
    return [wording.WHY_ROLE_KEY, wording.EXPECTED_GAIN, wording.CLOSEST_MADE,
            wording.WHY_RECALCULATED, wording.TIP_RECALCULATED, wording.why_picked_after([2, 3]),
            wording.why_picked_after([3]), wording.TIP_PICKED_AFTER, wording.why_compared_with(4),
            wording.tip_expected_gain("qlognei"), wording.tip_expected_gain("qlogei"),
            wording.TIP_CLOSEST_MADE, wording.LARGEST_DIFFERENCES_FROM_CLOSEST,
            wording.why_gain_said(wording.WHY_CHECKING), wording.repeat_after_generate(24)]


def test_the_new_strings_pass_the_vocabulary_guard():
    for text in _new_strings():
        assert not any(p.search(text) for p in _BANNED), text
        # Trial forms in a study of process settings alone.
        said = wording.for_project(_Settings(), text)
        assert not re.search(r"\b[Ff]ormulations?\b", said), said


def test_no_dash_chains_in_the_new_strings():
    """Style guide rule 9: at most one dash per line."""
    for text in _new_strings():
        assert text.count(" — ") <= 1, text


# ------------------------------------------------------------------ #
#  Links: the Why link, the Repeat link, the Explored link
# ------------------------------------------------------------------ #

@pytest.fixture
def links(monkeypatch):
    drawn = []

    def fake(label, target, section="top", fold=None, key=None, state=None):
        drawn.append({"label": label, "target": target, "section": section, "fold": fold,
                      "key": key, "state": state})
    monkeypatch.setattr(ui_insights, "link", fake)
    return drawn


def _round(opt, regime="warm", no=5):
    opt.pending_batch = [{"formulation": 99, "recipe": {}, "note": ""}]
    opt.pending_batch_no = no
    return bo_views.Generation(no, 99, [], regime, True, [99])


def _facts(opt, kind="explore", assumed=True, best=15, **kw):
    step = insights.NextStep(kind, assumed=assumed, **kw)
    return insights.StatusFacts(scored=20, next=step,
                                best=None if best is None else types.SimpleNamespace(
                                    formulation=best))


def test_the_why_link_lands_on_its_place_with_the_fold_open(links, monkeypatch):
    opt = _patty(rows=12)
    gen = _round(opt)
    monkeypatch.setattr(bo_views, "generation", lambda opt, r: gen)
    assert ui_insights._why_link(opt) is True
    [got] = links
    assert got["target"] == wording.TAB_BATCH
    assert got["section"] == ui_helpers.WHY_PLACE
    # v3: the place is the Round table on Make a round (no Why fold there).
    assert got["fold"] == ui_insights.WHY_FOLD_KEY


@pytest.mark.parametrize("kind", ["close", "explore", "see", "rising"])
def test_the_repeat_link_with_no_round_open(links, kind):
    opt = _patty(rows=12)
    assert ui_insights._repeat_link(opt, _facts(opt, kind, gain=2.0)) is True
    [got] = links
    assert got["label"] == wording.repeat_link(15)
    assert got["target"] == wording.TAB_BATCH
    # No Round open: the Add fold is not opened (adding first would open a
    # Round of the user's own alone); the top of Make a round says what to do.
    assert got["section"] == "top" and got["fold"] is None
    assert got["state"] == {ui_helpers.REPEAT_REQUEST_KEY: {
        "project": opt.project_name, "no": 15, "round": None}}
    assert got["key"] == ui_insights.REPEAT_LINK_KEY


def test_the_repeat_link_with_a_round_open_opens_the_add_fold(links, monkeypatch):
    """A Round open that the Why link does not explain (a cold Round here)."""
    opt = _patty(rows=12)
    gen = _round(opt, regime="cold", no=6)
    monkeypatch.setattr(bo_views, "generation", lambda opt, r: gen)
    assert ui_insights._repeat_link(opt, _facts(opt, "see")) is True
    [got] = links
    assert got["section"] == ui_helpers.ADD_OWN_PLACE
    assert got["fold"] == ui_helpers.ADD_OWN_KEY
    assert got["state"][ui_helpers.REPEAT_REQUEST_KEY]["round"] == 6


@pytest.mark.parametrize("facts_kw", [
    {"kind": "see", "assumed": False},              # the threshold is measured: no ask
    {"kind": "repeat", "assumed": False},           # a disagreeing Repeat: no ask
    {"kind": "misses", "assumed": False},
    {"kind": "early", "assumed": True},             # not an asking row
    {"kind": "see", "assumed": True, "best": None},  # no Best so far to repeat
])
def test_no_repeat_link_unless_next_asks_for_one(links, facts_kw):
    opt = _patty(rows=12)
    assert ui_insights._repeat_link(opt, _facts(opt, **facts_kw)) is False
    assert links == []


def test_the_why_link_outranks_the_repeat_link(links, monkeypatch):
    opt = _patty(rows=12)
    gen = _round(opt)
    monkeypatch.setattr(bo_views, "generation", lambda opt, r: gen)
    assert ui_insights._repeat_link(opt, _facts(opt, "see")) is False
    assert links == []


@pytest.mark.skip(reason="review cards prototype: pins the old Review layout (status lines, Explored fold, header); reconcile at the final merge")
def test_the_explored_link_lands_on_its_place_with_the_fold_open(links):
    opt = _patty(rows=12)
    [got] = links
    assert got["fold"] == ui_insights.EXPLORED_KEY


def test_a_link_can_leave_a_request_for_the_tab_it_opens():
    script = """
import streamlit as st
import ui_helpers
ui_helpers.link("Go", "2 · Make a round", section="here", fold="a_fold", key="go",
                state={"_a_request": {"no": 3}})
"""
    at = AppTest.from_string(script)
    at.run()
    at.button[0].click()
    at.run()
    assert at.session_state["_pending_tab"] == "2 · Make a round"
    assert at.session_state[ui_helpers.SCROLL_KEY] == "here"
    assert at.session_state["a_fold"] is True
    assert at.session_state["_a_request"] == {"no": 3}


# ------------------------------------------------------------------ #
#  Make a round: the Why fold
# ------------------------------------------------------------------ #

def _open_round(rounds=4, n=3):
    opt = _burger(rounds=rounds)
    ui_batch.ask_alone(opt, n_suggestions=n)
    return opt


def _app(opt, tab=wording.TAB_BATCH, state=None):
    # 600 s: the Generate run fits and asks, and timed out at 300 s on a
    # machine running six suites at once.
    at = AppTest.from_file(APP_PATH, default_timeout=600)
    at.session_state["_loaded_project"] = opt.project_name
    at.run()
    for k, v in (state or {}).items():
        at.session_state[k] = v
    at.session_state["main_tab"] = tab
    at.run()
    assert not at.exception, [e.message for e in at.exception]
    return at


def _block(node, key):
    """The keyed container `key` (its proto id ends with the key)."""
    proto = getattr(node, "proto", None)
    if getattr(node, "type", "") == "flex_container" and str(
            getattr(proto, "id", "")).endswith(f"-{key}"):
        return node
    for child in getattr(node, "children", {}).values():
        found = _block(child, key)
        if found is not None:
            return found
    return None


def _scripts(at):
    return [h.proto.body for h in at.get("html")]


@pytest.fixture
def why_open(monkeypatch):
    opt = _open_round()
    at = _app(opt, state={ui_insights.WHY_FOLD_KEY: True,
                          ui_helpers.SCROLL_KEY: ui_helpers.WHY_PLACE})
    return opt, at


def test_the_why_link_lands_on_the_round_table(why_open):
    """v3: Make a round has no Why fold; the link's place holds the Round
    table (ingredients down, the Round across)."""
    opt, at = why_open
    place = _block(at.tabs[1], ui_helpers.WHY_PLACE)
    assert place is not None
    assert [c for c in place.children.values() if getattr(c, "type", "") == "table"]
    assert not [e for e in at.tabs[1].expander if e.label == wording.WHY_THESE]
    # The landing: app.py scrolls to the place once, after the tabs.
    assert any(f".st-key-{ui_helpers.WHY_PLACE}" in body for body in _scripts(at))


# ------------------------------------------------------------------ #
#  Make a round: the Repeat request
# ------------------------------------------------------------------ #

def _request(opt, no, round_no=None):
    return {ui_helpers.REPEAT_REQUEST_KEY: {"project": opt.project_name, "no": no,
                                            "round": round_no}}


def _best_amounts(opt, no):
    index = opt.index_of_formulation(no)
    names = opt.recipe_history[index]
    start = insights.start_recipe(opt, index)
    return {v["name"]: round(float(start[v["name"]]), 2)
            for v in opt.varying_variables() if v["name"] in names}


def _add_fold(at):
    return next(e for e in at.tabs[1].expander
                if e.label == wording.ADD_OWN_EXPANDER)


def test_with_a_round_open_the_request_opens_the_add_fold_filled_in():
    opt = _open_round()
    best = opt.best_formulation_no()
    rows = [dict(r) for r in opt.pending_batch]
    at = _app(opt, state={**_request(opt, best, opt.pending_batch_no),
                          ui_helpers.SCROLL_KEY: ui_helpers.ADD_OWN_PLACE,
                          ui_helpers.ADD_OWN_KEY: True})
    assert at.session_state[ui_helpers.ADD_OWN_KEY] is True
    for name, value in _best_amounts(opt, best).items():
        assert at.session_state[ui_batch._own_key(name)] == value
    assert at.session_state["own_note"] == wording.repeat_of_formulation(best)
    assert ui_helpers.REPEAT_REQUEST_KEY not in at.session_state     # one shot
    place = _block(at.tabs[1], ui_helpers.ADD_OWN_PLACE)
    assert place is not None
    assert any(getattr(c, "label", "") == wording.ADD_OWN_EXPANDER
               for c in place.children.values())
    assert any(f".st-key-{ui_helpers.ADD_OWN_PLACE}" in body for body in _scripts(at))
    # Nothing is saved: the Round holds what it held.
    again = type(opt)(opt.project_name)
    assert [r["formulation"] for r in again.pending_batch] == [r["formulation"] for r in rows]


def test_with_no_round_open_the_request_waits_for_generate():
    opt = _burger(rounds=4)
    best = opt.best_formulation_no()
    at = _app(opt, state=_request(opt, best))
    assert wording.repeat_after_generate(best) in [c.value for c in at.tabs[1].caption]
    # The Add fold is not filled in yet: adding first would open a Round of
    # the user's own alone, with no Generate after it.
    assert at.session_state["own_note"] == ""
    assert at.session_state[ui_helpers.REPEAT_REQUEST_KEY]["no"] == best
    # Generate (the user's own click) starts the Round; the Repeat is then
    # ready in the opened fold, and not added.
    count = int(at.session_state["how_many"])
    at.session_state["main_tab"] = wording.TAB_BATCH
    next(b for b in at.tabs[1].button if b.key == "generate").click()
    at.run()
    at.session_state["main_tab"] = wording.TAB_BATCH
    at.run()
    assert not at.exception, [e.message for e in at.exception]
    again = type(opt)(opt.project_name)
    assert len(again.pending_batch) == count
    assert not any(r.get("note") for r in again.pending_batch)
    assert at.session_state["own_note"] == wording.repeat_of_formulation(best)
    for name, value in _best_amounts(opt, best).items():
        assert at.session_state[ui_batch._own_key(name)] == value
    assert at.session_state[ui_helpers.ADD_OWN_KEY] is True
    assert ui_helpers.REPEAT_REQUEST_KEY not in at.session_state


@pytest.mark.parametrize("stale", ["project", "round"])
def test_a_stale_request_is_dropped(stale):
    opt = _open_round()
    best = opt.best_formulation_no()
    request = _request(opt, best, opt.pending_batch_no)
    if stale == "project":
        request[ui_helpers.REPEAT_REQUEST_KEY]["project"] = "another project"
    else:
        request[ui_helpers.REPEAT_REQUEST_KEY]["round"] = opt.pending_batch_no + 1
    at = _app(opt, state=request)
    assert at.session_state["own_note"] == ""
    assert ui_helpers.REPEAT_REQUEST_KEY not in at.session_state


# ------------------------------------------------------------------ #
#  Insights: the Explored fold's place
# ------------------------------------------------------------------ #

def test_a_landing_clears_the_header_bar():
    """Measured at 1280 x 800: scrollIntoView put the Why fold's heading
    at y 0, behind Streamlit's 60 px header; the landing now subtracts the
    header's height."""
    script = ui_helpers.scroll_script(ui_helpers.WHY_PLACE, 1)
    assert f".st-key-{ui_helpers.WHY_PLACE}" in script
    assert 'stHeader' in script and "getBoundingClientRect().height" in script
    top = ui_helpers.scroll_script("top", 1)
    assert "main.scrollTop = 0" in top


def test_a_landing_follows_its_place_while_the_screen_fills_in():
    """Measured: the Round's predictions arrived after the landing and moved
    the Add fold down 55-430 px. The place is kept under the header for 8 s
    (80 x 100 ms), until the reader scrolls, clicks or types."""
    script = ui_helpers.scroll_script(ui_helpers.ADD_OWN_PLACE, 1)
    assert "follow(k - 1), 100" in script and "})(80);" in script
    for kind in ("wheel", "touchstart", "keydown", "mousedown"):
        assert f'"{kind}"' in script
