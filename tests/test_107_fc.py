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
import wording
from tests.test_app_ui import APP_PATH
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


def test_the_512_draws_are_the_samplers_own():
    import pathlib
    source = (pathlib.Path(__file__).resolve().parent.parent / "food_bo.py").read_text()
    assert "SobolQMCNormalSampler(sample_shape=torch.Size([512]))" in source


def _new_strings():
    return [wording.WHY_ROLE_KEY, 
            ]


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


def _round(opt, regime="warm", no=5):
    opt.pending_batch = [{"formulation": 99, "recipe": {}, "note": ""}]
    opt.pending_batch_no = no
    return bo_views.Generation(no, 99, [], regime, True, [99])


def _facts(opt, kind="explore", assumed=True, best=15, **kw):
    step = insights.NextStep(kind, assumed=assumed, **kw)
    return insights.StatusFacts(scored=20, next=step,
                                best=None if best is None else types.SimpleNamespace(
                                    formulation=best))


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


# ------------------------------------------------------------------ #
#  Make a round: the Repeat request
# ------------------------------------------------------------------ #


def _best_amounts(opt, no):
    index = opt.index_of_formulation(no)
    names = opt.recipe_history[index]
    start = insights.start_recipe(opt, index)
    return {v["name"]: round(float(start[v["name"]]), 2)
            for v in opt.varying_variables() if v["name"] in names}


def _add_fold(at):
    return next(e for e in at.tabs[1].expander
                if e.label == wording.ADD_OWN_EXPANDER)


# ------------------------------------------------------------------ #
#  Insights: the Explored fold's place
# ------------------------------------------------------------------ #


def test_a_landing_follows_its_place_while_the_screen_fills_in():
    """Measured: the Round's predictions arrived after the landing and moved
    the Add fold down 55-430 px. The place is kept under the header for 8 s
    (80 x 100 ms), until the reader scrolls, clicks or types."""
    script = ui_helpers.scroll_script(ui_helpers.ADD_OWN_PLACE, 1)
    assert "follow(k - 1), 100" in script and "})(80);" in script
    for kind in ("wheel", "touchstart", "keydown", "mousedown"):
        assert f'"{kind}"' in script
