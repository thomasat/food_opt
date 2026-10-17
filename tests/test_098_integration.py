"""0.8.0 BO views, Task 8: the wave as a whole (spec 2026-09-30-bo-views §5,
§7; 04 §5; review I10). At most one link to Insights on any screen, never
lit; the new screens say no retired word; the views' own costs at the
owner's bench shape (60 formulations x 4 measurements) stay inside the
speed spec's budgets; the lines the wave retired are gone."""
import os
import re
import time

import pytest
import torch
from streamlit.testing.v1 import AppTest

import bo_views
import insights
import insights_jobs
import ui_batch
import ui_insights
import wording
from tests import speed_projects
from tests.test_app_ui import APP_PATH
from tests.test_food_bo import _BANNED, _FTUE_ALLOWED
from tests.test_insights import _burger
from ui_helpers import INSIGHTS_OPENED_KEY


@pytest.fixture(autouse=True)
def _isolate(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    with bo_views.fresh():
        yield


def _run(at, tab):
    at.session_state["main_tab"] = tab
    at.run()
    assert not at.exception
    return at


def _links(at):
    return [(i, b) for i, t in enumerate(at.tabs) for b in t.button
            if str(b.key or "").startswith("link_")]


# ------------------------------------------------------------------ #
#  04 §5: at most one link on a screen, never the lit button
# ------------------------------------------------------------------ #

def test_each_screen_offers_at_most_one_link(monkeypatch):
    opt = _burger(rounds=4)
    at = AppTest.from_file(APP_PATH, default_timeout=300)
    at.session_state["_loaded_project"] = opt.project_name
    at.run()
    # No Round open: Results' E1; Make a round's E3 (Next step forced to a
    # boxed-in best) once Insights has worked its checks out.
    monkeypatch.setattr(insights, "next_step", lambda *a, **k: insights.NextStep("close"))
    _run(at, wording.TAB_INSIGHTS)
    del at.session_state[INSIGHTS_OPENED_KEY]
    seen = set()
    for index, tab in ((2, wording.TAB_RESULTS), (1, wording.TAB_BATCH),
                       (3, wording.TAB_INSIGHTS)):
        _run(at, tab)
        seen |= {b.key for i, b in _links(at) if i == index}
        per_tab = {}
        for i, b in _links(at):
            per_tab.setdefault(i, []).append(b)
            assert b.proto.type == "tertiary", b.key
        assert all(len(v) <= 1 for v in per_tab.values()), per_tab
    # v3: Make a round has no status line and no E3 (no verdicts or
    # forecasts there), so no link of its own.
    assert seen == {"link_round_checked"}


def test_the_new_screens_say_no_retired_word():
    """03 §3: no "likely range", "On track", "though", hedges or advice
    words; and the vocabulary guard over everything the two screens say."""
    opt = _burger(rounds=4, noise=0.4)
    ui_batch.ask_alone(opt, n_suggestions=3)
    at = AppTest.from_file(APP_PATH, default_timeout=300)
    at.session_state["_loaded_project"] = opt.project_name
    at.run()
    at.session_state[ui_insights.EACH_KEY] = True
    at.session_state[ui_insights.PROGRESS_KEY] = True
    at.session_state[ui_insights.HOW_KEY] = True
    said = []
    for tab in (wording.TAB_INSIGHTS, wording.TAB_BATCH):
        _run(at, tab)
        said += [m.value for m in at.markdown] + [c.value for c in at.caption]
    # T8 review: the audit's "keep in mind", "rough" and "a little" too.
    retired = re.compile(r"\b(likely|On track|though|treat|Barely|Not lately|stands out|"
                         r"worth|some luck|too few to be sure|[Kk]eep in mind|[Rr]ough|[Aa] little)\b")
    assert [t for t in said if retired.search(t)] == []
    # The one scoped allowance (first-time wave, review R-B2): Model and
    # method's lead, said as the exact string the guard allows.
    assert [t for t in said if t not in _FTUE_ALLOWED
            and any(p.search(t) for p in _BANNED)] == []


def test_the_lines_the_wave_retired_are_gone():
    for name in ("bias_tail", "PREDICTIONS_CAPTION"):
        assert not hasattr(wording, name), name
    assert not hasattr(ui_insights, "track_line")


# ------------------------------------------------------------------ #
#  Review I10: the views' own costs at 60 x 4, workers fitting the models
# ------------------------------------------------------------------ #

@pytest.fixture(scope="module")
def pool():
    p = insights_jobs.Pool(int(os.environ.get("FOODOPT_TIMING_WORKERS", "4")))
    assert p.wait_ready(120)
    yield p
    p.close()


@pytest.mark.skipif(os.environ.get("FOODOPT_SKIP_TIMING") == "1", reason="FOODOPT_SKIP_TIMING=1")
def test_the_views_at_sixty_by_four(pool):
    """With the models already fitted by the workers (the speed spec's own
    timing covers those): the §3.2 line, the pool, the Why fold for an open
    Round of 6 (an exact rebuild) and the whole Expected gain by
    Round history. CI bounds; on the owner's Mac about 0.02, 1.2, 0.4 and
    1.9 s."""
    opt = speed_projects.bench(60, 4)
    torch.manual_seed(0)
    memo = {}
    insights_jobs.Job(opt, pool, memo).wait(memo)
    with insights.memo_in_use(memo):
        insights.check_rounds(opt)
        t0 = time.perf_counter()
        line = bo_views.latest_gain(opt)
        t_line = time.perf_counter() - t0
        t1 = time.perf_counter()
        assert bo_views.feasible_pool(opt) is not None
        t_pool = time.perf_counter() - t1
        ui_batch.ask_alone(opt, n_suggestions=6)
        t2 = time.perf_counter()
        why = bo_views.why(opt)
        t_why = time.perf_counter() - t2
        t3 = time.perf_counter()
        history = bo_views.round_history(opt)
        t_history = time.perf_counter() - t3
    print(f"\nviews at 60x4: line {t_line:.2f} s, pool {t_pool:.2f} s, why {t_why:.2f} s, "
          f"history {t_history:.2f} s ({len(history)} Rounds)")
    assert line is not None and why.exact and len(why.rows) == 6
    assert t_line < 1.0
    assert t_pool < 3.0
    assert t_why < 3.0
    assert t_history < 8.0
