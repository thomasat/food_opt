"""0.8.0 speed, Plan A Tasks 8-10 on screen (spec 2026-09-30 §1, §3, §10;
review I1, I15, I16, V28): the fits start when a project opens or its data
changes, the tab fills in part by part behind one progress line, reopening
reads the cache beside the project, and Make a round's Overall-score
column shows "…" until it can show the widened range."""
import concurrent.futures as cf

import pytest
from streamlit.testing.v1 import AppTest

import insights
import insights_cache
import insights_jobs
import ui_batch
import ui_insights
import wording
from tests.test_app_ui import APP_PATH
from tests.test_insights import _burger


@pytest.fixture(autouse=True)
def _isolate(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(insights_jobs, "_JOBS", {})
    monkeypatch.setattr(insights_jobs, "_INFLIGHT", {})


class _Stuck:
    """A pool whose workers never answer: the screens wait for them."""

    broken = False

    def __init__(self):
        self.sent = []

    def submit(self, priority, request):
        self.sent.append((priority, request["kind"]))
        return cf.Future()

    def idle(self):
        return False


def _workers(monkeypatch, pool):
    """Workers on, with `pool` standing in for them (None: none started)."""
    monkeypatch.setattr(insights_jobs, "workers_on", lambda: pool is not None)
    monkeypatch.setattr(insights_jobs, "pool", lambda: pool)


@pytest.fixture
def stuck(monkeypatch):
    pool = _Stuck()
    _workers(monkeypatch, pool)
    return pool


@pytest.fixture
def five_rounds():
    """Fifteen scored formulations in five Rounds: three checked, and more
    than five Overall-score checks before the last one."""
    return _burger(rounds=5, noise=0.6)


def _app(project):
    at = AppTest.from_file(APP_PATH, default_timeout=300)
    at.session_state["_loaded_project"] = project
    return at


def _open(at, tab):
    at.run()
    at.session_state["main_tab"] = tab
    at.run()
    return at


def _texts(at):
    tab = at.tabs[3]
    return ([m.value for m in tab.markdown] + [c.value for c in tab.caption]
            + [e.label for e in tab.expander])


def _seed(at, opt, stage):
    """This session's memo holding exactly what `stage` needs."""
    memo = {}
    specs = insights_jobs.stage_specs(opt, insights_jobs.plan(opt), stage)
    insights_jobs.compute_inline(opt, memo, specs)
    at.session_state[ui_insights.MEMO_KEY] = (opt.project_name, memo)


# ------------------------------------------------------------------ #
#  Task 8: the fits start by themselves; the cache beside the project
# ------------------------------------------------------------------ #

def test_opening_a_project_sends_its_fits_overall_score_first(stuck, five_rounds):
    at = _app(five_rounds.project_name)
    at.run()
    assert not at.exception
    specs = insights_jobs.plan(five_rounds)
    assert len(stuck.sent) == sum(1 + len(insights.folds(len(s.rows))) for s in specs)
    assert stuck.sent[0] == (0, "main")


def test_a_change_to_the_data_sends_the_fits_again(stuck, five_rounds):
    at = _app(five_rounds.project_name)
    at.run()
    before = len(stuck.sent)
    opt = at.session_state["optimizer"]
    opt.tell({"Pea": 6.0, "Oil": 4.0, "Water": 90.0}, {"Firmness": 5.0, "Juiciness": 6.0},
             formulation_no=16, batch_no=6)                       # a result saved
    at.run()
    assert len(stuck.sent) > before


def test_with_too_few_scored_nothing_is_sent(stuck):
    opt = _burger(rounds=1)
    at = _app(opt.project_name)
    at.run()
    assert stuck.sent == []


def test_a_reopened_project_reads_its_cache_and_fits_nothing(five_rounds, monkeypatch):
    monkeypatch.setenv("FOODOPT_INSIGHTS_CACHE", "1")
    first = _open(_app(five_rounds.project_name), wording.TAB_INSIGHTS)
    assert not first.exception
    assert insights_cache.load(five_rounds.project_name).memo
    monkeypatch.setattr(insights, "_fit_gp", lambda *a: pytest.fail("fitted again"))
    monkeypatch.setattr(insights, "predicted_best", lambda *a: pytest.fail("searched again"))
    again = _open(_app(five_rounds.project_name), wording.TAB_INSIGHTS)
    assert not again.exception
    assert _texts(again) == _texts(first)


def test_a_project_with_everything_cached_starts_no_workers(five_rounds, monkeypatch):
    monkeypatch.setenv("FOODOPT_INSIGHTS_CACHE", "1")
    _open(_app(five_rounds.project_name), wording.TAB_INSIGHTS)     # fitted here and saved
    started = []
    monkeypatch.setattr(insights_jobs, "workers_on", lambda: True)
    monkeypatch.setattr(insights_jobs, "pool", lambda: started.append(1) or _Stuck())
    at = _open(_app(five_rounds.project_name), wording.TAB_INSIGHTS)
    assert not at.exception and started == []
    assert wording.MORE_CHARTS in _texts(at)


def test_the_cache_holds_what_this_data_uses_and_no_more(five_rounds, monkeypatch):
    monkeypatch.setenv("FOODOPT_INSIGHTS_CACHE", "1")
    _open(_app(five_rounds.project_name), wording.TAB_INSIGHTS)
    saved = insights_cache.load(five_rounds.project_name)
    assert set(saved.memo) == {s.key for s in insights_jobs.plan(five_rounds)}
    assert set(saved.rounds) <= set(insights.round_keys(five_rounds).values())
    assert saved.best[0] == insights.setup_key(five_rounds)


# ------------------------------------------------------------------ #
#  Task 9: the tab fills in part by part, behind one progress line
# ------------------------------------------------------------------ #

def test_while_nothing_is_in_one_progress_line_line_1_and_the_foot(stuck, five_rounds):
    at = _open(_app(five_rounds.project_name), wording.TAB_INSIGHTS)
    assert not at.exception
    [bar] = at.tabs[3].get("progress")
    total = insights_jobs.rounds_done(five_rounds, insights_jobs.plan(five_rounds), {})[1]
    assert bar.proto.text == wording.insights_progress(0, total)
    texts = " ".join(_texts(at))
    assert f"**{wording.Q_STAND}**" in texts and "Best so far" in texts
    for later in (wording.Q_TRUST, wording.Q_NEXT, wording.MORE_CHARTS):
        assert later not in texts
    assert "insights_foot" in [b.key for b in at.tabs[3].button]
    assert at.tabs[3].get("spinner") == []


def test_answer_1_once_its_checks_are_in(stuck, five_rounds):
    at = _app(five_rounds.project_name)
    at.run()
    _seed(at, five_rounds, "trust")
    at.session_state["main_tab"] = wording.TAB_INSIGHTS
    at.run()
    texts = " ".join(_texts(at))
    assert f"**{wording.Q_TRUST}**" in texts
    assert wording.Q_NEXT not in texts and wording.SHOW_PREDICTIONS not in texts
    assert len(at.tabs[3].get("progress")) == 1


def test_answers_2_and_3_once_the_overall_score_model_is_in(stuck, five_rounds):
    at = _app(five_rounds.project_name)
    at.run()
    _seed(at, five_rounds, "stand")
    at.session_state["main_tab"] = wording.TAB_INSIGHTS
    at.run()
    texts = " ".join(_texts(at))
    assert f"**{wording.Q_TRUST}**" in texts and f"**{wording.Q_NEXT}**" in texts
    assert wording.MORE_CHARTS not in texts
    assert at.tabs[3].get("spinner") == []


def test_each_part_reads_the_same_as_on_the_finished_tab(five_rounds, monkeypatch):
    _workers(monkeypatch, None)                              # no workers: all here
    done = _open(_app(five_rounds.project_name), wording.TAB_INSIGHTS)
    whole = [m.value for m in done.tabs[3].markdown]
    _workers(monkeypatch, _Stuck())
    at = _app(five_rounds.project_name)
    at.run()
    _seed(at, five_rounds, "stand")
    at.session_state["main_tab"] = wording.TAB_INSIGHTS
    at.run()
    part = [m.value for m in at.tabs[3].markdown]
    answers = part[:part.index(wording.HOW_MODELS)]        # up to the folds, which wait
    assert f"**{wording.Q_NEXT}**" in answers
    assert answers == whole[:len(answers)]


def test_with_everything_in_the_whole_tab_and_no_progress_line(stuck, five_rounds):
    at = _app(five_rounds.project_name)
    at.run()
    _seed(at, five_rounds, "all")
    at.session_state["main_tab"] = wording.TAB_INSIGHTS
    at.run()
    assert wording.MORE_CHARTS in _texts(at)
    assert at.tabs[3].get("progress") == []


def test_a_failed_worker_falls_back_to_fitting_here(five_rounds, monkeypatch):
    class _Broken(_Stuck):
        def submit(self, priority, request):
            fut = cf.Future()
            fut.set_exception(RuntimeError("the Insights workers stopped"))
            return fut
    _workers(monkeypatch, _Broken())
    at = _open(_app(five_rounds.project_name), wording.TAB_INSIGHTS)
    assert not at.exception
    assert wording.MORE_CHARTS in _texts(at)


# ------------------------------------------------------------------ #
#  Task 10: Make a round's column (I1) and its track-record line (I16)
# ------------------------------------------------------------------ #

def _column(at):
    [table] = [d for d in at.tabs[1].dataframe if wording.PREDICTED_SCORE_COLUMN in d.value]
    return list(table.value[wording.PREDICTED_SCORE_COLUMN])


def _with_open_round(opt):
    opt.set_pending_batch([{"Pea": 6.0, "Oil": 4.0, "Water": 90.0},
                           {"Pea": 7.0, "Oil": 3.0, "Water": 90.0}])
    return opt


def test_the_column_shows_dots_while_its_models_are_fitted(stuck, five_rounds, monkeypatch):
    _with_open_round(five_rounds)
    monkeypatch.setattr(insights, "_fit_gp", lambda *a: pytest.fail("fitted on Make a round"))
    at = _open(_app(five_rounds.project_name), wording.TAB_BATCH)
    assert not at.exception
    assert _column(at) == [wording.PREDICTION_PENDING] * 2


def test_the_column_is_widened_by_every_rounds_overall_score_check(five_rounds):
    _with_open_round(five_rounds)
    at = _open(_app(five_rounds.project_name), wording.TAB_BATCH)
    assert not at.exception
    fitted = insights.fit_all(five_rounds)
    factor = insights.score_widen(insights.check_rounds(five_rounds))
    assert factor > 1.0
    rows = ui_batch.open_rows(five_rounds)
    preds = insights.round_predictions(five_rounds, fitted, rows)
    want = [ui_insights.score_text(insights.widened(fitted[insights.SCORE], p[insights.SCORE],
                                                    factor))
            for p in preds.values()]
    assert _column(at) == want


def test_the_track_line_sits_under_the_table_once_the_checks_are_in(five_rounds):
    _with_open_round(five_rounds)
    at = _app(five_rounds.project_name)
    at.run()
    _seed(at, five_rounds, "all")
    at.session_state["main_tab"] = wording.TAB_BATCH
    at.run()
    t = insights.round_track(five_rounds, insights.check_rounds(five_rounds))
    line = (wording.round_track_holding(t.inside, t.total, t.see_insights) if not t.rough else
            wording.round_track_rough(t.inside, t.total,
                                      None if t.gap is None else f"{abs(t.gap):.1f}",
                                      (t.gap or 0) < 0, t.see_insights))
    assert line in [c.value for c in at.tabs[1].caption]


def test_values_typed_on_make_a_round_survive_the_column_filling_in(stuck, five_rounds):
    _with_open_round(five_rounds)
    at = _open(_app(five_rounds.project_name), wording.TAB_BATCH)
    assert _column(at) == [wording.PREDICTION_PENDING] * 2
    at.number_input(key="f16_Firmness").set_value(5.5)
    at.run()
    _seed(at, five_rounds, "all")                  # the workers' models land
    at.session_state["main_tab"] = wording.TAB_BATCH
    at.run()                                       # what the poller's rerun does
    assert not at.exception
    assert wording.PREDICTION_PENDING not in _column(at)
    assert at.number_input(key="f16_Firmness").value == 5.5
