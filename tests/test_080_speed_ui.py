"""0.8.0 speed, Plan A Tasks 8-10 on screen (spec 2026-09-30 §1, §3, §10;
review I1, I15, I16, V28): the fits start when a project opens or its data
changes, the tab fills in part by part behind one progress line, reopening
reads the cache beside the project, and Make a round's Overall-score
column shows "…" until it can show the widened range."""
import concurrent.futures as cf

import pytest
from tests.demo_build_reason import REVIEW_FOUR
from streamlit.testing.v1 import AppTest

import insights
import insights_cache
import insights_jobs
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


def _spinners(monkeypatch):
    """Every st.spinner entered from now on (AppTest drops a spinner once
    its block exits, so the element tree cannot tell; review C I-4)."""
    import streamlit as st
    seen = []
    real = st.spinner

    def spinner(text="", *a, **k):
        # The triangle in the collapsed "More charts" fold has a spinner of
        # its own for its own work, once it is drawn: not a wait of the tab.
        import inspect
        if not any(f.function == "_mixture_cached" for f in inspect.stack()):
            seen.append(text)
        return real(text, *a, **k)
    monkeypatch.setattr(st, "spinner", spinner)
    return seen


def _bars(monkeypatch):
    """Every progress bar's text from now on."""
    from streamlit.delta_generator import DeltaGenerator
    texts = []
    real = DeltaGenerator.progress

    def progress(self, value, text=None, **k):
        texts.append(text)
        return real(self, value, text=text, **k)
    monkeypatch.setattr(DeltaGenerator, "progress", progress)
    return texts


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
    assert len(stuck.sent) == sum(1 + len(insights.cv_folds(five_rounds, s.rows))
                                  for s in specs)
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


@pytest.mark.skip(reason=REVIEW_FOUR)
def test_a_project_with_everything_cached_starts_no_workers(five_rounds, monkeypatch):
    monkeypatch.setenv("FOODOPT_INSIGHTS_CACHE", "1")
    _open(_app(five_rounds.project_name), wording.TAB_INSIGHTS)     # fitted here and saved
    started = []
    monkeypatch.setattr(insights_jobs, "workers_on", lambda: True)
    monkeypatch.setattr(insights_jobs, "pool", lambda: started.append(1) or _Stuck())
    at = _open(_app(five_rounds.project_name), wording.TAB_INSIGHTS)
    assert not at.exception and started == []
    assert wording.PAIR_FOLD in _texts(at)


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

@pytest.mark.skip(reason=REVIEW_FOUR)
def test_while_nothing_is_in_one_progress_line_line_1_and_the_foot(stuck, five_rounds,
                                                                     monkeypatch):
    spins = _spinners(monkeypatch)
    at = _open(_app(five_rounds.project_name), wording.TAB_INSIGHTS)
    assert not at.exception
    [bar] = at.tabs[3].get("progress")
    total = insights_jobs.rounds_done(five_rounds, insights_jobs.plan(five_rounds), {})[1]
    assert bar.proto.text == wording.insights_progress(0, total)
    texts = " ".join(_texts(at))
    assert f"##### {wording.Q_STAND}" in texts and "Best so far" in texts
    for later in (wording.Q_TRUST, wording.Q_NEXT, wording.PAIR_FOLD):
        assert later not in texts
    assert "insights_foot" in [b.key for b in at.tabs[3].button]
    assert spins == []


@pytest.mark.skip(reason=REVIEW_FOUR)
def test_answer_1_once_its_checks_are_in(stuck, five_rounds):
    at = _app(five_rounds.project_name)
    at.run()
    _seed(at, five_rounds, "trust")
    at.session_state["main_tab"] = wording.TAB_INSIGHTS
    at.run()
    texts = " ".join(_texts(at))
    assert f"##### {wording.Q_TRUST}" in texts
    # Since ftue W1 the measured-against-predicted chart is answer 1's own
    # (inside Prediction figures); answers 2 and 3 still wait.
    assert wording.Q_NEXT not in texts
    assert len(at.tabs[3].get("progress")) == 1


@pytest.mark.skip(reason=REVIEW_FOUR)
def test_answers_2_and_3_once_the_overall_score_model_is_in(stuck, five_rounds, monkeypatch):
    at = _app(five_rounds.project_name)
    at.run()
    _seed(at, five_rounds, "stand")
    spins = _spinners(monkeypatch)
    at.session_state["main_tab"] = wording.TAB_INSIGHTS
    at.run()
    texts = " ".join(_texts(at))
    assert f"##### {wording.Q_TRUST}" in texts and f"##### {wording.Q_NEXT}" in texts
    assert wording.PAIR_FOLD not in texts
    assert spins == []                  # the predicted best's search has no spinner of its own


@pytest.mark.skip(reason=REVIEW_FOUR)
def test_with_everything_in_the_whole_tab_and_no_progress_line(stuck, five_rounds):
    at = _app(five_rounds.project_name)
    at.run()
    _seed(at, five_rounds, "all")
    at.session_state["main_tab"] = wording.TAB_INSIGHTS
    at.run()
    assert wording.PAIR_FOLD in _texts(at)
    assert at.tabs[3].get("progress") == []


@pytest.mark.skip(reason=REVIEW_FOUR)
def test_a_failed_worker_falls_back_to_fitting_here(five_rounds, monkeypatch):
    class _Broken(_Stuck):
        def submit(self, priority, request):
            fut = cf.Future()
            fut.set_exception(RuntimeError("the Insights workers stopped"))
            return fut
    _workers(monkeypatch, _Broken())
    at = _open(_app(five_rounds.project_name), wording.TAB_INSIGHTS)
    assert not at.exception
    assert wording.PAIR_FOLD in _texts(at)


def test_the_predicted_best_is_searched_once_its_models_are_in_whatever_the_tab(
        stuck, five_rounds):
    at = _app(five_rounds.project_name)
    at.run()
    assert ui_insights.BEST_KEY not in at.session_state
    _seed(at, five_rounds, "column")
    at.session_state["main_tab"] = wording.TAB_RESULTS
    at.run()
    assert at.session_state[ui_insights.BEST_KEY][0] == insights.setup_key(five_rounds)
    assert ui_insights.CACHE_KEY not in at.session_state


def test_every_check_is_worked_out_after_the_page_once_all_are_in(stuck, five_rounds):
    at = _app(five_rounds.project_name)
    at.run()
    _seed(at, five_rounds, "all")
    at.session_state["main_tab"] = wording.TAB_RESULTS
    at.run()
    key, value = at.session_state[ui_insights.CACHE_KEY]
    assert key == insights.data_key(five_rounds) and value["checks"]


@pytest.mark.skip(reason=REVIEW_FOUR)
def test_a_part_that_lands_while_the_page_is_drawn_is_drawn_by_one_more_run(
        stuck, five_rounds, monkeypatch):
    """After Save results the last fits often land between the tab's draw
    and keep_up: the page must not stay on the progress line with nothing
    left to wait for (browser: stuck until the next click)."""
    at = _app(five_rounds.project_name)
    at.run()
    _seed(at, five_rounds, "stand")
    real = ui_insights._catch_up
    landed = []

    def catch_up(opt):
        if not landed:                   # the rest lands after render drew
            landed.append(1)
            insights_jobs.compute_inline(opt, ui_insights.memo(opt), insights_jobs.plan(opt))
        real(opt)
    monkeypatch.setattr(ui_insights, "_catch_up", catch_up)
    at.session_state["main_tab"] = wording.TAB_INSIGHTS
    at.run()
    assert not at.exception and landed
    assert wording.PAIR_FOLD in _texts(at)
    assert at.tabs[3].get("progress") == []


# ------------------------------------------------------------------ #
#  Plan A review C I-1, I-3, I-4: workers that fail hand the tab to the
#  in-process path with its one counted bar, nothing is fitted quietly
#  on another tab, and the poller stops once nothing is in flight
# ------------------------------------------------------------------ #

def _fits(monkeypatch, where=None):
    """(name, rows) of every model fitted from now on; `where`, a list
    whose last item says what is running, tags each fit."""
    got = []
    real = insights._fit_gp

    def fit_gp(opt, name, rows):
        got.append((name, len(rows), where[-1] if where else None))
        return real(opt, name, rows)
    monkeypatch.setattr(insights, "_fit_gp", fit_gp)
    return got


def test_workers_that_fail_mid_fill_fit_nothing_quietly_on_another_tab(
        stuck, five_rounds, monkeypatch):
    at = _app(five_rounds.project_name)
    at.run()
    _seed(at, five_rounds, "column")
    at.session_state["main_tab"] = wording.TAB_INSIGHTS
    at.run()                                             # filling, workers fitting
    insights_jobs.current_job(five_rounds).failed = True    # a worker died
    fits = _fits(monkeypatch)
    at.session_state["main_tab"] = wording.TAB_RESULTS
    at.run()
    assert not at.exception
    assert fits == []                                    # nothing fitted on Results


@pytest.mark.skip(reason=REVIEW_FOUR)
def test_workers_that_fail_mid_fill_hand_insights_to_the_counted_bar(
        stuck, five_rounds, monkeypatch):
    at = _app(five_rounds.project_name)
    at.run()
    _seed(at, five_rounds, "column")
    at.session_state["main_tab"] = wording.TAB_INSIGHTS
    at.run()
    insights_jobs.current_job(five_rounds).failed = True
    where = ["page"]
    real = ui_insights._catch_up

    def catch_up(opt):
        where.append("catch-up")
        try:
            real(opt)
        finally:
            where.pop()
    monkeypatch.setattr(ui_insights, "_catch_up", catch_up)
    fits = _fits(monkeypatch, where)
    bars, spins = _bars(monkeypatch), _spinners(monkeypatch)
    at.session_state["main_tab"] = wording.TAB_INSIGHTS
    at.run()
    assert not at.exception
    assert wording.PAIR_FOLD in _texts(at)             # the whole tab, fitted here
    assert fits and not [f for f in fits if f[2] == "catch-up"]
    total = insights_jobs.rounds_done(five_rounds, insights_jobs.plan(five_rounds), {})[1]
    assert bars and all(t is not None and t.startswith("Checking predictions against your "
                                                       "results: ") and t.endswith(
        f"of {total} Rounds checked…") for t in bars)
    assert spins == []                                   # one indicator (review C M-2)


@pytest.mark.skip(reason=REVIEW_FOUR)
def test_workers_that_stop_while_the_page_is_drawn_rerun_it_once(stuck, five_rounds,
                                                                 monkeypatch):
    """Review C I-1's rarer variant: the workers are given up on between
    the filling draw and keep_up; the page is drawn again, here."""
    at = _app(five_rounds.project_name)
    at.run()
    real = ui_insights._fill_in
    stopped = []

    def fill_in(*a, **k):
        real(*a, **k)
        if not stopped:
            stopped.append(1)
            monkeypatch.setattr(insights_jobs, "_GAVE_UP", True)
            monkeypatch.setattr(insights_jobs, "workers_on", lambda: False)
    monkeypatch.setattr(ui_insights, "_fill_in", fill_in)
    at.session_state["main_tab"] = wording.TAB_INSIGHTS
    at.run()
    assert not at.exception and stopped
    assert wording.PAIR_FOLD in _texts(at)


def test_the_poller_polls_while_fits_are_in_flight_and_stops_once_they_are_in(
        stuck, five_rounds, monkeypatch):
    """Review C I-3, I-4: the poller's own step says "rerun" once nothing
    is in flight, whichever tab is open, so its fragment stops; a full run
    with nothing in flight starts no poller."""
    steps = []

    def poller(opt, stages):
        steps.append(ui_insights._poll_step(opt, stages))
        if len(steps) == 1:                              # then everything lands
            insights_jobs.compute_inline(opt, ui_insights.memo(opt), insights_jobs.plan(opt))
            steps.append(ui_insights._poll_step(opt, stages))
    monkeypatch.setattr(ui_insights, "_poller", poller)
    at = _app(five_rounds.project_name)
    at.run()                                             # Results: stages ()
    assert steps == [False, True]
    at.session_state["main_tab"] = wording.TAB_RESULTS
    at.run()
    assert steps == [False, True]                        # nothing in flight: no poller


def test_answer_1_waits_when_a_verdict_rounds_fit_failed(stuck, five_rounds):
    """Review C I-4: the settled guard. A measurement check in the
    verdict's Rounds failed, so the Rounds the verdict reads may not be the
    planned ones: answer 1 waits for everything."""
    at = _app(five_rounds.project_name)
    at.run()
    _seed(at, five_rounds, "trust")
    memo = at.session_state[ui_insights.MEMO_KEY][1]
    specs = insights_jobs.stage_specs(five_rounds, insights_jobs.plan(five_rounds), "trust")
    window = next(s for s in specs if s.name != insights.SCORE and s.stage != insights_jobs.NOW)
    memo[window.key] = {"error": "no fit"}
    at.session_state["main_tab"] = wording.TAB_INSIGHTS
    at.run()
    assert not at.exception
    assert f"##### {wording.Q_TRUST}" not in _texts(at)


# ------------------------------------------------------------------ #
#  Plan B Task 6 review I-1 (review I15's order): the header, line 1 and
#  the foot are drawn before anything is fitted or worked out
# ------------------------------------------------------------------ #

def _draw_log(monkeypatch):
    """What the Insights tab draws and works out, in order, from the
    moment render starts."""
    log = []

    def logged(module, name, event):
        real = getattr(module, name)

        def wrapper(*a, **k):
            log.append(event)
            return real(*a, **k)
        monkeypatch.setattr(module, name, wrapper)
    logged(ui_insights, "render", "render")
    logged(ui_insights, "header_text", "header")
    logged(ui_insights, "_stand_lines", "line 1")
    logged(ui_insights, "_foot", "foot")
    for name in ("fit_all", "check_rounds", "predicted_best", "_fit_gp"):
        logged(insights, name, "compute")
    return log


def _drawn_first(log):
    tab = log[log.index("render"):]
    first = tab.index("compute")
    return tab[:first], tab[first:]


@pytest.mark.skip(reason="review cards prototype: pins the old Review layout (status lines, Explored fold, header); reconcile at the final merge")
def test_a_cold_tab_draws_the_header_line_1_and_the_foot_before_fitting(five_rounds, monkeypatch):
    """No workers, nothing cached: the tab fits in this process."""
    at = _app(five_rounds.project_name)
    at.run()
    log = _draw_log(monkeypatch)
    at.session_state["main_tab"] = wording.TAB_INSIGHTS
    at.run()
    assert not at.exception
    before, after = _drawn_first(log)
    # Ftue R-I8: the button sits under the three status lines' places, so
    # it is drawn before Progress figures' line 1.
    assert before == ["render", "header", "foot", "line 1"]
    assert "foot" not in after
    # Then filled in, in screen order, each once.
    md = [m.value for m in at.tabs[3].markdown]
    heads = [md.index(f"##### {q}") for q in (wording.Q_TRUST, wording.Q_STAND, wording.Q_NEXT)]
    assert heads == sorted(heads)
    [line_1] = [i for i, m in enumerate(md) if "Best so far: Overall score " in m]
    assert line_1 == md.index(f"##### {wording.Q_STAND}") + 1
    [foot] = [b for b in at.tabs[3].button if b.key == "insights_foot"]
    assert wording.PAIR_FOLD in _texts(at)


@pytest.mark.skip(reason="review cards prototype: pins the old Review layout (status lines, Explored fold, header); reconcile at the final merge")
def test_a_filling_tab_draws_line_1_and_the_foot_before_working_anything_out(
        stuck, five_rounds, monkeypatch):
    """Workers still fitting, answers 2-3's models in: the partial checks
    and the predicted best are worked out here only after the foot."""
    at = _app(five_rounds.project_name)
    at.run()
    _seed(at, five_rounds, "stand")
    for key in (ui_insights.PARTIAL_KEY, ui_insights.BEST_KEY):
        if key in at.session_state:
            del at.session_state[key]
    log = _draw_log(monkeypatch)
    at.session_state["main_tab"] = wording.TAB_INSIGHTS
    at.run()
    assert not at.exception
    before, _ = _drawn_first(log)
    assert before == ["render", "header", "foot", "line 1"]       # ftue R-I8
    texts = " ".join(_texts(at))
    assert f"##### {wording.Q_TRUST}" in texts and f"##### {wording.Q_NEXT}" in texts


@pytest.mark.skip(reason=REVIEW_FOUR)
def test_line_1_is_drawn_again_in_its_place_when_the_checks_change_the_yardstick(
        five_rounds, monkeypatch):
    """Line 1 goes up with the yardstick known before the checks; when the
    checks leave a disagreeing Repeat out of it, line 1 is replaced, never
    repeated."""
    real = insights.noise_yardstick
    sd = insights.Yardstick(2.5, 2.5, 3, True)

    def yardstick(opt, checks=(), name=insights.SCORE):
        return sd if checks and name == insights.SCORE else real(opt, checks, name)
    monkeypatch.setattr(insights, "noise_yardstick", yardstick)
    at = _app(five_rounds.project_name)
    at.run()
    yards = []
    lines = ui_insights._stand_lines
    monkeypatch.setattr(ui_insights, "_stand_lines",
                        lambda opt, yard: yards.append(yard) or lines(opt, yard))
    at.session_state["main_tab"] = wording.TAB_INSIGHTS
    at.run()
    assert not at.exception
    assert yards[0] != sd and yards[-1] == sd
    md = [m.value for m in at.tabs[3].markdown]
    [line_1] = [i for i, m in enumerate(md) if "Best so far: Overall score " in m]
    assert line_1 == md.index(f"##### {wording.Q_STAND}") + 1
