"""The owner's hard rule, across sessions (review B2; Plan A review B I-3):
Streamlit runs each browser tab's script in a thread of one process, and
Insights still conditions, predicts, searches and (without workers) fits in
that process, each inside insights._isolated(), which forks and reseeds the
global torch RNG. One process-wide lock, insights.RNG_LOCK, is held inside
_isolated() and around the one ask() call site (ui_batch.ask_alone), so a
suggestion made in one tab is exactly the fixture's whatever another tab's
Insights is doing. The fixture is tests/fixtures/ask_unchanged.json, never
regenerated."""
import threading
import time

import pytest
import torch

import insights
import ui_batch
from tests.test_ask_unchanged import CASES, _spy, fixture, same
from tests.test_insights import _burger


@pytest.fixture(autouse=True)
def _isolate_cwd(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)


class _Busy:
    """Another tab's in-process Insights work, looping until stopped: the
    real conditioning and prediction path (insights.fit from a memo, then
    insights.predict), each step inside _isolated()."""

    def __init__(self):
        self.opt = _burger(rounds=4, noise=0.4)
        rows = list(range(len(self.opt.Y_history)))
        self.memo = {}
        with insights.memo_in_use(self.memo):
            insights.fit(self.opt, insights.SCORE, rows)         # fitted once, here
        self.rows = rows
        self.stop = threading.Event()
        self.steps = 0
        self.error = None

    def run(self):
        try:
            while not self.stop.is_set():
                with insights.memo_in_use(self.memo):
                    f = insights.fit(self.opt, insights.SCORE, self.rows)   # conditioning
                insights.predict(self.opt, f, [self.opt.X_history[0]] * 64)
                with insights._isolated():                        # and a draw of its own
                    torch.rand(20000)
                self.steps += 1
                time.sleep(0)
        except Exception as exc:     # surfaced by the test, never swallowed
            self.error = exc

    def __enter__(self):
        self.thread = threading.Thread(target=self.run, daemon=True)
        self.thread.start()
        while self.steps < 2:            # it is really running before ask() starts
            time.sleep(0.01)
        return self

    def __exit__(self, *exc):
        self.stop.set()
        self.thread.join(30)


def _ask(opt):
    """tests.test_ask_unchanged.run, through the app's call site."""
    seen = _spy(opt)
    out = ui_batch.ask_alone(opt, n_suggestions=3)
    return {"Y": list(opt.Y_history), "ask": out,
            "gp_X": seen[0][0].tolist(), "gp_Y": seen[0][1].tolist()}


@pytest.mark.parametrize("case", sorted(CASES))
def test_ask_is_the_fixtures_while_another_tab_works_out_insights_here(case):
    opt = CASES[case]()
    with _Busy() as busy:
        got = _ask(opt)
        assert busy.thread.is_alive()
    assert busy.error is None
    same(got, fixture()[case])


def test_the_lock_is_held_inside_isolated_and_around_ask(monkeypatch):
    held = []
    with insights._isolated():
        held.append(insights.RNG_LOCK._is_owned())
    opt = CASES["imported_blank"]()
    monkeypatch.setattr(opt, "ask", lambda **kw: held.append(insights.RNG_LOCK._is_owned()))
    ui_batch.ask_alone(opt, n_suggestions=1)
    assert held == [True, True]
    assert not insights.RNG_LOCK._is_owned()


def test_generate_asks_through_the_lock():
    """ui_batch._generate's one ask() goes through ask_alone, and no other
    screen calls ask()."""
    import inspect
    import pathlib
    src = inspect.getsource(ui_batch._generate)
    assert "ask_alone(" in src and "opt.ask(" not in src
    root = pathlib.Path(ui_batch.__file__).parent
    callers = [p.name for p in root.glob("*.py") if p.name not in ("food_bo.py", "benchmark.py", "theory.py")
               and ".ask(" in p.read_text().replace("opt.ask(**kwargs)", "")]
    assert callers == []
