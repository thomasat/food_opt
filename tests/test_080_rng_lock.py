"""The owner's hard rule, across sessions (review B2; Plan A review B I-3):
Streamlit runs each browser tab's script in a thread of one process, and
torch has one global RNG. One process-wide lock, insights.RNG_LOCK, is held
around the one ask() call site (ui_batch.ask_alone), so a suggestion made in
one tab is exactly the fixture's whatever another tab holding the lock is
doing with the RNG. The fixture is tests/fixtures/ask_unchanged.json, never
regenerated."""
import threading
import time

import pytest
import torch

import insights
import ui_batch
from tests.test_ask_unchanged import CASES, _spy, fixture, same


@pytest.fixture(autouse=True)
def _isolate_cwd(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)


class _Busy:
    """Another tab's work with torch's global RNG, looping until stopped,
    each step holding insights.RNG_LOCK as every in-process RNG user must
    (another tab's ask_alone holds it the same way)."""

    def __init__(self):
        self.stop = threading.Event()
        self.steps = 0
        self.error = None

    def run(self):
        try:
            while not self.stop.is_set():
                with insights.RNG_LOCK:                           # a draw of its own
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
def test_ask_is_the_fixtures_while_another_tab_uses_the_rng(case):
    opt = CASES[case]()
    with _Busy() as busy:
        got = _ask(opt)
        assert busy.thread.is_alive()
    assert busy.error is None
    same(got, fixture()[case])


def test_the_lock_is_held_around_ask(monkeypatch):
    held = []
    opt = CASES["imported_blank"]()
    monkeypatch.setattr(opt, "ask", lambda **kw: held.append(insights.RNG_LOCK._is_owned()))
    ui_batch.ask_alone(opt, n_suggestions=1)
    assert held == [True]
    assert not insights.RNG_LOCK._is_owned()


def test_generate_asks_through_the_lock():
    """ui_batch._generate's one ask() goes through ask_alone, and there is
    no other ask() in the app."""
    import inspect
    import pathlib
    src = inspect.getsource(ui_batch._generate)
    assert "ask_alone(" in src and "opt.ask(" not in src
    import ast
    root = pathlib.Path(ui_batch.__file__).parent
    # Real ask() calls only (an AST walk, so a docstring that names ask()
    # is not a caller), as (file, enclosing function).
    calls = []
    for path in sorted(root.glob("*.py")):
        if path.name in ("food_bo.py", "benchmark.py", "theory.py"):
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"))
        asks = {id(n) for n in ast.walk(tree) if isinstance(n, ast.Call)
                and isinstance(n.func, ast.Attribute) and n.func.attr == "ask"}
        placed = set()
        for fn in ast.walk(tree):
            if isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
                for node in ast.walk(fn):
                    if id(node) in asks:
                        calls.append((path.name, fn.name))
                        placed.add(id(node))
        calls += [(path.name, "<module>")] * len(asks - placed)
    assert sorted(set(calls)) == [("ui_batch.py", "ask_alone")]
