"""The owner's rule for the 0.8.0 review fixes: how suggestions are made does
not change. The golden test (test_complete_results_golden.py) covers a
project with no blanks. This one covers the projects the fixes are about:
formulations scored without a measurement added later, one imported
without a value, and one still waiting for results.

tests/fixtures/ask_unchanged.json is written ONCE, on insights-080 @ ce205c9
(before any review fix), by

    python -m tests.test_ask_unchanged

and committed. It is never regenerated. Every later commit keeps it green:
the same suggestions and the same rows handed to the model — bit-identical
on the Mac; within 0.1% on Windows (tests.golden.same)."""
import json
import os
import pathlib
import sys
import tempfile

import pytest

from food_bo import FoodOptimizer
from tests.golden import same

FIXTURE = pathlib.Path(__file__).parent / "fixtures" / "ask_unchanged.json"


def _truth(pea, oil):
    return {"Firmness": 2 + 0.9 * pea, "Juiciness": 1 + 1.2 * oil,
            "Cook loss": 30 - 3 * pea}


def _recipe(k):
    return {"Pea": 4 + (k * 0.37) % 4, "Oil": 2 + (k * 0.61) % 4}


def _base(name):
    opt = FoodOptimizer(name)
    opt.set_amount_unit("g")
    opt.add_ingredient("Pea", 4, 8)
    opt.add_ingredient("Oil", 2, 6)
    opt.add_objective("Firmness", 50, goal="target", target=6, min_val=0, max_val=10)
    opt.add_objective("Juiciness", 30, goal="max", min_val=0, max_val=10)
    return opt


def _tell(opt, k, drop=()):
    names = {o['name'] for o in opt.objectives}
    r = _recipe(k)
    truth = _truth(r["Pea"], r["Oil"])
    opt.tell(r, {n: v for n, v in truth.items() if n in names and n not in drop},
             formulation_no=k + 1, batch_no=k // 3 + 1)


def exempt_and_waiting(name="ask_unchanged_exempt"):
    """Rows 1-4 recorded before Cook loss was added (scored without it),
    rows 5-9 with it, except row 7, which waits for Cook loss."""
    opt = _base(name)
    for k in range(4):
        _tell(opt, k)
    opt.add_objective("Cook loss", 20, goal="min", min_val=0, max_val=40)
    for k in range(4, 9):
        _tell(opt, k, drop=("Cook loss",) if k == 6 else ())
    return opt


def imported_blank(name="ask_unchanged_import"):
    """Six rows with all three measurements and one imported without
    Cook loss (scored on what it has)."""
    opt = _base(name)
    opt.add_objective("Cook loss", 20, goal="min", min_val=0, max_val=40)
    for k in range(6):
        _tell(opt, k)
    r = _recipe(11)
    truth = _truth(r["Pea"], r["Oil"])
    opt.import_formulation(r, {"Firmness": truth["Firmness"],
                               "Juiciness": truth["Juiciness"]})
    return opt


def _spy(opt):
    seen = []
    real = opt._build_gp

    def spy(train_X_norm, train_Y, dim, input_tf):
        seen.append((train_X_norm.detach().clone(), train_Y.detach().clone()))
        return real(train_X_norm, train_Y, dim, input_tf)

    opt._build_gp = spy
    return seen


def run(opt):
    seen = _spy(opt)
    out = opt.ask(3)
    return {"Y": list(opt.Y_history), "ask": out,
            "gp_X": seen[0][0].tolist(), "gp_Y": seen[0][1].tolist()}


CASES = {"exempt_and_waiting": exempt_and_waiting, "imported_blank": imported_blank}


def fixture():
    return json.loads(FIXTURE.read_text(encoding="utf-8"))


@pytest.fixture(autouse=True)
def _isolate_cwd(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)


@pytest.mark.parametrize("case", sorted(CASES))
def test_suggestions_are_bit_identical(case):
    same(run(CASES[case]()), fixture()[case])


if __name__ == "__main__":
    here = os.getcwd()
    with tempfile.TemporaryDirectory() as tmp:
        os.chdir(tmp)
        data = {case: json.loads(json.dumps(run(make()), default=float))
                for case, make in CASES.items()}
        os.chdir(here)
    FIXTURE.write_text(json.dumps(data, indent=1) + "\n", encoding="utf-8")
    print(f"wrote {FIXTURE}", file=sys.stderr)
