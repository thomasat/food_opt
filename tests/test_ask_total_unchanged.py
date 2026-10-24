"""The owner's rule for the 0.8.0 BO-views wave (spec 2026-09-30-bo-views §1,
§8; review B1, B2): the one change to how suggestions are made is that a
recorded formulation more than 1 % off the Formulation total is learned at
the total. Every project with no such row suggests exactly what it did.

test_ask_unchanged.py and the golden test cover projects with no
Formulation total at all, so any rule that does nothing without a total
passes them. This fixture covers the projects that DO have a total, with
rows inside the ±0.5 % band or exactly on it, one row type each, plus one
project with no total whose Round was re-sized (not rescaled in this wave).

tests/fixtures/ask_total_unchanged.json is written ONCE, on insights-080 @
0bf00ba (before any batch-size code), by

    python -m tests.test_ask_total_unchanged

and committed on its own. It is never regenerated: every later commit keeps
it green — the suggestions and the rows handed to the model, bit-identical
on the Mac; within 0.1% on Windows (tests.golden.same)."""
import json
import os
import pathlib
import sys
import tempfile

import pytest

from food_bo import FoodOptimizer
from tests.golden import same

FIXTURE = pathlib.Path(__file__).parent / "fixtures" / "ask_total_unchanged.json"


def _truth(r):
    pea, oil = r.get("Pea", 6.0), r.get("Oil", 4.0)
    temp = r.get("Cook temperature", 75.0)
    return {"Firmness": 2 + 0.9 * pea + 0.01 * (temp - 75),
            "Juiciness": 1 + 1.2 * oil - 0.02 * (temp - 75)}


def _base(name, total=100.0, water="rest", rows=None):
    """Pea 4-8, Oil 2-6, then `rows(opt)` (more rows), then Water (80-94,
    or 0-100 when it fills to the total); two measurements; the
    Formulation total and Water's calculation as asked."""
    opt = FoodOptimizer(name)
    opt.set_amount_unit("g")
    opt.add_ingredient("Pea", 4, 8)
    opt.add_ingredient("Oil", 2, 6)
    if rows is not None:
        rows(opt)
    opt.add_ingredient("Water", *((0, 100) if water == "rest" else (80, 94)))
    opt.add_objective("Firmness", 50, goal="target", target=6, min_val=0, max_val=10)
    opt.add_objective("Juiciness", 30, goal="max", min_val=0, max_val=10)
    if total is not None:
        opt.set_formulation_total(total)
    if water == "rest":
        opt.set_formula("Water", "= rest")
    return opt


def _row(k):
    return {"Pea": 4 + (k * 0.37) % 4, "Oil": 2 + (k * 0.61) % 4}


def _tell(opt, recipe, no, batch_no):
    opt.tell(recipe, _truth(recipe), formulation_no=no, batch_no=batch_no)


def _warm(opt, rows=6, extra=None):
    """`rows` formulations told as Rounds of three, each filled to the
    total by the project's own calculations."""
    for k in range(rows):
        r = _row(k)
        if extra:
            r.update(extra(k))
        _tell(opt, opt.fill_formulas(r), k + 1, k // 3 + 1)
    return opt


def _record_open_round(opt):
    """Tell every row of the open Round, as Save results does."""
    for row in list(opt.pending_batch):
        rec = row["recipe"]
        _tell(opt, rec, row["formulation"], opt.pending_batch_no)


def band():
    """A total, Water not calculated: rows on the total and inside the
    ±0.5 % band, and a warm Round made by ask() and recorded (one of its
    rows moved to the band edge, as a snap the limits refused leaves it)."""
    opt = _base("total_band", water=None)
    for k in range(6):
        r = _row(k)
        r["Water"] = 100 - r["Pea"] - r["Oil"] - (0.45 if k == 2 else 0.0)
        _tell(opt, r, k + 1, k // 3 + 1)
    opt.ask(3)
    rows = list(opt.pending_batch)
    rows[0]["recipe"] = dict(rows[0]["recipe"],
                             Water=rows[0]["recipe"]["Water"] - 0.49)
    _record_open_round(opt)
    return opt


def fill_to_total():
    return _warm(_base("total_fill"))


def formula_constant():
    """A row worked out with a constant term: `= 0.5 + 0.05 × Pea`."""
    opt = _base("total_formula", rows=lambda o: o.add_ingredient("Salt", 0, 2))
    opt.set_formula("Salt", "= 0.5 + 0.05 * Pea")
    return _warm(opt)


def fixed_row():
    opt = _base("total_fixed", rows=lambda o: o.add_ingredient("Salt", 1.5, 1.5))
    return _warm(opt, extra=lambda k: {"Salt": 1.5})


def _spice_mix(opt):
    opt.add_premix("Spice mix", "portioned")
    opt.set_premix_parts("Spice mix", [{"name": "Pepper", "share": 50},
                                       {"name": "Cumin", "share": 50}])
    opt.add_ingredient("Spice mix", 1, 5)


def premix():
    opt = _base("total_premix", rows=_spice_mix)
    return _warm(opt, extra=lambda k: {"Spice mix": 1 + (k * 0.53) % 4})


def process_setting():
    opt = _base("total_process", rows=lambda o: o.add_process_parameter(
        "Cook temperature", 60, 90, unit="°C"))
    return _warm(opt, extra=lambda k: {"Cook temperature": 60 + (k * 7.3) % 30})


def own_at_total():
    """A warm Round with one formulation of the user's own at the total,
    recorded with the Round (its batch size is the project's default)."""
    opt = _warm(_base("total_own"))
    opt.ask(2)
    opt.add_to_pending_batch({"Pea": 6.0, "Oil": 4.0}, note="Own formulation")
    _record_open_round(opt)
    return opt


def past_at_total():
    """"Add a past formulation" at the total."""
    opt = _warm(_base("total_past"))
    opt.import_formulation(opt.fill_formulas({"Pea": 5.5, "Oil": 3.5}),
                           _truth({"Pea": 5.5, "Oil": 3.5}))
    return opt


def round_at_default():
    """A warm Round made at the project's default batch size: its stored
    size is the total."""
    opt = _warm(_base("total_default"))
    opt.ask(3)
    _record_open_round(opt)
    return opt


def no_total_resized():
    """No Formulation total: a Round re-sized to 500 g and recorded at 500 g.
    Not rescaled in this wave (review B1): unchanged."""
    opt = _base("no_total_resized", total=None, water=None)
    for k in range(6):
        r = _row(k)
        r["Water"] = 90.0
        _tell(opt, r, k + 1, k // 3 + 1)
    opt.ask(3)
    opt.scale_round(500)
    _record_open_round(opt)
    return opt


CASES = {f.__name__: f for f in (band, fill_to_total, formula_constant, fixed_row, premix,
                                 process_setting, own_at_total, past_at_total,
                                 round_at_default, no_total_resized)}


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
