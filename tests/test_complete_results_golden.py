"""Golden: what a project with no blanks scores and suggests, captured on the
code BEFORE the complete-results change (0.8.0, insights-080 @ c418337).

tests/fixtures/complete_results_golden.json is written ONCE, on the unchanged
code, by

    python -m tests.test_complete_results_golden

and committed. It is never regenerated: every later commit on the branch
keeps this file green, which is the proof that a project with no blanks and
no measurement added partway scores and suggests exactly what it did."""
import json
import os
import pathlib
import sys
import tempfile

import pytest
import torch

from food_bo import FoodOptimizer

FIXTURE = pathlib.Path(__file__).parent / "fixtures" / "complete_results_golden.json"


def _truth(pea, oil, salt):
    return {"Firmness": 2 + 0.9 * pea,
            "Juiciness": 1 + 1.2 * oil,
            "Cook loss": 30 - 3 * pea + 10 * salt}


def _recipe(k):
    return {"Pea": 4 + (k * 0.37) % 4, "Oil": 2 + (k * 0.61) % 4,
            "Salt": (k * 0.29) % 1}


def _project(name, robust=False, bo_config=None, rows=7):
    """Three ingredients, three measurements (one of each goal), `rows`
    fully measured formulations numbered 1..rows in Rounds of three.

    `name` is derived from the case only (never from call order): the
    project name feeds `_sobol_seed` (it hashes `project_name`), so a
    counter-based name would make the cold-start Sobol sequence, and
    therefore the recorded suggestions, depend on which other cases ran
    first in the same process -- e.g. `pytest -k qlogei` would then see
    different names, and therefore different seeds, than a full run."""
    opt = FoodOptimizer(name, robust=robust)
    opt.set_amount_unit("g")
    opt.add_ingredient("Pea", 4, 8)
    opt.add_ingredient("Oil", 2, 6)
    opt.add_ingredient("Salt", 0, 1)
    opt.add_objective("Firmness", 50, goal="target", target=6,
                      min_val=0, max_val=10)
    opt.add_objective("Juiciness", 30, goal="max", min_val=0, max_val=10)
    opt.add_objective("Cook loss", 20, goal="min", min_val=0, max_val=40)
    if bo_config:
        opt.set_bo_config(bo_config)
    for k in range(rows):
        recipe = _recipe(k)
        opt.tell(recipe, _truth(recipe["Pea"], recipe["Oil"], recipe["Salt"]),
                 formulation_no=k + 1, batch_no=k // 3 + 1)
    return opt


def _spy(opt):
    """Every (train_X_norm, train_Y) pair _ask_optimize hands _build_gp."""
    seen = []
    real = opt._build_gp

    def spy(train_X_norm, train_Y, dim, input_tf):
        seen.append((train_X_norm.detach().clone(), train_Y.detach().clone()))
        return real(train_X_norm, train_Y, dim, input_tf)

    opt._build_gp = spy
    return seen


CASES = {
    "default": dict(),
    "qlogei": dict(bo_config={"acquisition": "qlogei"}),
    "qucb": dict(bo_config={"acquisition": "qucb"}),
    "robust": dict(robust=True),
}


def _run_case(kind):
    """Cold ask on 3 rows, warm ask on 7, and the scores and GP inputs."""
    if kind == "open_round":
        return _run_open_round()
    cold = _project(f"golden_{kind}_3", rows=3, **CASES[kind])
    cold_seen = _spy(cold)
    cold_out = cold.ask(2)
    warm = _project(f"golden_{kind}_7", rows=7, **CASES[kind])
    seen = _spy(warm)
    warm_out = warm.ask(2)
    return {
        "Y_cold": list(cold.Y_history), "Y_warm": list(warm.Y_history),
        "cold": cold_out, "cold_gp_calls": len(cold_seen),
        "warm": warm_out,
        "gp_X": seen[0][0].tolist(), "gp_Y": seen[0][1].tolist(),
    }


def _run_open_round():
    """Round 3 open: two generated rows and one of the user's own; the own
    row and the first generated row recorded one sheet at a time; then the
    next ask."""
    opt = _project("golden_open_round", rows=6)
    first = opt.ask(2)
    own_no = opt.add_to_pending_batch({"Pea": 6.0, "Oil": 4.0, "Salt": 0.5},
                                      note="Own formulation")
    rows = {r['formulation']: r for r in opt.pending_batch}
    gen_no = min(n for n in rows if n != own_no)
    for no in (own_no, gen_no):
        rec = rows[no]['recipe']
        opt.tell(rec, _truth(rec["Pea"], rec["Oil"], rec["Salt"]),
                 formulation_no=no, batch_no=opt.pending_batch_no)
    seen = _spy(opt)
    second = opt.ask(2)
    return {"Y": list(opt.Y_history), "first": first, "second": second,
            "gp_X": seen[0][0].tolist(), "gp_Y": seen[0][1].tolist()}


KINDS = ["default", "qlogei", "qucb", "robust", "open_round"]


def _golden():
    return json.loads(FIXTURE.read_text())


@pytest.fixture(autouse=True)
def _isolate_cwd(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)


@pytest.mark.parametrize("kind", KINDS)
def test_scores_and_suggestions_are_bit_identical(kind):
    want = _golden()[kind]
    got = json.loads(json.dumps(_run_case(kind), default=float))
    for key in want:
        if key in ("gp_X", "gp_Y"):
            assert torch.equal(torch.tensor(got[key], dtype=torch.double),
                               torch.tensor(want[key], dtype=torch.double)), key
        else:
            assert got[key] == want[key], key


def _reference_utility(objectives, results):
    """_compute_utility as it was before 0.8.0, copied verbatim."""
    total_utility = 0.0
    for obj in objectives:
        raw_val = results.get(obj['name'])
        if raw_val is None:
            continue
        val = float(raw_val)
        min_v = obj.get('min_val', 0.0)
        max_v = obj.get('max_val', 10.0)
        rng = max_v - min_v
        if rng == 0:
            rng = 1.0
        norm_val = max(0.0, min(1.0, (val - min_v) / rng))
        if obj['goal'] == 'max':
            utility = norm_val
        elif obj['goal'] == 'min':
            utility = 1.0 - norm_val
        elif obj['goal'] == 'target':
            targ_val = obj.get('target', (max_v + min_v) / 2)
            norm_targ = (targ_val - min_v) / rng
            utility = max(0.0, 1.0 - abs(norm_val - norm_targ))
        else:
            utility = 0.0
        total_utility += obj['weight'] * utility
    return total_utility


def test_scores_equal_the_pre_change_formula():
    opt = _project("golden_formula_check", rows=7)
    for i, results in enumerate(opt.results_history):
        assert opt.Y_history[i] == _reference_utility(opt.objectives, results)


def test_a_reopened_project_scores_the_same():
    opt = _project("golden_reopen_check", rows=7)
    again = FoodOptimizer(opt.project_name)
    assert again.Y_history == opt.Y_history


if __name__ == "__main__":
    here = os.getcwd()
    with tempfile.TemporaryDirectory() as tmp:
        os.chdir(tmp)
        data = {kind: _run_case(kind) for kind in KINDS}
        os.chdir(here)
    FIXTURE.write_text(json.dumps(data, default=float, indent=1) + "\n")
    print(f"wrote {FIXTURE}", file=sys.stderr)
