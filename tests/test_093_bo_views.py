"""0.8.0 BO views, Task 3 (spec 2026-09-30-bo-views §3.4, §8; review B4,
I5, I9, I10, I11, I12, M5, M10): the suggestion model's acquisition rebuilt
on the display side. Exact — diff 0 against the object ask() built — for a
Round whose generation record still matches; recomputed otherwise; never
moving a suggestion; the same numbers from a worker."""
import json
import subprocess
import sys

import pytest
import torch

import bo_views
import insights
import insights_jobs
import wording
from food_bo import FoodOptimizer
from tests import demo_projects as demo


@pytest.fixture(autouse=True)
def _isolate(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    with bo_views.fresh():
        yield


def _truth(r):
    return {"Firmness": 2 + 0.9 * r["Pea"] - 0.2 * r["Oil"],
            "Juiciness": 1 + 1.2 * r["Oil"] - 0.05 * r["Pea"] ** 2 / 4}


def _project(name, bo_config=None, robust=False, rows=7):
    """Pea 4-8, Oil 2-6, Water filling to 100 g; `rows` formulations told."""
    opt = FoodOptimizer(name, robust=robust)
    opt.set_amount_unit("g")
    opt.add_ingredient("Pea", 4, 8)
    opt.add_ingredient("Oil", 2, 6)
    opt.add_ingredient("Water", 0, 100)
    opt.add_objective("Firmness", 50, goal="target", target=6, min_val=0, max_val=10)
    opt.add_objective("Juiciness", 30, goal="max", min_val=0, max_val=10)
    opt.set_formulation_total(100)
    opt.set_formula("Water", "= rest")
    if bo_config:
        opt.set_bo_config(bo_config)
    for k in range(rows):
        r = opt.fill_formulas({"Pea": 4 + (k * 0.37) % 4, "Oil": 2 + (k * 0.61) % 4})
        opt.tell(r, _truth(r), formulation_no=k + 1, batch_no=k // 3 + 1)
    return opt


def _grid(n=64):
    g = torch.Generator().manual_seed(5)
    return torch.rand(n, 2, generator=g, dtype=torch.double)


def _ask_and_capture(opt, n, P):
    """ask(n) as Generate calls it, the generation recorded first (the call
    site's job, review B4), and ask()'s own acquisition evaluated at P the
    moment it is built — before optimize_acqf changes it (02 M5)."""
    record = bo_views.generation_record(opt, n)
    seen = {}
    real = opt._build_acqf

    def spy(gp, X, Y):
        acqf = real(gp, X, Y)
        with torch.no_grad():
            seen["value"] = torch.cat([acqf(P[k:k + bo_views.CHUNK].unsqueeze(1)).reshape(-1)
                                       for k in range(0, P.shape[0], bo_views.CHUNK)])
        return acqf
    opt._build_acqf = spy
    with insights.RNG_LOCK:
        out = opt.ask(n)
    del opt._build_acqf
    record["picks"] = [int(row["formulation"]) for row in opt.pending_batch]
    opt.round_generation = {opt.pending_batch_no: record}
    return out, seen["value"]


@pytest.mark.parametrize("config,robust", [(None, False), ({"acquisition": "qlogei"}, False),
                                           ({"acquisition": "qucb"}, False), (None, True)])
def test_the_rebuilt_acquisition_is_the_one_ask_built(config, robust):
    """Review B4: diff 0, in-process, for a warm open Round whose record
    matches — qLogNEI, qLogEI, qUCB, and a robust project (Warp)."""
    opt = _project(f"exact_{robust}_{(config or {}).get('acquisition')}", config, robust)
    P = _grid()
    _, theirs = _ask_and_capture(opt, 3, P)
    gen = bo_views.generation(opt, opt.pending_batch_no)
    assert gen.exact and gen.regime == "warm" and len(gen.rows) == 7
    model = bo_views.rebuild(opt, gen)
    mean, sd, value = bo_views.evaluate_here(model, P)
    raw = theirs if bo_views.acquisition(opt) == "qucb" else theirs.exp()
    assert torch.equal(value, raw)


def test_rebuilding_never_moves_a_suggestion():
    """The global RNG is where it was, and the next Generate (here a
    regenerate of the open Round) gives what it would have given without
    the views."""
    a, b = _project("moves_a"), _project("moves_b")
    a.round_generation = {1 + a.next_batch_no() - 1: bo_views.generation_record(a, 2)}
    a.ask(2)
    b.ask(2)
    state = torch.get_rng_state()
    w = bo_views.why(a)
    assert w is not None and not w.cold and len(w.rows) == 2
    bo_views.round_history(a)
    # The real check: the global RNG state. ask() reseeds torch itself, so
    # the second assertion holds even if the views drew from it (T3 M-9);
    # the ask/golden fixture tests are the guard for ask() itself.
    assert torch.equal(state, torch.get_rng_state())
    assert a.ask(2, batch_no=a.pending_batch_no) == b.ask(2, batch_no=b.pending_batch_no)


def test_a_record_that_no_longer_matches_is_recomputed():
    """Review B4: a Share of score changed after Generate changes the rows'
    scores; the Round is then recomputed from today's records."""
    opt = _project("stale")
    _ask_and_capture(opt, 2, _grid(8))
    assert bo_views.generation(opt, opt.pending_batch_no).exact
    opt.update_objective("Firmness", weight=70)
    gen = bo_views.generation(opt, opt.pending_batch_no)
    assert not gen.exact and gen.seed == opt.round_generation[opt.pending_batch_no]["seed"]


def test_without_a_record_the_round_is_taken_from_todays_records():
    """02 §3: the seed is the Round's lowest Formulation number and the rows
    the fully measured ones before it (the demo projects have no record)."""
    opt = demo.load(demo.LINE_A)
    gen = bo_views.generation(opt, 6)
    assert (gen.seed, gen.exact, gen.regime, len(gen.rows)) == (22, False, "warm", 20)


def test_a_cold_round_has_no_acquisition():
    """Review I12: chosen to cover the allowed amounts; one line instead."""
    opt = _project("cold", rows=2)
    opt.ask(3)
    w = bo_views.why(opt)
    assert w.cold and w.rows == []
    assert bo_views.round_gain(opt, opt.pending_batch_no).why_none == "cold"


def test_a_worker_answers_what_this_process_works_out(tmp_path):
    """Review B4: a worker (one torch thread) agrees to 1e-9."""
    opt = _project("worker")
    _ask_and_capture(opt, 2, _grid(8))
    gen = bo_views.generation(opt, opt.pending_batch_no)
    model = bo_views.rebuild(opt, gen)
    P = _grid(32)
    request = {"id": 1, "kind": "acq", "cfg": None, "X": model.X.tolist(),
               "Y": model.Y.tolist(), "robust": False, "seed": gen.seed,
               "points": P.tolist(), "pending": None}
    proc = subprocess.run([sys.executable, insights_jobs.WORKER], cwd=tmp_path,
                          input=json.dumps(request) + "\n", capture_output=True, text=True,
                          timeout=180)
    hello, answer = [json.loads(line) for line in proc.stdout.splitlines()]
    assert hello == {"ready": True} and answer["id"] == 1
    mean, sd, value = bo_views.evaluate_here(model, P)
    for key, here in (("mean", mean), ("sd", sd), ("value", value)):
        assert torch.allclose(torch.tensor(answer[key], dtype=torch.double), here,
                              rtol=0, atol=1e-9), key


def test_without_workers_nothing_is_sent():
    assert not insights_jobs.workers_on()
    assert insights_jobs.ask_workers({"kind": "ping"}) is None


def test_outputs_are_cached_by_what_they_depend_on(monkeypatch):
    """Review B4, I11: outputs, not models; a second ask is not worked out
    again, and a different question is."""
    opt = _project("cache")
    _ask_and_capture(opt, 2, _grid(8))
    model = bo_views.rebuild(opt, bo_views.generation(opt, opt.pending_batch_no))
    calls = []
    real = bo_views.evaluate_here
    monkeypatch.setattr(bo_views, "evaluate_here", lambda *a, **k: calls.append(1) or real(*a, **k))
    P = _grid(16)
    bo_views.evaluate(opt, model, P)
    bo_views.evaluate(opt, model, P)
    assert len(calls) == 1
    bo_views.evaluate(opt, model, P[:8])
    assert len(calls) == 2


# ------------------------------------------------------------------ #
#  The pool of allowed formulations (review I9)
# ------------------------------------------------------------------ #

def _no_fill():
    """A total held as a band (Water varies): the pool holds it as an
    equality."""
    opt = FoodOptimizer("band")
    opt.set_amount_unit("g")
    opt.add_ingredient("Pea", 4, 8)
    opt.add_ingredient("Oil", 2, 6)
    opt.add_ingredient("Salt", 1.5, 1.5)
    opt.add_ingredient("Water", 80, 94)
    opt.add_objective("Firmness", 1, goal="max", min_val=0, max_val=10)
    opt.set_formulation_total(100)
    opt.add_quantity_constraint(["Pea", "Oil"], max_val=11)
    return opt


def test_the_pool_is_every_limit_kept_and_the_total_exact():
    opt = _no_fill()
    P = bo_views.feasible_pool(opt)
    assert P.shape == (bo_views.POOL_SIZE, 4)
    bounds = opt._get_bounds()
    for k in range(0, bo_views.POOL_SIZE, 97):
        r = opt._decode(insights.unnormalize(P[k], bounds).numpy())
        assert opt._check_constraints(r), r
        assert opt.ingredient_total(r) == pytest.approx(100.0, abs=1e-6)
        assert r["Salt"] == pytest.approx(1.5)
        assert r["Pea"] + r["Oil"] <= 11 + 1e-6


def test_the_pool_is_the_same_every_time():
    """A fixed seed (review M5): a role never flips between two opens."""
    a = bo_views.feasible_pool(_no_fill())
    with bo_views.fresh():
        b = bo_views.feasible_pool(_no_fill())
    assert torch.equal(a, b)


def test_the_role_is_the_compared_with_cells_word():
    """0.8.0 e2e fixes A2 (review I3): the Why fold's role is the Compared
    with cell's own word, by suggestion_role's distance rule, never the
    pool's percentiles; a row with a note is the user's own, as its cell
    says."""
    opt = demo.load(demo.LINE_A)
    for row in opt.pending_batch:
        assert bo_views.role(opt, row) == opt.suggestion_role(row["recipe"])
        assert opt.compared_with_text(row["recipe"]).lower().startswith(
            bo_views.role(opt, row))
    # Persona fixes: a row with a note says the note as its source.
    mine = dict(opt.pending_batch[0], note="Repeat of Formulation 15")
    assert bo_views.role(opt, mine) == "Repeat of Formulation 15"
    mine = dict(opt.pending_batch[0], note=wording.OWN_FORMULATION_NOTE)
    assert bo_views.role(opt, mine) == wording.OWN_FORMULATION_KIND


# ------------------------------------------------------------------ #
#  §3.1 and §3.2 on the demo projects (02 §4.3, §4.4)
# ------------------------------------------------------------------ #

def test_why_these_suggestions_on_line_a():
    """02 §4.3's table: F22 refines the best with the largest expected
    improvement; F24-F25 are balanced; picks 2-4 are valued with the
    earlier picks counted."""
    w = bo_views.why(demo.load(demo.LINE_A))
    assert not w.exact and not w.cold and w.kind == "qlognei"
    got = {r.formulation: r for r in w.rows}
    assert [got[n].pick for n in (22, 23, 24, 25)] == [1, 2, 3, 4]
    assert [got[n].pending for n in (22, 23, 24, 25)] == [False, True, True, True]
    assert got[22].value == pytest.approx(0.117, abs=0.01)
    # T3 I-4: picks 2-4 with the earlier picks as X_pending (02's 0.058,
    # 0.048, 0.030), not their own q = 1 values (0.081, 0.058, 0.051).
    assert [got[n].value for n in (23, 24, 25)] == pytest.approx([0.058, 0.049, 0.034],
                                                                  abs=0.01)
    assert got[23].value < 0.07
    # 0.8.0 e2e fixes A2: by distance. 22 and 23 are more than 0.15 from
    # Best so far (F15) but near tested ones; 24 and 25 move Mixing time
    # after fat by 105 s, more than 0.30 of its span from everything made.
    assert [got[n].role for n in (22, 23, 24, 25)] == [
        wording.ROLE_BALANCED, wording.ROLE_BALANCED, wording.ROLE_EXPLORES,
        wording.ROLE_EXPLORES]
    assert got[22].nearest == 18
    name, diff, unit = got[22].differences[0]
    assert unit == "g" and name in {v["name"] for v in demo.load(demo.LINE_A).variables}
    assert w.pool_mean.shape == (bo_views.POOL_SIZE,)


def test_the_expected_improvement_by_round_on_line_a():
    """02 §4.4's Line A row: 0.87, 1.10, 0.79, 0.26, 0.12 — recomputed from
    today's records (no generation records on the demo)."""
    h = bo_views.round_history(demo.load(demo.LINE_A))
    assert [g.round for g in h] == [2, 3, 4, 5, 6]
    assert [g.gain for g in h] == pytest.approx([0.87, 1.10, 0.79, 0.26, 0.12], abs=0.02)
    assert [g.best_measured for g in h] == pytest.approx([87.8, 88.4, 90.1, 90.5, 90.5], abs=0.1)
    assert all(not g.exact and g.highest is not None for g in h)


def test_the_latest_gain_is_the_open_rounds_else_the_last_made():
    opt = demo.load(demo.LINE_A)
    got = bo_views.latest_gain(opt)
    assert (got.round, got.highest) == (6, None)                # no pool for the line
    assert got.gain == pytest.approx(0.12, abs=0.02)
    opt.set_pending_batch(None)
    assert bo_views.latest_gain(opt).round == 5


def test_a_round_of_the_users_own_formulations_has_no_bar():
    opt = _project("own")
    opt.add_to_pending_batch({"Pea": 6.0, "Oil": 4.0}, note="Own formulation")
    assert bo_views.round_gain(opt, opt.pending_batch_no).why_none == "own"


def test_the_why_fold_never_starts_or_waits_for_the_workers(monkeypatch):
    """T3 I-3: the exact model is fitted once here and every question is
    answered with it; with the workers on, nothing is sent and the pool is
    never started."""
    opt = _project("pool")
    _ask_and_capture(opt, 3, _grid(8))
    monkeypatch.setattr(insights_jobs, "workers_on", lambda: True)
    started = []
    monkeypatch.setattr(insights_jobs, "pool", lambda: started.append(1))
    monkeypatch.setattr(insights_jobs, "ask_workers", lambda *a, **k: started.append(2))
    fits = []
    real = bo_views.fit_default
    monkeypatch.setattr(bo_views, "fit_default", lambda *a, **k: fits.append(1) or real(*a, **k))
    w = bo_views.why(opt)
    assert w.exact and len(w.rows) == 3
    assert started == [] and fits == [1]


def test_a_round_recorded_one_sheet_at_a_time_stays_exact():
    """Review Focus (plan): recording some of the open Round's rows adds
    rows the record does not name; its own rows still match, so the Round
    is still rebuilt exactly and every row is still explained."""
    opt = _project("sheets")
    _ask_and_capture(opt, 3, _grid(8))
    first = opt.pending_batch[0]
    opt.tell(first["recipe"], _truth(first["recipe"]), formulation_no=first["formulation"],
             batch_no=opt.pending_batch_no)
    gen = bo_views.generation(opt, opt.pending_batch_no)
    assert gen.exact and len(gen.rows) == 7
    w = bo_views.why(opt)
    assert w.exact and [r.pick for r in w.rows] == [1, 2, 3]


# ------------------------------------------------------------------ #
#  Fix round 1 (T3 review I-1, I-2, M-1, M-4, M-5, M-6, M-7)
# ------------------------------------------------------------------ #

def test_qlogei_picks_after_the_first_show_what_they_add():
    """T3 I-1: qLogEI puts X_pending into the q-batch, so its value with
    the earlier picks is the SET's; a pick's own is the increment
    value({earlier, x}) − value({earlier})."""
    opt = _project("qlogei_picks", {"acquisition": "qlogei"}, rows=9)
    _ask_and_capture(opt, 3, _grid(8))
    w = bo_views.why(opt)
    model = bo_views.rebuild(opt, bo_views.generation(opt, opt.pending_batch_no))
    recipes = [insights.at_total(opt, r["recipe"]) for r in opt.pending_batch]
    P = bo_views.points(opt, recipes)
    assert [r.pending for r in w.rows] == [False, True, True]
    for k in (1, 2):
        # botorch's own joint value of {x} ∪ X_pending, minus {earlier}'s.
        joint = float(bo_views.evaluate_here(model, P[k:k + 1], P[:k])[2][0])
        before = bo_views.joint_value(model, P[:k])
        assert w.rows[k].value == pytest.approx(max(0.0, joint - before), abs=1e-9)
        assert w.rows[k].value < joint - 1e-6      # never the set's own value


def test_qucb_rows_show_their_own_optimistic_score():
    """T3 I-1: qUCB has no conditional form; every row its q = 1 value,
    no "·"."""
    opt = _project("qucb_picks", {"acquisition": "qucb"}, rows=9)
    _ask_and_capture(opt, 3, _grid(8))
    w = bo_views.why(opt)
    model = bo_views.rebuild(opt, bo_views.generation(opt, opt.pending_batch_no))
    P = bo_views.points(opt, [insights.at_total(opt, r["recipe"]) for r in opt.pending_batch])
    mean, sd, value = bo_views.evaluate_here(model, P)
    assert [r.pending for r in w.rows] == [False, False, False]
    assert [r.value for r in w.rows] == pytest.approx(value.tolist(), abs=1e-12)
    assert [r.value for r in w.rows] == pytest.approx(
        (mean + 0.2 ** 0.5 * sd).tolist(), abs=0.05)


def test_generate_records_the_numbers_ask_issued():
    """T3 I-2: the Generate call site keeps the picks, in order, and they
    survive a save."""
    import ui_batch
    opt = _project("picks_recorded")
    ui_batch.ask_alone(opt, n_suggestions=3)
    record = opt.round_generation[opt.pending_batch_no]
    assert record["picks"] == [int(r["formulation"]) for r in opt.pending_batch]
    again = FoodOptimizer("picks_recorded")
    assert again.round_generation[opt.pending_batch_no]["picks"] == record["picks"]


def test_a_note_on_a_pick_never_makes_it_the_users_own():
    """T3 I-2: a Note typed on pick 1 at Save results. Once the Round is
    closed, its first pick is still pick 1 (not pick 2 at q = 1)."""
    opt = _project("noted", rows=9)
    _ask_and_capture(opt, 3, _grid(8))
    no = opt.pending_batch_no
    want = bo_views.round_gain(opt, no).gain
    second = next(r for r in bo_views.why(opt).rows if r.pick == 2)
    model = bo_views.rebuild(opt, bo_views.generation(opt, no))
    alone = bo_views.evaluate(opt, model, bo_views.points(opt, [insights.at_total(
        opt, opt.pending_batch[1]["recipe"])])).value[0]
    assert abs(float(alone) - want) > 1e-2 and second.formulation == opt.pending_batch[1][
        "formulation"]
    for k, row in enumerate(list(opt.pending_batch)):
        opt.tell(row["recipe"], _truth(row["recipe"]), formulation_no=row["formulation"],
                 batch_no=no, note="slightly dry" if k == 0 else "")
    opt.set_pending_batch(None)
    got = bo_views.round_gain(opt, no)
    # As recorded (amounts rounded at tell), still pick 1's value.
    assert got.why_none is None and got.gain == pytest.approx(want, abs=1e-3)


def test_an_edited_own_formulation_stays_the_users_own():
    """T3 I-2: editing an own row gives it original_recipe; it is still the
    user's own, not a suggestion "·", and never an earlier pick."""
    opt = _project("own_edited", rows=9)
    _ask_and_capture(opt, 2, _grid(8))
    own_no = opt.add_to_pending_batch({"Pea": 6.0, "Oil": 4.0}, note="Own formulation")
    opt.edit_pending_formulations({own_no: {"Pea": 6.5, "Oil": 4.0}})
    assert next(r for r in opt.pending_batch if r["formulation"] == own_no).get(
        "original_recipe")
    w = bo_views.why(opt)
    got = {r.formulation: r for r in w.rows}
    assert got[own_no].own and got[own_no].pick is None and not got[own_no].pending
    assert not got[own_no].edited
    assert [got[n].pick for n in opt.round_generation[opt.pending_batch_no]["picks"]] == [1, 2]


def test_a_pick_moved_to_not_scored_keeps_the_others_places():
    """T3 M-5: pick 1 moved to Not scored; the Round's first pick is still
    pick 1 (from the record), and in the open Round pick 2 is still pick 2,
    valued with pick 1 counted."""
    opt = _project("skipped_pick", rows=9)
    _ask_and_capture(opt, 3, _grid(8))
    no = opt.pending_batch_no
    want = bo_views.round_gain(opt, no).gain
    first, *rest = list(opt.pending_batch)
    opt.record_skipped(first["formulation"], no, first["recipe"])
    w = bo_views.why(opt)
    got = {r.formulation: r for r in w.rows}
    assert [got[r["formulation"]].pick for r in rest] == [2, 3]
    assert all(got[r["formulation"]].pending for r in rest)
    for row in rest:
        opt.tell(row["recipe"], _truth(row["recipe"]), formulation_no=row["formulation"],
                 batch_no=no)
    opt.set_pending_batch(None)
    assert bo_views.round_gain(opt, no).gain == pytest.approx(want, abs=1e-3)


def test_a_degenerate_pool_says_no_roles():
    """T3 M-1: hit-and-run returns POOL_SIZE rows or fails; what counts is
    how many distinct formulations it holds."""
    P = torch.rand(10, 3, dtype=torch.double).repeat(300, 1)
    assert P.shape[0] == bo_views.POOL_SIZE
    assert bo_views.pool_points(P) == 10 < bo_views.POOL_MIN
    assert bo_views.pool_points(None) == 0


def test_a_degenerate_pool_leaves_the_roles_alone(monkeypatch):
    """T3 M-1 said no roles from a pool of a few formulations; 0.8.0 e2e
    fixes A2 (review I3): the role is a distance, so it never reads the
    pool and is said all the same."""
    opt = _project("degenerate", rows=9)
    _ask_and_capture(opt, 2, _grid(8))
    one = bo_views.points(opt, [insights.at_total(opt, opt.pending_batch[0]["recipe"])])
    monkeypatch.setattr(bo_views, "feasible_pool", lambda o: one.repeat(bo_views.POOL_SIZE, 1))
    assert [r.role for r in bo_views.why(opt).rows] == [
        opt.suggestion_role(r["recipe"]) for r in opt.pending_batch]


def test_the_pool_key_follows_the_open_round_and_its_not_scored_rows():
    """T3 M-4: they decide which pre-mix make-up the pool is sampled with."""
    opt = _project("pool_key")
    before = bo_views.pool_key(opt)
    opt.ask(2)
    opened = bo_views.pool_key(opt)
    assert opened != before
    first = opt.pending_batch[0]
    opt.record_skipped(first["formulation"], opt.pending_batch_no, first["recipe"])
    assert bo_views.pool_key(opt) != opened


def test_the_pool_stays_cached_while_it_is_used():
    """T3 M-3: a hit moves the pool to the newest end of the cache."""
    opt = _no_fill()
    P = bo_views.feasible_pool(opt)
    for k in range(bo_views._CACHE_SIZE - 1):
        bo_views._remember(("x", k), k)
        assert bo_views.feasible_pool(opt) is P
    bo_views._remember(("x", "last"), 0)
    assert ("pool", bo_views.pool_key(opt)) in bo_views._CACHE


def test_a_display_model_of_other_rows_is_not_used():
    """T3 M-6: a recomputed Round never takes a model fitted on another
    number of rows."""
    opt = _project("wrong_rows", rows=9)
    gen = bo_views.generation(opt, 3)
    wrong = insights.fit(opt, insights.SCORE, list(range(9)))
    assert wrong.n_rows != len(gen.rows)
    model = bo_views.rebuild(opt, gen, wrong)
    assert model.gp is not wrong.model and model.X.shape[0] == len(gen.rows)



def test_the_next_rounds_figure_is_the_pick_generate_makes():
    """Final review I-1, coordinator rulings: with no Round open, the figure
    is ask()'s own first-pick search (q = 1 optimize_acqf, ask()'s bounds,
    limits and settings, from ask()'s RNG state) on the model the next
    Generate fits. On the Timing check it is at least — in fact exactly —
    Round 13's pick-1 figure after Generate: sequential=True makes ask()'s
    first pick this very search. The global torch, numpy and random states
    do not move."""
    import random
    import numpy as np
    import ui_batch
    opt = demo.load(demo.TIMING)
    states = (torch.get_rng_state(), np.random.get_state()[1].copy(), random.getstate())
    before = bo_views.next_model(opt)
    assert before.exact and before.gen.seed == opt.next_formulation_no
    ahead = bo_views.next_gain(opt, before)
    assert torch.equal(states[0], torch.get_rng_state())
    assert (states[1] == np.random.get_state()[1]).all() and states[2] == random.getstate()
    ui_batch.ask_alone(opt, n_suggestions=3)
    after = bo_views.latest_gain(opt)
    assert after.exact and after.round == opt.pending_batch_no
    assert ahead.gain >= after.gain - 1e-6
    assert ahead.gain == pytest.approx(after.gain, abs=1e-6) == pytest.approx(1.08, abs=0.02)
    # The same model: pick 1 valued on the model built before Generate.
    pick1 = insights.at_total(opt, opt.pending_batch[0]["recipe"])
    on_before = float(bo_views.evaluate(opt, before, bo_views.points(opt, [pick1])).value[0])
    assert on_before == pytest.approx(after.gain, abs=1e-6)


def test_the_search_uses_asks_own_settings():
    import inspect
    src = inspect.getsource(FoodOptimizer._ask_optimize)
    assert f"num_restarts={bo_views.ASK_RESTARTS}," in src
    assert f"raw_samples={bo_views.ASK_RAW_SAMPLES}," in src and "sequential=True" in src


def test_a_failed_search_falls_back_to_the_pool(monkeypatch):
    opt = _project("fallback", rows=9)
    model = bo_views.next_model(opt)
    monkeypatch.setattr(bo_views, "next_optimum", lambda *a: 1 / 0)
    got = bo_views.next_gain(opt, model)
    want = float(bo_views.evaluate(opt, model, bo_views.feasible_pool(opt)).value.max())
    assert got.round is None and got.gain == pytest.approx(want)
