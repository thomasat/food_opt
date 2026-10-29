"""Minor 15 (2026-09-29 review): flag a suggestion on Make a round when it
nearly repeats a formulation still waiting for results — display only, the
algorithm is unchanged.
Spec: docs/superpowers/specs/2026-09-29-review-fixes-design.md §1.3, §9."""
import copy
import io
import itertools
import re
import zipfile

import openpyxl
import pytest

import insights
import wording
from food_bo import FoodOptimizer
from tests.test_food_bo import _BANNED
from ui_batch import _near_copy_prefixes, _waiting_caption


@pytest.fixture(autouse=True)
def _isolate_cwd(tmp_path, monkeypatch):
    # tell()/ask() save "<project_name>.pkl" into the cwd.
    monkeypatch.chdir(tmp_path)


_n = itertools.count()


def _project(rows=6):
    """Pea 4-8 g (span 4), Oil 2-6 g (span 4). Firmness and Cook loss.
    `rows` fully measured formulations, three to a Round, at the same
    recipe (6 g Pea, 4 g Oil) — the search's own kind of suggestion is not
    what this file tests."""
    opt = FoodOptimizer(f"near_copy_{next(_n)}")
    opt.set_amount_unit("g")
    opt.add_ingredient("Pea", 4, 8)
    opt.add_ingredient("Oil", 2, 6)
    opt.add_objective("Firmness", 50, goal="target", target=6,
                      min_val=0, max_val=10)
    opt.add_objective("Cook loss", 50, goal="min", min_val=0, max_val=40)
    for k in range(rows):
        opt.tell({"Pea": 6.0, "Oil": 4.0}, {"Firmness": 6.0, "Cook loss": 10.0},
                 formulation_no=k + 1, batch_no=k // 3 + 1)
    return opt


def _waiting(opt, no, round_no, pea=6.0, oil=4.0, missing=("Cook loss",)):
    """Record formulation `no` in Round `round_no`, dropping `missing`: it
    stays in the history, still waiting for those results."""
    results = {"Firmness": 6.0, "Cook loss": 10.0}
    for name in missing:
        results.pop(name, None)
    opt.tell({"Pea": pea, "Oil": oil}, results, formulation_no=no,
             batch_no=round_no)


def _candidate(opt, pea, oil, batch_no):
    """Open `batch_no` with one suggestion at (pea, oil) and return its
    stored recipe — what near_copy and the batch table both read."""
    opt.set_pending_batch([{"Pea": pea, "Oil": oil}], batch_no=batch_no)
    return opt.pending_batch[0]['recipe']


# ------------------------------------------------------------------ #
#  insights.near_copy: the matching rule itself
# ------------------------------------------------------------------ #

def test_a_match_at_0_04_of_span_flags():
    opt = _project(rows=6)
    _waiting(opt, 7, 3)   # Pea 6.0, Oil 4.0, waiting for Cook loss
    recipe = _candidate(opt, 6.0 + 0.04 * 4, 4.0, batch_no=4)
    match = insights.near_copy(opt, recipe)
    assert match is not None
    assert match.formulation == 7
    assert match.waiting_on == ["Cook loss"]


def test_a_match_at_0_06_of_span_does_not_flag():
    opt = _project(rows=6)
    _waiting(opt, 7, 3)
    recipe = _candidate(opt, 6.0 + 0.06 * 4, 4.0, batch_no=4)
    assert insights.near_copy(opt, recipe) is None


def test_own_round_waiting_rows_are_ignored():
    """A waiting formulation recorded in the very Round being built is not a
    formulation the model has failed to learn from yet — it IS that Round —
    so an exact match to it inside its own Round is not flagged."""
    opt = _project(rows=6)
    _waiting(opt, 7, 3)
    # The open Round is Round 3 itself.
    recipe = _candidate(opt, 6.0, 4.0, batch_no=3)
    assert insights.near_copy(opt, recipe) is None
    # The same amounts, a different open Round, ARE flagged.
    recipe = _candidate(opt, 6.0, 4.0, batch_no=4)
    match = insights.near_copy(opt, recipe)
    assert match is not None and match.formulation == 7


def test_a_tie_in_distance_goes_to_the_lower_formulation_number():
    opt = _project(rows=6)
    _waiting(opt, 7, 3, pea=6.0, oil=4.0)   # 0.05 g away, either side
    _waiting(opt, 8, 3, pea=6.1, oil=4.0)
    recipe = _candidate(opt, 6.05, 4.0, batch_no=4)
    match = insights.near_copy(opt, recipe)
    assert match.formulation == 7


def test_a_categorical_choice_that_differs_is_never_a_match():
    """Same amounts, a different choice: distance() reads that as
    incomparable (None), so it can never be a near-copy match either."""
    opt = FoodOptimizer(f"near_copy_{next(_n)}")
    opt.set_amount_unit("g")
    opt.add_ingredient("Pea", 4, 8)
    opt.add_ingredient("Oil", 2, 6)
    # No public API makes a categorical row any more (old files only); it is
    # appended the way tests/test_insights_wave_b.py's own test does.
    opt.variables.append({'name': "Package", 'type': 'categorical',
                          'options': ["Tray", "Pouch"], 'category': 'process'})
    opt.add_objective("Firmness", 50, goal="target", target=6,
                      min_val=0, max_val=10)
    opt.add_objective("Cook loss", 50, goal="min", min_val=0, max_val=40)
    for k in range(6):
        opt.tell({"Pea": 6.0, "Oil": 4.0, "Package": "Tray"},
                 {"Firmness": 6.0, "Cook loss": 10.0},
                 formulation_no=k + 1, batch_no=k // 3 + 1)
    # Waiting for Cook loss, same amounts, the OTHER choice.
    opt.tell({"Pea": 6.0, "Oil": 4.0, "Package": "Pouch"}, {"Firmness": 6.0},
             formulation_no=7, batch_no=3)
    opt.set_pending_batch([{"Pea": 6.0, "Oil": 4.0, "Package": "Tray"}],
                          batch_no=4)
    recipe = opt.pending_batch[0]['recipe']
    assert insights.near_copy(opt, recipe) is None


def test_a_suggestion_with_no_waiting_formulations_is_never_flagged():
    opt = _project(rows=6)
    recipe = _candidate(opt, 6.0, 4.0, batch_no=2)
    assert insights.near_copy(opt, recipe) is None


# ------------------------------------------------------------------ #
#  ui_batch wiring: own formulations, the prefix text
# ------------------------------------------------------------------ #

def test_the_users_own_formulation_is_never_flagged():
    opt = _project(rows=6)
    _waiting(opt, 7, 3)
    opt.set_pending_batch(
        [{"formulation": 8, "recipe": {"Pea": 6.0, "Oil": 4.0},
          "note": "Repeat of Formulation 7"},
         {"formulation": 9, "recipe": {"Pea": 6.0, "Oil": 4.0}}],
        batch_no=4)
    prefixes = _near_copy_prefixes(opt, opt.pending_batch)
    assert 8 not in prefixes    # the user's own formulation: never flagged
    assert 9 in prefixes        # the suggestion beside it: flagged


def test_the_prefix_names_the_nearest_formulation_and_what_it_waits_for():
    assert wording.near_copy_prefix(20, "Cook loss") == \
        "Nearly the same as Formulation 20, still waiting for Cook loss. "
    assert wording.near_copy_prefix(20, "Cook loss and Juiciness") == \
        ("Nearly the same as Formulation 20, still waiting for Cook loss "
         "and Juiciness. ")


def test_the_prefix_uses_trial_wording_in_a_process_only_study():
    """No ingredients, only process settings dialled in: the table's own
    rows are Trials (wording.for_project), so a near-copy prefix has to say
    'Trial', never 'Formulation', beside it."""
    opt = FoodOptimizer(f"near_copy_{next(_n)}")
    opt.add_process_parameter("Temp", 0, 100, unit="°C")
    opt.add_process_parameter("Time", 0, 60, unit="min")
    opt.add_objective("Firmness", 50, goal="target", target=6,
                      min_val=0, max_val=10)
    opt.add_objective("Cook loss", 50, goal="min", min_val=0, max_val=40)
    for k in range(6):
        opt.tell({"Temp": 50.0, "Time": 30.0},
                 {"Firmness": 6.0, "Cook loss": 10.0},
                 formulation_no=k + 1, batch_no=k // 3 + 1)
    # Trial 7, Round 3: still waiting for Cook loss.
    opt.tell({"Temp": 50.0, "Time": 30.0}, {"Firmness": 6.0},
             formulation_no=7, batch_no=3)
    opt.set_pending_batch([{"Temp": 54.0, "Time": 30.0}], batch_no=4)
    prefixes = _near_copy_prefixes(opt, opt.pending_batch)
    no = opt.pending_batch[0]['formulation']
    assert prefixes[no] == (
        "Nearly the same as Trial 7, still waiting for Cook loss. ")
    assert "Formulation" not in prefixes[no]
    # And in the cell itself, on the table (the column header still reads
    # "Compared with Formulation N" — a pre-existing, separate for_project
    # gap the review notes as out of scope for this fix).
    frame = opt.batch_frame(opt.pending_batch, near_copies=prefixes)
    cell = frame[opt.compared_with_column()].iloc[0]
    assert cell.startswith("Nearly the same as Trial 7, still waiting for "
                          "Cook loss. ")


def test_the_flagged_cell_is_prefixed_on_the_batch_table():
    opt = _project(rows=6)
    _waiting(opt, 7, 3)
    opt.set_pending_batch([{"Pea": 6.0, "Oil": 4.0}], batch_no=4)
    near_copies = _near_copy_prefixes(opt, opt.pending_batch)
    frame = opt.batch_frame(opt.pending_batch, near_copies=near_copies)
    column = opt.compared_with_column()
    cell = frame[column].iloc[0]
    assert cell.startswith(
        "Nearly the same as Formulation 7, still waiting for Cook loss. ")
    # The rest of the cell is exactly what compared_with_text always says.
    assert cell.endswith(opt.compared_with_text({"Pea": 6.0, "Oil": 4.0}))


def test_without_the_flag_the_cell_is_unchanged():
    """`near_copies` defaults to None: every existing caller of batch_frame
    (and the workbook, which never calls it at all) is unaffected."""
    opt = _project(rows=6)
    _waiting(opt, 7, 3)
    opt.set_pending_batch([{"Pea": 6.0, "Oil": 4.0}], batch_no=4)
    frame = opt.batch_frame(opt.pending_batch)
    column = opt.compared_with_column()
    cell = frame[column].iloc[0]
    assert "Nearly the same as" not in cell
    assert cell == opt.compared_with_text({"Pea": 6.0, "Oil": 4.0})


# ------------------------------------------------------------------ #
#  The workbook is never touched
# ------------------------------------------------------------------ #

def _all_cells(data):
    book = openpyxl.load_workbook(io.BytesIO(data))
    for sheet in book.worksheets:
        for row in sheet.iter_rows(values_only=True):
            for cell in row:
                if isinstance(cell, str):
                    yield cell


def test_the_workbook_never_carries_the_near_copy_prefix():
    opt = _project(rows=6)
    _waiting(opt, 7, 3)
    opt.set_pending_batch([{"Pea": 6.0, "Oil": 4.0}], batch_no=4)
    # On screen this row is flagged (see test_the_flagged_cell_is_prefixed_
    # on_the_batch_table); the workbook is built from the same recipe.
    data = opt.workbook_bytes(opt.pending_batch, None)
    assert not any("Nearly the same as" in text for text in _all_cells(data))


def test_workbook_bytes_are_identical_whether_or_not_the_screen_flagged_the_row():
    """Rendering the on-screen flag (reading the waiting formulations and
    building a `near_copies` map, exactly as Make a round does) must never
    change what workbook_bytes writes: the workbook path never receives it,
    so the file is byte for byte the same before and after."""
    opt = _project(rows=6)
    _waiting(opt, 7, 3)
    opt.set_pending_batch([{"Pea": 6.0, "Oil": 4.0}], batch_no=4)
    before = _zip_parts(opt.workbook_bytes(opt.pending_batch, None))
    near_copies = _near_copy_prefixes(opt, opt.pending_batch)
    assert near_copies   # this suggestion really is flagged on screen
    opt.batch_frame(opt.pending_batch, near_copies=near_copies)
    after = _zip_parts(opt.workbook_bytes(opt.pending_batch, None))
    assert before == after


def _zip_parts(data):
    """{name: bytes} of a workbook, ignoring the save times openpyxl stamps
    (zip entry times are not part of the bytes compared; docProps/core.xml's
    created and modified elements are removed)."""
    with zipfile.ZipFile(io.BytesIO(data)) as zf:
        parts = {name: zf.read(name) for name in zf.namelist()}
    parts["docProps/core.xml"] = re.sub(
        rb"<dcterms:(created|modified)\b[^>]*>.*?</dcterms:\1>", b"",
        parts["docProps/core.xml"], flags=re.S)
    return parts


# ------------------------------------------------------------------ #
#  ask() is unaffected: the flag is display only
# ------------------------------------------------------------------ #

def test_ask_is_bit_identical_whether_or_not_a_round_gets_flagged():
    opt = _project(rows=6)
    _waiting(opt, 7, 3)
    opt.ask(2)   # opens a Round; a clone is taken right after
    untouched = copy.deepcopy(opt)
    # Render the flag exactly as Make a round would — read-only, whatever it
    # finds on this particular Round (a guaranteed match is exercised too,
    # as test_a_match_at_0_04_of_span_flags already covers on its own).
    near_copies = _near_copy_prefixes(opt, opt.pending_batch)
    opt.batch_frame(opt.pending_batch, near_copies=near_copies)
    _waiting_caption(opt)
    insights.near_copy(opt, {"Pea": 6.0, "Oil": 4.0})
    assert opt.ask(2) == untouched.ask(2)


def test_near_copy_never_writes_to_the_project():
    opt = _project(rows=6)
    _waiting(opt, 7, 3)
    opt.set_pending_batch([{"Pea": 6.0, "Oil": 4.0}], batch_no=4)
    # `storage` is a LocalStorage handle with no __eq__ of its own (a
    # deep-copied one never compares equal to the original by identity);
    # nothing here is expected to touch it anyway.
    skip = {"storage"}
    before = copy.deepcopy({k: v for k, v in opt.__dict__.items()
                            if k not in skip})
    insights.near_copy(opt, opt.pending_batch[0]['recipe'])
    _near_copy_prefixes(opt, opt.pending_batch)
    after = {k: v for k, v in opt.__dict__.items() if k not in skip}
    assert before.keys() == after.keys()
    for key in before:
        assert before[key] == after[key], key


# ------------------------------------------------------------------ #
#  Wording: the caption, process-only wording, the vocabulary guard
# ------------------------------------------------------------------ #

def test_the_waiting_caption_is_still_one_caption():
    opt = _project(rows=6)
    _waiting(opt, 7, 3)
    line = _waiting_caption(opt)
    assert line.count("\n") == 0
    assert line.endswith("so a suggestion may nearly repeat it.")


def test_new_strings_pass_the_vocabulary_guard():
    strings = [
        wording.near_copy_prefix(20, "Cook loss"),
        wording.near_copy_prefix(20, "Cook loss and Juiciness"),
        wording.waiting_line(1, "Cook loss"),
        wording.waiting_line(2, "Cook loss"),
        wording.waiting_line(2, ""),
    ]
    for text in strings:
        assert not any(p.search(text) for p in _BANNED), text
