"""Removing a process setting that took more than one value across the
recorded formulations is refused unless forced (stable 940a0ce, A10)."""
import pytest

import wording
from food_bo import FoodOptimizer


@pytest.fixture
def opt(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    o = FoodOptimizer(project_name="guard", robust=False)
    o.add_ingredient("A", 0, 10)
    o.add_process_parameter("T", 100, 250, baseline=150)
    o.add_objective("Taste", 1.0, goal="max")
    return o


def _names(o):
    return [v['name'] for v in o.variables]


def _varied(o):
    o.tell({"A": 1.0, "T": 150.0}, {"Taste": 5.0})
    o.tell({"A": 1.0, "T": 190.0}, {"Taste": 6.0})


def test_varied_setting_refused(opt):
    _varied(opt)
    before = _names(opt)
    with pytest.raises(ValueError) as caught:
        opt.remove_process_parameter("T")
    assert str(caught.value) == wording.setting_took_several_values("T", 2)
    assert _names(opt) == before


def test_varied_setting_removed_when_forced(opt):
    _varied(opt)
    opt.remove_process_parameter("T", force=True)
    assert _names(opt) == ["A"]


def test_constant_setting_removes_without_force(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    o = FoodOptimizer(project_name="guard2", robust=False)
    o.add_ingredient("A", 0, 10)
    o.add_objective("Taste", 1.0, goal="max")
    o.tell({"A": 2.0}, {"Taste": 6.0})
    # Added after results: earlier rows read its baseline, 150.
    o.add_process_parameter("T", 100, 250, baseline=150)
    o.tell({"A": 1.0, "T": 150.0}, {"Taste": 5.0})
    o.remove_process_parameter("T")
    assert _names(o) == ["A"]


def test_no_results_removes_without_force(opt):
    opt.remove_process_parameter("T")
    assert _names(opt) == ["A"]


def test_grid_delete_of_varied_setting_refused(opt):
    _varied(opt)
    frame = opt.ingredient_grid_frame()
    row = frame.index[frame[wording.NAME_LABEL] == "T"][0]
    errors, _ = opt.apply_ingredient_grid(frame.drop(index=row))
    assert errors == [(None, wording.setting_took_several_values("T", 2))]
    assert "T" in _names(opt)
    errors, _ = opt.apply_ingredient_grid(frame.drop(index=row), force={"T"})
    assert not errors
    assert _names(opt) == ["A"]


def test_sentence_is_short_and_in_vocabulary():
    text = wording.setting_took_several_values("T", 2)
    assert "2 values" in text and "Lowest = Highest" in text


def _with_choice(tmp_path, monkeypatch, picks):
    monkeypatch.chdir(tmp_path)
    o = FoodOptimizer(project_name="guard3", robust=False)
    o.add_ingredient("A", 0, 10)
    o.add_objective("Taste", 1.0, goal="max")
    o.variables.append({'name': "Pan", 'type': 'categorical',
                        'options': ["steel", "iron"], 'category': 'process'})
    for i, pick in enumerate(picks):
        o.tell({"A": 1.0 + i, "Pan": pick}, {"Taste": 5.0 + i})
    return o


def test_varied_categorical_refused_then_forced(tmp_path, monkeypatch):
    o = _with_choice(tmp_path, monkeypatch, ["steel", "iron"])
    with pytest.raises(ValueError) as caught:
        o.remove_process_parameter("Pan")
    assert str(caught.value) == wording.setting_took_several_values("Pan", 2)
    assert "Pan" in _names(o)
    o.remove_process_parameter("Pan", force=True)
    assert _names(o) == ["A"]


def test_constant_categorical_removes_without_force(tmp_path, monkeypatch):
    o = _with_choice(tmp_path, monkeypatch, ["steel", "steel"])
    o.remove_process_parameter("Pan")
    assert _names(o) == ["A"]


def test_an_older_files_blank_or_text_value_does_not_raise(opt):
    """A recorded value written by an older file can be a blank or a
    string; the count reads it the tolerant way the amounts are read."""
    opt.tell({"A": 1.0, "T": 150.0}, {"Taste": 5.0})
    opt.tell({"A": 1.0, "T": 150.0}, {"Taste": 6.0})
    opt.recipe_history[0]["T"] = ""
    opt.recipe_history[1]["T"] = "n/a"
    opt.remove_process_parameter("T")
    assert _names(opt) == ["A"]
