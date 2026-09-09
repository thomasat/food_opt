import pytest

from food_bo import FoodOptimizer
from storage import LocalStorage, StorageError


class FlakyStorage(LocalStorage):
    """LocalStorage that can be told to fail like a cloud backend."""
    persist_empty_on_init = False
    persist_after_load = False

    def __init__(self):
        super().__init__()
        self.fail_saves = False
        self.fail_loads = False

    def save(self, name, state):
        if self.fail_saves:
            raise StorageError("server unreachable — change NOT saved")
        super().save(name, state)

    def load(self, name):
        if self.fail_loads:
            raise StorageError("server unreachable — could not load")
        return super().load(name)

    def exists(self, name):
        if self.fail_loads:
            raise StorageError("server unreachable — could not load")
        return super().exists(name)


@pytest.fixture(autouse=True)
def in_tmp(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)


def test_default_storage_is_local_and_does_not_save_on_init():
    opt = FoodOptimizer("p")
    assert isinstance(opt.storage, LocalStorage)
    assert not opt.storage.exists("p")      # no phantom file until an edit is made


def test_no_ghost_row_when_backend_opts_out():
    s = FlakyStorage()
    FoodOptimizer("p", storage=s)
    assert not s.exists("p")                # cloud mode: no empty project


def test_save_error_flag_set_and_cleared():
    s = FlakyStorage()
    opt = FoodOptimizer("p", storage=s)
    s.fail_saves = True
    opt.add_ingredient("Water", 0, 100)     # internal save() swallows StorageError
    assert opt.save_error is not None
    s.fail_saves = False
    opt.add_ingredient("Flour", 0, 100)
    assert opt.save_error is None           # next good save self-heals


def test_load_failure_never_overwrites(tmp_path):
    s = FlakyStorage()
    opt = FoodOptimizer("p", storage=s)
    opt.add_ingredient("Water", 0, 100)     # now a real saved project exists
    s.fail_loads = True
    opt2 = FoodOptimizer("p", storage=s)    # backend down during load
    assert opt2.load_error is not None
    s.fail_loads = False
    opt3 = FoodOptimizer("p", storage=s)    # backend back: data intact
    assert opt3.load_error is None
    assert opt3.variables[0]["name"] == "Water"


def test_import_json_does_not_autosave():
    s = FlakyStorage()
    opt = FoodOptimizer("p", storage=s)
    opt.import_json({"project_name": "p", "variables": [], "objectives": []})
    assert not s.exists("p")                # caller decides when to persist


def test_fork_uses_storage_not_deepcopy():
    opt = FoodOptimizer("orig")
    opt.add_ingredient("Water", 0, 100)
    clone = opt.fork("branch")
    assert clone.storage is opt.storage
    assert clone.project_name == "branch"
    assert opt.storage.exists("branch")
    assert clone.variables[0]["name"] == "Water"


def _tiny_project(storage=None):
    opt = FoodOptimizer("t", storage=storage)
    opt.add_ingredient("Water", 0, 100)
    opt.add_objective("Taste", 1.0, "max", min_val=0, max_val=10)
    return opt


def test_tell_records_utc_timestamp():
    opt = _tiny_project()
    opt.tell({"Water": 10.0}, {"Taste": 5.0})
    assert len(opt.timestamps_history) == 1
    assert opt.timestamps_history[0].endswith("+00:00")  # UTC ISO


def test_timestamps_survive_roundtrip_and_old_files_pad_none():
    opt = _tiny_project()
    opt.tell({"Water": 10.0}, {"Taste": 5.0})
    state = opt.export_json()
    assert len(state["timestamps_history"]) == 1

    del state["timestamps_history"]        # simulate a pre-feature backup
    opt2 = FoodOptimizer("t2")
    opt2.import_json(state)
    assert opt2.timestamps_history == [None]


def test_delete_and_rewind_keep_lists_parallel():
    opt = _tiny_project()
    for v in (1.0, 2.0, 3.0):
        opt.tell({"Water": v}, {"Taste": v})
    opt.delete_result(1)
    assert len(opt.timestamps_history) == len(opt.results_history) == 2
    opt.rewind_to(0)
    assert len(opt.timestamps_history) == len(opt.results_history) == 1


def test_pending_batch_roundtrip_and_default():
    opt = _tiny_project()
    opt.set_pending_batch([{"Water": 3.0}])
    opt2 = FoodOptimizer("t")               # loads from disk
    assert opt2.pending_batch == [{"Water": 3.0}]

    state = opt.export_json()
    del state["pending_batch"]              # pre-feature backup
    opt3 = FoodOptimizer("t3")
    opt3.import_json(state)
    assert opt3.pending_batch is None


def test_design_space_mutations_clear_pending_batch():
    opt = _tiny_project()

    opt.set_pending_batch([{"Water": 3.0}])
    opt.add_ingredient("Sugar", 0, 50)
    assert opt.pending_batch is None

    opt.set_pending_batch([{"Water": 3.0, "Sugar": 1.0}])
    opt.add_objective("Crunch", 0.5, "max", min_val=0, max_val=10)
    assert opt.pending_batch is None

    opt.set_pending_batch([{"Water": 3.0, "Sugar": 1.0}])
    opt.add_process_parameter("Temp", 100, 200)
    assert opt.pending_batch is None

    opt.set_pending_batch([{"Water": 3.0, "Sugar": 1.0, "Temp": 150.0}])
    opt.deactivate_variable("Sugar")
    assert opt.pending_batch is None


def test_tell_and_edits_do_not_clear_pending_batch():
    opt = _tiny_project()
    opt.set_pending_batch([{"Water": 3.0}, {"Water": 6.0}])
    opt.tell({"Water": 3.0}, {"Taste": 5.0})
    assert opt.pending_batch is not None    # mid-batch: app clears at the end
    opt.edit_result(0, {"Taste": 6.0})
    assert opt.pending_batch is not None


def test_rewind_clears_pending_batch():
    opt = _tiny_project()
    opt.tell({"Water": 3.0}, {"Taste": 5.0})
    opt.tell({"Water": 6.0}, {"Taste": 6.0})
    opt.set_pending_batch([{"Water": 9.0}])
    opt.rewind_to(0)
    assert opt.pending_batch is None
