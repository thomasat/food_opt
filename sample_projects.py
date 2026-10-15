"""The app's samples: a real yogurt-style pilot, shipped as two projects,
and one example setup (a plant-based burger) with no results.

Rounds 1-2 were a random starting set drawn by the engine; Rounds 3-10 were
chosen by EGBO, an earlier version of this engine. The limits are the
pilot's own. The files under data/samples/yogurt_pilot
are rebuilt from the pilot's CSVs by build.py there, and
tests/test_sample_projects.py checks they still match.

The example setup (spec 2026-10-13 §9) shows how a project is set up:
calculated rows, pre-mixes, a process setting, records and the formulation
total, with illustrative values and no results. It is shipped from
data/samples/burger_setup (rebuilt by build.py there; checked by
tests/test_sample_burger.py) and is never part of the yogurt demo: opening
one never writes the other. Nothing here touches a project that is not one
of the three sample names."""
import json
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
SAMPLE_DIR = _ROOT / "data" / "samples" / "yogurt_pilot"
ROUND6 = "Yogurt pilot - Round 6 to score"
ALL10 = "Yogurt pilot - all 10 Rounds"
YOGURT_NAMES = (ROUND6, ALL10)
BURGER_DIR = _ROOT / "data" / "samples" / "burger_setup"
BURGER = "Example setup - plant-based burger"
NAMES = YOGURT_NAMES + (BURGER,)
ROUND6_RESULTS_FILE_NAME = "Round 6 results (pilot).xlsx"
ROUND6_RESULTS = SAMPLE_DIR / ROUND6_RESULTS_FILE_NAME
TEMPLATE_CSV = _ROOT / "data" / "ingredients_template.csv"
OPEN_NEXT_KEY = "_open_sample_next"
RESET_LABEL = "sample_reset"    # the archive label Start this sample over writes
_ROUND6 = 6                     # the pilot's Round 6, open in the shipped file


def _shipped_path(name):
    return (BURGER_DIR if name == BURGER else SAMPLE_DIR) / f"{name}.json"


def available():
    """The yogurt pilot's files are all here (Open the yogurt pilot)."""
    return ROUND6_RESULTS.is_file() and all(
        _shipped_path(n).is_file() for n in YOGURT_NAMES)


def burger_available():
    """The example setup's file is here (Open the burger example)."""
    return _shipped_path(BURGER).is_file()


def is_sample(name):
    return name in NAMES


def is_yogurt(name):
    """One of the two yogurt pilot projects: real results, the demo."""
    return name in YOGURT_NAMES


def untouched_example(opt):
    """The example setup as it ships: no formulation recorded or set aside
    and no Round open. Only then may the screen say it has no results."""
    return (getattr(opt, "project_name", None) == BURGER
            and not getattr(opt, "X_history", None)
            and not getattr(opt, "skipped", None)
            and not getattr(opt, "pending_batch", None))


def shipped_state(name):
    if not is_sample(name):
        raise ValueError(f"{name!r} is not a sample")
    with open(_shipped_path(name), encoding="utf-8") as fh:
        return json.load(fh)


def _write_round6_results(storage, overwrite):
    """A copy of the Round 6 results file beside the projects, so Browse
    files opens on it. Its name is not <name>.pkl, so it is never listed as
    a project or a copy. A storage with no folder on this disk gets none.

    A convenience only: a write that fails (a read-only folder, a folder of
    that name) is left unsaid, never blocking the samples or a restore —
    Download the pilot's Round 6 results is still on the screen. Returns
    whether the file is in place."""
    folder = getattr(storage, "folder", None)
    if folder is None:
        return False
    try:
        target = Path(folder()) / ROUND6_RESULTS_FILE_NAME
        if overwrite or not target.exists():
            target.write_bytes(ROUND6_RESULTS.read_bytes())
    except OSError:
        return False
    return target.is_file()


def ensure(storage, names=YOGURT_NAMES):
    """Write each of `names` (the yogurt pilot by default) that is not on
    disk; never overwrite one that is. With the Round 6 sample, its results
    file is written beside them the same way."""
    written = []
    for name in names:
        if not is_sample(name):
            raise ValueError(f"{name!r} is not a sample")
        if not storage.exists(name):
            storage.save(name, shipped_state(name))
            written.append(name)
    if ROUND6 in names:
        _write_round6_results(storage, overwrite=False)
    return written


def ensure_burger(storage):
    """The example setup alone: the yogurt pilot is never written by it."""
    return ensure(storage, (BURGER,))


def restore(name, storage):
    """Start this sample over: keep the current file as a copy, then put the
    shipped one back. Returns the copy's name, or None if there was no file.
    The copy is labelled sample_reset, which Saved copies lists as its own
    reason (wording.SAFETY_COPY_REASONS). The Round 6 sample also gets its
    results file back as shipped, so a demo started over has it."""
    state = shipped_state(name)          # refuses a non-sample before any write
    archived = storage.archive(name, RESET_LABEL, copy=False)
    storage.save(name, state)
    if name == ROUND6:
        _write_round6_results(storage, overwrite=True)
    return archived


def round6_results_bytes():
    return ROUND6_RESULTS.read_bytes()


def _round6_open(opt):
    return opt.pending_batch_no == _ROUND6 and bool(opt.pending_batch)


def offers_round6_download(opt):
    """The Round 6 sample while its Round 6 is open, whatever rows the
    reader added to it or moved out of it."""
    return opt.project_name == ROUND6 and _round6_open(opt)


def offers_what_next(opt):
    """The Round 6 sample once its Round 6 is recorded and closed, even if
    a row of it was moved to Not scored or Round 7 was made after it."""
    return (opt.project_name == ROUND6
            and _ROUND6 in {int(b) for b in opt.batch_history if b is not None}
            and not _round6_open(opt))
