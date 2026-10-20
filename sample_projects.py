"""The app's samples: a real yogurt-style pilot, shipped as two projects,
and one example setup (a plant-based burger) with no results.

Rounds 1-2 were a random starting set drawn by the engine; Rounds 3-10 were
chosen by an earlier version of this engine. The limits are the
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
import hashlib
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
ENGINE_PICK = "Engine's pick"   # in the Note of every Round 3-10 batch (build.note_of)

# Earlier shipped states of the yogurt pilot (release-080 builds before the
# pilot's own limits and notes, commits 7b184dc and 29b1b24; and db0d6b7,
# before the attributes took the pilot's names Tanginess and Uniformity), by
# fingerprint(). A copy on disk equal to one of them was written by an
# earlier build and never changed, so refresh() and ensure() put the current
# one in its place; a copy that differs in anything is the reader's and is
# kept (Start this sample over still replaces it, keeping a copy).
_SUPERSEDED = {
    ROUND6: {"18f97c374a4eecdc5f14432a98e576c9218a58e0cdb5635211d497d074e09d11",
             "7f82e37a3c1c7c0fadc045b34b562599bc8b6e7ad577afd70cd7704b2697bfee",
             # db0d6b7 (v3): the attributes still Tangy, Thickness uniformity.
             "24942d76a206680844e52c16c201c4de8744d77ba65cc031f7bd1ab45b6861f9",
             # 5dc62ff: the notes still named the method ("· EGBO").
             "15029a1823e7ef0a9d7bc43d9ee265d4799cd440ca611926de57a60da75b3b76"},
    ALL10: {"a7ab23c492a9e3a2c705473ed21c9c784e21c4caa5db18587a661c8ad1096a2b",
            "d6cd35236c4c2ff0d0a9aa78bcba8e9b0cfef1524b35a4e9d168bc05e887f94e",
            "6684ce3ed9328bc29ea80da1c0ffa75af0cbb7e98c7782a2f4bed69488d8d82b",
            "77a7bef9d04291636e15bddc93cf5bbbe8793e48a5abd703171dd4b4d1c2d19e"},
}


def _shipped_path(name):
    return (BURGER_DIR if name == BURGER else SAMPLE_DIR) / f"{name}.json"


def available():
    """The yogurt pilot's files are all here (Try the yogurt pilot)."""
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


# The pilot's panel as its provenance records it (data/samples/yogurt_pilot/
# README.md, build.py, wording.SAMPLE_PROVENANCE_HELP): each rating is the
# median of a trained panel of five, one per attribute per batch, no batch
# repeated. A project file records no panel, so only the pilot's two projects
# say so (Review's "(n = 5)", Results' "Panel median (n = 5)").
PILOT_PANEL_SIZE = 5


def panel_size(opt):
    """The tasting panel's size (its ratings are the panel's median) for a
    yogurt pilot sample, else None."""
    return PILOT_PANEL_SIZE if is_yogurt(getattr(opt, "project_name", None)) else None


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


def shipped_scored(name):
    """How many formulations the sample ships with results for (the yogurt
    pilot's sidebar line: 15 in the Round 6 sample, 30 in the all-10 one)."""
    return len(shipped_state(name).get("Y_history") or [])


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


def round6_results_on_disk(storage):
    """Whether the Round 6 results file is beside the projects: asked
    before and after opening the pilot, so its success line counts the file
    only when that click wrote it (S4)."""
    folder = getattr(storage, "folder", None)
    if folder is None:
        return False
    try:
        return (Path(folder()) / ROUND6_RESULTS_FILE_NAME).is_file()
    except OSError:
        return False


def fingerprint(state):
    """A project state's identity, whatever its file's indentation."""
    return hashlib.sha256(json.dumps(state, sort_keys=True).encode()).hexdigest()


def _superseded(storage, name):
    """`name` is on disk exactly as an earlier build shipped it."""
    if name not in _SUPERSEDED or not storage.exists(name):
        return False
    try:
        state = storage.load(name)
    except Exception:            # a damaged file is the reader's to see
        return False
    return state is not None and fingerprint(state) in _SUPERSEDED[name]


def refresh(storage):
    """Put the current yogurt pilot in place of a copy an earlier build
    wrote and nobody changed (_SUPERSEDED); run once per session at launch.
    Returns the names replaced. A copy that differs in anything is left
    exactly as it is."""
    replaced = []
    for name in YOGURT_NAMES:
        if _superseded(storage, name):
            storage.save(name, shipped_state(name))
            replaced.append(name)
    return replaced


def ensure(storage, names=YOGURT_NAMES):
    """Write each of `names` (the yogurt pilot by default) that is not on
    disk, or is on disk exactly as an earlier build shipped it (refresh);
    never overwrite one the reader has changed. With the Round 6 sample,
    its results file is written beside them the same way."""
    written = []
    for name in names:
        if not is_sample(name):
            raise ValueError(f"{name!r} is not a sample")
        if not storage.exists(name) or _superseded(storage, name):
            storage.save(name, shipped_state(name))
            written.append(name)
    if ROUND6 in names:
        _write_round6_results(storage, overwrite=False)
    return written


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
