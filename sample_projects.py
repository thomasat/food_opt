"""The app's samples: a real yogurt-style pilot, shipped as two projects.

Rounds 1-2 were the team's starting set; Rounds 3-10 were chosen by EGBO,
an earlier version of this engine. The files under data/samples/yogurt_pilot
are rebuilt from the pilot's CSVs by build.py there, and
tests/test_sample_projects.py checks they still match. Nothing here touches
a project that is not one of the two sample names."""
import json
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
SAMPLE_DIR = _ROOT / "data" / "samples" / "yogurt_pilot"
ROUND6 = "Yogurt pilot - Round 6 to score"
ALL10 = "Yogurt pilot - all 10 Rounds"
NAMES = (ROUND6, ALL10)
ROUND6_RESULTS_FILE_NAME = "Round 6 results (pilot).xlsx"
ROUND6_RESULTS = SAMPLE_DIR / ROUND6_RESULTS_FILE_NAME
TEMPLATE_CSV = _ROOT / "data" / "ingredients_template.csv"
OPEN_NEXT_KEY = "_open_sample_next"
_ROUND6_ROWS = {16, 17, 18}     # the pilot's Round 6: Formulations 16-18
_RECORDED_AFTER_ROUND6 = 18


def available():
    return ROUND6_RESULTS.is_file() and all(
        (SAMPLE_DIR / f"{n}.json").is_file() for n in NAMES)


def is_sample(name):
    return name in NAMES


def shipped_state(name):
    if not is_sample(name):
        raise ValueError(f"{name!r} is not a sample")
    with open(SAMPLE_DIR / f"{name}.json", encoding="utf-8") as fh:
        return json.load(fh)


def ensure(storage):
    """Write each sample that is not on disk; never overwrite one that is."""
    written = []
    for name in NAMES:
        if not storage.exists(name):
            storage.save(name, shipped_state(name))
            written.append(name)
    return written


def restore(name, storage):
    """Start this sample over: keep the current file as a copy, then put the
    shipped one back. Returns the copy's name, or None if there was no file."""
    state = shipped_state(name)          # refuses a non-sample before any write
    archived = storage.archive(name, "pre_restore", copy=False)
    storage.save(name, state)
    return archived


def round6_results_bytes():
    return ROUND6_RESULTS.read_bytes()


def offers_round6_download(opt):
    rows = opt.pending_batch or []
    return (opt.project_name == ROUND6
            and {r.get("formulation") for r in rows} == _ROUND6_ROWS)


def offers_what_next(opt):
    return opt.project_name == ROUND6 and len(opt.X_history) >= _RECORDED_AFTER_ROUND6
