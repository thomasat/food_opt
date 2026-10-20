"""The numbers behind the views that explain the search (spec
2026-09-30-bo-views §3; review B4, I5, I9, I10, I11, I12, M4, M5, M10).

The suggestion model's acquisition, rebuilt on the display side without
touching ask(): the same training rows (insights.xy, the rows ask() learns
from), the same builder, the same default fit and the Round's own seed, then
opt._build_acqf, built fresh for every evaluation (optimize_acqf changes the
object it is given, 02 M5). A Round whose generation was recorded (ui_batch
records it at the Generate call site) and whose rows still match is rebuilt
exactly; any other Round is recomputed from today's records with the
display model of the same rows, and says so.

Every step holds insights.RNG_LOCK inside a forked torch RNG, so nothing
here can move a suggestion. The model is fitted once in this process and
every question is answered here with it (fix round 1, T3 I-3): sending it to
a worker only fitted it again, and could start the pool. A worker (kind
"acq", `answer`) still gives the same numbers from JSON for a caller that
has not fitted. No screen text lives here."""


import insights
from food_bo import COLD_START_RUNS


# ------------------------------------------------------------------ #
#  Which rows, which seed: the generation record (review B4)
# ------------------------------------------------------------------ #

def _complete_rows(opt, before=None):
    n = len(opt.Y_history) if before is None else before
    return [i for i in range(n) if opt.fully_measured(i)]


def generation_key(opt, rows):
    """What a rebuild of a Round depends on: the rows as the suggestion
    model learns them (insights.xy), their scores, the model settings and
    robust. A record whose key still matches is rebuilt exactly."""
    X, Y = insights.xy(opt, insights.SCORE, rows)
    return insights._digest({"X": X.tolist(), "Y": Y.tolist(),
                             "cfg": getattr(opt, "bo_config", None),
                             "robust": bool(getattr(opt, "robust", False))})


def generation_record(opt, n):
    """The record the Generate call site keeps (review B4), made just
    before ask(n): the seed ask() will use (next_formulation_no), the
    formulation numbers of the rows it learns from, n, the regime, and the
    key of those rows. The call site adds "picks" after ask() returns: the
    formulation numbers ask() issued, in order (T3 I-2)."""
    rows = _complete_rows(opt)
    warm = opt.complete_count() >= COLD_START_RUNS
    return {"seed": int(opt.next_formulation_no),
            "rows": [int(opt.formulation_ids[i]) for i in rows], "n": int(n),
            "regime": "warm" if warm else "cold",
            "key": generation_key(opt, rows) if warm else ""}


                        # order (T3 I-2); [] without a record that has them


