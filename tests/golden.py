"""The one comparison every golden test uses (owner, 2026-10-24).

On the Mac a golden result must be bit-identical. On Windows the same code
runs on another BLAS and rounds floats differently (measured: at most
1.55e-4 g, relative 2.7e-5), so there numbers may differ by 0.1 % relative,
with a tiny absolute floor for zeros. Strings, keys and lengths must match
exactly everywhere."""
import json
import math
import sys

import torch

WINDOWS_REL = 1e-3
WINDOWS_ABS = 1e-12


def _close(got, want, where):
    if isinstance(want, dict):
        assert isinstance(got, dict) and set(got) == set(want), where
        for k in want:
            _close(got[k], want[k], f"{where}.{k}")
    elif isinstance(want, list):
        assert isinstance(got, list) and len(got) == len(want), where
        for i, (g, w) in enumerate(zip(got, want)):
            _close(g, w, f"{where}[{i}]")
    elif isinstance(want, (int, float)) and not isinstance(want, bool) \
            and isinstance(got, (int, float)) and not isinstance(got, bool):
        assert math.isclose(got, want, rel_tol=WINDOWS_REL, abs_tol=WINDOWS_ABS), \
            (where, got, want)
    else:
        assert got == want, where


def same(got, want):
    got = json.loads(json.dumps(got, default=float))
    for key in want:
        if sys.platform == "win32":
            _close(got[key], want[key], key)
        elif key in ("gp_X", "gp_Y"):
            assert torch.equal(torch.tensor(got[key], dtype=torch.double),
                               torch.tensor(want[key], dtype=torch.double)), key
        else:
            assert got[key] == want[key], key
