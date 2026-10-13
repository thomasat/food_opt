"""Test-wide settings.

Insights' worker processes are off in the test suite (FOODOPT_INSIGHTS_WORKERS
0): every screen then fits in the test's own process, as 0.8.0 did, so a
test sees the finished tab on its first run. The tests of the workers
start their own (tests/test_080_insights_worker.py and neighbours).

The cache beside the project is off too (FOODOPT_INSIGHTS_CACHE 0), so one
test's app never reads what another wrote; tests/test_080_insights_cache.py
and the speed UI tests turn it on."""
import importlib.util
import os
import shutil

import pytest

os.environ.setdefault("FOODOPT_INSIGHTS_WORKERS", "0")
os.environ.setdefault("FOODOPT_INSIGHTS_CACHE", "0")


R5 = "Yogurt pilot - Round 5"


@pytest.fixture(scope="session")
def pilot_dir(tmp_path_factory):
    """The shipped yogurt samples as the .pkl projects the app opens; Round 5
    is built from the same builder."""
    import sample_projects as sp
    d = tmp_path_factory.mktemp("pilot")
    for n in (sp.ROUND6, sp.ALL10):
        shutil.copy(sp.SAMPLE_DIR / f"{n}.json", d / f"{n}.pkl")
    spec = importlib.util.spec_from_file_location("yogurt_build", sp.SAMPLE_DIR / "build.py")
    b = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(b)
    b.build(R5, scored=5, out_dir=str(d))
    return d
