"""Test-wide fixtures: the pilot projects."""
import importlib.util
import shutil

import pytest


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
