"""Insights for a first-time user, fix round C (ledger, "Early walkthrough"
rulings; 05-walkthrough-retest-early.md §3, §5 items 2, 3, 5, 9;
06-statistician-cold-read.md I1). What stays: the sampler's 512 draws and
the landing that follows its place while the screen fills in. The Why
fold, its strings and the Insights status links left with the v3 Review."""
import pytest

import ui_helpers


@pytest.fixture(autouse=True)
def _isolate(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)


def test_the_512_draws_are_the_samplers_own():
    import pathlib
    source = (pathlib.Path(__file__).resolve().parent.parent / "food_bo.py").read_text()
    assert "SobolQMCNormalSampler(sample_shape=torch.Size([512]))" in source


def test_a_landing_follows_its_place_while_the_screen_fills_in():
    """Measured: the Round's predictions arrived after the landing and moved
    the Add fold down 55-430 px. The place is kept under the header for 8 s
    (80 x 100 ms), until the reader scrolls, clicks or types."""
    script = ui_helpers.scroll_script(ui_helpers.ADD_OWN_PLACE, 1)
    assert "follow(k - 1), 100" in script and "})(80);" in script
    for kind in ("wheel", "touchstart", "keydown", "mousedown"):
        assert f'"{kind}"' in script
