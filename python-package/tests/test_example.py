import runpy
from pathlib import Path

EXAMPLE = Path(__file__).resolve().parents[1] / "examples" / "quickstart.py"


def test_quickstart_runs(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    runpy.run_path(str(EXAMPLE), run_name="__main__")
    assert "Best" in capsys.readouterr().out
