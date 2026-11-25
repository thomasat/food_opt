"""The README's example runs as written."""
import re
from pathlib import Path

README = Path(__file__).resolve().parents[1] / "README.md"


def test_readme_example_runs(tmp_path, monkeypatch):
    blocks = re.findall(r"```python\n(.*?)```", README.read_text(), re.S)
    assert len(blocks) == 1
    monkeypatch.chdir(tmp_path)
    scope = {}
    exec(blocks[0], scope)
    assert len(scope["batch"]) == 3
    assert all(r["Coconut oil"] == 2 for r in scope["batch"])
    assert len(scope["opt"].history()) == 3
