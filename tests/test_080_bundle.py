"""0.8.0 speed, Plan A Task 6 (spec 2026-09-30 §10.5): the packaged app
carries the Insights modules, and a pool started from a copy of exactly
the bundled files, from another folder, runs its workers — the way the
app runs (`python -m streamlit run Resources/app.py` from ~/FoodOptimizer)."""
import json
import os
import pathlib
import re
import shutil
import subprocess
import sys

REPO = pathlib.Path(__file__).resolve().parent.parent
BUILD = REPO / "desktop" / "build_dmg.sh"


def bundle_list():
    line = next(l for l in BUILD.read_text().splitlines() if re.match(r"^for f in .*; do$", l))
    return line[len("for f in "):-len("; do")].split()


def test_the_bundle_carries_the_insights_modules():
    files = bundle_list()
    for f in ("insights_jobs.py", "insights_cache.py", "insights_worker.py",
              # The worker's "acq" kind imports it (spec 2026-09-30-bo-views §3.4).
              "bo_views.py"):
        assert f in files


def test_a_pool_from_the_bundled_files_runs_its_workers(tmp_path):
    resources = tmp_path / "Resources"
    resources.mkdir()
    for f in bundle_list():
        shutil.copy(REPO / f, resources / f)
    data = tmp_path / "FoodOptimizer"
    data.mkdir()
    script = (
        "import insights_jobs, json\n"
        "pool = insights_jobs.Pool(1)\n"
        "assert pool.wait_ready(120), 'no worker'\n"
        "print(json.dumps([insights_jobs.WORKER,\n"
        "                  pool.submit(0, {'id': 1, 'kind': 'ping'}).result(60)]))\n"
        "pool.close()\n")
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    env.update(PYTHONPATH=str(resources), PYTHONDONTWRITEBYTECODE="1")
    out = subprocess.run([sys.executable, "-c", script], cwd=data, env=env,
                         capture_output=True, text=True, timeout=180)
    worker, ping = json.loads(out.stdout.strip().splitlines()[-1])
    assert pathlib.Path(worker).parent == resources
    assert ping["threads"] == 1
    assert sorted(os.listdir(resources)) == sorted(bundle_list())    # nothing written there
    assert os.listdir(data) == []
