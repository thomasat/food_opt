"""Testing Level 2's packaged-app checks (test_e2e.sh tests 5-5g) against the
bundled Python, as the launcher runs it (-I -X utf8 -B), with the image's own
app folder as the resources; the image is unchanged afterwards."""
import hashlib
import importlib.util

from conftest import IMAGE, REPO, build_env, needs_image, record, run_image, slow

pytestmark = needs_image
CHECKS_FILE = REPO / "desktop" / "shared" / "packaged_app_checks.py"


def _sentinels():
    spec = importlib.util.spec_from_file_location("packaged_app_checks", CHECKS_FILE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return [s for names in module.SENTINELS.values() for s in names]


@slow
def test_packaged_app_checks_pass_with_the_bundled_python(tmp_path):
    projects = tmp_path / "FoodOptimizer"
    r = run_image(CHECKS_FILE, "--resources", IMAGE / "app", "--projects", projects,
                  home=tmp_path, timeout=1800)
    assert r.returncode == 0, r.stdout[-4000:] + r.stderr[-4000:]
    for sentinel in _sentinels():
        assert sentinel in r.stdout, sentinel
    text, _files, _size = build_env.file_manifest(IMAGE)
    assert hashlib.sha256(text.encode("utf-8")).hexdigest() == record()["image_sha256"]
