"""Tests of the shared Python launcher (desktop/shared/launcher). The
package is imported from the repo, so `import launcher` works here."""
import os
import sys

import pytest

from tests.shared_texts import REPO

SHARED = REPO / "desktop" / "shared"
if str(SHARED) not in sys.path:
    sys.path.insert(0, str(SHARED))

slow = pytest.mark.skipif(os.environ.get("FOODOPT_SLOW_TESTS") != "1",
                          reason="set FOODOPT_SLOW_TESTS=1")


def _kill(pid: int) -> None:
    from launcher import procinfo
    if procinfo.pid_alive(pid):
        procinfo.terminate(pid, grace=0.5)


@pytest.fixture
def world(tmp_path):
    from tests.desktop_launcher.harness import make_world
    w = make_world(tmp_path)
    yield w
    for launch in w.launches:
        if launch.proc.poll() is None:
            launch.proc.kill()
        if launch.proc.stdin is not None:
            launch.proc.stdin.close()
        try:
            launch.proc.wait(5)
        except Exception:
            pass
    for record in w.records():
        _kill(record["pid"])


def _impl_params():
    from tests.desktop_launcher.harness import IMPLS
    return [pytest.param(i, id=i.name,
                         marks=pytest.mark.skipif(i.name == "launcher.sh" and sys.platform != "darwin",
                                                  reason="launcher.sh runs on the Mac only"))
            for i in IMPLS]


@pytest.fixture(params=_impl_params())
def impl(request):
    return request.param


@pytest.fixture
def py():
    """The shared Python launcher, or the installed app's when
    FOODOPT_INSTALLED_APP is set (Level 2)."""
    from tests.desktop_launcher.harness import IMPLS, PythonImpl
    if os.environ.get("FOODOPT_INSTALLED_APP"):
        return IMPLS[0]
    return PythonImpl()


def healthy_launch(impl, world, timeout: float = 20, **kw):
    """Start a launch and wait until it answers its health check AND the
    launcher has said so itself: the server answers before launcher.sh's
    next health poll, and until then that launch is still in progress."""
    from tests.desktop_launcher.harness import wait_for

    def done():
        pp = world.port_pid()
        return (pp is not None and world.health() == 200
                and f"server healthy on port {pp[0]} (pid {pp[1]})" in world.log_text())

    launch = impl.start(world, **kw)
    if not wait_for(done, timeout):
        pytest.fail(f"{impl.name} never became healthy\n--- output ---\n{launch.output()}"
                    f"\n--- log ---\n{world.log_text()}")
    port, pid = world.port_pid()
    return launch, port, pid
