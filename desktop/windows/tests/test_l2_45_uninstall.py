"""Level 2: after modules 10-40 have run the installed app, its folder is
byte-for-byte what was installed, and a silent uninstall deletes the app
and its state but keeps the projects (spec section 5, Uninstall), even
when the state folder is the projects folder (Review Focus 2)."""
import pytest

import l2_support as l2


def test_install_folder_unchanged_after_runs(per_user):
    assert per_user.manifest
    assert l2.manifest(per_user.root) == per_user.manifest


def test_silent_uninstall_keeps_projects(per_user):
    home = per_user.home
    home.projects.mkdir(parents=True, exist_ok=True)
    planted = {home.projects / "My project.pkl": b"\x80\x04project bytes",
               home.projects / "Notes \u00e9.json": '{"note": "\u00e9t\u00e9"}'.encode("utf-8")}
    for path, data in planted.items():
        path.write_bytes(data)
    home.state.mkdir(parents=True, exist_ok=True)
    (home.state / "shell.log").write_text("x\n", encoding="utf-8")
    assert l2.uninstall(per_user.root, "user", env=home.env) == 0
    assert not per_user.root.exists()
    assert l2.uninstall_entry("user") is None
    dirs = l2.shortcut_dirs("user")
    for lnk in (dirs["programs"] / "Food Optimizer.lnk", dirs["desktop"] / "Food Optimizer.lnk",
                dirs["programs"] / "Food Optimizer - Start Here.lnk"):
        assert not lnk.exists(), lnk
    assert l2.wait_for(lambda: not home.state.exists(), 10), l2.last_uninstall_log()[-3000:]
    for path, data in planted.items():
        assert path.read_bytes() == data
    assert not l2.in_quick_access(home.projects)


# ---------- a state folder that is the projects folder (Review Focus 2) ----------

KEPT = "state folder kept (it holds the projects folder)"


def _uninstall_keeps_projects(installer, home, state):
    """A per-user install whose FOODOPT_STATE_DIR is `state`; after a silent
    uninstall the projects are all still there and the log says why."""
    env = dict(home.env, FOODOPT_STATE_DIR=str(state))
    result = l2.install(installer, "user", dir=home.root, env=env)
    assert result.rc == 0, result.log[-3000:]
    home.projects.mkdir(parents=True, exist_ok=True)
    keep = home.projects / "Keep me.pkl"
    keep.write_bytes(b"\x80\x04keep")
    assert l2.uninstall(home.root, "user", env=env) == 0
    log = l2.last_uninstall_log()
    assert not home.root.exists()
    assert keep.is_file(), log[-3000:]
    assert keep.read_bytes() == b"\x80\x04keep"
    assert KEPT in log, log[-3000:]


@pytest.mark.parametrize("written", ["same", "case", "slash", "double", "dots"])
def test_state_dir_equal_to_projects_is_kept(installer, clean_machine, l2_home, written):
    projects = str(l2_home.projects)
    parent, name = str(l2_home.projects.parent), l2_home.projects.name
    state = {"same": projects, "case": projects.upper(), "slash": projects + "\\",
             "double": parent + "\\\\" + name, "dots": parent + "\\.\\" + name}[written]
    _uninstall_keeps_projects(installer, l2_home, state)


def test_state_dir_holding_projects_is_kept(installer, clean_machine, l2_home):
    # The state folder is named FoodOptimizer and the projects live inside it.
    l2_home.projects = l2_home.home / "FoodOptimizer" / "Projects"
    l2_home.env["FOODOPT_PROJECTS_DIR"] = str(l2_home.projects)
    _uninstall_keeps_projects(installer, l2_home, l2_home.home / "FoodOptimizer")
