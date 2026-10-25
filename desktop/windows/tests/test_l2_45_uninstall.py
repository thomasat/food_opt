"""Level 2: after modules 10-40 have run the installed app, its folder is
byte-for-byte what was installed, and a silent uninstall deletes the app
and its state but keeps the projects (spec section 5, Uninstall)."""
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
