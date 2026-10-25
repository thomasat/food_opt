"""Level 2: a silent per-user install, as the limited user, to a folder
whose user name has a space and accents (spec Testing, Level 2; Review
Focus 5). Modules 10-45 share this one install (the per_user fixture)."""
import re

import l2_support as l2

FSI = "Food System Innovations"


def test_silent_per_user_install_needs_no_admin(per_user):
    assert per_user.result.rc == 0, per_user.result.log[-3000:]
    assert per_user.result.elevated is False
    assert l2.uninstall_entry("machine") is None
    entry = l2.uninstall_entry("user")
    assert entry is not None
    assert entry["DisplayName"] == "Food Optimizer"
    assert entry["DisplayVersion"] == l2.VERSION
    assert entry["Publisher"] == FSI
    assert entry["InstallLocation"] == str(per_user.root) + "\\"
    assert "unins000.exe" in entry["QuietUninstallString"]


def test_installed_files_are_the_image(per_user, dist_record):
    import hashlib
    assert hashlib.sha256(per_user.manifest.encode("utf-8")).hexdigest() == dist_record["image_sha256"]
    assert (per_user.root / "unins000.exe").is_file()
    assert (per_user.root / "unins000.dat").is_file()


def test_shortcuts_point_at_the_app(per_user):
    dirs = l2.shortcut_dirs("user")
    exe = str(per_user.root / "FoodOptimizer.exe")
    assert l2.shortcut_target(dirs["programs"] / "Food Optimizer.lnk").lower() == exe.lower()
    assert l2.shortcut_target(dirs["desktop"] / "Food Optimizer.lnk").lower() == exe.lower()
    assert (l2.shortcut_target(dirs["programs"] / "Food Optimizer - Start Here.lnk").lower()
            == str(per_user.root / "Start Here.txt").lower())


def test_projects_folder_is_made_and_pinned(per_user):
    assert per_user.home.projects.is_dir()
    assert l2.in_quick_access(per_user.home.projects), l2.quick_access()


def test_prewarm_ran_at_install(per_user):
    m = re.search(r"prewarm rc=0 secs=(\d+)", per_user.result.log)
    assert m, per_user.result.log[-3000:]
    l2.report("install_s", round(per_user.result.seconds, 1))
    l2.report("prewarm_s", int(m.group(1)))


def test_installer_and_app_carry_the_version(per_user, installer):
    info = l2.version_info(installer)
    assert info["ProductVersion"] == l2.VERSION
    assert info["CompanyName"] == FSI
    assert l2.version_info(per_user.root / "FoodOptimizer.exe")["FileVersion"] == l2.VERSION + ".0"
