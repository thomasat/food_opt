"""The Mac bundle's golden listing, derived from the shared manifest.

    python desktop/bundle_listing.py            check; prints a diff, exit 1 if stale
    python desktop/bundle_listing.py --write    rewrite desktop/mac_bundle_files.txt

Stdlib only. A file reaching the bundle is a decision: review the diff of
the golden list before committing it.
"""
import difflib
import pathlib
import sys

REPO = pathlib.Path(__file__).resolve().parent.parent
# What the Mac bundle carries besides the manifest's files.
MAC_ONLY_FILES = {"Info.plist", "MacOS/FoodOptimizer", "MacOS/uv",
                  "Resources/launcher.sh", "Resources/requirements.lock.txt",
                  "Resources/icon.icns"}


def read_manifest(repo=REPO):
    path = pathlib.Path(repo) / "desktop" / "shared" / "bundle_manifest.txt"
    return path.read_text(encoding="utf-8").splitlines()


def expected_bundle_listing(manifest):
    """Every entry under Contents/ the Mac build makes from this manifest:
    files, and each parent directory with a trailing /, sorted."""
    files = MAC_ONLY_FILES | {f"Resources/{entry}" for entry in manifest}
    dirs = set()
    for path in files:
        parts = path.split("/")[:-1]
        dirs |= {"/".join(parts[:i]) + "/" for i in range(1, len(parts) + 1)}
    return sorted(files | dirs)


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    repo = REPO
    if "--repo" in argv:
        at = argv.index("--repo")
        if at + 1 >= len(argv):
            print("usage: bundle_listing.py [--write] [--repo <path>]")
            return 2
        repo = pathlib.Path(argv[at + 1])
        del argv[at:at + 2]
    if argv not in ([], ["--write"]):
        print("usage: bundle_listing.py [--write] [--repo <path>]")
        return 2
    golden = repo / "desktop" / "mac_bundle_files.txt"
    want = "".join(f"{e}\n" for e in expected_bundle_listing(read_manifest(repo)))
    have = golden.read_text(encoding="utf-8") if golden.exists() else ""
    if argv:
        golden.write_bytes(want.encode("utf-8"))
        return 0
    if have == want:
        return 0
    sys.stdout.writelines(difflib.unified_diff(have.splitlines(True), want.splitlines(True),
                                               "mac_bundle_files.txt", "expected"))
    return 1


if __name__ == "__main__":
    sys.exit(main())
