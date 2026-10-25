"""The release texts, filled from the builds that are being released.

The IT sheet (desktop/windows/for_it_teams.md), the download email
(desktop/windows/download_email.md) and the release notes
(desktop/release_notes/<version>.md) are committed templates. A per-release
value is a {{key}} field; a claim no test backs yet is a PENDING(what)
marker. This fills them from the two build records, the SBOM and the Level 2
report, and renders the IT sheet to PDF with headless Chromium.

    python desktop/release_docs.py render --version X.Y.Z [--dist DIR] [--release] [--out DIR]

DIR holds what step 7 of the release ritual copies to the Mac: the dmg, the
exe, its .files.txt and .cdx.json, both *.build.txt records and report.json.
The output goes to --out (default desktop/dist/release-X.Y.Z/, ignored by
git): release-notes-X.Y.Z.md, FoodOptimizer-X.Y.Z-IT-sheet.html and .pdf,
download-email-X.Y.Z.txt.

A draft (no --release) shows every missing value and every pending claim as
"[pending: ...]". A release refuses them, and refuses a build that is not
signed, notarized, clean, from tag vX.Y.Z on both sides, or whose files no
longer match their records; it then writes nothing and exits 1 with
"refusing to render: <reason>".

Release tooling only (the Mac, at release time). markdown-it-py and
Playwright are imported where they are used, so the rest runs anywhere.
"""
import argparse
import hashlib
import html as html_lib
import importlib.util
import json
import pathlib
import re
import subprocess
import sys
from collections.abc import Mapping

DESKTOP = pathlib.Path(__file__).resolve().parent
REPO = DESKTOP.parent
URL_BASE = "https://github.com/thomasat/food_opt/releases/download"
TEMPLATES = {
    "it_sheet": DESKTOP / "windows" / "for_it_teams.md",
    "email": DESKTOP / "windows" / "download_email.md",
}

VALUE_KEYS = frozenset({
    "version", "exe_name", "exe_url", "dmg_name", "dmg_url", "it_sheet_name", "it_sheet_url",
    "sbom_name", "files_list_name", "exe_mb", "exe_sha256", "dmg_mb", "dmg_sha256", "commit",
    "image_files", "python_version", "vc_runtime_version", "inno_setup", "disk_free_gb",
    "unsigned_binaries", "image_gb", "signed", "sbom_components", "resigned_files",
    "licence_families", "gpl_statement", "wfp_events", "wfp_processes",
    # Added to the Windows record by the signing work; until then a draft
    # shows them as pending and a release refuses.
    "signer_subject", "signer_eku",
})
_WIN_COPIED = ("commit", "image_files", "python_version", "vc_runtime_version", "inno_setup",
               "disk_free_gb", "unsigned_binaries", "signed", "signer_subject", "signer_eku")

FIELD = re.compile(r"\{\{\s*([A-Za-z0-9_]+)\s*\}\}")
# One level of parentheses inside: PENDING(check before each release (owner)).
PENDING = re.compile(r"PENDING\(((?:[^()]|\([^()]*\))*)\)")
COMMENT = re.compile(r"<!--.*?-->", re.S)

NO_GPL = ("No component is licensed under the GPL, LGPL or AGPL "
          "(read from each package's metadata at build time).")


def assets(v: str) -> dict[str, str]:
    """The five files a release publishes, by role."""
    return {"dmg": f"FoodOptimizer-{v}.dmg", "exe": f"FoodOptimizer-{v}-Windows.exe",
            "sbom": f"FoodOptimizer-{v}-Windows.cdx.json", "files": f"FoodOptimizer-{v}-Windows.files.txt",
            "it_sheet": f"FoodOptimizer-{v}-IT-sheet.pdf"}


def records(v: str) -> dict[str, str]:
    """The two build records' file names (build_installer.ps1, build_dmg.sh)."""
    return {"win": f"FoodOptimizer-{v}-Windows.build.txt", "mac": f"FoodOptimizer-{v}.build.txt"}


def read_record(path) -> dict[str, str]:
    """A build record's key=value lines; a duplicate key is an error."""
    out: dict[str, str] = {}
    path = pathlib.Path(path)
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        key, sep, value = line.partition("=")
        if not sep:
            raise ValueError(f"{path.name}: not key=value: {line}")
        if key in out:
            raise ValueError(f"{path.name}: duplicate key {key}")
        out[key] = value
    return out


def _mb(n) -> str:
    return str(round(int(n) / 2**20))


def _gb(n) -> str:
    return f"{int(n) / 2**30:.1f}"


def sbom_summary(path) -> dict:
    """sbom.summary() of a CycloneDX file from sbom.py (loaded by path: it
    lives with the Windows build)."""
    spec = importlib.util.spec_from_file_location("foodopt_sbom", DESKTOP / "windows" / "sbom.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules.setdefault("foodopt_sbom", module)
    spec.loader.exec_module(module)
    return module.summary(json.loads(pathlib.Path(path).read_text(encoding="utf-8")))


def gpl_statement(summary: Mapping) -> str:
    """The sheet's licence statement, computed: never "no GPL" by hand."""
    licensed = list(summary.get("gpl_licensed", []))
    if licensed:
        text = "Components licensed under GPL-family terms: " + ", ".join(licensed) + "."
    else:
        text = NO_GPL
    seen = []
    for m in summary.get("gpl_mentions", []):
        item = f"{m['name']} {m['version']} ({m['file']})"
        if item not in seen:
            seen.append(item)
    if seen:
        text += " Licence notices that mention GPL-family terms: " + "; ".join(seen) + "."
    return text


def values(version, *, win=None, mac=None, sbom=None, l2=None, dist=None) -> dict[str, str]:
    """Every field value the inputs give; a missing input leaves its keys out."""
    a = assets(version)
    base = f"{URL_BASE}/v{version}/"
    vals = {"version": version,
            "exe_name": a["exe"], "exe_url": base + a["exe"],
            "dmg_name": a["dmg"], "dmg_url": base + a["dmg"],
            "it_sheet_name": a["it_sheet"], "it_sheet_url": base + a["it_sheet"],
            "sbom_name": a["sbom"], "files_list_name": a["files"]}
    if win:
        if "installer_bytes" in win:
            vals["exe_mb"] = _mb(win["installer_bytes"])
        if "installer_sha256" in win:
            vals["exe_sha256"] = win["installer_sha256"]
        if "image_bytes" in win:
            vals["image_gb"] = _gb(win["image_bytes"])
        vals.update({k: win[k] for k in _WIN_COPIED if k in win})
    if mac and "dmg_sha256" in mac:
        vals["dmg_sha256"] = mac["dmg_sha256"]
    if dist is not None and (pathlib.Path(dist) / a["dmg"]).is_file():
        vals["dmg_mb"] = _mb((pathlib.Path(dist) / a["dmg"]).stat().st_size)
    if sbom:
        vals["sbom_components"] = str(sbom["components"])
        vals["resigned_files"] = str(sbom["resigned_files"])
        fams = sorted(sbom["families"].items(), key=lambda kv: (-kv[1], kv[0]))
        vals["licence_families"] = "; ".join(f"{fam} ({n})" for fam, n in fams)
        vals["gpl_statement"] = gpl_statement(sbom)
    if l2 and "wfp_events_by_process" in l2:
        counts = l2["wfp_events_by_process"]
        vals["wfp_events"] = str(sum(counts.values()))
        vals["wfp_processes"] = ", ".join(sorted(counts))
    return vals


def fill(template: str, vals: Mapping[str, str], *, release: bool) -> str:
    """{{key}} -> its value; PENDING(what) -> "[pending: what]" in a draft.
    A release refuses a missing value or any marker; an unknown key is
    always an error. HTML comments stay (to_html drops them)."""
    def field(m):
        key = m.group(1)
        if key not in VALUE_KEYS:
            raise ValueError(f"unknown field: {key}")
        if key in vals:
            return str(vals[key])
        if release:
            raise ValueError(f"missing value: {key}")
        return f"[pending: {key}]"

    def pending(m):
        if release:
            raise ValueError(f"pending: {m.group(1)}")
        return f"[pending: {m.group(1)}]"
    return PENDING.sub(pending, FIELD.sub(field, template))


def _tag_commit(version: str) -> str:
    r = subprocess.run(["git", "-C", str(REPO), "rev-parse", "-q", "--verify", f"refs/tags/v{version}^{{commit}}"],
                       capture_output=True, text=True)
    return r.stdout.strip() if r.returncode == 0 else ""


def _sha256(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def check_release(version, win, mac, dist, l2, *, tag_commit: str | None = None) -> None:
    """Raise ValueError naming the first reason these inputs are not a
    release: versions, test or dirty builds, signing and notarization, the
    two records' commits and the tag, the size gate, each file's bytes
    against its record, and the Level 2 report."""
    on_file = (DESKTOP / "VERSION").read_text(encoding="utf-8").strip()
    if version != on_file:
        raise ValueError(f"version {version}: desktop/VERSION says {on_file}")
    if not win:
        raise ValueError(f"no Windows record ({records(version)['win']})")
    if not mac:
        raise ValueError(f"no Mac record ({records(version)['mac']})")
    for which, rec in (("Windows", win), ("Mac", mac)):
        if rec.get("version") != version:
            raise ValueError(f"version={rec.get('version')} ({which} record), not {version}")
    if win.get("test_build") != "no":
        raise ValueError(f"test_build={win.get('test_build')} (Windows record)")
    for which, rec in (("Windows", win), ("Mac", mac)):
        if rec.get("tree_clean") != "yes":
            raise ValueError(f"tree_clean={rec.get('tree_clean')} ({which} record)")
    for which, rec in (("Windows", win), ("Mac", mac)):
        if rec.get("signed") != "yes":
            raise ValueError(f"signed={rec.get('signed')} ({which} record)")
    if mac.get("notarized") != "yes":
        raise ValueError(f"notarized={mac.get('notarized')} (Mac record)")
    if win.get("commit") != mac.get("commit"):
        raise ValueError(f"commit: Windows {win.get('commit')} and Mac {mac.get('commit')} differ")
    tag = _tag_commit(version) if tag_commit is None else tag_commit
    if win.get("commit") != tag:
        raise ValueError(f"commit {win.get('commit')} is not tag v{version} ({tag or 'no such tag'})")
    if win.get("size_gate") != "pass":
        raise ValueError(f"size_gate={win.get('size_gate')} (Windows record)")
    a = assets(version)
    dist = pathlib.Path(dist)
    for role, rec, key in (("exe", win, "installer_sha256"), ("dmg", mac, "dmg_sha256"),
                           ("files", win, "files_list_sha256"), ("sbom", win, "sbom_sha256")):
        path = dist / a[role]
        if not path.is_file():
            raise ValueError(f"{key}: {a[role]} is not in {dist}")
        if _sha256(path) != rec.get(key):
            raise ValueError(f"{key}: {a[role]} does not match its record")
    if l2 is None:
        raise ValueError(f"no Level 2 report.json in {dist}")


# ---------- HTML and PDF ----------

CSS = """
body { font-family: -apple-system, "Segoe UI", Helvetica, Arial, sans-serif; font-size: 10.5pt;
       line-height: 1.4; color: #111; margin: 0; }
h1 { font-size: 18pt; margin: 0 0 8pt; } h2 { font-size: 13pt; margin: 16pt 0 6pt;
       border-bottom: 1px solid #999; padding-bottom: 2pt; break-after: avoid; }
h3 { font-size: 11pt; break-after: avoid; }
pre, code { font-family: Menlo, Consolas, "Courier New", monospace; font-size: 9pt;
            white-space: pre-wrap; word-break: break-all; }
pre { background: #f3f3f3; border: 1px solid #ccc; padding: 6pt; break-inside: avoid; }
table { border-collapse: collapse; width: 100%; margin: 6pt 0; break-inside: auto; }
th, td { border: 1px solid #999; padding: 3pt 5pt; text-align: left; vertical-align: top; }
th { background: #eee; }
tr { break-inside: avoid; }
.draft { border: 2px solid #b00; color: #b00; font-weight: bold; text-align: center;
         padding: 4pt; margin-bottom: 10pt; }
mark { background: #ffe08a; }
"""


def to_html(md: str, title: str, *, draft: bool) -> str:
    """One self-contained page: no typographic replacements (commands stay
    byte for byte), HTML comments dropped, inline CSS, nothing external."""
    from markdown_it import MarkdownIt
    body = MarkdownIt("commonmark", {"typographer": False}).enable("table").render(COMMENT.sub("", md))
    if draft:
        body = re.sub(r"\[pending: [^\]]*\]", lambda m: f"<mark>{m.group(0)}</mark>", body)
    banner = '<div class="draft">DRAFT - not for distribution</div>\n' if draft else ""
    page = (f'<!doctype html>\n<html lang="en">\n<head>\n<meta charset="utf-8">\n'
            f"<title>{html_lib.escape(title)}</title>\n<style>{CSS}</style>\n</head>\n"
            f"<body>\n{banner}{body}</body>\n</html>\n")
    if re.search(r"""<(?:link|script|img)\b[^>]*\b(?:src|href)\s*=\s*["']?\s*(?:https?:)?//""", page, re.I):
        raise ValueError("the sheet names an external file (link, script or image)")
    return page


def to_pdf(html: str, out, *, footer: str) -> None:
    """US Letter, 18 mm margins, a footer with page N of M. Every request
    but about: and data: is aborted; one is an error and no PDF is kept."""
    from playwright.sync_api import sync_playwright
    out = pathlib.Path(out)
    blocked = []

    def route(r):
        url = r.request.url
        if url.startswith(("about:", "data:")):
            r.continue_()
        else:
            blocked.append(url)
            r.abort()
    foot = (f'<div style="font-size:8px;width:100%;text-align:center;font-family:Helvetica,Arial,sans-serif;">'
            f'{html_lib.escape(footer)} - page <span class="pageNumber"></span> of '
            f'<span class="totalPages"></span></div>')
    with sync_playwright() as p:
        browser = p.chromium.launch()
        try:
            page = browser.new_page()
            page.route("**/*", route)
            page.set_content(html, wait_until="load")
            if blocked:
                raise RuntimeError(f"the sheet tried to load {blocked[0]}")
            page.pdf(path=str(out), format="Letter", print_background=True, display_header_footer=True,
                     header_template="<span></span>", footer_template=foot,
                     margin={"top": "18mm", "bottom": "18mm", "left": "18mm", "right": "18mm"})
        finally:
            browser.close()
    if blocked:
        out.unlink(missing_ok=True)
        raise RuntimeError(f"the sheet tried to load {blocked[0]}")


# ---------- render ----------

def _email_text(filled: str) -> str:
    """The sender comment's lines as plain lines on top, then the body."""
    m = re.match(r"\s*<!--(.*?)-->\s*", filled, re.S)
    if not m:
        return filled.strip() + "\n"
    head = [line.strip() for line in m.group(1).splitlines() if line.strip()]
    return "\n".join(head) + "\n\n" + filled[m.end():].strip() + "\n"


def _title(md: str) -> str:
    m = re.search(r"^# (.+)$", md, re.M)
    return m.group(1).strip() if m else "Food Optimizer"


def render(version, *, dist=None, out=None, release=False, templates=None, tag_commit=None) -> list[pathlib.Path]:
    """Fill and write the release texts; return the files written. Every
    check and every fill runs before the first write."""
    out = pathlib.Path(out) if out else DESKTOP / "dist" / f"release-{version}"
    win = mac = sbom = l2 = None
    if dist is not None:
        dist = pathlib.Path(dist)
        names = records(version)
        if (dist / names["win"]).is_file():
            win = read_record(dist / names["win"])
        if (dist / names["mac"]).is_file():
            mac = read_record(dist / names["mac"])
        if (dist / assets(version)["sbom"]).is_file():
            sbom = sbom_summary(dist / assets(version)["sbom"])
        if (dist / "report.json").is_file():
            l2 = json.loads((dist / "report.json").read_text(encoding="utf-8"))
    if release:
        if dist is None:
            raise ValueError("a release render needs --dist")
        check_release(version, win, mac, dist, l2, tag_commit=tag_commit)
    if templates is None:
        templates = {}
        for role, path in TEMPLATES.items():
            if path.is_file():
                templates[role] = path.read_text(encoding="utf-8")
        notes = DESKTOP / "release_notes" / f"{version}.md"
        if notes.is_file():
            templates["notes"] = notes.read_text(encoding="utf-8")
    if release:
        for role in ("it_sheet", "email", "notes"):
            if role not in templates:
                raise ValueError(f"no {role} template for {version}")
    vals = values(version, win=win, mac=mac, sbom=sbom, l2=l2, dist=dist)
    filled = {role: fill(text, vals, release=release) for role, text in templates.items()}
    pages = {}
    if "it_sheet" in filled:
        pages["it_sheet"] = to_html(filled["it_sheet"], _title(filled["it_sheet"]), draft=not release)
    out.mkdir(parents=True, exist_ok=True)
    written = []
    if "notes" in filled:
        path = out / f"release-notes-{version}.md"
        path.write_text(filled["notes"], encoding="utf-8")
        written.append(path)
    if "email" in filled:
        path = out / f"download-email-{version}.txt"
        path.write_text(_email_text(filled["email"]), encoding="utf-8")
        written.append(path)
    if "it_sheet" in pages:
        stem = pathlib.Path(assets(version)["it_sheet"]).stem
        html_path, pdf_path = out / f"{stem}.html", out / f"{stem}.pdf"
        html_path.write_text(pages["it_sheet"], encoding="utf-8")
        footer = f"Food Optimizer {version} for Windows: information for IT teams"
        to_pdf(pages["it_sheet"], pdf_path, footer=footer + ("" if release else " - DRAFT"))
        written += [html_path, pdf_path]
    return written


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="release_docs.py")
    sub = parser.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("render")
    r.add_argument("--version", required=True)
    r.add_argument("--dist")
    r.add_argument("--release", action="store_true")
    r.add_argument("--out")
    args = parser.parse_args(argv)
    try:
        written = render(args.version, dist=args.dist, out=args.out, release=args.release)
    except ValueError as e:
        print(f"refusing to render: {e}", file=sys.stderr)
        return 1
    for path in written:
        print(path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
