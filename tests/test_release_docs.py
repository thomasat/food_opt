"""desktop/release_docs.py (phase 6): the release texts (the IT sheet, the
download email, the release notes) are committed templates whose per-release
values are {{key}} fields and whose unproven claims are PENDING(...) markers.
A draft render shows both; a release render refuses them, and refuses any
build that is not the signed, notarized, clean, tagged one whose bytes match
its records. The IT sheet's PDF holds every command byte for byte and loads
nothing from the network."""
import hashlib
import importlib.util
import json
import pathlib
import re
import subprocess
import sys

import pytest

REPO = pathlib.Path(__file__).resolve().parent.parent
RELEASE_DOCS = REPO / "desktop" / "release_docs.py"
VERSION = (REPO / "desktop" / "VERSION").read_text(encoding="utf-8").strip()


def _load():
    spec = importlib.util.spec_from_file_location("foodopt_release_docs", RELEASE_DOCS)
    module = importlib.util.module_from_spec(spec)
    sys.modules["foodopt_release_docs"] = module
    spec.loader.exec_module(module)
    return module


rd = _load()

COMMIT = "c0ffee" + "0" * 34

# The phase 5 release record's keys (build_installer.ps1), phase 6's five,
# and the Mac record's (build_dmg.sh).
WIN_KEYS = ("version commit tree_clean test_build mac_lock_sha256 windows_lock_sha256 python_version "
            "python_nupkg_sha256 vc_runtime_version vc_redist_sha256 uv_version image_files image_bytes "
            "image_sha256 longest_path_chars unsigned_binaries dotnet_sdk shell_files inno_setup installer "
            "installer_bytes installer_sha256 disk_free_gb size_gate files_list_sha256 sbom_sha256 "
            "sbom_components sbom_gpl_licensed sbom_gpl_mentions signed").split()
MAC_KEYS = "version commit tree_clean mac_lock_sha256 windows_lock_sha256 dmg_sha256 signed notarized".split()


def _sha(path):
    return hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()


def _bom():
    def comp(ref, name, version, fam, extra=()):
        props = [{"name": "foodopt:licence_family", "value": fam},
                 {"name": "foodopt:licence_source", "value": "classifier"},
                 {"name": "foodopt:resigned_files", "value": "0"}, *extra]
        return {"bom-ref": ref, "name": name, "version": version, "type": "library",
                "licenses": [{"license": {"id": "MIT"}}], "properties": props}
    return {"bomFormat": "CycloneDX", "specVersion": "1.6", "version": 1,
            "metadata": {"component": {"bom-ref": "app:food-optimizer@x", "name": "Food Optimizer",
                                       "properties": [{"name": "foodopt:resigned_files", "value": "5"}]}},
            "components": [comp("pypi:a@1", "a", "1", "MIT"),
                           comp("pypi:numpy@2.4.3", "numpy", "2.4.3", "BSD", [
                               {"name": "foodopt:gpl_mention", "value": "numpy-2.4.3.dist-info/LICENSE.txt:3: GNU GPL"},
                               {"name": "foodopt:gpl_mention", "value": "numpy-2.4.3.dist-info/LICENSE.txt:9: GPL v3"}])]}


def make_dist(tmp, version=VERSION, win=None, mac=None, report=True):
    """A release input folder as step 7 of the ritual leaves it: the five
    assets' sources, both records (valid unless overridden) and report.json."""
    dist = pathlib.Path(tmp) / "in"
    dist.mkdir(parents=True, exist_ok=True)
    a = rd.assets(version)
    (dist / a["exe"]).write_bytes(b"MZ exe bytes" * 1000)
    (dist / a["dmg"]).write_bytes(b"dmg bytes" * 3000)
    (dist / a["files"]).write_text("ab" * 32 + "  app/app.py\n", encoding="utf-8")
    (dist / a["sbom"]).write_text(json.dumps(_bom()), encoding="utf-8")
    if report:
        (dist / "report.json").write_text(json.dumps({"wfp_events_by_process": {
            "python.exe": 20, "FoodOptimizer.exe": 3}}), encoding="utf-8")
    w = {k: "x" for k in WIN_KEYS}
    w.update(version=version, commit=COMMIT, tree_clean="yes", test_build="no", installer=a["exe"],
             installer_bytes=str((dist / a["exe"]).stat().st_size), installer_sha256=_sha(dist / a["exe"]),
             image_files="27770", image_bytes="1174185550", unsigned_binaries="299", disk_free_gb="2",
             size_gate="pass", files_list_sha256=_sha(dist / a["files"]), sbom_sha256=_sha(dist / a["sbom"]),
             python_version="3.13.7", vc_runtime_version="14.51.36247.0", inno_setup="6.5.4", signed="yes")
    w.update(win or {})
    m = {k: "x" for k in MAC_KEYS}
    m.update(version=version, commit=COMMIT, tree_clean="yes", dmg_sha256=_sha(dist / a["dmg"]),
             signed="yes", notarized="yes")
    m.update(mac or {})
    (dist / f"FoodOptimizer-{version}-Windows.build.txt").write_text(
        "".join(f"{k}={v}\n" for k, v in w.items()), encoding="utf-8")
    (dist / f"FoodOptimizer-{version}.build.txt").write_text(
        "".join(f"{k}={v}\n" for k, v in m.items()), encoding="utf-8")
    return dist


CLEAN_TEMPLATES = {"it_sheet": "# Sheet\n\n{{exe_sha256}}\n", "email": "<!--\nSubject: {{version}}\n-->\nHi\n",
                   "notes": "Notes {{version}}\n"}


# ---------- values, fields, markers ----------

def test_assets_are_the_five_release_files():
    assert rd.assets("1.2.3") == {
        "dmg": "FoodOptimizer-1.2.3.dmg", "exe": "FoodOptimizer-1.2.3-Windows.exe",
        "sbom": "FoodOptimizer-1.2.3-Windows.cdx.json", "files": "FoodOptimizer-1.2.3-Windows.files.txt",
        "it_sheet": "FoodOptimizer-1.2.3-IT-sheet.pdf"}


def test_values_follow_the_units():
    vals = rd.values("0.9.0", win={"installer_bytes": "249551715", "image_bytes": "1174185550"})
    assert vals["exe_mb"] == "238"
    assert vals["image_gb"] == "1.1"
    assert vals["exe_url"] == rd.URL_BASE + "/v0.9.0/FoodOptimizer-0.9.0-Windows.exe"
    assert vals["it_sheet_url"] == rd.URL_BASE + "/v0.9.0/FoodOptimizer-0.9.0-IT-sheet.pdf"
    assert "exe_sha256" not in vals
    assert set(vals) <= rd.VALUE_KEYS


def test_values_read_every_input(tmp_path):
    dist = make_dist(tmp_path, "0.9.0")
    win = rd.read_record(dist / "FoodOptimizer-0.9.0-Windows.build.txt")
    mac = rd.read_record(dist / "FoodOptimizer-0.9.0.build.txt")
    l2 = json.loads((dist / "report.json").read_text(encoding="utf-8"))
    summary = rd.sbom_summary(dist / rd.assets("0.9.0")["sbom"])
    vals = rd.values("0.9.0", win=win, mac=mac, sbom=summary, l2=l2, dist=dist)
    assert set(vals) == rd.VALUE_KEYS - {"signer_subject", "signer_eku"}
    assert (vals["wfp_events"], vals["wfp_processes"]) == ("23", "FoodOptimizer.exe, python.exe")
    assert vals["licence_families"] == "BSD (1); MIT (1)"
    assert (vals["sbom_components"], vals["resigned_files"]) == ("2", "5")
    assert vals["dmg_mb"] == str(round(27000 / 2**20))


def test_read_record_refuses_a_duplicate_key(tmp_path):
    path = tmp_path / "x.build.txt"
    path.write_text("a=1\nb=2\na=3\n", encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate key a"):
        rd.read_record(path)


def test_fill_draft_shows_what_is_missing():
    assert rd.fill("{{exe_sha256}} PENDING(x)", {}, release=False) == "[pending: exe_sha256] [pending: x]"
    assert rd.fill("PENDING(check (owner)) {{version}}", {"version": "1"}, release=False) == \
        "[pending: check (owner)] 1"


@pytest.mark.parametrize("release", [False, True])
def test_fill_refuses_an_unknown_field(release):
    with pytest.raises(ValueError, match="unknown field: nonsense"):
        rd.fill("{{nonsense}}", {"version": "1"}, release=release)


def test_fill_release_refuses_missing_and_pending():
    with pytest.raises(ValueError, match="missing value: exe_sha256"):
        rd.fill("{{exe_sha256}}", {}, release=True)
    with pytest.raises(ValueError, match="pending: x"):
        rd.fill("<!-- PENDING(x) -->", {}, release=True)
    assert rd.fill("<!-- a -->{{version}}", {"version": "1"}, release=True) == "<!-- a -->1"


def test_gpl_statement_is_computed():
    none = {"gpl_licensed": [], "gpl_mentions": []}
    assert rd.gpl_statement(none) == ("No component is licensed under the GPL, LGPL or AGPL "
                                      "(read from each package's metadata at build time).")
    one = {"gpl_licensed": [], "gpl_mentions": [
        {"name": "numpy", "version": "2.4.3", "file": "numpy.libs/LICENSE_gfortran.txt", "line": 1},
        {"name": "numpy", "version": "2.4.3", "file": "numpy.libs/LICENSE_gfortran.txt", "line": 7}]}
    assert rd.gpl_statement(one) == rd.gpl_statement(none) + (
        " Licence notices that mention GPL-family terms: numpy 2.4.3 (numpy.libs/LICENSE_gfortran.txt).")
    licensed = {"gpl_licensed": ["pypi:x@1", "pypi:y@2"], "gpl_mentions": []}
    text = rd.gpl_statement(licensed)
    assert "No component" not in text
    assert text == "Components licensed under GPL-family terms: pypi:x@1, pypi:y@2."


# ---------- release refusals (Review Focus 1, 4) ----------

def _case(name, tmp):
    if name == "test_build":
        return dict(dist=make_dist(tmp, win={"test_build": "yes"})), "test_build=yes"
    if name == "signed":
        return dict(dist=make_dist(tmp, win={"signed": "no"})), "signed=no"
    if name == "notarized":
        return dict(dist=make_dist(tmp, mac={"notarized": "no"})), "notarized=no"
    if name == "commits":
        return dict(dist=make_dist(tmp, mac={"commit": "f" * 40})), "commit"
    if name == "version":
        return dict(dist=make_dist(tmp, "9.9.9"), version="9.9.9"), "desktop/VERSION"
    if name == "exe_bytes":
        dist = make_dist(tmp)
        exe = dist / rd.assets(VERSION)["exe"]
        raw = bytearray(exe.read_bytes())
        raw[100] ^= 1
        exe.write_bytes(bytes(raw))
        return dict(dist=dist), "installer_sha256"
    if name == "report":
        return dict(dist=make_dist(tmp, report=False)), "report.json"
    if name == "pending":
        return dict(dist=make_dist(tmp), templates={**CLEAN_TEMPLATES, "it_sheet": "# S\n\nPENDING(y)\n"}), "pending: y"
    raise AssertionError(name)


CASES = ["test_build", "signed", "notarized", "commits", "version", "exe_bytes", "report", "pending"]


@pytest.mark.parametrize("name", CASES)
def test_release_render_refuses(tmp_path, name):
    kwargs, field = _case(name, tmp_path)
    args = dict(version=VERSION, out=tmp_path / "out", release=True, templates=CLEAN_TEMPLATES,
                tag_commit=COMMIT)
    args.update(kwargs)
    with pytest.raises(ValueError, match=re.escape(field)):
        rd.render(**args)
    assert not (tmp_path / "out").exists()


def test_a_clean_release_input_passes_the_checks(tmp_path):
    dist = make_dist(tmp_path)
    win = rd.read_record(dist / f"FoodOptimizer-{VERSION}-Windows.build.txt")
    mac = rd.read_record(dist / f"FoodOptimizer-{VERSION}.build.txt")
    rd.check_release(VERSION, win, mac, dist, {"wfp_events_by_process": {}}, tag_commit=COMMIT)
    with pytest.raises(ValueError, match=r"v%s" % re.escape(VERSION)):
        rd.check_release(VERSION, win, mac, dist, {}, tag_commit="f" * 40)


def test_release_render_writes_nothing_when_refused(tmp_path, capsys):
    dist = make_dist(tmp_path, win={"signed": "no"})
    out = tmp_path / "out"
    code = rd.main(["render", "--version", VERSION, "--dist", str(dist), "--release", "--out", str(out)])
    assert code == 1
    assert capsys.readouterr().err.startswith("refusing to render: signed=no")
    assert not out.exists() or not any(out.iterdir())


def test_render_output_is_ignored():
    assert subprocess.run(["git", "-C", str(REPO), "check-ignore", "-q", "desktop/dist/release-0.9.0/x.pdf"]).returncode == 0


# ---------- HTML and PDF (conda base: markdown-it-py, Playwright, PyMuPDF) ----------

FIXTURE_MD = """# Fixture sheet

<!-- for the sender only -->

| Code | Meaning |
|---|---|
| `0` | Installed -- "done" |

Prepared for José Müller.

```
"FoodOptimizer-0.9.0-Windows.exe" /VERYSILENT /SUPPRESSMSGBOXES /NORESTART /CLOSEAPPLICATIONS
```
"""


def test_html_is_self_contained():
    pytest.importorskip("markdown_it")
    html = rd.to_html(FIXTURE_MD, "Fixture", draft=True)
    assert "<table>" in html
    assert "for the sender only" not in html
    assert '<meta charset="utf-8">' in html
    assert not re.search(r"""src=["']?http|<link[^>]*http""", html)
    assert "José Müller" in html and "/VERYSILENT" in html
    assert 'Installed -- "done"' in html or "Installed -- &quot;done&quot;" in html
    assert "DRAFT - not for distribution" in html
    assert "DRAFT" not in rd.to_html(FIXTURE_MD, "Fixture", draft=False)


COMMANDS = [
    '"FoodOptimizer-0.9.0-Windows.exe" /VERYSILENT /SUPPRESSMSGBOXES /NORESTART /CLOSEAPPLICATIONS /ALLUSERS [/NODESKTOPSHORTCUT]',
    '"<install folder>\\unins000.exe" /VERYSILENT /SUPPRESSMSGBOXES /NORESTART',
    "HKEY_LOCAL_MACHINE\\SOFTWARE\\Microsoft\\Windows\\CurrentVersion\\Uninstall\\{DE876ED9-927A-45AD-AD9B-D6E4B4C9A4A3}_is1",
]


def _pdf_text(md, tmp_path, footer):
    fitz = pytest.importorskip("fitz")
    pytest.importorskip("playwright")
    pytest.importorskip("markdown_it")
    out = tmp_path / "x.pdf"
    rd.to_pdf(rd.to_html(md, "Fixture", draft=False), out, footer=footer)
    doc = fitz.open(out)
    return [page.get_text() for page in doc]


SHEET = REPO / "desktop" / "windows" / "for_it_teams.md"
# The real sheet's draft rendered to 10 Letter pages (2026-10-25); one page
# of slack before a layout change has to be looked at.
SHEET_PAGES = 11


def _fenced(md):
    out = []
    for indent, body in re.findall(r"^( *)```\n(.*?)^\1```$", md, re.M | re.S):
        out += [line[len(indent):] for line in body.strip("\n").splitlines()]
    return out


@pytest.mark.parametrize("case", ["fixture", "it-sheet"])
def test_pdf_keeps_commands_exact(tmp_path, case):
    if case == "fixture":
        md = "# Commands\n\n" + "".join(f"```\n{c}\n```\n\n" for c in COMMANDS)
        commands = COMMANDS
    else:
        md = rd.fill(SHEET.read_text(encoding="utf-8"), rd.values("0.9.0"), release=False)
        commands = [line for line in _fenced(md) if "/VERYSILENT" in line or "_is1" in line or "Get-" in line]
        assert len(commands) >= 7, commands
    pages = _pdf_text(md, tmp_path, footer="Fixture footer")
    flat = re.sub(r"\s+", "", "".join(pages))
    for command in commands:
        assert re.sub(r"\s+", "", command) in flat, command
    for bad in ("\u2013", "\u2014", "\u201c", "\u201d", "\u00ad"):
        assert bad not in "".join(pages), repr(bad)
    assert "Fixture footer" in pages[0]
    if case == "it-sheet":
        assert len(pages) <= SHEET_PAGES, len(pages)


def test_pdf_fetches_nothing(tmp_path):
    pytest.importorskip("playwright")
    html = '<!doctype html><html><body><p>x</p><img src="https://example.com/x.png"></body></html>'
    with pytest.raises(RuntimeError, match="example.com"):
        rd.to_pdf(html, tmp_path / "x.pdf", footer="x")
    assert not (tmp_path / "x.pdf").exists()


# ---------- the download email (Task 6) ----------

EMAIL = REPO / "desktop" / "windows" / "download_email.md"
SPEC_LINES = ["1. Open the downloaded file.", "2. Click Install. No admin rights needed.",
              "3. Open Food Optimizer from Start.",
              "Works offline. Needs Windows 10 or 11, {{disk_free_gb}} GB free.",
              "Blocked? Forward this email to IT."]
SAMPLE = {"version": "0.9.0", "exe_mb": "238", "disk_free_gb": "2", **rd.values("0.9.0")}


def _start_here_smartscreen():
    from tests.shared_texts import flowed, read_lf
    text = flowed(read_lf(REPO / "desktop" / "windows" / "start_here.txt"))
    return re.search(r"Windows may not recognize the app yet\..*?Run anyway\.", text).group(0)


def _email_body(text):
    return rd.COMMENT.sub("", text).strip()


def test_email_has_the_spec_lines_in_order():
    body = _email_body(EMAIL.read_text(encoding="utf-8"))
    lines = [line.strip() for line in body.splitlines()]
    at = [lines.index(line) for line in SPEC_LINES]
    assert at == sorted(at)
    assert body.splitlines()[0] == "Food Optimizer for Windows ({{exe_mb}} MB)"
    smart = next(i for i, line in enumerate(lines) if line.startswith("Windows may not recognize"))
    assert at[0] < smart < at[1]


def test_email_smartscreen_line_is_start_heres():
    body = " ".join(_email_body(EMAIL.read_text(encoding="utf-8")).split())
    assert _start_here_smartscreen() in body


def test_email_sender_lines():
    head = rd.COMMENT.match(EMAIL.read_text(encoding="utf-8")).group(0)
    assert "Subject: Food Optimizer {{version}} for Windows" in head
    assert "Attach: {{it_sheet_name}}" in head


def test_email_links_straight_to_the_exe():
    filled = rd.fill(EMAIL.read_text(encoding="utf-8"), SAMPLE, release=False)
    urls = re.findall(r"https?://\S+", filled)
    assert len(urls) == 1, urls
    assert re.fullmatch(r"https://github\.com/thomasat/food_opt/releases/download/v\d+\.\d+\.\d+/"
                        r"FoodOptimizer-\d+\.\d+\.\d+-Windows\.exe", urls[0])
    for word in ("Source code", "zip", "/archive/", "zipball"):
        assert word not in filled, word


def test_email_sentences_follow_the_copy_rules():
    filled = rd.fill(_email_body(EMAIL.read_text(encoding="utf-8")), SAMPLE, release=False)
    filled = re.sub(r"\[pending: [^\]]*\]", "", filled)
    exempt = _start_here_smartscreen()
    text = " ".join(filled.split()).replace(exempt, "")
    for sentence in re.split(r"(?<=[.!?])\s+", text):
        if sentence.startswith("https://"):
            continue
        assert len(sentence.split()) <= 10, sentence


def test_email_pending_markers_are_the_known_set():
    assert {m.group(1) for m in rd.PENDING.finditer(EMAIL.read_text(encoding="utf-8"))} == {
        "exact Edge download words"}


def test_render_writes_the_email_with_the_sender_lines_on_top(tmp_path):
    out = tmp_path / "out"
    rd.render("0.9.0", out=out, templates={"email": EMAIL.read_text(encoding="utf-8")})
    text = (out / "download-email-0.9.0.txt").read_text(encoding="utf-8")
    lines = text.splitlines()
    assert lines[:3] == ["Subject: Food Optimizer 0.9.0 for Windows", "Attach: FoodOptimizer-0.9.0-IT-sheet.pdf", ""]
    assert lines[3] == "Food Optimizer for Windows ([pending: exe_mb] MB)"
    assert "<!--" not in text


# ---------- the release notes and the root README (Task 8) ----------

NOTES = REPO / "desktop" / "release_notes" / f"{VERSION}.md"
README = REPO / "README.md"
UPDATE_LINE = "**Updating from an earlier version?** Close the app, run the new installer. Your projects are kept."


def test_notes_say_what_changed_in_three_lines():
    text = NOTES.read_text(encoding="utf-8")
    section = text[text.index("## What is new in"):]
    section = section[:section.index("\n## ", 5)]
    bullets = [line[2:] for line in section.splitlines() if line.startswith("- ")]
    assert len(bullets) == 3, bullets
    for bullet in bullets:
        for sentence in re.split(r"(?<=[.!?])\s+", bullet.strip()):
            assert len(sentence.split()) <= 15, sentence
    assert any("0.8.0" in b and "differ" in b for b in bullets), bullets


def test_notes_have_the_update_line_and_both_downloads():
    text = NOTES.read_text(encoding="utf-8")
    assert UPDATE_LINE in text
    assert "{{dmg_sha256}}" in text and "{{exe_sha256}}" in text
    a = rd.assets(VERSION)
    for name in (a["dmg"], a["exe"]):
        assert f"{rd.URL_BASE}/v{VERSION}/{name}" in text, name


def test_notes_point_support_to_email():
    text = NOTES.read_text(encoding="utf-8")
    assert "sohum.patnaik@fsi.org" in text
    assert "github.com/thomasat/food_opt/issues" not in text


def test_readme_and_notes_pending_markers():
    for path in (README, NOTES):
        assert {m.group(1) for m in rd.PENDING.finditer(path.read_text(encoding="utf-8"))} == {
            "exact SmartScreen words on a signed build"}, path.name


def test_notes_and_readme_carry_start_heres_smartscreen_line():
    smart = _start_here_smartscreen()
    for path in (README, NOTES):
        assert smart in " ".join(path.read_text(encoding="utf-8").split()), path.name


def test_notes_fields_are_known_values():
    fields = set(rd.FIELD.findall(NOTES.read_text(encoding="utf-8")))
    assert fields and fields <= rd.VALUE_KEYS
