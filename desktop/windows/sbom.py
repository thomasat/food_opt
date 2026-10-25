"""The Windows release's software bill of materials (CycloneDX 1.6 JSON).

build_installer.ps1 runs it in the build VM under the image's own Python,
as `python.exe -I -X utf8 -B sbom.py build ...`, after the size gate. It
reads what the image really ships: every *.dist-info in site-packages (each
checked against the Windows lock), the shell's FoodOptimizer.deps.json with
each NuGet package's .nuspec, and sbom_declared.json for what has no package
metadata (CPython, Microsoft's C++ runtime, the runtime libraries inside
CPython, Inno Setup). Every licence comes from that metadata, in a fixed
order, and an unknown one fails the build. Every GPL-family mention in a
shipped licence text is counted and listed, never hidden. Each unsigned
binary (the ones FSI re-signs) is attributed to exactly one component.

Stdlib only, one file: `-I` leaves this folder off sys.path.

    python sbom.py build <image> <lock> <out.json> --version V --commit C
        --unsigned FILE --signed yes|no --inno V [--nuget-root DIR]...
    python sbom.py survey <site-packages or wheel folder> <lock>

build prints sbom_components=, sbom_gpl_licensed=, sbom_gpl_mentions= and
sbom_sha256= lines, or one error line and exits 1. survey prints one TSV row
per package (name, version, ids, source, families, GPL mentions) and never
fails: an unknown licence prints UNKNOWN.
"""
import argparse
import csv
import email.message
import email.parser
import hashlib
import io
import json
import pathlib
import re
import sys
import uuid
import zipfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

HERE = pathlib.Path(__file__).resolve().parent
DECLARED = HERE / "sbom_declared.json"
SITE = "python/Lib/site-packages"
SUPPLIER = "Food System Innovations"
APP_REF = "app:food-optimizer"

LICENCE_FILE = r"(?i)(^|/)(licen[cs]e|copying|notice|third[-_]?party)[^/]*$|\.libs/[^/]*\.txt$"
GPL_FAMILY = r"\b(?:A|L)?GPL(?:v?\d|-\d|\b)|GNU (?:Lesser |Library |Affero )?General Public Licen[cs]e"
_LICENCE_FILE = re.compile(LICENCE_FILE)
_GPL = re.compile(GPL_FAMILY, re.I)

# Only classifiers that name one licence exactly. "BSD License" (2- or
# 3-clause?) and "Other/Proprietary License" are left out on purpose: they
# become a name such as "BSD (classifier)", with no SPDX id.
CLASSIFIER_SPDX = {
    "License :: OSI Approved :: MIT License": "MIT",
    "License :: OSI Approved :: Apache Software License": "Apache-2.0",
    "License :: OSI Approved :: Mozilla Public License 2.0 (MPL 2.0)": "MPL-2.0",
    "License :: OSI Approved :: Python Software Foundation License": "PSF-2.0",
    "License :: OSI Approved :: ISC License (ISCL)": "ISC",
    "License :: OSI Approved :: The Unlicense (Unlicense)": "Unlicense",
}

# The SPDX ids a free-text License: field may be taken as (a bare "BSD"
# fits the id pattern but is no SPDX id, so the list is explicit).
KNOWN_SPDX = frozenset({
    "0BSD", "Apache-2.0", "BSD-2-Clause", "BSD-3-Clause", "CC0-1.0", "HPND", "ISC",
    "MIT", "MIT-0", "MIT-CMU", "MPL-2.0", "PSF-2.0", "Python-2.0", "Unlicense", "Zlib",
    "blessing", "GPL-2.0-only", "GPL-2.0-or-later", "GPL-3.0-only", "GPL-3.0-or-later",
    "LGPL-2.1-only", "LGPL-2.1-or-later", "LGPL-3.0-only", "LGPL-3.0-or-later",
    "AGPL-3.0-only", "AGPL-3.0-or-later",
})
_ID = r"(?:LicenseRef-[A-Za-z0-9.-]+|[A-Za-z0-9][A-Za-z0-9.+-]*)"
SPDX_ID = re.compile(r"^\(*%s\)*(?:\s+(?:AND|OR|WITH)\s+\(*%s\)*)*$" % (_ID, _ID))

FAMILIES = ("MIT", "BSD", "Apache-2.0", "MPL-2.0", "PSF-2.0", "ISC", "Unlicense",
            "GPL-family", "Microsoft", "Intel", "Inno Setup", "Other")
# A short free-text licence name's family, by keyword (first match wins).
_NAME_FAMILY = (
    (_GPL, "GPL-family"),
    (re.compile(r"\bBSD\b"), "BSD"),
    (re.compile(r"\bMIT\b"), "MIT"),
    (re.compile(r"\bApache\b"), "Apache-2.0"),
    (re.compile(r"\bMPL\b|\bMozilla\b"), "MPL-2.0"),
    (re.compile(r"\bPSF\b|Python Software Foundation"), "PSF-2.0"),
    (re.compile(r"\bISC\b"), "ISC"),
    (re.compile(r"\bUnlicense\b"), "Unlicense"),
    (re.compile(r"\bIntel\b"), "Intel"),
    (re.compile(r"\bMicrosoft\b"), "Microsoft"),
    (re.compile(r"\bInno Setup\b"), "Inno Setup"),
)
SHORT_NAME = 60


class UnknownLicence(ValueError):
    pass


def normalize(name: str) -> str:
    """PEP 503."""
    return re.sub(r"[-_.]+", "-", name).lower()


# ---------- the lock ----------

@dataclass(frozen=True)
class LockEntry:
    name: str
    version: str
    hashes: tuple[str, ...]


def parse_lock(path) -> dict[str, LockEntry]:
    """Every `name==version` requirement of a uv lock with its hashes,
    keyed by normalize(name)."""
    entries: dict[str, list] = {}
    current = None
    for line in pathlib.Path(path).read_text(encoding="utf-8").splitlines():
        m = re.match(r"^([A-Za-z0-9][A-Za-z0-9._-]*)==([^\s;\\]+)", line)
        if m:
            current = normalize(m.group(1))
            entries[current] = [m.group(1), m.group(2), []]
        if current is not None:
            entries[current][2] += re.findall(r"--hash=sha256:([0-9a-f]+)", line)
    return {key: LockEntry(name, version, tuple(hashes))
            for key, (name, version, hashes) in entries.items()}


# ---------- the installed (or wheeled) distributions ----------

@dataclass(frozen=True)
class Dist:
    name: str
    version: str
    metadata: email.message.Message
    texts: dict[str, str]
    files: tuple[str, ...]


def _record_paths(raw: bytes) -> tuple[str, ...]:
    rows = csv.reader(io.StringIO(raw.decode("utf-8")))
    return tuple(row[0] for row in rows if row and row[0])


def _is_licence(rel: str, info: str) -> bool:
    return bool(_LICENCE_FILE.search(rel)) or rel.startswith(f"{info}/licenses/")


def _decode(raw: bytes) -> str:
    return raw.decode("utf-8", errors="replace")


def _dist(meta_raw: bytes, record_raw: bytes, info: str, read) -> Dist:
    # Parsed as text: a non-ASCII header (an author's name, a licence
    # text) parsed from bytes comes back as an email Header, not a str.
    meta = email.parser.Parser().parsestr(_decode(meta_raw))
    files = _record_paths(record_raw)
    texts = {}
    for rel in files:
        if _is_licence(rel, info):
            raw = read(rel)
            if raw is not None:
                texts[rel] = _decode(raw)
    return Dist(meta["Name"], meta["Version"], meta, texts, files)


def read_dists(source) -> dict[str, Dist]:
    """Every distribution in a site-packages folder (its *.dist-info
    folders) or in a folder of *.whl files (read with zipfile, never
    installed), keyed by normalize(name)."""
    source = pathlib.Path(source)
    out = {}
    wheels = sorted(source.glob("*.whl"))
    if wheels:
        for whl in wheels:
            with zipfile.ZipFile(whl) as z:
                names = set(z.namelist())
                infos = sorted({n.split("/")[0] for n in names
                                if n.count("/") == 1 and n.endswith(".dist-info/METADATA")})
                if len(infos) != 1:
                    raise ValueError(f"sbom: {whl.name} has {len(infos)} .dist-info folders")
                info = infos[0]
                read = lambda rel, z=z, names=names: z.read(rel) if rel in names else None
                d = _dist(z.read(f"{info}/METADATA"), z.read(f"{info}/RECORD"), info, read)
            out[normalize(d.name)] = d
        return out
    for info_dir in sorted(source.glob("*.dist-info")):
        meta, record = info_dir / "METADATA", info_dir / "RECORD"
        if not meta.is_file() or not record.is_file():
            raise ValueError(f"sbom: {info_dir.name} has no METADATA or RECORD")

        def read(rel, root=source):
            path = root / rel
            return path.read_bytes() if path.is_file() else None
        d = _dist(meta.read_bytes(), record.read_bytes(), info_dir.name, read)
        out[normalize(d.name)] = d
    return out


# ---------- licences ----------

@dataclass(frozen=True)
class Licence:
    ids: tuple[str, ...]
    expression: str | None
    name: str | None
    source: str


def spdx_ids(expression: str) -> tuple[str, ...]:
    """The licence ids of an SPDX expression: split on AND/OR/WITH and
    parentheses; the exception after WITH is not a licence."""
    tokens = re.findall(r"[A-Za-z0-9][A-Za-z0-9.:+-]*", expression)
    ids, skip = [], False
    for token in tokens:
        if token.upper() in ("AND", "OR"):
            continue
        if token.upper() == "WITH":
            skip = True
            continue
        if skip:
            skip = False
            continue
        if token not in ids:
            ids.append(token)
    return tuple(ids)


def _compound(expression: str) -> str | None:
    """The expression itself, unless it is one bare id."""
    return None if re.fullmatch(_ID, expression.strip()) else expression.strip()


def _known_expression(text: str) -> bool:
    if not SPDX_ID.match(text):
        return False
    return all(i in KNOWN_SPDX or i.startswith("LicenseRef-") for i in spdx_ids(text))


def _from_text(text: str, source: str) -> Licence | None:
    """An SPDX id or expression, or a short name with a family keyword."""
    text = text.strip()
    if not text or "\n" in text:
        return None
    if _known_expression(text):
        return Licence(spdx_ids(text), _compound(text), None, source)
    if len(text) <= SHORT_NAME and _name_family(text) != "Other":
        return Licence((), None, text, source)
    return None


def _name_family(name: str) -> str:
    for pattern, fam in _NAME_FAMILY:
        if pattern.search(name):
            return fam
    return "Other"


def licence_of(dist: Dist, overrides: Mapping[str, dict]) -> Licence:
    """The first of: License-Expression; an override (with evidence) for
    the name; the License classifiers, when every one names one licence;
    a License: field that is an SPDX id or expression; the classifiers
    with the ambiguous ones as names; a short License: name with a family
    keyword. Otherwise UnknownLicence."""
    meta = dist.metadata
    expression = (meta.get("License-Expression") or "").strip()
    if expression:
        return Licence(spdx_ids(expression), _compound(expression), None, "License-Expression")
    override = overrides.get(normalize(dist.name))
    if override is not None:
        lic = _from_text(override["licence"], "override")
        if lic is None:
            lic = Licence((), None, override["licence"], "override")
        return lic
    classifiers = [c.strip() for c in meta.get_all("Classifier") or [] if c.strip().startswith("License ::")]
    exact = [CLASSIFIER_SPDX[c] for c in classifiers if c in CLASSIFIER_SPDX]
    vague = [c for c in classifiers if c not in CLASSIFIER_SPDX]
    if classifiers and not vague:
        ids = tuple(dict.fromkeys(exact))
        return Licence(ids, None, None, "classifier")
    field = meta.get("License") or ""
    if field.strip() and _known_expression(field.strip()):
        return _from_text(field, "License")
    if vague:
        names = []
        for c in vague:
            label = c.split("::")[-1].strip()
            label = re.sub(r"\s+License$", "", label)
            names.append(f"{label} (classifier)")
        return Licence(tuple(dict.fromkeys(exact)), None, "; ".join(names), "classifier")
    lic = _from_text(field, "License")
    if lic is not None:
        return lic
    raise UnknownLicence(f"no licence for {dist.name} {dist.version}")


def _id_family(spdx: str) -> str:
    if spdx == "MIT" or spdx.startswith("MIT-"):
        return "MIT"
    if spdx.startswith("BSD-") or spdx == "0BSD":
        return "BSD"
    if spdx.startswith("Apache-"):
        return "Apache-2.0"
    if spdx.startswith("MPL-"):
        return "MPL-2.0"
    if spdx in ("PSF-2.0", "Python-2.0"):
        return "PSF-2.0"
    if spdx in ("ISC", "Unlicense"):
        return spdx
    if re.match(r"^(A|L)?GPL-", spdx):
        return "GPL-family"
    if spdx.startswith("LicenseRef-Microsoft"):
        return "Microsoft"
    if spdx.startswith("LicenseRef-Intel"):
        return "Intel"
    if spdx == "LicenseRef-Inno-Setup":
        return "Inno Setup"
    return "Other"


def family(lic: Licence) -> list[str]:
    """The licence families a licence belongs to, each once, in order."""
    out = [_id_family(i) for i in lic.ids]
    if lic.name:
        out += [_name_family(part) for part in lic.name.split("; ")]
    return list(dict.fromkeys(out))


def gpl_mentions(dist: Dist) -> list[tuple[str, int, str]]:
    """(file, line number, stripped line) of every GPL-family mention in the
    dist's shipped licence texts."""
    return _mentions(dist.texts)


def _mentions(texts: Mapping[str, str]) -> list[tuple[str, int, str]]:
    out = []
    for rel in sorted(texts):
        for i, line in enumerate(texts[rel].splitlines(), 1):
            if _GPL.search(line):
                out.append((rel, i, line.strip()))
    return out


# ---------- NuGet packages (the shell) ----------

def _nuspec_licence(path: pathlib.Path, pkg_id: str, version: str) -> Licence:
    import xml.etree.ElementTree as ET
    for el in ET.parse(path).getroot().iter():
        if el.tag == "license" or el.tag.endswith("}license"):
            kind, text = el.get("type"), (el.text or "").strip()
            if kind == "expression" and text:
                return Licence(spdx_ids(text), _compound(text), None, "nuspec")
            if kind == "file":
                return Licence((f"LicenseRef-{pkg_id}",), None, None, "nuspec")
    raise UnknownLicence(f"no licence for {pkg_id} {version} (its nuspec has no <license>)")


def nuget_components(deps_json, nuget_roots: Sequence) -> list[dict]:
    """The NuGet packages and runtime packs the shell's deps.json names,
    each with its licence (from <root>/<id>/<version>/<id>.nuspec, the first
    root that has it) and the file names it put in the flat publish."""
    deps = json.loads(pathlib.Path(deps_json).read_text(encoding="utf-8-sig"))
    assets: dict[str, list[str]] = {}
    for target in deps.get("targets", {}).values():
        for key, entry in target.items():
            names = assets.setdefault(key, [])
            for group in ("runtime", "native"):
                names += [rel.split("/")[-1] for rel in entry.get(group, {})]
            for rel, info in entry.get("resources", {}).items():
                names.append(f"{info.get('locale', '')}/{rel.split('/')[-1]}".lstrip("/"))
    out = []
    for key, lib in sorted(deps.get("libraries", {}).items()):
        kind = lib.get("type")
        if kind not in ("package", "runtimepack"):
            continue
        pkg_id, version = key.rsplit("/", 1)
        if kind == "runtimepack" and pkg_id.startswith("runtimepack."):
            pkg_id = pkg_id[len("runtimepack."):]
        low = pkg_id.lower()
        nuspec = next((p for p in (pathlib.Path(r) / low / version / f"{low}.nuspec" for r in nuget_roots)
                       if p.is_file()), None)
        if nuspec is None:
            raise ValueError(f"no nuspec for {pkg_id} {version}")
        out.append({"id": pkg_id, "version": version, "type": kind,
                    "licence": _nuspec_licence(nuspec, pkg_id, version),
                    "assets": sorted(set(assets.get(key, [])))})
    return out


# ---------- what the image's Python itself carries ----------

def runtime_versions() -> dict[str, str]:
    """The versions of the libraries built into this Python (the image's,
    when the build runs it), which CVE scanners match on."""
    import pyexpat
    import sqlite3
    import ssl
    import zlib
    openssl = re.search(r"OpenSSL (\S+)", ssl.OPENSSL_VERSION)
    return {"OpenSSL": openssl.group(1) if openssl else ssl.OPENSSL_VERSION,
            "SQLite": sqlite3.sqlite_version,
            "zlib": zlib.ZLIB_RUNTIME_VERSION,
            "Expat": pyexpat.EXPAT_VERSION.split("_")[-1]}


# ---------- attribution of the re-signed files ----------

def _pypi_ref(d: Dist) -> str:
    return f"pypi:{normalize(d.name)}@{d.version}"


def _nuget_ref(c: dict) -> str:
    return f"nuget:{c['id']}@{c['version']}"


def attribute(unsigned: Sequence[str], dists: Mapping[str, Dist], nuget: Sequence[dict],
              image_root_files: Sequence[str], *, app_ref: str = APP_REF) -> dict[str, list[str]]:
    """bom-ref -> the image-relative unsigned paths it owns. A site-packages
    path belongs to the dist whose RECORD lists it; a root-level file to the
    NuGet package whose deps.json assets name it, else (when it is one of
    the image's root files) to the app. Anything else is an error."""
    record = {}
    for d in dists.values():
        for rel in d.files:
            record[f"{SITE}/{rel}".casefold()] = _pypi_ref(d)
    assets = {}
    for c in nuget:
        for name in c["assets"]:
            assets[name.casefold()] = _nuget_ref(c)
    roots = {name.casefold() for name in image_root_files}
    out: dict[str, list[str]] = {}
    for path in unsigned:
        key = path.replace("\\", "/").casefold()
        owner = record.get(key) or assets.get(key)
        if owner is None and "/" not in key and key in roots:
            owner = app_ref
        if owner is None:
            raise ValueError(f"unattributed: {path}")
        out.setdefault(owner, []).append(path)
    return out


# ---------- the document ----------

def _pins(declared: pathlib.Path) -> dict:
    return json.loads((declared.parent / "pins.json").read_text(encoding="utf-8"))


def _licenses(lic: Licence) -> list[dict]:
    """CycloneDX: one expression, or one entry per SPDX id and per name
    (a LicenseRef is not an SPDX id, so it goes in an expression)."""
    if lic.expression or (len(lic.ids) == 1 and not lic.name and lic.ids[0].startswith("LicenseRef-")):
        return [{"expression": lic.expression or lic.ids[0]}]
    out = [{"license": {"id": i}} for i in lic.ids]
    out += [{"license": {"name": n}} for n in (lic.name.split("; ") if lic.name else [])]
    return out


def _parse_given(text: str, source: str) -> Licence:
    if _known_expression(text.strip()):
        return Licence(spdx_ids(text), _compound(text), None, source)
    return Licence((), None, text.strip(), source)


def _check_evidence(image: pathlib.Path, evidence: Mapping) -> None:
    for rel in ([evidence["file"]] if "file" in evidence else []) + list(evidence.get("files", [])):
        if not (image / rel).is_file():
            raise ValueError(f"evidence missing: {rel}")
    phrase = evidence.get("contains")
    if phrase is not None:
        text = _decode((image / evidence["file"]).read_bytes())
        if phrase not in text:
            raise ValueError(f"evidence phrase not in {evidence['file']}: {phrase}")


def _component(kind, ref, name, version, lic, *, purl=None, signing="vendor", resigned=0,
               extra=(), mentions=()) -> dict:
    props = [("foodopt:licence_source", lic.source), ("foodopt:licence_family", ",".join(family(lic))),
             ("foodopt:signing", signing), ("foodopt:resigned_files", str(resigned)), *extra]
    if mentions:
        props.append(("foodopt:gpl_mentions", str(len(mentions))))
        props += [("foodopt:gpl_mention", f"{f}:{n}: {line}") for f, n, line in mentions]
    c = {"type": kind, "bom-ref": ref, "name": name, "version": version, "licenses": _licenses(lic),
         "properties": [{"name": k, "value": v} for k, v in props]}
    if purl:
        c["purl"] = purl
    return c


def _version_from(spec: str, *, pins, inno, runtime, dists) -> str:
    kind, _, key = spec.partition(":")
    if kind == "pins":
        value = pins
        for part in key.split("."):
            value = value[part]
        return str(value)
    if kind == "arg" and key == "inno":
        return inno
    if kind == "runtime":
        return runtime[key]
    if kind == "dist" and normalize(key) in dists:
        return dists[normalize(key)].version
    raise ValueError(f"cannot resolve version_from {spec}")


def build(image, lock, declared, *, version: str, commit: str, unsigned: Sequence[str], signed: bool,
          inno: str, nuget_roots: Sequence = (), runtime: Mapping[str, str] | None = None) -> dict:
    """The CycloneDX 1.6 document for this image. Raises ValueError("sbom: ...")
    on any mismatch with the lock, an unknown licence, missing evidence or
    an unsigned file no component owns."""
    try:
        return _build(pathlib.Path(image), pathlib.Path(lock), pathlib.Path(declared), version=version,
                      commit=commit, unsigned=unsigned, signed=signed, inno=inno,
                      nuget_roots=nuget_roots, runtime=runtime)
    except ValueError as e:
        message = str(e)
        raise ValueError(message if message.startswith("sbom: ") else f"sbom: {message}") from None


def _build(image, lock, declared, *, version, commit, unsigned, signed, inno, nuget_roots, runtime):
    locked = parse_lock(lock)
    data = json.loads(declared.read_text(encoding="utf-8"))
    overrides = {normalize(k): v for k, v in data.get("licence_overrides", {}).items()}
    dists = read_dists(image / SITE)
    for key, entry in sorted(locked.items()):
        if key not in dists:
            raise ValueError(f"no metadata for {entry.name} {entry.version} in the image")
        if dists[key].version != entry.version:
            raise ValueError(f"{dists[key].name} {dists[key].version} in the image: the lock's version "
                             f"is {entry.version}")
    for key in sorted(set(dists) - set(locked)):
        raise ValueError(f"{dists[key].name} {dists[key].version} is not in the lock")
    for key, o in sorted(overrides.items()):
        if key in dists:
            _check_evidence(image, o["evidence"])
    deps = image / "FoodOptimizer.deps.json"
    nuget = nuget_components(deps, nuget_roots) if deps.is_file() else []
    app_ref = f"{APP_REF}@{version}"
    root_files = sorted(p.name for p in image.iterdir() if p.is_file())
    owned = attribute(list(unsigned), dists, nuget, root_files, app_ref=app_ref)
    word = "re-signed by Food System Innovations" if signed else "unsigned (internal build)"

    def signing(ref):
        n = len(owned.get(ref, ()))
        return (word if n else "vendor"), n

    components = []
    for key, d in sorted(dists.items()):
        ref = _pypi_ref(d)
        s, n = signing(ref)
        components.append(_component(
            "library", ref, d.name, d.version, licence_of(d, overrides),
            purl=f"pkg:pypi/{normalize(d.name)}@{d.version}", signing=s, resigned=n,
            extra=[("foodopt:lock_hashes", ",".join(locked[key].hashes))], mentions=gpl_mentions(d)))
    for c in nuget:
        ref = _nuget_ref(c)
        s, n = signing(ref)
        components.append(_component(
            "framework" if c["type"] == "runtimepack" else "library", ref, c["id"], c["version"],
            c["licence"], purl=f"pkg:nuget/{c['id']}@{c['version']}", signing=s, resigned=n))
    pins = _pins(declared)
    runtime = dict(runtime_versions() if runtime is None else runtime)
    for entry in data["components"]:
        _check_evidence(image, entry["evidence"])
        v = _version_from(entry["version_from"], pins=pins, inno=inno, runtime=runtime, dists=dists)
        ev = entry["evidence"]
        files = [ev["file"]] if "file" in ev else list(ev["files"])
        texts = {f: _decode((image / f).read_bytes()) for f in files if _LICENCE_FILE.search(f)}
        name = entry["ref"].split(":", 1)[-1]
        components.append(_component(
            entry["type"], f"{entry['ref']}@{v}", entry["name"], v, _parse_given(entry["licence"], "declared"),
            purl=f"pkg:generic/{name}@{v}", extra=[("foodopt:evidence", ", ".join(files) or "(none)")],
            mentions=_mentions(texts)))
    licence_text = (HERE.parents[1] / "LICENSE").read_text(encoding="utf-8")
    if not licence_text.startswith("MIT License"):
        raise ValueError("the repo's LICENSE is not the MIT licence")
    s, n = signing(app_ref)
    app = _component("application", app_ref, "Food Optimizer", version, Licence(("MIT",), None, None, "LICENSE"),
                     signing=s, resigned=n)
    app["supplier"] = {"name": SUPPLIER}
    components.sort(key=lambda c: c["bom-ref"])
    refs = [c["bom-ref"] for c in components]
    if len(refs) != len(set(refs)):
        raise ValueError("two components share a bom-ref")
    return {
        "bomFormat": "CycloneDX",
        "specVersion": "1.6",
        "version": 1,
        "serialNumber": "urn:uuid:" + str(uuid.uuid5(uuid.NAMESPACE_URL, f"food-optimizer/{version}/{commit}")),
        "metadata": {"component": app, "supplier": {"name": SUPPLIER},
                     "properties": [{"name": "foodopt:commit", "value": commit}]},
        "components": components,
    }


def dumps(bom: dict) -> bytes:
    return (json.dumps(bom, indent=1, sort_keys=True, ensure_ascii=False) + "\n").encode("utf-8")


def summary(bom: dict) -> dict:
    """What the IT sheet states, read back from the document."""
    def props(c, name):
        return [p["value"] for p in c.get("properties", []) if p["name"] == name]
    families: dict[str, int] = {}
    gpl_licensed, mentions, declared = [], [], []
    for c in bom["components"]:
        fams = [f for f in ",".join(props(c, "foodopt:licence_family")).split(",") if f]
        for f in fams:
            families[f] = families.get(f, 0) + 1
        if "GPL-family" in fams:
            gpl_licensed.append(c["bom-ref"])
        if props(c, "foodopt:licence_source") == ["declared"]:
            declared.append(c["bom-ref"])
        for value in props(c, "foodopt:gpl_mention"):
            m = re.match(r"^(.*?):(\d+): ", value)
            mentions.append({"name": c["name"], "version": c["version"], "file": m.group(1), "line": int(m.group(2))})
    resigned = sum(int(v) for c in [bom["metadata"]["component"], *bom["components"]]
                   for v in props(c, "foodopt:resigned_files"))
    return {"components": len(bom["components"]), "families": families, "gpl_licensed": gpl_licensed,
            "gpl_mentions": mentions, "declared": declared, "resigned_files": resigned}


# ---------- CLI ----------

def _overrides(declared) -> dict:
    data = json.loads(pathlib.Path(declared).read_text(encoding="utf-8"))
    return {normalize(k): v for k, v in data.get("licence_overrides", {}).items()}


def _survey(source, lock, declared) -> None:
    dists = read_dists(source)
    locked = parse_lock(lock)
    overrides = _overrides(declared) if pathlib.Path(declared).is_file() else {}
    print("name\tversion\tids\tsource\tfamilies\tgpl_mentions")
    for key in sorted(set(dists) | set(locked)):
        if key not in dists:
            print(f"{locked[key].name}\t{locked[key].version}\tMISSING\t-\t-\t-")
            continue
        d = dists[key]
        try:
            lic = licence_of(d, overrides)
            ids = " ".join(lic.ids) or "-"
            if lic.name:
                ids = f"{ids} [{lic.name}]" if lic.ids else f"[{lic.name}]"
            row = [ids, lic.source, ",".join(family(lic))]
        except UnknownLicence:
            row = ["UNKNOWN", "UNKNOWN", "-"]
        print("\t".join([d.name, d.version, *row, str(len(gpl_mentions(d)))]))


def _build_cli(args) -> int:
    unsigned = [line for line in pathlib.Path(args.unsigned).read_text(encoding="utf-8").splitlines() if line]
    try:
        bom = build(args.image, args.lock, DECLARED, version=args.version, commit=args.commit,
                    unsigned=unsigned, signed=args.signed == "yes", inno=args.inno,
                    nuget_roots=[pathlib.Path(r) for r in args.nuget_root])
    except ValueError as e:
        print(str(e))
        print(str(e), file=sys.stderr)
        return 1
    raw = dumps(bom)
    out = pathlib.Path(args.out)
    out.write_bytes(raw)
    s = summary(bom)
    print(f"sbom_components={s['components']}")
    print(f"sbom_gpl_licensed={len(s['gpl_licensed'])}")
    print(f"sbom_gpl_mentions={len(s['gpl_mentions'])}")
    print(f"sbom_sha256={hashlib.sha256(raw).hexdigest()}")
    return 0


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(prog="sbom.py")
    sub = parser.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    for name in ("image", "lock", "out"):
        b.add_argument(name)
    b.add_argument("--version", required=True)
    b.add_argument("--commit", required=True)
    b.add_argument("--unsigned", required=True)
    b.add_argument("--signed", required=True, choices=("yes", "no"))
    b.add_argument("--inno", required=True)
    b.add_argument("--nuget-root", action="append", default=[])
    survey = sub.add_parser("survey")
    survey.add_argument("source")
    survey.add_argument("lock")
    args = parser.parse_args(argv)
    if args.cmd == "build":
        return _build_cli(args)
    try:
        _survey(args.source, args.lock, DECLARED)
    except Exception as e:  # survey never fails: it is how we look
        print(f"survey: {e}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
