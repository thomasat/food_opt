"""Persistence backends for FoodOptimizer projects.

LocalStorage reproduces the historical file behavior exactly (JSON content in
<name>.pkl files in the working directory) and is the default everywhere —
including the desktop app, which must never touch the network. SupabaseStorage
(added in a later task) stores one row per project, scoped to a user.

This module must not import streamlit, and imports supabase only lazily.
"""
import glob
import json
import os
import re
import shutil


ARCHIVE_SUFFIX_RE = re.compile(
    r"_(archived|deleted|pre_rewind|pre_restore|pre_delete|pre_edit|pre_undo)(_\d+)?$"
)


def is_archive_name(name):
    return ARCHIVE_SUFFIX_RE.search(name) is not None


class StorageError(Exception):
    """A storage backend failure (network, API, write conflict).

    NOT raised for "project does not exist" — load() returns None for that.
    Messages are shown to end users, so keep them plain-language.
    """


class LocalStorage:
    persist_empty_on_init = False  # nothing on disk until the user creates or edits
    persist_after_load = True      # re-save after load only when the file is behind the
                                   # current CLASS_VERSION (see FoodOptimizer.load); an
                                   # up-to-date file is left untouched

    _CONFLICT = (
        "This project was changed in another window or tab. Click Reload "
        "project before continuing — changes made here were NOT saved."
    )

    def __init__(self):
        self._seen = {}  # project name -> st_mtime_ns when we last read or wrote it

    def _path(self, name):
        return f"{name}.pkl"

    def _stamp(self, name):
        try:
            return os.stat(self._path(name)).st_mtime_ns
        except FileNotFoundError:
            return None

    def list_projects(self):
        names = sorted(os.path.splitext(f)[0] for f in glob.glob("*.pkl"))
        return [n for n in names if not is_archive_name(n)]

    def list_archives(self):
        names = sorted(os.path.splitext(f)[0] for f in glob.glob("*.pkl"))
        return [n for n in names if is_archive_name(n)]

    def most_recent_project(self):
        projects = self.list_projects()
        if not projects:
            return None
        return max(projects, key=lambda n: os.stat(self._path(n)).st_mtime_ns)

    def exists(self, name):
        return os.path.exists(self._path(name))

    def load(self, name):
        seen = self.__dict__.setdefault("_seen", {})
        try:
            with open(self._path(name), 'rb') as f:
                raw = f.read()
        except FileNotFoundError:
            return None
        except OSError:
            raise StorageError(
                "This project file could not be opened. It may have been "
                "moved or deleted."
            )
        seen[name] = self._stamp(name)
        try:
            return json.loads(raw.decode('utf-8'))
        except (ValueError, UnicodeDecodeError):
            # No sniffing: a damaged file and a legacy pickle both start with
            # arbitrary bytes (the existing "damaged" tests use a pickle
            # header on purpose), so one message covers both cases.
            raise StorageError(
                "This project file is damaged, or was saved by an early version "
                "of Food Optimizer, and could not be opened. If you have a "
                "backup, use Restore from backup; otherwise check the "
                "FoodOptimizer > backups folder in your home folder for a "
                "recent copy. An early-version file can be converted by opening "
                "it in the version of Food Optimizer that created it and "
                "downloading a backup."
            )

    def save(self, name, state):
        seen = self.__dict__.setdefault("_seen", {})
        known = seen.get(name)
        current = self._stamp(name)
        if known is not None and current is not None and current != known:
            raise StorageError(self._CONFLICT)
        # Write-then-rename so a crash mid-write can't corrupt the file.
        # Local I/O errors propagate raw (never StorageError): tests assert
        # the RuntimeError path, and a local disk failure should be loud.
        data = json.dumps(state, indent=2)
        path = self._path(name)
        tmp = f"{path}.tmp"
        try:
            with open(tmp, 'w', encoding='utf-8') as f:
                f.write(data)
            os.replace(tmp, path)
        finally:
            if os.path.exists(tmp):
                os.remove(tmp)
        seen[name] = self._stamp(name)

    def is_stale(self, name):
        """True when the file changed on disk since this instance last read or wrote it."""
        seen = self.__dict__.setdefault("_seen", {})
        known = seen.get(name)
        current = self._stamp(name)
        return known is not None and current is not None and current != known

    def archive(self, name, label, copy=False):
        seen = self.__dict__.setdefault("_seen", {})
        path = self._path(name)
        if not os.path.exists(path):
            return None
        archive_name = f"{name}_{label}"
        counter = 1
        while os.path.exists(self._path(archive_name)):
            archive_name = f"{name}_{label}_{counter}"
            counter += 1
        if copy:
            shutil.copy2(path, self._path(archive_name))
        else:
            os.rename(path, self._path(archive_name))
            seen.pop(name, None)
        return archive_name
