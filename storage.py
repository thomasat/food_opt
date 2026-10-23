"""Persistence backends for FoodOptimizer projects.

LocalStorage reproduces the historical file behavior exactly (JSON content in
<name>.pkl files in the working directory) and is the default everywhere —
including the desktop app, which must never touch the network. SupabaseStorage
(added in a later task) stores one row per project, scoped to a user.

This module must not import streamlit, and imports supabase only lazily.
"""
import errno
import glob
import json
import os
import re
import shutil
import time
from datetime import datetime

import wording


# pre_rewind and pre_undo are no longer written — the acts they named are
# now Delete, which archives as pre_delete — but a project archived by an
# earlier version still wears them, and a copy that stops reading as a copy
# would reappear in the project list. sample_reset is Start this sample
# over's copy (sample_projects.restore).
ARCHIVE_SUFFIX_RE = re.compile(
    r"_(archived|deleted|pre_rewind|pre_restore|pre_delete|pre_edit"
    r"|pre_correction|pre_undo|sample_reset)(_\d+)?$"
)


def is_archive_name(name):
    return ARCHIVE_SUFFIX_RE.search(name) is not None


# Device names Windows will not save a file under, with any extension:
# CON.pkl and con.backup.pkl are both refused.
_WINDOWS_RESERVED = ({"CON", "PRN", "AUX", "NUL"}
                     | {f"COM{i}" for i in range(1, 10)}
                     | {f"LPT{i}" for i in range(1, 10)})


def is_reserved_name(name):
    """True when Windows reserves the part of `name` before its first dot."""
    return name.split(".", 1)[0].rstrip(" ").upper() in _WINDOWS_RESERVED


class StorageError(Exception):
    """A storage backend failure (network, API, write conflict).

    NOT raised for "project does not exist" — load() returns None for that.
    Messages are shown to end users, so keep them plain-language.
    """


# What the disk said, in the three words wording._DISK_TROUBLE has a fix
# for; anything else is None and gets the plain "could not be written".
# EDQUOT and ETXTBSY are looked up by name: a platform without one simply
# has no entry for it.
_DISK_CAUSES = {
    code: cause for code, cause in (
        (getattr(errno, "ENOSPC", None), 'full'),
        (getattr(errno, "EDQUOT", None), 'full'),
        (getattr(errno, "EACCES", None), 'read_only'),
        (getattr(errno, "EPERM", None), 'read_only'),
        (getattr(errno, "EROFS", None), 'read_only'),
        (getattr(errno, "EBUSY", None), 'in_use'),
        (getattr(errno, "ETXTBSY", None), 'in_use'),
    ) if code is not None
}

# Windows' sharing and lock violations: another program (antivirus,
# OneDrive, Excel) holds the file. They arrive as EACCES, which alone would
# read as a read-only folder.
_WINDOWS_IN_USE = (32, 33)


def disk_cause(error):
    """'full', 'read_only', 'in_use' or None for an OSError."""
    if getattr(error, "winerror", None) in _WINDOWS_IN_USE:
        return 'in_use'
    return _DISK_CAUSES.get(getattr(error, "errno", None))


def _retry_locked(fn, *args, attempts=5, delay=0.1):
    """fn(*args), tried again while another program holds the file for a
    moment; any other error, or the last try's, is raised."""
    for attempt in range(1, attempts + 1):
        try:
            return fn(*args)
        except PermissionError as e:
            if (attempt == attempts
                    or getattr(e, "winerror", None) not in _WINDOWS_IN_USE):
                raise
            time.sleep(delay * attempt)


class LocalStorage:
    persist_empty_on_init = False  # nothing on disk until the user creates or edits
    persist_after_load = True      # re-save after load only when the file is behind the
                                   # current CLASS_VERSION (see FoodOptimizer.load); an
                                   # up-to-date file is left untouched

    def __init__(self):
        self._seen = {}  # project name -> st_mtime_ns when we last read or wrote it

    def _path(self, name):
        return f"{name}.pkl"

    def folder(self):
        """The folder project files are written into, resolved the way
        _path is: the process's working directory."""
        return os.path.dirname(os.path.abspath(self._path("_")))

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

    # The project the reader last opened, by name, in one line beside the
    # projects. A fresh session reopens it: the newest file is not it when a
    # sample writes two projects at once, or a refresh rewrites one.
    _LAST_OPENED = ".last_opened"

    def remember_opened(self, name):
        """Note `name` as the project last opened. Best effort: a folder
        that refuses the write only loses the reopen."""
        try:
            with open(self._LAST_OPENED, "w", encoding="utf-8") as f:
                f.write(name)
        except OSError:
            pass

    def last_opened(self):
        """The project last opened, if it is still a project on disk."""
        try:
            with open(self._LAST_OPENED, encoding="utf-8") as f:
                name = f.read().strip()
        except (OSError, UnicodeDecodeError):
            return None
        return name if name in self.list_projects() else None

    def exists(self, name):
        return os.path.exists(self._path(name))

    def saved_at(self, name):
        """When the project file was last written, or None if there is no file.
        FoodOptimizer.last_saved_at only records saves made in this session, so
        a user who opens a project and changes nothing has no time to show; the
        file's own modification time is that missing fact."""
        try:
            stamp = os.stat(self._path(name)).st_mtime
        except (FileNotFoundError, OSError):
            return None
        return datetime.fromtimestamp(stamp).astimezone()

    def load(self, name):
        seen = self.__dict__.setdefault("_seen", {})
        try:
            with open(self._path(name), 'rb') as f:
                raw = f.read()
        except FileNotFoundError:
            return None
        except OSError:
            raise StorageError(wording.project_file_locked(name))
        seen[name] = self._stamp(name)
        try:
            return json.loads(raw.decode('utf-8'))
        except (ValueError, UnicodeDecodeError):
            # No sniffing: a damaged file and a legacy pickle both start with
            # arbitrary bytes (the existing "damaged" tests use a pickle
            # header on purpose), so one message covers both cases.
            raise StorageError(wording.project_file_unreadable(name))

    def save(self, name, state):
        seen = self.__dict__.setdefault("_seen", {})
        known = seen.get(name)
        current = self._stamp(name)
        if known is not None and current is not None and current != known:
            raise StorageError(wording.changed_elsewhere(name))
        # Write-then-rename so a crash mid-write can't corrupt the file.
        # A disk that refuses the write (full, read-only, file in use) is a
        # StorageError in plain words, so the save banner says which and the
        # work in memory survives; it was a traceback naming the .tmp file.
        # Anything else propagates raw: a crash must stay loud.
        data = json.dumps(state, indent=2)
        path = self._path(name)
        tmp = f"{path}.tmp"
        try:
            with open(tmp, 'w', encoding='utf-8') as f:
                f.write(data)
            _retry_locked(os.replace, tmp, path)
        except OSError as e:
            raise StorageError(wording.not_saved(name, disk_cause(e))) from e
        finally:
            try:
                if os.path.exists(tmp):
                    os.remove(tmp)
            except OSError:
                pass
        seen[name] = self._stamp(name)

    def delete_archive(self, name):
        """Take one saved copy off the disk for good.

        Only a copy: a name that is not an archive name is refused rather
        than deleted, because the one caller is the sidebar's tidy-up and a
        project file is not its to take.
        """
        if not is_archive_name(name):
            raise StorageError(
                "That is a project, not a saved copy. Only copies can be "
                "deleted here."
            )
        try:
            os.remove(self._path(name))
        except FileNotFoundError:
            return False
        except OSError as e:
            raise StorageError(wording.copy_not_deleted(disk_cause(e))) from e
        self.__dict__.setdefault("_seen", {}).pop(name, None)
        # What Insights worked out for the copy, if it was ever opened: it
        # goes with the copy.
        self._remove_insights(name)
        return True

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
        # The copy every destructive act takes first. A disk that refuses it
        # is a StorageError every caller already shows, never a traceback.
        try:
            if copy:
                shutil.copy2(path, self._path(archive_name))
            else:
                _retry_locked(os.rename, path, self._path(archive_name))
        except OSError as e:
            raise StorageError(wording.nothing_changed(name, disk_cause(e))) from e
        if not copy:
            seen.pop(name, None)
            # Delete project and Start over: what Insights worked out for the
            # project (measured values, make-ups) goes with it; the copy
            # gets its own only if it is ever opened (final review M-6).
            self._remove_insights(name)
        return archive_name

    @staticmethod
    def _remove_insights(name):
        """Take `name`'s Insights cache off the disk (insights_cache.path,
        spelled out here so this module stays free of torch)."""
        try:
            os.remove(os.path.join(".insights", f"{name}.insights.json"))
        except OSError:
            pass
