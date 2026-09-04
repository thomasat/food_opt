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
import pickle
import shutil


class StorageError(Exception):
    """A storage backend failure (network, API, write conflict).

    NOT raised for "project does not exist" — load() returns None for that.
    Messages are shown to end users, so keep them plain-language.
    """


class LocalStorage:
    persist_empty_on_init = True   # historical behavior: file created on init
    persist_after_load = True      # historical behavior: every load re-saves,
                                   # which is what migrates legacy pickles

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
        return sorted(os.path.splitext(f)[0] for f in glob.glob("*.pkl"))

    def exists(self, name):
        return os.path.exists(self._path(name))

    def load(self, name):
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
        self._seen[name] = self._stamp(name)
        try:
            return json.loads(raw.decode('utf-8'))
        except (ValueError, UnicodeDecodeError):
            try:
                return pickle.loads(raw)  # legacy project files were pickle
            except Exception:
                raise StorageError(
                    "This project file is damaged and could not be opened. "
                    "If you have a backup, use Restore from backup; otherwise "
                    "check the FoodOptimizer > backups folder in your home "
                    "folder for a recent copy."
                )

    def save(self, name, state):
        known = self._seen.get(name)
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
        self._seen[name] = self._stamp(name)

    def archive(self, name, label, copy=False):
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
            self._seen.pop(name, None)
        return archive_name
