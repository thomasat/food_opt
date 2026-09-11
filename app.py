import json
import os
import re as _re
from datetime import datetime

import streamlit as st
import pandas as pd

import storage as storage_backend
import ui_batch
import ui_results
import ui_setup
import wording
from food_bo import FoodOptimizer
from ui_helpers import (
    ARMED_KEY, TAB_BATCH, TAB_RESULTS, TAB_SETUP, clear_selection,
    confirm_action, confirmation_open, drain_clears, flash, landing_tab,
    open_rows, other_confirmation, park_clear, plural, render_flash,
    saved_line, saved_ok, take_clear,
)

STORAGE = storage_backend.LocalStorage()

_SAMPLE_CSV = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "data", "sample_ingredients.csv")

st.set_page_config(page_title=wording.APP_TITLE, layout="wide")
st.title(wording.APP_TITLE)
# The container the flash messages live in. The sidebar below runs later and
# can queue one of its own, so it is drained a second time once the sidebar
# has had its say — into this same slot, above the tabs.
_FLASH_BOX = render_flash()

_NAME_RE = _re.compile(r"[A-Za-z0-9][A-Za-z0-9 _.\-]{0,63}")


# Half-typed set-up and trial entries belong to the project they were typed
# in. Streamlit keeps a widget's value in session state under its key, so
# without this a measurement name typed in project A reappears in project B's
# form. Popping a key while its widget is on screen raises, which is why this
# only ever runs from a handler, before the tabs render.
_FORM_KEY_PREFIXES = (
    "meas_",                       # the measurement form, new and per-edit
    "var_",                        # the one add form: name, type, lowest,
                                   # highest, unit, baseline, and the picker
    "qc_",                         # amount limit min, max
    "tm_",                         # total limit min, max
    "prop_",                       # property limit metric, at least, at most,
                                   # and the box that names a new property
    "setprop_",                    # the Set property values editor
    "bo_",                         # advanced model settings
    # The three file uploaders. A file uploader cannot be emptied from session
    # state at all — assigning None is refused and popping the key leaves the
    # mounted widget holding the file — so each is keyed to its project
    # (ingredients_csv_<project>) and a new project renders a new, empty one.
    # These pops only clear the state the old widgets left behind.
    "ingredients_csv", "results_csv", "import_csv",
)
_GRID_KEY_RE = _re.compile(r"^f\d+_")   # tab 2: f7_Firmness, f7_note, f7_leave_out

# What each box on the set-up and trial forms holds in a project nobody has
# typed in yet. Popping a widget's key does NOT empty it: the widget is still
# mounted in the browser and posts its old value straight back, which carried
# a half-typed ingredient into the next project and offered to add it there.
# So the empty value is PARKED and assigned before the widget is created, the
# same pattern clear_selection has always used for a select box.
_FORM_FRESH = {
    "var_name": "", "var_low": 0.0, "var_high": 100.0, "var_base": None,
    "var_kind": "Ingredient", "unit_value": "",
    "prop_min": None, "prop_max": None, "prop_new": "",
    "qc_min": None, "qc_max": None, "tm_min": None, "tm_max": None,
    "meas_new_name": "", "meas_new_unit": "", "meas_new_goal": "max",
    "meas_new_target": 0.0, "meas_new_min": 0.0, "meas_new_max": 10.0,
    "meas_new_importance": 1.0,
    "qty_pick": [],
    "batch_size": 3, "repeat_best": False, "scale_total": None,
}
# The boxes whose empty value is None: the select boxes, and the add form's
# unit box, which empties to the newly opened project's own default (app.py
# passes it to drain_clears; it is not known here).
_FORM_EMPTIES_TO_NONE = ("correct_formulation", "delete_formulation",
                         "var_unit")

# The property boxes: one per property on the add form (var_prop_<name>) and
# one per property in the Set property values editor (setprop_<row>_<name>).
_PROPERTY_BOX_PREFIXES = ("var_prop_", "setprop_")


def _grid_fresh(key):
    """The empty value of one result-grid box: a note is text, Not made is a
    tick, and a measurement is an empty number box."""
    if key.endswith("_note"):
        return ""
    if key.endswith("_leave_out"):
        return False
    return None


def _reset_project_session():
    """Everything a project owns. A key left behind here follows the user into
    the next project — an uploaded sheet offered for import into a project it
    was never meant for, a confirmation already half-clicked, or a half-typed
    ingredient waiting in another project's form."""
    for k in ("optimizer", "current_batch", "_restore_candidate",
              "_results_upload", "_import_rows", "_editing_measurement",
              "_ingredients_loaded", "results_order", "show_amounts",
              "_pending_tab", "_var_kind_shown", "_props_for", ARMED_KEY):
        st.session_state.pop(k, None)
    for k in [k for k in st.session_state if isinstance(k, str)]:
        if k in _FORM_FRESH:
            park_clear(k, _FORM_FRESH[k])
        elif k.startswith(_PROPERTY_BOX_PREFIXES):
            # One box per property: the names are the project's own, so they
            # are parked by prefix rather than listed in _FORM_FRESH.
            park_clear(k, None)
        elif k in _FORM_EMPTIES_TO_NONE:
            clear_selection(k)
        elif _GRID_KEY_RE.match(k):
            park_clear(k, _grid_fresh(k))
        elif k.endswith("__pending") or k.startswith(_FORM_KEY_PREFIXES):
            st.session_state.pop(k, None)


def _open_project(name, create=False, made=False):
    """Open a project. `create` makes an empty one first; `made` says the
    project was built a moment ago by another handler (the sample), so the
    user is told it was created rather than opened."""
    if create:
        new_opt = FoodOptimizer(name, storage=STORAGE)
        new_opt.set_amount_unit("g")     # the default unit for a new project
        if new_opt.save_error:
            st.error(new_opt.save_error)
            return
    _reset_project_session()
    st.session_state["_loaded_project"] = name
    st.session_state["_land_on_open"] = True
    # Parked, not popped: the box is mounted in the browser and would post the
    # previous project's name straight back, leaving the sidebar naming one
    # project, the box another, and Open lit over the project just abandoned.
    clear_selection("project_select")
    flash("success",
          wording.project_created(name) if (create or made)
          else wording.project_opened(name))
    st.rerun()


def _open_sample_project():
    _name = "Sample project"
    if STORAGE.exists(_name):
        _open_project(_name)          # already created earlier; just open it
    else:
        try:
            _sample = FoodOptimizer(_name, storage=STORAGE)
            _sample.set_amount_unit("g")
            _sample.load_ingredients_from_csv(pd.read_csv(_SAMPLE_CSV))
            # A plant-based burger rated by a trained panel for intensity, 0 to
            # 10. Intensity has an optimum (10 juiciness is soggy, 10 firmness
            # is a puck), so both are targets; firmness matters a little more.
            _sample.add_objective("Juiciness", 1.0, goal="target", target=7,
                                  min_val=0, max_val=10, unit="/10")
            _sample.add_objective("Firmness", 1.5, goal="target", target=6,
                                  min_val=0, max_val=10, unit="/10")
        except ValueError as e:
            st.error(wording.sample_project_failed(e))
        else:
            if _sample.save_error:
                st.error(_sample.save_error)
            else:
                _open_project(_name, made=True)


def _held(opt):
    """Every formulation the project holds: scored, and left out. A left-out
    formulation keeps its number and its amounts, so a warning that counts
    only the scored ones undercounts what it is about to archive."""
    return len(opt.X_history) + len(opt.skipped)


def _batch_line(opt):
    """The one line under the title on tabs 1 and 2 once a trial exists."""
    if opt.pending_batch:
        return wording.batch_line_open(opt.pending_batch_no, len(open_rows(opt)))
    last = opt.last_batch_no()   # counts a trial whose rows were all left out
    if last is not None:
        return wording.batch_line_recorded(last)
    return None


# ================================================================== #
#  Sidebar: projects, backup and restore, manage project
# ================================================================== #

with st.sidebar:
    st.subheader(wording.PROJECTS_HEADER)

    try:
        existing_projects = STORAGE.list_projects()
    except storage_backend.StorageError as e:
        st.error(str(e))
        existing_projects = []

    # Returning users land where they left off; new users get the welcome
    # panel. Never auto-open on a shared host (Streamlit Community Cloud
    # mounts the repo at /mount/src): every visitor there shares one disk.
    _shared_host = os.path.isdir("/mount/src")
    if ("_loaded_project" not in st.session_state and existing_projects
            and not _shared_host):
        _recent = getattr(STORAGE, "most_recent_project", lambda: None)()
        st.session_state["_loaded_project"] = _recent or existing_projects[0]
        st.session_state["_land_on_open"] = True

    with st.form("new_project_form", clear_on_submit=True):
        new_name = st.text_input(wording.NEW_PROJECT_NAME_LABEL, key="new_project_name",
                                 placeholder=wording.NEW_PROJECT_PLACEHOLDER)
        # Grey: the sidebar never competes with the tab's one lit button, and
        # this one is on screen long before there is a name to create.
        if st.form_submit_button(wording.CREATE_PROJECT):
            name = new_name.strip()
            if not _NAME_RE.fullmatch(name):
                st.error(wording.NAME_RULE)
            elif name in existing_projects:
                st.error(wording.name_taken(name))
            elif storage_backend.is_archive_name(name) or STORAGE.exists(name):
                st.error(wording.NAME_COLLIDES_WITH_ARCHIVE)
            else:
                _open_project(name, create=True)

    if existing_projects:
        _active = st.session_state.get("_loaded_project")
        # A deleted project's name would otherwise stay in the box — popping
        # the key does not reach the browser — leaving Open lit over a
        # project that is no longer there.
        take_clear("project_select", fresh=_active)
        selected = st.selectbox(
            wording.OPEN_PROJECT_LABEL, existing_projects,
            index=existing_projects.index(_active) if _active in existing_projects else 0,
            key="project_select",
        )
        # The one sidebar control that ever lights up, and only while it would
        # do something: choosing a project in the box does not open it.
        _switching = selected != _active
        # Its position is reserved here and filled at the foot of the sidebar:
        # arming a confirmation does not rerun, so an Open drawn now would
        # still be coloured on the very run that puts a "Yes, reset" beside it.
        _open_slot = st.container()

    # On a first run the welcome panel already offers the sample, so the
    # sidebar shows it only once at least one project exists.
    if existing_projects and os.path.exists(_SAMPLE_CSV):
        if st.button(
            wording.TRY_SAMPLE_LABEL, key="sample_project_sidebar",
            help=wording.TRY_SAMPLE_HELP,
        ):
            _open_sample_project()

    project_name = st.session_state.get("_loaded_project")
    if project_name is None:
        opt = None
    else:
        if "optimizer" not in st.session_state:
            st.session_state.optimizer = FoodOptimizer(project_name, storage=STORAGE)
        elif getattr(st.session_state.optimizer, 'CLASS_VERSION', 0) < FoodOptimizer.CLASS_VERSION:
            st.session_state.optimizer = FoodOptimizer(
                st.session_state.optimizer.project_name, storage=STORAGE
            )
        opt = st.session_state.optimizer

    if opt is not None:
        # A project that failed to load must never look like an empty success.
        # The full sentence renders once, in the main body; here it would be
        # the same paragraph twice on one screen, so the sidebar only points.
        if getattr(opt, "load_error", None):
            st.error(wording.PROJECT_LOAD_ERROR_SIDEBAR_NOTE)

        st.divider()
        st.subheader(opt.project_name)

        if getattr(opt, "save_error", None):
            st.error(opt.save_error)
        else:
            # last_saved_at only records saves made in THIS session, so a
            # returning user would see no saved line until their first edit.
            _saved = getattr(opt, "last_saved_at", None) or getattr(
                opt.storage, "saved_at", lambda n: None)(opt.project_name)
            if _saved is not None:
                st.caption(saved_line(_saved))

        if getattr(opt, "load_error", None):
            # Never offer a "backup" of a project that failed to load — it
            # would be an empty file wearing the project's name.
            st.caption(wording.BACKUP_UNAVAILABLE)
        else:
            st.download_button(
                wording.DOWNLOAD_PROJECT_BACKUP,
                data=json.dumps(opt.export_json(), indent=2),
                file_name=f"{opt.project_name} backup {datetime.now():%Y-%m-%d}.json",
                mime="application/json",
            )

        # Any file name: what is inside decides, not the extension.
        uploaded_json = st.file_uploader(wording.RESTORE_FROM_BACKUP, key="restore_json")
        if uploaded_json is not None and st.button(wording.CHECK_THIS_BACKUP):
            try:
                st.session_state["_restore_candidate"] = json.loads(uploaded_json.read())
            except ValueError:
                st.session_state.pop("_restore_candidate", None)
                st.error(wording.BACKUP_UNREADABLE)

        candidate = st.session_state.get("_restore_candidate")
        if candidate is not None:
            try:
                summary = FoodOptimizer.validate_state(candidate)
            except ValueError as e:
                st.error(f"{e}{wording.RECENT_COPIES_HINT}")
                st.session_state.pop("_restore_candidate", None)
            else:
                # Both halves count the same way — scored and left out — or
                # replacing a project with its own backup reads as losing one.
                # A settings-only project is named by its settings: "0
                # ingredients" alone described a fully set-up project as empty.
                _holds = [plural(summary['formulations'], wording.FORMULATION),
                          plural(summary['ingredients'], wording.INGREDIENT)]
                if summary['settings']:
                    _holds.append(plural(summary['settings'], wording.PROCESS_SETTING))
                st.warning(wording.restore_backup_warning(
                    summary['name'], _holds, opt.project_name,
                    plural(_held(opt), wording.FORMULATION)))
                rc1, rc2 = st.columns(2)
                with rc1:
                    # A confirmation is the one thing to do while it is on
                    # screen, so it is lit — like every confirm_action.
                    if st.button(wording.YES_REPLACE, type="primary",
                                 use_container_width=True):
                        # Constructing FoodOptimizer below would re-stamp the
                        # shared _seen entry from the current file, so the
                        # stale check must happen first, against what THIS
                        # session loaded.
                        if getattr(opt.storage, "is_stale", lambda n: False)(opt.project_name):
                            st.error(storage_backend.LocalStorage._CONFLICT)
                        else:
                            try:
                                archived = STORAGE.archive(opt.project_name,
                                                           "pre_restore", copy=True)
                                state = dict(candidate)
                                state['project_name'] = opt.project_name
                                new_opt = FoodOptimizer(opt.project_name, storage=opt.storage)
                                new_opt.import_json(state)
                                new_opt.save()
                            except storage_backend.StorageError as e:
                                st.error(str(e))
                            except Exception:
                                st.error(wording.BACKUP_APPLY_FAILED)
                                st.session_state.pop("_restore_candidate", None)
                            else:
                                if new_opt.save_error:
                                    st.error(new_opt.save_error)
                                else:
                                    st.session_state.optimizer = new_opt
                                    st.session_state.pop("_restore_candidate", None)
                                    st.session_state.pop("current_batch", None)
                                    flash("success", wording.restored_flash(
                                        plural(len(new_opt.X_history), wording.FORMULATION),
                                        new_opt.project_name, archived))
                                    st.rerun()
                with rc2:
                    if st.button(wording.CANCEL, use_container_width=True, key="restore_cancel"):
                        st.session_state.pop("_restore_candidate", None)
                        st.rerun()

        with st.expander(wording.MANAGE_PROJECT):
            if confirm_action(
                "hard_reset", wording.START_OVER_LABEL,
                (wording.start_over_warning(opt.project_name)
                 if not _held(opt) else
                 wording.start_over_warning(
                     opt.project_name, plural(_held(opt), wording.FORMULATION))),
                confirm_label=wording.YES_START_OVER,
                disabled=other_confirmation("hard_reset"),
            ):
                _target = opt.project_name
                try:
                    archived = STORAGE.archive(_target, "archived", copy=False)
                except storage_backend.StorageError as e:
                    st.error(str(e))
                else:
                    # Archiving renames the file away, so the emptied project
                    # has to be written back under its own name. Without this
                    # the project vanished from Open project, the sidebar
                    # still named it, and Open lit up over a project that was
                    # no longer on disk.
                    _fresh = FoodOptimizer(_target, storage=STORAGE)
                    _fresh.save()
                    if _fresh.save_error:
                        st.error(_fresh.save_error)
                    else:
                        if archived:
                            flash("info", wording.copy_saved_as(archived))
                        _reset_project_session()
                        st.session_state["_loaded_project"] = _target
                        st.session_state["_land_on_open"] = True
                        st.rerun()

            # Same shape as Start this project over, but the project leaves the
            # list. The
            # file is renamed to an archive copy, never erased.
            if confirm_action(
                "delete_project", wording.DELETE_PROJECT_LABEL,
                (wording.delete_project_warning(opt.project_name)
                 if not _held(opt) else
                 wording.delete_project_warning(
                     opt.project_name, plural(_held(opt), wording.FORMULATION))),
                confirm_label=wording.YES_DELETE_IT,
                disabled=other_confirmation("delete_project"),
            ):
                _target = opt.project_name
                try:
                    archived = STORAGE.archive(_target, "deleted", copy=False)
                except storage_backend.StorageError as e:
                    st.error(str(e))
                else:
                    flash("info", wording.project_deleted_flash(_target, archived))
                    _reset_project_session()
                    st.session_state.pop("_loaded_project", None)
                    clear_selection("project_select")   # parked: see above
                    st.rerun()


    if existing_projects:
        with _open_slot:
            # Lit only while it would do something, and only while nothing else
            # is lit: an armed confirmation's Yes is the one coloured button in
            # the sidebar until it is answered.
            if st.button(
                wording.OPEN_BUTTON,
                type=("primary" if _switching and not confirmation_open()
                      else "secondary"),
                disabled=not _switching,
            ) and _switching:
                _open_project(selected)


# Anything the sidebar queued belongs on THIS run, above the tabs.
render_flash(_FLASH_BOX)


# ================================================================== #
#  Welcome panel (no project open)
# ================================================================== #

if opt is None:
    st.markdown(wording.WELCOME_HEADER)
    st.markdown(wording.welcome_steps())
    wc1, wc2 = st.columns(2)
    with wc1:
        if os.path.exists(_SAMPLE_CSV) and st.button(wording.TRY_SAMPLE_LABEL, type="primary"):
            _open_sample_project()
    with wc2:
        if os.path.exists(_SAMPLE_CSV):
            with open(_SAMPLE_CSV, "rb") as f:
                st.download_button(wording.DOWNLOAD_CSV_TEMPLATE, data=f.read(),
                                   file_name="ingredients_template.csv", mime="text/csv")
    st.stop()


# A damaged project must never be silently overwritten: every edit below
# calls save(), so pause the editing UI until the user restores a backup or
# hard-resets (both stay available in the sidebar).
if getattr(st.session_state.optimizer, "load_error", None):
    st.error(st.session_state.optimizer.load_error)
    st.info(wording.project_load_error_info())
    st.stop()

_save_banner_shown = False
if getattr(st.session_state.optimizer, "save_error", None):
    _err_opt = st.session_state.optimizer
    _save_banner_shown = True
    st.error(_err_opt.save_error)
    st.warning(wording.SAVE_ERROR_WARNING)
    b1, b2 = st.columns(2)
    with b1:
        st.download_button(wording.DOWNLOAD_BACKUP,
                           data=json.dumps(_err_opt.export_json(), indent=2),
                           file_name=f"{_err_opt.project_name} backup {datetime.now():%Y-%m-%d}.json",
                           mime="application/json", use_container_width=True,
                           key="download_after_save_error")
    with b2:
        if st.button(wording.RELOAD_PROJECT, key="reload_after_save_error", use_container_width=True):
            st.session_state.pop("optimizer", None)
            st.session_state.pop("current_batch", None)
            flash("info", wording.PROJECT_RELOADED)
            st.rerun()


# A trial generated before the ingredient list changed cannot be made. This
# lives in the main body, not the sidebar: the notice is about the trial the
# user is looking at, and so is the red banner when the discard fails to save.
_opt = st.session_state.optimizer
if getattr(_opt, "pending_batch", None):
    _var_names = {v['name'] for v in _opt.variables}
    try:
        # Ingredients only. A measurement is what you will score, not what you
        # weigh out, so deleting or editing one leaves every formulation in the
        # trial perfectly makeable.
        _batch_ok = all(set(r['recipe']) == _var_names for r in _opt.pending_batch)
    except (TypeError, AttributeError, KeyError):
        _batch_ok = False           # malformed backup rows; treat as a mismatch
    if not _batch_ok:
        _opt.set_pending_batch(None)
        # Only claimed once the discard reached the file. A failed write here
        # arrives after the banner above has rendered, which is exactly what
        # the end-of-script check at the foot of this file is for.
        if saved_ok(_opt):
            flash("info", wording.batch_discarded_notice())
            render_flash(_FLASH_BOX)   # this run, above the tabs


# ================================================================== #
#  The loop: three tabs
# ================================================================== #


# The landing rule. This is one of the six places allowed to change tabs (the
# other five are go_to_tab's callers: Next: make a trial, Back to set up,
# Save results, Save uploaded results, Start the next trial), and it fires only
# on the run that follows opening a project.
if st.session_state.pop("_land_on_open", False):
    st.session_state["main_tab"] = landing_tab(_opt)

# A move asked for on the previous run by go_to_tab. Both this and the landing
# rule above must land BEFORE st.tabs: the widget reads its key from session
# state only while it is being created, and on_change="rerun" is what binds the
# key to the frontend at all — without it "main_tab" is only a CSS class.
if "_pending_tab" in st.session_state:
    st.session_state["main_tab"] = st.session_state.pop("_pending_tab")

# Everything a project switch parked is assigned here: the previous run's
# widgets are gone and this run's tab widgets do not exist yet, which is the
# one moment Streamlit lets a widget's value be set.
drain_clears({"var_unit": _opt.amount_unit})

tab_setup, tab_batch, tab_results = st.tabs(
    [TAB_SETUP, TAB_BATCH, TAB_RESULTS], key="main_tab", on_change="rerun")

_line = _batch_line(_opt)

with tab_setup:
    # Tab 1 only: tab 2 carries the trial's own heading, and the line sat
    # directly above "Trial 1 · make these 3 formulations" saying it again.
    if _line:
        st.caption(_line)
    ui_setup.render(_opt, STORAGE)

with tab_batch:
    ui_batch.render(_opt, STORAGE)

with tab_results:
    ui_results.render(_opt, STORAGE)

# A save that failed during this run must be visible now, not after the next
# click: form submits do not rerun, and the banner above already rendered.
_opt_end = st.session_state.get("optimizer")
if _opt_end is not None and getattr(_opt_end, "save_error", None) and not _save_banner_shown:
    st.rerun()
