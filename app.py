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
from food_bo import FoodOptimizer
from ui_helpers import (
    ARMED_KEY, TAB_BATCH, TAB_RESULTS, TAB_SETUP, confirm_action,
    confirmation_open, flash, landing_tab, open_rows, other_confirmation,
    plural, render_flash, saved_line, saved_ok,
)

STORAGE = storage_backend.LocalStorage()

_SAMPLE_CSV = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "data", "sample_ingredients.csv")

st.set_page_config(page_title="Food Optimizer", layout="wide")
st.title("Food Optimizer")
# The container the flash messages live in. The sidebar below runs later and
# can queue one of its own, so it is drained a second time once the sidebar
# has had its say — into this same slot, above the tabs.
_FLASH_BOX = render_flash()

_NAME_RE = _re.compile(r"[A-Za-z0-9][A-Za-z0-9 _.\-]{0,63}")


# Half-typed set-up and batch entries belong to the project they were typed
# in. Streamlit keeps a widget's value in session state under its key, so
# without this a measurement name typed in project A reappears in project B's
# form. Popping a key while its widget is on screen raises, which is why this
# only ever runs from a handler, before the tabs render.
_FORM_KEY_PREFIXES = (
    "meas_",                       # the measurement form, new and per-edit
    "ing_",                        # ingredient name, min, max
    "pp_",                         # process setting name, min, max, baseline
    "qc_",                         # amount limit min, max
    "tm_",                         # total limit min, max
    "prop_",                       # property limit metric, min, max
    "bo_",                         # advanced model settings
)
_GRID_KEY_RE = _re.compile(r"^f\d+_")   # tab 2: f7_Firmness, f7_note, f7_leave_out


def _reset_project_session():
    """Everything a project owns. A key left behind here follows the user into
    the next project — an uploaded sheet offered for import into a project it
    was never meant for, a confirmation already half-clicked, or a half-typed
    ingredient waiting in another project's form."""
    for k in ("optimizer", "current_batch", "_restore_candidate",
              "_results_upload", "_import_rows", "_editing_measurement",
              "scale_total", "results_order", "show_amounts",
              "correct_formulation", "delete_formulation", "amount_unit",
              "_clear_correct_formulation", "_clear_delete_formulation",
              "qty_pick", "batch_size", "repeat_best", "_pending_tab",
              ARMED_KEY):
        st.session_state.pop(k, None)
    for k in [k for k in st.session_state
              if isinstance(k, str) and (
                  k.endswith("__pending")
                  or k.startswith(_FORM_KEY_PREFIXES)
                  or _GRID_KEY_RE.match(k))]:
        st.session_state.pop(k, None)


def _open_project(name, create=False):
    if create:
        new_opt = FoodOptimizer(name, storage=STORAGE)
        new_opt.set_amount_unit("g")     # the default unit for a new project
        if new_opt.save_error:
            st.error(new_opt.save_error)
            return
    _reset_project_session()
    st.session_state["_loaded_project"] = name
    st.session_state["_land_on_open"] = True
    flash("success", f"Created {name}." if create else f"Opened {name}.")
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
            st.error(f"The sample project could not be created: {e}")
        else:
            if _sample.save_error:
                st.error(_sample.save_error)
            else:
                _open_project(_name)


def _batch_line(opt):
    """The one line under the title on tabs 1 and 2 once a batch exists."""
    if opt.pending_batch:
        return f"Batch {opt.pending_batch_no} · {len(open_rows(opt))} to make"
    last = opt.last_batch_no()   # counts a batch whose rows were all left out
    if last is not None:
        return f"Batch {last} · recorded"
    return None


# ================================================================== #
#  Sidebar: projects, backup and restore, manage project
# ================================================================== #

with st.sidebar:
    st.subheader("Projects")

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
        new_name = st.text_input("New project name", key="new_project_name",
                                 placeholder="e.g. Oat cookie v2")
        # Grey: the sidebar never competes with the tab's one lit button, and
        # this one is on screen long before there is a name to create.
        if st.form_submit_button("Create project"):
            name = new_name.strip()
            if not _NAME_RE.fullmatch(name):
                st.error("Use 1 to 64 letters, numbers, spaces, hyphens, underscores or "
                         "periods, starting with a letter or number.")
            elif name in existing_projects:
                st.error(f"A project named {name} already exists. Open it below.")
            elif storage_backend.is_archive_name(name) or STORAGE.exists(name):
                st.error("That name is already used by a project or an archived copy. "
                         "Choose another.")
            else:
                _open_project(name, create=True)

    if existing_projects:
        _active = st.session_state.get("_loaded_project")
        selected = st.selectbox(
            "Open project", existing_projects,
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
            "Try the sample project", key="sample_project_sidebar",
            help="Opens a ready-made plant-based burger project with eight "
                 "ingredients and two trained-panel scores, Juiciness and "
                 "Firmness, each with a target intensity, so you can explore "
                 "before setting up your own.",
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
            st.error("This project could not be opened. See the message "
                     "on the right.")

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
            st.caption(
                "Backup download is unavailable while the project file "
                "cannot be read."
            )
        else:
            st.download_button(
                "Download project backup",
                data=json.dumps(opt.export_json(), indent=2),
                file_name=f"{opt.project_name} backup {datetime.now():%Y-%m-%d}.json",
                mime="application/json",
            )

        # Any file name: what is inside decides, not the extension.
        uploaded_json = st.file_uploader("Restore from backup", key="restore_json")
        if uploaded_json is not None and st.button("Check this backup"):
            try:
                st.session_state["_restore_candidate"] = json.loads(uploaded_json.read())
            except ValueError:
                st.session_state.pop("_restore_candidate", None)
                st.error(
                    "This file could not be read as a Food Optimizer backup. "
                    "If you have another copy, try that one; recent copies are "
                    "kept in FoodOptimizer › backups in your home folder."
                )

        candidate = st.session_state.get("_restore_candidate")
        if candidate is not None:
            try:
                summary = FoodOptimizer.validate_state(candidate)
            except ValueError as e:
                st.error(
                    f"{e} Recent copies of your own projects are kept in "
                    "FoodOptimizer › backups in your home folder."
                )
                st.session_state.pop("_restore_candidate", None)
            else:
                st.warning(
                    f"This backup contains project **{summary['name']}** with "
                    f"{plural(summary['experiments'], 'formulation')} and "
                    f"{plural(summary['ingredients'], 'ingredient')}. Replace "
                    f"**{opt.project_name}** "
                    f"({plural(len(opt.X_history), 'formulation')})? A copy of "
                    "the current project is kept first."
                )
                rc1, rc2 = st.columns(2)
                with rc1:
                    # A confirmation is the one thing to do while it is on
                    # screen, so it is lit — like every confirm_action.
                    if st.button("Yes, replace", type="primary",
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
                                st.error("This backup could not be applied. Your "
                                         "current project was not changed.")
                                st.session_state.pop("_restore_candidate", None)
                            else:
                                if new_opt.save_error:
                                    st.error(new_opt.save_error)
                                else:
                                    st.session_state.optimizer = new_opt
                                    st.session_state.pop("_restore_candidate", None)
                                    st.session_state.pop("current_batch", None)
                                    _done = (
                                        f"Restored "
                                        f"{plural(len(new_opt.X_history), 'formulation')} "
                                        f"into {new_opt.project_name}."
                                    )
                                    if archived:
                                        _done += (" A copy of the previous project "
                                                  f"was kept as {archived}.")
                                    flash("success", _done)
                                    st.rerun()
                with rc2:
                    if st.button("Cancel", use_container_width=True, key="restore_cancel"):
                        st.session_state.pop("_restore_candidate", None)
                        st.rerun()

        with st.expander("Manage project"):
            if confirm_action(
                "hard_reset", "Hard reset",
                (f"Start **{opt.project_name}** over? A copy of it is kept in your "
                 "projects folder and the project becomes empty."
                 if not opt.X_history else
                 f"Start **{opt.project_name}** over? Its "
                 f"{plural(len(opt.X_history), 'formulation')} and set-up are kept "
                 "as a copy in your projects folder, and the project becomes empty."),
                confirm_label="Yes, reset",
                disabled=other_confirmation("hard_reset"),
            ):
                _target = opt.project_name
                try:
                    archived = STORAGE.archive(_target, "archived", copy=False)
                except storage_backend.StorageError as e:
                    st.error(str(e))
                else:
                    if archived:
                        flash("info", f"A copy of the previous project was kept as {archived}.")
                    _reset_project_session()
                    st.session_state["_loaded_project"] = _target
                    st.session_state["_land_on_open"] = True
                    st.rerun()

            # Same shape as Hard reset, but the project leaves the list. The
            # file is renamed to an archive copy, never erased.
            if confirm_action(
                "delete_project", "Delete",
                (f"Delete **{opt.project_name}**? It has no formulations yet. A copy "
                 "of its set-up is kept in your projects folder, and it leaves this list."
                 if not opt.X_history else
                 f"Delete **{opt.project_name}** and its "
                 f"{plural(len(opt.X_history), 'formulation')}? A copy is kept in your "
                 "projects folder, and it leaves this list."),
                confirm_label="Yes, delete",
                disabled=other_confirmation("delete_project"),
            ):
                _target = opt.project_name
                try:
                    archived = STORAGE.archive(_target, "deleted", copy=False)
                except storage_backend.StorageError as e:
                    st.error(str(e))
                else:
                    if archived:
                        flash("info", f"Deleted {_target}. A copy was kept as {archived}.")
                    else:
                        flash("info", f"Deleted {_target}.")
                    _reset_project_session()
                    st.session_state.pop("_loaded_project", None)
                    st.session_state.pop("project_select", None)
                    st.rerun()


    if existing_projects:
        with _open_slot:
            # Lit only while it would do something, and only while nothing else
            # is lit: an armed confirmation's Yes is the one coloured button in
            # the sidebar until it is answered.
            if st.button(
                "Open",
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
    st.markdown("## Create your first project")
    st.markdown(
        "1. **Name a project** in the sidebar and click Create project.\n"
        "2. **Add ingredients** and the measurements you will record.\n"
        "3. **Make a batch**, make the formulations, and record the results."
    )
    wc1, wc2 = st.columns(2)
    with wc1:
        if os.path.exists(_SAMPLE_CSV) and st.button("Try the sample project", type="primary"):
            _open_sample_project()
    with wc2:
        if os.path.exists(_SAMPLE_CSV):
            with open(_SAMPLE_CSV, "rb") as f:
                st.download_button("Download ingredient CSV template", data=f.read(),
                                   file_name="ingredients_template.csv", mime="text/csv")
    st.stop()


# A damaged project must never be silently overwritten: every edit below
# calls save(), so pause the editing UI until the user restores a backup or
# hard-resets (both stay available in the sidebar).
if getattr(st.session_state.optimizer, "load_error", None):
    st.error(st.session_state.optimizer.load_error)
    st.info(
        "To protect the original file, editing is paused. In the sidebar on "
        "the left you can: restore a backup you downloaded earlier "
        "(Restore from backup), or start this project over "
        "(Manage project › Hard reset — the damaged file is archived, not deleted)."
    )
    st.stop()

_save_banner_shown = False
if getattr(st.session_state.optimizer, "save_error", None):
    _err_opt = st.session_state.optimizer
    _save_banner_shown = True
    st.error(_err_opt.save_error)
    st.warning("**Your last change was not saved.** Download a backup now, then click Reload project.")
    b1, b2 = st.columns(2)
    with b1:
        st.download_button("Download backup",
                           data=json.dumps(_err_opt.export_json(), indent=2),
                           file_name=f"{_err_opt.project_name} backup {datetime.now():%Y-%m-%d}.json",
                           mime="application/json", use_container_width=True,
                           key="download_after_save_error")
    with b2:
        if st.button("Reload project", key="reload_after_save_error", use_container_width=True):
            st.session_state.pop("optimizer", None)
            st.session_state.pop("current_batch", None)
            flash("info", "Project reloaded from the latest saved copy.")
            st.rerun()


# A batch generated before the ingredient list changed cannot be made. This
# lives in the main body, not the sidebar: the notice is about the batch the
# user is looking at, and so is the red banner when the discard fails to save.
_opt = st.session_state.optimizer
if getattr(_opt, "pending_batch", None):
    _var_names = {v['name'] for v in _opt.variables}
    try:
        # Ingredients only. A measurement is what you will score, not what you
        # weigh out, so removing or editing one leaves every formulation in the
        # batch perfectly makeable.
        _batch_ok = all(set(r['recipe']) == _var_names for r in _opt.pending_batch)
    except (TypeError, AttributeError, KeyError):
        _batch_ok = False           # malformed backup rows; treat as a mismatch
    if not _batch_ok:
        _opt.set_pending_batch(None)
        # Only claimed once the discard reached the file. A failed write here
        # arrives after the banner above has rendered, which is exactly what
        # the end-of-script check at the foot of this file is for.
        if saved_ok(_opt):
            flash("info",
                  "The open batch was discarded because the ingredient list or "
                  "its allowed amounts changed since it was generated.")
            render_flash(_FLASH_BOX)   # this run, above the tabs


# ================================================================== #
#  The loop: three tabs
# ================================================================== #


# The landing rule. This is one of the six places allowed to change tabs (the
# other five are go_to_tab's callers: Continue to make a batch, Back to set up,
# Save results, Save uploaded results, Start the next batch), and it fires only
# on the run that follows opening a project.
if st.session_state.pop("_land_on_open", False):
    st.session_state["main_tab"] = landing_tab(_opt)

# A move asked for on the previous run by go_to_tab. Both this and the landing
# rule above must land BEFORE st.tabs: the widget reads its key from session
# state only while it is being created, and on_change="rerun" is what binds the
# key to the frontend at all — without it "main_tab" is only a CSS class.
if "_pending_tab" in st.session_state:
    st.session_state["main_tab"] = st.session_state.pop("_pending_tab")

tab_setup, tab_batch, tab_results = st.tabs(
    [TAB_SETUP, TAB_BATCH, TAB_RESULTS], key="main_tab", on_change="rerun")

_line = _batch_line(_opt)

with tab_setup:
    if _line:
        st.caption(_line)
    ui_setup.render(_opt, STORAGE)

with tab_batch:
    if _line:
        st.caption(_line)
    ui_batch.render(_opt, STORAGE)

with tab_results:
    ui_results.render(_opt, STORAGE)

# A save that failed during this run must be visible now, not after the next
# click: form submits do not rerun, and the banner above already rendered.
_opt_end = st.session_state.get("optimizer")
if _opt_end is not None and getattr(_opt_end, "save_error", None) and not _save_banner_shown:
    st.rerun()
