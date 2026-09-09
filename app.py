import json
import os

import streamlit as st
import pandas as pd

import storage as storage_backend
from food_bo import FoodOptimizer
from ui_helpers import confirm_action, flash, render_flash

STORAGE = storage_backend.LocalStorage()

_SAMPLE_CSV = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "sample_ingredients.csv")


def _table_height(n_rows, max_rows=12):
    """Pixel height that shows up to max_rows rows of a st.dataframe without
    an inner scrollbar (35 px per row plus the header). Streamlit's default
    of a few visible rows made lists of ingredients hard to read."""
    return 38 + 35 * max(1, min(n_rows, max_rows)) + 2

st.set_page_config(page_title="Food Optimizer", layout="wide")
st.title("Food Optimizer")
render_flash()

# Defined early (before any st.stop(), including the welcome-panel one below)
# so the end-of-script save-error check always finds a defined name.
_save_banner_shown = False

# ================================================================== #
#  Sidebar: Project Management
# ================================================================== #

import re as _re
_NAME_RE = _re.compile(r"[A-Za-z0-9][A-Za-z0-9 _.\-]{0,63}")


def _reset_project_session():
    for k in ("optimizer", "current_batch", "_batch_id",
              "_last_saved", "_restore_candidate", "edit_idx", "rewind_idx",
              "edit_no", "rewind_no", "hist_order", "_results_upload",
              "_edit_warning_idx", "_edit_warning_n"):
        st.session_state.pop(k, None)
    for k in [k for k in st.session_state if k.endswith("__pending")]:
        st.session_state.pop(k, None)


def _open_project(name, create=False):
    if create:
        new_opt = FoodOptimizer(name, storage=STORAGE)
        new_opt.save()
        if new_opt.save_error:
            st.error(new_opt.save_error)
            return
    _reset_project_session()
    st.session_state["_loaded_project"] = name
    if create:
        flash("success", f"Created {name}.")
    else:
        flash("success", f"Opened {name}.")
    st.rerun()


def _open_sample_project():
    _name = "Sample project"
    if STORAGE.exists(_name):
        _open_project(_name)          # already created earlier; just open it
    else:
        try:
            _sample = FoodOptimizer(_name, storage=STORAGE)
            _sample.load_ingredients_from_csv(pd.read_csv(_SAMPLE_CSV))
            # A plant-based burger rated by a trained panel for intensity, 0 to
            # 10. Intensity has an optimum (10 juiciness is soggy, 10 firmness
            # is a puck), so both are targets; firmness weighted a little more.
            _sample.add_objective("Juiciness", 1.0, goal="target", target=7,
                                  min_val=0, max_val=10)
            _sample.add_objective("Firmness", 1.5, goal="target", target=6,
                                  min_val=0, max_val=10)
        except ValueError as e:
            st.error(f"The sample project could not be created: {e}")
        else:
            if _sample.save_error:
                st.error(_sample.save_error)
            else:
                _open_project(_name)


def _readiness(opt):
    """(ready: bool, reason: str, parts: list[str]) for the status strip."""
    n_ing = sum(1 for v in opt.variables if v.get('category', 'ingredient') == 'ingredient')
    n_obj = len(opt.objectives)
    parts = [
        f"Ingredients: {n_ing} {'✓' if n_ing else '✗ required'}",
        f"Objectives: {n_obj} {'✓' if n_obj else '✗ required'}",
        f"Constraints: {len(opt.constraints) + len(opt.quantity_constraints)} (optional)",
    ]
    if not n_ing:
        return False, "Add at least one ingredient in the Set up tab.", parts
    if not n_obj:
        return False, "Add at least one objective in the Set up tab.", parts
    return True, "", parts


with st.sidebar:
    st.subheader("Projects")

    try:
        existing_projects = STORAGE.list_projects()
    except storage_backend.StorageError as e:
        st.error(str(e))
        existing_projects = []

    # Returning users land where they left off; new users get the welcome panel.
    # Never auto-open on a shared host (Streamlit Community Cloud mounts the
    # repo at /mount/src): every visitor there shares one disk, so "most
    # recent" would be someone else's project. Same marker cloud-storage uses.
    _shared_host = os.path.isdir("/mount/src")
    if ("_loaded_project" not in st.session_state and existing_projects
            and not _shared_host):
        _recent = getattr(STORAGE, "most_recent_project", lambda: None)()
        st.session_state["_loaded_project"] = _recent or existing_projects[0]

    with st.form("new_project_form", clear_on_submit=True):
        new_name = st.text_input("New project name", key="new_project_name",
                                 placeholder="e.g. Oat cookie v2")
        if st.form_submit_button("Create project", type="primary"):
            name = new_name.strip()
            if not _NAME_RE.fullmatch(name):
                st.error("Use 1 to 64 letters, numbers, spaces, hyphens, underscores or "
                         "periods, starting with a letter or number.")
            elif name in existing_projects:
                st.error(f"A project named {name} already exists. Open it below.")
            elif storage_backend.is_archive_name(name) or STORAGE.exists(name):
                st.error("That name is already used by a project or an archive. "
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
        # Button rule used throughout the app: the button that moves you
        # forward is coloured (primary), housekeeping stays grey, and a
        # button that would do nothing is greyed out. Choosing a project in
        # the box does not open it, so Open lights up the moment the choice
        # differs from the project on screen.
        _switching = selected != _active
        if st.button("Open", type="primary" if _switching else "secondary",
                     disabled=not _switching) and _switching:
            _open_project(selected)

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
        # Restore a persisted suggestion batch (only if this session has none: the
        # CLASS_VERSION upgrade path must not clobber a live batch).
        if ("current_batch" not in st.session_state
                and not getattr(opt, "load_error", None)
                and getattr(opt, "pending_batch", None)):
            _var_names = {v['name'] for v in opt.variables}
            try:
                _batch_ok = bool(opt.objectives) and all(
                    set(r) == _var_names for r in opt.pending_batch
                )
            except (TypeError, AttributeError):
                _batch_ok = False   # malformed backup rows; treat as a mismatch
            if _batch_ok:
                st.session_state.current_batch = opt.pending_batch
            else:
                opt.set_pending_batch(None)
                st.info(
                    "A previously suggested batch was discarded because the "
                    "ingredient list or ranges changed since it was generated."
                )
        # A project that failed to load must never look like an empty success.
        if getattr(opt, "load_error", None):
            st.error(opt.load_error)
        _n_act = len(opt.active_variables())
        _n_all = len(opt.variables)
        n_paused = _n_all - _n_act
        _paused_suffix = f" · {n_paused} paused" if n_paused > 0 else ""
        st.caption(
            f"Active: **{opt.project_name}** | {len(opt.X_history)} experiments "
            f"| {_n_act} ingredients and settings in play{_paused_suffix}"
        )
        _saved = getattr(opt, "last_saved_at", None)
        if _saved is not None:
            st.caption(f"Saved {_saved:%H:%M} to this computer. Download a backup from Backup and restore any time.")

        # --- This project: backup, restore, reset, delete ---
        # The heading names the OPEN project (not the one selected in the box
        # above), so it is clear what every control below acts on, and it
        # changes the moment Open switches projects.
        st.divider()
        st.subheader(opt.project_name)
        st.caption("Everything below acts on this project.")
        st.markdown("**Backup and restore**")

        if getattr(opt, "load_error", None):
            # Never offer a "backup" of a project that failed to load — it would
            # be an empty file wearing the project's name.
            st.caption(
                "Backup download is unavailable while the project file "
                "cannot be read."
            )
        else:
            project_json = json.dumps(opt.export_json(), indent=2)
            st.download_button(
                "Download project backup",
                data=project_json,
                file_name=f"{opt.project_name}.json",
                mime="application/json",
            )

        uploaded_json = st.file_uploader("Restore from backup", type=["json"], key="restore_json")
        if uploaded_json is not None and st.button("Check this backup", type="primary"):
            try:
                st.session_state["_restore_candidate"] = json.loads(uploaded_json.read())
            except ValueError:
                st.session_state.pop("_restore_candidate", None)
                st.error(
                    "This backup file couldn't be read. Make sure it's a backup "
                    "downloaded from Food Optimizer (a .json file) and try again."
                )

        candidate = st.session_state.get("_restore_candidate")
        if candidate is not None:
            try:
                summary = FoodOptimizer.validate_state(candidate)
            except ValueError as e:
                st.error(str(e))
                st.session_state.pop("_restore_candidate", None)
            else:
                st.warning(
                    f"This backup contains project **{summary['name']}** with "
                    f"{summary['experiments']} experiments and {summary['ingredients']} "
                    f"ingredients. Replace **{opt.project_name}** "
                    f"({len(opt.X_history)} experiments)? The current project is "
                    "archived first."
                )
                rc1, rc2 = st.columns(2)
                with rc1:
                    if st.button("Yes, replace", type="primary", use_container_width=True):
                        # Constructing FoodOptimizer below would re-stamp the
                        # shared _seen entry from the current file, so a stale
                        # check must happen first, against what THIS session
                        # loaded — otherwise the coming save can't detect that
                        # another window changed the file in the meantime.
                        if getattr(opt.storage, "is_stale", lambda n: False)(opt.project_name):
                            st.error(storage_backend.LocalStorage._CONFLICT)
                        else:
                            try:
                                STORAGE.archive(opt.project_name, "pre_restore", copy=True)
                                state = dict(candidate)
                                state['project_name'] = opt.project_name
                                new_opt = FoodOptimizer(opt.project_name, storage=opt.storage)
                                new_opt.import_json(state)
                                new_opt.save()
                            except storage_backend.StorageError as e:
                                st.error(str(e))
                            except Exception:
                                st.error("This backup could not be applied. Your current project was not changed.")
                                st.session_state.pop("_restore_candidate", None)
                            else:
                                if new_opt.save_error:
                                    st.error(new_opt.save_error)
                                else:
                                    st.session_state.optimizer = new_opt
                                    st.session_state.pop("_restore_candidate", None)
                                    st.session_state.pop("current_batch", None)
                                    flash("success", f"Restored {len(new_opt.X_history)} experiments into {new_opt.project_name}.")
                                    st.rerun()
                with rc2:
                    if st.button("Cancel", use_container_width=True, key="restore_cancel"):
                        st.session_state.pop("_restore_candidate", None)
                        st.rerun()

        # --- Hard Reset ---
        st.divider()

        if confirm_action(
            "hard_reset", "Hard reset project",
            f"Start **{opt.project_name}** over? Its {len(opt.X_history)} experiment(s) "
            "and setup are moved to an archive copy, not deleted.",
            confirm_label="Yes, reset",
        ):
            _target = opt.project_name
            try:
                archived = STORAGE.archive(_target, "archived", copy=False)
            except storage_backend.StorageError as e:
                st.error(str(e))
            else:
                if archived:
                    flash("info", f"Your previous data was kept as an archive named {archived}.")
                _reset_project_session()
                st.session_state["_loaded_project"] = _target
                st.rerun()

        # --- Delete project ---
        # Same shape as Hard Reset, but the project leaves the list. The file
        # is renamed to an archive copy (never erased), so a wrong click costs
        # a rename in the projects folder, not data.
        if confirm_action(
            "delete_project", "Delete project",
            f"Delete **{opt.project_name}**? Its {len(opt.X_history)} experiment(s) "
            "and setup are moved to an archive copy in your projects folder, not "
            "erased, and the project leaves this list.",
            confirm_label="Yes, delete project",
        ):
            _target = opt.project_name
            try:
                archived = STORAGE.archive(_target, "deleted", copy=False)
            except storage_backend.StorageError as e:
                st.error(str(e))
            else:
                if archived:
                    flash("info", f"Deleted {_target}. A copy was kept as an archive "
                                  f"named {archived}.")
                else:
                    flash("info", f"Deleted {_target}.")
                _reset_project_session()
                # With no project pinned, the next run opens the most recent
                # remaining project, or the welcome panel if none is left.
                st.session_state.pop("_loaded_project", None)
                st.session_state.pop("project_select", None)
                st.rerun()


if opt is None:
    st.markdown("## Create your first project")
    st.markdown(
        "1. **Name a project** in the sidebar and click Create project.\n"
        "2. **Add ingredients** and say what you will measure.\n"
        "3. **Generate recipes**, make them, and enter the results."
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
# hard-resets (both stay available in the sidebar, as does switching project).
if getattr(st.session_state.optimizer, "load_error", None):
    st.error(st.session_state.optimizer.load_error)
    st.info(
        "To protect the original file, editing is paused. In the sidebar on "
        "the left you can: restore a backup you downloaded earlier "
        "(Restore from backup), or start this project over "
        "(Hard reset project — the damaged file is archived, not deleted)."
    )
    st.stop()

_save_banner_shown = False
if getattr(st.session_state.optimizer, "save_error", None):
    _opt = st.session_state.optimizer
    _save_banner_shown = True
    st.error(_opt.save_error)
    st.warning("**Your last change was not saved.** Download a backup now, then click Reload project.")
    b1, b2 = st.columns(2)
    with b1:
        st.download_button("Download backup", data=json.dumps(_opt.export_json(), indent=2),
                           file_name=f"{_opt.project_name}_backup.json", mime="application/json",
                           type="primary", use_container_width=True, key="download_after_save_error")
    with b2:
        if st.button("Reload project", key="reload_after_save_error", use_container_width=True):
            st.session_state.pop("optimizer", None)
            st.session_state.pop("current_batch", None)
            flash("info", "Project reloaded from the latest saved copy.")
            st.rerun()

# ================================================================== #
#  Tab 1: Setup & Config
# ================================================================== #

# Keyed so the chosen tab survives reruns: without a key the browser fell
# back to tab 1 whenever the content of tab 2 changed shape, e.g. the first
# time Generate recipes drew the batch and the results form.
tab_setup, tab_optimize = st.tabs(["1. Set up your project", "2. Run experiments"],
                                  key="main_tab")

_ready, _reason, _parts = _readiness(st.session_state.optimizer)
for _tab in (tab_setup, tab_optimize):
    with _tab:
        st.caption(" · ".join(_parts))

with tab_setup:
    col_a, col_b = st.columns(2)

    # -------------------------------------------------------------- #
    #  Column A: Ingredients, Process Parameters, Objectives
    # -------------------------------------------------------------- #
    with col_a:
        # --- A1. Ingredients ---
        st.subheader("Ingredients")
        # Standing instructions live in a caption and a tooltip, not an info
        # box: the box shouted the same four sentences on every visit. The
        # template sits right here because the only other place it is offered
        # (the first-run screen) is unreachable once a project is open.
        st.caption("One row per ingredient: Name, Min, Max, in your own units such as grams.")
        uploaded_csv = st.file_uploader(
            "Upload ingredients CSV", type=["csv"],
            help="Extra columns such as Cost or Protein per 100 g become "
                 "properties you can set limits on.",
        )
        if os.path.exists(_SAMPLE_CSV):
            with open(_SAMPLE_CSV, "rb") as _tf:
                st.download_button("Download CSV template", data=_tf.read(),
                                   file_name="ingredients_template.csv", mime="text/csv",
                                   key="ingredients_template_setup")
        df = None
        if uploaded_csv:
            try:
                df = pd.read_csv(uploaded_csv)
            except Exception:
                st.error(
                    "This file couldn't be read as a CSV. If it came from "
                    "Excel, use File > Save As and pick CSV format, then "
                    "try again."
                )
        if df is not None:
            st.dataframe(df, hide_index=True, height=_table_height(len(df)))
            if st.button("Load ingredients", type="primary"):
                try:
                    st.session_state.optimizer.load_ingredients_from_csv(df)
                    st.session_state.pop("current_batch", None)  # stale under new design space
                    st.success(f"Loaded {len(df)} ingredients.")
                except ValueError as e:
                    st.error(str(e))

        ingredient_vars = [
            v for v in st.session_state.optimizer.variables
            if v.get('category', 'ingredient') == 'ingredient'
        ]
        if ingredient_vars:
            st.caption("Current ingredients:")
            ing_df = pd.DataFrame([
                {
                    "Name": v['name'],
                    "Min": v['bounds'][0],
                    "Max": v['bounds'][1],
                    "Status": "active" if v.get('active', True) else "paused",
                }
                for v in ingredient_vars
            ])
            st.dataframe(ing_df, hide_index=True, height=_table_height(len(ing_df), max_rows=20))

        _has_hist = bool(st.session_state.optimizer.X_history)
        with st.form("add_ing_form", clear_on_submit=True):
            ic1, ic2, ic3 = st.columns([2, 1, 1])
            with ic1:
                ing_name = st.text_input("Ingredient name", key="ing_name")
            with ic2:
                ing_min = st.number_input(
                    "Min", value=0.0, key="ing_min", disabled=_has_hist,
                )
            with ic3:
                ing_max = st.number_input("Max", value=100.0, key="ing_max")
            if _has_hist:
                st.caption(
                    "Because experiments already exist, a new ingredient starts "
                    "at 0 in every past recipe, so its Min is fixed at 0. You "
                    "can raise it later once the model has data for it."
                )
            if st.form_submit_button("Add ingredient", type="primary"):
                try:
                    st.session_state.optimizer.add_ingredient(ing_name, ing_min, ing_max)
                except ValueError as e:
                    st.error(str(e))
                else:
                    st.session_state.pop("current_batch", None)
                    st.success(f"Added {ing_name.strip()}.")

        st.divider()

        # --- A2. Process Parameters ---
        st.subheader("Process settings (optional)")
        st.caption(
            "Add settings such as baking temperature or mixing time that the "
            "optimizer will also explore."
        )

        with st.form("process_param_form"):
            pp_cols = st.columns(4)
            with pp_cols[0]:
                pp_name = st.text_input(
                    "Setting name", placeholder="e.g. Baking temperature", key="pp_name")
            with pp_cols[1]:
                pp_min = st.number_input("Min value", value=0.0, key="pp_min")
            with pp_cols[2]:
                pp_max = st.number_input("Max value", value=100.0, key="pp_max")
            _pp_mid_run = bool(st.session_state.optimizer.X_history)
            with pp_cols[3]:
                if _pp_mid_run:
                    pp_base = st.number_input(
                        "Baseline", value=None, placeholder="required", key="pp_base",
                        help="The setting you used in every past batch, so those "
                             "results still count.",
                    )
                else:
                    pp_base = st.number_input(
                        "Baseline", value=0.0, key="pp_base",
                        help="Only needed once you have experiments: the setting used "
                             "in every past batch, between Min and Max. Past results "
                             "are recorded at this value.",
                    )
            if st.form_submit_button("Add process setting"):
                if _pp_mid_run and pp_base is None:
                    st.error("Enter the baseline: the setting you used in all your past batches.")
                else:
                    try:
                        st.session_state.optimizer.add_process_parameter(
                            pp_name, pp_min, pp_max,
                            baseline=(pp_base if _pp_mid_run else None),
                        )
                    except ValueError as e:
                        st.error(str(e))
                    else:
                        st.session_state.pop("current_batch", None)  # stale under new design space
                        st.success(f"Added process setting: {pp_name}")

        proc_vars = [
            v for v in st.session_state.optimizer.variables
            if v.get('category') == 'process'
        ]
        if proc_vars:
            st.caption("Current process settings:")
            for i, pv in enumerate(proc_vars):
                pc1, pc2 = st.columns([3, 1])
                with pc1:
                    _tag = "" if pv.get('active', True) else "  (paused)"
                    lo, hi = pv['bounds']
                    _base = pv.get('_absent_value')
                    _base_txt = "" if _base is None else f", baseline {_base:g}"
                    st.text(
                        f"{pv['name']}: {lo:g} to {hi:g}{_base_txt}{_tag}"
                    )
                with pc2:
                    if st.session_state.optimizer.X_history:
                        _go_pp = confirm_action(
                            f"rm_pp_{i}", "Remove",
                            f"Remove {pv['name']}? Past experiments will be recorded "
                            f"without it. A copy of the project is archived first.",
                            confirm_label="Yes, remove",
                        )
                    else:
                        _go_pp = st.button("Remove", key=f"rm_pp_{i}")
                    if _go_pp:
                        try:
                            STORAGE.archive(st.session_state.optimizer.project_name, "pre_delete", copy=True)
                        except storage_backend.StorageError as e:
                            st.error(str(e))
                        else:
                            st.session_state.optimizer.remove_process_parameter(pv['name'])
                            st.session_state.pop("current_batch", None)  # stale under new design space
                            flash("success", f"Removed {pv['name']}.")
                            st.rerun()

        st.divider()

        # --- B. Objectives ---
        st.subheader("Objectives — what you'll measure")
        st.caption(
            "For each measurement, say whether you want it high, low or at a "
            "target, and the range of values you expect. Results are scored on "
            "that range."
        )

        _goal_labels = {"max": "Higher is better", "min": "Lower is better", "target": "Hit a target"}
        with st.form("obj_form"):
            col_name, col_w = st.columns([2, 1])
            with col_name:
                obj_name = st.text_input("Measurement name, such as Chewiness")
            with col_w:
                obj_weight = st.slider(
                    "Weight", 0.1, 1.0, 1.0, step=0.1,
                    help="How much this matters relative to your other objectives. "
                         "Weights are relative and do not need to add up to 1.",
                )

            col_g, col_min, col_max = st.columns(3)
            with col_g:
                obj_goal = st.selectbox(
                    "Goal", list(_goal_labels), format_func=_goal_labels.get,
                    help="Whether you want this measurement higher, lower, or to hit a target.",
                )
            with col_min:
                obj_min = st.number_input("Range min", value=0.0,
                                          help="The lowest value you would realistically measure.")
            with col_max:
                obj_max = st.number_input("Range max", value=10.0,
                                          help="The highest value you would realistically measure.")

            obj_target = st.number_input("Target value (used when the goal is Hit a target)", value=5.0)

            if st.form_submit_button("Add or update objective", type="primary"):
                try:
                    replaced = st.session_state.optimizer.add_objective(
                        obj_name, obj_weight, obj_goal,
                        target=obj_target if obj_goal == 'target' else None,
                        min_val=obj_min, max_val=obj_max,
                    )
                except ValueError as e:
                    st.error(str(e))
                else:
                    st.session_state.pop("current_batch", None)  # stale under new design space
                    st.success(f"{'Updated' if replaced else 'Added'} {obj_name.strip()}.")

        if st.session_state.optimizer.objectives:
            _ceiling = st.session_state.optimizer.utility_ceiling()
            st.caption(
                f"Weights add up to {_ceiling:g}, so a perfect recipe scores {_ceiling:g}. "
                "Weights are relative: only their ratios matter."
            )
            _rows = [{
                "Measurement": o['name'],
                "Goal": _goal_labels.get(o['goal'], o['goal']),
                "Weight": o['weight'],
                "Range": f"{o['min_val']:g} – {o['max_val']:g}",
                "Target": "" if o.get('target') is None else f"{o['target']:g}",
            } for o in st.session_state.optimizer.objectives]
            st.dataframe(pd.DataFrame(_rows), hide_index=True)
            for i, obj in enumerate(st.session_state.optimizer.objectives):
                _has_history = bool(st.session_state.optimizer.X_history)
                if _has_history:
                    _go = confirm_action(
                        f"rm_obj_{i}", f"Remove {obj['name']}",
                        f"Remove {obj['name']}? Every stored Overall Score is recalculated without it.",
                        confirm_label="Yes, remove",
                    )
                else:
                    _go = st.button(f"Remove {obj['name']}", key=f"rm_obj_{i}")
                if _go:
                    st.session_state.optimizer.remove_objective(obj['name'])
                    st.session_state.pop("current_batch", None)
                    flash("success", f"Removed {obj['name']}.")
                    st.rerun()

    # -------------------------------------------------------------- #
    #  Column B: Screening Model, Constraints
    # -------------------------------------------------------------- #
    with col_b:
        # --- D. Property Constraints ---
        st.subheader("Limits on ingredient properties (optional)")
        st.caption(
            "Cap or floor a property of the whole recipe, such as sodium or cost "
            "per 100 g, using the property columns from your ingredient file."
        )

        available_props = set()
        for p in st.session_state.optimizer.ingredient_properties.values():
            available_props.update(p.keys())

        if available_props:
            c_metric = st.selectbox("Property", sorted(available_props))
            c_min = st.number_input("Min value", value=0.0, key="prop_c_min")
            c_max = st.number_input("Max value", value=100.0, key="prop_c_max")
            if st.button("Add property limit"):
                try:
                    st.session_state.optimizer.add_constraint(c_metric, min_val=c_min, max_val=c_max)
                except ValueError as e:
                    st.error(str(e))
                else:
                    st.success(f"Added a limit on {c_metric}.")
        else:
            st.write("Your ingredient file has no property columns, such as "
                     "Cost or Sodium per 100 g.")

        if st.session_state.optimizer.constraints:
            st.caption("Current property limits:")
            st.dataframe(pd.DataFrame(st.session_state.optimizer.constraints))
            for i, constr in enumerate(st.session_state.optimizer.constraints):
                if st.button(f"Remove {constr['metric']}", key=f"rm_constr_{i}"):
                    st.session_state.optimizer.remove_constraint(i)
                    st.rerun()

        st.divider()

        # --- E. Ingredient Quantity Constraints ---
        st.subheader("Limits on ingredient amounts (optional)")
        st.caption(
            "Limit the combined amount of a group of ingredients, for example "
            "Sugar plus Honey at most 50 g, or the total mass of the recipe."
        )

        ingredient_names = [
            v['name'] for v in st.session_state.optimizer.variables
            if v.get('category', 'ingredient') == 'ingredient'
        ]

        if ingredient_names:
            with st.form("qty_constraint_form"):
                selected_ings = st.multiselect(
                    "Ingredients to limit together",
                    ingredient_names,
                    help="The ingredients whose combined amount you want to limit.",
                )
                qc_cols = st.columns(2)
                with qc_cols[0]:
                    qc_min = st.number_input(
                        "Min sum", value=0.0, key="qc_min",
                        help="Lowest allowed total",
                    )
                with qc_cols[1]:
                    qc_max = st.number_input(
                        "Max sum", value=100.0, key="qc_max",
                        help="Highest allowed total",
                    )
                qc_use_min = st.checkbox("Set a minimum", value=False, key="qc_use_min")
                qc_use_max = st.checkbox("Set a maximum", value=True, key="qc_use_max")

                if st.form_submit_button("Add amount limit"):
                    if len(selected_ings) >= 1:
                        try:
                            st.session_state.optimizer.add_quantity_constraint(
                                selected_ings,
                                min_val=qc_min if qc_use_min else None,
                                max_val=qc_max if qc_use_max else None,
                            )
                        except ValueError as e:
                            st.error(str(e))
                        else:
                            st.success(f"Added an amount limit on: {', '.join(selected_ings)}")
                    else:
                        st.error("Choose at least one ingredient.")

            # Total mass shortcut
            st.caption("Or limit the total mass of the recipe:")
            tm_cols = st.columns(3)
            with tm_cols[0]:
                tm_min = st.number_input("Total mass min", value=0.0, key="tm_min")
            with tm_cols[1]:
                tm_max = st.number_input("Total mass max", value=100.0, key="tm_max")
            with tm_cols[2]:
                if st.button("Add total mass limit"):
                    try:
                        st.session_state.optimizer.add_total_mass_constraint(
                            min_val=tm_min, max_val=tm_max,
                        )
                    except ValueError as e:
                        st.error(str(e))
                    else:
                        st.success("Added a total mass limit.")

            # Show active quantity constraints
            qc_list = getattr(st.session_state.optimizer, 'quantity_constraints', [])
            if qc_list:
                st.caption("Current amount limits:")
                for i, qc in enumerate(qc_list):
                    if set(qc['ingredients']) == set(ingredient_names):
                        label = "Total mass (all ingredients)"
                    else:
                        label = " + ".join(qc['ingredients'])
                    bounds = []
                    if qc['min'] is not None:
                        bounds.append(f"at least {qc['min']:g}")
                    if qc['max'] is not None:
                        bounds.append(f"at most {qc['max']:g}")
                    qc_c1, qc_c2 = st.columns([4, 1])
                    with qc_c1:
                        st.text(f"{i + 1}. {label}: {' and '.join(bounds)}")
                    with qc_c2:
                        if st.button("Remove", key=f"rm_qc_{i}"):
                            st.session_state.optimizer.remove_quantity_constraint(i)
                            st.rerun()
        else:
            st.write("Load ingredients first to add amount limits.")

        st.divider()
        with st.expander("Advanced model settings (optional)"):
            st.caption(
                "Standard uses tested defaults and fits most projects. "
                "Expert-selected lets a specialist set the model's kernel, prior, "
                "noise handling and acquisition once at the start."
            )
            _opt = st.session_state.optimizer
            _cur_cfg = getattr(_opt, "bo_config", None)
            _mode = st.radio(
                "Model settings",
                ["Standard (default)", "Expert-selected"],
                index=1 if _cur_cfg else 0,
                key="bo_cfg_mode",
                horizontal=True,
            )
            if _mode == "Standard (default)":
                if _cur_cfg is not None and st.button("Revert to standard settings"):
                    _opt.set_bo_config(None)
                    flash("success", "Using default model settings.")
                    st.rerun()
            else:
                with st.form("bo_config_form"):
                    bc1, bc2 = st.columns(2)
                    with bc1:
                        _k = st.selectbox("Kernel", ["matern52", "matern32", "rbf", "linear", "poly2"])
                        _lp = st.selectbox("Lengthscale prior", ["default", "long", "short"])
                    with bc2:
                        _ns = st.selectbox("Noise", ["default", "low", "fixed_tiny"])
                        _aq = st.selectbox("Acquisition", ["qlognei", "qlogei", "qucb"])
                    st.caption(
                        "Note: `fixed_tiny` noise suits a deterministic objective, not a noisy "
                        "sensory panel — keep `default` unless you have a specific reason."
                    )
                    if st.form_submit_button("Apply expert settings"):
                        _opt.set_bo_config({
                            "kernel": _k, "lengthscale_prior": _lp,
                            "noise": _ns, "acquisition": _aq,
                        })
                        flash("success", "Model settings updated.")
                        st.rerun()
                if st.checkbox("Or paste expert settings as JSON", key="bo_cfg_paste"):
                    _txt = st.text_area(
                        "Expert settings JSON",
                        value='{"kernel": "matern52", "lengthscale_prior": "default", '
                              '"noise": "default", "acquisition": "qlognei"}',
                        key="bo_cfg_json",
                    )
                    if st.button("Apply pasted settings"):
                        try:
                            _opt.set_bo_config(json.loads(_txt))
                            flash("success", "Model settings updated.")
                            st.rerun()
                        except Exception as _e:
                            st.error(f"Invalid JSON: {_e}")
            if _cur_cfg:
                st.info("Active model settings: " + ", ".join(f"{k}: {v}" for k, v in _cur_cfg.items()))


# ================================================================== #
#  Tab 2: Optimization Loop
# ================================================================== #

with tab_optimize:
    _opt = st.session_state.optimizer
    if _opt.X_history:
        _best_i = _opt.best_index()
        _ceiling = _opt.utility_ceiling()
        sc1, sc2 = st.columns([1, 2])
        with sc1:
            st.metric("Best Overall Score", f"{_opt.Y_history[_best_i]:.3f}",
                      help=f"A perfect recipe would score {_ceiling:g} (the sum of your objective weights).")
            st.caption(f"Experiment {_best_i + 1} · out of a possible {_ceiling:g}")
            _best_recipe = _opt.recipe_history[_best_i] if _best_i < len(_opt.recipe_history) else {}
            st.table(pd.DataFrame(_opt.recipe_lines(_best_recipe),
                                  columns=["Ingredient or setting", "Amount"])
                      .set_index("Ingredient or setting")
                      .style.format({"Amount": "{:.2f}"}))
        with sc2:
            _chart = pd.DataFrame({
                "Experiment": range(1, len(_opt.Y_history) + 1),
                "Overall Score": [float(y) for y in _opt.Y_history],
                "Best so far": _opt.best_so_far(),
            }).set_index("Experiment")
            st.line_chart(_chart, height=220)
            st.caption("Each experiment's Overall Score, and the best so far. When the "
                       "top line stops rising, you are close to the best this "
                       "ingredient list can do.")

    _last = st.session_state.pop("_last_saved", None)
    if _last:
        _prev_best = max([float(y) for y in _opt.Y_history[:-len(_last)]] or [float('-inf')])
        _lines = []
        for item in _last:
            tag = " — **new best**" if item["score"] > _prev_best else ""
            _prev_best = max(_prev_best, item["score"])
            _lines.append(f"- Recipe {item['recipe']}: Overall Score {item['score']:.3f}{tag}")
        st.success("Results saved.\n" + "\n".join(_lines))
    st.divider()

    col_ask, col_tell = st.columns([1, 1.5])

    # -------------------------------------------------------------- #
    #  Ask: Generate Experiments
    # -------------------------------------------------------------- #
    with col_ask:
        st.subheader("Generate recipes")
        _n_hist = len(st.session_state.optimizer.X_history)
        if _n_hist < 5:   # matches ask(n_init_random=5)
            st.info(f"Getting started: {_n_hist} of 5 first recipes tried. The first five "
                    "are spread across your ranges so the optimizer can see how each "
                    "ingredient matters. After that, each batch aims closer to your targets.")
        else:
            st.caption(f"Learning from {_n_hist} experiments. Each new batch aims "
                       "closer to your targets.")
        batch_size = st.slider("How many recipes to try this round", 1, 10, 3,
                               help="Recipes you can realistically make before entering results.")
        if not _ready:
            st.caption(_reason)
        if st.button(f"Generate {batch_size} recipes", type="primary", disabled=not _ready):
            with st.spinner("Choosing the next recipes to try…"):
                try:
                    recipes = st.session_state.optimizer.ask(n_suggestions=batch_size)
                    st.session_state.current_batch = recipes
                    st.session_state["_batch_id"] = st.session_state.get("_batch_id", 0) + 1
                    st.session_state.pop("_results_upload", None)
                    st.session_state.optimizer.set_pending_batch(recipes)
                except ValueError as e:
                    st.error(str(e))
                except Exception:
                    # A raw traceback is a dead end for a nontechnical user.
                    st.error(
                        "The app hit an unexpected problem while choosing "
                        "recipes. Try again with fewer recipes. If it keeps "
                        "happening, loosen any limits you added recently, "
                        "or open Help › Get Help."
                    )

        if "current_batch" in st.session_state:
            _opt = st.session_state.optimizer
            df_batch = _opt.batch_frame(st.session_state.current_batch)
            st.markdown("**Your next recipes to make**")
            st.dataframe(df_batch.style.format({c: "{:.2f}" for c in df_batch.columns if c != "Recipe"}),
                         hide_index=True)
            st.download_button(
                "Download batch sheet (CSV)", data=_opt.batch_csv(st.session_state.current_batch),
                file_name=f"{_opt.project_name}_batch.csv", mime="text/csv",
            )
            with st.expander("Printable recipe cards"):
                for i, recipe in enumerate(st.session_state.current_batch):
                    st.markdown(f"**Recipe {i + 1}**")
                    st.table(pd.DataFrame(_opt.recipe_lines(recipe),
                                          columns=["Ingredient or setting", "Amount"])
                              .set_index("Ingredient or setting")
                              .style.format({"Amount": "{:.2f}"}))
                    _zero_names = [k for k, v in recipe.items() if float(v) == 0.0]
                    if _zero_names:
                        st.caption(f"Not used: {', '.join(_zero_names)}")

    # -------------------------------------------------------------- #
    #  Tell: Input Lab Results
    # -------------------------------------------------------------- #
    with col_tell:
        st.subheader("Enter results")

        batch = st.session_state.get("current_batch")
        if not batch:
            st.caption("Generate recipes on the left. A results form for each one appears here.")
        else:
            objs = st.session_state.optimizer.objectives
            batch_id = st.session_state.get("_batch_id", 0)
            # enter_to_submit=False removes Streamlit's "Press Enter to submit
            # form" hint under every focused field; with several fields per
            # recipe that hint was the loudest text on the form.
            with st.form("results_form", enter_to_submit=False):
                batch_inputs, skipped = {}, set()
                for i, recipe in enumerate(batch):
                    st.markdown(f"**Recipe {i + 1}**")
                    _lines = _opt.recipe_lines(recipe, limit=5)
                    _n_more = len(_opt.recipe_lines(recipe)) - len(_lines)
                    _caption = ", ".join(f"{k} {v:.2f}" for k, v in _lines)
                    if _n_more > 0:
                        _caption += f" and {_n_more} more"
                    st.caption(_caption)
                    with st.expander("Show full recipe"):
                        st.table(pd.DataFrame(_opt.recipe_lines(recipe),
                                              columns=["Ingredient or setting", "Amount"])
                                  .set_index("Ingredient or setting")
                                  .style.format({"Amount": "{:.2f}"}))
                    if st.checkbox("Not made or failed — leave this recipe out",
                                   key=f"b{batch_id}_skip{i}"):
                        skipped.add(i)
                    # Wrap at four measurements per row so a pilot with TPA,
                    # colorimeter and shear readings (10+ objectives) stays legible.
                    _per_row = 4
                    rec_scores = {}
                    for j, obj in enumerate(objs):
                        if j % _per_row == 0:
                            cols = st.columns(min(_per_row, len(objs) - j))
                        with cols[j % _per_row]:
                            # One label (the measurement) and the range as the
                            # placeholder, which disappears once they type.
                            rec_scores[obj['name']] = st.number_input(
                                obj['name'],
                                min_value=float(obj['min_val']),
                                max_value=float(obj['max_val']),
                                value=None,
                                placeholder=f"{obj['min_val']:g}–{obj['max_val']:g}",
                                key=f"b{batch_id}_r{i}o{j}",
                            )
                    batch_inputs[i] = rec_scores
                    st.divider()

                if st.form_submit_button("Save results", type="primary"):
                    missing = [
                        f"Recipe {i + 1} {name}"
                        for i, scores in batch_inputs.items() if i not in skipped
                        for name, v in scores.items() if v is None
                    ]
                    kept = [(i, r) for i, r in enumerate(batch) if i not in skipped]
                    if missing:
                        st.error(
                            "Enter a value for: " + ", ".join(missing) + ". "
                            "Tick 'Not made or failed' to leave a recipe out."
                        )
                    elif not kept:
                        st.error("Every recipe is marked as left out, so there is nothing to save.")
                    else:
                        opt_ = st.session_state.optimizer
                        try:
                            for i, recipe in kept:
                                opt_.tell(recipe, batch_inputs[i])
                        except (ValueError, TypeError) as e:
                            st.error(f"Could not save these results: {e}")
                        else:
                            n = len(opt_.Y_history)
                            st.session_state["_last_saved"] = [
                                {"recipe": i + 1, "score": float(opt_.Y_history[n - len(kept) + k])}
                                for k, (i, _) in enumerate(kept)
                            ]
                            opt_.set_pending_batch(None)
                            del st.session_state.current_batch
                            st.session_state.pop("_results_upload", None)
                            st.rerun()

            with st.expander("Or upload results from a CSV"):
                st.caption(
                    "Download the batch sheet on the left, fill in one column per "
                    "measurement, and upload it here. Rows are matched by Recipe number; "
                    "recipes you leave out stay on the bench."
                )
                up = st.file_uploader("Results sheet", type=["csv"], key=f"b{batch_id}_results_csv")
                if up is not None and st.button("Check this sheet", type="primary", key=f"b{batch_id}_check"):
                    try:
                        st.session_state["_results_upload"] = pd.read_csv(up)
                    except Exception:
                        st.session_state.pop("_results_upload", None)
                        st.error("This file couldn't be read as a CSV. If it came from Excel, use File > Save As and pick CSV format.")
                _sheet = st.session_state.get("_results_upload")
                if _sheet is not None:
                    _opt = st.session_state.optimizer
                    try:
                        parsed = _opt.parse_batch_results(_sheet, batch)
                    except ValueError as e:
                        st.error(str(e))
                        st.session_state.pop("_results_upload", None)
                    else:
                        st.info(f"Found results for {len(parsed)} of {len(batch)} recipes: "
                                + ", ".join(f"Recipe {i + 1}" for i, _ in parsed) + ".")
                        if st.button("Save uploaded results", type="primary", key=f"b{batch_id}_save_upload"):
                            try:
                                for i, results in parsed:
                                    _opt.tell(batch[i], results)
                            except (ValueError, TypeError) as e:
                                st.error(f"Could not save these results: {e}")
                            else:
                                done = {i for i, _ in parsed}
                                n = len(_opt.Y_history)
                                st.session_state["_last_saved"] = [
                                    {"recipe": i + 1, "score": float(_opt.Y_history[n - len(parsed) + k])}
                                    for k, (i, _) in enumerate(parsed)
                                ]
                                remaining = [r for i, r in enumerate(batch) if i not in done]
                                _opt.set_pending_batch(remaining or None)
                                if remaining:
                                    st.session_state.current_batch = remaining
                                else:
                                    del st.session_state.current_batch
                                st.session_state.pop("_results_upload", None)
                                st.session_state["_batch_id"] = batch_id + 1
                                st.rerun()

    st.divider()

    # -------------------------------------------------------------- #
    #  Experiment History (with Edit / Delete / Rewind)
    # -------------------------------------------------------------- #
    if st.session_state.optimizer.X_history:
        st.subheader("Experiment history")
        _opt = st.session_state.optimizer
        _order = st.radio("Order", ["Most recent first", "Best first"], horizontal=True,
                          key="hist_order", label_visibility="collapsed")
        hist_df = _opt.history_frame()
        if _order == "Best first":
            hist_df = hist_df.sort_values("Overall Score", ascending=False)
        else:
            hist_df = hist_df.sort_values("Experiment", ascending=False)
        _best_exp = (_opt.best_index() or 0) + 1
        _num_cols = [
            c for c in hist_df.columns
            if c not in ("Experiment", "Date") and pd.api.types.is_numeric_dtype(hist_df[c])
        ]
        _fmt = {c: "{:.2f}" for c in _num_cols if c != "Overall Score"}
        if "Overall Score" in hist_df.columns:
            _fmt["Overall Score"] = "{:.3f}"
        st.dataframe(
            hist_df.style
                .format(_fmt, na_rep="")
                .apply(lambda r: ["background-color: rgba(46,110,78,.18)" if r["Experiment"] == _best_exp else "" for _ in r], axis=1),
            hide_index=True,
        )
        st.caption(f"Highlighted: experiment {_best_exp}, the best so far. "
                   f"Overall Score is out of {_opt.utility_ceiling():g}.")

        st.download_button(
            "Download history (CSV)", data=_opt.history_csv(),
            file_name=f"{_opt.project_name}_history.csv", mime="text/csv",
        )

        # --- Edit / Delete ---
        st.caption("Edit or delete a past result:")
        results_history = getattr(st.session_state.optimizer, 'results_history', [])

        if results_history:
            edit_no = st.number_input(
                "Experiment number to edit", min_value=1,
                max_value=len(st.session_state.optimizer.X_history), value=1, step=1, key="edit_no",
            )
            edit_idx = int(edit_no) - 1

            if edit_idx < len(results_history):
                current_results = results_history[edit_idx]
                st.caption(
                    f"Current results for experiment {edit_idx + 1}: "
                    + " · ".join(f"{k} {v:g}" for k, v in current_results.items())
                )

                if not st.session_state.optimizer.objectives:
                    st.info("Add a measurement in the Set up tab before editing past results.")
                else:
                    with st.form("edit_form", enter_to_submit=False):
                        st.markdown(f"**Edit results for experiment {edit_idx + 1}:**")
                        new_results = {}
                        _edit_objs = st.session_state.optimizer.objectives
                        _per_row = 4
                        for j, obj in enumerate(_edit_objs):
                            if j % _per_row == 0:
                                edit_cols = st.columns(min(_per_row, len(_edit_objs) - j))
                            with edit_cols[j % _per_row]:
                                _has_val = obj['name'] in current_results
                                current_val = float(current_results[obj['name']]) if _has_val else None
                                new_results[obj['name']] = st.number_input(
                                    obj['name'],
                                    min_value=float(obj['min_val']),
                                    max_value=float(obj['max_val']),
                                    value=current_val,
                                    placeholder=f"{obj['min_val']:g}–{obj['max_val']:g}",
                                    key=f"edit_{edit_idx}_{j}",
                                )

                        ec1, ec2 = st.columns(2)
                        with ec1:
                            if st.form_submit_button("Update result"):
                                blank_missing = [
                                    obj['name'] for obj in _edit_objs
                                    if new_results[obj['name']] is None and obj['name'] not in current_results
                                ]
                                if blank_missing:
                                    st.error(
                                        "Enter a value for "
                                        + ", ".join(blank_missing)
                                        + " or leave it blank only for measurements this experiment never had."
                                    )
                                else:
                                    final_results = dict(current_results)
                                    for obj in _edit_objs:
                                        v = new_results[obj['name']]
                                        if v is not None:
                                            final_results[obj['name']] = v
                                    st.session_state.optimizer.edit_result(edit_idx, final_results)
                                    n_after = len(st.session_state.optimizer.X_history) - 1 - edit_idx
                                    if n_after > 0:
                                        st.session_state._edit_warning_idx = edit_idx
                                        st.session_state._edit_warning_n = n_after
                                    flash("success", f"Updated experiment {edit_idx + 1}.")
                                    st.rerun()

                if st.session_state.get('_edit_warning_idx') is not None:
                    warn_idx = st.session_state._edit_warning_idx
                    warn_n = st.session_state._edit_warning_n
                    st.warning(
                        f"Experiments after {warn_idx + 1} ({warn_n} total) were based on "
                        f"the pre-edit ratings and may no longer be valid. "
                        f"Consider rewinding to {warn_idx + 1}."
                    )
                    del st.session_state._edit_warning_idx
                    del st.session_state._edit_warning_n

                _row = st.session_state.optimizer.recipe_history[edit_idx] if edit_idx < len(st.session_state.optimizer.recipe_history) else {}
                _top = st.session_state.optimizer.recipe_lines(_row, limit=3)
                _desc = ", ".join(f"{k} {v:.2f}" for k, v in _top)
                if len(_row) > len(_top):
                    _desc += f" and {len(_row) - len(_top)} more"
                if confirm_action(
                    "delete_exp",
                    f"Delete experiment {edit_idx + 1}",
                    f"Delete experiment {edit_idx + 1} ({_desc})? A copy of the project is archived first.",
                    confirm_label="Yes, delete",
                ):
                    try:
                        STORAGE.archive(st.session_state.optimizer.project_name, "pre_delete", copy=True)
                    except storage_backend.StorageError as e:
                        st.error(str(e))
                    else:
                        st.session_state.optimizer.delete_result(edit_idx)
                        flash("success", f"Deleted experiment {edit_idx + 1}.")
                        st.rerun()
            else:
                st.warning(
                    f"Experiment {edit_idx + 1} did not store its measurements, so it "
                    "cannot be edited. Newer experiments can."
                )
        else:
            st.info(
                "These experiments were recorded before measurements were stored, "
                "so they cannot be edited. New results can."
            )

        # --- Rewind ---
        st.divider()
        st.caption("Rewind to a past experiment:")
        st.info("Rewind keeps experiments 1 through N and **discards all later ones**. "
                "The current project is archived first, so nothing is permanently lost.")
        rewind_no = st.number_input(
            "Keep experiments up to", min_value=1,
            max_value=len(st.session_state.optimizer.X_history),
            value=len(st.session_state.optimizer.X_history), step=1, key="rewind_no",
        )
        rewind_idx = int(rewind_no) - 1
        n_discard = len(st.session_state.optimizer.X_history) - 1 - rewind_idx
        if n_discard > 0:
            st.warning(f"This will discard {n_discard} experiment(s) "
                       f"({rewind_idx + 2} through {len(st.session_state.optimizer.X_history)}).")
        if confirm_action(
            "rewind", "Rewind",
            f"Discard {n_discard} experiment(s) and keep 1 through {rewind_idx + 1}? "
            "A copy of the project is archived first.",
            confirm_label="Yes, rewind", disabled=(n_discard == 0),
        ):
            pname = st.session_state.optimizer.project_name
            try:
                archived = STORAGE.archive(pname, "pre_rewind", copy=True)
            except storage_backend.StorageError as e:
                st.error(str(e))
            else:
                if archived:
                    flash("info", f"Archived current state as {archived}")
                st.session_state.optimizer.rewind_to(rewind_idx)
                st.session_state.pop("current_batch", None)
                flash("success", f"Rewound to experiment {rewind_idx + 1}.")
                st.rerun()

    st.divider()

    # -------------------------------------------------------------- #
    #  Overall Score Explanation
    # -------------------------------------------------------------- #
    with st.expander("How is the Overall Score calculated?"):
        st.markdown(f"""
Each measurement is turned into a score between 0 and 1 using the range you set: the low end of the range scores 0, the high end scores 1, and values outside the range count as the nearest end.

- **Higher is better:** a higher measurement gives a higher score.
- **Lower is better:** a lower measurement gives a higher score.
- **Hit a target:** the score is 1 at the target and falls off evenly on either side.

Each score is multiplied by its measurement's weight and the results are added up. Your objectives' weights add up to {_opt.utility_ceiling():g}, so a perfect recipe scores {_opt.utility_ceiling():g}.

**Example:** Chewiness (higher is better, weight 0.6, range 0 to 10) measured at 8 scores 0.8, weighted 0.48. Sweetness (target 5, weight 0.4, range 0 to 10) measured at 6 is 0.1 of the range away from the target, so it scores 0.9, weighted 0.36. Overall Score 0.84.
        """)

    st.divider()

    # -------------------------------------------------------------- #
    #  Bulk Import Historical Experiments
    # -------------------------------------------------------------- #
    with st.expander("Import past experiments from a CSV"):
        # Name the exact columns this project needs instead of a made-up example.
        _imp_opt = st.session_state.optimizer
        _imp_cols = ([v["name"] for v in _imp_opt.variables]
                     + [o["name"] for o in _imp_opt.objectives])
        if _imp_cols:
            st.caption("One row per past experiment. The columns must match these names "
                       "exactly: " + ", ".join(_imp_cols) + ".")
        else:
            st.caption("One row per past experiment. Add ingredients and measurements "
                       "first; the columns must match their names exactly.")

        import_csv = st.file_uploader("Upload experiments CSV", type=["csv"], key="import_csv")
        import_df = None
        if import_csv is not None:
            try:
                import_df = pd.read_csv(import_csv)
            except Exception:
                st.error(
                    "This file couldn't be read as a CSV. If it came from "
                    "Excel, use File > Save As and pick CSV format, then "
                    "try again."
                )
        if import_df is not None:
            st.dataframe(import_df, hide_index=True)

            var_names = [v['name'] for v in st.session_state.optimizer.variables]
            obj_names = [o['name'] for o in st.session_state.optimizer.objectives]
            required = var_names + obj_names
            missing = [c for c in required if c not in import_df.columns]

            if missing:
                st.error(f"Missing columns: {', '.join(missing)}")
            else:
                nan_cols = [c for c in required if import_df[c].isna().any()]
                if nan_cols:
                    st.error(f"These columns have blank cells: {', '.join(nan_cols)}")
                elif st.button("Import all rows", type="primary"):
                    imported = 0
                    import_error = None
                    try:
                        for _, row in import_df.iterrows():
                            recipe = {name: float(row[name]) for name in var_names}
                            results = {name: float(row[name]) for name in obj_names}
                            st.session_state.optimizer.tell(recipe, results)
                            imported += 1
                    except (ValueError, TypeError) as e:
                        import_error = e
                    if import_error is not None:
                        st.error(
                            f"Stopped at row {imported + 1}: {import_error} "
                            f"The {imported} row(s) before it were imported "
                            f"and saved."
                        )
                    if import_error is None and imported:
                        flash("success", f"Imported {imported} experiments.")
                        st.rerun()

    st.divider()

    # -------------------------------------------------------------- #
    #  Adaptive EGBO: revise the design space mid-run (arm 2)
    # -------------------------------------------------------------- #
    with st.expander("Change the ingredient list mid-project (add, pause, or remove)"):
        _opt = st.session_state.optimizer

        _act = _opt.active_variables()
        _inact = _opt.inactive_variables()
        ac1, ac2 = st.columns(2)
        with ac1:
            st.metric("In play", len(_act))
        with ac2:
            st.metric("Paused (data kept)", len(_inact))

        st.markdown(
            "**1. Share your progress with an expert.** Copy the summary "
            "below and send it along with the other ingredients you could "
            "add."
        )
        st.code(_opt.export_trajectory(), language="text")

        st.markdown(
            "**2. Add an ingredient or process setting.** Your past "
            "experiments are updated automatically."
        )
        _has_hist = bool(_opt.X_history)
        # Radio outside the form so the fields react to the type choice.
        add_type = st.radio(
            "Type", ["Ingredient", "Process setting"],
            horizontal=True, key="adaptive_add_type",
        )
        with st.form("adaptive_add_form"):
            new_name = st.text_input("Name", key="adaptive_new_name")
            nc1, nc2, nc3 = st.columns(3)
            with nc1:
                new_min = st.number_input(
                    "Min", value=0.0, key="adaptive_new_min",
                    disabled=(add_type == "Ingredient" and _has_hist),
                    help="An ingredient added now starts at 0 in every past recipe, "
                         "so its minimum is 0.",
                )
            with nc2:
                new_max = st.number_input("Max", value=10.0, key="adaptive_new_max")
            with nc3:
                new_base = st.number_input(
                    "Baseline (process only)", value=0.0, key="adaptive_new_base",
                    disabled=(add_type == "Ingredient"),
                    help="The setting used in every past batch, between Min and Max. "
                         "Past results are recorded at this value.",
                )
            if add_type == "Ingredient":
                st.caption(
                    "Ingredient: counted as 0 in every past recipe. Its minimum is 0 "
                    "and the optimizer decides how much to use."
                )
            else:
                st.caption(
                    "Process setting: past batches used one fixed value, so enter it "
                    "as the baseline, between Min and Max."
                )
            if st.form_submit_button("Add"):
                nm = new_name.strip()
                if not nm:
                    st.error("Enter a name.")
                else:
                    try:
                        if add_type == "Ingredient":
                            _opt.add_ingredient(nm, new_min, new_max)
                        else:
                            _opt.add_process_parameter(
                                nm, new_min, new_max,
                                baseline=(new_base if _has_hist else None),
                            )
                        st.session_state.pop("current_batch", None)  # stale under new dim
                        flash(
                            "success",
                            f"Added {add_type.lower()} {nm} ({new_min:g} to {new_max:g}). "
                            f"Generate a new batch."
                        )
                        st.rerun()
                    except ValueError as e:
                        st.error(str(e))

        st.divider()
        st.markdown(
            "**3. Pause an ingredient.** Paused ingredients are fixed at 0 "
            "(process settings at their baseline) while new recipes are "
            "chosen. Nothing is deleted and you can resume it any time."
        )

        if len(_act) > 1:
            _off = st.multiselect(
                "Pause",
                [v['name'] for v in _act],
                key="egbo_deactivate",
                help="Paused ingredients are fixed at 0; paused process "
                     "settings are fixed at their baseline or lower bound.",
            )
            if st.button("Pause selected", disabled=not _off):
                try:
                    for nm in _off:
                        _opt.deactivate_variable(nm)
                    st.session_state.pop("current_batch", None)
                    flash("success", f"Paused: {', '.join(_off)}. Generate a new batch.")
                    st.rerun()
                except ValueError as e:
                    st.error(str(e))
        else:
            st.caption("At least two items must be in play before one can be paused.")

        if _inact:
            st.caption("Paused — fixed while new recipes are chosen:")
            st.dataframe(
                pd.DataFrame([
                    {
                        "Name": v['name'],
                        "Type": v.get('category', 'ingredient'),
                        "Fixed at": _opt._frozen_value(v),
                    }
                    for v in _inact
                ]),
                hide_index=True,
            )
            _on = st.multiselect(
                "Resume",
                [v['name'] for v in _inact],
                key="egbo_reactivate",
            )
            if st.button("Resume selected", disabled=not _on):
                for nm in _on:
                    _opt.reactivate_variable(nm)
                st.session_state.pop("current_batch", None)
                flash("success", f"Resumed: {', '.join(_on)}. Generate a new batch.")
                st.rerun()

        # Checkbox rather than an expander: Streamlit forbids nested expanders.
        if st.checkbox("Show permanent deletion (rarely needed)", key="egbo_show_del"):
            st.caption(
                "Deleting removes the ingredient from every past experiment. It is "
                "refused if the ingredient was ever used in an amount above 0, "
                "because that would rewrite past experiments into recipes nobody "
                "made. Pausing keeps the data."
            )
            _ing_names = [
                v['name'] for v in _opt.variables
                if v.get('category', 'ingredient') == 'ingredient'
            ]
            if _ing_names:
                _del = st.selectbox("Ingredient", _ing_names, key="egbo_del_pick")
                _force = st.checkbox(
                    "Force delete even if it was used (discards that information)",
                    key="egbo_del_force",
                )
                if confirm_action(
                    "egbo_del", "Delete permanently",
                    f"Delete {_del} from this project for good? Past experiments will be "
                    "recorded without it. A copy of the project is archived first.",
                    confirm_label="Yes, delete",
                ):
                    try:
                        STORAGE.archive(_opt.project_name, "pre_delete", copy=True)
                        _opt.remove_ingredient(_del, force=_force)
                        st.session_state.pop("current_batch", None)
                        flash("success", f"Deleted '{_del}'.")
                        st.rerun()
                    except (ValueError, storage_backend.StorageError) as e:
                        st.error(str(e))
            else:
                st.caption("No ingredients loaded.")

# A save that failed during this run must be visible now, not after the next
# click: form submits do not rerun, and the banner above already rendered.
_opt_end = st.session_state.get("optimizer")
if _opt_end is not None and getattr(_opt_end, "save_error", None) and not _save_banner_shown:
    st.rerun()
