import json
import os

import streamlit as st
import pandas as pd

import storage as storage_backend
from food_bo import FoodOptimizer
from ui_helpers import confirm_action, flash, render_flash

STORAGE = storage_backend.LocalStorage()

st.set_page_config(page_title="Food Optimizer", layout="wide")
st.title("Food Optimizer")
render_flash()

# ================================================================== #
#  Sidebar: Project Management
# ================================================================== #

import re as _re
_NAME_RE = _re.compile(r"[A-Za-z0-9][A-Za-z0-9 _.\-]{0,63}")


def _reset_project_session():
    for k in ("optimizer", "current_batch", "show_backup_warning", "_batch_id",
              "_last_saved", "_restore_candidate", "edit_idx", "rewind_idx",
              "edit_no", "rewind_no", "hist_order"):
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
        if st.form_submit_button("Create project"):
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
        if st.button("Open") and selected != _active:
            _open_project(selected)

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
                    "design space changed since it was generated."
                )
        # A project that failed to load must never look like an empty success.
        if getattr(opt, "load_error", None):
            st.error(opt.load_error)
        _n_act = len(opt.active_variables())
        _n_all = len(opt.variables)
        _pruned = f" | {_n_all - _n_act} pruned" if _n_all > _n_act else ""
        st.caption(
            f"Active: **{opt.project_name}** | {len(opt.X_history)} experiments "
            f"| {_n_act} active vars{_pruned}"
        )

        # --- Backup & Restore ---
        st.divider()
        st.subheader("Backup & Restore")

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
                "Download Project Backup",
                data=project_json,
                file_name=f"{opt.project_name}.json",
                mime="application/json",
            )

        uploaded_json = st.file_uploader("Restore from backup", type=["json"], key="restore_json")
        if uploaded_json is not None and st.button("Check this backup"):
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
                        try:
                            STORAGE.archive(opt.project_name, "pre_restore", copy=True)
                            state = dict(candidate)
                            state['project_name'] = opt.project_name
                            opt.import_json(state)
                            opt.save()
                        except storage_backend.StorageError as e:
                            st.error(str(e))
                        else:
                            if opt.save_error:
                                st.error(opt.save_error)
                            else:
                                st.session_state.pop("_restore_candidate", None)
                                st.session_state.pop("current_batch", None)
                                flash("success", f"Restored {len(opt.X_history)} experiments into {opt.project_name}.")
                                st.rerun()
                with rc2:
                    if st.button("Cancel", use_container_width=True, key="restore_cancel"):
                        st.session_state.pop("_restore_candidate", None)
                        st.rerun()

        # --- Hard Reset ---
        st.divider()

        if confirm_action(
            "hard_reset", "Hard Reset Project",
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


if opt is None:
    st.markdown("## Create your first project")
    st.markdown(
        "1. **Name a project** in the sidebar and click Create project.\n"
        "2. **Add ingredients** and say what you will measure.\n"
        "3. **Generate recipes**, make them, and enter the results."
    )
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
        "(Hard Reset Project — the damaged file is archived, not deleted)."
    )
    st.stop()

if getattr(st.session_state.optimizer, "save_error", None):
    st.error(st.session_state.optimizer.save_error)
    if st.button("Reload project", key="reload_after_save_error"):
        st.session_state.pop("optimizer", None)
        st.session_state.pop("current_batch", None)
        flash("info", "Project reloaded from the latest saved copy.")
        st.rerun()

# ================================================================== #
#  Tab 1: Setup & Config
# ================================================================== #

tab_setup, tab_optimize = st.tabs(["1. Setup & Config", "2. Optimization Loop"])

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
        st.subheader("A. Ingredients (CSV)")
        st.info("Upload CSV with columns: Name, Min, Max. Optional: Cost, Protein, etc.")

        uploaded_csv = st.file_uploader("Upload Ingredients CSV", type=["csv"])
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
            st.dataframe(df.head(), height=150)
            if st.button("Load Ingredients"):
                try:
                    st.session_state.optimizer.load_ingredients_from_csv(df)
                    st.session_state.pop("current_batch", None)  # stale under new design space
                    st.success(f"Loaded {len(df)} ingredients!")
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
                    "Status": "active" if v.get('active', True) else "pruned",
                }
                for v in ingredient_vars
            ])
            st.dataframe(ing_df, hide_index=True, height=150)

        st.divider()

        # --- A2. Process Parameters ---
        st.subheader("A2. Process Parameters")
        st.caption(
            "Add processing variables (e.g., baking temperature, mixing time) "
            "that the optimizer will also explore."
        )

        with st.form("process_param_form"):
            pp_cols = st.columns(4)
            with pp_cols[0]:
                pp_name = st.text_input(
                    "Parameter Name", placeholder="e.g. Baking_Temp", key="pp_name")
            with pp_cols[1]:
                pp_min = st.number_input("Min Value", value=0.0, key="pp_min")
            with pp_cols[2]:
                pp_max = st.number_input("Max Value", value=100.0, key="pp_max")
            with pp_cols[3]:
                pp_base = st.number_input(
                    "Baseline", value=0.0, key="pp_base",
                    help="Only needed once you have experiments: the value this "
                         "parameter had in ALL past batches (past experiments "
                         "encode at this value; must be between Min and Max).",
                )
            if st.form_submit_button("Add Process Parameter"):
                try:
                    st.session_state.optimizer.add_process_parameter(
                        pp_name, pp_min, pp_max,
                        baseline=(pp_base if st.session_state.optimizer.X_history
                                  else None),
                    )
                except ValueError as e:
                    st.error(str(e))
                else:
                    st.session_state.pop("current_batch", None)  # stale under new design space
                    st.success(f"Added process parameter: {pp_name}")

        proc_vars = [
            v for v in st.session_state.optimizer.variables
            if v.get('category') == 'process'
        ]
        if proc_vars:
            st.caption("Current process parameters:")
            for i, pv in enumerate(proc_vars):
                pc1, pc2 = st.columns([3, 1])
                with pc1:
                    _tag = "" if pv.get('active', True) else "  (pruned)"
                    st.text(
                        f"{pv['name']}: [{pv['bounds'][0]}, {pv['bounds'][1]}]{_tag}"
                    )
                with pc2:
                    if st.button("Remove", key=f"rm_pp_{i}"):
                        st.session_state.optimizer.remove_process_parameter(pv['name'])
                        st.session_state.pop("current_batch", None)  # stale under new design space
                        st.rerun()

        st.divider()

        # --- B. Objectives ---
        st.subheader("Objectives — what you'll measure")
        st.caption(
            "For each measurement, say whether higher or lower is better and the "
            "range of values you expect. Results are scored on that range."
        )

        _goal_labels = {"max": "Higher is better", "min": "Lower is better", "target": "Hit a target"}
        with st.form("obj_form"):
            col_name, col_w = st.columns([2, 1])
            with col_name:
                obj_name = st.text_input("Measurement name (e.g. Chewiness)")
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
                    help="Whether you want this measurement as high as possible, as low as possible, or at a specific value.",
                )
            with col_min:
                obj_min = st.number_input("Range Min", value=0.0,
                                          help="The lowest value you would realistically measure.")
            with col_max:
                obj_max = st.number_input("Range Max", value=10.0,
                                          help="The highest value you would realistically measure.")

            obj_target = st.number_input("Target value (only used for 'Hit a target')", value=5.0)

            if st.form_submit_button("Add or update objective"):
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
        # --- C. Low-Fidelity Model ---
        st.subheader("C. Low-Fidelity Model (Optional)")
        # The old ".pkl model" upload is intentionally disabled: loading a
        # pickle file runs whatever code is inside it, so accepting one from
        # another person would be a security risk. This advanced screening
        # feature can return in a safe format if a pilot user needs it.
        st.caption(
            "Advanced screening-model upload is turned off in this version. "
            "Contact us if you need it."
        )

        st.divider()

        # --- D. Property Constraints ---
        st.subheader("D. Property Constraints")
        st.caption(
            "Constraints on computed properties (e.g., total cost, total protein) "
            "based on ingredient properties from your CSV."
        )

        available_props = set()
        for p in st.session_state.optimizer.ingredient_properties.values():
            available_props.update(p.keys())

        if available_props:
            c_metric = st.selectbox("Constraint Metric", sorted(available_props))
            c_min = st.number_input("Min Value", value=0.0, key="prop_c_min")
            c_max = st.number_input("Max Value", value=100.0, key="prop_c_max")
            if st.button("Add Property Constraint"):
                try:
                    st.session_state.optimizer.add_constraint(c_metric, min_val=c_min, max_val=c_max)
                except ValueError as e:
                    st.error(str(e))
                else:
                    st.success("Property constraint added!")
        else:
            st.write("No properties found in CSV.")

        if st.session_state.optimizer.constraints:
            st.caption("Active property constraints:")
            st.dataframe(pd.DataFrame(st.session_state.optimizer.constraints))
            for i, constr in enumerate(st.session_state.optimizer.constraints):
                if st.button(f"Remove {constr['metric']}", key=f"rm_constr_{i}"):
                    st.session_state.optimizer.remove_constraint(i)
                    st.rerun()

        st.divider()

        # --- E. Ingredient Quantity Constraints ---
        st.subheader("E. Ingredient Quantity Constraints")
        st.caption(
            "Set upper/lower limits on the **sum of ingredient quantities**. "
            "Useful for synergistic effects (e.g., Sugar + Honey <= 50g) "
            "or constraining total recipe mass."
        )

        ingredient_names = [
            v['name'] for v in st.session_state.optimizer.variables
            if v.get('category', 'ingredient') == 'ingredient'
        ]

        if ingredient_names:
            with st.form("qty_constraint_form"):
                selected_ings = st.multiselect(
                    "Select Ingredients for Sum Constraint",
                    ingredient_names,
                    help="Select the ingredients whose combined amount you want to limit.",
                )
                qc_cols = st.columns(2)
                with qc_cols[0]:
                    qc_min = st.number_input(
                        "Min Sum", value=0.0, key="qc_min",
                        help="Leave at 0 for no lower bound",
                    )
                with qc_cols[1]:
                    qc_max = st.number_input(
                        "Max Sum", value=100.0, key="qc_max",
                        help="Upper limit for the sum",
                    )
                qc_use_min = st.checkbox("Apply minimum bound", value=False, key="qc_use_min")
                qc_use_max = st.checkbox("Apply maximum bound", value=True, key="qc_use_max")

                if st.form_submit_button("Add Quantity Constraint"):
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
                            st.success(f"Added quantity constraint on: {', '.join(selected_ings)}")
                    else:
                        st.error("Select at least 1 ingredient.")

            # Total mass shortcut
            st.caption("Or quickly add a total mass constraint:")
            tm_cols = st.columns(3)
            with tm_cols[0]:
                tm_min = st.number_input("Total Mass Min", value=0.0, key="tm_min")
            with tm_cols[1]:
                tm_max = st.number_input("Total Mass Max", value=100.0, key="tm_max")
            with tm_cols[2]:
                if st.button("Add Total Mass Constraint"):
                    try:
                        st.session_state.optimizer.add_total_mass_constraint(
                            min_val=tm_min, max_val=tm_max,
                        )
                    except ValueError as e:
                        st.error(str(e))
                    else:
                        st.success("Total mass constraint added!")

            # Show active quantity constraints
            qc_list = getattr(st.session_state.optimizer, 'quantity_constraints', [])
            if qc_list:
                st.caption("Active quantity constraints:")
                for i, qc in enumerate(qc_list):
                    label = " + ".join(qc['ingredients'])
                    bounds = []
                    if qc['min'] is not None:
                        bounds.append(f"min={qc['min']}")
                    if qc['max'] is not None:
                        bounds.append(f"max={qc['max']}")
                    qc_c1, qc_c2 = st.columns([4, 1])
                    with qc_c1:
                        st.text(f"[{i}] {label}: {', '.join(bounds)}")
                    with qc_c2:
                        if st.button("Remove", key=f"rm_qc_{i}"):
                            st.session_state.optimizer.remove_quantity_constraint(i)
                            st.rerun()
        else:
            st.write("Load ingredients first to add quantity constraints.")

        st.divider()
        with st.expander("Advanced: model settings (most people can skip this)"):
            st.caption(
                "Leave this on Standard unless you know the statistics behind "
                "the optimizer. Standard uses sensible defaults. 'Expert-selected' "
                "lets a specialist fix the model's kernel, prior, noise handling "
                "and acquisition once at the start."
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
                if _cur_cfg is not None and st.button("Apply: revert to defaults"):
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
                    if st.form_submit_button("Apply expert config"):
                        _opt.set_bo_config({
                            "kernel": _k, "lengthscale_prior": _lp,
                            "noise": _ns, "acquisition": _aq,
                        })
                        flash("success", "Model settings updated.")
                        st.rerun()
                if st.checkbox("Or paste an expert config as JSON", key="bo_cfg_paste"):
                    _txt = st.text_area(
                        "Expert config JSON",
                        value='{"kernel": "matern52", "lengthscale_prior": "default", '
                              '"noise": "default", "acquisition": "qlognei"}',
                        key="bo_cfg_json",
                    )
                    if st.button("Apply pasted config"):
                        try:
                            _opt.set_bo_config(json.loads(_txt))
                            flash("success", "Model settings updated.")
                            st.rerun()
                        except Exception as _e:
                            st.error(f"Invalid JSON: {_e}")
            if _cur_cfg:
                st.info(f"Active model settings: {_cur_cfg}")


# ================================================================== #
#  Tab 2: Optimization Loop
# ================================================================== #

with tab_optimize:
    # -- Backup warning after new data is recorded --
    if st.session_state.get("show_backup_warning"):
        st.warning("**Your experiment data has changed. Download a backup now!**", icon="\u26a0\ufe0f")
        opt = st.session_state.optimizer
        backup_json = json.dumps(opt.export_json(), indent=2)
        col_dl, col_dismiss = st.columns([1, 1])
        with col_dl:
            st.download_button(
                "\u2b07 Download Backup Now",
                data=backup_json,
                file_name=f"{opt.project_name}_backup.json",
                mime="application/json",
                type="primary",
                use_container_width=True,
            )
        with col_dismiss:
            if st.button("Dismiss", use_container_width=True):
                del st.session_state["show_backup_warning"]
                st.rerun()
        st.divider()

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
            st.markdown("\n".join(f"- {k}: {v:.2f}" for k, v in _best_recipe.items()))
        with sc2:
            _chart = pd.DataFrame({
                "Experiment": range(1, len(_opt.Y_history) + 1),
                "Overall Score": [float(y) for y in _opt.Y_history],
                "Best so far": _opt.best_so_far(),
            }).set_index("Experiment")
            st.line_chart(_chart, height=220)
            st.caption("Each experiment's Overall Score, and the best score reached so far. "
                       "A flattening line means the optimizer is converging.")

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
        st.subheader("Generate Recipes")
        _n_hist = len(st.session_state.optimizer.X_history)
        if _n_hist < 5:   # matches ask(n_init_random=5)
            st.info(f"Exploration phase: {_n_hist} of 5 baseline experiments done. The first "
                    "recipes are spread across your ranges to map the space; the optimizer "
                    "starts learning from your results after that.")
        else:
            st.caption(f"Optimizing — the model has learned from {_n_hist} experiments.")
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
                    st.session_state.optimizer.set_pending_batch(recipes)
                except ValueError as e:
                    st.error(str(e))
                except Exception:
                    # A raw traceback is a dead end for a nontechnical user.
                    st.error(
                        "The optimizer hit an unexpected problem while "
                        "generating recipes. Try again (a smaller batch "
                        "often helps); if this keeps happening, relax any "
                        "recently added constraints or use Help > Email "
                        "Support."
                    )

        if "current_batch" in st.session_state:
            st.info("Suggested Batch:")

            proc_names = {
                v['name'] for v in st.session_state.optimizer.variables
                if v.get('category') == 'process'
            }
            df_batch = pd.DataFrame(st.session_state.current_batch)

            ing_cols = [c for c in df_batch.columns if c not in proc_names]
            if ing_cols:
                st.caption("Ingredients:")
                st.dataframe(df_batch[ing_cols].style.format("{:.2f}"), hide_index=True)

            proc_cols = [c for c in df_batch.columns if c in proc_names]
            if proc_cols:
                st.caption("Process Parameters:")
                st.dataframe(df_batch[proc_cols].style.format("{:.2f}"), hide_index=True)

    # -------------------------------------------------------------- #
    #  Tell: Input Lab Results
    # -------------------------------------------------------------- #
    with col_tell:
        st.subheader("Enter Lab Results")

        batch = st.session_state.get("current_batch")
        if not batch:
            st.caption("Generate recipes on the left. A results form for each one appears here.")
        else:
            objs = st.session_state.optimizer.objectives
            batch_id = st.session_state.get("_batch_id", 0)
            with st.form("results_form"):
                batch_inputs, skipped = {}, set()
                for i, recipe in enumerate(batch):
                    st.markdown(f"**Recipe {i + 1}**")
                    st.caption(" · ".join(f"{k} {v:.2f}" for k, v in recipe.items()))
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
                            rec_scores[obj['name']] = st.number_input(
                                f"{obj['name']} ({obj['min_val']:g}–{obj['max_val']:g})",
                                min_value=float(obj['min_val']),
                                max_value=float(obj['max_val']),
                                value=None,
                                placeholder="enter measurement",
                                key=f"b{batch_id}_r{i}o{j}",
                            )
                    batch_inputs[i] = rec_scores
                    st.divider()

                if st.form_submit_button("Save Results"):
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
                            st.session_state.show_backup_warning = True
                            st.rerun()

    st.divider()

    # -------------------------------------------------------------- #
    #  Adaptive EGBO: revise the design space mid-run (arm 2)
    # -------------------------------------------------------------- #
    with st.expander("Adaptive EGBO: export trajectory & revise design space"):
        _opt = st.session_state.optimizer

        _act = _opt.active_variables()
        _inact = _opt.inactive_variables()
        ac1, ac2 = st.columns(2)
        with ac1:
            st.metric("Active variables |S|", len(_act))
        with ac2:
            st.metric("Pruned (still in GP)", len(_inact))

        st.markdown(
            "**1. Export the trajectory** to query the expert for ingredients to add "
            "or remove. Append your candidate pool (minus the active variables) "
            "before sending."
        )
        st.code(_opt.export_trajectory(), language="text")

        st.markdown(
            "**2. Add the expert's suggested variable(s)** to the design space. The "
            "history re-encodes automatically."
        )
        _has_hist = bool(_opt.X_history)
        # Radio outside the form so the fields react to the type choice.
        add_type = st.radio(
            "Type", ["Ingredient", "Process parameter"],
            horizontal=True, key="adaptive_add_type",
        )
        with st.form("adaptive_add_form"):
            new_name = st.text_input("Name", key="adaptive_new_name")
            nc1, nc2, nc3 = st.columns(3)
            with nc1:
                new_min = st.number_input(
                    "Min", value=0.0, key="adaptive_new_min",
                    disabled=(add_type == "Ingredient" and _has_hist),
                    help="Ingredients added mid-run have min 0 (absent in prior recipes).",
                )
            with nc2:
                new_max = st.number_input("Max", value=10.0, key="adaptive_new_max")
            with nc3:
                new_base = st.number_input(
                    "Baseline (process only)", value=0.0, key="adaptive_new_base",
                    disabled=(add_type == "Ingredient"),
                    help="The value used in ALL prior batches. Past experiments encode "
                         "at this value; must lie within [Min, Max].",
                )
            if add_type == "Ingredient":
                st.caption(
                    "Ingredient: absent (0) in every past recipe; mid-run its min is 0 "
                    "and BO decides how much to use."
                )
            else:
                st.caption(
                    "Process parameter: past batches ran at a fixed setting, so give the "
                    "Baseline (that setting) — it must lie within [Min, Max]."
                )
            if st.form_submit_button("Add to design space"):
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
                        st.session_state.show_backup_warning = True
                        flash(
                            "success",
                            f"Added {add_type.lower()} '{nm}' "
                            f"(bounds {_opt.variables[-1]['bounds']}). "
                            f"History re-encoded — generate a new batch."
                        )
                        st.rerun()
                    except ValueError as e:
                        st.error(str(e))

        st.divider()
        st.markdown(
            "**3. Prune the design space.** Deactivating a variable removes it from "
            "the active set without deleting anything: past experiments stay in the "
            "GP, and only the acquisition search is restricted (the variable is held "
            "at its pinned value). Fully reversible — this is what lets the active "
            "set shrink as well as grow, so expert false positives don't accumulate."
        )

        if len(_act) > 1:
            _off = st.multiselect(
                "Deactivate (prune from the active set)",
                [v['name'] for v in _act],
                key="egbo_deactivate",
                help="Ingredients pin at 0; process parameters pin at their baseline "
                     "or lower bound.",
            )
            if st.button("Deactivate selected", disabled=not _off):
                try:
                    for nm in _off:
                        _opt.deactivate_variable(nm)
                    st.session_state.pop("current_batch", None)
                    st.session_state.show_backup_warning = True
                    flash("success", f"Pruned: {', '.join(_off)} — generate a new batch.")
                    st.rerun()
                except ValueError as e:
                    st.error(str(e))
        else:
            st.caption("At least 2 active variables are needed before pruning.")

        if _inact:
            st.caption("Currently pruned — pinned during search, still in the GP:")
            st.dataframe(
                pd.DataFrame([
                    {
                        "Name": v['name'],
                        "Type": v.get('category', 'ingredient'),
                        "Pinned at": _opt._frozen_value(v),
                    }
                    for v in _inact
                ]),
                hide_index=True,
            )
            _on = st.multiselect(
                "Reactivate (return to the active set)",
                [v['name'] for v in _inact],
                key="egbo_reactivate",
            )
            if st.button("Reactivate selected", disabled=not _on):
                for nm in _on:
                    _opt.reactivate_variable(nm)
                st.session_state.pop("current_batch", None)
                st.session_state.show_backup_warning = True
                flash("success", f"Reactivated: {', '.join(_on)} — generate a new batch.")
                st.rerun()

        # Checkbox rather than an expander: Streamlit forbids nested expanders.
        if st.checkbox("Show permanent deletion (rarely needed)", key="egbo_show_del"):
            st.caption(
                "Deletion drops the ingredient's column from the encoded history. "
                "It is refused if the ingredient was ever used at a nonzero amount, "
                "since that would rewrite past experiments into recipes nobody ran. "
                "Deactivation above is almost always what you want."
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
                    f"Delete {_del} from this project for good? Past experiments are kept but "
                    "re-encoded without it. A copy of the project is archived first.",
                    confirm_label="Yes, delete",
                ):
                    try:
                        STORAGE.archive(_opt.project_name, "pre_delete", copy=True)
                        _opt.remove_ingredient(_del, force=_force)
                        st.session_state.pop("current_batch", None)
                        st.session_state.show_backup_warning = True
                        flash("success", f"Deleted '{_del}'.")
                        st.rerun()
                    except (ValueError, storage_backend.StorageError) as e:
                        st.error(str(e))
            else:
                st.caption("No ingredients loaded.")

    st.divider()

    # -------------------------------------------------------------- #
    #  Bulk Import Historical Experiments
    # -------------------------------------------------------------- #
    with st.expander("Import Historical Experiments (CSV)"):
        st.markdown(
            "Upload a CSV to bulk-import past experiments. "
            "Columns must match your **ingredient names** and **objective names** exactly."
        )
        st.caption(
            "Example: if you have ingredients `flour, sugar, butter` and objectives "
            "`Chewiness, Flavor`, your CSV needs columns: "
            "`flour, sugar, butter, Chewiness, Flavor`"
        )

        import_csv = st.file_uploader("Upload Experiments CSV", type=["csv"], key="import_csv")
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
                st.error(f"Missing columns: {missing}")
            else:
                nan_cols = [c for c in required if import_df[c].isna().any()]
                if nan_cols:
                    st.error(f"Columns with missing/NaN values: {nan_cols}")
                elif st.button("Import All Rows", type="primary"):
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
                    if imported:
                        st.session_state.show_backup_warning = True
                    if import_error is None and imported:
                        flash("success", f"Imported {imported} experiments.")
                        st.rerun()

    st.divider()

    # -------------------------------------------------------------- #
    #  Utility Score Explanation
    # -------------------------------------------------------------- #
    with st.expander("How is the Utility Score calculated?"):
        st.markdown("""
**The Utility Score** is a weighted combination of all your objectives, computed as follows:

1. **Normalize** each raw metric value to [0, 1] using the range you defined:
   - `normalized = (value - range_min) / (range_max - range_min)`
   - Values outside the range are clamped to [0, 1]

2. **Convert to utility** based on the optimization goal:
   - **Maximize**: `utility = normalized` (higher raw value = higher utility)
   - **Minimize**: `utility = 1 - normalized` (lower raw value = higher utility)
   - **Target**: `utility = max(0, 1 - |normalized - normalized_target|)`
     (closer to target = higher utility, with linear penalty for deviation)

3. **Weighted sum**: `Total Utility = sum(weight_i * utility_i)` across all objectives

**Example:** If you have Chewiness (goal=max, weight=0.6, range 0-10) and Sweetness
(goal=target at 5, weight=0.4, range 0-10):
- Chewiness score of 8 -> normalized = 0.8 -> utility = 0.8 -> weighted = 0.48
- Sweetness score of 6 -> normalized = 0.6, target_norm = 0.5 -> utility = 1 - 0.1 = 0.9 -> weighted = 0.36
- **Total Utility = 0.48 + 0.36 = 0.84**
        """)

    # -------------------------------------------------------------- #
    #  Experiment History (with Edit / Delete / Rewind)
    # -------------------------------------------------------------- #
    if st.session_state.optimizer.X_history:
        st.subheader("Experiment History")
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
                st.caption(f"Current results for experiment {edit_idx + 1}:")
                st.json(current_results)

                with st.form("edit_form"):
                    st.markdown(f"**Edit results for experiment {edit_idx + 1}:**")
                    new_results = {}
                    edit_cols = st.columns(len(st.session_state.optimizer.objectives))
                    for j, obj in enumerate(st.session_state.optimizer.objectives):
                        with edit_cols[j]:
                            current_val = float(current_results.get(obj['name'], 0.0))
                            new_results[obj['name']] = st.number_input(
                                obj['name'], value=current_val,
                                key=f"edit_{edit_idx}_{j}",
                            )

                    ec1, ec2 = st.columns(2)
                    with ec1:
                        if st.form_submit_button("Update Result"):
                            st.session_state.optimizer.edit_result(edit_idx, new_results)
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
                _desc = " · ".join(f"{k} {v:.2f}" for k, v in _row.items())
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
                    f"Experiment {edit_idx + 1} was recorded before edit tracking was enabled. "
                    "Only newer experiments can be edited."
                )
        else:
            st.info(
                "Edit capability is available for experiments recorded from this version onward. "
                "Older experiments (without stored raw results) cannot be edited."
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
