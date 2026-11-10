import json
import os
import glob
import shutil

import streamlit as st
import pandas as pd

from food_bo import DEFAULT_BO_CONFIG, FoodOptimizer

st.set_page_config(page_title="Food Optimizer", layout="wide")
st.title("Formulation Assistant")

# ================================================================== #
#  Sidebar: Project Management
# ================================================================== #

with st.sidebar:
    st.subheader("Project Management")

    existing_pkls = sorted(glob.glob("*.pkl"))
    existing_projects = [os.path.splitext(f)[0] for f in existing_pkls]

    project_name = st.text_input("Project Name", "my_project")

    if st.button("Create / Switch Project"):
        st.session_state.pop("optimizer", None)
        st.session_state.pop("current_batch", None)
        st.session_state["_loaded_project"] = project_name
        st.rerun()

    if existing_projects:
        st.caption("Or load an existing project:")
        selected = st.selectbox(
            "Existing Projects",
            ["(none)"] + existing_projects,
            key="project_select",
        )
        if selected != "(none)" and st.button("Load Selected Project"):
            st.session_state.pop("optimizer", None)
            st.session_state.pop("current_batch", None)
            st.session_state["_loaded_project"] = selected
            st.rerun()

    if "_loaded_project" in st.session_state:
        project_name = st.session_state["_loaded_project"]

    # Initialize or upgrade optimizer
    if "optimizer" not in st.session_state:
        st.session_state.optimizer = FoodOptimizer(project_name)
        st.success(f"Initialized: {project_name}")
    elif getattr(st.session_state.optimizer, 'CLASS_VERSION', 0) < FoodOptimizer.CLASS_VERSION:
        st.session_state.optimizer = FoodOptimizer(
            st.session_state.optimizer.project_name
        )
        st.success("Upgraded session to latest version.")

    opt = st.session_state.optimizer
    if getattr(opt, "load_error", None):
        st.error(opt.load_error)
    if not opt.X_history and not opt.pending_batch:
        st.error(
            "**This project has no experiments.** If a campaign is under way, the app "
            "has probably restarted after sleeping: **restore your latest backup** "
            "(below) before doing anything else."
        )
    _n_act = len(opt.active_variables())
    _n_all = len(opt.variables)
    _pruned = f" | {_n_all - _n_act} pruned" if _n_all > _n_act else ""
    st.caption(
        f"Active: **{opt.project_name}** | {len(opt.X_history)} experiments "
        f"| {opt.completed_rounds()} rounds | {_n_act} active vars{_pruned}"
    )

    # --- Backup & Restore ---
    st.divider()
    st.subheader("Backup & Restore")
    st.caption(
        "The downloaded backup **is** the record: the server copy can be lost "
        "when the app restarts. Download after every round and keep every file."
    )

    project_json = json.dumps(opt.export_json(), indent=2)
    st.download_button(
        "Download Project Backup",
        data=project_json,
        file_name=opt.backup_filename(),
        mime="application/json",
    )

    uploaded_json = st.file_uploader("Restore from backup", type=["json"], key="restore_json")
    if uploaded_json is not None:
        if st.button("Restore Project"):
            try:
                state = json.loads(uploaded_json.getvalue())
                state['project_name'] = st.session_state.optimizer.project_name
                st.session_state.optimizer.import_json(state)
                st.session_state.pop("current_batch", None)
                st.session_state["_restored_msg"] = (
                    f"Restored {len(st.session_state.optimizer.X_history)} experiments "
                    f"({st.session_state.optimizer.completed_rounds()} rounds) into "
                    f"{st.session_state.optimizer.project_name}."
                )
                st.rerun()
            except (ValueError, KeyError, TypeError) as e:
                st.error(f"Restore failed, nothing was changed: {e}")
    if st.session_state.get("_restored_msg"):
        st.success(st.session_state.pop("_restored_msg"))

    # --- Hard Reset ---
    st.divider()

    if st.button("Hard Reset Project"):
        fname = f"{project_name}.pkl"
        if os.path.exists(fname):
            archive_name = f"{project_name}_archived.pkl"
            counter = 1
            while os.path.exists(archive_name):
                archive_name = f"{project_name}_archived_{counter}.pkl"
                counter += 1
            os.rename(fname, archive_name)
            st.info(f"Archived as {archive_name}")
        st.session_state.pop("optimizer", None)
        st.session_state.pop("_loaded_project", None)
        st.session_state.pop("current_batch", None)
        st.rerun()


# ================================================================== #
#  Tab 1: Setup & Config
# ================================================================== #

tab_setup, tab_optimize = st.tabs(["1. Setup & Config", "2. Optimization Loop"])

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
        if uploaded_csv:
            df = pd.read_csv(uploaded_csv)
            st.dataframe(df.head(), height=150)
            if st.button("Load Ingredients"):
                try:
                    st.session_state.optimizer.load_ingredients_from_csv(df)
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
            pp_cols = st.columns(3)
            with pp_cols[0]:
                pp_name = st.text_input("Parameter Name", placeholder="e.g. Baking_Temp")
            with pp_cols[1]:
                pp_min = st.number_input("Min Value", value=0.0, key="pp_min")
            with pp_cols[2]:
                pp_max = st.number_input("Max Value", value=100.0, key="pp_max")
            if st.form_submit_button("Add Process Parameter"):
                if st.session_state.optimizer.X_history:
                    st.error(
                        "Experiments already exist. Add process parameters mid-run in the "
                        "Adaptive EGBO panel (Optimization Loop tab), which asks for the "
                        "baseline setting used so far."
                    )
                elif pp_name:
                    try:
                        st.session_state.optimizer.add_process_parameter(pp_name, pp_min, pp_max)
                        st.success(f"Added process parameter: {pp_name}")
                    except ValueError as e:
                        st.error(str(e))

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
                    if st.button("Remove", key=f"rm_pp_{i}",
                                 disabled=bool(st.session_state.optimizer.X_history),
                                 help="After experiments exist, deactivate instead "
                                      "(Adaptive EGBO panel)."):
                        try:
                            st.session_state.optimizer.remove_process_parameter(pv['name'])
                            st.rerun()
                        except ValueError as e:
                            st.error(str(e))

        st.divider()

        # --- B. Objectives ---
        st.subheader("B. Objectives (Normalized)")
        st.caption("Define the valid range for each metric to normalize scores (0-1).")

        with st.form("obj_form"):
            col_name, col_w = st.columns([2, 1])
            with col_name:
                obj_name = st.text_input("Metric Name (e.g. Chewiness)")
            with col_w:
                obj_weight = st.slider("Weight", 0.0, 1.0, 0.1)

            col_g, col_min, col_max = st.columns(3)
            with col_g:
                obj_goal = st.selectbox("Goal", ["max", "min", "target"])
            with col_min:
                obj_min = st.number_input("Range Min", value=0.0)
            with col_max:
                obj_max = st.number_input("Range Max", value=10.0)

            obj_target = st.number_input("Target Value (If Goal=Target)", value=5.0)

            if st.form_submit_button("Add Objective"):
                if not obj_name.strip():
                    st.error("Objective name cannot be empty.")
                elif obj_min >= obj_max:
                    st.error("Range Min must be less than Range Max.")
                else:
                    try:
                        st.session_state.optimizer.add_objective(
                            obj_name.strip(), obj_weight, obj_goal,
                            target=obj_target if obj_goal == 'target' else None,
                            min_val=obj_min, max_val=obj_max,
                        )
                        st.success(f"Added {obj_name}")
                    except ValueError as e:
                        st.error(str(e))

        if st.session_state.optimizer.objectives:
            st.write("Active Objectives:")
            st.dataframe(pd.DataFrame(st.session_state.optimizer.objectives))
            for i, obj in enumerate(st.session_state.optimizer.objectives):
                if st.button(f"Remove {obj['name']}", key=f"rm_obj_{i}"):
                    st.session_state.optimizer.remove_objective(obj['name'])
                    st.rerun()

    # -------------------------------------------------------------- #
    #  Column B: Constraints, BO settings, campaign settings
    # -------------------------------------------------------------- #
    with col_b:
        # --- C. Property Constraints ---
        st.subheader("C. Property Constraints")
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
                st.session_state.optimizer.add_constraint(c_metric, min_val=c_min, max_val=c_max)
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

        # --- D. Ingredient Quantity Constraints ---
        st.subheader("D. Ingredient Quantity Constraints")
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
                    help="Select 2+ ingredients to constrain their combined quantity",
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
                        st.session_state.optimizer.add_quantity_constraint(
                            selected_ings,
                            min_val=qc_min if qc_use_min else None,
                            max_val=qc_max if qc_use_max else None,
                        )
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
                    st.session_state.optimizer.add_total_mass_constraint(
                        min_val=tm_min, max_val=tm_max,
                    )
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
        st.subheader("E. BO Hyperparameters")
        _opt = st.session_state.optimizer
        _cur_cfg = getattr(_opt, "bo_config", None)
        if _opt.X_history:
            st.info(
                "**Locked for this campaign** (experiments exist): "
                f"{_cur_cfg if _cur_cfg else 'Standard (library defaults)'}"
            )
        else:
            st.caption(
                "Fixed **once**, before the first recipe; the GP still refits "
                "lengthscales and noise from data every round. Campaign default: "
                "Matérn-5/2, default lengthscale prior, learned noise, qLogNEI "
                "(the configuration tested in simulation)."
            )
            if _cur_cfg != DEFAULT_BO_CONFIG:
                if st.button("Use campaign config (recommended)", type="primary"):
                    _opt.set_bo_config(dict(DEFAULT_BO_CONFIG))
                    st.rerun()
            _mode = st.radio(
                "Other options",
                ["Keep current", "Standard (library defaults)", "Custom"],
                key="bo_cfg_mode", horizontal=True,
            )
            if _mode == "Standard (library defaults)" and _cur_cfg is not None:
                if st.button("Apply: library defaults"):
                    _opt.set_bo_config(None)
                    st.rerun()
            elif _mode == "Custom":
                with st.form("bo_config_form"):
                    bc1, bc2 = st.columns(2)
                    with bc1:
                        _k = st.selectbox("Kernel", ["matern52", "matern32", "rbf", "linear", "poly2"])
                        _lp = st.selectbox("Lengthscale prior", ["default", "long", "short"])
                    with bc2:
                        _ns = st.selectbox("Noise", ["default", "low", "fixed_tiny"])
                        _aq = st.selectbox("Acquisition", ["qlognei", "qlogei", "qucb"])
                    st.caption(
                        "`fixed_tiny` noise suits a deterministic objective, not a noisy "
                        "sensory panel — keep `default` unless you have a specific reason."
                    )
                    if st.form_submit_button("Apply custom config"):
                        _opt.set_bo_config({
                            "kernel": _k, "lengthscale_prior": _lp,
                            "noise": _ns, "acquisition": _aq,
                        })
                        st.rerun()
            st.info(f"Current: {_cur_cfg if _cur_cfg else 'Standard (library defaults)'}")

        st.divider()
        st.subheader("F. Adaptive EGBO campaign settings")
        _ce = st.number_input(
            "Expert check every N rounds (0 = no scheduled checks)",
            min_value=0, max_value=20, value=int(_opt.check_every), step=1, key="check_every_in",
        )
        if int(_ce) != int(_opt.check_every):
            _opt.set_check_every(_ce)
            st.rerun()

        _pool = st.text_area(
            "Candidates the expert may add (ingredients or process changes, one per line, "
            "with a plausible range). Included in the expert report.",
            value=_opt.candidate_pool, key="cand_pool_in", height=120,
        )
        if st.button("Save candidate list"):
            _opt.set_candidate_pool(_pool)
            st.success("Saved.")

        st.markdown("**Balance ingredient** (e.g. water): not searched over, but filled "
                    "up to the batch total, so the batch size stays fixed.")
        _bal = _opt.balance
        if _bal:
            st.info(f"Balance: **{_bal['name']}** fills the batch to {_bal['total']:g} "
                    f"(kept between {_bal['min']:g} and "
                    f"{'no limit' if _bal['max'] is None else format(_bal['max'], 'g')}).")
        if not _opt.X_history:
            with st.form("balance_form"):
                bl1, bl2, bl3, bl4 = st.columns(4)
                with bl1:
                    _bn = st.text_input("Ingredient", value=_bal['name'] if _bal else "Tap water")
                with bl2:
                    _bt = st.number_input("Batch total", value=float(_bal['total']) if _bal else 1000.0)
                with bl3:
                    _bmin = st.number_input("Min amount", value=float(_bal['min']) if _bal else 0.0)
                with bl4:
                    _bmax = st.number_input("Max amount (0 = none)",
                                            value=float(_bal['max'] or 0.0) if _bal else 0.0)
                bc1, bc2 = st.columns(2)
                with bc1:
                    _set_bal = st.form_submit_button("Set balance ingredient")
                with bc2:
                    _clr_bal = st.form_submit_button("Clear", disabled=not _bal)
                if _set_bal:
                    try:
                        _opt.set_balance(_bn.strip(), _bt, _bmin, _bmax or None)
                        st.rerun()
                    except ValueError as e:
                        st.error(str(e))
                if _clr_bal:
                    _opt.set_balance(None, 0)
                    st.rerun()


# ================================================================== #
#  Tab 2: Optimization Loop
# ================================================================== #

with tab_optimize:
    _opt = st.session_state.optimizer
    _r_done = _opt.completed_rounds()
    _pb = _opt.pending_batch

    # -- Round status and the expert-check schedule --
    s1, s2, s3 = st.columns(3)
    with s1:
        st.metric("Rounds completed", _r_done)
    with s2:
        st.metric("Current round", _pb['round'] if _pb else _r_done + 1)
    with s3:
        if _opt.check_every:
            _next = (_r_done // _opt.check_every + (0 if _opt.check_due() else 1)) * _opt.check_every
            st.metric("Expert check after round", _next)
        else:
            st.metric("Expert check after round", "—")
    if _opt.check_due():
        st.warning(
            f"**Expert check due after round {_r_done}.** Open *Adaptive EGBO* below: "
            f"prepare the expert report, make any changes, and record the decision "
            f"(also when nothing changes) before generating round {_r_done + 1}.",
            icon="\U0001F9D1‍\U0001F52C",
        )

    # -- Backup warning after new data is recorded --
    if st.session_state.get("show_backup_warning"):
        st.warning("**Your experiment data has changed. Download a backup now!**", icon="⚠️")
        backup_json = json.dumps(_opt.export_json(), indent=2)
        col_dl, col_dismiss = st.columns([1, 1])
        with col_dl:
            st.download_button(
                "⬇ Download Backup Now",
                data=backup_json,
                file_name=_opt.backup_filename(),
                mime="application/json",
                type="primary",
                use_container_width=True,
            )
        with col_dismiss:
            if st.button("Dismiss", use_container_width=True):
                del st.session_state["show_backup_warning"]
                st.rerun()
        st.divider()

    col_ask, col_tell = st.columns([1, 1.5])
    _proc_names = {v['name'] for v in _opt.variables if v.get('category') == 'process'}

    # -------------------------------------------------------------- #
    #  Ask: Generate Experiments
    # -------------------------------------------------------------- #
    with col_ask:
        st.subheader("Generate Experiments")
        batch_size = st.slider("Batch Size", 1, 10, 5)

        _override = False
        if _opt.check_due():
            _override = st.checkbox(
                "Generate without recording the expert check (off-protocol)",
                key="skip_check",
            )
        _empty = not _opt.X_history and not _pb
        _new_ok = False
        if _empty:
            st.error(
                "No experiments in this project. If the campaign has already started, "
                "**restore your latest backup** (sidebar) first: generating now would "
                "start over with round-1 recipes."
            )
            _new_ok = st.checkbox("This is a brand-new campaign (no earlier rounds)",
                                  key="confirm_new_campaign")
        if _pb:
            st.caption(
                f"A round-{_pb['round']} batch is waiting for scores. Generating again "
                f"replaces it — only do that if it hasn't been made yet."
            )
        if st.button(f"Generate {batch_size} Recipes", type="primary", key="generate_btn",
                     disabled=(_opt.check_due() and not _override) or (_empty and not _new_ok)):
            if not _opt.variables:
                st.error("Please load ingredients in Setup tab first!")
            elif not _opt.objectives:
                st.error("Please define objectives in Setup tab first!")
            else:
                with st.spinner("Optimizing..."):
                    try:
                        _opt.ask(n_suggestions=batch_size)
                        st.session_state.show_backup_warning = True
                        st.rerun()
                    except ValueError as e:
                        st.error(str(e))

        if _opt.pending_batch:
            _pb = _opt.pending_batch
            st.info(f"Round {_pb['round']} batch (generated {_pb.get('time', '')}):")
            df_batch = pd.DataFrame(_pb['recipes'])
            df_batch.index = [f"Recipe #{i + 1}" for i in range(len(df_batch))]

            ing_cols = [c for c in df_batch.columns if c not in _proc_names]
            if ing_cols:
                st.caption("Ingredients:")
                st.dataframe(df_batch[ing_cols].style.format("{:.3g}"))

            proc_cols = [c for c in df_batch.columns if c in _proc_names]
            if proc_cols:
                st.caption("Process Parameters:")
                st.dataframe(df_batch[proc_cols].style.format("{:.3g}"))
            st.download_button(
                "Download batch (CSV)", df_batch.to_csv(),
                file_name=f"{_opt.project_name}_R{_pb['round']:02d}_batch.csv",
            )

    # -------------------------------------------------------------- #
    #  Tell: Input Lab Results
    # -------------------------------------------------------------- #
    with col_tell:
        st.subheader("Input Lab Results")

        if _opt.pending_batch:
            _pb = _opt.pending_batch
            with st.form("results_form"):
                batch_inputs = {}
                for i, recipe in enumerate(_pb['recipes']):
                    st.markdown(f"**Recipe #{i + 1}**")
                    cols = st.columns(len(_opt.objectives))
                    rec_scores = {}
                    for j, obj in enumerate(_opt.objectives):
                        with cols[j]:
                            rec_scores[obj['name']] = st.number_input(
                                f"{obj['name']} ({obj['min_val']}-{obj['max_val']})",
                                key=f"r{_pb['round']}_{i}o{j}",
                            )
                    batch_inputs[i] = rec_scores
                    st.divider()

                if st.form_submit_button("Save Results"):
                    for i, recipe in enumerate(_pb['recipes']):
                        _opt.tell(recipe, batch_inputs[i], role='bo')
                    _opt.clear_pending_batch()
                    st.session_state.show_backup_warning = True
                    st.session_state.pop("expert_report", None)
                    st.rerun()
        else:
            st.caption("Generate a batch first.")

        with st.expander("Add a non-BO sample (e.g. an extra recipe the team wants tasted)"):
            if _opt.variables and _opt.objectives:
                _m_df = pd.DataFrame([
                    {"Variable": v['name'], "Amount": float(v.get('_absent_value', 0.0))}
                    for v in _opt.variables if v['type'] == 'continuous'
                ])
                with st.form("manual_form"):
                    _m_ed = st.data_editor(_m_df, hide_index=True, disabled=["Variable"],
                                           key="manual_editor")
                    _m_round = st.number_input("Round", min_value=0, value=max(_r_done, 1), step=1)
                    _m_gp = st.checkbox("Use in the model", value=True)
                    cols = st.columns(len(_opt.objectives))
                    _m_scores = {}
                    for j, obj in enumerate(_opt.objectives):
                        with cols[j]:
                            _m_scores[obj['name']] = st.number_input(
                                f"{obj['name']} ({obj['min_val']}-{obj['max_val']})",
                                key=f"man_o{j}",
                            )
                    if st.form_submit_button("Save sample"):
                        _rec = {r["Variable"]: float(r["Amount"]) for _, r in _m_ed.iterrows()}
                        _opt.tell(_rec, _m_scores, role='manual', in_gp=_m_gp,
                                  round_num=int(_m_round))
                        st.session_state.show_backup_warning = True
                        st.rerun()
            else:
                st.caption("Set up ingredients and objectives first.")

    st.divider()

    # -------------------------------------------------------------- #
    #  Finalists: rank by predicted utility, not the best single score
    # -------------------------------------------------------------- #
    with st.expander("Finalists (ranked by the model's predicted utility)"):
        st.caption(
            "A single tasting is noisy, so the best observed score is often a lucky one. "
            "The model's predicted utility pools evidence from similar recipes. Re-taste "
            "the top 2–3 with more panelists before choosing the final recipe."
        )
        if st.button("Compute finalists"):
            with st.spinner("Fitting the model..."):
                _fin = _opt.finalists(5)
            _rows = []
            for f in _fin:
                row = {"Exp #": f['index'], "Round": f['round'], "Observed": round(f['observed'], 3)}
                if f['predicted'] is not None:
                    row["Predicted"] = round(f['predicted'], 3)
                    row["± sd"] = round(f['sd'], 3)
                rec = _opt.recipe_history[f['index']]
                row.update({k: round(v, 3) for k, v in rec.items() if v})
                _rows.append(row)
            st.dataframe(pd.DataFrame(_rows), hide_index=True)

    # -------------------------------------------------------------- #
    #  Adaptive EGBO: revise the design space at the scheduled checks
    # -------------------------------------------------------------- #
    with st.expander("Adaptive EGBO: expert check & revise design space",
                     expanded=_opt.check_due()):
        _act = _opt.active_variables()
        _inact = _opt.inactive_variables()
        ac1, ac2 = st.columns(2)
        with ac1:
            st.metric("Active variables |S|", len(_act))
        with ac2:
            st.metric("Pruned (still in GP)", len(_inact))

        st.markdown(
            "**1. Prepare the expert report.** It has the objective definitions, the best "
            "recipes with each attribute's gap and share of the loss, the current ranges "
            "(flagging variables stuck at a limit), all "
            "experiments by round, and earlier decisions. Give it to the expert (FSA)."
        )
        if st.button("Prepare expert report"):
            with st.spinner("Fitting the model..."):
                st.session_state.expert_report = _opt.expert_report()
        if st.session_state.get("expert_report"):
            _txt, _snap = st.session_state.expert_report
            st.download_button(
                "Download report (.md)", _txt,
                file_name=f"{_opt.project_name}_check_after_R{_snap['after_round']:02d}.md",
            )
            with st.container(height=400):
                st.markdown(_txt)

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
                new_min = st.number_input("Min", value=0.0, key="adaptive_new_min")
            with nc2:
                new_max = st.number_input("Max", value=10.0, key="adaptive_new_max")
            with nc3:
                new_base = st.number_input(
                    "Amount / setting used so far", value=0.0, key="adaptive_new_base",
                    disabled=not _has_hist,
                    help="The amount (ingredient) or setting (process) used in ALL earlier "
                         "recipes. 0 for a brand-new ingredient. Must lie within [Min, Max].",
                )
            if add_type == "Ingredient":
                st.caption(
                    "Ingredient: give the amount used in every earlier recipe — 0 if it is "
                    "new, or its fixed amount if it was in the base recipe but not being "
                    "varied (e.g. lecithin held at 5 g)."
                )
            else:
                st.caption(
                    "Process parameter: past batches ran at a fixed setting — give that "
                    "setting. It must lie within [Min, Max]."
                )
            if st.form_submit_button("Add to design space"):
                nm = new_name.strip()
                if not nm:
                    st.error("Enter a name.")
                else:
                    try:
                        if add_type == "Ingredient":
                            _opt.add_ingredient(nm, new_min, new_max,
                                                baseline=(new_base if _has_hist else None))
                        else:
                            _opt.add_process_parameter(
                                nm, new_min, new_max,
                                baseline=(new_base if _has_hist else None),
                            )
                        st.session_state.pop("current_batch", None)  # stale under new dim
                        st.session_state.show_backup_warning = True
                        st.success(
                            f"Added {add_type.lower()} '{nm}' "
                            f"(bounds {_opt.variables[-1]['bounds']}). "
                            f"History re-encoded — generate a new batch."
                        )
                        st.rerun()
                    except ValueError as e:
                        st.error(str(e))

        st.divider()
        st.markdown(
            "**3. Widen or narrow a range.** Past experiments are kept: after a "
            "narrowing, recipes outside the new range still inform the model, and only "
            "new recipes are restricted to it."
        )
        _cont = [v for v in _opt.variables if v['type'] == 'continuous']
        if _cont:
            _rv_name = st.selectbox(
                "Variable", [v['name'] for v in _cont],
                key="range_pick_" + "|".join(v['name'] for v in _cont),
                format_func=lambda n: (
                    f"{n}  [{_opt._var_by_name(n)['bounds'][0]:g}, "
                    f"{_opt._var_by_name(n)['bounds'][1]:g}]"
                    + ("" if _opt._var_by_name(n).get('active', True) else "  (pruned)")
                ),
            )
            _rv = _opt._var_by_name(_rv_name)
            with st.form("range_form"):
                rr1, rr2 = st.columns(2)
                with rr1:
                    _rlo = st.number_input("New min", value=float(_rv['bounds'][0]),
                                           key=f"rng_lo_{_rv_name}", format="%.4g")
                with rr2:
                    _rhi = st.number_input("New max", value=float(_rv['bounds'][1]),
                                           key=f"rng_hi_{_rv_name}", format="%.4g")
                if st.form_submit_button("Apply new range"):
                    try:
                        _opt.set_variable_bounds(_rv_name, _rlo, _rhi)
                        st.session_state.show_backup_warning = True
                        st.session_state["_range_msg"] = (
                            f"'{_rv_name}' range is now [{_rlo:g}, {_rhi:g}] — generate a new batch.")
                        st.rerun()
                    except ValueError as e:
                        st.error(str(e))
            if st.session_state.get("_range_msg"):
                st.success(st.session_state.pop("_range_msg"))

        st.divider()
        st.markdown(
            "**4. Prune the design space.** Deactivating a variable removes it from "
            "the active set without deleting anything: past experiments stay in the "
            "GP, and only the acquisition search is restricted (the variable is held "
            "at the amount you choose). Fully reversible — this is what lets the active "
            "set shrink as well as grow, so expert false positives don't accumulate."
        )

        if len(_act) > 1:
            _off = st.selectbox("Deactivate (stop varying)", [v['name'] for v in _act],
                                key="egbo_deactivate_" + "|".join(v['name'] for v in _act))
            _offv = _opt._var_by_name(_off)
            _best_i = max(range(len(_opt.Y_history)), key=lambda i: _opt.Y_history[i]) \
                if _opt.Y_history else None
            _default_hold = float(_opt.recipe_history[_best_i].get(
                _off, _offv.get('_absent_value', 0.0))) if _best_i is not None \
                else float(_offv.get('_absent_value', _opt._frozen_value(_offv)))
            with st.form(f"hold_form_{_off}"):
                _hold = st.number_input(
                    f"Hold {_off} at (range {_offv['bounds'][0]:g}–{_offv['bounds'][1]:g})",
                    value=_default_hold, key=f"hold_{_off}", format="%.4g",
                    help="Defaults to its amount in the best recipe so far. Use 0 to drop "
                         "an ingredient entirely.",
                )
                if st.form_submit_button("Deactivate"):
                    try:
                        _opt.deactivate_variable(_off, value=_hold)
                        st.session_state.show_backup_warning = True
                        st.session_state["_prune_msg"] = (
                            f"'{_off}' is now held at {_hold:g} — generate a new batch.")
                        st.rerun()
                    except ValueError as e:
                        st.error(str(e))
            if st.session_state.get("_prune_msg"):
                st.success(st.session_state.pop("_prune_msg"))
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
                key="egbo_reactivate_" + "|".join(v['name'] for v in _inact),
            )
            if st.button("Reactivate selected", disabled=not _on):
                for nm in _on:
                    _opt.reactivate_variable(nm)
                st.session_state.pop("current_batch", None)
                st.session_state.show_backup_warning = True
                st.success(f"Reactivated: {', '.join(_on)} — generate a new batch.")
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
                if st.button("Delete permanently", key="egbo_del_btn"):
                    try:
                        _opt.remove_ingredient(_del, force=_force)
                        st.session_state.pop("current_batch", None)
                        st.session_state.show_backup_warning = True
                        st.success(f"Deleted '{_del}'.")
                        st.rerun()
                    except ValueError as e:
                        st.error(str(e))
            else:
                st.caption("No ingredients loaded.")

        st.divider()
        st.markdown(
            "**5. Record the check** — every scheduled check, including when nothing "
            "changes. The report prepared in step 1 is saved with it."
        )
        if not st.session_state.get("expert_report"):
            st.caption("Prepare the expert report (step 1) first, so it is saved with the decision.")
        with st.form("record_check_form"):
            _dec = st.text_input("Decision (e.g. 'no change', or 'added X; widened Y to 0–8%')")
            _why = st.text_area("Reasoning (which attributes and evidence it was based on)")
            _by = st.text_input("Decided by (e.g. FSA)")
            if st.form_submit_button("Record check",
                                     disabled=not st.session_state.get("expert_report")):
                if not _dec.strip():
                    st.error("Enter the decision (write 'no change' if nothing changed).")
                else:
                    _opt.record_check(_dec, _why, by=_by,
                                      report=st.session_state.expert_report[1])
                    st.session_state.show_backup_warning = True
                    st.session_state.pop("expert_report", None)
                    st.rerun()

        if _opt.design_log:
            st.caption("Design log (checks and changes after data exists):")
            st.dataframe(pd.DataFrame([
                {
                    "After round": e.get('after_round'),
                    "Type": e.get('type'),
                    "What": (e.get('decision') if e.get('type') == 'check' else
                             ", ".join(f"{k}={v}" for k, v in e.items()
                                       if k not in ('type', 'after_round', 'time', 'report',
                                                    'rationale', 'by', 'decision'))),
                    "Reasoning": e.get('rationale', ''),
                    "By": e.get('by', ''),
                    "Time": e.get('time'),
                }
                for e in _opt.design_log
            ]), hide_index=True)

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
        if import_csv is not None:
            import_df = pd.read_csv(import_csv)
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
                    for _, row in import_df.iterrows():
                        recipe = {name: float(row[name]) for name in var_names}
                        results = {name: float(row[name]) for name in obj_names}
                        st.session_state.optimizer.tell(recipe, results, role='import')
                        imported += 1
                    st.session_state.show_backup_warning = True
                    st.success(f"Imported {imported} experiments!")
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
        st.session_state.optimizer._sync_meta()

        hist = []
        for i, x in enumerate(st.session_state.optimizer.X_history):
            row = st.session_state.optimizer._decode(x)
            row['_index'] = i
            _meta = st.session_state.optimizer.meta_history[i]
            row['Round'] = _meta.get('round')
            row['Kind'] = _meta.get('role') + ("" if _meta.get('in_gp', True) else " (not in model)")
            row['Total_Utility_Score'] = st.session_state.optimizer.Y_history[i]

            res_hist = getattr(st.session_state.optimizer, 'results_history', [])
            if i < len(res_hist):
                for key, val in res_hist[i].items():
                    row[f"[Result] {key}"] = val
            hist.append(row)

        hist_df = pd.DataFrame(hist).sort_values('Total_Utility_Score', ascending=False)
        hist_df.insert(0, 'Exp #', hist_df['_index'])
        st.dataframe(hist_df.drop(columns=['_index']), hide_index=True)

        # --- Edit / Delete ---
        st.caption("Edit or delete a past result:")
        results_history = getattr(st.session_state.optimizer, 'results_history', [])

        if results_history:
            edit_idx = st.number_input(
                "Experiment Index to Edit (0-based)",
                min_value=0,
                max_value=len(st.session_state.optimizer.X_history) - 1,
                value=0, step=1, key="edit_idx",
            )

            if edit_idx < len(results_history):
                current_results = results_history[edit_idx]
                st.caption(f"Current results for experiment #{edit_idx}:")
                st.json(current_results)

                with st.form("edit_form"):
                    st.markdown(f"**Edit results for experiment #{edit_idx}:**")
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
                            st.success(f"Updated experiment #{edit_idx}!")
                            st.rerun()

                if st.session_state.get('_edit_warning_idx') is not None:
                    warn_idx = st.session_state._edit_warning_idx
                    warn_n = st.session_state._edit_warning_n
                    st.warning(
                        f"Experiments after #{warn_idx} ({warn_n} total) were based on "
                        f"the pre-edit ratings and may no longer be valid. "
                        f"Consider rewinding to #{warn_idx}."
                    )
                    del st.session_state._edit_warning_idx
                    del st.session_state._edit_warning_n

                if st.button(f"Delete Experiment #{edit_idx}", key="delete_btn"):
                    st.session_state.optimizer.delete_result(edit_idx)
                    st.success(f"Deleted experiment #{edit_idx}!")
                    st.rerun()
            else:
                st.warning(
                    f"Experiment #{edit_idx} was recorded before edit tracking was enabled. "
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
        st.info(
            "Rewind keeps experiments 0 through N and **discards all later ones**. "
            "The current project is archived first so nothing is permanently lost."
        )
        rewind_idx = st.number_input(
            "Keep experiments up to (0-based)",
            min_value=0,
            max_value=len(st.session_state.optimizer.X_history) - 1,
            value=len(st.session_state.optimizer.X_history) - 1,
            step=1, key="rewind_idx",
        )
        n_discard = len(st.session_state.optimizer.X_history) - 1 - rewind_idx
        if n_discard > 0:
            st.warning(
                f"This will discard {n_discard} experiment(s) "
                f"(#{rewind_idx + 1} through #{len(st.session_state.optimizer.X_history) - 1})."
            )
        if st.button("Rewind", disabled=(n_discard == 0), key="rewind_btn"):
            pname = st.session_state.optimizer.project_name
            fname = f"{pname}.pkl"
            if os.path.exists(fname):
                archive_name = f"{pname}_pre_rewind.pkl"
                counter = 1
                while os.path.exists(archive_name):
                    archive_name = f"{pname}_pre_rewind_{counter}.pkl"
                    counter += 1
                shutil.copy2(fname, archive_name)
                st.info(f"Archived current state as {archive_name}")
            st.session_state.optimizer.rewind_to(rewind_idx)
            st.session_state.pop("current_batch", None)
            st.success(f"Rewound to experiment #{rewind_idx}!")
            st.rerun()
