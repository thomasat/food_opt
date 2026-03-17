import json
import streamlit as st
import pandas as pd
import pickle
import os
import glob
from food_bo import FoodOptimizer

st.set_page_config(page_title="Food Optimizer", layout="wide")
st.title("Formulation Assistant")

# --- SIDEBAR: Project Management ---
with st.sidebar:
    st.subheader("Project Management")

    # Scan for existing projects
    existing_pkls = sorted(glob.glob("*.pkl"))
    existing_projects = [os.path.splitext(f)[0] for f in existing_pkls]

    project_name = st.text_input("Project Name", "Cookie_Project_v4")

    if existing_projects:
        st.caption("Or load an existing project:")
        selected = st.selectbox(
            "Existing Projects",
            ["(none)"] + existing_projects,
            key="project_select"
        )
        if selected != "(none)" and st.button("Load Selected Project"):
            project_name = selected
            if "optimizer" in st.session_state:
                del st.session_state.optimizer
            if "current_batch" in st.session_state:
                del st.session_state.current_batch
            st.session_state["_loaded_project"] = selected
            st.rerun()

    # Restore loaded project name
    if "_loaded_project" in st.session_state:
        project_name = st.session_state["_loaded_project"]

    if "optimizer" not in st.session_state:
        st.session_state.optimizer = FoodOptimizer(project_name)
        st.success(f"Initialized: {project_name}")
    elif getattr(st.session_state.optimizer, 'CLASS_VERSION', 0) < FoodOptimizer.CLASS_VERSION:
        # Re-instantiate with current class to pick up new methods/attrs
        st.session_state.optimizer = FoodOptimizer(
            st.session_state.optimizer.project_name
        )
        st.success("Upgraded session to latest version.")

    # Show current project info
    opt = st.session_state.optimizer
    st.caption(f"Active: **{opt.project_name}** | "
               f"{len(opt.X_history)} experiments")

    # --- Backup: Download / Upload JSON ---
    st.divider()
    st.subheader("Backup & Restore")

    # Download
    project_json = json.dumps(opt.export_json(), indent=2)
    st.download_button(
        "Download Project Backup",
        data=project_json,
        file_name=f"{opt.project_name}.json",
        mime="application/json",
    )

    # Upload / Restore
    uploaded_json = st.file_uploader("Restore from backup", type=["json"], key="restore_json")
    if uploaded_json is not None:
        if st.button("Restore Project"):
            state = json.loads(uploaded_json.read())
            # Keep current project name to avoid desync
            state['project_name'] = st.session_state.optimizer.project_name
            st.session_state.optimizer.import_json(state)
            st.success(f"Restored {len(st.session_state.optimizer.X_history)} experiments "
                        f"into {st.session_state.optimizer.project_name}")
            st.rerun()

    st.divider()

    if st.button("Hard Reset Project"):
        fname = f"{project_name}.pkl"
        if os.path.exists(fname):
            # Archive instead of deleting
            archive_name = f"{project_name}_archived.pkl"
            counter = 1
            while os.path.exists(archive_name):
                archive_name = f"{project_name}_archived_{counter}.pkl"
                counter += 1
            os.rename(fname, archive_name)
            st.info(f"Archived as {archive_name}")
        if "optimizer" in st.session_state:
            del st.session_state.optimizer
        if "_loaded_project" in st.session_state:
            del st.session_state["_loaded_project"]
        if "current_batch" in st.session_state:
            del st.session_state.current_batch
        st.rerun()

# --- TABS ---
tab_setup, tab_optimize = st.tabs(["1. Setup & Config", "2. Optimization Loop"])

# ==========================================
# TAB 1: SETUP
# ==========================================
with tab_setup:
    col_a, col_b = st.columns(2)

    with col_a:
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

        # Show current ingredients
        ingredients = [v for v in st.session_state.optimizer.variables
                       if v.get('category', 'ingredient') == 'ingredient']
        if ingredients:
            st.caption("Current ingredients:")
            ing_df = pd.DataFrame([
                {"Name": v['name'], "Min": v['bounds'][0], "Max": v['bounds'][1]}
                for v in ingredients
            ])
            st.dataframe(ing_df, hide_index=True, height=150)

        st.divider()

        # --- A2. Process Parameters ---
        st.subheader("A2. Process Parameters")
        st.caption("Add processing variables (e.g., baking temperature, mixing time) "
                   "that the optimizer will also explore.")

        with st.form("process_param_form"):
            pp_cols = st.columns(3)
            with pp_cols[0]:
                pp_name = st.text_input("Parameter Name", placeholder="e.g. Baking_Temp")
            with pp_cols[1]:
                pp_min = st.number_input("Min Value", value=0.0, key="pp_min")
            with pp_cols[2]:
                pp_max = st.number_input("Max Value", value=100.0, key="pp_max")
            if st.form_submit_button("Add Process Parameter"):
                if pp_name:
                    st.session_state.optimizer.add_process_parameter(pp_name, pp_min, pp_max)
                    st.success(f"Added process parameter: {pp_name}")

        # Show current process parameters
        proc_vars = [v for v in st.session_state.optimizer.variables
                     if v.get('category') == 'process']
        if proc_vars:
            st.caption("Current process parameters:")
            for i, pv in enumerate(proc_vars):
                pc1, pc2 = st.columns([3, 1])
                with pc1:
                    st.text(f"{pv['name']}: [{pv['bounds'][0]}, {pv['bounds'][1]}]")
                with pc2:
                    if st.button("Remove", key=f"rm_pp_{i}"):
                        st.session_state.optimizer.remove_process_parameter(pv['name'])
                        st.rerun()

        st.divider()

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
                    st.session_state.optimizer.add_objective(
                        obj_name.strip(),
                        obj_weight,
                        obj_goal,
                        target=obj_target if obj_goal == 'target' else None,
                        min_val=obj_min,
                        max_val=obj_max
                    )
                    st.success(f"Added {obj_name}")

        if st.session_state.optimizer.objectives:
            st.write("Active Objectives:")
            st.dataframe(pd.DataFrame(st.session_state.optimizer.objectives))
            for i, obj in enumerate(st.session_state.optimizer.objectives):
                if st.button(f"Remove {obj['name']}", key=f"rm_obj_{i}"):
                    st.session_state.optimizer.remove_objective(obj['name'])
                    st.rerun()

    with col_b:
        st.subheader("C. Low-Fidelity Model (Optional)")
        uploaded_pkl = st.file_uploader("Upload Model .pkl", type=["pkl"])
        if uploaded_pkl:
            try:
                model = pickle.load(uploaded_pkl)
                st.session_state.optimizer.load_screening_model(model)
                st.success("Model loaded!")
            except Exception as e:
                st.error(f"Error loading pickle: {e}")

        st.divider()

        # --- D. Property Constraints ---
        st.subheader("D. Property Constraints")
        st.caption("Constraints on computed properties (e.g., total cost, total protein) "
                   "based on ingredient properties from your CSV.")
        props = set()
        for p in st.session_state.optimizer.ingredient_properties.values():
            props.update(p.keys())

        if props:
            c_metric = st.selectbox("Constraint Metric", list(props))
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

        # --- E. Ingredient Quantity Constraints ---
        st.subheader("E. Ingredient Quantity Constraints")
        st.caption(
            "Set upper/lower limits on the **sum of ingredient quantities**. "
            "Useful for synergistic effects (e.g., Sugar + Honey <= 50g) "
            "or constraining total recipe mass."
        )

        ingredients = [v['name'] for v in st.session_state.optimizer.variables
                       if v.get('category', 'ingredient') == 'ingredient']

        if ingredients:
            # Sum constraint on selected ingredients
            with st.form("qty_constraint_form"):
                selected_ings = st.multiselect(
                    "Select Ingredients for Sum Constraint",
                    ingredients,
                    help="Select 2+ ingredients to constrain their combined quantity"
                )
                qc_cols = st.columns(2)
                with qc_cols[0]:
                    qc_min = st.number_input("Min Sum", value=0.0, key="qc_min",
                                             help="Leave at 0 for no lower bound")
                with qc_cols[1]:
                    qc_max = st.number_input("Max Sum", value=100.0, key="qc_max",
                                             help="Upper limit for the sum")
                qc_use_min = st.checkbox("Apply minimum bound", value=False, key="qc_use_min")
                qc_use_max = st.checkbox("Apply maximum bound", value=True, key="qc_use_max")

                if st.form_submit_button("Add Quantity Constraint"):
                    if len(selected_ings) >= 1:
                        st.session_state.optimizer.add_quantity_constraint(
                            selected_ings,
                            min_val=qc_min if qc_use_min else None,
                            max_val=qc_max if qc_use_max else None
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
                        min_val=tm_min, max_val=tm_max
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

# ==========================================
# TAB 2: OPTIMIZATION LOOP
# ==========================================
with tab_optimize:
    col1, col2 = st.columns([1, 1.5])

    # COLUMN 1: ASK
    with col1:
        st.subheader("Generate Experiments")
        batch_size = st.slider("Batch Size", 1, 10, 3)

        if st.button(f"Generate {batch_size} Recipes", type="primary"):
            if not st.session_state.optimizer.variables:
                st.error("Please load ingredients in Setup tab first!")
            elif not st.session_state.optimizer.objectives:
                st.error("Please define objectives in Setup tab first!")
            else:
                with st.spinner("Optimizing..."):
                    try:
                        recipes = st.session_state.optimizer.ask(n_suggestions=batch_size)
                        st.session_state.current_batch = recipes
                    except ValueError as e:
                        st.error(str(e))

        if "current_batch" in st.session_state:
            st.info("Suggested Batch:")

            # Separate ingredients and process parameters for display
            proc_names = {v['name'] for v in st.session_state.optimizer.variables
                          if v.get('category') == 'process'}
            ing_names = [v['name'] for v in st.session_state.optimizer.variables
                         if v.get('category', 'ingredient') == 'ingredient']

            df_res = pd.DataFrame(st.session_state.current_batch)

            # Show ingredients
            ing_cols = [c for c in df_res.columns if c not in proc_names]
            if ing_cols:
                st.caption("Ingredients:")
                st.dataframe(df_res[ing_cols].style.format("{:.2f}"), hide_index=True)

            # Show process parameters separately
            proc_cols = [c for c in df_res.columns if c in proc_names]
            if proc_cols:
                st.caption("Process Parameters:")
                st.dataframe(df_res[proc_cols].style.format("{:.2f}"), hide_index=True)

    # COLUMN 2: TELL
    with col2:
        st.subheader("Input Lab Results")

        if "current_batch" in st.session_state and st.session_state.current_batch:
            with st.form("results_form"):
                batch_inputs = {}
                for i, recipe in enumerate(st.session_state.current_batch):
                    st.markdown(f"**Recipe #{i+1}**")
                    cols = st.columns(len(st.session_state.optimizer.objectives))
                    rec_scores = {}
                    for j, obj in enumerate(st.session_state.optimizer.objectives):
                        with cols[j]:
                            rec_scores[obj['name']] = st.number_input(
                                f"{obj['name']} ({obj['min_val']}-{obj['max_val']})",
                                key=f"r{i}o{j}"
                            )
                    batch_inputs[i] = rec_scores
                    st.divider()

                if st.form_submit_button("Save Results"):
                    for i, recipe in enumerate(st.session_state.current_batch):
                        st.session_state.optimizer.tell(recipe, batch_inputs[i])
                    del st.session_state.current_batch
                    st.success("Saved!")
                    st.rerun()

    st.divider()

    # --- Bulk Import Historical Experiments ---
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
            elif st.button("Import All Rows", type="primary"):
                imported = 0
                for _, row in import_df.iterrows():
                    recipe = {name: float(row[name]) for name in var_names}
                    results = {name: float(row[name]) for name in obj_names}
                    st.session_state.optimizer.tell(recipe, results)
                    imported += 1
                st.success(f"Imported {imported} experiments!")
                st.rerun()

    st.divider()

    # --- Utility Score Explanation ---
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

    # --- History with Edit/Delete ---
    if st.session_state.optimizer.X_history:
        st.subheader("Experiment History")

        hist = []
        for i, x in enumerate(st.session_state.optimizer.X_history):
            r = st.session_state.optimizer._decode(x)
            r['_index'] = i
            r['Total_Utility_Score'] = st.session_state.optimizer.Y_history[i]

            # Include raw results if available
            res_hist = getattr(st.session_state.optimizer, 'results_history', [])
            if i < len(res_hist):
                for key, val in res_hist[i].items():
                    r[f"[Result] {key}"] = val

            hist.append(r)

        hist_df = pd.DataFrame(hist).sort_values('Total_Utility_Score', ascending=False)
        hist_df.insert(0, 'Exp #', hist_df['_index'])
        st.dataframe(hist_df.drop(columns=['_index']), hide_index=True)

        # --- Edit Results ---
        st.caption("Edit or delete a past result:")
        results_history = getattr(st.session_state.optimizer, 'results_history', [])

        if results_history:
            edit_idx = st.number_input(
                "Experiment Index to Edit (0-based)",
                min_value=0,
                max_value=len(st.session_state.optimizer.X_history) - 1,
                value=0,
                step=1,
                key="edit_idx"
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
                                f"{obj['name']}",
                                value=current_val,
                                key=f"edit_{edit_idx}_{j}"
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
            step=1,
            key="rewind_idx",
        )
        n_discard = len(st.session_state.optimizer.X_history) - 1 - rewind_idx
        if n_discard > 0:
            st.warning(
                f"This will discard {n_discard} experiment(s) "
                f"(#{rewind_idx + 1} through #{len(st.session_state.optimizer.X_history) - 1})."
            )
        if st.button("Rewind", disabled=(n_discard == 0), key="rewind_btn"):
            # Archive current state before rewinding
            pname = st.session_state.optimizer.project_name
            fname = f"{pname}.pkl"
            if os.path.exists(fname):
                archive_name = f"{pname}_pre_rewind.pkl"
                counter = 1
                while os.path.exists(archive_name):
                    archive_name = f"{pname}_pre_rewind_{counter}.pkl"
                    counter += 1
                import shutil
                shutil.copy2(fname, archive_name)
                st.info(f"Archived current state as {archive_name}")
            st.session_state.optimizer.rewind_to(rewind_idx)
            if "current_batch" in st.session_state:
                del st.session_state.current_batch
            st.success(f"Rewound to experiment #{rewind_idx}!")
            st.rerun()
