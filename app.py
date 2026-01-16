import streamlit as st
import pandas as pd
import pickle
from food_bo import FoodOptimizer

st.set_page_config(page_title="BO for Food", layout="wide")
st.title("Bayesian Optimization for Food")

# --- SIDEBAR: Project Name ---
with st.sidebar:
    project_name = st.text_input("Project Name", "Cookie_Project_v2")
    
    if "optimizer" not in st.session_state:
        st.session_state.optimizer = FoodOptimizer(project_name)
        st.success(f"Initialized {project_name}")
    
    # Reset Button
    if st.button("⚠️ Hard Reset Project"):
        import os
        if os.path.exists(f"{project_name}.pkl"):
            os.remove(f"{project_name}.pkl")
        del st.session_state.optimizer
        st.rerun()

# --- TABS ---
tab_setup, tab_optimize = st.tabs(["🛠️ 1. Setup & Config", "🚀 2. Optimization Loop"])

# ==========================================
# TAB 1: SETUP (Ingredients, Model, Constraints)
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
                st.session_state.optimizer.load_ingredients_from_csv(df)
                st.success(f"Loaded {len(df)} ingredients!")

        st.divider()
        st.subheader("B. Objectives")
        obj_name = st.text_input("Objective Name", "Texture")
        obj_weight = st.slider("Weight", 0.0, 1.0, 1.0)
        obj_goal = st.selectbox("Goal", ["max", "min"])
        if st.button("Add Objective"):
            st.session_state.optimizer.add_objective(obj_name, obj_weight, obj_goal)
            st.success(f"Added {obj_name}")

    with col_b:
        st.subheader("C. Low-Fidelity Model (Optional)")
        uploaded_pkl = st.file_uploader("Upload Model .pkl", type=["pkl"])
        if uploaded_pkl:
            try:
                model = pickle.load(uploaded_pkl)
                st.session_state.optimizer.load_screening_model(model)
                st.success("Model loaded! It will screen initial suggestions.")
            except Exception as e:
                st.error(f"Error loading pickle: {e}")

        st.divider()
        st.subheader("D. Constraints")
        # Dynamic constraints based on CSV properties
        props = set()
        for p in st.session_state.optimizer.ingredient_properties.values():
            props.update(p.keys())
        
        if props:
            c_metric = st.selectbox("Constraint Metric", list(props))
            c_min = st.number_input("Min Value", value=0.0)
            c_max = st.number_input("Max Value", value=100.0)
            if st.button("Add Constraint"):
                st.session_state.optimizer.add_constraint(c_metric, min_val=c_min)
                st.success("Constraint Added")
        else:
            st.write("No properties (Cost/Protein) found in CSV.")

# ==========================================
# TAB 2: OPTIMIZATION LOOP
# ==========================================
with tab_optimize:
    col1, col2 = st.columns([1, 1.5]) 

    # COLUMN 1: ASK 
    with col1:
        st.subheader("Generate Experiments")
        batch_size = st.slider("Batch Size", 1, 10, 3)
        
        if st.button(f"✨ Generate {batch_size} Recipes", type="primary"):
            if not st.session_state.optimizer.variables:
                st.error("Please load ingredients in Setup tab first!")
            elif not st.session_state.optimizer.objectives:
                st.error("Please define objectives in Setup tab first!")
            else:
                with st.spinner("Optimizing..."):
                    recipes = st.session_state.optimizer.ask(n_suggestions=batch_size)
                    st.session_state.current_batch = recipes
        
        if "current_batch" in st.session_state:
            st.info(f"🧪 Suggested Batch:")
            df_res = pd.DataFrame(st.session_state.current_batch)
            st.dataframe(df_res.style.format("{:.2f}"), hide_index=True)

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
                                f"{obj['name']}", key=f"r{i}o{j}"
                            )
                    batch_inputs[i] = rec_scores
                    st.divider()
                
                if st.form_submit_button("💾 Save Results"):
                    for i, recipe in enumerate(st.session_state.current_batch):
                        st.session_state.optimizer.tell(recipe, batch_inputs[i])
                    st.session_state.optimizer = st.session_state.optimizer 
                    del st.session_state.current_batch 
                    st.success("Saved!")
                    st.rerun()

    st.divider()
    if st.session_state.optimizer.X_history:
        st.subheader("History")
        # Decode history
        hist = []
        for i, x in enumerate(st.session_state.optimizer.X_history):
            r = st.session_state.optimizer._decode(x)
            r['Score'] = st.session_state.optimizer.Y_history[i]
            hist.append(r)
        st.dataframe(pd.DataFrame(hist).sort_values('Score', ascending=False))