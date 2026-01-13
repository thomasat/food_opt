import streamlit as st
import pandas as pd
from food_bo import FoodOptimizer


st.set_page_config(page_title="Bayesian Optimization", layout="wide")

st.title("Bayesian Optimization for Food")

with st.sidebar:
    st.header("Project Config")
    project_name = st.text_input("Project Name", "Nokara_Hard_Cheese_v1")
    
    if "optimizer" not in st.session_state:
        opt = FoodOptimizer(project_name)
        
        if not opt.variables:
            opt.add_ingredient("Nokara_Percent", 40, 70)
            opt.add_ingredient("Fermentation Time in Incubator (hrs)", 24, 72) 
            
        if not opt.objectives:
            opt.add_objective("Liking", weight=0.6, goal="max")
            opt.add_objective("Similarity to Animal-Based Cheese", weight=0.4, goal="max")
            
        st.session_state.optimizer = opt
        st.success(f"Loaded {project_name}")

    st.subheader("Parameters Defined")
    for var in st.session_state.optimizer.variables:
        if var['type'] == 'continuous':
            st.code(f"{var['name']}: {var['bounds']}")
        else:
            st.code(f"{var['name']}: {var['options']}")


col1, col2 = st.columns(2)

with col1:
    st.subheader("1. Generate New Recipe")
    
    if st.button("✨ Ask AI for Next Recipe", type="primary"):
        recipe = st.session_state.optimizer.ask()
        st.session_state.current_recipe = recipe
    
    if "current_recipe" in st.session_state:
        st.info("🧪 Test This Formulation:")
        
        recipe_df = pd.DataFrame([st.session_state.current_recipe])
        st.dataframe(recipe_df, hide_index=True)

with col2:
    st.subheader("2. Input Lab Results")
    
    if "current_recipe" in st.session_state:
        with st.form("results_form"):
            st.write("Rate the sensory attributes (0-10):")
            
            inputs = {}
            for obj in st.session_state.optimizer.objectives:
                inputs[obj['name']] = st.slider(
                    f"{obj['name']} ({obj['goal']})", 
                    0.0, 10.0, 5.0, 0.5
                )
            
            submitted = st.form_submit_button("💾 Save Results")
            
            if submitted:
                st.session_state.optimizer.tell(
                    st.session_state.current_recipe, 
                    inputs
                )
                st.success("Data saved! Model updated.")
                # Clear the current recipe to force a new 'Ask'
                del st.session_state.current_recipe
                st.rerun()
    else:
        st.write("Generate a recipe first to enter data.")

st.divider()
st.subheader("📊 Experiment History")

# Combine X (Ingredients) and Y (Scores) for display
if st.session_state.optimizer.X_history:
    # We need to decode the history from the optimizer to make it readable
    history_data = []
    for i, x_vec in enumerate(st.session_state.optimizer.X_history):
        # Decode inputs
        row = st.session_state.optimizer._decode(x_vec)
        # Add score
        row['Weighted_Score'] = st.session_state.optimizer.Y_history[i]
        history_data.append(row)
        
    hist_df = pd.DataFrame(history_data)
    st.dataframe(hist_df, use_container_width=True)
else:
    st.text("No data yet. Start baking!")