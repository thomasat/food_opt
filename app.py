import streamlit as st
import pandas as pd
from food_bo import FoodOptimizer

# Page Config
st.set_page_config(page_title="Food Optimization", layout="wide")
st.title("🍪 Bayesian Optimization for Food")

# --- SIDEBAR: Project Setup ---
with st.sidebar:
    st.header("Project Config")
    project_name = st.text_input("Project Name", "High_Protein_Cookie_v1")
    
    if "optimizer" not in st.session_state:
        opt = FoodOptimizer(project_name)
        
        # Defaults for Demo
        if not opt.variables:
            opt.add_ingredient("Pea_Protein_Percent", 0, 20)
            opt.add_ingredient("Coconut_Oil_Percent", 10, 25) 
            opt.add_ingredient("Baking_Time_Mins", 8, 14)
            
        if not opt.objectives:
            opt.add_objective("Moistness", weight=0.6, goal="max")
            opt.add_objective("Beany_Flavor", weight=0.4, goal="min")
            
        st.session_state.optimizer = opt
        st.success(f"Loaded {project_name}")

    st.subheader("Ingredients Defined")
    for var in st.session_state.optimizer.variables:
        if var['type'] == 'continuous':
            st.code(f"{var['name']}: {var['bounds']}")
        else:
            st.code(f"{var['name']}: {var['options']}")

# --- MAIN PANEL ---

col1, col2 = st.columns([1, 1.5]) # Make Results column slightly wider

# COLUMN 1: ASK (Generate Batch)
with col1:
    st.subheader("1. Generate Experiments")
    
    # Batch Size Slider
    batch_size = st.slider("Batch Size (Recipes to Test)", 1, 10, 3)
    
    if st.button(f"✨ Generate {batch_size} New Recipes", type="primary"):
        with st.spinner("AI is thinking..."):
            # Returns a LIST of recipes now
            recipes = st.session_state.optimizer.ask(n_suggestions=batch_size)
            st.session_state.current_batch = recipes
    
    # Display Batch
    if "current_batch" in st.session_state:
        st.info(f"🧪 Suggested Batch ({len(st.session_state.current_batch)} recipes):")
        
        # Show as simple dataframe
        df = pd.DataFrame(st.session_state.current_batch)
        st.dataframe(df, hide_index=True)

# COLUMN 2: TELL (Input Results for Batch)
with col2:
    st.subheader("2. Input Lab Results")
    
    if "current_batch" in st.session_state and st.session_state.current_batch:
        
        with st.form("batch_results_form"):
            st.write("Enter sensory data for the recipes on the left:")
            
            # Dictionary to hold user inputs for the whole batch
            # Key: index, Value: dict of objective scores
            batch_inputs = {} 
            
            # Loop through the batch and create an entry section for each
            for i, recipe in enumerate(st.session_state.current_batch):
                st.markdown(f"**Recipe #{i+1}**")
                # Show recipe details in small text
                st.caption(str(recipe))
                
                cols = st.columns(len(st.session_state.optimizer.objectives))
                recipe_scores = {}
                
                # Create a slider for each objective
                for j, obj in enumerate(st.session_state.optimizer.objectives):
                    with cols[j]:
                        # Unique key is crucial for Streamlit widgets in loops!
                        val = st.number_input(
                            f"{obj['name']} (0-10)", 
                            min_value=0.0, max_value=10.0, step=0.5,
                            key=f"rec_{i}_obj_{j}"
                        )
                        recipe_scores[obj['name']] = val
                
                batch_inputs[i] = recipe_scores
                st.divider()
            
            submitted = st.form_submit_button("💾 Save All Results")
            
            if submitted:
                # Save each recipe in the batch loop
                for i, recipe in enumerate(st.session_state.current_batch):
                    scores = batch_inputs[i]
                    st.session_state.optimizer.tell(recipe, scores)
                
                st.success(f"Saved {len(st.session_state.current_batch)} results! Model updated.")
                del st.session_state.current_batch # Clear batch after save
                st.rerun()
                
    else:
        st.write("Generate a batch to enter results.")

# --- BOTTOM: History ---
st.divider()
st.subheader("📊 Experiment History")

if st.session_state.optimizer.X_history:
    history_data = []
    for i, x_vec in enumerate(st.session_state.optimizer.X_history):
        row = st.session_state.optimizer._decode(x_vec)
        row['Weighted_Score'] = st.session_state.optimizer.Y_history[i]
        history_data.append(row)
        
    hist_df = pd.DataFrame(history_data)
    # Sort by score to show winners top
    hist_df = hist_df.sort_values(by="Weighted_Score", ascending=False)
    st.dataframe(hist_df, use_container_width=True)
else:
    st.text("No data yet. Start baking!")