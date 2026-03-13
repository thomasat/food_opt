---
marp: true
theme: default
paginate: true
style: |
  section {
    font-family: 'Segoe UI', Arial, sans-serif;
  }
  h1 {
    color: #2c3e50;
  }
  h2 {
    color: #34495e;
  }
  .columns {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 1rem;
  }
---

# Formulation Assistant

### AI-Powered Food Product Optimization

Bayesian Optimization for Ingredient Formulation

---

# The Challenge

- Developing new food formulations requires **many trial-and-error iterations**
- Each lab trial is **expensive** and **time-consuming**
- Sensory panel evaluations add further cost per iteration
- Traditional approaches (DOE, one-variable-at-a-time) scale poorly with many ingredients

**Goal:** Minimize the number of lab experiments needed to find the optimal recipe.

---

# Our Solution: Bayesian Optimization

A **smart experimental design** system that:

1. **Learns** from every experiment you run
2. **Models** the relationship between ingredients and sensory outcomes
3. **Suggests** the most informative next experiment
4. **Converges** on the optimal formulation in fewer iterations

> Typically reaches near-optimal results in **10-15 iterations** vs. 50-100+ with traditional methods.

---

# How It Works

![bg right:40% fit](https://upload.wikimedia.org/wikipedia/commons/0/02/Bayesian_optimization.gif)

### The Ask-Tell Loop

1. **Setup** - Define ingredients, ranges, and objectives
2. **Ask** - System suggests a batch of recipes to try
3. **Lab Trial** - You prepare and evaluate the recipes
4. **Tell** - Input sensory panel ratings back into the system
5. **Repeat** - System learns and suggests improved recipes

---

# Step 1: Setup - Ingredients

Upload your ingredient list via CSV with ranges:

| Name | Min | Max | Cost | Protein |
|------|-----|-----|------|---------|
| Flour | 30 | 60 | 0.5 | 10.3 |
| Sugar | 5 | 25 | 0.8 | 0.0 |
| Butter | 10 | 30 | 2.1 | 0.9 |
| Eggs | 5 | 15 | 1.5 | 12.6 |

- **Min/Max**: allowed quantity range per ingredient
- **Additional columns** (Cost, Protein, etc.): used for property constraints

---

# Step 1: Setup - Process Parameters

Optimize **process variables** alongside ingredients:

- Baking Temperature (150-220 C)
- Mixing Time (5-20 min)
- Resting Time (0-60 min)

The optimizer explores ingredient *and* process combinations jointly.

---

# Step 1: Setup - Objectives

Define what "good" means for your product:

| Metric | Goal | Weight | Range |
|--------|------|--------|-------|
| Chewiness | Maximize | 0.6 | 0-10 |
| Sweetness | Target = 5 | 0.3 | 0-10 |
| Graininess | Minimize | 0.1 | 0-10 |

- **Maximize / Minimize / Hit a Target** for each attribute
- **Weights** reflect relative importance
- **Ranges** normalize scores to a common 0-1 scale

---

# Step 1: Setup - Constraints

### Property Constraints
Based on ingredient properties from your CSV:
- Total Cost <= 5.00 per unit
- Total Protein >= 8g per serving

### Ingredient Quantity Constraints
Direct limits on ingredient amounts:
- Sugar + Honey <= 50g (synergistic sweeteners)
- Total recipe mass between 90g and 110g

All constraints are enforced **during optimization** - every suggestion is feasible.

---

# Step 2: The Optimization Engine

### Cold Start (First 5 Experiments)
- Uses **Sobol quasi-random sequences** for space-filling exploration
- Ensures broad coverage of the ingredient space
- Respects all constraints from the start

### Bayesian Optimization (Experiment 6+)
- Fits a **Gaussian Process (GP)** model to all data so far
- Uses **Noisy Expected Improvement** acquisition to balance:
  - **Exploration**: trying under-explored regions
  - **Exploitation**: refining promising areas

---

# Utility Score Calculation

Each experiment gets a single **Utility Score** combining all objectives:

```
For each objective:
  1. Normalize:  (value - range_min) / (range_max - range_min)
  2. Apply goal:
     - Maximize:  utility = normalized
     - Minimize:  utility = 1 - normalized
     - Target:    utility = max(0, 1 - |normalized - target|)
  3. Weight:     weighted = weight * utility

Total Utility = sum of all weighted utilities
```

**Example:** Chewiness=8 (w=0.6) + Sweetness=6 (target=5, w=0.4) = **0.876**

---

# Managing Experiments

### Edit Past Results
- Correct sensory ratings if a panel needs to be re-evaluated
- System warns that later experiments may be invalidated

### Rewind
- Roll back to any past experiment
- Discard subsequent iterations that are no longer valid
- Original state is **archived automatically** (nothing lost)

### Delete
- Remove individual outlier experiments

---

# Workflow: Correcting a Past Evaluation

**Scenario:** At iteration 6, you realize iteration 4's sensory panel was flawed.

1. **Edit** experiment #4 with corrected sensory ratings
2. **Rewind** to experiment #4 (discards #5 and #6)
3. System **re-optimizes** from the corrected data
4. New suggestion for experiment #5 reflects the updated ratings

The archive preserves all prior work for reference.

---

# Two Operating Modes

### Standard Mode (Default)
- Faster convergence on **smooth** formulation landscapes
- Best for typical ingredient synergies
- Recommended starting point

### Robust Mode
- Uses **Input Warping** to handle non-linear effects
- Better for formulations with sharp thresholds
  (e.g., phase separation, texture cliffs)
- Slower initial learning, more resilient long-term

---

# Project Management

- **Multiple projects** - Run separate optimization campaigns in parallel
- **Save/Load** - All state persisted automatically (pickle files)
- **Archive on reset** - Hard reset archives rather than deletes
- **Version tracking** - Automatic session upgrades when software updates

---

# Technology Stack

| Component | Technology |
|-----------|-----------|
| Optimization Engine | BoTorch + GPyTorch (Meta AI) |
| Surrogate Model | Gaussian Process (SingleTaskGP) |
| Acquisition Function | qLogNoisyExpectedImprovement |
| User Interface | Streamlit (interactive web app) |
| Language | Python + PyTorch |

---

# Key Benefits

- **Fewer experiments** to reach optimal formulation (cost + time savings)
- **Multi-objective** optimization with flexible goals
- **Constraint-aware** suggestions (cost, nutrition, mass limits)
- **Batch suggestions** - run multiple experiments per iteration
- **Full audit trail** with edit, delete, and rewind capabilities
- **No ML expertise required** - intuitive web interface

---

# Next Steps

1. **Define** your ingredient list and ranges
2. **Set** sensory objectives and constraints
3. **Run** the first batch of 3-5 experiments
4. **Evaluate** with your sensory panel
5. **Iterate** - the system improves with every data point

---

# Thank You

### Questions?

*Formulation Assistant - Smarter experiments, faster results.*
