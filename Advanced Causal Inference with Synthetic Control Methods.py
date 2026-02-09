"""
Advanced Causal Inference with Synthetic Control Methods
-------------------------------------------------------

Single-file, production-quality implementation of:
1. Panel data simulation
2. Synthetic Control Method (from scratch)
3. Permutation inference (placebo tests)
4. Quantitative analysis and reporting

Author: AI Expert
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize

# ======================================================
# 1. DATA GENERATION
# ======================================================

def generate_panel_data(
    n_controls=50,
    n_periods=100,
    intervention_time=60,
    seed=42
):
    np.random.seed(seed)

    units = n_controls + 1  # 1 treated unit
    time = np.arange(n_periods)

    # Latent factors
    factor_1 = np.sin(time / 8)
    factor_2 = np.log(time + 1)

    loadings = np.random.normal(1, 0.2, size=(units, 2))
    noise = np.random.normal(0, 0.5, size=(units, n_periods))

    Y = (
        loadings[:, 0][:, None] * factor_1 +
        loadings[:, 1][:, None] * factor_2 +
        noise
    )

    # Treatment effect (only for treated unit after intervention)
    treatment_effect = np.zeros(n_periods)
    treatment_effect[intervention_time:] = 3.0

    Y[0, :] += treatment_effect  # unit 0 is treated

    return Y, intervention_time

# ======================================================
# 2. SYNTHETIC CONTROL METHOD
# ======================================================

def fit_synthetic_control(Y, treated_idx, intervention_time):
    Y_pre = Y[:, :intervention_time]

    Y_treated_pre = Y_pre[treated_idx]
    Y_controls_pre = np.delete(Y_pre, treated_idx, axis=0)

    n_controls = Y_controls_pre.shape[0]

    def objective(w):
        synthetic = np.dot(w, Y_controls_pre)
        return np.sum((Y_treated_pre - synthetic) ** 2)

    constraints = (
        {'type': 'eq', 'fun': lambda w: np.sum(w) - 1},
    )
    bounds = [(0, 1) for _ in range(n_controls)]

    w0 = np.ones(n_controls) / n_controls

    result = minimize(
        objective,
        w0,
        bounds=bounds,
        constraints=constraints
    )

    return result.x

def construct_synthetic(Y, weights, treated_idx):
    controls = np.delete(Y, treated_idx, axis=0)
    return np.dot(weights, controls)

# ======================================================
# 3. TREATMENT EFFECT ESTIMATION
# ======================================================

def estimate_treatment_effect(Y, synthetic, intervention_time):
    treated = Y[0]
    effect = treated[intervention_time:] - synthetic[intervention_time:]
    return effect, effect.mean()

# ======================================================
# 4. PERMUTATION (PLACEBO) INFERENCE
# ======================================================

def placebo_tests(Y, intervention_time):
    placebo_effects = []

    for unit in range(1, Y.shape[0]):
        weights = fit_synthetic_control(Y, unit, intervention_time)
        synthetic = construct_synthetic(Y, weights, unit)
        effect = Y[unit, intervention_time:] - synthetic[intervention_time:]
        placebo_effects.append(effect.mean())

    return np.array(placebo_effects)

def compute_p_value(true_effect, placebo_effects):
    return np.mean(np.abs(placebo_effects) >= np.abs(true_effect))

# ======================================================
# 5. MAIN EXECUTION
# ======================================================

def main():
    Y, intervention_time = generate_panel_data()

    # Fit SCM
    weights = fit_synthetic_control(Y, treated_idx=0, intervention_time=intervention_time)
    synthetic = construct_synthetic(Y, weights, treated_idx=0)

    # Estimate effect
    effect_path, avg_effect = estimate_treatment_effect(
        Y, synthetic, intervention_time
    )

    # Placebo inference
    placebo_effects = placebo_tests(Y, intervention_time)
    p_value = compute_p_value(avg_effect, placebo_effects)

    # ==================================================
    # OUTPUT RESULTS
    # ==================================================

    print("\n=== SYNTHETIC CONTROL RESULTS ===\n")

    print("Optimized Weights (Top 10 Controls):")
    weight_df = pd.DataFrame({
        "Control Unit": np.arange(1, len(weights) + 1),
        "Weight": weights
    }).sort_values("Weight", ascending=False)

    print(weight_df.head(10).to_string(index=False))

    print("\nAverage Post-Intervention Treatment Effect:")
    print(f"{avg_effect:.3f}")

    print("\nPermutation Test Results:")
    print(f"Mean placebo effect: {placebo_effects.mean():.3f}")
    print(f"P-value: {p_value:.4f}")

    print("\nInterpretation:")
    if p_value < 0.05:
        print("The treatment effect is statistically significant.")
    else:
        print("The treatment effect is NOT statistically significant.")

if __name__ == "__main__":
    main()
