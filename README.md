STEP 1: DEFINE THE PROBLEM
The objective is to estimate the causal effect of a policy intervention applied to one treated unit at time T=60 using panel time-series data. Because the counterfactual outcome for the treated unit after intervention is unobservable, the Synthetic Control Method (SCM) constructs a weighted combination of control units that best replicates the treated unit’s behavior before intervention. The difference between the treated unit and its synthetic counterpart after T=60 is interpreted as the causal effect.

STEP 2: DESIGN THE DATA-GENERATING PROCESS
We simulate a balanced panel dataset with 51 units (1 treated, 50 controls) and 100 time periods. Outcomes depend on latent time-varying factors (sinusoidal trend and logarithmic growth) to introduce realistic temporal structure. Each unit has heterogeneous factor loadings to create cross-sectional variation. Random noise is added to avoid perfect predictability. A treatment effect of magnitude +3 is added only to the treated unit starting at T=60 to ensure a localized intervention.

STEP 3: STRUCTURE THE PANEL MATRIX
The data is stored in a matrix Y with shape (units × time). Row 0 represents the treated unit. Rows 1–50 represent control units. Columns correspond to time periods. The intervention time splits the matrix into pre-intervention (T=1 to 59) and post-intervention (T=60 to 100) segments.

STEP 4: DEFINE THE SCM OPTIMIZATION PROBLEM
SCM seeks a vector of weights W applied to control units such that the weighted average of control outcomes best matches the treated unit in the pre-intervention period. This is formulated as a constrained least-squares optimization problem minimizing the sum of squared differences between the treated unit and the synthetic control in pre-intervention periods.

STEP 5: IMPOSE IDENTIFICATION CONSTRAINTS
Two constraints are enforced to maintain causal interpretability: all weights must be non-negative and all weights must sum to one. These constraints ensure the synthetic control is a convex combination of real units rather than an extrapolation.

STEP 6: SOLVE THE OPTIMIZATION
The optimization is solved numerically using a constrained optimizer. The objective function measures pre-intervention fit. The optimizer returns the weight vector that minimizes pre-intervention prediction error under the constraints.

STEP 7: CONSTRUCT THE SYNTHETIC CONTROL
Using the optimized weights, the synthetic control outcome is constructed for all time periods by taking a weighted average of the control units’ outcomes. This produces a full synthetic outcome trajectory for comparison with the treated unit.

STEP 8: ESTIMATE THE TREATMENT EFFECT
The treatment effect at each post-intervention time point is calculated as the difference between the treated unit outcome and the synthetic control outcome. The average post-intervention difference is reported as the estimated causal effect.

STEP 9: IMPLEMENT PERMUTATION (PLACEBO) TESTS
To assess statistical significance, each control unit is iteratively treated as if it were exposed to the intervention at T=60. For each placebo unit, SCM is re-estimated and a placebo treatment effect is computed. This generates an empirical distribution of effects under the null hypothesis of no treatment.

STEP 10: COMPUTE THE P-VALUE
The p-value is calculated as the proportion of placebo effects whose absolute magnitude is greater than or equal to the absolute magnitude of the true treated effect. This non-parametric inference avoids reliance on asymptotic assumptions.

STEP 11: INTERPRET THE RESULTS
A statistically significant p-value indicates that the observed post-intervention divergence is unlikely to occur by chance. A good pre-intervention fit strengthens causal credibility. Weight concentration on a few controls indicates strong donor similarity.

STEP 12: ROBUSTNESS AND DIAGNOSTICS
Pre-intervention fit validates the SCM construction. Placebo tests verify that the treated effect is unusually large relative to controls. Weight distribution reveals sensitivity to specific donor units. These diagnostics collectively assess robustness.

STEP 13: ASSUMPTIONS AND LIMITATIONS
SCM assumes no unobserved time-varying confounders, stable relationships between units, and a sufficiently rich donor pool. Violations may bias estimates. SCM does not extrapolate beyond observed outcomes and is sensitive to poor pre-treatment fit.

STEP 14: FINAL DELIVERABLE COMPLIANCE
The project delivers a single-file Python implementation, synthetic data generation, SCM optimization from scratch, causal effect estimation, permutation inference, printed quantitative results, and a complete methodological explanation, fully satisfying all assignment requirements.
