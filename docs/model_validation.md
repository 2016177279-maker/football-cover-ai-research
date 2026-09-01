# Model Validation

## Validation Philosophy

The project treats predictive performance and model risk as separate from descriptive association. Strong in-sample or random-split performance is not considered sufficient evidence of real-world generalization.

## Grouped Holdout

Videos are grouped by channel in the holdout design to reduce leakage of creator-specific structure between train and test sets.

Representative grouped-holdout performance:

| Target | R² |
|---|---:|
| Views | 0.602 |
| Velocity | 0.650 |
| Engagement | 0.150 |

These results suggest that channel-aware models can capture meaningful cross-sectional structure for views and velocity, while engagement is more difficult to explain.

## Temporal Validation

A later-period test for Velocity produced:

- **R² = -0.233**

This is an important model-risk result. It indicates that relationships learned from historical data are not stable enough to support claims of reliable future-performance prediction.

The project therefore does not describe the model as a guaranteed forecasting system.

## Robustness and Uncertainty

Representative checks include:

- clustered standard errors;
- alternative controlled specifications;
- grouped train/test splits;
- temporal out-of-time evaluation;
- Bootstrap uncertainty analysis;
- model-interpretation consistency checks;
- candidate-level prediction intervals in the AIGC PoC.

## Decision Implication

Because temporal generalization is unstable, model output is used as one input to decision support rather than as an autonomous decision rule.

The application layer therefore includes:

- uncertainty intervals;
- abstention logic;
- feature re-extraction for generated candidates;
- human review before accepting a candidate.

## What the Validation Does Not Prove

The validation does not prove that changing a thumbnail feature will causally increase future views or engagement. Demonstrating causal uplift would require a stronger experimental or quasi-experimental design, such as randomized A/B testing.