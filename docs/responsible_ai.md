# Responsible AI and Model Risk

## Scope

This project uses public platform data, predictive models and generative AI. The research therefore separates useful decision support from claims that the evidence cannot justify.

## Core Safeguards

### 1. Association is not causality
Regression and machine-learning results are presented as modeled associations or predictive relationships. The project does not claim that changing a thumbnail feature will necessarily cause higher future performance.

### 2. No guaranteed uplift claims
The AIGC layer must not state or imply guaranteed increases in views, engagement or other performance metrics.

### 3. Temporal instability is disclosed
A later-period validation result showed materially weaker generalization. This is treated as model-risk evidence and motivates uncertainty and abstention rather than being omitted from the public summary.

### 4. Semantic preservation
Generated research candidates must preserve the underlying content semantics. They must not invent or alter:

- match or competition identity;
- player or club affiliation;
- scorelines;
- transfers;
- injuries;
- trophies;
- quotes;
- breaking-news claims;
- other unsupported factual events.

### 5. Human-in-the-loop review
The final PoC is decision support rather than autonomous optimization. Candidate outputs can be reviewed, rejected or abstained from.

### 6. Uncertainty is part of the output
Prediction intervals and candidate-level uncertainty are exposed where applicable instead of presenting a single deterministic score as truth.

### 7. Data minimization
The public repository does not need to redistribute every raw asset. Credentials, private data, confidential materials and unnecessary large raw files are excluded.

## Intended Use

The project is intended for:

- academic and portfolio demonstration;
- exploratory content analytics;
- model-validation research;
- evidence-backed design experimentation.

It is not intended to be used as:

- an automated causal decision engine;
- a guaranteed CTR or views optimizer;
- a source of invented football news or factual claims;
- a system for accessing non-public YouTube user data.