# Methodology

## Research Design

The project studies associations between thumbnail visual characteristics and public content-performance indicators in football-related YouTube videos.

The analysis is observational. It is designed for association, prediction and decision support rather than causal attribution.

## Sampling and Data Quality

The final frozen dataset contains **5,200 unique videos**. Construction of the research sample follows a reproducible sequence:

1. collect public football-related candidates;
2. filter for association-football relevance;
3. resolve uncertain cases through human review;
4. remove duplicates;
5. apply structured stratification across time and content categories;
6. freeze the final sample before downstream modeling.

The project uses explicit schema/version control and separates raw collection, eligibility decisions, final sampling and downstream feature/model outputs.

## Feature Engineering

Thumbnail images are converted into structured analytical variables using several feature families.

### Basic visual features
- dimensions and aspect ratio;
- brightness / contrast;
- color and saturation summaries;
- edge / visual-complexity measures;
- text-area measures;
- face / person indicators;
- composition-related variables.

### Advanced visual features
- OCR-derived text burden;
- center saliency;
- person / face / object geometry;
- visual entropy;
- aesthetic-coherence proxy;
- semantic representations using CLIP-style embeddings;
- selected scene and content-semantic indicators.

### Controls
Public video and channel metadata are used to control for non-visual differences such as channel scale, publication timing and content category where appropriate.

## Statistical Analysis

The analytical stack combines classical and predictive methods.

Representative methods include:

- multivariate regression;
- channel-clustered standard errors;
- fixed-effects / controlled specifications where applicable;
- nonlinear machine-learning models;
- grouped holdout validation;
- temporal validation;
- Bootstrap uncertainty analysis;
- SHAP / PDP / ICE / ALE-style interpretation and heterogeneity analysis.

## Why Grouped Validation Matters

Random row-level train/test splits can leak channel-specific structure when many videos come from the same creator. The project therefore uses grouped validation so that videos from the same channel are not mechanically split across train and test partitions.

This is intended to provide a more realistic estimate of generalization across creators.

## Why Temporal Validation Matters

A model that fits historical cross-sectional structure may still fail when applied to later periods. Temporal validation is therefore treated as a separate model-risk test.

The project reports temporal deterioration rather than hiding it. This is one reason the final application uses uncertainty, abstention and human review rather than presenting model output as guaranteed future performance.

## Interpretation Principle

The project separates three questions:

1. **Association:** which visual characteristics are statistically related to performance indicators?
2. **Prediction:** how much out-of-sample structure can models capture?
3. **Causality:** can a visual change be claimed to cause performance uplift?

Only the first two are addressed directly. Causal claims would require a stronger identification design such as randomized experiments or valid quasi-experimental variation.