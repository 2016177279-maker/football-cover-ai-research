# Football Content Performance Analytics & AI Optimization

An end-to-end **data analytics + applied AI research project** that studies how visual characteristics of football-related YouTube thumbnails are associated with public content-performance indicators, and translates validated evidence into constrained AIGC optimization rules.

> **Final research dataset: 5,200 football-related videos.**
>
> **Core workflow:** Business Question → Data Collection → Data Governance → Multimodal Feature Engineering → Statistical / ML Analysis → Model Validation → Business Insights → AIGC Optimization PoC.

This repository is designed as a public research and portfolio artifact. It uses publicly available YouTube data and does **not** contain private platform data, user credentials, API keys, confidential employer materials, or a commercial prediction service.

**Project policies:** [Privacy Policy](PRIVACY.md) · [Terms of Use](TERMS.md)

---

## 1. Business Question

The project begins with a practical content-analytics question:

> **Which thumbnail visual characteristics are consistently associated with stronger content performance in football social media, after accounting for channel, time and content differences?**

The goal is not only to build a predictive model. The project is structured to turn a broad business question into a reproducible analytical workflow and then convert evidence into decision rules that can support content optimization.

---

## 2. What the Project Demonstrates

### Data analytics
- public-data collection and quality control;
- eligibility filtering, deduplication and stratified sampling;
- KPI and variable-system design;
- feature engineering for structured and unstructured data;
- regression, machine learning and robustness analysis;
- model interpretation and business-insight translation.

### Applied AI
- multimodal visual feature extraction using Computer Vision, OCR and CLIP-style semantic representations;
- evidence-backed AIGC prompt optimization;
- candidate comparison, re-extraction and ranking;
- Streamlit proof of concept with human-in-the-loop decision support.

### Governance and validation
- frozen datasets and explicit schema/version control;
- lineage and reproducibility checks;
- grouped holdout and temporal validation;
- prediction intervals and abstention logic;
- explicit separation between **association, prediction and causality**.

---

## 3. Research Pipeline

```text
Public YouTube Data
        ↓
Eligibility Filtering & Human Review
        ↓
Deduplication / QC / Stratified Sampling
        ↓
Frozen Final Dataset (N = 5,200)
        ↓
Basic + Advanced Visual Features
        ↓
Regression + Machine Learning
        ↓
Grouped / Temporal / Robustness Validation
        ↓
SHAP-style Interpretation & Heterogeneity Analysis
        ↓
Evidence-backed Design Rules
        ↓
Prompt Optimizer + Streamlit PoC
        ↓
Human Review / Abstention
```

---

## 4. Analytical Methods

The project combines classical statistics, machine learning and multimodal feature engineering.

**Visual feature groups** include:
- text burden and OCR-derived variables;
- person / face / object geometry;
- composition and center saliency;
- visual entropy and saturation;
- aesthetic coherence;
- semantic representations and content controls.

**Modeling and validation** include:
- multivariate regression;
- channel-clustered standard errors;
- channel fixed effects and heterogeneity checks;
- machine-learning models;
- grouped holdout evaluation;
- temporal validation;
- Bootstrap-based uncertainty analysis;
- SHAP / PDP / ICE / ALE-style interpretation where applicable.

See [Methodology](docs/methodology.md) and [Model Validation](docs/model_validation.md).

---

## 5. Selected Validation Results

On grouped holdout evaluation, representative model performance reached:

| Target | Grouped Holdout R² |
|---|---:|
| Views | 0.602 |
| Velocity | 0.650 |
| Engagement | 0.150 |

Temporal validation is materially harder: a later-period Velocity test produced **R² = -0.233**. This is treated as an important model-risk finding rather than hidden as a failure.

**Interpretation:** the models can capture useful cross-sectional structure, but temporal generalization is unstable. The system therefore does not present itself as a guaranteed future-performance predictor.

See [Model Validation](docs/model_validation.md).

---

## 6. Evidence-backed Business Insights

Selected modeled associations for an interquartile-range change include:

- **Aesthetic coherence:** +28.2% modeled Velocity association (95% CI +16.5% to +41.0%).
- **Text coverage:** -23.6% (95% CI -30.4% to -16.2%).
- **Saturation:** +18.5% (95% CI +4.5% to +34.3%), but machine-learning evidence is mixed.
- **Visual entropy:** -13.6% (95% CI -21.4% to -5.1%).

These results are used as **decision evidence**, not causal uplift claims. The practical design direction prioritizes lower text burden, stronger aesthetic coherence and clearer visual focus while treating complexity and saturation more cautiously.

See [Business Insights](docs/business_insights.md).

---

## 7. AIGC Optimization PoC

The final research output translates model evidence into constrained optimization rules rather than asking a generative model to redesign thumbnails freely.

Representative intervention families include:

- **B1 — Text simplification:** reduce redundant or competing text while preserving factual meaning.
- **B2 — Aesthetic coherence:** improve visual consistency without inventing new event facts.
- **B3 — Visual focus:** strengthen one clear near-center focal subject and reduce avoidable clutter.

The Streamlit PoC supports:
- original-thumbnail analysis;
- candidate comparison and feature re-extraction;
- propagation-score comparison;
- **90% prediction intervals**;
- candidate ranking and abstention;
- up to two human decision rounds.

The public portfolio includes an offline-safe test and demo layer so reviewers can inspect the workflow without YouTube API credentials, production data, model checkpoints or image-generation keys.

---

## 8. Responsible AI / Model Risk Principles

This project deliberately avoids several common overclaims:

- it is **not a CTR predictor**;
- it does **not** guarantee higher future views or engagement;
- observational associations are not presented as causal effects;
- generated candidates must preserve the underlying team, player, match, event and title semantics;
- the system must not invent scores, transfers, injuries, trophies, quotes or news events;
- unstable temporal generalization is disclosed;
- uncertainty, abstention and human review are part of the decision process.

See [Responsible AI](docs/responsible_ai.md).

---

## 9. Repository Guide

- [Project Overview](docs/project_overview.md) — recruiter-friendly summary of the problem, workflow and outputs.
- [Methodology](docs/methodology.md) — sampling, features, modeling and analytical design.
- [Data Governance](docs/data_governance.md) — freezing, lineage, privacy and reproducibility principles.
- [Model Validation](docs/model_validation.md) — grouped holdout, temporal validation and model-risk interpretation.
- [Business Insights](docs/business_insights.md) — evidence translated into decision-oriented recommendations.
- [Responsible AI](docs/responsible_ai.md) — constraints, uncertainty and human-in-the-loop safeguards.

Public code and example artifacts should include only materials that are independently owned and safe to disclose. Large raw datasets, thumbnails, credentials and confidential materials are intentionally excluded.

---

## 10. Use of the YouTube Data API

The project uses the YouTube Data API with an API key to retrieve public information required for academic research, which may include video IDs, titles, descriptions, publication dates, public statistics, public channel metadata, publicly accessible thumbnails and public search results.

The project does not use OAuth to access user-authorized private data and does not obtain or store Google user passwords or private YouTube information.

---

## 11. Independent Project Notice

Football Content Performance Analytics & AI Optimization is an independent research project. It is not endorsed by, sponsored by, affiliated with, or operated by Google or YouTube.

Use of YouTube API Services remains subject to the YouTube Terms of Service, YouTube API Services Terms of Service, YouTube API Services Developer Policies and Google Privacy Policy.

This repository should not be interpreted as an official product or endorsement of any employer, university, platform or football organization.