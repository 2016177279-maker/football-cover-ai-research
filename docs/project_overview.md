# Project Overview

## Problem

Football social-media thumbnails are highly visual, but content-performance decisions are often based on intuition. This project asks whether public video and thumbnail data can be turned into a reproducible evidence base for content decisions.

The analytical question is:

> Which thumbnail visual characteristics are associated with stronger public content-performance indicators after accounting for channel, time and content differences?

## Final Scope

- Final frozen research sample: **5,200 unique videos**.
- Domain: football / association-football content on YouTube.
- Data: public video, channel and thumbnail information retrieved for research.
- Outputs: structured dataset, multimodal visual features, statistical / ML analysis, model validation, business insights, AIGC optimization rules and a Streamlit proof of concept.

## End-to-End Workflow

1. Define the research and decision question.
2. Collect public YouTube video candidates under quota and checkpoint constraints.
3. Filter for football relevance and resolve uncertain cases through human review.
4. Deduplicate and construct a stratified final sample.
5. Freeze schema, dataset and version lineage before downstream modeling.
6. Extract Basic CV, OCR, saliency, face/object geometry, aesthetic and semantic features.
7. Join public engagement and channel controls.
8. Estimate regression and machine-learning models.
9. Evaluate grouped holdout, temporal generalization and robustness.
10. Translate stable evidence into constrained AIGC design rules.
11. Re-extract candidate features and compare candidates in a human-in-the-loop Streamlit PoC.

## Why This Is a Data Analytics Project

The core value is not image generation. The project demonstrates the full chain from a broad business question to evidence and decision support:

**Business Question → Data → Quality Control → Feature Engineering → Modeling → Validation → Insight → Decision Rule → PoC**

This structure makes the project relevant to business analytics, data analytics, technology risk, digital transformation and applied-AI consulting roles.

## Skills Demonstrated

- Python-based data processing and analysis
- Data cleaning, deduplication and sample design
- KPI and variable-system design
- Computer Vision, OCR and multimodal feature extraction
- Regression and machine learning
- Robustness and uncertainty analysis
- Model interpretability
- Responsible AI / model-risk thinking
- Human-in-the-loop product prototyping
- Reproducibility, versioning and Git workflow

## Public Disclosure Boundary

This public repository is a portfolio representation, not a dump of the working directory. It intentionally excludes:

- API keys and `.env` files;
- private credentials;
- confidential employer or university material;
- large raw data exports and thumbnail caches;
- any material the project owner does not have the right to publish.

Only independently owned code, documentation, public-data methodology and disclosure-safe examples should be committed.