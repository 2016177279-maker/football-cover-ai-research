# Data Governance and Reproducibility

## Final Dataset

The final research dataset contains **5,200 unique videos** and is treated as a frozen analytical asset for downstream modeling.

The project distinguishes between:

- raw collection outputs;
- eligibility / human-review decisions;
- final sampled data;
- visual-feature tables;
- engagement / channel controls;
- model outputs;
- AIGC candidate outputs.

This separation reduces accidental overwriting and makes it easier to trace how a result was produced.

## Version and Lineage Principles

Key research assets are versioned explicitly. The workflow records, where applicable:

- collection run identifiers;
- filter versions;
- schema versions;
- sampling-protocol versions;
- frozen dataset hashes or manifests;
- model configuration and evaluation outputs.

The principle is that the final analytical result should be reproducible from declared inputs rather than reconstructed from memory.

## Human Review

Borderline football-relevance cases are resolved through structured human review rather than silently forcing uncertain examples into automated labels.

Human-label workflows are separated from model predictions to reduce confirmation bias where practical.

## Public Repository Boundary

This repository is intentionally a disclosure-safe portfolio version. It should not contain:

- `.env` files;
- YouTube or other API credentials;
- OAuth tokens or private account information;
- confidential employer material;
- private business data;
- large raw exports or full thumbnail caches unless publication rights and platform terms clearly permit it;
- local absolute file paths containing personal information.

Instead, the public repository should use documentation, code, schemas, synthetic examples and small disclosure-safe samples to demonstrate the workflow.

## Reproducibility vs. Data Redistribution

Reproducible research does not require republishing every raw artifact. The preferred public pattern is:

1. document how data were obtained;
2. publish code and schemas where disclosure is permitted;
3. publish small example data or synthetic fixtures;
4. document frozen-sample statistics and validation results;
5. avoid redistributing material when platform terms, privacy, confidentiality or ownership are unclear.

## Research Integrity

The project does not interpret observational associations as causal effects. Negative or unstable validation results are preserved as part of the research record rather than removed for presentation purposes.