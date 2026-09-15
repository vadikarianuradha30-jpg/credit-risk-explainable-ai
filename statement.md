# Project Statement

## Title
Explainable AI Loan/Credit Risk Approval System with Bias Audit

## Problem Statement
Financial institutions increasingly rely on machine learning models to
approve or reject loan applications. However, these models are often
"black boxes" — they produce a decision without explaining the reasoning
behind it, and they can unintentionally treat certain demographic groups
unfairly even when applicants have similar financial profiles. This lack
of transparency and accountability creates real risks: applicants cannot
understand why they were rejected, and biased outcomes can go undetected
until they cause harm or regulatory issues.

## Scope
This project builds a credit risk prediction system that goes beyond a
standard classifier by adding two additional layers:

1. **Explainability** — for any individual prediction, the system reports
   which factors most influenced the decision and by how much, using SHAP
   (SHapley Additive exPlanations).
2. **Fairness Auditing** — the system checks whether approval rates differ
   significantly across demographic groups (age, sex) and flags any gap
   that may indicate bias.

The project uses the German Credit Risk dataset (1,000 historical loan
applications) and is implemented as a fully command-line-executable
Python application.

## Target Users
- Bank risk analysts who need to justify credit decisions
- Regulators/auditors reviewing AI systems for fairness compliance
- Students and researchers studying explainable and responsible AI

## High-Level Features
- Data preprocessing pipeline (cleaning, encoding, train/test split)
- Credit risk prediction model (Random Forest / Logistic Regression)
- Per-applicant explanation of predictions (SHAP waterfall plots)
- Global feature importance summary across all predictions
- Fairness audit comparing approval rates across age and sex groups,
  with an automatic flag when the gap exceeds 10%
- Unit tests covering the data pipeline
- Fully executable via command line, no GUI required

## Key Finding (from this project's results)
The trained model achieved 77% accuracy on held-out test data. The
fairness audit revealed a negligible 1.6% approval-rate gap by sex, but
a notable 13.8% approval-rate gap by age (30-and-above applicants
approved at a higher rate than under-30 applicants) — demonstrating why
explainability and bias auditing must be built into credit AI systems
rather than added as an afterthought.