# Explainable AI Loan/Credit Risk Approval System with Bias Audit

An AI-based loan approval system that predicts credit risk, explains
each prediction using SHAP, and audits the model for fairness across
demographic groups (age, sex).

## Overview

This project trains a machine learning model on the German Credit Risk
dataset to predict whether a loan applicant is a good or bad credit
risk. Unlike a typical classifier, it also:
- Explains *why* each prediction was made (explainability module)
- Checks whether the model treats different demographic groups fairly
  (fairness audit module)

All functionality runs from the command line — no GUI required.

## Features
- Data cleaning and preprocessing pipeline
- Random Forest / Logistic Regression credit risk classifier
- SHAP-based explanation for individual predictions and global feature
  importance
- Fairness audit comparing approval rates by age and sex
- Unit tests for the data pipeline

## Tech Stack
- Python 3.8+
- pandas, numpy — data handling
- scikit-learn — model training
- SHAP — explainability
- matplotlib — plots
- pytest — testing

## Project Structure
credit-risk-explainable-ai/
├── data/
│ └── german_credit_data.csv
├── src/
│ ├── data_processing.py
│ ├── train_model.py
│ ├── explain.py
│ └── fairness_audit.py
├── tests/
│ └── test_data_processing.py
├── results/ (created automatically when scripts run)
├── report/
│ └── diagrams/
├── requirements.txt
├── statement.md
└── README.md


## Setup Instructions

### 1. Prerequisites
- Python 3.8 or higher installed
- pip (comes with Python)

### 2. Clone the repository

git clone https://github.com/vadikarianuradha30-jpg/credit-risk-explainable-ai
cd credit-risk-explainable-ai


### 3. Install dependencies

pip install -r requirements.txt


### 4. Dataset
The dataset (`german_credit_data.csv`) is already included in the
`data/` folder. No additional download is required.

## How to Run

Run each script from the project root folder.

### Step 1 — Preprocess data (optional, for verification)

python src/data_processing.py

Prints the shape and sample of the processed training/test data.

### Step 2 — Train the model

python src/train_model.py --model random_forest

Trains the model, prints accuracy/precision/recall/F1, and saves the
trained model to `results/random_forest_model.pkl`.

You can also try:

python src/train_model.py --model logistic


### Step 3 — Explain predictions
Explain a single applicant's prediction (replace `5` with any test-set
index):

python src/explain.py --applicant_id 5

Generate the overall feature importance summary:

python src/explain.py --summary

Both save chart images to the `results/` folder.

### Step 4 — Run the fairness audit

python src/fairness_audit.py --check sex
python src/fairness_audit.py --check age

Prints approval rates per group and appends findings to
`results/audit_report.md`.

## Testing

Run the unit test suite from the project root:

pytest tests/

All tests should pass, verifying the data pipeline works correctly
(correct split sizes, no missing values, proper target encoding).

## Sample Results
- Model accuracy: ~77%
- Fairness audit found a 13.8% approval-rate gap between age groups,
  flagged as a potential fairness concern
- Full details available in `results/audit_report.md` after running
  the fairness audit scripts
