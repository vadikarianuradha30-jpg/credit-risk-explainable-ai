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