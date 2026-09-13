# PHSO Machine Learning Technical Assessment

## Overview

This repository contains my solution to the Parliamentary and Health Service Ombudsman (PHSO) machine learning technical assessment.

The objective is to predict the `SeverityScore` for unseen holdback cases and provide an indication of whether a case is likely to require progression to investigation.

## Approach

The analysis followed these main steps:

1. Loaded and inspected the supplied case data.
2. Reviewed the target distribution and missing values.
3. Prepared numerical and categorical features using preprocessing pipelines.
4. Split the training data into training and validation sets using stratification.
5. Established a Dummy Classifier baseline.
6. Compared Logistic Regression, Random Forest and HistGradientBoosting models.
7. Evaluated models using accuracy, macro F1 and weighted F1.
8. Evaluated the binary investigation decision separately, with particular attention to recall.
9. Performed feature importance and feature ablation analysis.
10. Selected the Random Forest model for the final predictions.
11. Applied the final model to the 500 holdback cases.
12. Performed quality checks on the final predictions.

## Model Selection

Random Forest provided the strongest overall performance among the models evaluated.

On the validation set, the selected Random Forest achieved:

- Accuracy: 0.775
- Macro F1: 0.780
- Weighted F1: 0.773
- Investigation recall: 0.908

The model was selected based on its overall classification performance and its ability to identify cases requiring investigation.

## Holdback Predictions

The final model was applied to the 500 holdback cases.

Quality checks confirmed:

- 500 holdback cases were processed
- 500 predictions were generated
- 500 unique case references were retained
- No missing severity predictions
- No missing investigation probabilities
- No missing investigation decisions

The final severity predictions are provided in `phso_predictions.csv`.

## Repository Contents

- `ML_Technical_Assessment.ipynb` - analysis, modelling and evaluation notebook
- `phso_predictions.csv` - final predictions for the holdback cases
- `requirements.txt` - Python dependencies
- `.gitignore` - files excluded from version control

## Reproducibility

The analysis was developed in Python using a Jupyter/Google Colab environment.

To install the required packages:

```bash
pip install -r requirements.txt
