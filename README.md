# Income Classification

This repository implements a binary income-classification pipeline (predicting income >50K) using a custom logistic regression learner trained with gradient ascent. It includes data cleaning, exploratory visualizations, a preprocessing pipeline, a hand-written logistic regression classifier, performance plots (accuracy, log-likelihood, ROC), and a short report summarizing results and conclusions.

## Files

- proj1_data_analysis.ipynb — data cleaning, EDA, and visualization functions (correlation heatmap, histograms).
- proj1_classification.ipynb — preprocessing pipeline, custom LogisticRegression class (gradient ascent), training loop, and plotting utilities (accuracy, log-likelihood, ROC).
- train_data.csv, test_data.csv — original raw datasets
- cleaned_train_data.csv, cleaned_test_data.csv — cleaned datasets produced by the cleaning routine.
- proj1_report.pdf — written report with figures, interpretations, and learning-rate comparisons.
- README.md — this file.

## Project Overview

- **Dataset**: tabular census-style income data (categorical + numeric features; income label <=50K or >50K).
- **Objective**: implement and analyze a logistic regression classifier from scratch to predict income >50K, exploring preprocessing choices and learning-rate behavior.
- **Audience**: instructors, students, and practitioners learning model implementation and evaluation.

## Results

1. Correlations
    - Strongest positive correlation with income: capital.gain (≈ 0.31)
    - Weakest correlation with income: capital.loss (≈ 0.15)
    - Age shows moderate positive correlation with income (≈ 0.24)
2. Feature distributions
    - Most samples earn <=50K across education levels except Masters, Prof-school, Doctorate where >50K is more common.
    - Those earning >50K tend to work at least 30 hours/week; most samples cluster around 40 hours/week.
3. Learning rate comparison
    - Learning rate 0.75 reaches higher accuracy faster (≈0.84 vs 0.80 by ~100 iterations) and increases average log-likelihood more quickly than 0.05.
4. ROC / AUC
    - For learning rate 0.75: train AUC ≈ 0.90, test AUC ≈ 0.89
    - For learning rate 0.05: train AUC ≈ 0.87, test AUC ≈ 0.86

## Interpretation

- A straightforward logistic regression trained with gradient ascent and well-engineered features can achieve strong discrimination (test AUC ≈ 0.89).
- Important predictors include capital.gain and hours worked; adding nonlinear features or richer models may improve results.
- Preprocessing (encoding, scaling, outlier handling) and hyperparameter choices (learning rate) significantly affect convergence and performance.


