# CodeAlpha_CreditScoring
Task 1 - Credit Scoring Model (CodeAlpha ML Internship)

**Goal:** predict whether a borrower will become seriously delinquent within 2 years.
**Data:** Kaggle "Give Me Some Credit" (`cs-training.csv`). If the file is not supplied the script
generates a synthetic dataset with the same columns so the pipeline can be tested.
**Feature engineering:** outlier capping, placeholder-code cleaning (96/98), missing-income flag,
total/weighted late payments, log income, monthly debt payment, income per dependent, credit lines per year of adult age.
**Models:** Logistic Regression, Decision Tree, Random Forest (class-weighted for imbalance).
**Metrics:** Precision, Recall, F1, ROC-AUC, plus an F1-optimal decision threshold.

## Run
```
pip install -r requirements.txt
python credit_scoring.py --data cs-training.csv
```
Outputs: model_comparison.csv, roc_curves.png, confusion_matrix.png, feature_importance.png, credit_model.joblib.
