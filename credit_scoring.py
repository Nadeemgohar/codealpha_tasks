"""
CodeAlpha ML Internship - Task 1: Credit Scoring Model
Predict whether a borrower will become seriously delinquent (default).

Data
----
Works with the Kaggle "Give Me Some Credit" file (cs-training.csv, target = SeriousDlqin2yrs):
    python credit_scoring.py --data cs-training.csv
If no file is given, a SYNTHETIC dataset with the same columns is generated so the
pipeline can be run and tested end-to-end. Use the real file for your final submission.
"""
import argparse, warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (precision_score, recall_score, f1_score, roc_auc_score,
                             confusion_matrix, classification_report, RocCurveDisplay,
                             precision_recall_curve)

SEED = 42
TARGET = "SeriousDlqin2yrs"


# ------------------------------------------------------------------ data
def make_synthetic(n=30000, seed=SEED):
    rng = np.random.default_rng(seed)
    age = rng.integers(21, 85, n)
    income = np.clip(rng.lognormal(8.6, 0.55, n), 800, 40000)
    util = np.clip(rng.beta(1.2, 3.0, n) * rng.choice([1, 1, 1, 1.6], n), 0, 3)
    debt_ratio = np.clip(rng.gamma(2.0, 0.22, n), 0, 5)
    late30 = rng.poisson(np.clip(0.15 + 1.2 * util * (age < 40), 0, 4))
    late60 = rng.poisson(0.35 * late30 + 0.03)
    late90 = rng.poisson(0.45 * late60 + 0.02)
    open_lines = np.clip(rng.poisson(8, n), 0, 30)
    re_loans = np.clip(rng.poisson(1.0, n), 0, 6)
    deps = np.clip(rng.poisson(0.8, n), 0, 6).astype(float)

    z = (-3.3 + 1.3 * np.minimum(util, 1.5) + 0.35 * late30 + 0.5 * late60 + 0.8 * late90
         + 0.5 * debt_ratio - 0.018 * (age - 45) - 0.00004 * income + 0.05 * deps)
    p = 1 / (1 + np.exp(-z))
    y = (rng.random(n) < p).astype(int)

    df = pd.DataFrame({
        TARGET: y,
        "RevolvingUtilizationOfUnsecuredLines": util,
        "age": age,
        "NumberOfTime30-59DaysPastDueNotWorse": late30,
        "DebtRatio": debt_ratio,
        "MonthlyIncome": income,
        "NumberOfOpenCreditLinesAndLoans": open_lines,
        "NumberOfTimes90DaysLate": late90,
        "NumberRealEstateLoansOrLines": re_loans,
        "NumberOfTime60-89DaysPastDueNotWorse": late60,
        "NumberOfDependents": deps,
    })
    df.loc[rng.random(n) < 0.20, "MonthlyIncome"] = np.nan      # realistic missingness
    df.loc[rng.random(n) < 0.03, "NumberOfDependents"] = np.nan
    return df


def load_data(path):
    if path:
        df = pd.read_csv(path)
        df = df.drop(columns=[c for c in df.columns if c.lower().startswith("unnamed")], errors="ignore")
        print(f"Loaded real dataset: {path}  {df.shape}")
    else:
        df = make_synthetic()
        print(f"No --data given -> using SYNTHETIC data {df.shape}")
    return df


# ------------------------------------------------------------------ feature engineering
def engineer(df):
    df = df.copy()
    # clean obvious outliers / coding errors seen in the real dataset
    df["age"] = df["age"].clip(18, 100)
    df["RevolvingUtilizationOfUnsecuredLines"] = df["RevolvingUtilizationOfUnsecuredLines"].clip(upper=3)
    df["DebtRatio"] = df["DebtRatio"].clip(upper=df["DebtRatio"].quantile(0.99))
    for c in ["NumberOfTime30-59DaysPastDueNotWorse", "NumberOfTime60-89DaysPastDueNotWorse",
              "NumberOfTimes90DaysLate"]:
        df[c] = df[c].where(df[c] < 20, np.nan)         # 96/98 are placeholder codes

    late = ["NumberOfTime30-59DaysPastDueNotWorse", "NumberOfTime60-89DaysPastDueNotWorse",
            "NumberOfTimes90DaysLate"]
    df["IncomeMissing"] = df["MonthlyIncome"].isna().astype(int)
    df["TotalLatePayments"] = df[late].sum(axis=1, skipna=True)
    df["AnyLate"] = (df["TotalLatePayments"] > 0).astype(int)
    df["SeverePaymentScore"] = (df[late[0]].fillna(0) * 1 + df[late[1]].fillna(0) * 2
                                + df[late[2]].fillna(0) * 3)
    df["LogIncome"] = np.log1p(df["MonthlyIncome"])
    df["MonthlyDebtPayment"] = df["DebtRatio"] * df["MonthlyIncome"]
    df["IncomePerDependent"] = df["MonthlyIncome"] / (df["NumberOfDependents"].fillna(0) + 1)
    df["CreditLinesPerYearOfAge"] = df["NumberOfOpenCreditLinesAndLoans"] / (df["age"] - 17)
    return df


# ------------------------------------------------------------------ main
def main(path):
    df = engineer(load_data(path))
    X, y = df.drop(columns=[TARGET]), df[TARGET]
    print(f"Default rate: {y.mean():.2%}  (imbalanced -> using class_weight='balanced' + ROC-AUC/F1)\n")

    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=SEED)

    models = {
        "Logistic Regression": Pipeline([("imp", SimpleImputer(strategy="median")),
                                         ("sc", StandardScaler()),
                                         ("clf", LogisticRegression(max_iter=3000, class_weight="balanced"))]),
        "Decision Tree": Pipeline([("imp", SimpleImputer(strategy="median")),
                                   ("clf", DecisionTreeClassifier(max_depth=6, min_samples_leaf=50,
                                                                  class_weight="balanced", random_state=SEED))]),
        "Random Forest": Pipeline([("imp", SimpleImputer(strategy="median")),
                                   ("clf", RandomForestClassifier(n_estimators=300, min_samples_leaf=20,
                                                                  class_weight="balanced_subsample",
                                                                  n_jobs=-1, random_state=SEED))]),
    }

    cv = StratifiedKFold(5, shuffle=True, random_state=SEED)
    rows = []
    for name, m in models.items():
        cv_auc = cross_val_score(m, X_tr, y_tr, cv=cv, scoring="roc_auc").mean()
        m.fit(X_tr, y_tr)
        proba = m.predict_proba(X_te)[:, 1]
        pred = (proba >= 0.5).astype(int)
        rows.append({"Model": name, "CV ROC-AUC": cv_auc,
                     "Precision": precision_score(y_te, pred, zero_division=0),
                     "Recall": recall_score(y_te, pred),
                     "F1": f1_score(y_te, pred),
                     "ROC-AUC": roc_auc_score(y_te, proba)})
    res = pd.DataFrame(rows).set_index("Model").round(4)
    print("=== Model comparison (test set, threshold 0.5) ===")
    print(res.to_string(), "\n")
    res.to_csv("model_comparison.csv")

    best_name = res["ROC-AUC"].idxmax()
    best = models[best_name]
    proba = best.predict_proba(X_te)[:, 1]

    # choose a business threshold that maximises F1
    p, r, t = precision_recall_curve(y_te, proba)
    f1s = 2 * p[:-1] * r[:-1] / (p[:-1] + r[:-1] + 1e-12)
    thr = t[f1s.argmax()]
    pred = (proba >= thr).astype(int)
    print(f"Best model: {best_name}  |  F1-optimal threshold = {thr:.3f}")
    print(classification_report(y_te, pred, target_names=["good", "default"]))

    # plots
    fig, ax = plt.subplots(figsize=(6, 5))
    for n, m in models.items():
        RocCurveDisplay.from_estimator(m, X_te, y_te, name=n, ax=ax)
    ax.plot([0, 1], [0, 1], "k:", alpha=.4); ax.set_title("ROC curves")
    plt.tight_layout(); plt.savefig("roc_curves.png", dpi=120); plt.close()

    plt.figure(figsize=(4.5, 4))
    sns.heatmap(confusion_matrix(y_te, pred), annot=True, fmt="d", cmap="Blues",
                xticklabels=["good", "default"], yticklabels=["good", "default"])
    plt.xlabel("Predicted"); plt.ylabel("Actual"); plt.title(f"{best_name} @ thr={thr:.2f}")
    plt.tight_layout(); plt.savefig("confusion_matrix.png", dpi=120); plt.close()

    rf = models["Random Forest"].named_steps["clf"]
    imp = pd.Series(rf.feature_importances_, index=X.columns).sort_values().tail(12)
    plt.figure(figsize=(7, 5)); imp.plot.barh(); plt.title("Feature importance (Random Forest)")
    plt.tight_layout(); plt.savefig("feature_importance.png", dpi=120); plt.close()

    joblib.dump({"model": best, "threshold": thr, "features": list(X.columns)}, "credit_model.joblib")
    print("Saved credit_model.joblib, plots, model_comparison.csv")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=None, help="path to cs-training.csv (optional)")
    main(ap.parse_args().data)
