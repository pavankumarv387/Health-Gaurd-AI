"""Train Logistic Regression + Random Forest for Diabetes and Heart Disease,
save models, metrics and the diabetes -> heart link analysis.

Put diabetes.csv and heart.csv (your two datasets, with header row) next to this file.
Run:  python train.py
"""
import os
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, ConfusionMatrixDisplay)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

os.makedirs("models", exist_ok=True)
os.makedirs("results", exist_ok=True)

# Columns are assigned by position so slight header spelling differences don't matter.
D_COLS = ["Pregnancies", "Glucose", "BloodPressure", "SkinThickness", "Insulin",
          "BMI", "DiabetesPedigreeFunction", "Age", "Outcome"]
H_COLS = ["age", "sex", "chest_pain_type", "resting_bp", "cholesterol",
          "fasting_blood_sugar", "resting_ecg", "max_heart_rate", "exercise_angina",
          "oldpeak", "slope", "num_major_vessels", "thalassemia", "target"]

diab = pd.read_csv("diabetes.csv")
diab.columns = D_COLS
heart = pd.read_csv("heart.csv")
heart.columns = H_COLS

# ---- cleaning ----
for c in ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]:
    diab[c] = diab[c].replace(0, np.nan)          # 0 is not a valid reading
    diab[c] = diab[c].fillna(diab[c].median())
heart = heart.fillna(heart.median(numeric_only=True))
heart["target"] = (heart["target"] > 0).astype(int)  # 1 = disease present


def train(name, df, target):
    X, y = df.drop(columns=[target]), df[target]
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    models = {
        "lr": make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)),
        "rf": RandomForestClassifier(n_estimators=200, random_state=42),
    }
    rows = []
    for key, m in models.items():
        m.fit(Xtr, ytr)
        pred, prob = m.predict(Xte), m.predict_proba(Xte)[:, 1]
        rows.append({"disease": name, "model": key.upper(),
                     "accuracy": accuracy_score(yte, pred),
                     "precision": precision_score(yte, pred),
                     "recall": recall_score(yte, pred),
                     "f1": f1_score(yte, pred),
                     "roc_auc": roc_auc_score(yte, prob)})
        joblib.dump(m, f"models/{name}_{key}.joblib")
        ConfusionMatrixDisplay.from_predictions(yte, pred)
        plt.title(f"{name} - {key.upper()}")
        plt.savefig(f"results/cm_{name}_{key}.png", dpi=150, bbox_inches="tight")
        plt.close()
    return rows, models["rf"], X.columns


rows_d, _, _ = train("diabetes", diab, "Outcome")
rows_h, rf_heart, heart_cols = train("heart", heart, "target")
metrics = pd.DataFrame(rows_d + rows_h).round(3)
metrics.to_csv("results/metrics.csv", index=False)
print("\n=== Model comparison ===\n", metrics.to_string(index=False))

# ---- Diabetes -> Heart link analysis ----
# heart.csv contains fasting_blood_sugar (>120 mg/dl = 1), a diabetes marker.
ct = pd.crosstab(heart["fasting_blood_sugar"], heart["target"])
chi2, p, _, _ = chi2_contingency(ct)
rate = heart.groupby("fasting_blood_sugar")["target"].mean()
print("\n=== Heart disease rate by fasting blood sugar ===")
print(rate.round(3).to_string())
print(f"Chi-square p-value: {p:.4f}  ({'significant' if p < 0.05 else 'not significant'} at 0.05)")

imp = pd.Series(rf_heart.feature_importances_, index=heart_cols).sort_values()
print("\nFasting-sugar importance rank (1 = most important):",
      int(imp.rank(ascending=False)["fasting_blood_sugar"]), "of", len(imp))

fig, ax = plt.subplots(1, 2, figsize=(11, 4))
rate.rename({0: "FBS <= 120", 1: "FBS > 120"}).plot.bar(ax=ax[0], rot=0, color=["#4c9", "#e66"])
ax[0].set_title("Heart disease rate vs fasting blood sugar")
ax[0].set_ylabel("Proportion with heart disease")
imp.plot.barh(ax=ax[1])
ax[1].set_title("Random Forest feature importance (heart)")
plt.tight_layout()
plt.savefig("results/diabetes_heart_link.png", dpi=150)
print("\nSaved: models/*.joblib, results/metrics.csv, results/*.png")
