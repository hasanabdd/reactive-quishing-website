# ============================================================
# Phase 1: XGBoost Hyperparameter Tuning (Tree Structure)
# This file is a SAFE COPY of the original training pipeline.
# The original qr_model.py is NOT modified.
# ============================================================

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
import pickle

# -----------------------------
# 1. Load Dataset
# -----------------------------
data0 = pd.read_csv("5.urldata.csv")

# -----------------------------
# 2. Data Preprocessing
# -----------------------------
data = data0.drop(['Domain'], axis=1).copy()
data = data.sample(frac=1, random_state=42).reset_index(drop=True)

y = data['Label']
X = data.drop('Label', axis=1)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=12
)

print("Dataset loaded successfully")
print("Training samples:", X_train.shape[0])
print("Testing samples :", X_test.shape[0])

# -----------------------------
# 3. Phase 1: Tree Structure Tuning
# -----------------------------
configs = [
    {"n_estimators": 200, "max_depth": 5, "learning_rate": 0.1},
    {"n_estimators": 300, "max_depth": 6, "learning_rate": 0.1},
    {"n_estimators": 400, "max_depth": 6, "learning_rate": 0.05},
    {"n_estimators": 500, "max_depth": 7, "learning_rate": 0.05},
]

best_model = None
best_test_acc = 0.0
best_config = None

print("\n=== Phase 1: XGBoost Tree Structure Tuning ===")

for cfg in configs:
    print("\nTesting configuration:", cfg)

    model = XGBClassifier(
        n_estimators=cfg["n_estimators"],
        max_depth=cfg["max_depth"],
        learning_rate=cfg["learning_rate"],
        eval_metric="logloss",
        random_state=42
    )

    model.fit(X_train, y_train)

    train_acc = accuracy_score(y_train, model.predict(X_train))
    test_acc = accuracy_score(y_test, model.predict(X_test))

    print(f"Training Accuracy: {train_acc:.4f}")
    print(f"Testing  Accuracy: {test_acc:.4f}")

    if test_acc > best_test_acc:
        best_test_acc = test_acc
        best_model = model
        best_config = cfg

# -----------------------------
# 4. Save Best Phase 1 Model
# -----------------------------
pickle.dump(best_model, open("XGBoost_phase1_best.dat", "wb"))

print("\n=== Phase 1 Completed ===")
print("Best Configuration:", best_config)
print(f"Best Phase 1 Test Accuracy: {best_test_acc:.4f}")
print("Model saved as: XGBoost_phase1_best.dat")
