# ============================================================
# Phase 2: XGBoost Regularization & Sampling
# ============================================================

import pandas as pd
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

# -----------------------------
# 3. Phase 2 Model (Regularized)
# -----------------------------
model = XGBClassifier(
    n_estimators=300,
    max_depth=5,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.1,
    reg_lambda=1.0,
    eval_metric="logloss",
    random_state=42
)

model.fit(X_train, y_train)

train_acc = accuracy_score(y_train, model.predict(X_train))
test_acc = accuracy_score(y_test, model.predict(X_test))

print("\n=== Phase 2 Results ===")
print(f"Training Accuracy: {train_acc:.4f}")
print(f"Testing  Accuracy: {test_acc:.4f}")

# -----------------------------
# 4. Save Phase 2 Model
# -----------------------------
pickle.dump(model, open("XGBoost_phase2_best.dat", "wb"))
print("Model saved as: XGBoost_phase2_best.dat")
