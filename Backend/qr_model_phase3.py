# ============================================================
# Phase 3: XGBoost Controlled Boosting (Final Optimization)
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
# 3. Phase 3 Model
# -----------------------------
model = XGBClassifier(
    n_estimators=800,
    max_depth=7,
    learning_rate=0.03,
    subsample=1.0,
    colsample_bytree=1.0,
    reg_alpha=0.0,
    reg_lambda=1.0,
    eval_metric="logloss",
    random_state=42
)

model.fit(X_train, y_train)

train_acc = accuracy_score(y_train, model.predict(X_train))
test_acc = accuracy_score(y_test, model.predict(X_test))

print("\n=== Phase 3 Results ===")
print(f"Training Accuracy: {train_acc:.4f}")
print(f"Testing  Accuracy: {test_acc:.4f}")

# -----------------------------
# 4. Save Phase 3 Model
# -----------------------------
pickle.dump(model, open("XGBoost_phase3_best.dat", "wb"))
print("Model saved as: XGBoost_phase3_best.dat")
