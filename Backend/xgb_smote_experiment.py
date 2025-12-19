import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE

# -----------------------------
# Load Dataset
# -----------------------------
data0 = pd.read_csv("5.urldata.csv")
data = data0.drop(['Domain'], axis=1).copy()
data = data.sample(frac=1, random_state=42).reset_index(drop=True)

y = data['Label']
X = data.drop('Label', axis=1)

# -----------------------------
# Train / Test Split (IMPORTANT)
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=12
)

# -----------------------------
# Apply SMOTE (TRAINING ONLY)
# -----------------------------
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

print("Original training samples:", X_train.shape[0])
print("SMOTE training samples   :", X_train_smote.shape[0])

# -----------------------------
# Train XGBoost
# -----------------------------
model = XGBClassifier(
    n_estimators=400,
    max_depth=7,
    learning_rate=0.1,
    eval_metric="logloss",
    random_state=42
)

model.fit(X_train_smote, y_train_smote)

train_acc = accuracy_score(y_train_smote, model.predict(X_train_smote))
test_acc = accuracy_score(y_test, model.predict(X_test))

print("\n=== SMOTE Experiment Results ===")
print(f"Training Accuracy (SMOTE): {train_acc:.4f}")
print(f"Testing  Accuracy         : {test_acc:.4f}")
