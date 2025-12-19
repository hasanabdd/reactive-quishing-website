import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
import pickle

# -----------------------------
# Load Dataset
# -----------------------------
data0 = pd.read_csv("5.urldata.csv")
data = data0.drop(['Domain'], axis=1).copy()
data = data.sample(frac=1, random_state=42).reset_index(drop=True)

y = data['Label']
X = data.drop('Label', axis=1)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=12
)

print("RF Phase 1: Capacity Expansion")

configs = [
    {"n_estimators": 200, "max_depth": 10},
    {"n_estimators": 400, "max_depth": 15},
    {"n_estimators": 600, "max_depth": None}
]

best_acc = 0
best_model = None
best_cfg = None

for cfg in configs:
    print("\nTesting:", cfg)

    rf = RandomForestClassifier(
        n_estimators=cfg["n_estimators"],
        max_depth=cfg["max_depth"],
        random_state=42,
        n_jobs=-1
    )

    rf.fit(X_train, y_train)

    train_acc = accuracy_score(y_train, rf.predict(X_train))
    test_acc = accuracy_score(y_test, rf.predict(X_test))

    print(f"Train Acc: {train_acc:.4f}")
    print(f"Test  Acc: {test_acc:.4f}")

    if test_acc > best_acc:
        best_acc = test_acc
        best_model = rf
        best_cfg = cfg

pickle.dump(best_model, open("RF_phase1_best.dat", "wb"))
print("\nBest Config:", best_cfg)
print("Best Test Accuracy:", best_acc)
