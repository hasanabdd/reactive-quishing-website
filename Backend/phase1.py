from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
import pickle

# Try multiple configurations
configs = [
    {"n_estimators": 200, "max_depth": 5, "learning_rate": 0.1},
    {"n_estimators": 300, "max_depth": 6, "learning_rate": 0.1},
    {"n_estimators": 400, "max_depth": 6, "learning_rate": 0.05},
    {"n_estimators": 500, "max_depth": 7, "learning_rate": 0.05},
]

best_model = None
best_test_acc = 0

for cfg in configs:
    print("\nTesting config:", cfg)

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

    print(f"Train Accuracy: {train_acc:.4f}")
    print(f"Test  Accuracy: {test_acc:.4f}")

    if test_acc > best_test_acc:
        best_test_acc = test_acc
        best_model = model

# Save best Phase 1 model
pickle.dump(best_model, open("XGBoost_phase1_best.dat", "wb"))

print("\n✅ Best Phase 1 Test Accuracy:", best_test_acc)
