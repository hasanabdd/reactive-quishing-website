import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os
import keras
from keras.layers import Input, Dense
from keras import regularizers
import tensorflow as tf
from keras.models import Model
from sklearn import metrics
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_curve,
    auc,
    ConfusionMatrixDisplay
)
import matplotlib.pyplot as plt


data0 = pd.read_csv('5.urldata.csv')
data0.head()
data0.shape
data0.columns
data0.info()
axs = data0.hist(bins=30, figsize=(22, 20), layout=(6, 3))
plt.subplots_adjust(hspace=0.6, wspace=0.4)
plt.show()

# heatmap
plt.figure(figsize=(15, 13))
sns.heatmap(data0.select_dtypes(include=[np.number]).corr())
plt.show()

# Data Preprocessing
data0.describe()
data = data0.drop(['Domain'], axis=1).copy()
data.isnull().sum()
data = data.sample(frac=1).reset_index(drop=True)
data.head()

# Splitting the Data
y = data['Label']
X = data.drop('Label', axis=1)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=12)
X_train.shape, X_test.shape

# Machine Learning Models
from sklearn.metrics import accuracy_score
ML_Model = []
acc_train = []
acc_test = []

def storeResults(model, a, b):
    ML_Model.append(model)
    acc_train.append(round(a, 3))
    acc_test.append(round(b, 3))

def evaluate_model(model, X_test, y_test, model_name, save_dir="results"):
    """
    Evaluates a trained model and saves:
    - Confusion Matrix
    - ROC Curve (if supported)

    Parameters:
    model      : trained ML model
    X_test     : test features
    y_test     : test labels
    model_name : string name of the model
    save_dir   : directory to save figures
    """

    # Create directory if it does not exist
    os.makedirs(save_dir, exist_ok=True)

    # Predictions
    y_pred = model.predict(X_test)

    # Probabilities (if supported)
    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_test)[:, 1]
    else:
        y_prob = None

    # Metrics
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    print(f"\n===== {model_name} Evaluation =====")
    print(f"Accuracy  : {acc:.4f}")
    print(f"Precision : {prec:.4f}")
    print(f"Recall    : {rec:.4f}")
    print(f"F1 Score  : {f1:.4f}")

    # -----------------------------
    # Confusion Matrix
    # -----------------------------
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap="Blues")

    cm_path = os.path.join(
        save_dir,
        f"{model_name.replace(' ', '_')}_confusion_matrix.png"
    )
    plt.title(f"{model_name} – Confusion Matrix")
    plt.savefig(cm_path, dpi=300, bbox_inches="tight")
    plt.close()

    # -----------------------------
    # ROC Curve
    # -----------------------------
    if y_prob is not None:
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        roc_auc = auc(fpr, tpr)

        plt.figure()
        plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.4f}")
        plt.plot([0, 1], [0, 1], linestyle="--")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title(f"{model_name} – ROC Curve")
        plt.legend(loc="lower right")

        roc_path = os.path.join(
            save_dir,
            f"{model_name.replace(' ', '_')}_roc_curve.png"
        )
        plt.savefig(roc_path, dpi=300, bbox_inches="tight")
        plt.close()

# Decision Tree
from sklearn.tree import DecisionTreeClassifier
tree = DecisionTreeClassifier(max_depth=5)
tree.fit(X_train, y_train)
y_test_tree = tree.predict(X_test)
y_train_tree = tree.predict(X_train)
acc_train_tree = accuracy_score(y_train, y_train_tree)
acc_test_tree = accuracy_score(y_test, y_test_tree)
print("Decision Tree: Accuracy on training Data: {:.3f}".format(acc_train_tree))
print("Decision Tree: Accuracy on test Data: {:.3f}".format(acc_test_tree))
plt.figure(figsize=(9, 7))
n_features = X_train.shape[1]
plt.barh(range(n_features), tree.feature_importances_, align='center')
plt.yticks(np.arange(n_features), X_train.columns)
plt.xlabel("Feature importance")
plt.ylabel("Feature")
plt.show()
storeResults('Decision Tree', acc_train_tree, acc_test_tree)

# Random Forest
from sklearn.ensemble import RandomForestClassifier
forest = RandomForestClassifier(max_depth=5)
forest.fit(X_train, y_train)
y_test_forest = forest.predict(X_test)
y_train_forest = forest.predict(X_train)
acc_train_forest = accuracy_score(y_train, y_train_forest)
acc_test_forest = accuracy_score(y_test, y_test_forest)
print("Random forest: Accuracy on training Data: {:.3f}".format(acc_train_forest))
print("Random forest: Accuracy on test Data: {:.3f}".format(acc_test_forest))
plt.figure(figsize=(9, 7))
plt.barh(range(n_features), forest.feature_importances_, align='center')
plt.yticks(np.arange(n_features), X_train.columns)
plt.xlabel("Feature importance")
plt.ylabel("Feature")
plt.show()
storeResults('Random Forest', acc_train_forest, acc_test_forest)

# Multilayer Perceptrons
from sklearn.neural_network import MLPClassifier
mlp = MLPClassifier(alpha=0.001, hidden_layer_sizes=([100, 100, 100]))
mlp.fit(X_train, y_train)
y_test_mlp = mlp.predict(X_test)
y_train_mlp = mlp.predict(X_train)
acc_train_mlp = accuracy_score(y_train, y_train_mlp)
acc_test_mlp = accuracy_score(y_test, y_test_mlp)
print("Multilayer Perceptrons: Accuracy on training Data: {:.3f}".format(acc_train_mlp))
print("Multilayer Perceptrons: Accuracy on test Data: {:.3f}".format(acc_test_mlp))
storeResults('Multilayer Perceptrons', acc_train_mlp, acc_test_mlp)

# XGBoost
from xgboost import XGBClassifier
xgb = XGBClassifier(learning_rate=0.4, max_depth=7)
xgb.fit(X_train, y_train)
y_test_xgb = xgb.predict(X_test)
y_train_xgb = xgb.predict(X_train)
acc_train_xgb = accuracy_score(y_train, y_train_xgb)
acc_test_xgb = accuracy_score(y_test, y_test_xgb)
print("XGBoost: Accuracy on training Data: {:.3f}".format(acc_train_xgb))
print("XGBoost : Accuracy on test Data: {:.3f}".format(acc_test_xgb))
storeResults('XGBoost', acc_train_xgb, acc_test_xgb)

# Autoencoder (Keras/TensorFlow version)
input_dim = X_train.shape[1]
encoding_dim = input_dim

input_layer = Input(shape=(input_dim, ))
encoder = Dense(encoding_dim, activation="relu", activity_regularizer=regularizers.l1(10e-4))(input_layer)
encoder = Dense(int(encoding_dim), activation="relu")(encoder)
encoder = Dense(int(encoding_dim - 2), activation="relu")(encoder)
code = Dense(int(encoding_dim - 4), activation='relu')(encoder)
decoder = Dense(int(encoding_dim - 2), activation='relu')(code)
decoder = Dense(int(encoding_dim), activation='relu')(decoder)
decoder = Dense(input_dim, activation='relu')(decoder)
autoencoder = Model(inputs=input_layer, outputs=decoder)
autoencoder.summary()

autoencoder.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
history = autoencoder.fit(X_train, X_train, epochs=10, batch_size=64, shuffle=True, validation_split=0.2)

acc_train_auto = autoencoder.evaluate(X_train, X_train)[1]
acc_test_auto = autoencoder.evaluate(X_test, X_test)[1]
print('\nAutoencoder: Accuracy on training Data: {:.3f}'.format(acc_train_auto))
print('Autoencoder: Accuracy on test Data: {:.3f}'.format(acc_test_auto))
storeResults('AutoEncoder', acc_train_auto, acc_test_auto)

# Support Vector Machine
from sklearn.svm import SVC
svm = SVC(kernel='linear', C=1.0, random_state=12)
svm.fit(X_train, y_train)
y_test_svm = svm.predict(X_test)
y_train_svm = svm.predict(X_train)
acc_train_svm = accuracy_score(y_train, y_train_svm)
acc_test_svm = accuracy_score(y_test, y_test_svm)
print("SVM: Accuracy on training Data: {:.3f}".format(acc_train_svm))
print("SVM : Accuracy on test Data: {:.3f}".format(acc_test_svm))
storeResults('SVM', acc_train_svm, acc_test_svm)


evaluate_model(tree, X_test, y_test, "Decision Tree")
evaluate_model(forest, X_test, y_test, "Random Forest")
evaluate_model(mlp, X_test, y_test, "Multilayer Perceptron")
evaluate_model(xgb, X_test, y_test, "XGBoost")
evaluate_model(svm, X_test, y_test, "Support Vector Machine")

# Comparison of Models
results = pd.DataFrame({
    'ML Model': ML_Model,
    'Train Accuracy': acc_train,
    'Test Accuracy': acc_test
})
results.sort_values(by=['Test Accuracy', 'Train Accuracy'], ascending=False)

# Saving XGBoost model
import pickle
pickle.dump(xgb, open("XGBoostClassifier.pickle.dat", "wb"))

# Testing saved model
loaded_model = pickle.load(open("XGBoostClassifier.pickle.dat", "rb"))
loaded_model
