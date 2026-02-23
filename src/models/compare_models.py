from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
import pandas as pd
import joblib
import os


def compare_models(X_train, X_test, y_train, y_test):
    # Create directory
    os.makedirs("../../models", exist_ok=True)

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=100, random_state=42
        ),
        "SVM": SVC(probability=True, random_state=42),
        "KNN": KNeighborsClassifier(n_neighbors=5),
    }

    results = {}

    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        accuracy = accuracy_score(y_test, y_pred)

        results[name] = {"model": model, "accuracy": accuracy}

        print(f"\n{name}:")
        print(f"Accuracy: {accuracy:.4f}")
        print(classification_report(y_test, y_pred))

    # Save best model
    best_model_name = max(results, key=lambda x: results[x]["accuracy"])
    best_model = results[best_model_name]["model"]
    joblib.dump(best_model, "../../models/best_model.pkl")

    print(f"\n✓ Best model saved: {best_model_name}")
    print(f"  Accuracy: {results[best_model_name]['accuracy']:.4f}")

    return best_model


if __name__ == "__main__":
    X_train = pd.read_csv("../../data/processed/X_train.csv")
    X_test = pd.read_csv("../../data/processed/X_test.csv")
    y_train = pd.read_csv("../../data/processed/y_train.csv").values.ravel()
    y_test = pd.read_csv("../../data/processed/y_test.csv").values.ravel()

    best_model = compare_models(X_train, X_test, y_train, y_test)
