import mlflow
import pandas as pd
import joblib
import os
import shutil
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score


def train_with_mlflow():
    # Load data
    X_train = pd.read_csv("../../data/processed/X_train.csv")
    X_test = pd.read_csv("../../data/processed/X_test.csv")
    y_train = pd.read_csv("../../data/processed/y_train.csv").values.ravel()
    y_test = pd.read_csv("../../data/processed/y_test.csv").values.ravel()

    # ✅ FIX 1: Capture run object at start (critical for run ID)
    with mlflow.start_run(run_name="banking_risk_gb") as run:
        # Train model
        model = GradientBoostingClassifier(
            n_estimators=200, max_depth=5, learning_rate=0.1, random_state=42
        )
        model.fit(X_train, y_train)

        # Evaluate
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)

        # Log metrics and parameters
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_param("n_estimators", 200)
        mlflow.log_param("max_depth", 5)
        mlflow.log_param("learning_rate", 0.1)

        # Save model to project directory
        model_save_path = "../../models/production_model.pkl"
        joblib.dump(model, model_save_path)

        # ✅ FIX 2: Get run ID BEFORE ending run
        run_id = run.info.run_id

        # Create MLflow artifacts directory manually
        artifacts_dir = run.info.artifact_uri.replace("file://", "")
        model_artifact_dir = os.path.join(artifacts_dir, "model")
        os.makedirs(model_artifact_dir, exist_ok=True)

        # Copy model file using shutil (avoids distutils error)
        shutil.copy(model_save_path, os.path.join(model_artifact_dir, "model.pkl"))

        # Log model path for traceability
        mlflow.log_param("model_path", os.path.abspath(model_save_path))

        # ✅ FIX 3: Print run ID INSIDE context manager (before run ends)
        print(f"\n{'=' * 60}")
        print(f"✓ MODEL TRAINING COMPLETE")
        print(f"✓ Test Accuracy: {accuracy:.4f}")
        print(f"✓ MLflow Run ID: {run_id}")
        print(f"✓ Model saved to: {os.path.abspath(model_save_path)}")
        print(f"✓ Model artifact copied to MLflow: {model_artifact_dir}/model.pkl")
        print(f"{'=' * 60}\n")

    # ✅ NO mlflow.active_run() AFTER end_run() - avoids AttributeError
    return run_id, accuracy


if __name__ == "__main__":
    run_id, accuracy = train_with_mlflow()
    # Optional: Access run_id safely after function completes
    print(f"Completed run: {run_id} | Final accuracy: {accuracy:.4f}")
