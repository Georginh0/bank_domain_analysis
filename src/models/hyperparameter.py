from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import GradientBoostingClassifier
import pandas as pd
import joblib


def optimize_model(X_train, y_train):
    # Define parameter grid
    param_grid = {
        "n_estimators": [100, 200],
        "max_depth": [3, 5, 7],
        "learning_rate": [0.01, 0.1],
        "subsample": [0.8, 1.0],
    }

    # Initialize model
    gbc = GradientBoostingClassifier(random_state=42)

    # Grid search
    grid_search = GridSearchCV(
        estimator=gbc, param_grid=param_grid, cv=5, scoring="accuracy", n_jobs=-1
    )

    grid_search.fit(X_train, y_train)

    print("Best Parameters:", grid_search.best_params_)
    print("Best Score:", grid_search.best_score_)

    # Save optimized model
    joblib.dump(grid_search.best_estimator_, "../../models/optimized_model.pkl")

    return grid_search.best_estimator_


# Usage
X_train = pd.read_csv("../../data/processed/X_train.csv")
y_train = pd.read_csv("../../data/processed/y_train.csv").values.ravel()

optimized_model = optimize_model(X_train, y_train)
