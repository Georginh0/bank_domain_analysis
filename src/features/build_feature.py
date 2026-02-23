from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import pandas as pd
import joblib
import os


def scale_and_reduce(X_train, X_test):
    # Create directory
    os.makedirs("../../models", exist_ok=True)

    # Standard scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Save scaler
    joblib.dump(scaler, "../../models/scaler.pkl")

    # PCA for dimensionality reduction
    pca = PCA(n_components=0.95)
    X_train_pca = pca.fit_transform(X_train_scaled)
    X_test_pca = pca.transform(X_test_scaled)

    # Save PCA
    joblib.dump(pca, "../../models/pca.pkl")

    print(" Feature scaling and reduction complete")
    print(f"  - Original features: {X_train.shape[1]}")
    print(f"  - Reduced features: {X_train_pca.shape[1]}")
    print(f"  - Variance explained: {sum(pca.explained_variance_ratio_):.3f}")

    return X_train_pca, X_test_pca


if __name__ == "__main__":
    X_train = pd.read_csv("../../data/processed/X_train.csv")
    X_test = pd.read_csv("../../data/processed/X_test.csv")
    X_train_pca, X_test_pca = scale_and_reduce(X_train, X_test)
