import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


def _clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.loc[:, ~df.columns.astype(str).str.match(r"^Unnamed")]

    cols = (
        df.columns.astype(str)
        .str.strip()
        .str.replace(r"\s+", "_", regex=True)
        .str.replace(r"[^0-9A-Za-z_]+", "", regex=True)
    )

    cols = [c if c not in ("", "_") else "DROP_ME" for c in cols]
    df.columns = cols
    df = df.drop(columns=["DROP_ME"], errors="ignore")

    return df


def load_and_preprocess():
    ROOT = Path(__file__).resolve().parents[2]
    RAW = ROOT / "data" / "raw" / "Banking.csv"
    PROCESSED = ROOT / "data" / "processed"
    MODELS = ROOT / "models"

    PROCESSED.mkdir(parents=True, exist_ok=True)
    MODELS.mkdir(parents=True, exist_ok=True)

    # Load data
    df = pd.read_csv(RAW)
    df = _clean_columns(df)

    cols_to_drop = [
        "id",
        "Name",
        "Location_ID",
        "Joined_Bank",
        "Banking_Contact",
        "BRId",
        "GenderId",
        "IAId",
        "Risk_Weighting",
    ]
    df = df.drop(columns=cols_to_drop, errors="ignore")

    # Drop rows with missing values
    df = df.dropna()

    # Outlier treatment (IQR capping)
    numerical_cols = [
        "Age",
        "Estimated_Income",
        "Superannuation_Savings",
        "Amount_of_Credit_Cards",
        "Credit_Card_Balance",
        "Bank_Loans",
        "Bank_Deposits",
        "Checking_Accounts",
        "Saving_Accounts",
        "Foreign_Currency_Account",
        "Business_Lending",
        "Properties_Owned",
    ]

    print("Applying outlier treatment (IQR capping)...")
    for col in numerical_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            Q1, Q3 = df[col].quantile([0.25, 0.75])
            IQR = Q3 - Q1
            lower, upper = Q1 - 3 * IQR, Q3 + 3 * IQR
            df[col] = df[col].clip(lower=lower, upper=upper)

    # Target column
    target = "Fee_Structure"
    if target not in df.columns:
        raise ValueError(
            f"Target column '{target}' not found. Columns: {list(df.columns)}"
        )

    print(f"\nTarget variable distribution ({target}):")
    print(df[target].value_counts())

    # Remove rare classes
    class_counts = df[target].value_counts()
    valid_classes = class_counts[class_counts >= 2].index
    df = df[df[target].isin(valid_classes)]

    print(f"\nAfter removing rare classes: {len(df)} samples remaining")
    print(df[target].value_counts())

    # Class weights (saved for training)
    total = len(df)
    n_classes = df[target].nunique()
    class_weights = {
        cls: total / (n_classes * cnt) for cls, cnt in df[target].value_counts().items()
    }
    joblib.dump(class_weights, MODELS / "class_weights.pkl")

    print("\nClass weights:")
    for cls, w in class_weights.items():
        print(f"  {cls}: {w:.2f}x")

    # Encode target
    le_target = LabelEncoder()
    y = le_target.fit_transform(df[target])
    joblib.dump(le_target, MODELS / "le_target.pkl")

    # Encode categorical features
    categorical_cols = ["Nationality", "Occupation", "Loyalty_Classification"]
    for col in categorical_cols:
        if col in df.columns:
            le = LabelEncoder()
            df[col] = df[col].astype(str)
            df[f"{col}_encoded"] = le.fit_transform(df[col])
            joblib.dump(le, MODELS / f"le_{col}.pkl")
            df = df.drop(columns=[col])

    # Features
    X = df.drop(columns=[target])

    # Drop any non-feature junk columns that might still sneak in
    X = X.drop(columns=["_"], errors="ignore")

    # Ensure all remaining columns numeric
    non_numeric = X.select_dtypes(exclude=[np.number]).columns.tolist()
    if non_numeric:
        # Try coercing them to numeric (sometimes numbers stored as strings)
        for c in non_numeric:
            X[c] = pd.to_numeric(X[c], errors="coerce")

        non_numeric2 = X.select_dtypes(exclude=[np.number]).columns.tolist()
        if non_numeric2:
            raise ValueError(
                f"Non-numeric columns found: {non_numeric2}. Fix encoding/dropping."
            )

    # Final NA cleanup after coercion
    X = X.dropna()
    y = y[X.index]

    print(f"\nFeatures used ({X.shape[1]}): {list(X.columns)}")

    # Split (stratified)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print("✓ Stratified split successful")

    # Save processed
    X_train.to_csv(PROCESSED / "X_train.csv", index=False)
    X_test.to_csv(PROCESSED / "X_test.csv", index=False)
    pd.DataFrame(y_train, columns=["target"]).to_csv(
        PROCESSED / "y_train.csv", index=False
    )
    pd.DataFrame(y_test, columns=["target"]).to_csv(
        PROCESSED / "y_test.csv", index=False
    )

    print("\n✓ Data preprocessing complete")
    print(f"  - Training samples: {len(X_train)}")
    print(f"  - Test samples: {len(X_test)}")
    print(f"  - Features: {X_train.shape[1]}")
    print(f"  - Classes: {len(np.unique(y))}")

    return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    load_and_preprocess()
