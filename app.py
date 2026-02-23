#!/usr/bin/env python
"""
Banking Risk API - FIXED FOR PORT CONFLICT & DEPRECATION WARNINGS
✅ Port changed to 5001 (avoid macOS AirPlay conflict)
✅ Removed deprecated __version__ checks
✅ Added PORT environment variable support
✅ Streamlined startup diagnostics
"""

import sys
import os
from flask import Flask, request, jsonify
import pandas as pd
import joblib
import numpy as np

app = Flask(__name__)

# Load models with error handling
try:
    model = joblib.load("models/production_model.pkl")
    scaler = joblib.load("models/scaler.pkl")
    pca = joblib.load("models/pca.pkl")
    le_target = joblib.load("models/le_target.pkl")
    le_nationality = joblib.load("models/le_Nationality.pkl")
    le_occupation = joblib.load("models/le_Occupation.pkl")
    le_loyalty = joblib.load("models/le_Loyalty_Classification.pkl")
    print("✓ All models loaded successfully")
except Exception as e:
    print(f"❌ Model loading failed: {e}", file=sys.stderr)
    sys.exit(1)


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No JSON data provided"}), 400

        # Create feature dataframe
        features = pd.DataFrame(
            [
                {
                    "Age": data["age"],
                    "Estimated_Income": data["estimated_income"],
                    "Superannuation_Savings": data["superannuation_savings"],
                    "Amount_of_Credit_Cards": data["amount_of_credit_cards"],
                    "Credit_Card_Balance": data["credit_card_balance"],
                    "Bank_Loans": data["bank_loans"],
                    "Bank_Deposits": data["bank_deposits"],
                    "Checking_Accounts": data["checking_accounts"],
                    "Saving_Accounts": data["saving_accounts"],
                    "Foreign_Currency_Account": data["foreign_currency_account"],
                    "Business_Lending": data["business_lending"],
                    "Properties_Owned": data["properties_owned"],
                    "Nationality_encoded": le_nationality.transform(
                        [data["nationality"]]
                    )[0],
                    "Occupation_encoded": le_occupation.transform([data["occupation"]])[
                        0
                    ],
                    "Loyalty_Classification_encoded": le_loyalty.transform(
                        [data["loyalty_classification"]]
                    )[0],
                }
            ]
        )

        # Preprocess and predict
        features_scaled = scaler.transform(features)
        features_pca = pca.transform(features_scaled)
        prediction_encoded = model.predict(features_pca)[0]
        prediction_label = le_target.inverse_transform([prediction_encoded])[0]
        probability = model.predict_proba(features_pca)[0].max()

        return jsonify(
            {
                "risk_prediction": int(prediction_encoded),
                "risk_level": str(prediction_label),
                "confidence": float(probability),
                "message": f"Customer risk level: {prediction_label} (Confidence: {probability:.1%})",
            }
        )

    except KeyError as e:
        return jsonify({"error": f"Missing required field: {str(e)}"}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "healthy", "model_loaded": True})


@app.route("/", methods=["GET"])
def home():
    return jsonify(
        {
            "service": "Banking Risk Analysis API",
            "version": "1.0",
            "port": os.environ.get("PORT", 5001),
            "endpoints": {
                "POST /predict": "Risk prediction endpoint",
                "GET /health": "Health check",
            },
        }
    )


if __name__ == "__main__":
    # Use PORT env var or default to 5001 (avoids macOS AirPlay conflict on 5000)
    port = int(os.environ.get("PORT", 5001))

    print(f"\n{'=' * 60}")
    print(f"🚀 BANKING RISK API STARTING")
    print(f"   Model: Gradient Boosting Classifier")
    print(f"   Target: FeeStructure (High/Mid/Low)")
    print(f"   Port: {port} (changed from 5000 to avoid conflict)")
    print(f"   Health check: http://localhost:{port}/health")
    print(f"   API docs: http://localhost:{port}/")
    print(f"{'=' * 60}\n")

    app.run(debug=False, host="0.0.0.0", port=port)
