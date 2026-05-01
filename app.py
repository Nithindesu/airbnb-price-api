from flask import Flask, request, jsonify
import pickle
import pandas as pd
import numpy as np

from predict_utils import build_features

app = Flask(__name__)
with open("models/model.pkl", "rb") as f:
    model = pickle.load(f)

with open("models/train_reference.pkl", "rb") as f:
    train_reference = pickle.load(f)

@app.route("/")
def home():
    return jsonify({
        "message": "Airbnb Price Prediction API is running"
    })

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        input_df = pd.DataFrame([data])

        # Add all missing columns expected by the feature builder
        for col in train_reference.columns:
            if col not in input_df.columns:
                input_df[col] = np.nan

        # Match training column order
        input_df = input_df[train_reference.columns]

        # Convert raw input into the same engineered feature matrix
        _, X_input, _ = build_features(train_reference, input_df)

        pred_log = model.predict(X_input)
        pred_price = np.expm1(pred_log)
        pred_price = max(0, float(pred_price[0]))

        return jsonify({
            "predicted_price": round(pred_price, 2)
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    app.run(debug=True)