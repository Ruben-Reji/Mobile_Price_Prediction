from flask import Flask, render_template, request, jsonify
import joblib
import pandas as pd
import os

app = Flask(__name__)

# ✅ Load trained model and feature order
model_data = joblib.load(r"C:\Users\rejir\Downloads\mobile-price-prediction\models\mobile_price_model.joblib")
model = model_data["model"]
feature_order = model_data["features"]

# Homepage
@app.route("/")
def index():
    return render_template("index.html")

# API endpoint for prediction
@app.route("/api/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()

        # Create DataFrame from input
        df = pd.DataFrame([data])

        # ✅ Ensure column order matches training
        df = df.reindex(columns=feature_order, fill_value=0)

        # Make prediction
        prediction = model.predict(df)[0]

        return jsonify({
            "class": int(prediction),
            "label": price_label(prediction)
        })
    except Exception as e:
        return jsonify({"error": str(e)})

# Helper function to map class → label
def price_label(pred_class):
    mapping = {
        0: "Low",
        1: "Medium",
        2: "High",
        3: "Very High"
    }
    return mapping.get(pred_class, "Unknown")

if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)
