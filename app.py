from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import joblib
from tensorflow.keras.models import load_model

# Initialize Flask app
app = Flask(__name__, template_folder='templates')

# =====================================
# Load your trained models and scalers
# =====================================
print("🔄 Loading model and scalers...")

feature_scaler = joblib.load("feature_scaler.pkl")
target_scaler = joblib.load("target_scaler.pkl")
feature_cols = joblib.load("feature_cols.pkl")
lookback = joblib.load("lookback.pkl")

lstm_model = load_model("lstm_model.h5", compile=False)
gan_generator = load_model("gan_generator.h5", compile=False)

processed = pd.read_csv("processed_data.csv")
processed["Datetime"] = pd.to_datetime(processed["Datetime"])

print("✅ Models and data loaded successfully!")

# =====================================
# ROUTES
# =====================================

@app.route("/")
def home():
    """Serve the index.html file."""
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    """Predict power consumption for given datetime."""
    data = request.get_json()

    if not data or "datetime" not in data:
        return jsonify(success=False, message="Missing datetime"), 400

    try:
        req_dt = pd.to_datetime(data["datetime"])
    except Exception as e:
        return jsonify(success=False, message=f"Invalid datetime format: {e}"), 400

    # Ensure enough data for sequence
    if len(processed) < lookback:
        return jsonify(success=False, message="Not enough historical data"), 400

    # Take last (lookback - 1) records
    history = processed.iloc[-(lookback - 1):].copy()

    # Generate new row dynamically
    new_row = {}
    new_row["Datetime"] = req_dt
    new_row["hour"] = req_dt.hour
    new_row["day"] = req_dt.day
    new_row["month"] = req_dt.month
    new_row["dayofweek"] = req_dt.dayofweek
    new_row["quarter"] = req_dt.quarter
    new_row["is_weekend"] = 1 if req_dt.dayofweek >= 5 else 0
    new_row["hour_sin"] = np.sin(2 * np.pi * new_row["hour"] / 24)
    new_row["hour_cos"] = np.cos(2 * np.pi * new_row["hour"] / 24)
    new_row["month_sin"] = np.sin(2 * np.pi * new_row["month"] / 12)
    new_row["month_cos"] = np.cos(2 * np.pi * new_row["month"] / 12)
    new_row["day_sin"] = np.sin(2 * np.pi * new_row["dayofweek"] / 7)
    new_row["day_cos"] = np.cos(2 * np.pi * new_row["dayofweek"] / 7)

    last_row = processed.iloc[-1]
    new_row["power_roll_mean_24"] = last_row["power_roll_mean_24"]
    new_row["power_roll_std_24"] = last_row["power_roll_std_24"]
    new_row["power_roll_max_24"] = last_row["power_roll_max_24"]
    new_row["power_roll_min_24"] = last_row["power_roll_min_24"]
    new_row["power_lag_1"] = last_row["PowerConsumption_Zone1"]
    new_row["power_lag_24"] = last_row["power_lag_24"]
    new_row["power_lag_168"] = last_row["power_lag_168"]
    new_row["power_diff_1"] = 0
    new_row["power_diff_24"] = 0

    new_row_df = pd.DataFrame([new_row])
    full_seq = pd.concat([history, new_row_df], ignore_index=True)

    features = full_seq[feature_cols].values
    scaled_features = feature_scaler.transform(features)
    X = scaled_features.reshape(1, scaled_features.shape[0], scaled_features.shape[1])

    # LSTM + GAN prediction
    y_pred_scaled = lstm_model.predict(X, verbose=0).flatten()
    lstm_features = X[0, -1, :]
    gan_input = np.concatenate([lstm_features, y_pred_scaled]).reshape(1, -1)
    gan_corr = gan_generator.predict(gan_input, verbose=0).flatten()
    refined_scaled = y_pred_scaled * 0.7 + gan_corr * 0.3

    prediction_actual = target_scaler.inverse_transform(refined_scaled.reshape(-1, 1)).flatten()[0]

    return jsonify(success=True, prediction=float(prediction_actual))

if __name__ == "__main__":
    app.run(debug=True)
