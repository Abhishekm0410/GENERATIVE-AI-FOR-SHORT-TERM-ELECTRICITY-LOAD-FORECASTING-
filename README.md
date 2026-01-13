⚡ Generative AI for Short-Term Electricity Load Forecasting

This project implements a short-term electricity load forecasting system using deep learning techniques. It combines a Long Short-Term Memory (LSTM) network with a Generative Adversarial Network (GAN) to improve forecasting accuracy by learning complex temporal patterns and enhancing data representation.

The project covers the complete pipeline from data preprocessing and model training to evaluation, visualization, and deployment through a web application.

📌 Project Overview

Accurate short-term electricity load forecasting is essential for efficient power grid operation, demand–supply planning, and energy optimization. Traditional statistical models often fail to capture non-linear temporal dependencies present in electricity consumption data.

This project addresses these challenges by using:

🧠 LSTM networks for sequential time-series learning

🔁 GAN-generated synthetic sequences to improve generalization

🛠️ An end-to-end machine learning pipeline with deployment support

✨ Key Features

📈 Short-term electricity load forecasting using LSTM

🧪 GAN-assisted data augmentation for improved robustness

⚖️ Feature and target normalization using saved scalers

📊 Multiple evaluation metrics for model assessment

🖼️ Visualization of trends, errors, and forecast comparisons

🌐 Flask-based web application for real-time predictions

🗂️ Repository Structure
├── app.py
├── train.py
├── plot_results.py
├── power_data.csv
├── processed_data.csv
│
├── lstm_model.h5
├── gan_generator.h5
│
├── scaler.pkl
├── target_scaler.pkl
├── feature_scaler.pkl
├── feature_cols.pkl
├── lookback.pkl
│
├── metrics.csv
│
├── training_results.png
├── error_analysis.png
├── metrics_summary.png
├── trend_plot.png
├── weekly_patterns.png
├── 6month_forecast.png
├── 6_months_comparison.png
│
├── requirements.txt
└── README.md

🧰 Technologies Used

🐍 Python

🤖 TensorFlow / Keras

📐 NumPy

📊 Pandas

📉 Scikit-learn

📈 Matplotlib

🌍 Flask

🔬 Methodology
🧹 Data Preprocessing

Cleaning and formatting raw electricity load data

Handling missing values

Feature engineering

Normalization using scalers

Time-series window creation using a lookback period

🏗️ Model Training

LSTM model trained for short-term load forecasting

GAN trained to generate realistic synthetic load sequences

Synthetic data improves temporal pattern learning

Models and preprocessing artifacts saved for reuse

📊 Evaluation

Performance measured using standard regression metrics

Error distribution and residual analysis

Comparison of actual vs predicted load

Visualization for interpretability

🚀 Deployment

Flask-based web application for inference

Reuse of trained models and scalers

Real-time electricity load prediction

⚙️ Installation

Install dependencies using:

pip install -r requirements.txt

🏃 Training the Model

Run the training pipeline:

python train.py


This script preprocesses data, trains the LSTM and GAN models, and saves all artifacts.

🌐 Running the Application

Start the Flask app:

python app.py


The application runs locally and provides real-time load forecasting.

📈 Results

Accurate short-term electricity load predictions

Improved generalization through GAN-generated sequences

Clear visual insights through trend and error plots

Stable performance across multiple forecast horizons

💡 Use Cases

Power grid load management

Energy demand forecasting

Smart grid analytics

Time-series forecasting research

👤 Author

Abhishek Maheshwari
