
import gradio as gr
import numpy as np
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image

BASE = os.path.dirname(os.path.abspath(__file__))

# Load ML models
rf    = joblib.load(os.path.join(BASE, "rf_cv.pkl"))
xgb   = joblib.load(os.path.join(BASE, "xgb_cv.pkl"))
svr   = joblib.load(os.path.join(BASE, "svr_cv.pkl"))
scaler= joblib.load(os.path.join(BASE, "scaler_cv.pkl"))
stack = joblib.load(os.path.join(BASE, "stack_cv.pkl"))

# Load CV model
cv_model = load_model(os.path.join(BASE, "leaf_classifier.h5"))

FEATURES = ['pesticides_tonnes', 'avg_temp', 'N', 'P', 'K',
            'temperature', 'humidity', 'ph', 'NPK_total', 'health_score']

def classify_leaf(image):
    if image is None:
        return 0.5, "No image — using default health score 0.5"
    img = Image.fromarray(image).resize((224, 224))
    arr = img_to_array(img) / 255.0
    arr = np.expand_dims(arr, axis=0)
    score = float(cv_model.predict(arr, verbose=0)[0][0])
    label = "Healthy 🟢" if score > 0.5 else "Diseased 🔴"
    return round(score, 4), f"{label} (confidence: {score:.2%})"

def predict_yield(image, pesticides, avg_temp, N, P, K,
                  temperature, humidity, ph):

    health_score, health_label = classify_leaf(image)
    NPK_total = N + P + K

    input_data = np.array([[pesticides, avg_temp, N, P, K,
                            temperature, humidity, ph,
                            NPK_total, health_score]])
    input_df = pd.DataFrame(input_data, columns=FEATURES)
    input_sc = scaler.transform(input_df)

    rf_pred    = round(float(rf.predict(input_df)[0]), 3)
    xgb_pred   = round(float(xgb.predict(input_df)[0]), 3)
    svr_pred   = round(float(svr.predict(input_sc)[0]), 3)
    stack_pred = round(float(stack.predict(input_df)[0]), 3)

    # Bar chart
    fig, ax = plt.subplots(figsize=(8, 3))
    models_list = ["Random Forest", "XGBoost", "SVR", "Ensemble"]
    preds_list  = [rf_pred, xgb_pred, svr_pred, stack_pred]
    colors      = ["#2e8b57", "#e07b39", "#5b7fa6", "#8b2e8b"]
    bars = ax.barh(models_list, preds_list, color=colors, edgecolor="white")
    ax.set_xlabel("Predicted Yield (Ton/Ha)")
    ax.set_title("Crop Yield Predictions — Punjab, Pakistan")
    for bar, val in zip(bars, preds_list):
        ax.text(val + 0.005, bar.get_y() + bar.get_height()/2,
                str(val), va="center", fontweight="bold")
    plt.tight_layout()

    # SHAP
    explainer = shap.TreeExplainer(xgb)
    shap_vals = explainer.shap_values(input_df)
    fig2, ax2 = plt.subplots(figsize=(8, 5))
    shap.waterfall_plot(
        shap.Explanation(
            values=shap_vals[0],
            base_values=explainer.expected_value,
            data=input_df.iloc[0],
            feature_names=FEATURES),
        show=False)
    plt.tight_layout()

    result = (
        "🌿 Leaf Health    : " + health_label + chr(10) +
        "🌲 Random Forest  : " + str(rf_pred)    + " Ton/Ha" + chr(10) +
        "⚡ XGBoost        : " + str(xgb_pred)   + " Ton/Ha" + chr(10) +
        "🔵 SVR            : " + str(svr_pred)   + " Ton/Ha" + chr(10) +
        "🏆 Ensemble       : " + str(stack_pred) + " Ton/Ha"
    )
    return result, fig, fig2


with gr.Blocks(title="Crop Yield Predictor — Punjab Pakistan") as demo:

    gr.Markdown("# 🌾 Crop Yield Prediction — Punjab, Pakistan")
    gr.Markdown("### Multi-Model Ensemble + Computer Vision Leaf Health Analysis")
    gr.Markdown("---")

    with gr.Row():
        with gr.Column():
            gr.Markdown("### 📷 Leaf Health (Computer Vision)")
            image_input = gr.Image(label="Upload Crop Leaf Image (optional)",
                                   type="numpy")
            gr.Markdown("*Upload a maize leaf image — CV model detects healthy/diseased*")

        with gr.Column():
            gr.Markdown("### 🌡️ Climate Parameters")
            pesticides  = gr.Slider(0, 30000, value=6000,
                                    label="Pesticides (tonnes)")
            avg_temp    = gr.Slider(15.0, 35.0, value=24.0,
                                    label="Average Temperature (C)")

        with gr.Column():
            gr.Markdown("### 🌱 Soil Parameters")
            N           = gr.Slider(0, 140, value=56,  label="Nitrogen (N)")
            P           = gr.Slider(0, 100, value=42,  label="Phosphorus (P)")
            K           = gr.Slider(0, 120, value=61,  label="Potassium (K)")
            temperature = gr.Slider(15.0, 35.0, value=24.0,
                                    label="Soil Temperature (C)")
            humidity    = gr.Slider(50.0, 100.0, value=86.0,
                                    label="Humidity (%)")
            ph          = gr.Slider(4.0, 9.0, value=6.4, label="Soil pH")

    predict_btn = gr.Button("🔍 Predict Yield", variant="primary")

    gr.Markdown("### 📊 Results")
    with gr.Row():
        result_text = gr.Textbox(label="Predictions", lines=6)
    with gr.Row():
        chart1 = gr.Plot(label="Model Comparison")
        chart2 = gr.Plot(label="SHAP Feature Importance")

    predict_btn.click(
        fn=predict_yield,
        inputs=[image_input, pesticides, avg_temp,
                N, P, K, temperature, humidity, ph],
        outputs=[result_text, chart1, chart2]
    )

    gr.Markdown("---")
    gr.Markdown(
        "**Region:** Bahawalpur, Punjab Pakistan | "
        "**CV Model:** MobileNetV2 (99.87% accuracy) | "
        "**Ensemble R²:** 0.9906"
    )

demo.launch()
