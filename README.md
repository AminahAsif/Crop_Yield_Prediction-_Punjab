# Crop Yield Prediction for Punjab, Pakistan - Multi-Model Ensemble, Computer Vision Leaf Health & Explainability

## Project Objective

This project builds an end-to-end crop yield prediction system for Punjab, Pakistan. It combines three tabular regression models (Random Forest, XGBoost, SVR) in a Ridge-based Stacking Ensemble with a Computer Vision module: a MobileNetV2 transfer-learning classifier that reads a maize-leaf photograph and returns a healthy/diseased probability. The system also explains each prediction with SHAP and is deployed as an interactive Gradio web app.

The core findings: the Stacking Ensemble reaches **R² = 0.9906** (MAE 0.0462 Ton/Ha) on the final 10-feature configuration, and the leaf classifier reaches **99.87% validation accuracy** (771 / 772 images). Baseline tabular results were independently cross-checked in Orange Data Mining.

The evaluation is also reported critically. The dataset contains only 46 distinct crop-year combinations (each repeated 9 times), so random-split scores overstate real forecasting ability: holding out whole crop-years drops XGBoost to R² = -1.89. And the `health_score` feature fed to the yield models is a rescaled copy of `NPK_total` (the training data has no paired leaf images), so it adds no measurable accuracy. Both points are documented below rather than hidden.

---

## Live Demo

*(Click the image below to open the live demo on Hugging Face Spaces)*

<div align="center">
  <a href="https://huggingface.co/spaces/AminahAsif/Crop-yield-punjab">
    <img src="Results/app_cv.png" alt="Crop Yield Prediction App" width="800" style="border-radius: 8px;">
  </a>
</div>

**[Open Live Demo →](https://huggingface.co/spaces/AminahAsif/Crop-yield-punjab)**

Optionally upload a maize leaf image, set climate and soil parameters, and get: leaf health label with confidence, yield predictions from Random Forest, XGBoost, SVR and the Ensemble, a model comparison chart, and a SHAP waterfall plot.

---

## Key Results

| Metric | Result |
|--------|--------|
| Stacking Ensemble test R² (10 features) | **0.9906** |
| Stacking Ensemble MAE / RMSE | **0.0462 / 0.0780** Ton/Ha |
| Stacking Ensemble 5-fold CV R² | 0.9923 |
| XGBoost test R² | 0.9894 |
| Random Forest test R² | 0.9735 |
| SVR test R² | 0.2240 |
| Orange validation, baseline features (RF / Gradient Boosting / SVM R²) | 0.999 / 0.998 / 0.964 |
| MobileNetV2 leaf classifier accuracy | **99.87%** (771 / 772) |
| Leaf classifier macro F1 | 0.9985 |
| Effect of adding `health_score` (Phase 2 vs Phase 3) | none (differences <= 0.0005 R²) |
| XGBoost R², random split vs group-aware split | 0.9894 vs **-1.89** |
| Change in Ensemble output from leaf score 0 to 1 | 0.004 Ton/Ha |

**Honest note on the results:** the Ensemble is effectively XGBoost (Ridge meta-learner weights: XGBoost 0.967, Random Forest 0.070, SVR -0.027), which is why the two scores are so close. SVR performs poorly (R² = 0.224) once the `year` feature is removed, because the yield depends on step-like crop-year identifiers that tree models split on easily and an RBF surface with fixed settings (C = 100, gamma = 0.1, not re-tuned) does not capture. Permutation importance shows the models rely almost entirely on `pesticides_tonnes` and `N`; both are constant within a crop-year, so the high R² partly reflects memorisation of crop-year identifiers. See the [project report](docs/CropYield_Report_with_CV_Module.docx) (Sections 7.4, 7.5 and 10) for the full analysis.

---

## Model Comparison

![Model Comparison](Results/model_comparison_cv.png)

*MAE, RMSE and R² for Random Forest, XGBoost, SVR and the Stacking Ensemble on the 10-feature configuration.*

![Actual vs Predicted](Results/avp_cv.png)

*Actual vs predicted yield on the held-out test set.*

---

## System Architecture

![System Architecture](Results/arch.png)

The pipeline is structured across five phases, each building on the last.

### 1. Data

- **Sources (all Kaggle):** wheat data of Punjab (12 years), Crop Yield Prediction dataset (FAO + World Bank, filtered to Pakistan, 1,449 rows), Crop District-Level dataset for Punjab (2,200 rows, 13 crops), and the PlantVillage leaf-image dataset (maize classes only)
- **Merge:** Datasets 2 and 3 joined on crop type, giving **414 rows** (207 wheat + 207 maize), 1990 to 2013 (23 distinct years, 2003 absent), no missing values
- **Feature engineering:** `NPK_total = N + P + K`; target converted from hg/ha to Ton/Ha
- **Feature-set evolution:** Phase 1, 13 features (baseline); Phase 2, 9 features (removed `year`, constant `rainfall_mm`, its interaction term and the crop label); Phase 3, 10 features (Phase 2 plus `health_score`)
- **Scaling:** StandardScaler for SVR only; tree models are unscaled

### 2. Tabular Regression Models and Stacking

- **Random Forest:** 200 trees, `max_depth` 10 (Phase 1) and 8 (Phases 2 and 3)
- **XGBoost:** 200 estimators, learning rate 0.05, `max_depth` 6 (Phase 1) and 5 (Phases 2 and 3)
- **SVR:** RBF kernel, C = 100, gamma = 0.1, epsilon = 0.1, inside a StandardScaler pipeline
- **Stacking Ensemble:** the three models as level-1 learners, Ridge Regression (alpha = 1.0) as meta-learner, 5-fold out-of-fold predictions
- **Protocol:** 80/20 train-test split, 5-fold cross-validation on the training portion, `random_state = 42`
- **Orange Data Mining:** independent 5-fold CV of Random Forest, Gradient Boosting and SVM on the **baseline** features only (the CV module and Phases 2 and 3 were evaluated in Python, not Orange)

### 3. Computer Vision Module - Leaf Health Classification

![CV Architecture](Results/cvarch.png)

- **Data:** PlantVillage maize classes (healthy, common rust, northern leaf blight, gray leaf spot); the three disease classes merged into one "diseased" label; 3,852 images split 80/20 per class (3,080 train / 772 validation)
- **Model:** MobileNetV2 (ImageNet weights, frozen) with GlobalAveragePooling, Dense(128, ReLU), Dropout(0.3) and a sigmoid output; binary cross-entropy, Adam
- **Augmentation:** rotation 20 degrees, zoom 0.2, horizontal flip, brightness 0.8 to 1.2 (training only)
- **Training:** EarlyStopping on validation accuracy (patience 3, best weights restored); training stopped after five epochs
- **Result:** 99.87% validation accuracy; all 539 diseased leaves and 232 of 233 healthy leaves correct

<p align="center">
  <img src="Results/confusion_matrix.png" alt="Confusion Matrix" width="420">
  <img src="Results/training_curves.png" alt="Training Curves" width="560">
</p>

| Class | Precision | Recall | F1 | Support |
|-------|-----------|--------|----|---------|
| Diseased | 0.9981 | 1.0000 | 0.9991 | 539 |
| Healthy | 1.0000 | 0.9957 | 0.9979 | 233 |
| Macro average | 0.9991 | 0.9979 | 0.9985 | 772 |

- **How the output reaches the yield models:** at inference the classifier's probability is passed as `health_score` (default 0.5 when no image is uploaded). At training time there are no paired leaf images, so `health_score` was constructed by rescaling `NPK_total` between the classifier's mean healthy score (0.9921) and mean diseased score (0.0007). It therefore takes only two values (maize 0.9921, wheat 0.0007) and mirrors `NPK_total`.

![health_score vs NPK_total](Results/hs.png)

### 4. Explainability and Critical Evaluation

- **SHAP (TreeExplainer):** per-prediction waterfall plot for XGBoost in the app; global SHAP importance shows `pesticides_tonnes` and `N` dominate, and `health_score` contributes nothing
- **Ablation:** adding `health_score` changes R² by at most 0.0005 (XGBoost and Ensemble identical to four decimals)
- **Permutation importance:** `pesticides_tonnes` 1.894, `N` 0.391, `avg_temp` 0.011, all others 0.000
- **Group-aware validation:** the 414 rows are 46 crop-year groups repeated 9 times each; holding out whole groups gives XGBoost R² = -1.89
- **App sensitivity:** changing only the leaf score across its full range moves the Ensemble prediction by 0.004 Ton/Ha (XGBoost unchanged, Random Forest 0.043 Ton/Ha), so the leaf image currently has almost no influence on the predicted yield

<p align="center">
  <img src="Results/ablation.png" alt="Ablation" width="470">
  <img src="Results/split.png" alt="Random vs group split" width="320">
</p>

![SHAP Importance](Results/shap_cv_bar.png)

### 5. Deployment

- Gradio app on Hugging Face Spaces with optional leaf-image upload
- Slider inputs: pesticides (0 to 30,000 t), average temperature (15 to 35 C), N (0 to 140), P (0 to 100), K (0 to 120), soil temperature, humidity (50 to 100%), pH (4 to 9)
- Outputs: leaf-health label and confidence, four model predictions, model comparison bar chart, SHAP waterfall plot

---

## Quick Start

### Prerequisites

- Python 3.10+ with a recent TensorFlow / Keras 3 (the leaf classifier was saved with Keras 3.13)
- The trained artefacts in the `models/` folder: the five `.pkl` files in `models/tabular_cv_10_features/` and `models/leaf_classifier.h5` (see `models/README.md`)

### Installation

```bash
pip install -r requirements.txt
```

### Run the demo locally

```bash
git clone https://github.com/AminahAsif/Crop-Yield-Prediction-Punjab.git
cd Crop-Yield-Prediction-Punjab
python app_cv.py
```

### Reproduce the pipeline

Open `Crop_yield_Prediction_with_CV_module.ipynb` in Google Colab. It covers data download through the Kaggle API, preprocessing and merging, model training, stacking, SHAP, CV classifier training and export of the app files. All random seeds are fixed at 42.

Datasets:
- [Crop Yield Prediction Dataset (patelris)](https://www.kaggle.com/datasets/patelris/crop-yield-prediction-dataset)
- [PlantVillage Dataset (abdallahalidev)](https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset)
- Wheat Data of Punjab Pakistan (mhaiderali37) and Crop District Level Dataset Punjab Pakistan (talhanazir168), both on Kaggle

---

## Repository Structure

```
.
├── app_cv.py                                   # Gradio application
├── requirements.txt
├── Crop_yield_Prediction_with_CV_module.ipynb  # Full Colab notebook
├── data/
│   ├── raw/                                    # Original Kaggle files (pesticides, rainfall, temp, yield, yield_df, district crop data, wheat data)
│   └── processed/                              # merged_dataset, final_ml_dataset, final_ml_dataset_cv (414 rows), orange_final, district_data, wheat_trends
├── models/
│   ├── tabular_final_9_features/                # Phase 2 models (rf, xgb, svr, scaler, stack _final.pkl)
│   ├── tabular_cv_10_features/                  # Phase 3 models used by the app (rf, xgb, svr, scaler, stack _cv.pkl)
│   ├── leaf_classifier.h5                      # MobileNetV2 leaf classifier (11.5 MB)
│   └── README.md
├── Results/                                    # All result figures and metric tables (model_results_*.csv, comparison and SHAP plots, CV figures)
└── docs/CropYield_Report_with_CV_Module.docx   # Full project report
```

---

## Limitations

- Only two crops (wheat, maize) after the merge; the CV classifier covers maize only
- Rainfall is a single constant value (494 mm) and soil data are crop-level averages, not district or field measurements
- Training data ends in 2013
- `health_score` in the training data is a rescaled `NPK_total`, not a per-sample image measurement, and gives no measurable gain
- Random-split metrics overstate generalisation to unseen years (group-aware R² = -1.89)
- The leaf classifier was selected and evaluated on the same validation set (no separate test set), and PlantVillage may contain near-duplicate images of the same leaf, so 99.87% is best read as an upper estimate; the images are also taken under controlled conditions, unlike field photographs
- The CV module and Phases 2 and 3 were not replicated in Orange

**Future work:** leave-one-year-out validation as the headline metric; paired leaf-image and yield data so `health_score` is measured; district-level seasonal weather and NDVI; more crops and locally captured Bahawalpur leaf images; re-tuning or removing SVR; live weather API and an Urdu mobile version.

---

## Supporting Documents

- [Project Report](docs/CropYield_Report_with_CV_Module.docx) - full methodology, results, critical evaluation and references



## Disclaimer

This system is an academic prototype. It is **not validated for real agricultural planning** and should not be used to make farming, financial or policy decisions.

---
