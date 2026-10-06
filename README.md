# Crop Yield Prediction for Punjab, Pakistan

**A tree-ensemble and leaf-health prototype, and what random validation hides**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![DOI](https://img.shields.io/badge/DOI-to%20be%20added-lightgrey.svg)](#citation)
[![Live demo](https://img.shields.io/badge/Live%20demo-Hugging%20Face-orange.svg)](https://huggingface.co/spaces/AminahAsif/Crop-yield-punjab)

Machine-learning prototype for predicting wheat and maize yield in Punjab, Pakistan (Random Forest, XGBoost, SVR and a stacked ensemble, with a MobileNetV2 leaf-health classifier and SHAP explanations), **together with an honest audit of how well such models really forecast**.

> **Please read this first.** The scores reported for the original project (for example R² = 0.99 in [`Version_1`](Version_1/)) come from a random split of a table that turned out to contain only **46 independent observations**, because every crop-year was copied nine times by a data join. They overstate real forecasting skill. The corrected evaluation, the replications and the preprint are in [`Extended Version`](Extended%20Version/).

---

## Contents

1. [Repository layout](#repository-layout)
2. [What the project does](#what-the-project-does)
3. [Main findings of the extended evaluation](#main-findings-of-the-extended-evaluation)
4. [Version_1: the original project](#version_1-the-original-project)
5. [Extended Version: audit, replication and preprint](#extended-version-audit-replication-and-preprint)
6. [Data sources and licences](#data-sources-and-licences)
7. [Limitations](#limitations)
8. [Authors and credits](#authors-and-credits)
9. [Citation](#citation)
10. [Licence](#licence)
11. [Disclaimer](#disclaimer)

---

## Repository layout

```
Crop_Yield_Prediction-_Punjab/
├── README.md             this file
├── LICENSE               MIT licence (code)
├── Version_1/            the original project: data, notebook, Gradio app, trained models, figures, course report
└── Extended Version/     everything added afterwards: audit, replications, scripts, results, figures, notebook, preprint files
```

Both folders are flat: files are grouped by type rather than in sub-folders.

| Folder | Main contents |
|--------|---------------|
| [`Version_1`](Version_1/) | `app_cv.py` (Gradio app), `Crop_yield_Prediction_with_CV_module.ipynb` (full original notebook), `CropYield_Report_with_CV_Module.docx` (course report), `final_ml_dataset_cv.csv` and a `data/` folder, result figures (`*.png`: model comparison, ablation, SHAP, confusion matrix, architecture, ...), trained models |
| [`Extended Version`](Extended%20Version/) | analysis scripts (`experiments.py`, `extra.py`, `detrend.py`, `detrend2.py`, `district.py`, ...), result tables (`*.csv`), paper figures (`*.pdf`, `*.png`), the Colab notebook `crop_yield_extension_colab.ipynb`, LaTeX figure files (`*.tex`), the deduplicated table `crop_year_dataset.csv`, and a `data/` folder |

**Live demo:** [Hugging Face Space](https://huggingface.co/spaces/AminahAsif/Crop-yield-punjab)

---

## What the project does

- **Tabular yield models.** Random Forest, XGBoost and support-vector regression predict yield (Ton/Ha) for wheat and maize from pesticide use, temperature and soil variables. A Ridge meta-learner stacks the three.
- **Leaf-health module.** A MobileNetV2 classifier (ImageNet weights, frozen backbone, small dense head) labels a maize-leaf photograph as healthy or diseased. Its probability is passed to the yield models as a `health_score` feature.
- **Explanations and app.** SHAP waterfall plots explain the XGBoost prediction, and a Gradio web app puts everything behind sliders and an optional image upload.
- **Audit and replication (Extended Version).** The same question is re-asked with validation that respects time (leave-one-year-out, forward chaining), on a longer FAOSTAT table, on a district-level wheat panel, and with an unseen-image test of the classifier.

---

## Main findings of the extended evaluation

| # | Finding | Key evidence |
|---|---------|--------------|
| 1 | The 414-row national table holds only **46 independent observations** | 2 crops x 23 years x 9 copies; a temperature join copied every crop-year nine times |
| 2 | Random splits inflate the scores | XGBoost R² 0.994 (random 5-fold) vs **-0.204** (leave-one-year-out) on the same rows |
| 3 | A per-crop **linear trend** beats every model that is not given the year | Leave-one-year-out RMSE 0.243 vs 0.59-0.74 (46 rows); 0.292 vs 0.84-1.06 on the FAOSTAT table (70 rows) |
| 4 | Giving the models the year helps interpolation, not forecasting | RF + year: leave-one-year-out 0.141 / 0.202; forward chaining 0.333 / 0.336 vs 0.317 / 0.339 for the trend (ties) |
| 5 | The **pesticide series is imputed** in both data releases; the gain it seemed to give vanishes on the longer table; national temperature change adds nothing | Kaggle table: exact straight line from 2001; FAOSTAT: 4 distinct values, 27 of 35 flagged as imputed |
| 6 | District-level wheat panel (36 districts x 13 seasons): most variation is spatial | District identity explains 85.4% of the variance, season 6.7%; district mean + common trend matches the learned models |
| 7 | The leaf classifier is accurate on its benchmark and transfers reasonably to unseen diseased leaves, but the yield models do not use it | 99.87% on PlantVillage validation; **324 / 336 (96.4%)** recall on unseen images after removing near-duplicates; the ensemble moves only 0.004 Ton/Ha across the whole score range |
| 8 | The stacking ensemble is effectively XGBoost | Meta-learner weight 0.967 on XGBoost |

**What we do not claim:** that weather has no effect on yield (our weather data are a national annual anomaly), that learned models can never beat a trend, that images cannot help predict yield (no paired data exist), or anything about the classifier's **false-alarm rate** on healthy field leaves (our unseen images are all diseased).

---

## Version_1: the original project

### Data

Three public Kaggle datasets were merged (crop yield 1990-2013 for Pakistan, district-level soil and climate data for Punjab, and Punjab wheat statistics for plots), plus the PlantVillage maize images for the classifier. The merged table (`final_ml_dataset_cv.csv`) has **414 rows** (207 wheat, 207 maize), the nine tabular features and `health_score`.

### Models

| Model | Settings |
|-------|----------|
| Random Forest | 200 trees, max depth 8 (10 in the 13-feature baseline) |
| XGBoost | 200 trees, learning rate 0.05, max depth 5 (6 in the baseline) |
| SVR | RBF kernel, C = 100, gamma = 0.1, epsilon = 0.1, standardised inputs |
| Stacking | RF, XGBoost, SVR -> Ridge (alpha = 1), 5-fold out-of-fold predictions |
| Leaf classifier | MobileNetV2 (frozen) + GlobalAveragePooling + Dense 128 (ReLU) + Dropout 0.3 + Dense 1 (sigmoid); 164,097 of 2,422,081 parameters trainable |

All seeds are fixed at 42. The baseline tabular models were also cross-checked in Orange Data Mining (5-fold CV).

### Original results (random 80/20 split; see the warning at the top)

| Model | MAE | RMSE | Test R² |
|-------|-----|------|---------|
| Random Forest | 0.0715 | 0.1309 | 0.9735 |
| XGBoost | 0.0509 | 0.0827 | 0.9894 |
| SVR | 0.4639 | 0.7086 | 0.2240 |
| Stacking ensemble | 0.0462 | 0.0780 | 0.9906 |

Leaf classifier: 99.87% accuracy on 772 held-out PlantVillage images (771 correct).

### The web app

`app_cv.py` is a Gradio app: sliders for pesticide use (0-30,000 t), temperature (15-35 C), N, P, K, soil temperature, humidity and pH, an optional leaf-image upload, and as output the four model predictions, a comparison chart, the leaf label and a SHAP waterfall plot. Run it from inside `Version_1/`:

```bash
git clone https://github.com/AminahAsif/Crop_Yield_Prediction-_Punjab.git
cd Crop_Yield_Prediction-_Punjab/Version_1
pip install gradio numpy pandas scikit-learn xgboost shap matplotlib joblib tensorflow Pillow
python app_cv.py
```

The app loads the trained models (`*_cv.pkl` and `leaf_classifier.h5`) from the location set at the top of `app_cv.py`; check that path matches where you keep the files. Note that the confidence shown for a *diseased* leaf should be the probability of the diseased class (1 - P(healthy)); the first release showed P(healthy) for every leaf, and the Hugging Face Space may still run that earlier file.

---

## Extended Version: audit, replication and preprint

### Contents

| Files | What they are |
|-------|---------------|
| `experiments.py`, `extra.py` | the three validation protocols on the deduplicated 46-row table; the 414-row random-vs-leave-one-year-out check |
| `detrend.py`, `detrend2.py` | trend + residual models and the feature ablation |
| `district.py` | district-level wheat panel analysis |
| `make_figs.py`, `make_svg.py`, `tables.py` | figures and LaTeX tables for the paper |
| `crop_year_dataset.csv` | the deduplicated 46-row crop-year table |
| `protocol_results.csv`, `extra414.csv`, `detrend_results.csv`, `detrend_ablation.csv`, `district_results.csv` | raw result tables behind the paper |
| `crop_yield_extension_colab.ipynb` | Colab notebook: downloads FAOSTAT, rebuilds the 1990-2024 table, re-runs the evaluation, and runs the unseen-image test with near-duplicate removal |
| `*.pdf`, `*.png`, `*.tex` (figures) | the figures of the preprint and the small LaTeX files that include them |
| `data/` | data used by the Extended Version scripts |

The scripts were written with absolute paths from the authoring machine; edit the paths at the top of each script before re-running. Seeds are fixed at 42.

### How to reproduce

1. **Experiments on the Kaggle-derived table.** Run `experiments.py`, `extra.py`, `detrend.py` and `detrend2.py` after pointing their paths to `crop_year_dataset.csv` and to `Version_1/final_ml_dataset_cv.csv`.
2. **FAOSTAT replication and unseen-image test.** Open `crop_yield_extension_colab.ipynb` in Google Colab. Put `leaf_classifier.h5` (from `Version_1`) and, for the PlantVillage download, your `kaggle.json` in a Drive folder called `crop_yield_project`, then run the cells top to bottom. Part 4 (healthy leaves from other public datasets) is optional and was **not** used in the paper; its automatic download may be blocked by the data host.
3. **District-level wheat panel.** `district.py` expects the raw file `punjab_wheat_district_yield_raw.csv`. It is **not** redistributed here because the portal lists no licence: download *Area sown, production and yield per hectare of wheat by districts, Punjab* from [Open Data Pakistan](https://opendata.com.pk) (Pakistan Bureau of Statistics) and save it under that name.
4. **The paper.** The LaTeX source compiles with pdfLaTeX and BibTeX (main file `main.tex`, bibliography `references.bib`; the `.tex` figure files expect the figures in a `figures/` folder and the tables in a `tables/` folder).

---

## Data sources and licences

- **Kaggle datasets** (crop yield, district-level crop data, wheat statistics, PlantVillage, Corn or Maize Leaf Disease): each keeps its own licence; check it before reusing the data.
- **FAOSTAT** (yield, pesticide use, temperature change): downloaded by the notebook from FAO's bulk-download service; see FAO's terms of use.
- **Pakistan Bureau of Statistics district wheat yields** (via Open Data Pakistan): not redistributed; see above.

---

## Limitations

Small samples (46 and 70 national observations, 464 district-season observations), national or crop-level data, no growing-season weather, a constructed `health_score` (no leaf images are paired with yield records), an unseen-image test that contains no healthy leaves, and a single seed for the classifier. The full list is in the preprint (Limitations section).

---

## Authors and credits

**Authors of the preprint**

- **Amina Asif**, Department of Artificial Intelligence, The Islamia University of Bahawalpur, Bahawalpur, Pakistan
- **Qurat ul Ain Asif**, Key Laboratory of Xinjiang Endemic Phytomedicine Resources, Ministry of Education; School of Pharmacy, Shihezi University, Shihezi 832002, China
- **Noor Bakhat Asif**, Department of Physics, Government College Women University Faisalabad, Faisalabad, Pakistan

**Original course project (`Version_1`).** Done in the Data Mining course, Department of Artificial Intelligence, The Islamia University of Bahawalpur (session 2023 to 2027), by Amina Asif and Kasa e Fatima under the supervision of Miss Maryam Aslam.

---

## Citation

If you use this work, please cite the preprint:

> Asif, A., Asif, Q. U. A., Asif, N. B. (2026). *Crop Yield Prediction for Punjab, Pakistan: A Tree-Ensemble and Leaf-Health Prototype, and What Random Validation Hides.* Preprint. **DOI: _to be added_**

```bibtex
@misc{asif2026cropyield,
  title  = {Crop Yield Prediction for Punjab, Pakistan: A Tree-Ensemble and Leaf-Health Prototype, and What Random Validation Hides},
  author = {Asif, Amina and Asif, Qurat ul Ain and Asif, Noor Bakhat},
  year   = {2026},
  note   = {Preprint},
  doi    = {to be added},
  url    = {https://github.com/AminahAsif/Crop_Yield_Prediction-_Punjab}
}
```

<!-- When the DOI is issued:
     1. In the badge at the top, replace  DOI-to%20be%20added-lightgrey  with  DOI-<your DOI>-blue  and the link (#citation) with https://doi.org/<your DOI>.
     2. In this section, replace "to be added" (twice) with the DOI.
     An arXiv submission gets a DOI of the form 10.48550/arXiv.<number>; a GitHub release archived on Zenodo gets its own DOI. -->

---

## Licence

The **code** in this repository is released under the [MIT Licence](LICENSE). Third-party datasets keep their own licences (see [Data sources and licences](#data-sources-and-licences)).

---

## Disclaimer

This is research code and an academic prototype. It is **not validated for real agricultural planning** and must not be used to make farming, financial or policy decisions.
