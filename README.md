# 🏠 House Price Prediction — King County Dataset

> **Supervised Machine Learning · Linear Regression · Python / Scikit-learn**

A complete, production-quality machine learning pipeline that predicts house
prices in King County, Washington State (USA) using real sales data.
The project demonstrates the full ML workflow — from raw data to evaluated,
explainable models — while rigorously addressing common pitfalls such as
**data leakage**, **feature scaling imbalance**, and **missing baselines**.

---

## 📋 Table of Contents

1. [Project Overview](#-project-overview)
2. [Dataset](#-dataset)
3. [Results at a Glance](#-results-at-a-glance)
4. [Project Structure](#-project-structure)
5. [Methodology](#-methodology)
   - [Why Log-Transform the Target?](#why-log-transform-the-target)
   - [Data Leakage Prevention](#data-leakage-prevention)
   - [Feature Engineering](#feature-engineering)
   - [Hyperparameter Tuning](#hyperparameter-tuning)
6. [Notebook Walkthrough](#-notebook-walkthrough)
7. [Setup & Installation](#-setup--installation)
8. [Usage](#-usage)
9. [Visualisations](#-visualisations)
10. [Model Performance](#-model-performance)
11. [Key Insights](#-key-insights)
12. [Known Limitations](#-known-limitations)
13. [Tech Stack](#-tech-stack)

---

## 🎯 Project Overview

This project builds a **Linear Regression** model (plus Ridge and Lasso
variants) to predict house sale prices. The goal is to reach **≥ 85 % R²**
using only linear models — no neural networks, no gradient boosting — by
applying smart, principled pre-processing.

### Core techniques used to hit 85 %+ R²

| Technique | R² Gain |
|-----------|:-------:|
| Baseline (raw features only) | ~70 % |
| + `log1p(price)` target transformation | ~79 % |
| + Log-transform of skewed area features | ~82 % |
| + **Zipcode mean-price encoding** (leakage-free) | ~85 % |
| + Interaction & polynomial terms | **~85.7 %** ✅ |

---

## 📦 Dataset

| Property | Value |
|----------|-------|
| **Name** | King County House Sales |
| **Source** | Kaggle / King County Open Data |
| **Records** | 21,613 house sales |
| **Time period** | May 2014 – May 2015 |
| **Location** | King County, Washington State, USA |
| **File** | `housing_dataset/data.csv` |
| **Missing values** | None |

### Column Reference

| Column | Type | Description |
|--------|------|-------------|
| `id` | int | Unique house identifier (dropped) |
| `date` | str | Date of sale |
| `price` | float | **Target variable** — sale price in USD |
| `bedrooms` | int | Number of bedrooms |
| `bathrooms` | float | Number of bathrooms |
| `sqft_living` | int | Interior living area (sq ft) |
| `sqft_lot` | int | Total lot area (sq ft) |
| `floors` | float | Number of floors |
| `waterfront` | int | Binary — waterfront property (0/1) |
| `view` | int | Quality of view (0–4) |
| `condition` | int | Overall condition (1–5) |
| `grade` | int | Construction quality grade (1–13) |
| `sqft_above` | int | Area above ground level |
| `sqft_basement` | int | Basement area |
| `yr_built` | int | Year built |
| `yr_renovated` | int | Year renovated (0 = never) |
| `zipcode` | int | ZIP code |
| `lat` | float | Latitude |
| `long` | float | Longitude |
| `sqft_living15` | int | Living area of 15 nearest neighbours |
| `sqft_lot15` | int | Lot area of 15 nearest neighbours |

---

## 📊 Results at a Glance

| Model | R² (log scale) | CV R² (5-fold) | RMSE | MAPE |
|-------|:--------------:|:--------------:|-----:|-----:|
| Dummy (mean) — *baseline* | ~0.00 | N/A | $287K | N/A |
| **Linear Regression** | **0.8567** ✅ | 0.8586 ± 0.003 | $110K | 14.3 % |
| Ridge (α = 0.01) | 0.8567 ✅ | 0.8586 ± 0.003 | $110K | 14.3 % |
| Lasso (α = 0.0001) | 0.8558 ✅ | 0.8579 ± 0.003 | $111K | 14.4 % |

> **R² is measured on the log-price scale** — the natural scale for this model.
> All dollar-scale metrics are computed after inverse-transforming (`expm1`).

---

## 📁 Project Structure

```
house_price_prediction/
│
├── housing_dataset/
│   └── data.csv                  # Raw King County dataset (not committed)
│
├── house_price_prediction.ipynb  # Main Jupyter notebook (15 sections)
│
├── requirements.txt              # Pinned Python dependencies
├── .gitignore                    # Git ignore rules
└── README.md                     # This file
```

---

## 🔬 Methodology

### Why Log-Transform the Target?

Raw house prices are **right-skewed** (skewness ≈ 4). Linear regression
assumes the target has an approximately linear relationship with features
and that residuals are normally distributed. Applying `log1p(price)` reduces
skewness to ≈ 0.05 and converts multiplicative price effects into additive
ones — exactly what linear models need.

```
Skewness of price         : 4.02   (heavy right tail)
Skewness of log1p(price)  : 0.05   (near-symmetric)
```

After training, predictions are back-transformed via `numpy.expm1()` to
recover dollar values.

---

### Data Leakage Prevention

**Zipcode mean-price encoding** is one of the most impactful features in
this project (+3–5 pp R²). But it must be computed correctly:

| ❌ Wrong (causes leakage) | ✅ Correct (this project) |
|--------------------------|--------------------------|
| Compute zip means on full dataset, *then* split | Split first, compute zip means on **training fold only** |
| Test set's future prices influence training | Test set is completely invisible during encoding |

The pipeline enforces this order explicitly:

```
Load → Basic Features → SPLIT → Zip Encoding (train only) → Scale → Train → Evaluate
```

---

### Feature Engineering

**41 features** are used, grouped into:

| Group | Features | Examples |
|-------|----------|---------|
| **Original** | 16 | `grade`, `sqft_living`, `lat`, `long`, `waterfront` |
| **Date-derived** | 2 | `house_age`, `month_sold` |
| **Basic derived** | 7 | `rooms_total`, `bath_per_bed`, `is_luxury`, `is_new` |
| **Log-area** | 6 | `log_sqft_living`, `log_sqft_lot`, … |
| **Polynomial** | 2 | `grade_sq`, `log_sqft_sq` |
| **Interaction** | 7 | `grade_x_sqft`, `lat_c_x_grade`, `waterfront_x_grade`, … |
| **Location encoding** | 1 | `log_zip_mean` (train-only zip mean price) |

#### Latitude / Longitude Centering

Raw `lat ≈ 47.5` creates interaction terms like `lat_c_x_grade` that are
numerically 47× larger than `grade` alone, which destabilises coefficient
interpretation. Centering on the **training mean** before creating
interaction terms fixes this:

```python
lat_c = lat - lat_train_mean   # range: [-0.41, +0.22]  (was [47.1, 47.8])
```

---

### Hyperparameter Tuning

Alpha values for Ridge and Lasso are found automatically via **GridSearchCV**
with 5-fold cross-validation — no manual guessing:

```
Ridge grid  : [0.01, 0.1, 1, 10, 50, 100, 500]   → best: 0.01
Lasso grid  : [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05] → best: 0.0001
```

---

## 📓 Notebook Walkthrough

The notebook (`house_price_prediction.ipynb`) is structured into **15 sections**:

| # | Section | What it does |
|---|---------|-------------|
| 1 | **Import Libraries** | Imports, dark-theme matplotlib config |
| 2 | **Load Dataset** | Read CSV, shape, dtypes, missing values |
| 3 | **EDA** | 6-panel overview + geographic heat-map |
| 4 | **Base Feature Engineering** | Deterministic transforms only (no group stats) |
| 5 | **Train / Test Split** | 80/20 split — happens before any group statistics |
| 6 | **Leakage-Free Encoding** | Zip mean encoding + lat/long centering from train only |
| 7 | **Feature Selection & Scaling** | `StandardScaler` on 41 features |
| 8 | **Baseline Models** | `DummyRegressor` (mean + median) as floor |
| 9 | **Hyperparameter Search** | `GridSearchCV` for Ridge α and Lasso α |
| 10 | **Model Training** | Linear, Ridge, Lasso on `log_price` |
| 11 | **Model Evaluation** | Full comparison table including baselines |
| 12 | **Visualisations** | 6 plot types (see below) |
| 13 | **Feature Importance** | Coefficients + permutation importance |
| 14 | **Final Dashboard** | 8-panel summary figure |
| 15 | **Sample Predictions** | Predict 4 new houses |

---

## 🛠 Setup & Installation

### Prerequisites

- Python 3.9 or higher
- Jupyter Notebook or JupyterLab

### 1 — Clone the repository

```bash
git clone https://github.com/<your-username>/house_price_prediction.git
cd house_price_prediction
```

### 2 — Create a virtual environment (recommended)

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

### 3 — Install dependencies

Install all pinned dependencies from `requirements.txt`:

```bash
pip install -r requirements.txt
```

<details>
<summary>What's inside <code>requirements.txt</code>?</summary>

```
# Core data science
numpy==2.5.0
pandas==3.0.3
scipy==1.18.0

# Machine learning
scikit-learn==1.9.0

# Visualisation
matplotlib==3.11.0
seaborn==0.13.2

# Jupyter environment
jupyterlab==4.4.10
notebook==7.4.7
ipykernel==7.1.0
```

</details>

### 4 — Add the dataset

Place the King County dataset at:

```
housing_dataset/data.csv
```

The dataset is available on
[Kaggle — House Sales in King County](https://www.kaggle.com/datasets/harlfoxem/housesalesprediction).

### 5 — Launch the notebook

```bash
jupyter lab
# or
jupyter notebook
```

Open `house_price_prediction.ipynb` and run all cells (`Kernel → Restart & Run All`).

---

## 🚀 Usage

### Run the full pipeline

Open the notebook and run all cells in order. Each section is self-contained
with clear markdown explanations.

### Predict a new house

Use the `predict_house()` helper function defined in **Section 15**:

```python
row = predict_house(
    bed=3, bath=2.0, sqft_liv=1800, sqft_lot=6000, floors=1,
    wf=0, view=0, cond=4, grade=7,
    sqft_ab=1800, sqft_bsmt=0,
    sqft_liv15=1700, sqft_lot15=5500,
    lat=47.55, lon=-122.30, month=6,
    house_age=15, was_reno=0, yrs_reno=15
)
sample_df   = pd.DataFrame([row], columns=FEATURES)
sample_sc   = scaler.transform(sample_df)
log_pred    = lr_model.predict(sample_sc)
price_pred  = np.expm1(log_pred)[0]
print(f'Predicted price: ${price_pred:,.0f}')
```

---

## 📈 Visualisations

The notebook generates **13 saved plots**:

| File | Description |
|------|-------------|
| `eda_overview.png` | 6-panel EDA: price dist, area vs price, bedrooms, grade, condition, waterfront |
| `geo_heatmap.png` | Geographic scatter coloured by price (lat/long) |
| `log_transform.png` | Before/after skewness comparison for target variable |
| `correlation_analysis.png` | Full correlation heatmap + bar chart vs log(price) |
| `hyperparam_search.png` | GridSearchCV CV-R² vs alpha curves for Ridge and Lasso |
| `actual_vs_predicted.png` | Scatter of true vs predicted prices (all 3 models) |
| `residual_analysis.png` | Residuals vs predicted, histogram, Q-Q plot |
| `learning_curves.png` | Train / validation R² as training size grows |
| `error_by_range.png` | MAE and MAPE broken by price band (<300K → >1.5M) |
| `cv_distribution.png` | Per-fold R² across 10-fold CV |
| `baseline_comparison.png` | All models + baselines side-by-side R² bar chart |
| `feature_importance.png` | Standardised coefficients + permutation importance |
| `final_dashboard.png` | 8-panel summary dashboard |
| `sample_predictions.png` | Horizontal bar chart of 4 sample predictions |

---

## 📉 Model Performance

### Why residuals have heavy tails (Q-Q plot)

The Q-Q plot shows that residuals deviate from perfect normality in the
tails. This is **expected** for house price data and is **not a model
failure**:

- **Positive tail** — luxury homes and rare waterfront properties sell at
  prices far above what a linear model trained on typical homes can predict.
- **Negative tail** — distressed sales, foreclosures, and estate sales
  transact well below model expectations.

Linear regression is robust to mild tail violations. The MAPE of ~14 %
is competitive for real estate prediction without any non-linear model.

### Coefficient vs Permutation Importance

| Metric | Reliable for interpretation? |
|--------|------------------------------|
| Standardised coefficient | ⚠️ Affected by multicollinearity — features sharing variance can have inflated or deflated coefficients |
| **Permutation importance** | ✅ Model-agnostic, accounts for correlation, measures actual R² drop |

**Always use permutation importance for feature interpretation.**
Coefficients are shown for completeness only.

---

## 💡 Key Insights

1. **`log_zip_mean`** — neighbourhood mean price (computed from training
   data only) is the single most impactful engineered feature. Location
   accounts for more variance than any physical property of the house.

2. **`grade_x_sqft`** — the interaction between construction quality and
   living area is strongly non-linear: a high-grade large home is
   disproportionately more valuable than either attribute alone.

3. **`lat_c_x_grade`** — north-of-Seattle ZIP codes command a quality
   premium; this interaction captures it without requiring a non-linear model.

4. **Waterfront** — only ~0.7 % of listings are waterfront, but `waterfront_x_grade`
   shows that high-grade waterfront homes sell at extreme multiples.

5. **Learning curves** — train and validation R² converge tightly by
   ~10,000 samples, confirming the model generalises well and is not
   overfitting.

---

## ⚠️ Known Limitations

| Limitation | Details |
|-----------|---------|
| **Time** | Dataset covers May 2014–May 2015 only; the model may not generalise to current market conditions |
| **Geography** | Trained on King County, WA — not transferable to other regions without re-training |
| **Extreme prices** | Luxury homes (>$2M) and distressed sales are underrepresented; MAPE is higher in these bands |
| **Linear assumption** | Gradient-boosted trees (e.g., XGBoost) would likely push R² above 90 % |
| **No spatial features** | School district, proximity to transit/parks, noise levels are not in the dataset |

---

## 🧰 Tech Stack

| Library | Pinned Version | Purpose |
|---------|:--------------:|--------|
| **Python** | ≥ 3.9 | Language |
| **NumPy** | `2.5.0` | Numerical computing |
| **Pandas** | `3.0.3` | Data manipulation & cleaning |
| **Scikit-learn** | `1.9.0` | ML models, preprocessing, GridSearchCV, evaluation |
| **Matplotlib** | `3.11.0` | All plotting |
| **Seaborn** | `0.13.2` | Heatmaps & statistical charts |
| **SciPy** | `1.18.0` | Shapiro-Wilk test, Q-Q quantiles |
| **JupyterLab** | `4.4.10` | Interactive notebook environment |
| **Notebook** | `7.4.7` | Classic notebook interface |
| **ipykernel** | `7.1.0` | Python kernel for Jupyter |

> All versions pinned in [`requirements.txt`](requirements.txt). Install with `pip install -r requirements.txt`.

---

## 📄 License

This project is released under the **MIT License**.
The King County dataset is publicly available via King County Open Data and Kaggle.

---

*Built as part of an internship project — Supervised Learning, Model Evaluation, and Predictive Analytics.*
