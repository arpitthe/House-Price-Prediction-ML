# 🏠 King County House Price Prediction

An end-to-end Machine Learning project that predicts house prices in **King County, Washington** using **Linear Regression**, **Ridge Regression**, and **Lasso Regression**.

The project demonstrates the complete supervised learning workflow—from data exploration and feature engineering to model evaluation and interpretation—using over **21,000 real-world housing records**.

---

## 📌 Project Overview

This project focuses on building a robust Linear Regression pipeline capable of predicting house prices while following proper Machine Learning practices such as:

- Exploratory Data Analysis (EDA)
- Data Cleaning & Preprocessing
- Feature Engineering
- Feature Scaling
- Target Transformation
- Cross Validation
- Hyperparameter Tuning
- Model Evaluation
- Residual Analysis
- Feature Importance Analysis

Instead of relying on complex ensemble models, this project shows how carefully engineered features can significantly improve the performance of classical linear models.

---

# 📊 Dataset

**Dataset:** King County House Sales Dataset

- **Records:** 21,613
- **Location:** King County, Washington (USA)
- **Time Period:** May 2014 – May 2015
- **Target Variable:** House Sale Price

The dataset contains various features including:

- Bedrooms
- Bathrooms
- Living Area
- Lot Area
- Floors
- Grade
- Condition
- Waterfront
- View
- Latitude
- Longitude
- Zipcode
- Nearby Property Information

---

# 🚀 Features

✔ Complete Exploratory Data Analysis

✔ Missing Value Analysis

✔ Correlation Analysis

✔ Feature Engineering

✔ Log Transformation

✔ Standard Scaling

✔ Train/Test Split

✔ Cross Validation

✔ Linear Regression

✔ Ridge Regression

✔ Lasso Regression

✔ Hyperparameter Tuning

✔ Residual Analysis

✔ Learning Curves

✔ Feature Importance

✔ Geographic Price Visualization

✔ Model Comparison Dashboard

---

# 🛠️ Tech Stack

- Python
- NumPy
- Pandas
- Matplotlib
- Seaborn
- Scikit-learn
- SciPy
- Jupyter Notebook

---

# 📈 Model Performance

| Model | R² Score | RMSE | MAE | MAPE |
|--------|----------|------|------|------|
| Linear Regression | **0.855** | ~$111K | ~$73K | 14.4% |
| Ridge Regression | 0.853 | ~$112K | ~$74K | 14.5% |
| Lasso Regression | 0.852 | ~$113K | ~$74K | 14.6% |

The Linear Regression model achieved the best overall performance while maintaining excellent generalization across cross-validation folds.

---

# 🧠 Feature Engineering

Several additional features were engineered to improve prediction performance.

Examples include:

- House Age
- Years Since Renovation
- Total Rooms
- Bathrooms per Bedroom
- Luxury Home Indicator
- Log-Transformed Area Features
- Polynomial Features
- Interaction Features
- Zipcode Mean Price Encoding (Leakage-Free)

These engineered features improved model performance substantially compared to using only the raw dataset.

---

# 📊 Project Visualizations

The notebook generates several analytical visualizations including:

- House Price Distribution
- Correlation Heatmap
- Geographic Price Heatmap
- Feature Importance
- Learning Curves
- Residual Analysis
- Actual vs Predicted Prices
- Prediction Error Analysis
- Cross Validation Results
- Final Model Dashboard

---

# ⚙ Installation

Clone the repository

```bash
git clone https://github.com/yourusername/House-Price-Prediction.git
```

Move into the project directory

```bash
cd House-Price-Prediction
```

Install dependencies

```bash
pip install -r requirements.txt
```

Launch Jupyter Notebook

```bash
jupyter notebook
```

Run all notebook cells sequentially.

---

# 📌 Project Workflow

```
Raw Dataset
      │
      ▼
Exploratory Data Analysis
      │
      ▼
Data Cleaning
      │
      ▼
Feature Engineering
      │
      ▼
Train-Test Split
      │
      ▼
Feature Scaling
      │
      ▼
Model Training
      │
      ▼
Hyperparameter Tuning
      │
      ▼
Model Evaluation
      │
      ▼
Visualization & Insights
```

---

# 📚 Key Learnings

During this project I gained practical experience with:

- Data preprocessing
- Feature engineering
- Linear regression assumptions
- Model evaluation metrics
- Cross-validation
- Hyperparameter tuning
- Residual analysis
- Model interpretation
- Data visualization
- Building complete Machine Learning workflows

---

# 📌 Future Improvements

- Random Forest Regression
- XGBoost Regression
- LightGBM
- CatBoost
- Flask/FastAPI Deployment
- Interactive Web Application
- Docker Support

---

# 📄 License

This project is intended for educational and learning purposes.

The dataset belongs to **King County Open Data** and is publicly available through Kaggle.

---

## ⭐ If you found this project useful, consider giving it a star.
