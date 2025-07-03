# 🏡 House Prices - Advanced Regression Techniques
<p align="center">
  <a href="https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques">
    <img src="https://img.shields.io/badge/Kaggle-House_Prices_Competition-blue?style=for-the-badge&logo=kaggle&logoColor=white" alt="House Prices Competition"/>
  </a>
</p>


Welcome to my second Kaggle project in the **SSJ3-ML-Journey** series! This competition is focused on **predicting house prices** using advanced regression techniques and **creative feature engineering** with real-world data from Ames, Iowa.

---

## 🎯 Objective

To build a regression model that accurately predicts the **SalePrice** of homes based on a diverse set of **79 explanatory variables** covering physical attributes, quality metrics, and location data.

---

## 📦 Dataset Overview

- `train.csv` – 1460 rows × 81 columns (with target `SalePrice`)
- `test.csv` – 1459 rows × 80 columns (excluding `SalePrice`)
- `data_description.txt` – Detailed descriptions of all columns
- `sample_submission.csv` – Template for Kaggle submissions

### 📌 Target Variable

| Feature     | Description                        |
|-------------|------------------------------------|
| `SalePrice` | Final price of the house in dollars |

### 📌 Sample Feature Highlights

| Feature        | Description                                      |
|----------------|--------------------------------------------------|
| `OverallQual`  | Overall material and finish quality              |
| `GrLivArea`    | Ground living area in square feet                |
| `GarageCars`   | Number of garage car spaces                      |
| `YearBuilt`    | Year of house construction                       |
| `Neighborhood` | Location within Ames city limits                 |
| `LotFrontage`  | Linear feet of street connected to property      |
| `BsmtQual`     | Height of basement                               |

Full feature list can be found in `data_description.txt`.

---

## 🧪 Planned Workflow

- ✅ Data Loading and Cleaning
- ✅ Missing Value Treatment
- ✅ Exploratory Data Analysis (EDA)
- ✅ Feature Engineering
- ✅ Data Transformation (scaling, encoding)
- ✅ Model Training (Linear, Ridge, Lasso, RF, XGBoost)
- ✅ Hyperparameter Tuning (GridSearchCV / Optuna)
- ✅ Model Evaluation (MAE, RMSE, R²)
- ✅ Kaggle Submission

---

## 📚 Key Skills Practiced

- 🧠 **Creative Feature Engineering**
- 📏 **Regression Modeling & Tuning**
- 🧹 **Smart Missing Value Imputation**
- 📊 **Statistical EDA + Visual Insights**
- 🛠️ **Model Evaluation & Ensemble Approaches**

---

## 🧪 Exploratory Data Analysis (Planned Sections)

> _To be updated as the project progresses._

- 🔍 Distribution of `SalePrice` (target)
- 🔁 Correlation heatmap of numerical features
- 🧱 Feature impact analysis: `OverallQual`, `GrLivArea`, `Neighborhood`, etc.
- ❌ Outlier detection
- 🎛️ Skewness and log transformations
- 📐 Feature selection based on VIF & correlation

---

## 📈 Modeling Approach

> _To be updated with results later._

| Model        | MAE   | RMSE  | R² Score |
|--------------|-------|-------|----------|
| LinearReg    | TBD   | TBD   | TBD      |
| Ridge        | TBD   | TBD   | TBD      |
| Lasso        | TBD   | TBD   | TBD      |
| XGBoost      | TBD   | TBD   | TBD      |
| StackedModel | TBD   | TBD   | TBD      |

---

## 📊 Visualizations

> _To be added here:_

- `EDA_SalePrice_Distribution.png`
- `Correlation_Heatmap.png`
- `Feature_Importance_XGBoost.png`
- `Prediction_vs_Actual.png`
- `Residuals_Plot.png`

---

## 💾 Submission Format

```csv
Id,SalePrice
1461,169000
1462,187000
```

---

## 🛠️ Future Work

- 🔍 SHAP interpretation of tree-based models  
- 🧪 Feature selection via **Lasso Regression**  
- 🤝 Ensemble methods: **Stacking** / **Blending**  
- 🛠️ Advanced preprocessing using `ColumnTransformer` pipelines  
- 🤖 AutoML experiments (e.g., **AutoSklearn**, **Optuna**)  
- 📉 Feature reduction & visualization using **PCA** / **t-SNE**

---

## 📚 Learnings (To Be Updated)

- 🧼 Importance of data cleaning in real estate datasets  
- 🎯 Handling **skewness** in target variable  
- 🔠 Encoding and managing **categorical + ordinal features**  
- 🛡️ Tuning **regularization** to avoid overfitting

---

## ✍️ Author

- **Name**: Guna Venkat Doddi  
- **Project**: Part of `SSJ3-Kaggle-Projects` repository  
- **Contact**: [![GitHub - Guna Venkat Doddi](https://img.shields.io/badge/GitHub-Guna--Venkat--Doddi-black?logo=github&style=flat-square)](https://github.com/Guna-Venkat)

---

