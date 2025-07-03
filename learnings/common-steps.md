# 🧠 Common Machine Learning Workflow (SSJ3 Journey)

This document outlines the **common ML pipeline** steps shared across most structured datasets like **Titanic** (classification) and **House Prices** (regression). It serves as a base template before diverging into task-specific modeling.

---

## 🗂️ 1. Data Loading

- Use `pandas.read_csv()` to load datasets.
- Check initial shape and use `.head()`, `.info()`, `.describe()` for quick exploration.

```python
train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')
```

## 🧹 2. Data Cleaning

- Identify and handle missing values.
- Convert data types appropriately  
  (e.g., `object → category`, `str → datetime`).
- Drop redundant or identifier columns  
  (e.g., `PassengerId`, `Id`).

---

## 🔍 3. Exploratory Data Analysis (EDA)

- Visualize target distribution using `seaborn` or `matplotlib`.
- Use `.value_counts()` or `.groupby()` to explore categorical features.
- Plot a correlation heatmap to identify multicollinearity.
- Detect outliers and highly skewed features.

---

## 🧱 4. Feature Engineering

- Create new features:  
  `FamilySize = SibSp + Parch + 1`  
  `TotalSF = TotalBsmtSF + 1stFlrSF + 2ndFlrSF`
- Apply transformations for skewed features  
  (e.g., `log`, `Box-Cox`).
- Use domain knowledge to map ordinal categories  
  (e.g., `Ex: 5`, `Gd: 4`, ..., `None: 0`).

---

## 🔠 5. Encoding Categorical Features

- **Ordinal Encoding** – for features with natural order  
  (e.g., `ExterQual`, `BsmtCond`)
- **Label Encoding** – when meaningful numeric mapping exists.
- **OneHot Encoding** – only when:
  - Feature has low cardinality
  - No natural ordering exists

❗ Avoid OneHot for high-cardinality columns unless using tree-based models.

---

## ⚖️ 6. Feature Scaling

- Apply `StandardScaler` or `RobustScaler` for normalization.
- Not required for tree-based models like RandomForest or XGBoost.
- Always **scale after splitting** train/test data to avoid data leakage.

---

## 🧪 7. Model Building (Generic Flow)

```python
from sklearn.model_selection import train_test_split

X = train_df.drop('target', axis=1)
y = train_df['target']

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.1, stratify=y
)
```

## 🔧 Algorithms

### 🧠 Classification Models
- `LogisticRegression`
- `RandomForestClassifier`
- `XGBoostClassifier`

### 📐 Regression Models
- `LinearRegression`
- `Lasso`
- `Ridge`
- `XGBoostRegressor`

---

## 📊 8. Evaluation Metrics

### 📈 Regression Metrics

| Metric | Description                  |
|--------|------------------------------|
| MAE    | Mean Absolute Error          |
| RMSE   | Root Mean Squared Error      |
| R²     | Coefficient of Determination |

### 📉 Classification Metrics

| Metric     | Description                              |
|------------|------------------------------------------|
| Accuracy   | % of correct predictions                 |
| Precision  | TP / (TP + FP)                           |
| Recall     | TP / (TP + FN)                           |
| F1 Score   | Harmonic mean of precision & recall      |
| ROC-AUC    | Area under the ROC curve (probabilities) |

---

## 🧠 Model Interpretability (Optional)

- `.coef_` – for linear models  
- `.feature_importances_` – for tree-based models  
- `permutation_importance` – model-agnostic  
- **SHAP** – advanced interpretability (e.g., XGBoost, LightGBM)

---

## ✅ Checklist

- [x] Missing values handled
- [x] Categorical data encoded
- [x] Numeric features scaled
- [x] Evaluation metrics computed
- [x] Hyperparameters tuned

---

## 📝 Notes

- Logic is largely reusable across structured datasets like Titanic and House Prices.
- Final **submission format** depends on problem type:
  - **Classification** → predicted class label
  - **Regression** → numeric prediction value
- Avoid overfitting via:
  - Regularization (`Ridge`, `Lasso`)
  - Tree pruning or max depth
  - Cross-validation
  - Early stopping (for boosting models)
