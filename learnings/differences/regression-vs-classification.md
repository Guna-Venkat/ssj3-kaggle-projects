# 📚 Regression vs Classification — Core ML Problem Types

Machine Learning problems are often divided into two main types: **Regression** and **Classification**. Knowing which type your problem falls into is critical for choosing the right preprocessing, models, and evaluation strategies.

---

## 🧠 What’s the Difference?

| Aspect           | Regression                                 | Classification                             |
|------------------|---------------------------------------------|---------------------------------------------|
| Output Type      | Continuous numeric value (e.g., price)      | Discrete category or class (e.g., survived) |
| Target Variable  | Real number                                 | Label (binary, multiclass)                  |
| Goal             | Predict a quantity                          | Predict a class                             |
| Example          | House price prediction                      | Titanic survival prediction                 |
| Evaluation       | Error/Accuracy-based                        | Accuracy, Precision, Recall, etc.           |

---

## 🔧 Algorithm Types

### 📐 Regression Algorithms
- `LinearRegression`
- `Ridge`, `Lasso`, `ElasticNet`
- `RandomForestRegressor`
- `XGBoostRegressor`, `LightGBMRegressor`
- `SVR` (Support Vector Regressor)
- `KNeighborsRegressor`

### 🧠 Classification Algorithms
- `LogisticRegression`
- `RandomForestClassifier`
- `XGBoostClassifier`, `LightGBMClassifier`
- `KNeighborsClassifier`
- `SVC` (Support Vector Classifier)
- `NaiveBayes`

---

## 📊 Evaluation Metrics

### 📈 Regression Metrics

| Metric | Description                        |
|--------|------------------------------------|
| MAE    | Mean Absolute Error                |
| RMSE   | Root Mean Squared Error            |
| R²     | Coefficient of Determination       |

### 📉 Classification Metrics

| Metric     | Description                             |
|------------|-----------------------------------------|
| Accuracy   | Overall % of correct predictions        |
| Precision  | TP / (TP + FP)                          |
| Recall     | TP / (TP + FN)                          |
| F1 Score   | Harmonic mean of precision and recall   |
| ROC-AUC    | Area under ROC curve (probabilistic)    |

---

## 🧹 Data Preprocessing Differences

| Step                    | Regression                         | Classification                        |
|-------------------------|-------------------------------------|----------------------------------------|
| Handling Skewed Target  | Often use log/Box-Cox               | Not needed                             |
| Feature Scaling         | Required for linear/SVM models      | Often optional (except SVM/KNN)        |
| Label Encoding          | For ordinal categoricals            | Required for class labels              |
| OneHot Encoding         | Low-cardinality features            | Useful for non-ordinal categoricals    |
| Target Transformation   | e.g., log(SalePrice)                | Not applicable                         |

---

## 📦 Sample Use Cases

### ✅ Regression Problems
- Predicting house prices
- Forecasting sales
- Estimating age from images
- Predicting flight delays in minutes

### ✅ Classification Problems
- Identifying spam emails
- Predicting loan approval
- Diagnosing diseases (e.g., diabetes)
- Image classification (dog vs cat)

---

## 💾 Submission Format Tips

| Problem Type    | Submission Format Example                     |
|------------------|-----------------------------------------------|
| Regression       | `Id,SalePrice`                                |
| Classification   | `PassengerId,Survived`                        |

---

## 🛡️ Overfitting Prevention

| Technique            | Applicable to Both |
|----------------------|--------------------|
| Cross-validation     | ✅                 |
| Regularization       | ✅ (`Lasso`, `Ridge`) |
| Pruning (Tree Models)| ✅                 |
| Early Stopping       | ✅ (Boosting Models) |
| Feature Selection    | ✅                 |

---

## 🧠 Final Thoughts

- Use **regression** when your target is a real-valued number.
- Use **classification** when your target is a category or class label.
- Many preprocessing techniques apply to both, but metric choice and model evaluation differ significantly.
- Keep a consistent modeling template and reuse where possible — only change the model and evaluation logic based on the problem type.

---

## 📝 Learning Projects

| Dataset        | Problem Type    | Target Variable | Model Examples         |
|----------------|------------------|------------------|-------------------------|
| Titanic        | Classification   | `Survived`       | LogisticRegression, XGBoostClassifier |
| House Prices   | Regression       | `SalePrice`      | LinearRegression, XGBoostRegressor    |

---

> ✅ Use this guide as a reusable reference when approaching any tabular machine learning dataset.
