# 📊 Evaluation Metrics: Regression vs Classification

Evaluation metrics help quantify how well your model is performing. Choosing the right metric depends on the **type of problem** — regression or classification.

---

## 📈 Regression Metrics

Regression problems predict **continuous numeric values** (e.g., house prices, temperatures, etc.).

| Metric | Description |
|--------|-------------|
| **MAE** (Mean Absolute Error) | Average of absolute differences between predicted and actual values. Less sensitive to outliers. |
| **RMSE** (Root Mean Squared Error) | Square root of average of squared errors. More sensitive to large errors/outliers. |
| **R² Score** (Coefficient of Determination) | Measures how well the regression line approximates the real data. Ranges from -∞ to 1.0. Higher is better. |
| **MAPE** (Mean Absolute Percentage Error) | Error as a percentage. Avoid if actual values can be 0. |
| **MSLE** (Mean Squared Log Error) | Useful when target is exponential or highly skewed. |

> ✅ Use RMSE if you care more about **large errors**.  
> ✅ Use MAE for **balanced sensitivity** to errors.  
> ✅ Use R² to understand **explained variance**.

---

## 📉 Classification Metrics

Classification problems predict **categories** (e.g., spam or not, survived or not).

| Metric     | Description |
|------------|-------------|
| **Accuracy**       | % of correctly predicted labels. Best for balanced datasets. |
| **Precision**      | TP / (TP + FP): How many predicted positives were actually correct? |
| **Recall**         | TP / (TP + FN): How many actual positives were correctly predicted? |
| **F1 Score**       | Harmonic mean of precision and recall. Good for imbalanced classes. |
| **ROC-AUC**        | Measures separability. Area under ROC curve. 1 = perfect, 0.5 = random. |
| **Log Loss**       | Penalizes confident wrong predictions. Good for probabilistic classifiers. |
| **Confusion Matrix** | Table that shows TP, FP, TN, FN counts. Great for visualizing errors. |

> ✅ Use **Precision/Recall** for **imbalanced data** (e.g., fraud detection).  
> ✅ Use **ROC-AUC** when comparing **classifier thresholds**.  
> ✅ Use **F1 Score** when both precision and recall matter equally.

---

## 🧪 Summary Table

| Problem Type   | Metric Type        | When to Use                           |
|----------------|--------------------|----------------------------------------|
| Regression     | MAE                | Understand typical prediction error    |
| Regression     | RMSE               | Penalize large errors more heavily     |
| Regression     | R² Score           | Measure model fit (variance explained) |
| Classification | Accuracy           | Balanced datasets                      |
| Classification | Precision/Recall   | Imbalanced datasets                    |
| Classification | F1 Score           | Balanced tradeoff between P & R        |
| Classification | ROC-AUC            | Evaluate probabilistic classifiers     |

---

## 📝 Notes

- Always evaluate multiple metrics for a **well-rounded view** of model performance.
- For imbalanced classification tasks, **accuracy alone is misleading**.
- Use **cross-validation** to average performance across folds.
- Use **grid search** or **randomized search** to find hyperparameters that optimize your desired metric.

---

## 📌 Example Use Cases

| Dataset       | Problem Type   | Metrics Used         |
|---------------|----------------|-----------------------|
| Titanic       | Classification | Accuracy, F1 Score    |
| House Prices  | Regression     | MAE, RMSE, R²         |
| Credit Fraud  | Classification | Precision, Recall, AUC|
| Salary Prediction | Regression | MAE, R²               |

---

> ✅ Pro Tip: Track and log **all metrics** across experiments to compare model improvements over time.
