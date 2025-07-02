# 🚢 Titanic - Machine Learning from Disaster

Welcome aboard one of the most iconic beginner-friendly machine learning challenges — **Titanic: ML from Disaster**. This repository contains my end-to-end exploration and model development for the Kaggle Titanic competition, as part of my **SSJ3-ML-Journey**.

## 🎯 Objective

Predict which passengers survived the Titanic shipwreck using supervised classification techniques. The model learns from the **training dataset** (`train.csv`) and makes predictions on the **test dataset** (`test.csv`).

## 🧠 Challenge Context

On April 15, 1912, the Titanic tragically sank, taking 1502 out of 2224 lives. While luck played a role, **factors such as age, gender, and class** heavily influenced survival odds. The task is to identify these patterns and build a predictive model to answer:

> “What sorts of people were more likely to survive?”

## 📁 Dataset Overview

| Feature          | Description                                                    |
|------------------|----------------------------------------------------------------|
| `Survived`       | Target variable (0 = No, 1 = Yes)                              |
| `Pclass`         | Ticket class (1 = 1st, 2 = 2nd, 3 = 3rd)                        |
| `Sex`            | Gender                                                         |
| `Age`            | Age in years (may contain missing or estimated values)         |
| `SibSp`          | # of siblings / spouses aboard                                 |
| `Parch`          | # of parents / children aboard                                 |
| `Ticket`         | Ticket number                                                  |
| `Fare`           | Passenger fare                                                 |
| `Cabin`          | Cabin number (may be missing)                                  |
| `Embarked`       | Port of Embarkation (C = Cherbourg, Q = Queenstown, S = Southampton) |

> Additional file: `gender_submission.csv` – baseline prediction assuming only females survived.

## 📊 Planned Workflow

- ✅ Load & Understand the Data  
- ✅ Clean & Handle Missing Values  
- ✅ Perform EDA (Exploratory Data Analysis)  
- ✅ Feature Engineering  
- ✅ Model Selection (Logistic Regression, Decision Trees, etc.)  
- ✅ Evaluate Model  
- ✅ Submit to Kaggle  
- 🔜 Add visual dashboards using `Plotly` / `Altair`  
- 🔜 Share decision flow and pipeline as **tree diagrams / flowcharts**

## 🔍 Tools & Libraries

- `pandas`, `numpy`, `matplotlib`, `seaborn`
- `scikit-learn`, `logistic-regression`, `Decision-Trees`, `Random-Forests`, `SVMs`
- `plotly`, `matplotlib`, `seaborn` (for interactive EDA - planned)
- `graphviz` or `diagrams.net` (for flowchart design - planned)

## 🗂️ Folder Structure

```
titanic_classification/
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_modeling.ipynb
├── data/
│   ├── raw/
│   ├── ├── train.csv
│   ├── ├── test.csv
│   ├── ├── gender_submission.csv
├── outputs/
│   ├── submission.csv
│   ├── visualizations/
├── README.md
```
