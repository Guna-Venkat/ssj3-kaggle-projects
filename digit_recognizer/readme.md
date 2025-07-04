# 🧠 Digit Recognizer – Dimensionality Reduction + ML/DL Exploration

[![Kaggle Competition](https://img.shields.io/badge/Kaggle-Digit%20Recognizer-20BEFF?logo=kaggle&logoColor=white)](https://www.kaggle.com/competitions/digit-recognizer/overview)

---

## 📌 Competition Overview

### 🎯 Goal

Your objective is to classify grayscale images of hand-drawn digits (0–9). The dataset used is the **classic MNIST** dataset—commonly known as the *"Hello World"* of computer vision.

You’ll build models to correctly predict the digit in each image. The competition evaluates your solution based on **categorization accuracy**.

---

## 📂 Dataset Description

- **Train File**: `train.csv` – contains **785 columns**:
  - `label`: the correct digit (0–9)
  - `pixel0` to `pixel783`: intensity of each pixel (values from 0 to 255)
  
- **Test File**: `test.csv` – contains **784 columns**:
  - `pixel0` to `pixel783`: no label, your model will predict this

- Each image is **28x28 pixels**, unrolled into a 784-dimensional vector.

---

## 👶 Ideal Starting Point

This competition is ideal if:

- You know some **Python or R**
- You have **machine learning basics**
- You're **new to computer vision**

It’s a great opportunity to practice with:

- 🧠 **Neural networks** (basic and deep)
- 🧪 **Dimensionality reduction** (PCA, Autoencoders)
- ⚙️ **ML techniques** like SVM, Random Forest, kNN
- 📊 **Data visualization** and segmentation

---

## 🎯 Project-Specific Goals

This project is a comprehensive experimentation ground to:

- 🔍 Reduce dimensionality with **PCA and Autoencoders**
- 🧩 Use **image segmentation** via clustering (k-means)
- ⚙️ Compare performance of **classical ML models** (SVC, RF, kNN)
- 🧠 Build **CNN-based deep learning** pipelines
- 🔗 Use **ensembling techniques** (voting, stacking) for better accuracy
- ⚡ Compare **model performance** with and without reduction
- 🧪 Learn through **applied research**

---

## 🚀 Approach Overview

```text
1. Data Exploration
2. Dimensionality Reduction (PCA, Autoencoders)
3. Clustering / Segmentation (kMeans on pixels/features)
4. Traditional ML Models (SVC, RF)
5. CNN-based Deep Learning Models
6. Data Augmentation
7. Ensemble Strategies (Voting, Stacking)
8. Evaluation, Visualization, Saving Outputs
```

## 📊 Dimensionality Reduction Techniques

### 🟢 PCA
- Reduce dimensionality while retaining **95–98% variance**.
- **Benefit**: Speeds up training for traditional ML models.

### 🔵 Autoencoders
- Capture **non-linear feature representations**.
- **Benefit**: Often better than PCA for complex data.

---

## 🧩 Image Segmentation Strategy

We experiment with:

- **k-Means Clustering** on flattened pixel space or on extracted CNN features.
- Represent each image with **cluster memberships** or **cluster-based statistics**.
- Save transformed features to `.npy` or `.pkl` files for reuse.

---

## 🤖 Machine Learning Models

| Model              | Input Type        | Notes                            |
|-------------------|-------------------|----------------------------------|
| **SVC**            | PCA / clustered   | Good baseline for clean data     |
| **Random Forest**  | PCA / clustered   | Handles noise well               |
| **CNN**            | Raw + augmented   | End-to-end learning              |
| **Autoencoder + ML**| Compressed vector | Dimensionality-aware combo       |

---

## 🧬 Data Augmentation (for CNNs)

- **Techniques**: Rotation, shifting, zooming, flipping  
- **Libraries**: `ImageDataGenerator`, `Albumentations`

---

## 🧠 Ensemble Techniques

### ✅ Soft Voting
- Blend predictions using **average of probabilities**.

### ✅ Stacking
- Use outputs of base models as input to a **meta-model** (e.g., Logistic Regression).

---

## 📈 Evaluation Metrics

- ✅ Accuracy on test set  
- ✅ Confusion matrix  
- ✅ Per-class performance  
- ✅ Training time vs performance comparison

---

## 📁 Project Structure

```
digit-recognizer/
├── data/                  # Dataset files (train.csv, test.csv, etc.)
├── notebooks/             # Jupyter notebooks for each experiment
│   ├── 01_explore_data.ipynb         # Initial EDA
│   ├── 02_pca_classical_ml.ipynb     # PCA + SVC, RF
│   ├── 03_autoencoder_ml.ipynb       # Autoencoder + SVC
│   ├── 04_kmeans_features.ipynb      # kMeans segmentation + ML
│   ├── 05_cnn_baseline.ipynb         # Baseline CNN
│   ├── 06_cnn_augmented.ipynb        # CNN with augmentations
│   ├── 07_ensemble_models.ipynb      # Voting/stacking ensembles
│   └── 08_visualizations.ipynb       # t-SNE, UMAP plots
├── models/                # Saved models (Pickle, H5, etc.)
├── outputs/               # Predictions and logs
├── requirements.txt       # List of required packages
└── README.md              # Project overview
```

---

## 🧪 Planned Experiments

| Experiment                       | Status | Notes                                |
|----------------------------------|--------|--------------------------------------|
| Baseline CNN                     | ⬜️     |                                      |
| PCA + SVC                        | ⬜️     | Try 95% and 98% retained variance    |
| PCA + RF                         | ⬜️     | Check for overfitting                |
| Autoencoder + SVC                | ⬜️     | Compare vs PCA                       |
| kMeans Cluster Features + ML     | ⬜️     | Convert clusters into feature sets   |
| CNN with Augmentation            | ⬜️     | Improve generalization               |
| Stacked Model (SVC+RF+CNN)       | ⬜️     | Final ensemble                       |

---

## 🛠 Tools & Libraries Used

- **Python**, **NumPy**, **Pandas**, **Scikit-learn**  
- **TensorFlow / Keras**  
- **Matplotlib / Seaborn**  
- **UMAP / t-SNE** (for visualization)  
- **Gradio** (for interactive demos, optional)

---

## 🧠 Learnings & Reflections (To be updated)

- 📌 Comparative impact of **PCA vs Autoencoder**  
- 📌 When to prefer **classical ML over DL**  
- 📌 Efficiency of **ensembles in small datasets**  
- 📌 Real-world applications of **dimensionality reduction**

---

## 📜 License

**MIT License** – Free to use with attribution.  
Digit data provided by [Kaggle](https://www.kaggle.com/competitions/digit-recognizer/overview).

---

## 👨‍💻 Author

- **Name**: Guna Venkat Doddi  
- **Project**: Part of `SSJ3-Kaggle-Projects` repository  
- **Contact**: [![GitHub - Guna Venkat Doddi](https://img.shields.io/badge/GitHub-Guna--Venkat--Doddi-black?logo=github&style=flat-square)](https://github.com/Guna-Venkat)s