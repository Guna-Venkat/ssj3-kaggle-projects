# 🧠 Amazon Product Review Classification — NLP Project

<p align="center">
  <a href="https://www.kaggle.com/datasets/bittlingmayer/amazonreviews">
    <img src="https://img.shields.io/badge/Kaggle-Amazon_Reviews_Dataset-blue?style=for-the-badge&logo=kaggle&logoColor=white" alt="Kaggle Amazon Reviews Dataset"/>
  </a>
</p>

This repository contains my end-to-end experimentation and implementation of **Natural Language Processing techniques** on the Amazon Product Reviews dataset, as part of my **SSJ3-NLP-Journey**. The goal is to classify reviews as **positive or negative** using a progressive roadmap of NLP techniques — from classical ML models to deep learning and transformer-based architectures.

---

## 🎯 Objective

Build multiple NLP models to classify product reviews into **positive (1)** or **negative (0)** sentiment classes, showcasing progression from:
- 🔹 Classical ML → 
- 🔸 Deep Learning (LSTM, Embeddings) →
- 🟣 Transformers (BERT) →
- 🧪 Ensembles →
- 🚀 Deployed UI Interface

---

## 📁 Dataset Overview

| Feature   | Description                                     |
|-----------|-------------------------------------------------|
| `reviewText` | The actual product review written by users     |
| `label`   | Target variable (0 = Negative, 1 = Positive)     |
| `summary` | Short summary/title of the review                |

📝 Note: For this task, we primarily use `reviewText` and `label`.

---

## 🧭 Workflow & Techniques

### ✅ NLP Roadmap (Inspired by Vision Project: SVC → CNN → TL)

| Level | Technique | Tools | Description |
|-------|-----------|-------|-------------|
| 🟢 Basic | TF-IDF + Logistic Regression / SVC | `scikit-learn` | Classic baseline for short text |
| 🟢 Basic | CountVectorizer + Naive Bayes | `sklearn` | Fast, interpretable probabilistic model |
| 🟡 Intermediate | Word Embeddings (GloVe/FastText) + MLP | `gensim`, `keras` | Word vectors for semantic representation |
| 🟡 Intermediate | LSTM / GRU | `TensorFlow`, `Keras` | Sequence modeling with RNNs |
| 🔵 Advanced | BERT (Fine-tuning) | `HuggingFace Transformers` | State-of-the-art transformer classification |
| 🔵 Advanced | RAG (Optional) | `Haystack`, `LangChain` | Retrieval-Augmented QA-style modeling |
| 🧪 Bonus | Ensemble (LR + LSTM + BERT) | `sklearn`, `custom` | Voting classifier to test robustness |
| 🚀 Deployment | Streamlit / Gradio App | `streamlit`, `gradio` | Showcase model predictions + metrics in a UI |

---

## 🧰 Tools & Libraries

- `pandas`, `numpy`, `matplotlib`, `seaborn`
- `scikit-learn`, `xgboost`
- `keras`, `tensorflow`, `gensim`
- `transformers`, `datasets` (Hugging Face)
- `plotly`, `gradio`, `streamlit`, `haystack`
- `nltk`, `spacy`, `wordcloud` (text preprocessing)

---

## 📊 Planned Folder Structure

```
amazon_review_nlp/
├── notebooks/
│ ├── 01_eda_preprocessing.ipynb
│ ├── 02_classical_models.ipynb
│ ├── 03_embeddings_mlp.ipynb
│ ├── 04_lstm_models.ipynb
│ ├── 05_bert_finetuning.ipynb
│ ├── 06_ensemble.ipynb
│ ├── 07_streamlit_ui.ipynb
├── data/
│ ├── raw/
│ ├── processed/
├── outputs/
│ ├── plots/
│ ├── models/
│ ├── metrics/
├── app/
│ ├── streamlit_app.py
│ ├── requirements.txt
├── README.md
```


---

## 📈 Visualizations

- ✅ Word Frequency and WordClouds  
- ✅ TF-IDF Feature Importances  
- ✅ t-SNE Plots for Word Embeddings  
- ✅ Model-wise Accuracy/F1 Comparison  
- ✅ Confusion Matrix Heatmaps  
- ✅ Attention Scores (for BERT) *(optional)*  
- 🔜 Visual Embedding Explorer (via TensorBoard Projector)

---

## 🤖 Models Compared

| Model | Accuracy | F1 Score | Remarks |
|-------|----------|----------|---------|
| TF-IDF + Logistic Regression | TBD | TBD | Classical strong baseline |
| CountVectorizer + Naive Bayes | TBD | TBD | Fast & interpretable |
| GloVe Embeddings + MLP | TBD | TBD | Better semantics |
| LSTM + Embedding Layer | TBD | TBD | Sequence-aware |
| BERT Fine-tuned | TBD | TBD | SOTA for text |
| Ensemble (LR + LSTM + BERT) | TBD | TBD | Robust output |
| RAG | TBD | TBD | For QnA-type use cases |

---

## ⚙️ Model Training Pipeline

- Clean & tokenize text (`nltk`, `re`, `spacy`)
- Train-test split
- Classical models with TF-IDF & CountVectorizer
- Embedding matrix (GloVe/FastText) for DL models
- Sequence padding for RNNs
- Transformer tokenization and fine-tuning (HuggingFace)
- Cross-validation and metric logging
- Ensemble logic for final robustness check
- Model saving for UI usage

---

## 🚀 Streamlit / Gradio UI

### 🔍 Features:
- Live prediction from user review input
- Choose backend model (TF-IDF, LSTM, BERT)
- Show predicted sentiment + probability
- Visualize:
  - Confusion matrix
  - Accuracy/F1 score per model
  - Word importance (for classical)
  - BERT attention (optional)

---

## 🧾 Sample Input

```
"Great quality headphones for the price. Bass is excellent. Highly recommend!"
→ Predicted: POSITIVE (confidence: 0.97)
```


---

## 📤 Future Work

- ✅ UI with comparison across models  
- 🔜 Add SHAP/Attention Interpretability for deep models  
- 🔜 Hyperparameter optimization using Optuna  
- 🔜 Add dataset explorer inside UI (search & filter reviews)  
- 🔜 Deploy using Hugging Face Spaces / Streamlit Cloud  

---

## 📚 Learnings

- Classical models perform well with TF-IDF for short reviews  
- Word embeddings & RNNs improve contextual understanding  
- BERT fine-tuning provides SOTA performance out of the box  
- Visualizing embeddings and word importance gives insights into model behavior  
- Building a UI boosts usability and showcases applied ML skills  

---

## ✍️ Author

- **Name**: Guna Venkat Doddi  
- **Project**: Part of `SSJ3-NLP-Projects` repository  
- **Contact**: [![GitHub - Guna Venkat Doddi](https://img.shields.io/badge/GitHub-Guna--Venkat--Doddi-black?logo=github&style=flat-square)](https://github.com/Guna-Venkat)

---