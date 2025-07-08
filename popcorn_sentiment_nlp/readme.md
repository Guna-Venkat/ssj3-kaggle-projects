# 🍿 IMDB - Popcorn Sentiment Classification (NLP)

<p align="center">
  <a href="https://www.kaggle.com/competitions/word2vec-nlp-tutorial">
    <img src="https://img.shields.io/badge/Kaggle-Popcorn_Review_Classification-orange?style=for-the-badge&logo=kaggle&logoColor=white" alt="Kaggle Popcorn Sentiment"/>
  </a>
</p>

Welcome to **Popcorn NLP Sentiment Classification**, a classic binary text classification problem from the **IMDB Movie Review** dataset.  
This repository contains an end-to-end exploration using both traditional NLP and modern deep learning pipelines — part of the **SSJ3-ML-Journey**.

This competition is based on the **Word2Vec NLP tutorial** developed at Kaggle, which explores semantic relationships in text using efficient deep-learning-inspired techniques. The dataset and challenge are grounded in real-world movie review sentiment analysis.

---

## 🎯 Objective

Predict whether a movie review expresses **positive** or **negative** sentiment based on its text.

You’ll classify reviews using:

- 🧠 **Traditional models**: TF-IDF + Naive Bayes / SVM / Logistic Regression  
- 🤖 **Optional upgrade**: Transformer-based models (RoBERTa / BERT)

---

## 🧾 Tutorial Background

This competition demonstrates how **Word2Vec** can be used for sentiment classification, using an IMDB movie review dataset.  
While deep learning models such as RNNs or DNNs can capture rich patterns in text, **Word2Vec** provides a **computationally efficient** way to model word semantics. This tutorial helps beginners understand:

- Basic NLP preprocessing (Part 1)
- Word2Vec training & usage (Parts 2 & 3)
- Exploratory rather than prescriptive modeling

---

## 📁 Dataset Overview

The dataset contains **100,000 IMDB reviews**, split into labeled, unlabeled, and test sets.

| File                  | Description                              |
|-----------------------|------------------------------------------|
| `labeledTrainData.tsv`| 25,000 labeled training samples (`id`, `review`, `sentiment`) |
| `testData.tsv`        | 25,000 test samples (`id`, `review`) — to predict sentiment |
| `unlabeledTrainData.tsv` | 50,000 additional unlabeled reviews for Word2Vec training |
| `sampleSubmission.csv`| Submission template for test predictions |

### 📌 Sentiment Labels
- `0` = Negative review (IMDB rating < 5)
- `1` = Positive review (IMDB rating ≥ 7)

> Each review is multi-paragraph, and **no movie appears in both train and test sets**.

---

## 📊 Evaluation Metric

- 🧮 **AUC (Area Under ROC Curve)** — model performance is judged based on this metric.

---

## 🔍 Workflow

```
1. Load Data & Initial Inspection
2. Text Preprocessing (cleaning, stopwords, stemming)
3. Feature Extraction (TF-IDF, n-grams)
4. Traditional ML Modeling (NB, SVM, LR)
5. Evaluation & Visualization
6. Advanced Experiments (BERT/RoBERTa - optional)
```

## 🧪 Observations

| Technique               | Accuracy | Notes                                      |
|-------------------------|----------|--------------------------------------------|
| TF-IDF + Naive Bayes    | ~0.87    | Simple, fast, and interpretable            |
| TF-IDF + SVM            | ~0.90    | Better generalization                      |
| TF-IDF + LogisticR      | ~0.89    | Stable baseline                            |
| Transformers (BERT)     | ~0.94+   | Requires more compute, better performance  |

## 📦 Project Structure

```
popcorn_sentiment_nlp/
├── data/                        # Raw and processed dataset
│   ├── train.tsv
│   ├── test.tsv
│   ├── sampleSubmission.csv
├── notebooks/
│   ├── 01_clean_text_lemmatize.ipynb
│   ├── 02_train_word2vec.ipynb
│   ├── 03_glove_fasttext_compare.ipynb 
│   ├── 04_tfidf_ml_models.ipynb
│   ├── 05_nn_on_embeddings.ipynb
│   ├── 06_rnn_lstm.ipynb
│   ├── 07_bert_transformer.ipynb
│   ├── 08_error_analysis.ipynb
├── models/                      # Trained model files (.pkl)
├── outputs/                     # Submission files, reports
├── visualizations/              # Wordclouds, confusion matrices
├── requirements.txt
└── README.md
```

## 🧼 Text Cleaning & Preprocessing

- Remove HTML tags, punctuation  
- Lowercase conversion  
- Stopword removal (NLTK)  
- *Optional:* Stemming / Lemmatization  

```python
from nltk.corpus import stopwords
from bs4 import BeautifulSoup
import re

def clean_text(text):
    text = BeautifulSoup(text, "html.parser").get_text()
    text = re.sub("[^a-zA-Z]", " ", text).lower()
    text = " ".join([w for w in text.split() if w not in stopwords.words("english")])
    return text
```

# 🔤 Feature Engineering

| Vectorizer        | Settings                          |
|-------------------|-----------------------------------|
| TF-IDF            | `max_features=10000`, `ngram=(1,2)` |
| CountVectorizer   | *(optional)*                      |

```python
from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer(ngram_range=(1,2), max_features=10000)
X = vectorizer.fit_transform(cleaned_reviews)
```

# 🤖 Machine Learning Models

| Model              | Input     | Remarks                          |
|-------------------|-----------|----------------------------------|
| Naive Bayes       | TF-IDF    | Fast, interpretable              |
| Logistic Regression | TF-IDF  | Good for high-dim sparse         |
| SVM (Linear)      | TF-IDF    | Excellent decision boundary      |
| BERT / RoBERTa    | Raw text  | SOTA performance (optional)      |

---

# 📈 Evaluation Metrics

- Accuracy  
- F1 Score  
- Confusion Matrix  
- Training time  
- Misclassified examples  

---

# 📊 Visualizations

- Wordclouds (positive vs negative)  
- Confusion matrix  
- N-gram bar charts  
- Incorrect predictions samples  

# 🧪 Optional Upgrades

- 🔁 Try **Stemming** vs **Lemmatization**
- 🔎 Interpret model coefficients (top positive/negative words)
- 📚 Upgrade to **BERT** or **RoBERTa** (via HuggingFace)
- ⚖️ Use **class weights** or **balancing** for imbalanced datasets
- 🧪 Use **GridSearchCV** for hyperparameter tuning

---

# 🧠 Learnings

- Text cleaning has a **huge impact** on model performance  
- **TF-IDF** is still powerful in classical ML pipelines  
- **BERT** offers massive performance gain at compute cost  
- Importance of **visualizing** and **interpreting** misclassifications  

## ✍️ Author

- **Name**: Guna Venkat Doddi  
- **Project**: Part of `SSJ3-Kaggle-Projects` repository  
- **Contact**: [![GitHub - Guna Venkat Doddi](https://img.shields.io/badge/GitHub-Guna--Venkat--Doddi-black?logo=github&style=flat-square)](https://github.com/Guna-Venkat)

---