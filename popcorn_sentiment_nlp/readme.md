# 🍿 IMDB - Popcorn Sentiment Classification (NLP)

<p align="center">
  <a href="https://www.kaggle.com/competitions/word2vec-nlp-tutorial">
    <img src="https://img.shields.io/badge/Kaggle-Popcorn_Review_Classification-orange?style=for-the-badge&logo=kaggle&logoColor=white" alt="Kaggle Popcorn Sentiment"/>
  </a>
</p>

Welcome to **Popcorn NLP Sentiment Classification**, a classic binary text classification problem from the **IMDB Movie Review** dataset.  
This repository contains an end-to-end exploration using both traditional NLP and modern deep learning pipelines — part of the **SSJ3-ML-Journey**.

---

## 🎯 Objective

Predict whether a movie review expresses **positive** or **negative** sentiment based on its text.

You’ll classify reviews using:

- 🧠 **Traditional models**: TF-IDF + Naive Bayes / SVM / Logistic Regression  
- 🤖 **Optional upgrade**: Transformer-based models (RoBERTa / BERT)

---

## 📁 Dataset Overview

| File                  | Description                              |
|-----------------------|------------------------------------------|
| `train.tsv`           | Contains `id`, `review`, and `sentiment` |
| `test.tsv`            | Contains `id` and `review`               |
| `sampleSubmission.csv`| Template for final predictions            |

- Reviews are raw HTML/text strings from IMDB movie reviews.
- **Labels**:  
  - `0` = Negative  
  - `1` = Positive

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
│   ├── 01_eda_preprocessing.ipynb
│   ├── 02_tfidf_logreg_nb_svm.ipynb
│   ├── 03_transformer_finetune.ipynb  (optional)
│   ├── 04_visualizations.ipynb
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