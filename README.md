# 🔍 Fake News Detector

Text-based fake news classifier using **NLP + TF-IDF + Random Forest**.

---

## 📁 Project Structure

```
fake_news_detector/
├── data/
│   ├── True.csv          ← download from Kaggle (see below)
│   └── Fake.csv
├── models/               ← auto-created after training
│   ├── rf_model.pkl
│   ├── tfidf.pkl
│   └── preprocessor.pkl
├── outputs/              ← charts saved here
│   ├── eda_distribution.png
│   ├── wordclouds.png
│   └── random_forest_evaluation.png
├── fake_news_detector.py ← ML pipeline (train + evaluate)
├── detector.py                ← Streamlit demo app
├── requirements.txt
└── README.md
```

---

## 🚀 Quick Start

### 1 — Install dependencies
```bash
pip install -r requirements.txt
python -c "import nltk; nltk.download('all')"
```

### 2 — Get the dataset
Download the **ISOT Fake News Dataset** (most popular, ~44 k articles):

| Source | Link |
|--------|------|
| Kaggle (recommended) | [kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset) |
| University of Victoria | [uvic.ca/ecs/ece/isot/datasets/fake-news](https://www.uvic.ca/ecs/ece/isot/datasets/fake-news/index.php) |

Place `True.csv` and `Fake.csv` inside the `data/` folder.

### 3 — Train the model
```bash
python fake_news_detector.py
```
This will:
- Run EDA and save charts to `outputs/`
- Generate word clouds
- Train a Logistic Regression baseline + Random Forest
- Print a full classification report
- Save model artefacts to `models/`

### 4 — Launch the demo app
```bash
streamlit run detector.py
```
Open http://localhost:8501 in your browser.

---

## 🧠 Technical Overview

### Pipeline

```
Raw Text
  ↓
Combine title + body
  ↓
Clean  →  lowercase, remove URLs/HTML/punctuation/digits
  ↓
Tokenise  (NLTK word_tokenize)
  ↓
Remove stopwords  +  lemmatise  (WordNetLemmatizer)
  ↓
TF-IDF Vectorisation
  │  max_features = 50 000
  │  ngram_range  = (1, 2)   ← unigrams + bigrams
  │  sublinear_tf = True     ← log(1 + tf)
  │  min_df = 2, max_df = 0.95
  ↓
Random Forest Classifier
  │  n_estimators  = 200
  │  max_features  = "sqrt"
  │  class_weight  = "balanced"
  │  random_state  = 42
  ↓
Prediction  +  probability score
```

### Why these choices?

| Choice | Reason |
|--------|--------|
| TF-IDF over Bag-of-Words | Down-weights common words automatically |
| Bigrams | Captures "breaking news", "fake news", "deep state", etc. |
| `sublinear_tf` | Reduces dominance of very frequent terms |
| `balanced` weights | Handles class imbalance gracefully |
| `sqrt` max_features | Standard RF de-correlation trick |
| Lemmatisation over stemming | Produces real words → better readability |

### Expected performance (ISOT dataset)

| Model | Accuracy | AUC |
|-------|----------|-----|
| Logistic Regression (baseline) | ~98% | ~0.998 |
| **Random Forest** | **~99%** | **~0.999** |

> The ISOT dataset is relatively clean. Real-world performance on diverse
> social-media posts will be lower.

---

## 📊 Outputs

| File | Description |
|------|-------------|
| `outputs/eda_distribution.png` | Class balance + text-length histograms |
| `outputs/wordclouds.png` | Most common words in real vs. fake articles |
| `outputs/logistic_regression_(baseline)_evaluation.png` | Baseline metrics |
| `outputs/random_forest_evaluation.png` | RF confusion matrix, ROC, top features |

---

## 🔧 Customisation

### Change the dataset
If your CSV has different column names, edit `load_data()` in
`fake_news_detector.py`. The only hard requirement is a `label` column
(0 = real, 1 = fake) and at least one text column.

### Tune the model
```python
# In main() — try more trees or a depth limit
rf = train_random_forest(X_train_v, y_train, n_estimators=300)
```

### Add more models
```python
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC

nb  = MultinomialNB().fit(X_train_v, y_train)
svc = LinearSVC(class_weight="balanced").fit(X_train_v, y_train)
```

---

## 📚 Concepts Covered

- **NLP pre-processing**: tokenisation, stopword removal, lemmatisation
- **TF-IDF**: term frequency–inverse document frequency weighting
- **Random Forest**: ensemble of decision trees (bagging)
- **Evaluation**: accuracy, precision, recall, F1, ROC-AUC
- **Confusion matrix**: TP / TN / FP / FN breakdown
- **Class imbalance**: `class_weight="balanced"` strategy
- **Model persistence**: pickle for saving/loading artefacts

---

## ⚠️ Limitations & Ethics

1. **Dataset bias** — model reflects the specific news sources in training data.
2. **Satire** — intentionally false-seeming content may be misclassified.
3. **Domain shift** — performance degrades on social-media posts or non-English text.
4. **Not a fact-checker** — always verify with Reuters, AP News, or Snopes.
5. **Dual use** — understanding how fake news is detected also reveals how to evade detection.

## Author
Ankita Dey(AIML-A6/May-9814)
Developed as an AI/ML Minor Project — AI InternsElite May2026
