"""
Fake News Detector — Complete ML Pipeline
=========================================
Dataset : Kaggle "News Articles" dataset (True.csv + Fake.csv)
          OR any CSV with columns: title, text, label (0=Real, 1=Fake)

Quick start
-----------
1.  pip install -r requirements.txt
2.  python -c "import nltk; nltk.download('all')"
3.  Place True.csv and Fake.csv inside the data/ folder
4.  python fake_news_detector.py
"""

import os
import re
import string
import pickle
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression          # quick baseline
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    accuracy_score, classification_report,
    confusion_matrix, roc_auc_score, roc_curve,
    ConfusionMatrixDisplay,
)
from sklearn.pipeline import Pipeline

warnings.filterwarnings("ignore")

# --------------------------------------------------------------------------- #
# NLTK downloads (safe to call multiple times)
# --------------------------------------------------------------------------- #
for pkg in ("stopwords", "punkt", "punkt_tab", "wordnet", "omw-1.4"):
    nltk.download(pkg, quiet=True)

os.makedirs("outputs", exist_ok=True)
os.makedirs("models", exist_ok=True)


# ============================================================
# 1. DATA LOADING
# ============================================================

def load_data(path: str = "data/WELFake_Dataset.csv") -> pd.DataFrame:
    print("📂  Loading data …")
    df = pd.read_csv(path)
    df = df.rename(columns={"label": "label"})
    df = df.dropna(subset=["text"]).reset_index(drop=True)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    real_count = (df["label"] == 1).sum()
    fake_count = (df["label"] == 0).sum()
    print(f"✅  {len(df):,} articles loaded  ({real_count:,} real · {fake_count:,} fake)")
    return df         


# ============================================================
# 2. EXPLORATORY DATA ANALYSIS
# ============================================================

def run_eda(df: pd.DataFrame) -> pd.DataFrame:
    """Print statistics and save distribution charts."""
    print("\n📊  Dataset overview")
    print(f"    Shape   : {df.shape}")
    print(f"    Columns : {list(df.columns)}")
    print(f"\n    Label distribution:\n{df['label'].value_counts().to_string()}")

    missing = df.isnull().sum()
    if missing.any():
        print(f"\n    Missing values:\n{missing[missing > 0].to_string()}")

    # Compute text length
    df = df.copy()
    df["text_length"] = (df.get("text", pd.Series(dtype=str))
                           .fillna("").apply(len))

    fig, axes = plt.subplots(1, 2, figsize=(13, 4))

    # — class distribution —
    counts = df["label"].value_counts().sort_index()
    bars = axes[0].bar(
        ["Real (0)", "Fake (1)"], counts.values,
        color=["#2ecc71", "#e74c3c"], edgecolor="white", alpha=0.88,
    )
    for bar, v in zip(bars, counts.values):
        axes[0].text(bar.get_x() + bar.get_width() / 2,
                     v + counts.max() * 0.01,
                     f"{v:,}", ha="center", fontsize=11, fontweight="bold")
    axes[0].set_title("Class Distribution", fontsize=13, fontweight="bold", pad=10)
    axes[0].set_ylabel("Article count")
    axes[0].spines[["top", "right"]].set_visible(False)

    # — text length histogram —
    bins = 60
    axes[1].hist(df[df["label"] == 0]["text_length"],
                 bins=bins, alpha=0.6, color="#2ecc71", label="Real")
    axes[1].hist(df[df["label"] == 1]["text_length"],
                 bins=bins, alpha=0.6, color="#e74c3c", label="Fake")
    axes[1].set_title("Text Length Distribution", fontsize=13,
                       fontweight="bold", pad=10)
    axes[1].set_xlabel("Character count")
    axes[1].set_ylabel("Frequency")
    axes[1].legend(frameon=False)
    axes[1].spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    plt.savefig("outputs/eda_distribution.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("    Chart saved → outputs/eda_distribution.png")
    return df


# ============================================================
# 3. TEXT PREPROCESSING
# ============================================================

class TextPreprocessor:
    """Clean, normalise, and lemmatise news text."""

    def __init__(self):
        self.stop_words = set(stopwords.words("english"))
        self.lemmatizer = WordNetLemmatizer()

    def clean(self, text: str) -> str:
        if not isinstance(text, str) or not text.strip():
            return ""

        text = text.lower()
        text = re.sub(r"https?://\S+|www\.\S+", " ", text)   # URLs
        text = re.sub(r"<[^>]+>", " ", text)                  # HTML tags
        text = re.sub(r"\[.*?\]|\(.*?\)", " ", text)          # brackets
        text = re.sub(r"[%s]" % re.escape(string.punctuation), " ", text)
        text = re.sub(r"\d+", " ", text)                       # digits
        text = re.sub(r"\s+", " ", text).strip()

        tokens = word_tokenize(text)
        tokens = [
            self.lemmatizer.lemmatize(t)
            for t in tokens
            if t not in self.stop_words and len(t) > 2
        ]
        return " ".join(tokens)

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        print("\n🔧  Preprocessing text …")
        df = df.copy()

        # Combine title + text for richer signal
        title_col = df.get("title", pd.Series([""] * len(df))).fillna("")
        text_col  = df.get("text",  pd.Series([""] * len(df))).fillna("")
        df["content"] = title_col + " " + text_col

        df["cleaned"] = df["content"].apply(self.clean)

        # Drop empty rows
        before = len(df)
        df = df[df["cleaned"].str.len() > 10].reset_index(drop=True)
        print(f"✅  {len(df):,} articles remain  "
              f"({before - len(df):,} dropped after cleaning)")
        return df


# ============================================================
# 4. FEATURE EXTRACTION — TF-IDF
# ============================================================

def build_tfidf(max_features: int = 50_000,
                ngram_range: tuple = (1, 2)) -> TfidfVectorizer:
    return TfidfVectorizer(
        max_features=max_features,
        ngram_range=ngram_range,
        min_df=2,
        max_df=0.95,
        sublinear_tf=True,          # log(1 + tf) — shrinks high-freq terms
        strip_accents="unicode",
        analyzer="word",
        token_pattern=r"\b[a-zA-Z]{3,}\b",
    )


# ============================================================
# 5. MODEL TRAINING — RANDOM FOREST + BASELINE
# ============================================================

def train_random_forest(X_train, y_train,
                        n_estimators: int = 200) -> RandomForestClassifier:
    print(f"\n🌲  Training Random Forest  (n_estimators={n_estimators}) …")
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        min_samples_split=5,
        min_samples_leaf=2,
        max_features="sqrt",
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    print("✅  Training complete!")
    return model


def train_baseline(X_train, y_train) -> LogisticRegression:
    """Logistic Regression for a quick comparison baseline."""
    print("\n📊  Training Logistic Regression baseline …")
    lr = LogisticRegression(max_iter=1_000, class_weight="balanced",
                            random_state=42, n_jobs=-1)
    lr.fit(X_train, y_train)
    return lr


# ============================================================
# 6. EVALUATION
# ============================================================

def evaluate(model, tfidf: TfidfVectorizer,
             X_test_raw, X_test_tfidf, y_test,
             model_name: str = "Random Forest") -> dict:
    """Full evaluation: metrics + 3 charts."""
    print(f"\n📈  Evaluating {model_name} …")

    y_pred = model.predict(X_test_tfidf)
    y_prob = model.predict_proba(X_test_tfidf)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)

    print(f"\n{'─'*50}")
    print(f"  Model    : {model_name}")
    print(f"  Accuracy : {acc:.4f}  ({acc*100:.2f}%)")
    print(f"  ROC-AUC  : {auc:.4f}")
    print(f"\n{classification_report(y_test, y_pred, target_names=['Real','Fake'])}")
    print(f"{'─'*50}")

    # ── Charts ──────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(f"{model_name} — Evaluation", fontsize=14,
                 fontweight="bold", y=1.02)

    # 1. Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Real", "Fake"],
                yticklabels=["Real", "Fake"],
                ax=axes[0], linewidths=0.5)
    axes[0].set_title("Confusion Matrix")
    axes[0].set_ylabel("Actual")
    axes[0].set_xlabel("Predicted")

    # 2. ROC curve
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    axes[1].plot(fpr, tpr, lw=2, color="#3498db",
                 label=f"AUC = {auc:.3f}")
    axes[1].plot([0, 1], [0, 1], "k--", lw=1)
    axes[1].fill_between(fpr, tpr, alpha=0.10, color="#3498db")
    axes[1].set_title("ROC Curve")
    axes[1].set_xlabel("False Positive Rate")
    axes[1].set_ylabel("True Positive Rate")
    axes[1].legend(frameon=False)
    axes[1].spines[["top", "right"]].set_visible(False)

    # 3. Top-20 feature importances (RF only)
    if hasattr(model, "feature_importances_"):
        feat_names = np.array(tfidf.get_feature_names_out())
        importances = model.feature_importances_
        top_idx = np.argsort(importances)[-20:][::-1]
        axes[2].barh(range(20), importances[top_idx],
                     color="#9b59b6", alpha=0.8)
        axes[2].set_yticks(range(20))
        axes[2].set_yticklabels(feat_names[top_idx], fontsize=9)
        axes[2].invert_yaxis()
        axes[2].set_title("Top 20 Feature Importances")
        axes[2].set_xlabel("Importance score")
        axes[2].spines[["top", "right"]].set_visible(False)
    else:
        axes[2].axis("off")
        axes[2].text(0.5, 0.5, "N/A for this model",
                     ha="center", va="center", transform=axes[2].transAxes)

    plt.tight_layout()
    slug = model_name.lower().replace(" ", "_")
    out_path = f"outputs/{slug}_evaluation.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"    Chart saved → {out_path}")

    return {"model": model_name, "accuracy": acc, "auc": auc}


# ============================================================
# 7. WORD CLOUDS
# ============================================================

def generate_wordclouds(df: pd.DataFrame) -> None:
    try:
        from wordcloud import WordCloud
    except ImportError:
        print("⚠️  wordcloud not installed — skipping word clouds.")
        return

    print("\n☁️  Generating word clouds …")
    real_text = " ".join(df[df["label"] == 0]["cleaned"])
    fake_text = " ".join(df[df["label"] == 1]["cleaned"])

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    for ax, text, cmap, title, color in [
        (axes[0], real_text, "Greens",
         "Real News — Common Words", "#27ae60"),
        (axes[1], fake_text, "Reds",
         "Fake News — Common Words", "#c0392b"),
    ]:
        wc = WordCloud(width=800, height=400, background_color="white",
                       colormap=cmap, max_words=150, random_state=42)
        wc.generate(text)
        ax.imshow(wc, interpolation="bilinear")
        ax.set_title(title, fontsize=13, fontweight="bold", color=color)
        ax.axis("off")

    plt.tight_layout()
    plt.savefig("outputs/wordclouds.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("    Chart saved → outputs/wordclouds.png")


# ============================================================
# 8. SAVE / LOAD
# ============================================================

def save_artefacts(model, tfidf: TfidfVectorizer,
                   preprocessor: TextPreprocessor,
                   path: str = "models/") -> None:
    os.makedirs(path, exist_ok=True)
    with open(f"{path}rf_model.pkl", "wb") as f:
        pickle.dump(model, f)
    with open(f"{path}tfidf.pkl", "wb") as f:
        pickle.dump(tfidf, f)
    with open(f"{path}preprocessor.pkl", "wb") as f:
        pickle.dump(preprocessor, f)
    print(f"\n💾  Artefacts saved → {path}")


def load_artefacts(path: str = "models/"):
    with open(f"{path}rf_model.pkl", "rb") as f:
        model = pickle.load(f)
    with open(f"{path}tfidf.pkl", "rb") as f:
        tfidf = pickle.load(f)
    with open(f"{path}preprocessor.pkl", "rb") as f:
        preprocessor = pickle.load(f)
    return model, tfidf, preprocessor


# ============================================================
# 9. SINGLE-ARTICLE PREDICTION
# ============================================================

def predict(text: str,
            model, tfidf: TfidfVectorizer,
            preprocessor: TextPreprocessor) -> dict:
    cleaned   = preprocessor.clean(text)
    features  = tfidf.transform([cleaned])
    pred      = model.predict(features)[0]
    proba     = model.predict_proba(features)[0]
    label     = "FAKE" if pred == 1 else "REAL"
    return {
        "label":       label,
        "confidence":  round(float(proba[pred]) * 100, 2),
        "prob_real":   round(float(proba[0]) * 100, 2),
        "prob_fake":   round(float(proba[1]) * 100, 2),
    }


# ============================================================
# MAIN PIPELINE
# ============================================================

def main():
    print("=" * 56)
    print("  FAKE NEWS DETECTOR — ML Pipeline")
    print("=" * 56)

    # 1. Load
    df = load_data()

    # 2. EDA
    df = run_eda(df)

    # 3. Preprocess
    preprocessor = TextPreprocessor()
    df = preprocessor.fit_transform(df)

    # 4. Word clouds
    generate_wordclouds(df)

    # 5. Train / test split  (stratified)
    X, y = df["cleaned"], df["label"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"\n📦  Split : {len(X_train):,} train · {len(X_test):,} test")

    # 6. TF-IDF
    print(f"\n🔢  Vectorising with TF-IDF (max=50 000, ngram=(1,2)) …")
    tfidf = build_tfidf()
    X_train_v = tfidf.fit_transform(X_train)
    X_test_v  = tfidf.transform(X_test)
    print(f"✅  Feature matrix : {X_train_v.shape}")

    # 7. Baseline (LR) — fast sanity check
    lr = train_baseline(X_train_v, y_train)
    baseline_res = evaluate(lr, tfidf, X_test, X_test_v, y_test,
                            model_name="Logistic Regression (Baseline)")

    # 8. Main model (RF)
    rf = train_random_forest(X_train_v, y_train, n_estimators=200)
    rf_res = evaluate(rf, tfidf, X_test, X_test_v, y_test,
                      model_name="Random Forest")

    # 9. Compare
    print("\n🏆  Model comparison")
    print(f"    {'Model':<35} {'Accuracy':>10}  {'AUC':>8}")
    print(f"    {'─'*55}")
    for r in [baseline_res, rf_res]:
        print(f"    {r['model']:<35} {r['accuracy']:>10.4f}  {r['auc']:>8.4f}")

    # 10. Save best model (RF)
    save_artefacts(rf, tfidf, preprocessor)

    # 11. Demo prediction
    demo_articles = [
        ("REAL DEMO",
         "NASA scientists confirm that the James Webb Space Telescope has "
         "captured images of galaxies forming just 300 million years after "
         "the Big Bang, offering new insights into the early universe."),
        ("FAKE DEMO",
         "SHOCKING: Government admits chemtrails are real and contain "
         "mind-control agents. Whistleblower reveals secret programme to "
         "keep population docile. Share before this gets deleted!!!"),
    ]

    print("\n🔍  Demo predictions")
    print("    " + "─" * 52)
    for tag, article in demo_articles:
        result = predict(article, rf, tfidf, preprocessor)
        icon = "🟢" if result["label"] == "REAL" else "🔴"
        print(f"    [{tag}]")
        print(f"    {icon} {result['label']}  "
              f"(confidence {result['confidence']:.1f}%  |  "
              f"real {result['prob_real']:.1f}%  fake {result['prob_fake']:.1f}%)")
        print()

    print("✅  Pipeline complete!  Check outputs/ for charts.")
    return rf, tfidf, preprocessor


if __name__ == "__main__":
    main()
