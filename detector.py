import os
import re
import string
import pickle
import warnings
import time
from PIL import Image

import streamlit as st
import nltk

warnings.filterwarnings("ignore")

for pkg in ("stopwords", "punkt", "punkt_tab", "wordnet", "omw-1.4"):
    nltk.download(pkg, quiet=True)

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

st.set_page_config(
    page_title="Fake News Detector",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
  .hero { text-align:center; padding:36px 0 20px; }
  .hero h1 { font-size:2.6rem; font-weight:800; letter-spacing:-1px; margin:0; }
  .hero p  { font-size:1.05rem; color:#888; margin-top:8px; }

  .stat-row { display:grid; grid-template-columns:repeat(4,1fr); gap:14px; margin:24px 0; }
  .stat-card { background:linear-gradient(135deg,#1e1e2f,#2a2a3d);
               border:1px solid #3a3a55; border-radius:14px;
               padding:18px 14px; text-align:center; }
  .stat-val  { font-size:2rem; font-weight:700; color:#7c83fd; }
  .stat-lbl  { font-size:.75rem; color:#aaa; margin-top:4px; text-transform:uppercase;
               letter-spacing:.06em; }

  .result-banner { padding:24px 28px; border-radius:14px; margin:18px 0; text-align:center; }
  .result-banner h2 { margin:0 0 6px; font-size:2rem; }
  .result-banner p  { margin:0; font-size:1rem; color:#222222; opacity:1; font-weight:600 }
  .real-banner { background:#d4f5e2; border:2px solid #27ae60; }
  .real-banner h2 { color:#155724; }
  .fake-banner { background:#fde8e8; border:2px solid #e74c3c; }
  .fake-banner h2 { color:#721c24; }

  .prob-row { display:flex; gap:12px; margin:10px 0; }
  .pill { flex:1; padding:14px; border-radius:10px; text-align:center; }
  .pill-real { background:#d4f5e2; }
  .pill-fake { background:#fde8e8; }
  .pill-val  { font-size:1.8rem; font-weight:700; color:#111111}
  .pill-lbl  { font-size:.75rem; color:#222222; margin-top:2px; }

  .gauge-wrap  { display:flex; align-items:center; gap:14px; margin:18px 0; }
  .gauge-bar   { flex:1; height:22px; border-radius:11px; background:#f0f0f0; overflow:hidden; }
  .gauge-fill  { height:100%; border-radius:11px; transition:width .6s ease; }
  .gauge-label { font-size:18px; font-weight:700; min-width:60px; text-align:right; }

  .chip-list { display:flex; flex-wrap:wrap; gap:6px; margin:10px 0; }
  .chip { background:#eef; color:#334; padding:3px 10px; border-radius:20px; font-size:.8rem; }

  .section-head { font-size:1.2rem; font-weight:700; border-left:4px solid #7c83fd;
                  padding-left:12px; margin:24px 0 14px; }

  .compare-table { width:100%; border-collapse:collapse; margin-top:10px; }
  .compare-table th { background:#7c83fd22; color:#7c83fd; padding:10px 14px; text-align:left;
                      font-size:.8rem; text-transform:uppercase; letter-spacing:.05em; }
  .compare-table td { padding:10px 14px; border-top:1px solid #2a2a3d; font-size:.95rem; }
  .compare-table tr:hover td { background:#ffffff08; }
  .winner { color:#2ecc71; font-weight:600; }

  div[data-testid="stTabs"] button { font-size:.95rem; }
</style>
""", unsafe_allow_html=True)


@st.cache_resource(show_spinner=False)
def get_preprocessor():
    stop_words = set(stopwords.words("english"))
    lemmatizer = WordNetLemmatizer()
    def clean(text: str) -> str:
        if not isinstance(text, str) or not text.strip():
            return ""
        text = text.lower()
        text = re.sub(r"https?://\S+|www\.\S+", " ", text)
        text = re.sub(r"<[^>]+>", " ", text)
        text = re.sub(r"\[.*?\]|\(.*?\)", " ", text)
        text = re.sub(r"[%s]" % re.escape(string.punctuation), " ", text)
        text = re.sub(r"\d+", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        tokens = word_tokenize(text)
        tokens = [lemmatizer.lemmatize(t) for t in tokens
                  if t not in stop_words and len(t) > 2]
        return " ".join(tokens)
    return clean

@st.cache_resource(show_spinner=False)
def load_model():
    if not os.path.exists("models/rf_model.pkl"):
        return None, None
    with open("models/rf_model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("models/tfidf.pkl", "rb") as f:
        tfidf = pickle.load(f)
    return model, tfidf

def predict(text, model, tfidf, clean_fn):
    cleaned  = clean_fn(text)
    features = tfidf.transform([cleaned])
    pred     = model.predict(features)[0]
    proba    = model.predict_proba(features)[0]
    return {
        "label":      "FAKE" if pred == 1 else "REAL",
        "confidence": float(proba[pred]) * 100,
        "prob_real":  float(proba[0]) * 100,
        "prob_fake":  float(proba[1]) * 100,
        "cleaned":    cleaned,
    }

def load_img(path):
    if os.path.exists(path):
        return Image.open(path)
    return None


model, tfidf = load_model()
clean_fn     = get_preprocessor()


with st.sidebar:
    
    st.markdown("### 📚 Dataset")
    st.markdown(
        "Trained on the [WeLFake Dataset](https://www.kaggle.com/datasets/"
        "saurabhshahane/fake-news-classification) — 72,095 articles."
    )
    st.markdown("---")
    st.markdown("### 🔬 How it works")
    st.markdown(
        "1. **Preprocess** — clean and lemmatise text\n"
        "2. **TF-IDF** — convert to weighted term frequencies\n"
        "3. **Random Forest** — ensemble of 200 decision trees\n"
        "4. **Probability** — vote-based confidence score"
    )
    st.markdown("---")
    st.caption("⚠️ For educational purposes only.")


st.markdown("""
<div class="hero">
  <h1>🔍 Fake News Detector</h1>
  <p>NLP · TF-IDF · Random Forest · 72,095 articles trained</p>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="stat-row">
  <div class="stat-card">
    <div class="stat-val">95%</div>
    <div class="stat-lbl">Accuracy</div>
  </div>
  <div class="stat-card">
    <div class="stat-val">0.99</div>
    <div class="stat-lbl">ROC-AUC</div>
  </div>
  <div class="stat-card">
    <div class="stat-val">72k</div>
    <div class="stat-lbl">Articles trained</div>
  </div>
  <div class="stat-card">
    <div class="stat-val">50k</div>
    <div class="stat-lbl">TF-IDF features</div>
  </div>
</div>
""", unsafe_allow_html=True)

st.markdown("---")

tab1, tab2, tab3 = st.tabs(["🔍 Detect", "📊 Model Performance", "☁️ Dataset Insights"])


with tab1:
    if model is None:
        st.error("No trained model found. Run `python fake_news_detector.py` first.")
        st.stop()

    EXAMPLES = {
        "Select an example …": "",
        "🟢 Real — Federal Reserve": (
            "The Federal Reserve raised interest rates by 25 basis points on Wednesday, "
            "citing persistent inflation concerns. Fed Chair Jerome Powell said policymakers "
            "remain committed to bringing inflation back to the 2% target, though he "
            "acknowledged the risk of slowing economic growth."
        ),
        "🔴 Fake — Chemtrail conspiracy": (
            "BREAKING: Government whistleblower reveals that chemtrails contain secret "
            "mind-control chemicals sprayed on the population. Deep-state documents LEAKED "
            "showing global elites plan to use this to remain in power. Share before this "
            "gets deleted by the censors!!!"
        ),
        "🟢 Real — Climate report": (
            "A new report by the Intergovernmental Panel on Climate Change warns that global "
            "average temperatures could rise by 1.5°C above pre-industrial levels by the "
            "early 2030s if current trends continue. The report calls for immediate cuts "
            "in greenhouse-gas emissions."
        ),
        "🔴 Fake — Election fraud": (
            "MILLIONS of fake ballots discovered in warehouse! Election was STOLEN — sources "
            "with inside knowledge say up to 20 million fraudulent votes were cast. Mainstream "
            "media is hiding the TRUTH. Retweet so everyone can see what's really happening!!!"
        ),
    }

    col1, col2 = st.columns([2, 1])

    with col1:
        example_choice = st.selectbox("Load an example", list(EXAMPLES.keys()))
        default_text   = EXAMPLES[example_choice]
        user_text = st.text_area(
            "📰 Article text",
            value=default_text,
            height=200,
            placeholder="Paste headline or full article here …",
        )
        analyse_btn = st.button("🔍 Analyse", type="primary",
                                disabled=not user_text.strip())

    with col2:
        st.markdown("### 📌 Tips")
        st.markdown(
            "- Longer text gives **better results**\n"
            "- Include the **headline** for extra signal\n"
            "- Trained on English text only\n"
            "- Satire may be misclassified\n"
            "- Always verify with a trusted fact-checker"
        )

    if analyse_btn and user_text.strip():
        with st.spinner("Analysing …"):
            time.sleep(0.4)
            result = predict(user_text, model, tfidf, clean_fn)

        label      = result["label"]
        confidence = result["confidence"]
        prob_real  = result["prob_real"]
        prob_fake  = result["prob_fake"]
        is_fake    = label == "FAKE"
        banner_cls = "fake-banner" if is_fake else "real-banner"
        icon       = "🔴" if is_fake else "🟢"
        verdict    = "This article appears to be FAKE" if is_fake else "This article appears to be REAL"
        bar_color  = "#e74c3c" if is_fake else "#27ae60"

        st.markdown(f"""
        <div class="result-banner {banner_cls}">
          <h2>{icon} {label}</h2>
          <p>{verdict} — model confidence: <strong>{confidence:.1f}%</strong></p>
        </div>""", unsafe_allow_html=True)

        st.markdown(f"""
        <div class="prob-row">
          <div class="pill pill-real">
            <div class="pill-val">{prob_real:.1f}%</div>
            <div class="pill-lbl">Probability REAL</div>
          </div>
          <div class="pill pill-fake">
            <div class="pill-val">{prob_fake:.1f}%</div>
            <div class="pill-lbl">Probability FAKE</div>
          </div>
        </div>""", unsafe_allow_html=True)

        st.markdown(f"""
        <div class="gauge-wrap">
          <span style="font-size:.85rem;color:#777;min-width:80px">Confidence</span>
          <div class="gauge-bar">
            <div class="gauge-fill" style="width:{confidence:.1f}%;background:{bar_color};"></div>
          </div>
          <span class="gauge-label" style="color:{bar_color}">{confidence:.1f}%</span>
        </div>""", unsafe_allow_html=True)

        with st.expander("🔬 See cleaned tokens (what the model sees)"):
            tokens = result["cleaned"].split()[:80]
            chips  = "".join(f'<span class="chip">{t}</span>' for t in tokens)
            ellipsis = "…" if len(result["cleaned"].split()) > 80 else ""
            st.markdown(
                f'<div class="chip-list">{chips}</div>'
                f'<p style="font-size:.8rem;color:#888">'
                f'{len(result["cleaned"].split())} tokens total{ellipsis}</p>',
                unsafe_allow_html=True,
            )

        st.markdown("---")
        st.caption(
            "⚠️ **Disclaimer**: For educational purposes only. "
            "Verify with [Reuters](https://www.reuters.com), "
            "[AP News](https://apnews.com), or [Snopes](https://www.snopes.com)."
        )


with tab2:
    st.markdown('<div class="section-head">Model Comparison</div>', unsafe_allow_html=True)

    st.markdown("""
    <table class="compare-table">
      <thead>
        <tr>
          <th>Model</th><th>Accuracy</th><th>ROC-AUC</th>
          <th>F1 — Real</th><th>F1 — Fake</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <td>Logistic Regression (Baseline)</td>
          <td>95.75%</td><td>0.9914</td><td>0.96</td><td>0.96</td>
        </tr>
        <tr>
          <td class="winner">✅ Random Forest (Final)</td>
          <td class="winner">95.00%</td>
          <td class="winner">0.9906</td>
          <td class="winner">0.95</td>
          <td class="winner">0.95</td>
        </tr>
      </tbody>
    </table>
    """, unsafe_allow_html=True)

    st.markdown('<div class="section-head">Random Forest Evaluation</div>', unsafe_allow_html=True)
    rf_chart = load_img("outputs/random_forest_evaluation.png")
    if rf_chart:
        st.image(rf_chart, use_container_width=True,
                 caption="Confusion Matrix · ROC Curve · Top 20 Feature Importances")
    else:
        st.warning("Run the pipeline first to generate charts.")

    st.markdown('<div class="section-head">Baseline — Logistic Regression</div>', unsafe_allow_html=True)
    lr_chart = load_img("outputs/logistic_regression_(baseline)_evaluation.png")
    if lr_chart:
        st.image(lr_chart, use_container_width=True,
                 caption="Logistic Regression baseline evaluation")
    else:
        st.info("Chart not found — check outputs/ folder.")


with tab3:
    st.markdown('<div class="section-head">Class Distribution & Text Length</div>', unsafe_allow_html=True)
    eda_chart = load_img("outputs/eda_distribution.png")
    if eda_chart:
        st.image(eda_chart, use_container_width=True,
                 caption="Left: class balance  ·  Right: article length distribution")
    else:
        st.warning("EDA chart not found. Run the pipeline first.")

    st.markdown('<div class="section-head">Word Clouds — Real vs Fake</div>', unsafe_allow_html=True)
    wc_chart = load_img("outputs/wordclouds.png")
    if wc_chart:
        st.image(wc_chart, use_container_width=True,
                 caption="Most frequent words after preprocessing — Green: Real · Red: Fake")
    else:
        st.warning("Word cloud not found. Run the pipeline first.")

    st.markdown("---")
    st.markdown("""
    **Dataset**: WeLFake (Weighted Ensemble of 4 datasets)
    | Split | Count |
    |-------|-------|
    | Total articles | 72,095 |
    | Real news | 37,067 (51.4%) |
    | Fake news | 35,028 (48.6%) |
    | Train set (80%) | 57,640 |
    | Test set (20%) | 14,410 |
    """)
