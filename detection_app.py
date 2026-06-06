# ============================================================
# app.py — Cyberbullying Detection (DistilBERT + SVM + LR + Cost Metrics + LIME)
# ============================================================

import os
import time
import pickle
import numpy as np
import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn.functional as F

# Optional: memory (RAM) metrics
try:
    import psutil
except ImportError:
    psutil = None

# ============================================================
# 1️⃣  Page setup
# ============================================================
st.set_page_config(page_title="Cyberbullying Detection", layout="wide")
st.title("🤖 Cyberbullying Detection System")
st.markdown(
    "Classify tweets as **Cyberbullying** or **Not Cyberbullying** using "
    "DistilBERT, SVM, or Logistic Regression."
)

# ============================================================
# 2️⃣  Load models (all cached)
# ============================================================
# try:
#     from google.colab import drive
#     drive.mount("/content/drive")
# except Exception:
#     pass

BASE_DIR  = "/CyberBullying Detection System"
BERT_PATH = f"{BASE_DIR}/distilbert_saved_model"
SVM_PATH  = f"{BASE_DIR}/svm.pkl"
LR_PATH   = f"{BASE_DIR}/logistic_regression.pkl"

# ── Vectoriser.
_TFIDF_CANDIDATES = [
    f"{BASE_DIR}/tfidf_vectorizer.pkl",
    f"{BASE_DIR}/tfidf.pkl",
    f"{BASE_DIR}/vectorizer.pkl",
    f"{BASE_DIR}/tfidf_vectoriser.pkl",
]

@st.cache_resource
def load_bert():
    if not os.path.isdir(BERT_PATH):
        raise FileNotFoundError(
            f"DistilBERT folder not found at: {BERT_PATH}\n"
            "Ensure the folder contains config.json, tokenizer files, and model weights."
        )
    tokenizer = AutoTokenizer.from_pretrained(BERT_PATH, local_files_only=True)
    bert      = AutoModelForSequenceClassification.from_pretrained(BERT_PATH, local_files_only=True)
    dev       = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bert.to(dev)
    bert.eval()
    return tokenizer, bert, dev

@st.cache_resource
def load_sklearn_model(path: str):
    with open(path, "rb") as f:
        return pickle.load(f)

@st.cache_resource
def load_tfidf_vectorizer():
    """
    Tries common filenames for the TF-IDF vectoriser.
    Returns the vectoriser, or None if no file is found (pipeline models don't need it).
    """
    from sklearn.pipeline import Pipeline
    from sklearn.feature_extraction.text import TfidfVectorizer

    for candidate in _TFIDF_CANDIDATES:
        if os.path.isfile(candidate):
            with open(candidate, "rb") as f:
                vec = pickle.load(f)
            return vec

    # If no separate file found, check if the models are already pipelines
    # (pipelines handle vectorisation internally — no separate file needed)
    return None

def _apply_vectorizer(texts, vectorizer):
    """
    Transforms a list of raw strings.
    - If vectorizer is available: applies TF-IDF transform.
    - If None (model is a pipeline): passes texts straight through.
    Raises a clear error if neither works.
    """
    if vectorizer is not None:
        return vectorizer.transform(texts)   # sparse matrix → sklearn-ready
    return texts                              # pipeline handles it internally

tokenizer, bert_model, device = load_bert()
svm_model  = load_sklearn_model(SVM_PATH)
lr_model   = load_sklearn_model(LR_PATH)
tfidf_vec  = load_tfidf_vectorizer()

if tfidf_vec is None:
    st.sidebar.info(
        "ℹ️ No separate TF-IDF vectoriser file found. "
        "Assuming SVM & LR models are sklearn Pipelines that include vectorisation."
    )

# ============================================================
# 3️⃣  Prediction functions
# ============================================================
CLASS_NAMES = ["Not Cyberbullying", "Cyberbullying"]
LABEL_MAP   = {0: "Not Cyberbullying", 1: "Cyberbullying"}

def _bert_predict_proba(texts):
    """LIME-compatible: list[str] → (N, 2) ndarray."""
    results = []
    for t in texts:
        inputs = tokenizer(
            t, truncation=True, padding=True, max_length=128, return_tensors="pt"
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            probs = F.softmax(bert_model(**inputs).logits, dim=-1)[0].cpu().numpy()
        results.append(probs)
    return np.array(results)

def _bert_predict_label(text):
    probs      = _bert_predict_proba([text])[0]
    pred_class = int(np.argmax(probs))
    return LABEL_MAP[pred_class], float(probs[pred_class]), probs

def _sklearn_predict_proba(sk_model, texts):
    """
    texts: list of raw strings.
    Applies TF-IDF vectorisation if a separate vectoriser was loaded,
    then runs the classifier.  Works for both bare classifiers and pipelines.
    """
    X = _apply_vectorizer(texts, tfidf_vec)
    try:
        return sk_model.predict_proba(X).astype(float)
    except AttributeError:
        # Classifier doesn't support predict_proba — use decision_function + softmax
        scores = sk_model.decision_function(X)
        if scores.ndim == 1:
            scores = np.column_stack([-scores, scores])
        e = np.exp(scores - scores.max(axis=1, keepdims=True))
        return (e / e.sum(axis=1, keepdims=True)).astype(float)

def _sklearn_predict_label(sk_model, text):
    probs      = _sklearn_predict_proba(sk_model, [text])[0]
    pred_class = int(np.argmax(probs))
    return LABEL_MAP[pred_class], float(probs[pred_class]), probs

# ============================================================
# 4️⃣  Adapter objects — uniform interface across all models
# ============================================================
class DistilBERTAdapter:
    name = "DistilBERT"
    def predict_label(self, text):  return _bert_predict_label(text)
    def predict_proba(self, texts): return _bert_predict_proba(texts)

class SKLearnAdapter:
    def __init__(self, sk_model, name):
        self.sk_model = sk_model
        self.name     = name
    def predict_label(self, text):
        return _sklearn_predict_label(self.sk_model, text)
    def predict_proba(self, texts):
        # texts is always list[str] — vectorisation handled inside _sklearn_predict_proba
        return _sklearn_predict_proba(self.sk_model, texts)

MODELS = {
    "DistilBERT":          DistilBERTAdapter(),
    "SVM":                 SKLearnAdapter(svm_model, "SVM"),
    "Logistic Regression": SKLearnAdapter(lr_model,  "Logistic Regression"),
}

ALL_OPTION       = "🔀 All Models"
DROPDOWN_OPTIONS = [ALL_OPTION] + list(MODELS.keys())

# ============================================================
# 5️⃣  Computational cost helpers (all models)
# ============================================================
def _folder_size_bytes(path: str) -> int:
    total = 0
    for root, _, files in os.walk(path):
        for f in files:
            fp = os.path.join(root, f)
            try:
                total += os.path.getsize(fp)
            except OSError:
                pass
    return total

def _count_parameters(m: torch.nn.Module) -> int:
    return sum(p.numel() for p in m.parameters())

def _dtype_of_model(m: torch.nn.Module) -> str:
    for p in m.parameters():
        return str(p.dtype).replace("torch.", "")
    return "unknown"

def _memory_metrics(dev: torch.device):
    out = {}
    if dev.type == "cuda":
        torch.cuda.synchronize()
        out["GPU allocated (MB)"] = torch.cuda.memory_allocated() / (1024 ** 2)
        out["GPU reserved (MB)"]  = torch.cuda.memory_reserved()  / (1024 ** 2)
    else:
        if psutil is not None:
            p = psutil.Process(os.getpid())
            out["Process RAM RSS (MB)"] = p.memory_info().rss / (1024 ** 2)
        else:
            out["Process RAM RSS (MB)"] = None
    return out

@st.cache_data
def get_static_costs(path: str) -> dict:
    return {"Model folder size (MB)": _folder_size_bytes(path) / (1024 ** 2)}

def benchmark_inference(text: str, n_runs: int = 20, warmup: int = 3, max_length: int = 128):
    tok_times, fwd_times, total_times = [], [], []

    def _sync():
        if device.type == "cuda":
            torch.cuda.synchronize()

    for _ in range(warmup):
        inputs = tokenizer(text, truncation=True, padding=True,
                           max_length=max_length, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        _sync()
        with torch.no_grad():
            _ = bert_model(**inputs)
        _sync()

    for _ in range(n_runs):
        start = time.perf_counter()
        t0    = time.perf_counter()
        inputs = tokenizer(text, truncation=True, padding=True,
                           max_length=max_length, return_tensors="pt")
        t1    = time.perf_counter()
        inputs = {k: v.to(device) for k, v in inputs.items()}
        _sync()
        with torch.no_grad():
            _ = bert_model(**inputs)
        _sync()
        t2  = time.perf_counter()
        end = time.perf_counter()

        tok_times.append((t1 - t0) * 1000.0)
        fwd_times.append((t2 - t1) * 1000.0)
        total_times.append((end - start) * 1000.0)

    def summarize(arr):
        arr = np.array(arr)
        return {
            "mean_ms": float(arr.mean()),
            "std_ms":  float(arr.std(ddof=1)) if len(arr) > 1 else 0.0,
            "p95_ms":  float(np.percentile(arr, 95)),
            "min_ms":  float(arr.min()),
            "max_ms":  float(arr.max()),
            "raw_ms":  arr,
        }

    return {
        "tokenization": summarize(tok_times),
        "forward":      summarize(fwd_times),
        "total":        summarize(total_times),
    }

def benchmark_sklearn(text: str, sk_model, n_runs: int = 20, warmup: int = 3):
    """
    Benchmarks a sklearn model (vectorisation + predict) in milliseconds.
    Returns the same summarize structure as benchmark_inference, but only
    'vectorization', 'predict', and 'total' — no separate forward/tokenization.
    """
    vec_times, pred_times, total_times = [], [], []

    for _ in range(warmup):
        X = _apply_vectorizer([text], tfidf_vec)
        _ = sk_model.predict_proba(X)

    for _ in range(n_runs):
        start = time.perf_counter()

        t0 = time.perf_counter()
        X  = _apply_vectorizer([text], tfidf_vec)
        t1 = time.perf_counter()

        _ = sk_model.predict_proba(X)
        t2 = time.perf_counter()

        end = time.perf_counter()

        vec_times.append((t1 - t0) * 1000.0)
        pred_times.append((t2 - t1) * 1000.0)
        total_times.append((end - start) * 1000.0)

    def summarize(arr):
        arr = np.array(arr)
        return {
            "mean_ms": float(arr.mean()),
            "std_ms":  float(arr.std(ddof=1)) if len(arr) > 1 else 0.0,
            "p95_ms":  float(np.percentile(arr, 95)),
            "min_ms":  float(arr.min()),
            "max_ms":  float(arr.max()),
            "raw_ms":  arr,
        }

    return {
        "vectorization": summarize(vec_times),
        "predict":       summarize(pred_times),
        "total":         summarize(total_times),
    }

# ============================================================
# 6️⃣  LIME explainability helpers
# ============================================================
def _get_lime_explainer():
    try:
        from lime.lime_text import LimeTextExplainer
        return LimeTextExplainer(class_names=CLASS_NAMES)
    except ImportError:
        return None

def explain_text_lime(text, predict_proba_fn, num_samples=2500):
    explainer = _get_lime_explainer()
    if explainer is None:
        return None
    exp = explainer.explain_instance(
        text, predict_proba_fn, num_features=20, num_samples=num_samples
    )
    return {"token_weights": dict(exp.as_list()), "exp_object": exp}

def highlight_text_with_lime(text, token_weights):
    if not token_weights:
        return text
    max_abs = max(abs(v) for v in token_weights.values()) or 1.0
    out = []
    for word in text.split():
        clean  = word.strip(".,!?\"'()[]").lower()
        weight = token_weights.get(clean, token_weights.get(word, 0.0))
        intensity = int(min(abs(weight) / max_abs * 200, 200))
        if weight > 0:
            color = f"rgba(0,180,0,{intensity/255:.2f})"
        elif weight < 0:
            color = f"rgba(220,0,0,{intensity/255:.2f})"
        else:
            color = "transparent"
        out.append(
            f'<span style="background-color:{color};border-radius:3px;padding:1px 3px">{word}</span>'
        )
    return " ".join(out)

def top_token_contributions(token_weights, top_k=10):
    return sorted(token_weights.items(), key=lambda x: abs(x[1]), reverse=True)[:top_k]

def make_plain_english_explanation(text, verdict_label, predict_proba_fn,
                                   max_reasons=5, num_samples=2500):
    lime_result = explain_text_lime(text, predict_proba_fn, num_samples=num_samples)
    if lime_result is None:
        return {
            "plain_english_reasons": ["LIME not installed — run `pip install lime`."],
            "text": text,
            "lime": {"token_weights": {}},
        }
    reasons = []
    for word, weight in top_token_contributions(lime_result["token_weights"], top_k=max_reasons):
        direction = "increases" if weight > 0 else "decreases"
        reasons.append(
            f'The word **"{word}"** {direction} the likelihood of **{verdict_label}** '
            f'(contribution: {weight:+.3f}).'
        )
    return {"plain_english_reasons": reasons, "text": text, "lime": lime_result}

# ============================================================
# 7️⃣  Sidebar
# ============================================================
st.sidebar.header("⚙️ Settings")

selected_option = st.sidebar.selectbox(
    "Select model for analysis:",
    DROPDOWN_OPTIONS,
    index=0,
    help="'All Models' runs all three simultaneously and shows results side-by-side."
)

enable_xai   = st.sidebar.checkbox("Show 'Why?' explanation (LIME)", value=True)
lime_samples = st.sidebar.slider("LIME samples (quality vs speed)", 500, 4000, 2500, step=500)

with st.sidebar.expander("🔬 Benchmark settings", expanded=False):
    n_runs             = st.slider("Benchmark runs", 5, 100, 20, 5)
    warmup             = st.slider("Warmup runs", 0, 10, 3, 1)
    show_latency_chart = st.checkbox("Show latency chart", value=True)

# ============================================================
# 8️⃣  Shared UI helpers
# ============================================================
def render_model_card(container, model_name, label, confidence, probs):
    """Renders a colour-coded result card inside the given container."""
    with container:
        bg = "#d4edda" if label == "Not Cyberbullying" else "#f8d7da"
        st.markdown(
            f"""
            <div style="background:{bg};border-radius:8px;padding:12px 16px;margin-bottom:6px">
                <h4 style="margin:0 0 4px 0">🔹 {model_name}</h4>
                <b>Prediction:</b> {label}<br>
                <b>Confidence:</b> {confidence*100:.2f}%
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.progress(int(confidence * 100))
        st.json({
            "Cyberbullying":     f"{probs[1]*100:.2f}%",
            "Not Cyberbullying": f"{probs[0]*100:.2f}%",
        })

def render_cost_metrics(user_input, model_name: str):
    """
    Benchmarks and displays computational cost for any model.
    - DistilBERT: tokenization + forward pass + total, plus memory metrics.
    - SVM / LR:   vectorization + predict + total.
    Also renders a per-model latency line chart and, when called for all
    three together, a side-by-side comparison bar of mean total latency.
    """
    st.subheader(f"📊 Computational Cost — {model_name} Inference")

    if model_name == "DistilBERT":
        with st.spinner("Benchmarking DistilBERT…"):
            bench  = benchmark_inference(user_input, n_runs=n_runs, warmup=warmup)
            static = get_static_costs(BERT_PATH)
            params = _count_parameters(bert_model)
            dtype  = _dtype_of_model(bert_model)
            mem    = _memory_metrics(device)

        c1, c2, c3 = st.columns(3)
        c1.metric("Total latency (mean)",  f"{bench['total']['mean_ms']:.2f} ms")
        c2.metric("Forward pass (mean)",   f"{bench['forward']['mean_ms']:.2f} ms")
        c3.metric("Tokenization (mean)",   f"{bench['tokenization']['mean_ms']:.2f} ms")

        cost_table = {
            "Device":                  str(device),
            "Model dtype":             dtype,
            "Parameters":              f"{params:,}",
            "Model folder size (MB)":  f"{static['Model folder size (MB)']:.2f}",
            "Total latency mean (ms)": f"{bench['total']['mean_ms']:.2f}",
            "Total latency p95 (ms)":  f"{bench['total']['p95_ms']:.2f}",
            "Forward mean (ms)":       f"{bench['forward']['mean_ms']:.2f}",
            "Forward p95 (ms)":        f"{bench['forward']['p95_ms']:.2f}",
            "Tokenization mean (ms)":  f"{bench['tokenization']['mean_ms']:.2f}",
            "Tokenization p95 (ms)":   f"{bench['tokenization']['p95_ms']:.2f}",
        }
        for k, v in mem.items():
            cost_table[k] = "N/A" if v is None else f"{v:.2f}"
        st.dataframe(cost_table, use_container_width=True)

        if show_latency_chart:
            st.write("**Latency across runs (ms)**")
            st.line_chart({
                "tokenization_ms": bench["tokenization"]["raw_ms"],
                "forward_ms":      bench["forward"]["raw_ms"],
                "total_ms":        bench["total"]["raw_ms"],
            })

    else:
        # SVM or Logistic Regression
        sk_model = MODELS[model_name].sk_model
        with st.spinner(f"Benchmarking {model_name}…"):
            bench      = benchmark_sklearn(user_input, sk_model, n_runs=n_runs, warmup=warmup)
            model_file = SVM_PATH if model_name == "SVM" else LR_PATH
            file_kb    = os.path.getsize(model_file) / 1024

        c1, c2, c3 = st.columns(3)
        c1.metric("Total latency (mean)",    f"{bench['total']['mean_ms']:.2f} ms")
        c2.metric("Predict call (mean)",     f"{bench['predict']['mean_ms']:.2f} ms")
        c3.metric("Vectorization (mean)",    f"{bench['vectorization']['mean_ms']:.2f} ms")

        cost_table = {
            "Model file size (KB)":         f"{file_kb:.2f}",
            "Total latency mean (ms)":      f"{bench['total']['mean_ms']:.2f}",
            "Total latency p95 (ms)":       f"{bench['total']['p95_ms']:.2f}",
            "Predict mean (ms)":            f"{bench['predict']['mean_ms']:.2f}",
            "Predict p95 (ms)":             f"{bench['predict']['p95_ms']:.2f}",
            "Vectorization mean (ms)":      f"{bench['vectorization']['mean_ms']:.2f}",
            "Vectorization p95 (ms)":       f"{bench['vectorization']['p95_ms']:.2f}",
        }
        st.dataframe(cost_table, use_container_width=True)

        if show_latency_chart:
            st.write("**Latency across runs (ms)**")
            st.line_chart({
                "vectorization_ms": bench["vectorization"]["raw_ms"],
                "predict_ms":       bench["predict"]["raw_ms"],
                "total_ms":         bench["total"]["raw_ms"],
            })

    return bench  # caller can use this for cross-model comparisons


def render_cost_metrics_all(user_input):
    """
    Runs benchmarks for all three models and shows:
    1. Individual cost tables + latency charts (in expanders).
    2. A unified comparison table + bar chart of mean total latency.
    """
    st.subheader("📊 Computational Cost — All Models")

    # --- Run all benchmarks ---
    with st.spinner("Benchmarking all models…"):
        bert_bench = benchmark_inference(user_input, n_runs=n_runs, warmup=warmup)
        svm_bench  = benchmark_sklearn(user_input, svm_model,  n_runs=n_runs, warmup=warmup)
        lr_bench   = benchmark_sklearn(user_input, lr_model,   n_runs=n_runs, warmup=warmup)

    all_benches = {
        "DistilBERT":          bert_bench,
        "SVM":                 svm_bench,
        "Logistic Regression": lr_bench,
    }

    # --- Top-level comparison metrics ---
    c1, c2, c3 = st.columns(3)
    c1.metric("DistilBERT total (mean)",          f"{bert_bench['total']['mean_ms']:.2f} ms")
    c2.metric("SVM total (mean)",                 f"{svm_bench['total']['mean_ms']:.2f} ms")
    c3.metric("Logistic Regression total (mean)", f"{lr_bench['total']['mean_ms']:.2f} ms")

    # --- Summary comparison table ---
    st.write("**Side-by-side summary**")
    import pandas as pd
    summary_rows = []
    for mname, bench in all_benches.items():
        row = {"Model": mname,
               "Total mean (ms)":  f"{bench['total']['mean_ms']:.3f}",
               "Total p95 (ms)":   f"{bench['total']['p95_ms']:.3f}",
               "Total min (ms)":   f"{bench['total']['min_ms']:.3f}",
               "Total max (ms)":   f"{bench['total']['max_ms']:.3f}",
               "Total std (ms)":   f"{bench['total']['std_ms']:.3f}"}
        # Sub-stage labels differ per model type
        if mname == "DistilBERT":
            row["Stage 1 (Tokenization) mean (ms)"] = f"{bench['tokenization']['mean_ms']:.3f}"
            row["Stage 2 (Forward pass) mean (ms)"] = f"{bench['forward']['mean_ms']:.3f}"
        else:
            row["Stage 1 (Vectorization) mean (ms)"] = f"{bench['vectorization']['mean_ms']:.3f}"
            row["Stage 2 (Predict call) mean (ms)"]  = f"{bench['predict']['mean_ms']:.3f}"
        summary_rows.append(row)
    st.dataframe(pd.DataFrame(summary_rows).set_index("Model"), use_container_width=True)

    # --- Latency comparison bar chart ---
    if show_latency_chart:
        st.write("**Mean total latency comparison (ms)**")
        bar_data = pd.DataFrame({
            "Model":           list(all_benches.keys()),
            "Mean total (ms)": [b["total"]["mean_ms"] for b in all_benches.values()],
        }).set_index("Model")
        st.bar_chart(bar_data)

        # Per-model latency-over-runs line chart in expanders
        st.write("**Per-model latency across runs (ms)**")
        for mname, bench in all_benches.items():
            with st.expander(f"📈 {mname} — latency across {n_runs} runs", expanded=False):
                if mname == "DistilBERT":
                    st.line_chart({
                        "tokenization_ms": bench["tokenization"]["raw_ms"],
                        "forward_ms":      bench["forward"]["raw_ms"],
                        "total_ms":        bench["total"]["raw_ms"],
                    })
                else:
                    st.line_chart({
                        "vectorization_ms": bench["vectorization"]["raw_ms"],
                        "predict_ms":       bench["predict"]["raw_ms"],
                        "total_ms":         bench["total"]["raw_ms"],
                    })

    # --- Extra DistilBERT-specific static info ---
    with st.expander("🔍 DistilBERT model details", expanded=False):
        static = get_static_costs(BERT_PATH)
        params = _count_parameters(bert_model)
        dtype  = _dtype_of_model(bert_model)
        mem    = _memory_metrics(device)
        details = {
            "Device":                 str(device),
            "Model dtype":            dtype,
            "Parameters":             f"{params:,}",
            "Model folder size (MB)": f"{static['Model folder size (MB)']:.2f}",
        }
        for k, v in mem.items():
            details[k] = "N/A" if v is None else f"{v:.2f}"
        st.dataframe(details, use_container_width=True)

    # --- sklearn model file sizes ---
    with st.expander("🔍 SVM & LR model details", expanded=False):
        svm_kb = os.path.getsize(SVM_PATH) / 1024
        lr_kb  = os.path.getsize(LR_PATH)  / 1024
        st.dataframe({
            "SVM file size (KB)":                f"{svm_kb:.2f}",
            "Logistic Regression file size (KB)": f"{lr_kb:.2f}",
        }, use_container_width=True)

def render_lime(user_input, label, adapter):
    """Runs LIME and renders the explanation block."""
    with st.spinner(f"Generating LIME explanation for {adapter.name}…"):
        xai = make_plain_english_explanation(
            text=user_input,
            verdict_label=label,
            predict_proba_fn=adapter.predict_proba,
            max_reasons=5,
            num_samples=lime_samples,
        )
    st.markdown("**Plain-English explanation:**")
    for r in xai["plain_english_reasons"]:
        st.write(f"- {r}")
    st.markdown("**Highlighted influential words:**")
    st.caption("🟢 Green = pushes toward prediction &nbsp;&nbsp; 🔴 Red = pushes away")
    st.markdown(
        highlight_text_with_lime(xai["text"], xai["lime"]["token_weights"]),
        unsafe_allow_html=True
    )
    st.markdown("**Top token contributions:**")
    for word, weight in top_token_contributions(xai["lime"]["token_weights"], top_k=10):
        direction = "➡️ toward prediction" if weight > 0 else "⬅️ away from prediction"
        st.write(f"- **{word}**: {weight:+.3f} {direction}")

# ============================================================
# 9️⃣  Main input + Analyze button
# ============================================================
st.write("---")
user_input = st.text_area(
    "✏️ Enter a tweet or comment:",
    placeholder="Type something here...",
    height=110
)

if st.button("Analyze", type="primary"):
    if not user_input.strip():
        st.warning("Please enter some text first.")
        st.stop()

    st.write("---")

    # ──────────────────────────────────────────────────────────
    # CASE A: All Models
    # ──────────────────────────────────────────────────────────
    if selected_option == ALL_OPTION:
        st.subheader("📊 All Models — Results")

        with st.spinner("Running all three models…"):
            all_results = {
                name: adapter.predict_label(user_input)
                for name, adapter in MODELS.items()
            }

        # Three equal columns, one card per model
        cols = st.columns(3)
        for col, (model_name, (label, conf, probs)) in zip(cols, all_results.items()):
            render_model_card(col, model_name, label, conf, probs)

        # All-model cost metrics + comparison
        st.write("---")
        render_cost_metrics_all(user_input)

        # LIME for each model — collapsible so page stays tidy
        if enable_xai:
            st.write("---")
            st.subheader("🧠 LIME Explanations — All Models")
            for model_name, (label, conf, probs) in all_results.items():
                with st.expander(f"Why did **{model_name}** predict '{label}'?", expanded=False):
                    render_lime(user_input, label, MODELS[model_name])

    # ──────────────────────────────────────────────────────────
    # CASE B: Single model
    # ──────────────────────────────────────────────────────────
    else:
        adapter = MODELS[selected_option]
        st.subheader(f"✅ Result — {selected_option}")

        with st.spinner(f"Analyzing with {selected_option}…"):
            label, confidence, probs = adapter.predict_label(user_input)

        render_model_card(st.container(), selected_option, label, confidence, probs)

        st.write("### Probabilities:")
        st.json({
            "Cyberbullying":     f"{probs[1]*100:.2f}%",
            "Not Cyberbullying": f"{probs[0]*100:.2f}%",
        })

        # Cost metrics for the selected model (works for all three)
        st.write("---")
        render_cost_metrics(user_input, selected_option)

        # LIME
        if enable_xai:
            st.write("---")
            st.subheader("🧠 Why was this classified this way?")
            render_lime(user_input, label, adapter)

st.write("---")
st.markdown("Built with ❤️ using **DistilBERT**, **SVM**, **Logistic Regression**, and **Streamlit**.")