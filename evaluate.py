# ============================================================
# Cyberbullying Detection — Standalone Model Evaluation
# Mirrors the exact preprocessing used during training
# Run this as a single cell in Google Colab
# ============================================================

# ── 0. Install deps ───────────────────────────────────────────────────────────
import subprocess, sys
for pkg in ["transformers", "torch", "scikit-learn", "matplotlib", "seaborn", "pandas", "numpy"]:
    subprocess.run([sys.executable, "-m", "pip", "install", pkg, "-q"], check=False)

# ── 1. Imports ────────────────────────────────────────────────────────────────
import os, re, pickle, warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import (
    confusion_matrix, accuracy_score, precision_score,
    recall_score, f1_score, roc_auc_score, average_precision_score,
    roc_curve, precision_recall_curve, classification_report,
)
warnings.filterwarnings("ignore")

# ── 2. Mount Google Drive ─────────────────────────────────────────────────────
from google.colab import drive
drive.mount("/content/drive")

# ── 3. Paths ──────────────────────────────────────────────────────────────────
BASE_DIR   = "/content/drive/MyDrive/Up/CyberBullying Detection System"
BERT_PATH  = f"{BASE_DIR}/distilbert_saved_model"
SVM_PATH   = f"{BASE_DIR}/svm.pkl"
LR_PATH    = f"{BASE_DIR}/logistic_regression.pkl"
TFIDF_PATH = f"{BASE_DIR}/tfidf_vectorizer.pkl"
CSV_PATH   = f"{BASE_DIR}/cyberbullying_datasets.csv"
TEXT_COL   = "text"
LABEL_COL  = "label"

CLASS_NAMES  = ["Not Cyberbullying", "Cyberbullying"]   # 0, 1
MODEL_COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c"]

# ── 4. Preprocessing — identical to training ─────────────────────────────────
def clean_text(s):
    """Exact same function used during training."""
    s = str(s).lower()
    s = re.sub(r"http\S+|www\.\S+", "", s)   # remove URLs
    s = re.sub(r"@\w+", "", s)                # remove @mentions
    s = re.sub(r"#", "", s)                   # remove hashtag symbol
    s = re.sub(r"[^a-z\s]", "", s)            # keep only letters + spaces
    s = re.sub(r"\s+", " ", s).strip()
    return s

def encode_label(x):
    """
    Exact same binary mapping used during training:
      not_cyberbullying / 0 / 0.0 / 2 / 2.0  →  0
      everything else (gender, religion, age,
        ethnicity, other_cyberbullying, 1, 1.0) →  1
    """
    x = str(x).lower().strip()
    return 0 if x in {"not_cyberbullying", "0", "0.0", "2", "2.0"} else 1

# ── 5. Load & preprocess dataset ─────────────────────────────────────────────
print("Loading dataset…")
df = pd.read_csv(CSV_PATH).dropna(subset=[TEXT_COL, LABEL_COL])
print(f"  Raw rows: {len(df):,}")
print(f"  Raw label distribution:\n{df[LABEL_COL].value_counts().to_string()}\n")

df["clean_text"] = df[TEXT_COL].apply(clean_text)
df["label_int"]  = df[LABEL_COL].apply(encode_label)

texts  = df["clean_text"].tolist()   # cleaned text → fed to all models
y_true = df["label_int"].tolist()

print(f"  After encoding:")
print(f"    Not Cyberbullying (0): {y_true.count(0):,}")
print(f"    Cyberbullying     (1): {y_true.count(1):,}")
print(f"    Total            : {len(y_true):,}")

# ── 6. Load models ────────────────────────────────────────────────────────────
print("\nLoading models…")
device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = AutoTokenizer.from_pretrained(BERT_PATH, local_files_only=True)
bert      = AutoModelForSequenceClassification.from_pretrained(BERT_PATH, local_files_only=True)
bert.to(device).eval()

with open(SVM_PATH,   "rb") as f: svm_model = pickle.load(f)
with open(LR_PATH,    "rb") as f: lr_model  = pickle.load(f)
with open(TFIDF_PATH, "rb") as f: tfidf_vec = pickle.load(f)
print(f"  ✅ All models loaded  (DistilBERT on: {device})")

# ── 7. Prediction functions ───────────────────────────────────────────────────
def predict_bert(texts, batch_size=32):
    """Returns (N, 2) probability array."""
    all_proba = []
    for i in range(0, len(texts), batch_size):
        batch  = texts[i : i + batch_size]
        inputs = tokenizer(batch, truncation=True, padding=True,
                           max_length=128, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            proba = F.softmax(bert(**inputs).logits, dim=-1).cpu().numpy()
        all_proba.append(proba)
        if (i // batch_size) % 10 == 0:
            print(f"    DistilBERT: {min(i+batch_size, len(texts))}/{len(texts)}", end="\r")
    print()
    return np.vstack(all_proba)

def predict_sklearn(sk_model, texts):
    """Vectorises with TF-IDF then predicts. Returns (N, 2) array."""
    X = tfidf_vec.transform(texts)
    try:
        return sk_model.predict_proba(X).astype(float)
    except AttributeError:
        scores = sk_model.decision_function(X)
        if scores.ndim == 1:
            scores = np.column_stack([-scores, scores])
        e = np.exp(scores - scores.max(axis=1, keepdims=True))
        return (e / e.sum(axis=1, keepdims=True)).astype(float)

# ── 8. Run predictions ────────────────────────────────────────────────────────
print("\nRunning predictions…")
print("  → DistilBERT")
bert_proba = predict_bert(texts)
bert_pred  = np.argmax(bert_proba, axis=1)

print("  → SVM")
svm_proba  = predict_sklearn(svm_model, texts)
svm_pred   = np.argmax(svm_proba, axis=1)

print("  → Logistic Regression")
lr_proba   = predict_sklearn(lr_model, texts)
lr_pred    = np.argmax(lr_proba, axis=1)
print("  ✅ Done")

# ── 9. Compute metrics ────────────────────────────────────────────────────────
def compute_metrics(y_true, y_pred, y_proba_pos):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return {
        "Accuracy":      round(accuracy_score(y_true, y_pred),                          4),
        "Precision":     round(precision_score(y_true, y_pred,    zero_division=0),     4),
        "Recall":        round(recall_score(y_true, y_pred,       zero_division=0),     4),
        "F1":            round(f1_score(y_true, y_pred,           zero_division=0),     4),
        "ROC-AUC":       round(roc_auc_score(y_true, y_proba_pos),                      4),
        "Avg Precision": round(average_precision_score(y_true, y_proba_pos),            4),
        "TP": int(tp), "FP": int(fp), "TN": int(tn), "FN": int(fn),
    }

results = {
    "DistilBERT":          compute_metrics(y_true, bert_pred, bert_proba[:, 1]),
    "SVM":                 compute_metrics(y_true, svm_pred,  svm_proba[:, 1]),
    "Logistic Regression": compute_metrics(y_true, lr_pred,   lr_proba[:, 1]),
}

# ── 10. Print tables ──────────────────────────────────────────────────────────
print("\n" + "="*72)
print("  OVERALL METRICS SUMMARY")
print("="*72)
summary_df = pd.DataFrame(results).T
print(summary_df.to_string())

for mname, y_pred in zip(results.keys(), [bert_pred, svm_pred, lr_pred]):
    print(f"\n{'='*72}")
    print(f"  PER-CLASS REPORT — {mname}")
    print("="*72)
    print(classification_report(y_true, y_pred,
                                 target_names=CLASS_NAMES, zero_division=0))

# ── 11. Charts ────────────────────────────────────────────────────────────────
sns.set_style("whitegrid")
model_names = list(results.keys())
all_preds   = [bert_pred,  svm_pred,  lr_pred]
all_probas  = [bert_proba, svm_proba, lr_proba]

# ── 11a. Confusion Matrices ───────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
fig.suptitle("Confusion Matrices", fontsize=15, fontweight="bold")
for ax, mname, y_pred in zip(axes, model_names, all_preds):
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues", ax=ax,
        xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
        linewidths=0.5, linecolor="grey",
        annot_kws={"size": 14, "weight": "bold"},
    )
    r  = results[mname]
    ax.set_title(mname, fontsize=12, pad=8)
    ax.set_xlabel(
        f"Predicted\nTP={r['TP']}  FP={r['FP']}  FN={r['FN']}  TN={r['TN']}",
        fontsize=9,
    )
    ax.set_ylabel("Actual", fontsize=10)
plt.tight_layout()
plt.savefig("/content/confusion_matrices.png", dpi=150, bbox_inches="tight")
plt.show()
print("💾 Saved: /content/confusion_matrices.png")

# ── 11b. ROC Curves ───────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 5.5))
for mname, y_proba, color in zip(model_names, all_probas, MODEL_COLORS):
    fpr, tpr, _ = roc_curve(y_true, y_proba[:, 1])
    auc = roc_auc_score(y_true, y_proba[:, 1])
    ax.plot(fpr, tpr, color=color, lw=2.2, label=f"{mname}  (AUC = {auc:.4f})")
ax.plot([0,1],[0,1], "k--", lw=1.2, label="Random baseline")
ax.fill_between([0,1],[0,1], alpha=0.04, color="grey")
ax.set_xlabel("False Positive Rate", fontsize=12)
ax.set_ylabel("True Positive Rate", fontsize=12)
ax.set_title("ROC Curves", fontsize=14, fontweight="bold")
ax.legend(fontsize=10); ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("/content/roc_curves.png", dpi=150, bbox_inches="tight")
plt.show()
print("💾 Saved: /content/roc_curves.png")

# ── 11c. Precision-Recall Curves ─────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 5.5))
for mname, y_proba, color in zip(model_names, all_probas, MODEL_COLORS):
    prec, rec, _ = precision_recall_curve(y_true, y_proba[:, 1])
    ap = average_precision_score(y_true, y_proba[:, 1])
    ax.plot(rec, prec, color=color, lw=2.2, label=f"{mname}  (AP = {ap:.4f})")
baseline = sum(y_true) / len(y_true)
ax.axhline(baseline, color="grey", lw=1.2, linestyle="--",
           label=f"Random baseline ({baseline:.3f})")
ax.set_xlabel("Recall", fontsize=12)
ax.set_ylabel("Precision", fontsize=12)
ax.set_title("Precision-Recall Curves", fontsize=14, fontweight="bold")
ax.legend(fontsize=10); ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("/content/pr_curves.png", dpi=150, bbox_inches="tight")
plt.show()
print("💾 Saved: /content/pr_curves.png")

# ── 11d. Metrics Comparison Bar Chart ────────────────────────────────────────
metric_keys = ["Accuracy", "Precision", "Recall", "F1", "ROC-AUC"]
x     = np.arange(len(metric_keys))
width = 0.24

fig, ax = plt.subplots(figsize=(11, 5))
for i, (mname, color) in enumerate(zip(model_names, MODEL_COLORS)):
    vals = [results[mname][k] for k in metric_keys]
    bars = ax.bar(x + i*width, vals, width, label=mname,
                  color=color, alpha=0.87, edgecolor="white")
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.006,
                f"{v:.3f}", ha="center", va="bottom", fontsize=8, fontweight="bold")
ax.set_xticks(x + width)
ax.set_xticklabels(metric_keys, fontsize=11)
ax.set_ylim(0, 1.15)
ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
ax.set_ylabel("Score", fontsize=11)
ax.set_title("Model Comparison — Classification Metrics (binary)",
             fontsize=13, fontweight="bold")
ax.legend(fontsize=10); ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig("/content/metrics_comparison.png", dpi=150, bbox_inches="tight")
plt.show()
print("💾 Saved: /content/metrics_comparison.png")

# ── 12. Final summary ─────────────────────────────────────────────────────────
print("\n✅ Evaluation complete — charts saved to /content/")
print("\n📋 Quick summary:")
for mname in model_names:
    r = results[mname]
    print(f"  {mname:<22}  Acc={r['Accuracy']:.4f}  "
          f"F1={r['F1']:.4f}  AUC={r['ROC-AUC']:.4f}")