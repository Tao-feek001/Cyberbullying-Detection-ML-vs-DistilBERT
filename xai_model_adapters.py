# xai/model_adapters.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple, Union
import numpy as np

try:
    import torch
    import torch.nn.functional as F
except Exception:
    torch = None
    F = None


# -----------------------------
# Shared output schema
# -----------------------------
@dataclass
class ModelAdapter:
    """
    Unifies different model types under one API:
      - predict_proba(texts) -> np.ndarray of shape (n, 2)
      - predict_label(text) -> (label_str, confidence_float, probs_np)
    """
    name: str
    class_names: Tuple[str, str] = ("Not Cyberbullying", "Cyberbullying")
    predict_proba: Callable[[List[str]], np.ndarray] = None


    def predict_label(self, text: str) -> Tuple[str, float, np.ndarray]:
        probs = self.predict_proba([text])[0]  # shape (2,)
        pred_idx = int(np.argmax(probs))
        confidence = float(probs[pred_idx])
        return self.class_names[pred_idx], confidence, probs


# -----------------------------
# sklearn adapter (LR/RF/SVM)
# -----------------------------
def make_sklearn_text_adapter(
    name: str,
    model,
    vectorizer=None,
    class_names: Tuple[str, str] = ("Not Cyberbullying", "Cyberbullying"),
    positive_class_index: int = 1,
) -> ModelAdapter:
    """
    Works with:
      - Pipeline(vectorizer+classifier)  (vectorizer can be None)
      - classifier + separate vectorizer (pass vectorizer)
    Requirements:
      - model must provide either predict_proba OR decision_function.
        If decision_function is used, we convert scores to probabilities.

    For SVM:
      - Best: train with probability=True (SVC) OR wrap in CalibratedClassifierCV.
      - If neither, decision_function is used and we do a softmax/sigmoid fallback.
    """

    def _vectorize(texts: List[str]):
        if vectorizer is None:
            # assume model is already a Pipeline that can accept raw texts
            return texts
        return vectorizer.transform(texts)

    def _to_proba_from_scores(scores: np.ndarray) -> np.ndarray:
        """
        Convert decision_function outputs to pseudo-probabilities.
        - Binary: scores shape (n,) or (n,1) => sigmoid
        - Multiclass: scores shape (n,k) => softmax
        """
        scores = np.asarray(scores)

        if scores.ndim == 1:
            # binary raw margin
            p_pos = 1.0 / (1.0 + np.exp(-scores))
            p_neg = 1.0 - p_pos
            proba = np.vstack([p_neg, p_pos]).T
            return proba

        if scores.ndim == 2 and scores.shape[1] == 1:
            s = scores[:, 0]
            p_pos = 1.0 / (1.0 + np.exp(-s))
            p_neg = 1.0 - p_pos
            return np.vstack([p_neg, p_pos]).T

        # softmax for (n, k)
        exps = np.exp(scores - scores.max(axis=1, keepdims=True))
        proba = exps / exps.sum(axis=1, keepdims=True)

        # If k != 2, we still try to pull out negative/positive based on index
        if proba.shape[1] == 2:
            return proba

        # For safety, map to 2-class format using chosen positive index
        pos = proba[:, positive_class_index]
        neg = 1.0 - pos
        return np.vstack([neg, pos]).T

    def predict_proba_batch(texts: List[str]) -> np.ndarray:
        X = _vectorize(texts)

        # Case 1: predict_proba exists (ideal)
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(X)
            proba = np.asarray(proba)
            # Ensure binary format (n,2)
            if proba.ndim == 2 and proba.shape[1] >= 2:
                # If classifier returns classes in different order, you can re-order here
                return proba[:, :2]
            # If returns (n,), fix
            if proba.ndim == 1:
                proba = np.vstack([1 - proba, proba]).T
            return proba

        # Case 2: decision_function fallback
        if hasattr(model, "decision_function"):
            scores = model.decision_function(X)
            return _to_proba_from_scores(scores)

        # Case 3: last-resort: use predict and make hard probs
        preds = model.predict(X)
        preds = np.asarray(preds).astype(int)
        proba = np.zeros((len(preds), 2), dtype=float)
        proba[np.arange(len(preds)), preds] = 1.0
        return proba

    return ModelAdapter(
        name=name,
        class_names=class_names,
        predict_proba=predict_proba_batch
    )


# -----------------------------
# DistilBERT adapter
# -----------------------------
def make_distilbert_adapter(
    name: str,
    tokenizer,
    model,
    device,
    class_names: Tuple[str, str] = ("Not Cyberbullying", "Cyberbullying"),
    max_length: int = 128,
) -> ModelAdapter:
    """
    Provides predict_proba(texts) for HuggingFace classification models.
    """
    if torch is None or F is None:
        raise RuntimeError("PyTorch not available. Install torch to use DistilBERT adapter.")

    model.eval()

    def predict_proba_batch(texts: List[str]) -> np.ndarray:
        inputs = tokenizer(
            texts,
            truncation=True,
            padding=True,
            max_length=max_length,
            return_tensors="pt"
        ).to(device)

        with torch.no_grad():
            outputs = model(**inputs)
            probs = F.softmax(outputs.logits, dim=-1).detach().cpu().numpy()

        # enforce shape (n,2)
        probs = np.asarray(probs)
        if probs.shape[1] != 2:
            # try to compress to 2 classes if needed
            pos = probs[:, 1]
            neg = 1.0 - pos
            probs = np.vstack([neg, pos]).T
        return probs

    return ModelAdapter(
        name=name,
        class_names=class_names,
        predict_proba=predict_proba_batch
    )
