# xai/xai_text_explain.py
from __future__ import annotations

from typing import Dict, List, Tuple
from xai_explain_text_with_lime import explain_text_lime


def make_plain_english_explanation(
    text: str,
    verdict_label: str,
    confidence: float,
    probs,
    predict_proba_fn,
    class_names: Tuple[str, str] = ("Not Cyberbullying", "Cyberbullying"),
    max_reasons: int = 5,
) -> Dict:
    """
    Returns a unified explanation object you can show in Streamlit.
    Works for LR/SVM/RF/DistilBERT as long as predict_proba_fn is standardised.
    """
    lime_out = explain_text_lime(
        text=text,
        predict_proba_fn=predict_proba_fn,
        class_names=class_names,
        num_features=12,
        num_samples=2500,
    )

    lime_list = lime_out["lime_list"]  # [(token, weight), ...]

    pushing_for = [(w, wt) for (w, wt) in lime_list if wt > 0]
    pushing_against = [(w, wt) for (w, wt) in lime_list if wt < 0]

    pushing_for = sorted(pushing_for, key=lambda x: x[1], reverse=True)[:max_reasons]
    pushing_against = sorted(pushing_against, key=lambda x: x[1])[:max_reasons]  # most negative

    reasons: List[str] = []

    if verdict_label == class_names[1]:  # Cyberbullying
        if pushing_for:
            top = ", ".join([f"“{w}”" for w, _ in pushing_for[:3]])
            reasons.append(f"The prediction moved toward **Cyberbullying** mainly because of {top}, which resembles insulting/targeted wording.")
        if pushing_against:
            top = ", ".join([f"“{w}”" for w, _ in pushing_against[:3]])
            reasons.append(f"However, words like {top} reduced the cyberbullying score slightly, suggesting some neutral context.")
    else:  # Not Cyberbullying
        if pushing_for:
            top = ", ".join([f"“{w}”" for w, _ in pushing_for[:3]])
            reasons.append(f"The prediction moved toward **Not Cyberbullying** because {top} looks more neutral/supportive rather than abusive.")
        if pushing_against:
            top = ", ".join([f"“{w}”" for w, _ in pushing_against[:3]])
            reasons.append(f"Still, {top} increased the cyberbullying score a bit, but not enough to flip the final decision.")

    if not reasons:
        reasons.append("The model’s decision was influenced by the overall tone and phrasing rather than one or two standout words.")

    return {
        "text": text,
        "verdict": verdict_label,
        "confidence": float(confidence),
        "probs": probs,
        "lime": {
            "target_class": lime_out["target_class"],
            "target_class_name": lime_out["target_class_name"],
            "token_weights": lime_list,
            "pushing_for": pushing_for,
            "pushing_against": pushing_against,
        },
        "plain_english_reasons": reasons,
    }
