# xai/explain_text_with_lime.py
from __future__ import annotations

from typing import Dict, List, Tuple
import numpy as np
from lime.lime_text import LimeTextExplainer


def explain_text_lime(
    text: str,
    predict_proba_fn,
    class_names: Tuple[str, str] = ("Not Cyberbullying", "Cyberbullying"),
    num_features: int = 12,
    num_samples: int = 2500,
) -> Dict:
    """
    Universal LIME explainer for binary text classification.

    predict_proba_fn must accept List[str] and return np.ndarray shape (n, 2)
    """
    explainer = LimeTextExplainer(class_names=list(class_names))

    # Choose the model's predicted class for THIS input
    probs = np.asarray(predict_proba_fn([text]))[0]
    target_class = int(probs.argmax())

    exp = explainer.explain_instance(
        text_instance=text,
        classifier_fn=predict_proba_fn,
        num_features=num_features,
        num_samples=num_samples,
        labels=(target_class,),
    )

    lime_list = exp.as_list(label=target_class)  # [(token, weight), ...]

    return {
        "target_class": target_class,
        "target_class_name": class_names[target_class],
        "lime_list": lime_list,
        "probs": probs,
    }
