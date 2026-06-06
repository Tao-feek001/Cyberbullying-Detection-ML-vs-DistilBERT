# Cyberbullying-Detection-ML-vs-DistilBERT
A comparative NLP research system evaluating traditional machine learning (Logistic Regression, SVM, Random Forest) against a fine-tuned DistilBERT transformer for cyberbullying detection on a unified corpus of 230,363 Twitter-style posts, with explainable AI (LIME), statistical testing, latency benchmarking, and a Streamlit deployment prototype.


## Project overview
This project is based on my dissertation research on cyberbullying detection in
social media text. It includes:
- A cleaned, unified Twitter-style corpus built from multiple public datasets
  (not included in this repo for size/licensing reasons).
- Classical models trained on TF–IDF features: Logistic Regression, Linear SVM,
  Random Forest (saved as `.pkl` files).
- A fine-tuned DistilBERT model for binary classification: cyberbullying vs
  non-cyberbullying.
- A Streamlit app for interactive prediction and model comparison.
- LIME-based explanations for model predictions.


## Repository structure
- `Dataset_Combination/`
  - `data.py` – utilities for loading and combining the Twitter datasets.
  - (Large CSVs are ignored; see Data section.)
- `detection_app.py` – Streamlit web app for cyberbullying detection.
- `evaluate.py` – scripts for evaluating models and computing metrics.
- `logistic_regression.pkl`, `svm.pkl`, `random_forest.pkl`,
  `tfidf_vectorizer.pkl` – trained classical models and vectorizer.
- `xai_explain_text_with_lime.py`, `xai_model_adapters.py`,
  `xai_xai_text_explain.py` – LIME/XAI utilities for text explanations.
- `README.md` – project documentation.
- `.gitignore` – excludes large datasets and model checkpoints from the repo.


## Data
The original dataset is a unified Twitter-style corpus constructed by merging
several publicly available cyberbullying / toxic-comment datasets, cleaning the
labels, and mapping them to a binary cyberbullying vs non-cyberbullying
schema.

Because of size and licensing, the full CSV files are **not included** in this
repository. To reproduce the experiments:

1. Obtain the source datasets listed in the dissertation report.
2. Place the combined CSVs in `Dataset_Combination/`.
3. Update any file paths in `Dataset_Combination/data.py` if needed.


## How to run

### 1. Environment setup

```bash
# create and activate a virtual env (example)
conda create -n cyberbullying python=3.10
conda activate cyberbullying

# install dependencies
pip install -r requirements.txt
```

### 2. Run the Streamlit app

```bash
streamlit run detection_app.py
```

Then open the URL shown in the terminal (usually http://localhost:8501) and
enter example tweets to see predictions from each model.

### 3. Re-run evaluation

```bash
python evaluate.py
```

This will reload the trained models and compute the evaluation metrics reported
in the dissertation (accuracy, precision, recall, F1, ROC–AUC, etc.).

## Key results

- DistilBERT outperforms the TF–IDF baselines on the held-out test set, with
  higher F1, ROC–AUC, and better recall on the cyberbullying class.
- Classical models are faster and remain competitive on clear, explicit abuse
  but struggle with subtle or context-dependent cases.
- LIME explanations show that DistilBERT uses contextual cues beyond simple
  keywords, especially on ambiguous examples.


  ## Author

Developed by Taofeek Abimbolu as part of my BSc dissertation in Computing at
the University of Northampton (2026).

If you use this project in academic work, please cite the dissertation report
or link to this repository.
