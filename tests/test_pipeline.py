"""Synthetic-data tests for the cross-validated scoring + statistical comparison.

These don't need the Enron dataset or BERT weights: they build tiny toy datasets
so the per-fold CV scoring, the paired-t-test comparison, and the GaussianNB BERT
path can be exercised end-to-end. (The torch/transformers import is lazy, so only
scikit-learn / scipy / pandas / numpy are required here.)
"""

import numpy as np
import pandas as pd

from train_evaluate import (
    train_classifiers,
    evaluate_models,
    train_classifiers_with_bert_features,
    bert_evaluate_models,
)
from statistical_tests import compare_classifiers

SPAM = ["free", "money", "win", "cash", "prize"]
HAM = ["meeting", "report", "schedule", "project", "team"]


def _toy_text_df(n=60, seed=0):
    """Two clearly separable token-bag classes (message column = list of tokens)."""
    rng = np.random.default_rng(seed)
    messages, labels = [], []
    for i in range(n):
        vocab, label = (SPAM, 1) if i % 2 == 0 else (HAM, 0)
        messages.append(list(rng.choice(vocab, size=4)))
        labels.append(label)
    return pd.DataFrame({"message": messages, "label": labels})


def test_bow_training_produces_per_fold_scores():
    _, scores, _ = train_classifiers(_toy_text_df())
    assert scores, "no classifiers scored"
    for s in scores.values():
        assert len(s["fold_scores"]) == 5
        assert 0.0 <= s["accuracy"] <= 1.0


def test_compare_classifiers_runs_paired_ttest_on_folds():
    _, scores_a, _ = train_classifiers(_toy_text_df(seed=0))
    models, _, vec = train_classifiers(_toy_text_df(seed=0))
    scores_b = evaluate_models(models, vec, _toy_text_df(seed=1))

    result = compare_classifiers(scores_a, scores_b, label1="A", label2="B")
    assert len(result) == len(scores_a)
    assert {"Classifier", "t-statistic", "p-value"}.issubset(result.columns)


def test_bert_path_handles_negative_features_without_offset():
    """GaussianNB must cope with real-valued (negative) features — no `+= 10` hack."""
    df = _toy_text_df(n=40)
    feats = np.random.default_rng(2).standard_normal((len(df), 16))  # has negatives
    assert feats.min() < 0

    models, scores = train_classifiers_with_bert_features(df, feats)
    assert all(len(s["fold_scores"]) == 5 for s in scores.values())

    eval_scores = bert_evaluate_models(models, feats, df)
    assert all("fold_scores" in s for s in eval_scores.values())
