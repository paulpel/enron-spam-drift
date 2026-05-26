from scipy.stats import ttest_rel
import numpy as np
import pandas as pd


def compare_classifiers(scores1, scores2, metric="accuracy", label1="Set 1", label2="Set 2"):
    """
    Compare two conditions per classifier with a paired t-test over CV folds.

    Each classifier must expose a ``fold_scores`` list (one accuracy per
    cross-validation fold) in both score dicts; the paired t-test is run on
    those equal-length samples. Comparing two single accuracy values (as an
    earlier version did) is statistically meaningless — hence the fold scores.

    :param scores1: per-classifier scores for the first condition.
    :type scores1: dict
    :param scores2: per-classifier scores for the second condition.
    :type scores2: dict
    :param metric: metric name used for the displayed mean (default ``accuracy``).
    :type metric: str
    :param label1: label for the first condition.
    :type label1: str
    :param label2: label for the second condition.
    :type label2: str
    :return: DataFrame with the mean metric per condition, t-statistic and p-value.
    :rtype: pandas.DataFrame
    """
    comparisons = []

    for clf in scores1:
        folds1 = scores1[clf].get("fold_scores")
        folds2 = scores2[clf].get("fold_scores")
        if folds1 is None or folds2 is None or len(folds1) != len(folds2):
            raise ValueError(
                f"{clf}: a paired t-test needs equal-length per-fold scores; "
                "populate 'fold_scores' for both conditions."
            )

        t_stat, p_value = ttest_rel(folds1, folds2)
        comparisons.append(
            {
                "Classifier": clf,
                f"{label1} {metric}": float(np.mean(folds1)),
                f"{label2} {metric}": float(np.mean(folds2)),
                "t-statistic": t_stat,
                "p-value": p_value,
            }
        )

    return pd.DataFrame(comparisons)
