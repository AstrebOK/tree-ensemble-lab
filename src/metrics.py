"""Метрики и интерпретация: OOB, энтропия/Джини/информационный выигрыш, MDI и permutation importance, G-mean."""
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.metrics import confusion_matrix, roc_auc_score


def g_mean_score(y_true, y_pred):
    """G-mean = sqrt(Recall(+) * Recall(-)): recall миноритарного класса на recall мажоритарного."""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    recall_pos = tp / (tp + fn) if (tp + fn) else 0.0   # recall класса "мошенничество"
    recall_neg = tn / (tn + fp) if (tn + fp) else 0.0   # recall класса "легитимно" (специфичность)
    return float(np.sqrt(recall_pos * recall_neg))


def oob_roc_auc(rf, y_train):
    """ROC-AUC по out-of-bag предсказаниям Random Forest (нужен oob_score=True при обучении)."""
    return float(roc_auc_score(y_train, rf.oob_decision_function_[:, 1]))


def entropy(y):
    """Энтропия Шеннона H = -sum(p_i * log2(p_i)) для вектора меток."""
    _, counts = np.unique(np.asarray(y), return_counts=True)
    p = counts / counts.sum()
    return float(-(p * np.log2(p)).sum())


def gini(y):
    """Индекс Джини G = 1 - sum(p_k^2) для вектора меток."""
    _, counts = np.unique(np.asarray(y), return_counts=True)
    p = counts / counts.sum()
    return float(1.0 - (p ** 2).sum())


def information_gain(y, mask, impurity=entropy):
    """Информационный выигрыш разбиения: I(родитель) - взвешенная сумма I(потомков).

    mask - булев массив: True уходит в левый узел, False - в правый.
    """
    y = np.asarray(y)
    mask = np.asarray(mask, dtype=bool)
    left, right = y[mask], y[~mask]
    if len(left) == 0 or len(right) == 0:
        return 0.0
    w_left, w_right = len(left) / len(y), len(right) / len(y)
    return float(impurity(y) - w_left * impurity(left) - w_right * impurity(right))


def mdi_importances(model, feature_names):
    """MDI-важность (feature_importances_) в виде таблицы, отсортированной по убыванию."""
    imp = model.feature_importances_
    assert len(imp) == len(feature_names), f"Несовпадение: {len(imp)} важностей vs {len(feature_names)} имён"
    return (pd.DataFrame({"feature": feature_names, "importance": imp})
            .sort_values("importance", ascending=False).reset_index(drop=True))


def permutation_importance_table(pipe, X, y, scoring="roc_auc", n_repeats=10, seed=42):
    """Permutation importance по ИСХОДНЫМ признакам (pipeline целиком): падение метрики при перемешивании признака."""
    res = permutation_importance(pipe, X, y, scoring=scoring, n_repeats=n_repeats,
                                 random_state=seed, n_jobs=1)
    return (pd.DataFrame({"feature": X.columns, "importance": res.importances_mean,
                          "std": res.importances_std})
            .sort_values("importance", ascending=False).reset_index(drop=True))
