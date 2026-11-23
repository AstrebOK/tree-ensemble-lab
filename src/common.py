"""Общие функции для ЛР №6 (вариант 3): загрузка данных, разбиение, предобработка, оценка, графики."""
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (accuracy_score, average_precision_score, f1_score,
                             precision_score, recall_score, roc_auc_score,
                             ConfusionMatrixDisplay)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import plot_tree

from metrics import g_mean_score

SEED = 42                                            # фиксируем генератор случайных чисел
ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "credit_card_fraud_2026.csv"
FIG_DIR = ROOT / "figures"
TARGET = "is_fraud"                                  # целевая переменная
ID_COLS = ["transaction_id"]                         # идентификатор - не признак
CLASS_NAMES = ["Легитимно", "Мошенничество"]


def load_data(path=DATA_PATH):
    """Читает csv; булевы признаки переводит в 0/1 (бинарные флаги)."""
    df = pd.read_csv(path)
    bool_cols = df.select_dtypes(include="bool").columns
    df[bool_cols] = df[bool_cols].astype(int)
    return df


def get_feature_lists(df):
    """Делит признаки на числовые (включая бинарные флаги) и категориальные."""
    features = [c for c in df.columns if c not in ID_COLS + [TARGET]]
    numeric = [c for c in features if pd.api.types.is_numeric_dtype(df[c])]
    categorical = [c for c in features if c not in numeric]
    return numeric, categorical


def make_preprocessor(numeric_cols, categorical_cols):
    """Числовые признаки - как есть (деревьям масштаб не важен), категориальные - OneHotEncoder."""
    return ColumnTransformer(
        transformers=[
            ("num", "passthrough", numeric_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_cols),
        ],
        verbose_feature_names_out=False,
    )


def split_data(df, test_size=0.2, seed=SEED):
    """Сначала разбиение train/test (stratify=y), и только потом любая предобработка."""
    X = df.drop(columns=ID_COLS + [TARGET])
    y = df[TARGET]
    return train_test_split(X, y, test_size=test_size, stratify=y, random_state=seed)


def split_train_val(X_train, y_train, val_size=0.25, seed=SEED):
    """Из обучающей части выделяет validation-выборку для подбора гиперпараметров."""
    return train_test_split(X_train, y_train, test_size=val_size, stratify=y_train, random_state=seed)


def evaluate(model, X, y):
    """Метрики классификации (класс 1 = мошенничество - миноритарный)."""
    y_pred = model.predict(X)
    y_proba = model.predict_proba(X)[:, 1]
    return {
        "accuracy": accuracy_score(y, y_pred),
        "precision": precision_score(y, y_pred, zero_division=0),
        "recall": recall_score(y, y_pred),
        "f1": f1_score(y, y_pred),
        "roc_auc": roc_auc_score(y, y_proba),
        "pr_auc": average_precision_score(y, y_proba),
        "g_mean": g_mean_score(y, y_pred),
    }


def get_feature_names(preprocessor):
    """Имена признаков в том же порядке, что и после ColumnTransformer."""
    names = []
    for name, transformer, cols in preprocessor.transformers_:
        if name == "remainder":
            continue
        if hasattr(transformer, "get_feature_names_out"):
            names.extend(transformer.get_feature_names_out(cols).tolist())
        else:                                        # passthrough - берём исходные названия
            names.extend(list(cols) if not isinstance(cols, str) else [cols])
    return names


def save_fig(name, dpi=150):
    """Сохраняет текущий график в figures/ (папка не коммитится)."""
    FIG_DIR.mkdir(exist_ok=True)
    plt.savefig(FIG_DIR / f"{name}.png", dpi=dpi, bbox_inches="tight")


def plot_confusion(model, X, y, title="Матрица ошибок", ax=None):
    """Матрица ошибок для обученной модели (pipeline или оценщик)."""
    disp = ConfusionMatrixDisplay.from_estimator(model, X, y, display_labels=CLASS_NAMES,
                                                 cmap="Blues", values_format="d", ax=ax)
    disp.ax_.set_title(title)
    disp.ax_.grid(False)
    return disp


def plot_decision_tree(tree, feature_names, title="Дерево решений", figsize=(22, 9), fontsize=9):
    """Визуализация дерева через sklearn.tree.plot_tree с подписями признаков и классов."""
    plt.figure(figsize=figsize)
    plot_tree(tree, feature_names=list(feature_names), class_names=CLASS_NAMES,
              filled=True, rounded=True, fontsize=fontsize)
    plt.title(title)


def plot_top_importances(importances_df, title, top_n=15, figsize=(9, 6), xlabel="Важность признака"):
    """Горизонтальная столбчатая диаграмма топ-N признаков (таблица: feature, importance)."""
    import seaborn as sns
    fi = importances_df.head(top_n).iloc[::-1]       # разворачиваем, чтобы топ был сверху
    plt.figure(figsize=figsize)
    ax = sns.barplot(data=fi, x="importance", y="feature", hue="feature", palette="viridis", legend=False)
    ax.bar_label(ax.containers[0], fmt="%.4f", padding=3, fontsize=8)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("Признак")
    plt.tight_layout()
    return fi.iloc[::-1].reset_index(drop=True)


def make_pipeline_dt(preprocessor, **tree_params):
    """Pipeline: предобработка + DecisionTreeClassifier с class_weight='balanced'."""
    from sklearn.base import clone
    from sklearn.pipeline import Pipeline
    from sklearn.tree import DecisionTreeClassifier
    params = dict(class_weight="balanced", random_state=SEED)
    params.update(tree_params)
    return Pipeline([("prep", clone(preprocessor)), ("clf", DecisionTreeClassifier(**params))])
