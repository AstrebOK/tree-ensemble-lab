"""Юнит-тесты (на синтетических данных - датасет для запуска не нужен). Запуск: pytest -q"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from common import get_feature_lists, make_preprocessor, split_data, split_train_val  # noqa: E402
from metrics import entropy, g_mean_score, gini, information_gain  # noqa: E402


def make_df(n=1000, fraud_share=0.05, seed=0):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "transaction_id": np.arange(n),
        "amount_usd": rng.exponential(100, n),
        "is_foreign_transaction": rng.integers(0, 2, n),
        "channel": rng.choice(["Online", "POS", "ATM"], n),
        "is_fraud": (rng.random(n) < fraud_share).astype(int),
    })


def test_split_is_stratified_and_has_no_id():
    df = make_df()
    X_tr, X_te, y_tr, y_te = split_data(df)
    assert len(X_te) == 200 and len(X_tr) == 800
    assert abs(y_tr.mean() - y_te.mean()) < 0.01          # доли классов сохранены (stratify)
    assert "transaction_id" not in X_tr.columns and "is_fraud" not in X_tr.columns


def test_split_train_val_does_not_touch_test():
    df = make_df()
    X_tr, X_te, y_tr, _ = split_data(df)
    X_fit, X_val, _, _ = split_train_val(X_tr, y_tr)
    assert set(X_val.index).isdisjoint(X_te.index) and set(X_fit.index).isdisjoint(X_val.index)


def test_preprocessor_handles_unknown_category():
    df = make_df()
    X_tr, X_te, _, _ = split_data(df)
    num, cat = get_feature_lists(df)
    prep = make_preprocessor(num, cat).fit(X_tr)
    X_new = X_te.copy()
    X_new["channel"] = "Неизвестный канал"                # handle_unknown="ignore" - ошибки быть не должно
    assert prep.transform(X_new).shape[0] == len(X_new)


def test_entropy_gini_and_information_gain():
    y = np.array([0, 0, 1, 1])
    assert np.isclose(entropy(y), 1.0) and np.isclose(gini(y), 0.5)
    perfect = np.array([True, True, False, False])         # идеальное разбиение
    assert np.isclose(information_gain(y, perfect), 1.0)


def test_g_mean():
    y_true = np.array([0, 0, 0, 0, 1, 1])
    y_pred = np.array([0, 0, 0, 1, 1, 0])                  # recall(+)=0.5, recall(-)=0.75
    assert np.isclose(g_mean_score(y_true, y_pred), np.sqrt(0.5 * 0.75))
