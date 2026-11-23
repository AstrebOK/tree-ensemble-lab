"""Ансамблевые модели варианта 3: Random Forest, Gradient Boosting, Stacking."""
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_sample_weight

SEED = 42


class BalancedGradientBoostingClassifier(GradientBoostingClassifier):
    """GradientBoostingClassifier из sklearn не имеет class_weight, поэтому веса объектов
    считаются внутри fit по принципу class_weight='balanced' (аналог scale_pos_weight в XGBoost).
    Так модель корректно работает внутри cross_validate и StackingClassifier."""

    def fit(self, X, y, sample_weight=None, monitor=None):
        sw = compute_sample_weight("balanced", y)
        if sample_weight is not None:
            sw = sw * np.asarray(sample_weight)
        return super().fit(X, y, sample_weight=sw, monitor=monitor)


def build_rf(**params):
    """Random Forest с балансировкой классов и OOB-оценкой.

    class_weight="balanced_subsample": тот же принцип, что "balanced", но веса пересчитываются
    в каждой bootstrap-выборке. С "balanced" OOB-оценка в используемой версии scikit-learn (1.9.1)
    вырождается (ROC-AUC = 0.0), поэтому для Random Forest выбран "balanced_subsample".
    """
    base = dict(n_estimators=200, class_weight="balanced_subsample", oob_score=True, n_jobs=-1, random_state=SEED)
    base.update(params)
    return RandomForestClassifier(**base)


def build_gb(**params):
    """Gradient Boosting (GBDT) с балансировкой классов через веса объектов."""
    base = dict(n_estimators=200, learning_rate=0.1, max_depth=3, random_state=SEED)
    base.update(params)
    return BalancedGradientBoostingClassifier(**base)


def build_stacking(rf_params=None, gb_params=None, passthrough=False, cv=5):
    """Stacking: базовые модели RF + GB + логистическая регрессия, метамодель - логистическая регрессия.

    Разнообразие базовых моделей (два бэггинг/бустинг-ансамбля на деревьях и линейная модель)
    даёт метамодели менее скоррелированные предсказания. Мета-признаки - вероятности класса 1,
    полученные out-of-fold (cv). passthrough=True добавляет к мета-признакам исходные признаки.
    """
    rf_params = dict(rf_params or {}, oob_score=False)      # OOB внутри стекинга не нужен
    lr_base = Pipeline([("scaler", StandardScaler()),
                        ("lr", LogisticRegression(class_weight="balanced", max_iter=2000, random_state=SEED))])
    meta = Pipeline([("scaler", StandardScaler()),
                     ("lr", LogisticRegression(class_weight="balanced", max_iter=2000, random_state=SEED))])
    return StackingClassifier(
        estimators=[("rf", build_rf(**rf_params)), ("gb", build_gb(**(gb_params or {}))), ("lr", lr_base)],
        final_estimator=meta, stack_method="predict_proba", cv=cv, passthrough=passthrough, n_jobs=1)


def make_pipeline(preprocessor, clf):
    """Предобработка + модель в одном Pipeline (предобработка обучается только на train)."""
    from sklearn.base import clone
    return Pipeline([("prep", clone(preprocessor)), ("clf", clf)])
