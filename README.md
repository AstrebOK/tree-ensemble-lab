# Лабораторная работа № 6. Деревья решений и ансамблевые методы (вариант 3)

**Вариант 3:** датасет `credit_card_fraud_2026.csv` (выявление мошенничества с кредитными картами), методы – **Random Forest + Gradient Boosting + Stacking**.

## Аннотация
Бинарная классификация мошеннических транзакций (`is_fraud`) на несбалансированных данных (1,7% мошенничества, дисбаланс 1 : 58). Построены и сравнены одиночные деревья решений и три ансамбля,
гиперпараметры подобраны по validation-выборке, качество оценено кросс-валидацией и на независимом тесте, проведён анализ важности признаков.

## Данные
20 000 транзакций, 24 признака (13 числовых, 6 бинарных флагов, 5 категориальных) и целевая переменная `is_fraud`; пропусков нет. Файл в репозиторий не выкладывается: положите `credit_card_fraud_2026.csv`
в папку `data/` (см. `data/README.md`).

## Установка и запуск
```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cd notebooks && jupyter notebook                        # выполнить ноутбуки 01 -> 02 -> 03
pytest -q tests                                         # юнит-тесты
```

## Структура
```
notebooks/01_data_processing.ipynb       # этапы 0–2: загрузка, EDA, предобработка, разбиение
notebooks/02_decision_tree.ipynb         # этап 3: дерево решений
notebooks/03_ensembles_comparison.ipynb  # этапы 4–6: RF, GB, Stacking, важность признаков
src/common.py  src/models.py  src/metrics.py
tests/test_common.py
```

## Результаты (test, 4 000 объектов, из них 68 мошеннических)
| Модель | CV ROC-AUC | Test ROC-AUC | Test F1 | Test Recall | Test G-mean |
|:--|--:|--:|--:|--:|--:|
| Decision Tree (max_depth=3) | 0.794 | 0.822 | 0.139 | 0.529 | 0.688 |
| Decision Tree (min_samples_leaf=10) | 0.617 | 0.619 | 0.122 | 0.294 | 0.526 |
| Random Forest | 0.918 | 0.932 | 0.292 | 0.279 | 0.526 |
| Gradient Boosting | 0.925 | 0.933 | 0.199 | 0.809 | 0.849 |
| Stacking | 0.935 | 0.940 | 0.182 | 0.824 | 0.849 |

Лучший ансамбль по CV ROC-AUC – Stacking (0,935 ± 0,014). Stacking и Gradient Boosting находят более 80% мошеннических транзакций (порог 0,5), но с низкой precision (≈ 0,10–0,11);
Random Forest точнее (precision 0,31), но находит лишь 28%.

## Интерпретация
Наиболее важные признаки (MDI и permutation importance): `cvv_retry_count`, `merchant_risk_score`, `velocity_score`, `ip_country_mismatch`, отсутствие аутентификации (`auth_method = No Authentication`),
`billing_shipping_mismatch`, `is_new_merchant`. Это согласуется с EDA.

## Ограничения
* В тесте всего 68 мошеннических транзакций – метрики имеют заметный разброс, различия между ансамблями в пределах погрешности.
* Порог классификации 0,5 не оптимален для дисбаланса; в продакшене его подбирают по стоимости ошибок.
* Для Random Forest используется `class_weight="balanced_subsample"`: с `"balanced"` OOB-оценка в scikit-learn 1.9.1 вырождается.
* SMOTE не применялся (используются веса классов). SHAP summary plot строится ячейкой в ноутбуке 03 при установленном пакете `shap`.
