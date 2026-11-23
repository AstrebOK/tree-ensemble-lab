# -*- coding: utf-8 -*-
"""
Лабораторная работа №7. Анализ многомерных данных.
Вариант 11: «Климат и экология».
Признаки: X1 - температура, X2 - давление, X3 - влажность, X4 - осадки.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Ellipse
from scipy.stats import chi2

# Настройки для красивых и четких графиков
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams['figure.figsize'] = (15, 11)
plt.rcParams['font.size'] = 11
plt.rcParams['axes.titlesize'] = 13
plt.rcParams['axes.titleweight'] = 'bold'

# ==========================================
# 1. Генерация данных (имитация метеоданных)
# ==========================================
np.random.seed(11)
N = 120

cov_matrix = np.array([
    [25.0, -15.0,  0.5,  0.2],
    [-15.0, 20.0, -0.3,  0.1],
    [0.5,  -0.3,  15.0,  8.0],
    [0.2,   0.1,   8.0,  6.0]
])
mean_vector = [20, 750, 60, 5]

data = np.random.multivariate_normal(mean_vector, cov_matrix, size=N)

# 2 аномальных дня (экстремальная жара и мороз)
data[0] = [38, 720, 90, 50]
data[1] = [-15, 780, 30, 0]

df = pd.DataFrame(data, columns=['X1_Темп', 'X2_Давление', 'X3_Влажность', 'X4_Осадки'])

print("="*70)
print("ВАРИАНТ 11: АНАЛИЗ МЕТЕОДАННЫХ")
print("="*70)

# ==========================================
# ЗАДАНИЕ 1 & 2. Вычисления
# ==========================================
S = df.cov().values
R = df.corr().values
det_S = np.linalg.det(S)
eigvals = np.linalg.eigvalsh(S)
cond_S = eigvals[-1] / eigvals[0]

p = R.shape[0]
sign, logdet_R = np.linalg.slogdet(R)
chi2_stat = -(N - 1 - (2*p + 5)/6) * logdet_R
p_value = chi2.sf(chi2_stat, p*(p-1)//2)

print(f"Опред. |S| = {det_S:.3e}, cond(S) = {cond_S:.2f}")
print(f"Критерий Бартлетта: p-value = {p_value:.3e} (H0 отклоняется)")

# ==========================================
# ЗАДАНИЕ 3. Параметры для эллипсов
# ==========================================
mu_12 = df[['X1_Темп', 'X2_Давление']].mean().values
S_12 = df[['X1_Темп', 'X2_Давление']].cov().values

vals, vecs = np.linalg.eigh(S_12)
order = np.argsort(vals)[::-1]
vals, vecs = vals[order], vecs[:, order]
angle = np.degrees(np.arctan2(vecs[1, 0], vecs[0, 0]))
if angle <= -90: angle += 180
if angle > 90: angle -= 180

# ==========================================
# ЗАДАНИЕ 4. ВИЗУАЛИЗАЦИЯ
# ==========================================
inv_S = np.linalg.inv(S)
diff = df.values - df.mean().values
d2 = np.sum(diff @ inv_S * diff, axis=1)

fig, axes = plt.subplots(2, 2, figsize=(15, 11))
fig.suptitle("Вариант 11. Анализ метеоданных", fontsize=18, fontweight='bold', y=0.98)

# 1. Рассеяние (Температура, Давление)
ax1 = axes[0, 0]
ax1.scatter(df['X1_Темп'], df['X2_Давление'], alpha=0.7, color='#3498db', edgecolor='k', s=50)
for c2, ls, label in [(2.30, '--', '68%'), (5.99, '-', '95%')]:
    # ИСПРАВЛЕНИЕ: w и h - это скаляры
    w = 2 * np.sqrt(vals[0] * c2)
    h = 2 * np.sqrt(vals[1] * c2)
    e = Ellipse(xy=mu_12, width=w, height=h, angle=angle,
                fill=False, edgecolor='#e74c3c', linestyle=ls, linewidth=2, label=f'Эллипс {label}')
    ax1.add_patch(e)
ax1.plot(mu_12[0], mu_12[1], 'k+', markersize=14, markeredgewidth=2, label='Центр')
ax1.set_title(f"Рассеяние (Температура, Давление)\nНаклон главной оси: {angle:.1f}°")
ax1.set_xlabel("Температура (°C)")
ax1.set_ylabel("Давление (мм рт. ст.)")
ax1.legend(loc='lower left', framealpha=0.9)

# 2. Рассеяние (Влажность, Осадки)
ax2 = axes[0, 1]
sns.scatterplot(data=df, x='X3_Влажность', y='X4_Осадки', color='#2ecc71',
                alpha=0.7, edgecolor='k', s=50, ax=ax2)
ax2.set_title("Связь влажности и осадков")
ax2.set_xlabel("Влажность (%)")
ax2.set_ylabel("Осадки (мм)")

# 3. Тепловая карта
ax3 = axes[1, 0]
sns.heatmap(pd.DataFrame(R, columns=df.columns, index=df.columns),
            annot=True, cmap='RdBu_r', fmt=".2f", ax=ax3, vmin=-1, vmax=1,
            linewidths=0.5, linecolor='white', annot_kws={"size": 12})
ax3.set_title("Корреляционная матрица R\n(визуализация блочной структуры)")

# 4. QQ-график
ax4 = axes[1, 1]
sorted_d2 = np.sort(d2)
theoretical_q = chi2.ppf((np.arange(1, N + 1) - 0.5) / N, df=p)

# Доверительный коридор (±30% от теоретического значения)
lower_bound = theoretical_q * 0.7
upper_bound = theoretical_q * 1.3

ax4.fill_between(theoretical_q, lower_bound, upper_bound,
                 color='gray', alpha=0.3, label='Зона нормы (±30%)')

ax4.scatter(theoretical_q, sorted_d2, alpha=0.8, color='#9b59b6', edgecolor='k', s=40, label='Наблюдения')

# ИСПРАВЛЕНИЕ: расширяем лимиты осей, чтобы выбросы точно попали в кадр
max_val = max(max(theoretical_q), max(sorted_d2)) * 1.15
ax4.plot([0, max_val], [0, max_val], 'r--', linewidth=2, label='Идеал (y=x)')

# Выделяем 2 выброса
ax4.scatter(theoretical_q[-2:], sorted_d2[-2:], color='red', edgecolor='darkred', s=100, zorder=5, label='Выбросы')

ax4.set_title("QQ-график Махаланобиса")
# ИСПРАВЛЕНИЕ: правильный вывод LaTeX-символа chi
ax4.set_xlabel(r"Теоретические квантили $\chi^2_4$")
ax4.set_ylabel(r"Выборочные $d^2$")
ax4.set_xlim(-2, max_val)
ax4.set_ylim(-2, max_val)

# ИСПРАВЛЕНИЕ: легенда сдвинута в правый нижний угол (lower right), чтобы не закрывать выбросы
ax4.legend(loc='lower right', framealpha=0.9)

plt.tight_layout()
plt.show()