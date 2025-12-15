"""
Step 1.2: Визуализация дисбаланса классов
=========================================

Цель: Наглядно показать дисбаланс классов через графики.

Что делаем:
1. Загружаем данные
2. Считаем распределение loan_status (0/1)
3. Строим 2 графика:
   - Столбчатая диаграмма (bar chart)
   - Круговая диаграмма (pie chart)
4. Сохраняем графики в PNG

Зачем нужно:
Дисбаланс 6:1 означает что модель будет склонна предсказывать класс 0
(не одобрен). Нужно учитывать это при обучении через class_weight='balanced'.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from pathlib import Path

# =============================================================================
# АВТООПРЕДЕЛЕНИЕ КОРНЯ ПРОЕКТА
# =============================================================================

PROJECT_ROOT = Path(__file__).parent.parent.parent
os.chdir(PROJECT_ROOT)

# =============================================================================
# НАСТРОЙКИ
# =============================================================================

# Пути
DATA_PATH = 'data/raw/train.csv'
FIGURES_PATH = 'results/step1_first_look/figures'

# Создаём папку для графиков
os.makedirs(FIGURES_PATH, exist_ok=True)

# Настройки графиков
# plt.style.use('seaborn-v0_8-darkgrid')  # Стиль для красоты
sns.set_palette("husl")  # Цветовая палитра

print("="*60)
print("STEP 1.2: ВИЗУАЛИЗАЦИЯ ДИСБАЛАНСА КЛАССОВ")
print("="*60)

# =============================================================================
# ЗАГРУЗКА ДАННЫХ
# =============================================================================

print("\n1. Загрузка данных...")
df = pd.read_csv(DATA_PATH)
print(f" Загружено {len(df):,} записей")

# =============================================================================
# ПОДСЧЁТ РАСПРЕДЕЛЕНИЯ
# =============================================================================

print("\n2. Анализ целевой переменной...")

# Считаем количество каждого класса
target_counts = df['loan_status'].value_counts().sort_index()
target_pct = df['loan_status'].value_counts(normalize=True).sort_index() * 100

print(f"\n   Класс 0 (НЕ одобрен): {target_counts[0]:,} ({target_pct[0]:.2f}%)")
print(f"   Класс 1 (одобрен):    {target_counts[1]:,} ({target_pct[1]:.2f}%)")
print(f"   Соотношение: {target_counts[0]/target_counts[1]:.2f}:1")

# =============================================================================
# ВИЗУАЛИЗАЦИЯ 1: СТОЛБЧАТАЯ ДИАГРАММА
# =============================================================================

print("\n3. Создание графиков...")

# Создаём фигуру с 2 подграфиками (1 строка, 2 столбца)
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# --- График 1: Bar Chart ---
ax1 = axes[0]

# Строим столбцы
bars = ax1.bar(
    ['НЕ одобрен (0)', 'Одобрен (1)'],  # Метки по оси X
    target_counts.values,                 # Высота столбцов
    color=['#FF6B6B', '#4ECDC4'],        # Цвета
    edgecolor='black',                    # Обводка
    alpha=0.7                             # Прозрачность
)

# Добавляем значения НАД столбцами
for i, (count, pct) in enumerate(zip(target_counts.values, target_pct.values)):
    ax1.text(
        i,                                # Позиция по X
        count + 1000,                     # Позиция по Y (чуть выше столбца)
        f'{count:,}\n({pct:.1f}%)',      # Текст
        ha='center',                      # Выравнивание по горизонтали
        va='bottom',                      # Выравнивание по вертикали
        fontsize=12,
        fontweight='bold'
    )

# Настройки графика
ax1.set_ylabel('Количество заявок', fontsize=12, fontweight='bold')
ax1.set_title('Распределение классов (loan_status)', fontsize=14, fontweight='bold')
ax1.grid(axis='y', alpha=0.3)  # Сетка только по Y
ax1.set_ylim(0, max(target_counts.values) * 1.15)  # Чуть выше макс значения

# =============================================================================
# ВИЗУАЛИЗАЦИЯ 2: КРУГОВАЯ ДИАГРАММА
# =============================================================================

# --- График 2: Pie Chart ---
ax2 = axes[1]

# Строим круг
wedges, texts, autotexts = ax2.pie(
    target_counts.values,                          # Значения
    labels=['НЕ одобрен (0)', 'Одобрен (1)'],     # Метки
    autopct='%1.1f%%',                             # Формат процентов
    colors=['#FF6B6B', '#4ECDC4'],                # Цвета
    explode=[0.05, 0],                             # "Выдвинуть" первый сектор
    startangle=90,                                 # Начальный угол
    textprops={'fontsize': 12, 'fontweight': 'bold'}
)

# Делаем проценты белыми и жирными
for autotext in autotexts:
    autotext.set_color('white')
    autotext.set_fontweight('bold')
    autotext.set_fontsize(14)

ax2.set_title('Доля классов', fontsize=14, fontweight='bold')

# =============================================================================
# СОХРАНЕНИЕ
# =============================================================================

# Компактное размещение элементов
plt.tight_layout()

# Сохраняем
output_file = os.path.join(FIGURES_PATH, '01_target_distribution.png')
plt.savefig(output_file, dpi=300, bbox_inches='tight')
print(f"\n График сохранён: {output_file}")

# Показываем график (закомментируй если не нужно)
# plt.show()

plt.close()

# =============================================================================
# ВЫВОДЫ
# =============================================================================

print("\n" + "="*60)
print("КЛЮЧЕВЫЕ ВЫВОДЫ:")
print("="*60)

print(f"\n СИЛЬНЫЙ ДИСБАЛАНС КЛАССОВ: {target_counts[0]/target_counts[1]:.1f}:1")
print(f"\n   Это означает:")
print(f"   - На каждую ОДОБРЕННУЮ заявку приходится {target_counts[0]/target_counts[1]:.0f} ОТКЛОНЁННЫХ")
print(f"   - Модель без настроек будет склонна всегда предсказывать класс 0")
print(f"\n  РЕШЕНИЕ:")
print(f"   - Использовать class_weight='balanced' при обучении")
print(f"   - Метрика: ROC-AUC (не Accuracy!)")
print(f"   - Фокус на Recall для класса 1 (не пропустить одобренные)")

print("\n" + "="*60)
print("STEP 1.2 ЗАВЕРШЁН")
print("="*60)
print(f"\nГрафик сохранён в: {FIGURES_PATH}")
print("\nСледующий шаг: python src/step1_first_look/step1_3_baseline_0.py")