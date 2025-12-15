"""
Step 1.3: Baseline Model 0 (точка отсчёта)
==========================================

Цель: Создать простейшую модель для установки baseline (точки отсчёта).

Что делаем:
1. Загружаем данные
2. Разделяем на train/validation (80/20)
3. Обучаем Logistic Regression с дефолтными параметрами
4. Оцениваем качество на validation
5. Сохраняем метрики

Зачем нужен baseline:
Это "точка отсчёта" для всех будущих улучшений. Любая новая модель должна
быть ЛУЧШЕ baseline, иначе нет смысла в усложнениях.

Важно:
- Используем class_weight='balanced' из-за дисбаланса 6:1
- Основная метрика: ROC-AUC (не Accuracy!)
- Не делаем никаких преобразований данных (чистый baseline)
"""

import pandas as pd
import numpy as np
import os
import sys
from pathlib import Path

# =============================================================================
# АВТООПРЕДЕЛЕНИЕ КОРНЯ ПРОЕКТА
# =============================================================================

PROJECT_ROOT = Path(__file__).parent.parent.parent
os.chdir(PROJECT_ROOT)

# Добавляем корень проекта в sys.path чтобы работали импорты
sys.path.insert(0, str(PROJECT_ROOT))

# Scikit-learn для машинного обучения
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# Наши функции для расчёта метрик
from src.utils.metrics import calculate_all_metrics, print_metrics, save_metrics

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
RESULTS_PATH = 'results/step1_first_look/tables'

# Создаём папку для результатов
os.makedirs(RESULTS_PATH, exist_ok=True)

# Параметры
RANDOM_STATE = 42  # Для воспроизводимости результатов
TEST_SIZE = 0.2    # 20% данных на validation

print("="*60)
print("STEP 1.3: BASELINE MODEL 0")
print("="*60)

# =============================================================================
# 1. ЗАГРУЗКА ДАННЫХ
# =============================================================================

print("\n1. Загрузка данных...")
df = pd.read_csv(DATA_PATH)
print(f" Загружено {len(df):,} записей")

# =============================================================================
# 2. ПОДГОТОВКА ДАННЫХ
# =============================================================================

print("\n2. Подготовка данных...")

# Удаляем ненужные столбцы
# - id: просто идентификатор, не несёт информации
# - cb_person_default_on_file: категориальный (пока не обрабатываем)
cols_to_drop = ['id', 'cb_person_default_on_file']

# Проверяем какие категориальные признаки есть
categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
print(f"\n   Категориальные признаки (пропускаем в baseline):")
for col in categorical_cols:
    print(f"      - {col}")

# Удаляем ВСЕ категориальные признаки в baseline
# (в следующих шагах будем их обрабатывать)
cols_to_drop.extend(categorical_cols)

# Признаки (X) и целевая переменная (y)
X = df.drop(columns=cols_to_drop + ['loan_status'])
y = df['loan_status']

print(f"\n   Признаков для обучения: {X.shape[1]}")
print(f"   Список признаков:")
for i, col in enumerate(X.columns, 1):
    print(f"      {i}. {col}")

# =============================================================================
# 3. РАЗДЕЛЕНИЕ НА TRAIN/VALIDATION
# =============================================================================

print(f"\n3. Разделение на train/validation ({int((1-TEST_SIZE)*100)}/{int(TEST_SIZE*100)})...")

# train_test_split разделяет данные случайным образом
# stratify=y сохраняет пропорцию классов (6:1) в обеих выборках
X_train, X_val, y_train, y_val = train_test_split(
    X, y,
    test_size=TEST_SIZE,      # 20% на validation
    random_state=RANDOM_STATE, # Для воспроизводимости
    stratify=y                 # Сохраняем пропорцию классов
)

print(f"\n   Train: {len(X_train):,} записей")
print(f"   Validation: {len(X_val):,} записей")

# Проверяем что пропорция классов сохранилась
train_ratio = (y_train == 0).sum() / (y_train == 1).sum()
val_ratio = (y_val == 0).sum() / (y_val == 1).sum()
print(f"\n   Соотношение классов в train: {train_ratio:.2f}:1")
print(f"   Соотношение классов в validation: {val_ratio:.2f}:1")

# =============================================================================
# 4. ОБУЧЕНИЕ МОДЕЛИ
# =============================================================================

print("\n4. Обучение Logistic Regression...")

# Создаём модель
# class_weight='balanced' автоматически подбирает веса классов
# чтобы компенсировать дисбаланс 6:1
model = LogisticRegression(
    class_weight='balanced',  # КРИТИЧНО для несбалансированных данных!
    random_state=RANDOM_STATE,
    max_iter=1000            # Максимум итераций для сходимости
)

# Обучаем модель
# fit() находит оптимальные веса для признаков
model.fit(X_train, y_train)

print(" Модель обучена")

# =============================================================================
# 5. ПРЕДСКАЗАНИЯ
# =============================================================================

print("\n5. Получение предсказаний на validation...")

# predict() возвращает предсказанный класс (0 или 1)
y_pred = model.predict(X_val)

# predict_proba() возвращает вероятности для каждого класса
# [:, 1] берём вероятность класса 1 (одобрен)
y_pred_proba = model.predict_proba(X_val)[:, 1]

print(f" Получено {len(y_pred):,} предсказаний")

# =============================================================================
# 6. ОЦЕНКА КАЧЕСТВА
# =============================================================================

print("\n6. Оценка качества модели...")

# Используем нашу функцию из src/utils/metrics.py
# Она считает все метрики сразу: ROC-AUC, Accuracy, Precision, Recall, F1, Confusion Matrix
metrics = calculate_all_metrics(y_val, y_pred, y_pred_proba)

# Красиво выводим метрики
print_metrics(metrics, title="Baseline Model 0 - Validation Metrics")

# =============================================================================
# 7. ИНТЕРПРЕТАЦИЯ РЕЗУЛЬТАТОВ
# =============================================================================

print("\n" + "="*60)
print("ИНТЕРПРЕТАЦИЯ МЕТРИК:")
print("="*60)

print(f"\n ROC-AUC: {metrics['ROC-AUC']:.4f}")
print(f"   - Основная метрика для несбалансированных данных")
print(f"   - 0.5 = случайное угадывание")
print(f"   - 1.0 = идеальная модель")
if metrics['ROC-AUC'] > 0.7:
    print(f"   -  {metrics['ROC-AUC']:.4f} = хороший результат для baseline!")
elif metrics['ROC-AUC'] > 0.6:
    print(f"   -   {metrics['ROC-AUC']:.4f} = средний результат, есть куда расти")
else:
    print(f"   -  {metrics['ROC-AUC']:.4f} = слабый результат")

print(f"\n Accuracy: {metrics['Accuracy']:.4f}")
print(f"   - НЕ главная метрика при дисбалансе!")
print(f"   - Может быть высокой даже если модель плохая")
print(f"   - Если предсказывать всегда 0 → Accuracy = 85.76%")

print(f"\n Precision (для класса 1): {metrics['Precision']:.4f}")
print(f"   - Из тех, кого модель предсказала как 'одобрен'")
print(f"   - Сколько реально одобрены")
print(f"   - Важно если НЕ хотим ошибочно одобрять кредиты")

print(f"\n Recall (для класса 1): {metrics['Recall']:.4f}")
print(f"   - Из всех реально одобренных")
print(f"   - Сколько модель нашла")
print(f"   - Важно если НЕ хотим пропускать одобренные кредиты")

print(f"\n F1-Score: {metrics['F1-Score']:.4f}")
print(f"   - Баланс между Precision и Recall")
print(f"   - Гармоническое среднее")

print(f"\n Confusion Matrix:")
print(f"   TN={metrics['TN']:,} | FP={metrics['FP']:,}")
print(f"   FN={metrics['FN']:,} | TP={metrics['TP']:,}")
print(f"\n   TN (True Negative): правильно предсказали 'НЕ одобрен'")
print(f"   FP (False Positive): ошибочно предсказали 'одобрен'")
print(f"   FN (False Negative): пропустили 'одобрен'")
print(f"   TP (True Positive): правильно предсказали 'одобрен'")

# =============================================================================
# 8. СОХРАНЕНИЕ РЕЗУЛЬТАТОВ
# =============================================================================

print("\n" + "="*60)
print("СОХРАНЕНИЕ РЕЗУЛЬТАТОВ:")
print("="*60)

# Добавляем название модели
metrics['Model'] = 'Baseline_0_LogisticRegression'

# Сохраняем в CSV
output_file = os.path.join(RESULTS_PATH, '01_baseline_0_metrics.csv')
save_metrics(metrics, output_file)

# =============================================================================
# 9. СОЗДАНИЕ ГРАФИКОВ
# =============================================================================

print("\n9. Создание графиков...")

# Импортируем функции визуализации
from src.utils.plotting import plot_all_model_visualizations

# Папка для графиков
FIGURES_PATH = 'results/step1_first_look/figures'

# Создаём все графики
plot_all_model_visualizations(
    y_true=y_val,
    y_pred=y_pred,
    y_pred_proba=y_pred_proba,
    metrics_dict=metrics,
    model_name='Baseline_0',
    save_dir=FIGURES_PATH
)

# =============================================================================
# ЗАВЕРШЕНИЕ
# =============================================================================

print("\n" + "="*60)
print("STEP 1.3 ЗАВЕРШЁН")
print("="*60)
print(f"\n Baseline модель создана: Logistic Regression")
print(f" ROC-AUC на validation: {metrics['ROC-AUC']:.4f}")
print(f" Метрики сохранены: {output_file}")
print(f"\n Это наша точка отсчёта!")
print(f" Все следующие модели должны быть ЛУЧШЕ {metrics['ROC-AUC']:.4f}")
print("\nСледующий этап: Step 2 - EDA (детальный анализ данных)")