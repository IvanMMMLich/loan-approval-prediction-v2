"""
BASELINE V1: RANDOM FOREST CLASSIFIER

КОНТЕКСТ:
После анализа на Step 3 (19 методов feature importance и inter-feature correlations)
выявили следующее:
1. Мультиколлинеарность между признаками (loan_grade ↔ loan_int_rate: η²=0.907)
2. Нелинейные зависимости (Mutual Information показал пороговую логику)
3. Категориальные признаки с множественными категориями

РЕШЕНИЕ:
Переход на tree-based модели (Random Forest), так как они:
- Устойчивы к мультиколлинеарности
- Ловят нелинейные зависимости
- Хорошо работают с категориальными признаками
- Не требуют feature scaling

ГИПЕРПАРАМЕТРЫ:
- n_estimators=100: количество деревьев в ансамбле (баланс скорость/качество)
- max_depth=10: максимальная глубина дерева (предотвращение переобучения)
- min_samples_split=20: минимум сэмплов для разделения узла (регуляризация)
- min_samples_leaf=10: минимум сэмплов в листе (сглаживание предсказаний)
- class_weight='balanced': учёт дисбаланса классов (85.8% / 14.2%)
- random_state=42: воспроизводимость результатов
- n_jobs=-1: параллельное выполнение (все ядра CPU)

КОНСИСТЕНТНОЕ КОДИРОВАНИЕ:
Применяем маппинги из Method 8 (Logistic Regression Coefficients):
- Логика: выше число = лучше клиент = РЕЖЕ одобряют (субстандартный кредитор)
"""

import sys
from pathlib import Path

# Добавляем корневую директорию проекта в sys.path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    roc_auc_score, accuracy_score, precision_score, recall_score, 
    f1_score, classification_report, confusion_matrix
)

# Импорт готовых функций визуализации из utils
from src.utils.plotting import (
    plot_confusion_matrix,
    plot_roc_curve,
    plot_all_metrics_bars,
    plot_precision_recall_f1,
    plot_all_model_visualizations
)

# ============================================================================
# НАСТРОЙКИ
# ============================================================================

RANDOM_STATE = 42
TEST_SIZE = 0.2

# Пути
DATA_PATH = project_root / 'data' / 'raw' / 'train.csv'
OUTPUT_DIR = project_root / 'results' / 'step3_importance' / 'baseline_v1'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
(OUTPUT_DIR / 'tables').mkdir(exist_ok=True)
(OUTPUT_DIR / 'figures').mkdir(exist_ok=True)

# ============================================================================
# КОНСИСТЕНТНОЕ КОДИРОВАНИЕ (из Method 8)
# ============================================================================

# Логика: выше число = лучше клиент = РЕЖЕ одобряют
GRADE_MAPPING = {
    'A': 7,  # Лучший грейд: 4.92% approval
    'B': 6,
    'C': 5,
    'D': 4,
    'E': 3,
    'F': 2,
    'G': 1   # Худший грейд: 81.82% approval
}

DEFAULT_MAPPING = {
    'N': 1,  # Не было дефолта (лучше)
    'Y': 0   # Был дефолт (хуже)
}

HOME_MAPPING = {
    'OWN': 4,       # 1.37% approval
    'MORTGAGE': 3,  # 5.97% approval
    'OTHER': 2,     # 16.85% approval
    'RENT': 1       # 22.26% approval
}

INTENT_MAPPING = {
    'VENTURE': 6,         # 9.28% approval
    'EDUCATION': 5,       # 10.48% approval
    'PERSONAL': 4,        # 12.35% approval
    'HOMEIMPROVEMENT': 3, # 13.55% approval
    'MEDICAL': 2,         # 16.67% approval
    'DEBTCONSOLIDATION': 1  # 18.93% approval
}

# ============================================================================
# ЗАГРУЗКА И ПОДГОТОВКА ДАННЫХ
# ============================================================================

print("=" * 80)
print("BASELINE V1: RANDOM FOREST CLASSIFIER")
print("=" * 80)
print()

print("Загрузка данных...")
df = pd.read_csv(DATA_PATH)
print(f"Загружено: {df.shape[0]:,} строк, {df.shape[1]} столбцов")
print()

print("Применение консистентного кодирования...")
# Категориальные признаки
df['loan_grade'] = df['loan_grade'].map(GRADE_MAPPING)
df['cb_person_default_on_file'] = df['cb_person_default_on_file'].map(DEFAULT_MAPPING)
df['person_home_ownership'] = df['person_home_ownership'].map(HOME_MAPPING)
df['loan_intent'] = df['loan_intent'].map(INTENT_MAPPING)

print("Кодирование применено:")
print(f"  - loan_grade: {GRADE_MAPPING}")
print(f"  - cb_person_default_on_file: {DEFAULT_MAPPING}")
print(f"  - person_home_ownership: {HOME_MAPPING}")
print(f"  - loan_intent: {INTENT_MAPPING}")
print()

# Разделение на признаки и таргет
X = df.drop(columns=['loan_status', 'id'])
y = df['loan_status']

print(f"Признаки (X): {X.shape[1]} столбцов")
print(f"Таргет (y): {y.value_counts().to_dict()}")
print(f"   Дисбаланс: {y.value_counts(normalize=True).to_dict()}")
print()

# ============================================================================
# TRAIN/VAL SPLIT
# ============================================================================

print("Train/Validation Split...")
X_train, X_val, y_train, y_val = train_test_split(
    X, y, 
    test_size=TEST_SIZE, 
    random_state=RANDOM_STATE, 
    stratify=y
)

print(f"Train: {X_train.shape[0]:,} строк")
print(f"Validation: {X_val.shape[0]:,} строк")
print()

# ============================================================================
# ОБУЧЕНИЕ RANDOM FOREST
# ============================================================================

print("Обучение Random Forest...")
print()
print("Гиперпараметры:")
print("  - n_estimators=100       # 100 деревьев (баланс скорость/качество)")
print("  - max_depth=10           # Глубина 10 (предотвращение переобучения)")
print("  - min_samples_split=20   # Мин. 20 сэмплов для разделения (регуляризация)")
print("  - min_samples_leaf=10    # Мин. 10 сэмплов в листе (сглаживание)")
print("  - class_weight='balanced' # Учёт дисбаланса 85.8% / 14.2%")
print("  - random_state=42        # Воспроизводимость")
print("  - n_jobs=-1              # Все ядра CPU")
print()

rf_model = RandomForestClassifier(
    n_estimators=100,          # Количество деревьев
    max_depth=10,              # Максимальная глубина
    min_samples_split=20,      # Минимум для разделения
    min_samples_leaf=10,       # Минимум в листе
    class_weight='balanced',   # Учёт дисбаланса
    random_state=RANDOM_STATE,
    n_jobs=-1                  # Параллельное выполнение
)

rf_model.fit(X_train, y_train)
print("Модель обучена!")
print()

# ============================================================================
# ПРЕДСКАЗАНИЯ
# ============================================================================

print("Предсказания на Validation...")
y_pred = rf_model.predict(X_val)
y_pred_proba = rf_model.predict_proba(X_val)[:, 1]
print("Предсказания готовы")
print()

# ============================================================================
# МЕТРИКИ
# ============================================================================

print("Расчёт метрик...")

# Основные метрики
roc_auc = roc_auc_score(y_val, y_pred_proba)
accuracy = accuracy_score(y_val, y_pred)
precision = precision_score(y_val, y_pred)
recall = recall_score(y_val, y_pred)
f1 = f1_score(y_val, y_pred)

# Specificity (для полноты)
cm = confusion_matrix(y_val, y_pred)
tn, fp, fn, tp = cm.ravel()
specificity = tn / (tn + fp)

# Словарь метрик для визуализации
metrics_dict = {
    'ROC-AUC': roc_auc,
    'Accuracy': accuracy,
    'Precision': precision,
    'Recall': recall,
    'F1-Score': f1,
    'Specificity': specificity
}

# Classification Report
class_report = classification_report(y_val, y_pred, output_dict=True)

print()
print("=" * 80)
print("РЕЗУЛЬТАТЫ BASELINE V1 (RANDOM FOREST)")
print("=" * 80)
print(f"ROC-AUC:      {roc_auc:.4f}")
print(f"Accuracy:     {accuracy:.4f} ({accuracy*100:.2f}%)")
print(f"Precision:    {precision:.4f}")
print(f"Recall:       {recall:.4f}")
print(f"F1-Score:     {f1:.4f}")
print(f"Specificity:  {specificity:.4f}")
print()
print("Confusion Matrix:")
print(f"TN: {tn:>6,}  |  FP: {fp:>6,}")
print(f"FN: {fn:>6,}  |  TP: {tp:>6,}")
print("=" * 80)
print()

# ============================================================================
# СОХРАНЕНИЕ МЕТРИК
# ============================================================================

print("Сохранение метрик...")

# Таблица основных метрик
metrics_df = pd.DataFrame({
    'Metric': ['ROC-AUC', 'Accuracy', 'Precision', 'Recall', 'F1-Score', 'Specificity'],
    'Value': [roc_auc, accuracy, precision, recall, f1, specificity]
})
metrics_path = OUTPUT_DIR / 'tables' / 'metrics.csv'
metrics_df.to_csv(metrics_path, index=False)
print(f"Сохранено: {metrics_path}")

# Classification Report
class_report_df = pd.DataFrame(class_report).transpose()
report_path = OUTPUT_DIR / 'tables' / 'classification_report.csv'
class_report_df.to_csv(report_path)
print(f"Сохранено: {report_path}")

# Feature Importance
feature_importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': rf_model.feature_importances_
}).sort_values('Importance', ascending=False)

fi_path = OUTPUT_DIR / 'tables' / 'feature_importance.csv'
feature_importance.to_csv(fi_path, index=False)
print(f"Сохранено: {fi_path}")

# ============================================================================
# ВИЗУАЛИЗАЦИЯ (используем готовые функции из plotting.py)
# ============================================================================

print()
print("Создание визуализаций...")

# Используем функцию plot_all_model_visualizations для создания всех 4 графиков
figures_dir = str(OUTPUT_DIR / 'figures')
plot_all_model_visualizations(
    y_true=y_val,
    y_pred=y_pred,
    y_pred_proba=y_pred_proba,
    metrics_dict=metrics_dict,
    model_name="Baseline_v1_RF",
    save_dir=figures_dir
)

print()
print("=" * 80)
print("BASELINE V1 ЗАВЕРШЁН!")
print("=" * 80)
print()
print(f"Результаты сохранены в: {OUTPUT_DIR}")
print()