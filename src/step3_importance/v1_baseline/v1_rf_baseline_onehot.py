"""
BASELINE V1 (ONE-HOT): RANDOM FOREST CLASSIFIER

ОТЛИЧИЕ ОТ baseline_v1:
В baseline_v1 использовали ORDINAL ENCODING (ручное упорядочивание):
  - loan_grade: A=7, B=6, C=5, D=4, E=3, F=2, G=1
  - person_home_ownership: OWN=4, MORTGAGE=3, OTHER=2, RENT=1
  - Логика: выше число = лучше клиент = реже одобряют

В ЭТОЙ МОДЕЛИ используем ONE-HOT ENCODING (автоматическое):
  - loan_grade → 7 столбцов: grade_A, grade_B, ..., grade_G (0/1)
  - person_home_ownership → 4 столбца: home_OWN, home_MORTGAGE, home_OTHER, home_RENT (0/1)
  - Модель САМА решает какая категория важнее

ЦЕЛЬ СРАВНЕНИЯ:
Проверить влияет ли ручное упорядочивание категорий на качество модели.

ГИПЕРПАРАМЕТРЫ:
Точно такие же как в baseline_v1 для честного сравнения:
- n_estimators=100
- max_depth=10
- min_samples_split=20
- min_samples_leaf=10
- class_weight='balanced'
- random_state=42
- n_jobs=-1
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
from src.utils.plotting import plot_all_model_visualizations

# ============================================================================
# НАСТРОЙКИ
# ============================================================================

RANDOM_STATE = 42
TEST_SIZE = 0.2

# Пути
DATA_PATH = project_root / 'data' / 'raw' / 'train.csv'
OUTPUT_DIR = project_root / 'results' / 'step3_importance' / 'baseline_v1_onehot'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
(OUTPUT_DIR / 'tables').mkdir(exist_ok=True)
(OUTPUT_DIR / 'figures').mkdir(exist_ok=True)

# ============================================================================
# ЗАГРУЗКА И ПОДГОТОВКА ДАННЫХ
# ============================================================================

print("=" * 80)
print("BASELINE V1 (ONE-HOT): RANDOM FOREST CLASSIFIER")
print("=" * 80)
print()

print("Загрузка данных...")
df = pd.read_csv(DATA_PATH)
print(f"Загружено: {df.shape[0]:,} строк, {df.shape[1]} столбцов")
print()

# ============================================================================
# ONE-HOT ENCODING (автоматическое кодирование)
# ============================================================================

print("Применение One-Hot Encoding...")

# Удаляем id и таргет перед кодированием
X = df.drop(columns=['loan_status', 'id'])
y = df['loan_status']

# Определяем категориальные признаки
categorical_features = ['loan_grade', 'person_home_ownership', 'loan_intent', 'cb_person_default_on_file']

print(f"Категориальные признаки: {categorical_features}")
print()

# Применяем One-Hot Encoding
X_encoded = pd.get_dummies(X, columns=categorical_features, drop_first=False, dtype=int)

print("One-Hot Encoding применён:")
print(f"  - Было признаков: {X.shape[1]}")
print(f"  - Стало признаков: {X_encoded.shape[1]}")
print(f"  - Добавлено: {X_encoded.shape[1] - X.shape[1]} dummy переменных")
print()
print("Новые столбцы:")
new_cols = [col for col in X_encoded.columns if col not in X.columns]
for col in new_cols:
    print(f"  - {col}")
print()

# ============================================================================
# TRAIN/VAL SPLIT
# ============================================================================

print("Train/Validation Split...")
X_train, X_val, y_train, y_val = train_test_split(
    X_encoded, y, 
    test_size=TEST_SIZE, 
    random_state=RANDOM_STATE, 
    stratify=y
)

print(f"Train: {X_train.shape[0]:,} строк, {X_train.shape[1]} признаков")
print(f"Validation: {X_val.shape[0]:,} строк, {X_val.shape[1]} признаков")
print()

# ============================================================================
# ОБУЧЕНИЕ RANDOM FOREST
# ============================================================================

print("Обучение Random Forest...")
print()
print("Гиперпараметры (те же что в baseline_v1 для честного сравнения):")
print("  - n_estimators=100")
print("  - max_depth=10")
print("  - min_samples_split=20")
print("  - min_samples_leaf=10")
print("  - class_weight='balanced'")
print("  - random_state=42")
print("  - n_jobs=-1")
print()

rf_model = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    min_samples_split=20,
    min_samples_leaf=10,
    class_weight='balanced',
    random_state=RANDOM_STATE,
    n_jobs=-1
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

# Specificity
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
print("РЕЗУЛЬТАТЫ BASELINE V1 (ONE-HOT ENCODING)")
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

# Feature Importance (топ-20, т.к. признаков много)
feature_importance = pd.DataFrame({
    'Feature': X_encoded.columns,
    'Importance': rf_model.feature_importances_
}).sort_values('Importance', ascending=False)

# Сохраняем все
fi_path = OUTPUT_DIR / 'tables' / 'feature_importance.csv'
feature_importance.to_csv(fi_path, index=False)
print(f"Сохранено: {fi_path}")

# Топ-20 для удобства
fi_top20_path = OUTPUT_DIR / 'tables' / 'feature_importance_top20.csv'
feature_importance.head(20).to_csv(fi_top20_path, index=False)
print(f"Сохранено (топ-20): {fi_top20_path}")

# ============================================================================
# ВИЗУАЛИЗАЦИЯ
# ============================================================================

print()
print("Создание визуализаций...")

figures_dir = str(OUTPUT_DIR / 'figures')
plot_all_model_visualizations(
    y_true=y_val,
    y_pred=y_pred,
    y_pred_proba=y_pred_proba,
    metrics_dict=metrics_dict,
    model_name="Baseline_v1_RF_onehot",
    save_dir=figures_dir
)

print()
print("=" * 80)
print("BASELINE V1 (ONE-HOT) ЗАВЕРШЁН!")
print("=" * 80)
print()
print(f"Результаты сохранены в: {OUTPUT_DIR}")
print()
print("СРАВНЕНИЕ С ORDINAL ENCODING:")
print("Теперь можно сравнить метрики из:")
print("  - results/step3_importance/baseline_v1/tables/metrics.csv (Ordinal)")
print("  - results/step3_importance/baseline_v1_onehot/tables/metrics.csv (One-Hot)")
print()