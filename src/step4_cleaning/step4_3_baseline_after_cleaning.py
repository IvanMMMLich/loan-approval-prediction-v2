"""
Step 4.3: Baseline модель после очистки данных

Описание:
Запускаем простой Random Forest на очищенных данных (после step4_1 и step4_2),
чтобы оценить влияние очистки на качество модели.

Что было сделано до этого:
    step4_1: Удалены признаки (id, person_age, cb_person_cred_hist_length)
    step4_2: Capping выбросов (person_income, person_emp_length)

Модель:
    Random Forest с базовыми параметрами (без тюнинга)
    
Кодирование:
    Категориальные признаки кодируются LabelEncoder
    (временное решение, в step5 сделаем правильное Ordinal/OneHot)
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, 
    f1_score, roc_auc_score, confusion_matrix,
    classification_report
)
import sys

# Добавляем путь к utils
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.append(str(PROJECT_ROOT / 'src'))

from utils.plotting import plot_all_model_visualizations

# ============================================================================
# НАСТРОЙКИ
# ============================================================================

INPUT_DIR = PROJECT_ROOT / 'data' / 'processed'
RESULTS_DIR = PROJECT_ROOT / 'results' / 'step4_cleaning' / 'baseline_after_cleaning'

TRAIN_FILE = INPUT_DIR / 'train_step4_2.csv'

TARGET_COL = 'loan_status'

CATEGORICAL_FEATURES = [
    'person_home_ownership',
    'loan_intent',
    'loan_grade',
    'cb_person_default_on_file'
]

MODEL_NAME = 'Baseline_v2_after_cleaning'

# ============================================================================
# ГЛАВНАЯ ФУНКЦИЯ
# ============================================================================

def main():
    print("\n" + "="*70)
    print("STEP 4.3: BASELINE МОДЕЛЬ ПОСЛЕ ОЧИСТКИ")
    print("="*70 + "\n")
    
    # Создание папок
    figures_dir = RESULTS_DIR / 'figures'
    tables_dir = RESULTS_DIR / 'tables'
    figures_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)
    
    # -------------------------------------------------------------------------
    # 1. ЗАГРУЗКА ДАННЫХ
    # -------------------------------------------------------------------------
    print("1. ЗАГРУЗКА ДАННЫХ")
    print("-" * 40)
    
    df = pd.read_csv(TRAIN_FILE)
    print(f"\n   Файл: {TRAIN_FILE.name}")
    print(f"   Размер: {df.shape[0]:,} строк × {df.shape[1]} столбцов")
    print(f"\n   Признаки: {list(df.columns)}")
    print()
    
    # -------------------------------------------------------------------------
    # 2. ПОДГОТОВКА ДАННЫХ
    # -------------------------------------------------------------------------
    print("2. ПОДГОТОВКА ДАННЫХ")
    print("-" * 40)
    
    # Разделение на X и y
    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]
    
    print(f"\n   X: {X.shape}")
    print(f"   y: {y.shape}")
    print(f"\n   Баланс классов:")
    print(f"      0 (отказ):    {(y == 0).sum():,} ({(y == 0).mean()*100:.2f}%)")
    print(f"      1 (одобрено): {(y == 1).sum():,} ({(y == 1).mean()*100:.2f}%)")
    
    # Кодирование категориальных признаков
    print(f"\n   Кодирование категориальных признаков (LabelEncoder):")
    
    X_encoded = X.copy()
    label_encoders = {}
    
    for feature in CATEGORICAL_FEATURES:
        if feature in X_encoded.columns:
            le = LabelEncoder()
            X_encoded[feature] = le.fit_transform(X_encoded[feature])
            label_encoders[feature] = le
            print(f"      {feature}: {len(le.classes_)} категорий")
    
    print()
    
    # -------------------------------------------------------------------------
    # 3. РАЗДЕЛЕНИЕ НА TRAIN/VALIDATION
    # -------------------------------------------------------------------------
    print("3. РАЗДЕЛЕНИЕ НА TRAIN/VALIDATION")
    print("-" * 40)
    
    X_train, X_val, y_train, y_val = train_test_split(
        X_encoded, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )
    
    print(f"\n   Train: {X_train.shape[0]:,} строк ({X_train.shape[0]/len(X)*100:.0f}%)")
    print(f"   Val:   {X_val.shape[0]:,} строк ({X_val.shape[0]/len(X)*100:.0f}%)")
    print()
    
    # -------------------------------------------------------------------------
    # 4. ОБУЧЕНИЕ МОДЕЛИ
    # -------------------------------------------------------------------------
    print("4. ОБУЧЕНИЕ RANDOM FOREST")
    print("-" * 40)
    
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_split=20,
        min_samples_leaf=10,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )
    
    print(f"\n   Параметры модели:")
    print(f"      n_estimators:     {model.n_estimators}")
    print(f"      max_depth:        {model.max_depth}")
    print(f"      min_samples_split: {model.min_samples_split}")
    print(f"      min_samples_leaf: {model.min_samples_leaf}")
    print(f"      class_weight:     {model.class_weight}")
    
    print(f"\n   Обучение...")
    model.fit(X_train, y_train)
    print(f"   ✓ Модель обучена")
    print()
    
    # -------------------------------------------------------------------------
    # 5. ПРЕДСКАЗАНИЯ
    # -------------------------------------------------------------------------
    print("5. ПРЕДСКАЗАНИЯ НА VALIDATION")
    print("-" * 40)
    
    y_pred = model.predict(X_val)
    y_pred_proba = model.predict_proba(X_val)[:, 1]
    
    print(f"\n   Предсказания сделаны для {len(y_val):,} примеров")
    print()
    
    # -------------------------------------------------------------------------
    # 6. РАСЧЁТ МЕТРИК
    # -------------------------------------------------------------------------
    print("6. РАСЧЁТ МЕТРИК")
    print("-" * 40)
    
    # Confusion matrix
    tn, fp, fn, tp = confusion_matrix(y_val, y_pred).ravel()
    
    # Метрики
    metrics_dict = {
        'ROC-AUC': roc_auc_score(y_val, y_pred_proba),
        'Accuracy': accuracy_score(y_val, y_pred),
        'Precision': precision_score(y_val, y_pred),
        'Recall': recall_score(y_val, y_pred),
        'F1-Score': f1_score(y_val, y_pred),
        'Specificity': tn / (tn + fp)
    }
    
    print(f"\n   ┌{'─'*30}┬{'─'*12}┐")
    print(f"   │ {'Метрика':<28} │ {'Значение':>10} │")
    print(f"   ├{'─'*30}┼{'─'*12}┤")
    for metric, value in metrics_dict.items():
        print(f"   │ {metric:<28} │ {value:>10.4f} │")
    print(f"   └{'─'*30}┴{'─'*12}┘")
    
    print(f"\n   Confusion Matrix:")
    print(f"      TP: {tp:,}  FP: {fp:,}")
    print(f"      FN: {fn:,}  TN: {tn:,}")
    print()
    
    # -------------------------------------------------------------------------
    # 7. FEATURE IMPORTANCE
    # -------------------------------------------------------------------------
    print("7. FEATURE IMPORTANCE")
    print("-" * 40)
    
    importance_df = pd.DataFrame({
        'Признак': X_train.columns,
        'Importance': model.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    importance_df['Importance (%)'] = importance_df['Importance'] * 100
    importance_df['Ранг'] = range(1, len(importance_df) + 1)
    importance_df = importance_df[['Ранг', 'Признак', 'Importance', 'Importance (%)']]
    
    print(f"\n   Топ признаки:")
    for _, row in importance_df.head(5).iterrows():
        print(f"      {row['Ранг']}. {row['Признак']:<25} {row['Importance (%)']:>6.2f}%")
    print()
    
    # -------------------------------------------------------------------------
    # 8. ВИЗУАЛИЗАЦИЯ (через utils/plotting.py)
    # -------------------------------------------------------------------------
    print("8. СОЗДАНИЕ ВИЗУАЛИЗАЦИЙ")
    print("-" * 40)
    
    plot_all_model_visualizations(
        y_true=y_val,
        y_pred=y_pred,
        y_pred_proba=y_pred_proba,
        metrics_dict=metrics_dict,
        model_name=MODEL_NAME,
        save_dir=str(figures_dir)
    )
    
    print()
    
    # -------------------------------------------------------------------------
    # 9. СОХРАНЕНИЕ РЕЗУЛЬТАТОВ
    # -------------------------------------------------------------------------
    print("9. СОХРАНЕНИЕ РЕЗУЛЬТАТОВ")
    print("-" * 40)
    
    # Метрики
    metrics_df = pd.DataFrame([metrics_dict])
    metrics_df.insert(0, 'Model', MODEL_NAME)
    metrics_path = tables_dir / 'metrics.csv'
    metrics_df.to_csv(metrics_path, index=False)
    print(f"\n   Метрики: {metrics_path}")
    
    # Feature Importance
    importance_path = tables_dir / 'feature_importance.csv'
    importance_df.to_csv(importance_path, index=False)
    print(f"   Importance: {importance_path}")
    
    # Classification Report
    report = classification_report(y_val, y_pred, output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    report_path = tables_dir / 'classification_report.csv'
    report_df.to_csv(report_path)
    print(f"   Report: {report_path}")
    
    # -------------------------------------------------------------------------
    # 10. ИТОГОВАЯ СВОДКА
    # -------------------------------------------------------------------------
    print("\n" + "="*70)
    print("STEP 4.3 ЗАВЕРШЁН")
    print("="*70)
    
    print(f"""
РЕЗУЛЬТАТЫ BASELINE ПОСЛЕ ОЧИСТКИ:

    ┌{'─'*30}┬{'─'*12}┐
    │ {'Метрика':<28} │ {'Значение':>10} │
    ├{'─'*30}┼{'─'*12}┤
    │ {'ROC-AUC':<28} │ {metrics_dict['ROC-AUC']:>10.4f} │
    │ {'Accuracy':<28} │ {metrics_dict['Accuracy']:>10.4f} │
    │ {'F1-Score':<28} │ {metrics_dict['F1-Score']:>10.4f} │
    └{'─'*30}┴{'─'*12}┘

ТОП-3 ПРИЗНАКА:

    1. {importance_df.iloc[0]['Признак']:<25} {importance_df.iloc[0]['Importance (%)']:>6.2f}%
    2. {importance_df.iloc[1]['Признак']:<25} {importance_df.iloc[1]['Importance (%)']:>6.2f}%
    3. {importance_df.iloc[2]['Признак']:<25} {importance_df.iloc[2]['Importance (%)']:>6.2f}%

СРАВНЕНИЕ С BASELINE ДО ОЧИСТКИ:

    Нужно сравнить с результатами из step3_importance/baseline_v1/
    
СЛЕДУЮЩИЙ ШАГ:

    step5_feature_engineering — создание новых признаков
    """)
    
    return model, metrics_dict, importance_df


if __name__ == '__main__':
    model, metrics, importance = main()