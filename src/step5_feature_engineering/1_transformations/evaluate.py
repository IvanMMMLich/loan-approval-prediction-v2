"""
Step 5.1: Evaluate — Random Forest Baseline + Метрики + Анализ

============================================================================
ЧТО ДЕЛАЕМ:
============================================================================

1. Загружаем данные после трансформации (Box-Cox для income)
2. Кодируем категориальные признаки (Ordinal Encoding)
3. Обучаем Random Forest baseline
4. Считаем метрики (ROC-AUC, Accuracy, Precision, Recall, F1, Specificity)
5. Визуализация: Confusion Matrix, ROC-кривая, метрики
6. Анализ: Pearson, Spearman корреляции + Permutation Importance

============================================================================
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from pathlib import Path
import sys

# Добавляем путь к utils
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.append(str(PROJECT_ROOT / 'src'))

from utils.metrics import calculate_all_metrics, print_metrics, save_metrics
from utils.plotting import plot_all_model_visualizations
from utils.analysis import run_full_analysis

# ============================================================================
# НАСТРОЙКИ ПУТЕЙ
# ============================================================================

DATA_DIR = PROJECT_ROOT / 'data' / 'processed' / 'step5' / '1_transformations'
RESULTS_DIR = PROJECT_ROOT / 'results' / 'step5_feature_engineering' / '1_transformations'

# Входные файлы
TRAIN_FILE = DATA_DIR / 'train.csv'

# Создаём директории для результатов
(RESULTS_DIR / 'figures').mkdir(parents=True, exist_ok=True)
(RESULTS_DIR / 'tables').mkdir(parents=True, exist_ok=True)


# ============================================================================
# ORDINAL ENCODING ДЛЯ КАТЕГОРИАЛЬНЫХ ПРИЗНАКОВ
# ============================================================================

ORDINAL_MAPPINGS = {
    'loan_grade': {'A': 7, 'B': 6, 'C': 5, 'D': 4, 'E': 3, 'F': 2, 'G': 1},
    'person_home_ownership': {'OWN': 4, 'MORTGAGE': 3, 'OTHER': 2, 'RENT': 1},
    'loan_intent': {
        'VENTURE': 6, 'EDUCATION': 5, 'PERSONAL': 4,
        'HOMEIMPROVEMENT': 3, 'MEDICAL': 2, 'DEBTCONSOLIDATION': 1
    },
    'cb_person_default_on_file': {'N': 1, 'Y': 0}
}


def encode_categorical(df):
    """Применяет Ordinal Encoding к категориальным признакам."""
    df_encoded = df.copy()
    
    for col, mapping in ORDINAL_MAPPINGS.items():
        if col in df_encoded.columns:
            df_encoded[col] = df_encoded[col].map(mapping)
    
    return df_encoded


# ============================================================================
# ГЛАВНАЯ ФУНКЦИЯ
# ============================================================================

def main():
    print("\n" + "="*70)
    print("STEP 5.1: EVALUATE — RANDOM FOREST + METRICS + ANALYSIS")
    print("="*70)
    
    # ========================================================================
    # 1. ЗАГРУЗКА ДАННЫХ
    # ========================================================================
    print("\n" + "-"*70)
    print("1. ЗАГРУЗКА ДАННЫХ")
    print("-"*70)
    
    train = pd.read_csv(TRAIN_FILE)
    print(f"\nЗагружено: {train.shape[0]:,} строк, {train.shape[1]} столбцов")
    
    # Разделяем на X и y
    TARGET = 'loan_status'
    X = train.drop(columns=[TARGET])
    y = train[TARGET]
    
    print(f"\nПризнаки: {list(X.columns)}")
    print(f"Таргет: {TARGET}")
    print(f"Баланс классов: {y.value_counts().to_dict()}")
    
    # ========================================================================
    # 2. КОДИРОВАНИЕ КАТЕГОРИАЛЬНЫХ
    # ========================================================================
    print("\n" + "-"*70)
    print("2. КОДИРОВАНИЕ КАТЕГОРИАЛЬНЫХ ПРИЗНАКОВ")
    print("-"*70)
    
    X_encoded = encode_categorical(X)
    
    print("\nOrdinal Encoding применён:")
    for col in ORDINAL_MAPPINGS.keys():
        if col in X.columns:
            print(f"   • {col}")
    
    # ========================================================================
    # 3. TRAIN/VAL SPLIT
    # ========================================================================
    print("\n" + "-"*70)
    print("3. TRAIN/VALIDATION SPLIT")
    print("-"*70)
    
    X_train, X_val, y_train, y_val = train_test_split(
        X_encoded, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )
    
    print(f"\nTrain: {X_train.shape[0]:,} строк")
    print(f"Val:   {X_val.shape[0]:,} строк")
    
    # ========================================================================
    # 4. ОБУЧЕНИЕ RANDOM FOREST
    # ========================================================================
    print("\n" + "-"*70)
    print("4. ОБУЧЕНИЕ RANDOM FOREST")
    print("-"*70)
    
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_split=20,
        min_samples_leaf=10,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )
    
    print("\nПараметры модели:")
    print(f"   • n_estimators: 100")
    print(f"   • max_depth: 10")
    print(f"   • min_samples_split: 20")
    print(f"   • min_samples_leaf: 10")
    print(f"   • class_weight: balanced")
    
    print("\nОбучение...")
    model.fit(X_train, y_train)
    print("Готово!")
    
    # ========================================================================
    # 5. ПРЕДСКАЗАНИЯ И МЕТРИКИ
    # ========================================================================
    print("\n" + "-"*70)
    print("5. ПРЕДСКАЗАНИЯ И МЕТРИКИ")
    print("-"*70)
    
    # Предсказания
    y_pred = model.predict(X_val)
    y_pred_proba = model.predict_proba(X_val)[:, 1]
    
    # Считаем метрики
    metrics = calculate_all_metrics(y_val, y_pred, y_pred_proba)
    
    # Выводим метрики
    print_metrics(metrics, title="RF after Box-Cox Transformation")
    
    # Сохраняем метрики
    save_metrics(metrics, RESULTS_DIR / 'tables' / 'metrics.csv')
    
    # ========================================================================
    # 6. ВИЗУАЛИЗАЦИЯ МОДЕЛИ
    # ========================================================================
    print("\n" + "-"*70)
    print("6. ВИЗУАЛИЗАЦИЯ МОДЕЛИ")
    print("-"*70)
    
    plot_all_model_visualizations(
        y_true=y_val,
        y_pred=y_pred,
        y_pred_proba=y_pred_proba,
        metrics_dict=metrics,
        model_name="RF_BoxCox_Transformations",
        save_dir=RESULTS_DIR / 'figures'
    )
    
    # ========================================================================
    # 7. АНАЛИЗ ПРИЗНАКОВ
    # ========================================================================
    print("\n" + "-"*70)
    print("7. АНАЛИЗ ПРИЗНАКОВ (Pearson, Spearman, Permutation Importance)")
    print("-"*70)
    
    # Новые признаки для выделения (в данном случае нет новых, только трансформация)
    new_features = ['person_income_boxcox']  # person_income трансформирован, но не новый столбец
    
    run_full_analysis(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        model=model,
        new_features=new_features,
        save_dir=RESULTS_DIR,
        top_n=10
    )
    
    # ========================================================================
    # ИТОГИ
    # ========================================================================
    print("\n" + "="*70)
    print("ГОТОВО!")
    print("="*70)
    
    print(f"\nРезультаты сохранены в: {RESULTS_DIR}")
    print(f"\nКлючевые метрики:")
    print(f"   • ROC-AUC:  {metrics['ROC-AUC']:.4f}")
    print(f"   • Accuracy: {metrics['Accuracy']:.4f}")
    print(f"   • F1-Score: {metrics['F1-Score']:.4f}")
    
    print(f"\nСравните с baseline моделью!")


if __name__ == '__main__':
    main()