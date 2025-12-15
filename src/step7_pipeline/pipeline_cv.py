"""
Step 7-9: Pipeline + Cross-Validation + Feature Analysis

============================================================================
ЧТО ДЕЛАЕМ:
============================================================================

1. Создаём Pipeline без data leakage:
   - Box-Cox для person_income
   - Ordinal Encoding для категориальных
   - LightGBM с лучшими параметрами

2. 5-Fold Cross-Validation:
   - Надёжная оценка качества
   - Среднее ± std для всех метрик

3. Feature Importance:
   - Анализ важности признаков
   - Рекомендации по отбору

============================================================================
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import PowerTransformer, FunctionTransformer, OrdinalEncoder
from lightgbm import LGBMClassifier
from pathlib import Path
import sys
import warnings
warnings.filterwarnings('ignore')

# Добавляем путь к utils
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.append(str(PROJECT_ROOT / 'src'))

from utils.metrics import calculate_all_metrics, print_metrics

# ============================================================================
# НАСТРОЙКИ ПУТЕЙ
# ============================================================================

# Берём ОРИГИНАЛЬНЫЕ данные из step4 (до всех трансформаций!)
DATA_DIR = PROJECT_ROOT / 'data' / 'processed'
RESULTS_DIR = PROJECT_ROOT / 'results' / 'step7_final_pipeline'

TRAIN_FILE = DATA_DIR / 'train_step4_2.csv'
TEST_FILE = DATA_DIR / 'test_step4_2.csv'

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
(RESULTS_DIR / 'tables').mkdir(parents=True, exist_ok=True)


# ============================================================================
# КОНФИГУРАЦИЯ
# ============================================================================

# Лучшие параметры из Optuna
BEST_PARAMS = {
    'n_estimators': 836,
    'max_depth': 12,
    'learning_rate': 0.033018,
    'num_leaves': 23,
    'min_child_samples': 40,
    'subsample': 0.647898,
    'colsample_bytree': 0.574083,
    'reg_alpha': 4.743694,
    'reg_lambda': 0.408280,
    'class_weight': 'balanced',
    'random_state': 42,
    'n_jobs': -1,
    'verbose': -1
}

# Признаки по типам
NUMERIC_FEATURES = ['person_income', 'person_emp_length', 'loan_amnt', 
                    'loan_int_rate', 'loan_percent_income']

CATEGORICAL_FEATURES = ['person_home_ownership', 'loan_intent', 
                        'loan_grade', 'cb_person_default_on_file']

# Ordinal Encoding категории (в порядке!)
ORDINAL_CATEGORIES = [
    ['RENT', 'OTHER', 'MORTGAGE', 'OWN'],  # person_home_ownership
    ['DEBTCONSOLIDATION', 'MEDICAL', 'HOMEIMPROVEMENT', 
     'PERSONAL', 'EDUCATION', 'VENTURE'],  # loan_intent
    ['G', 'F', 'E', 'D', 'C', 'B', 'A'],  # loan_grade
    ['Y', 'N']  # cb_person_default_on_file
]

TARGET = 'loan_status'


# ============================================================================
# СОЗДАНИЕ PIPELINE
# ============================================================================

def create_pipeline():
    """
    Создаёт Pipeline с правильной обработкой данных.
    
    Порядок:
    1. ColumnTransformer:
       - Box-Cox для person_income
       - Passthrough для остальных числовых
       - OrdinalEncoder для категориальных
    2. LightGBM
    """
    
    # Препроцессор для числовых признаков
    # Box-Cox только для income (остальные passthrough)
    numeric_transformer = ColumnTransformer(
        transformers=[
            ('boxcox_income', PowerTransformer(method='box-cox', standardize=True), 
             ['person_income']),
            ('passthrough_numeric', 'passthrough', 
             ['person_emp_length', 'loan_amnt', 'loan_int_rate', 'loan_percent_income'])
        ],
        remainder='drop'
    )
    
    # Препроцессор для категориальных
    categorical_transformer = OrdinalEncoder(
        categories=ORDINAL_CATEGORIES,
        handle_unknown='use_encoded_value',
        unknown_value=-1
    )
    
    # Общий препроцессор
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, NUMERIC_FEATURES),
            ('cat', categorical_transformer, CATEGORICAL_FEATURES)
        ],
        remainder='drop'
    )
    
    # Полный Pipeline
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', LGBMClassifier(**BEST_PARAMS))
    ])
    
    return pipeline


# ============================================================================
# CROSS-VALIDATION
# ============================================================================

def run_cross_validation(X, y, n_splits=5):
    """
    Запускает 5-Fold Stratified Cross-Validation.
    
    Returns:
        dict: Средние метрики и их std
    """
    
    pipeline = create_pipeline()
    
    # Stratified K-Fold
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    # Метрики для CV
    scoring = {
        'roc_auc': 'roc_auc',
        'accuracy': 'accuracy',
        'precision': 'precision',
        'recall': 'recall',
        'f1': 'f1'
    }
    
    print(f"\n   Запуск {n_splits}-Fold Cross-Validation...")
    print(f"   Это займёт пару минут...\n")
    
    # Cross-validation
    cv_results = cross_validate(
        pipeline, X, y,
        cv=cv,
        scoring=scoring,
        return_train_score=False,
        n_jobs=-1
    )
    
    # Собираем результаты
    results = {}
    for metric in scoring.keys():
        scores = cv_results[f'test_{metric}']
        results[metric] = {
            'mean': scores.mean(),
            'std': scores.std(),
            'scores': scores
        }
    
    return results


# ============================================================================
# FEATURE IMPORTANCE
# ============================================================================

def analyze_feature_importance(X, y):
    """
    Анализирует важность признаков.
    """
    
    pipeline = create_pipeline()
    pipeline.fit(X, y)
    
    # Получаем важность из LightGBM
    importance = pipeline.named_steps['classifier'].feature_importances_
    
    # Имена признаков после трансформации
    feature_names = (
        ['person_income_boxcox'] + 
        ['person_emp_length', 'loan_amnt', 'loan_int_rate', 'loan_percent_income'] +
        CATEGORICAL_FEATURES
    )
    
    # DataFrame с важностью
    importance_df = pd.DataFrame({
        'Признак': feature_names,
        'Importance': importance
    }).sort_values('Importance', ascending=False)
    
    importance_df['Importance_pct'] = (
        importance_df['Importance'] / importance_df['Importance'].sum() * 100
    )
    
    return importance_df, pipeline


# ============================================================================
# ГЛАВНАЯ ФУНКЦИЯ
# ============================================================================

def main():
    print("\n" + "="*70)
    print("STEP 7-9: PIPELINE + CROSS-VALIDATION + FEATURE ANALYSIS")
    print("="*70)
    
    # ========================================================================
    # 1. ЗАГРУЗКА ДАННЫХ
    # ========================================================================
    print("\n" + "-"*70)
    print("1. ЗАГРУЗКА ДАННЫХ")
    print("-"*70)
    
    train = pd.read_csv(TRAIN_FILE)
    print(f"\nЗагружено: {train.shape[0]:,} строк, {train.shape[1]} столбцов")
    print(f"Данные из: step4_2 (ОРИГИНАЛЬНЫЕ, до трансформаций)")
    
    X = train.drop(columns=[TARGET])
    y = train[TARGET]
    
    print(f"\nПризнаки ({len(X.columns)}):")
    print(f"   Числовые: {NUMERIC_FEATURES}")
    print(f"   Категориальные: {CATEGORICAL_FEATURES}")
    
    # ========================================================================
    # 2. PIPELINE STRUCTURE
    # ========================================================================
    print("\n" + "-"*70)
    print("2. СТРУКТУРА PIPELINE")
    print("-"*70)
    
    print("""
    Pipeline:
    ┌─────────────────────────────────────────────────────────┐
    │  1. PREPROCESSOR (ColumnTransformer)                    │
    │     ├── person_income → Box-Cox + Standardize           │
    │     ├── person_emp_length, loan_amnt, ... → Passthrough │
    │     └── категориальные → OrdinalEncoder                 │
    ├─────────────────────────────────────────────────────────┤
    │  2. CLASSIFIER (LightGBM с лучшими параметрами)         │
    └─────────────────────────────────────────────────────────┘
    
    ✅ Все трансформации внутри Pipeline
    ✅ fit() только на train
    ✅ Нет data leakage!
    """)
    
    # ========================================================================
    # 3. 5-FOLD CROSS-VALIDATION
    # ========================================================================
    print("\n" + "-"*70)
    print("3. 5-FOLD CROSS-VALIDATION")
    print("-"*70)
    
    cv_results = run_cross_validation(X, y, n_splits=5)
    
    print("\n   📊 РЕЗУЛЬТАТЫ CV:")
    print(f"   {'Метрика':<15} {'Mean':>10} {'± Std':>10}")
    print(f"   {'-'*35}")
    
    for metric, values in cv_results.items():
        print(f"   {metric:<15} {values['mean']:>10.4f} ± {values['std']:>9.4f}")
    
    print(f"\n   Детали по фолдам (ROC-AUC):")
    for i, score in enumerate(cv_results['roc_auc']['scores'], 1):
        print(f"      Fold {i}: {score:.4f}")
    
    # Сохраняем CV результаты
    cv_df = pd.DataFrame({
        'Metric': list(cv_results.keys()),
        'Mean': [v['mean'] for v in cv_results.values()],
        'Std': [v['std'] for v in cv_results.values()]
    })
    cv_df.to_csv(RESULTS_DIR / 'tables' / 'cv_results.csv', index=False)
    
    # ========================================================================
    # 4. FEATURE IMPORTANCE
    # ========================================================================
    print("\n" + "-"*70)
    print("4. FEATURE IMPORTANCE")
    print("-"*70)
    
    importance_df, fitted_pipeline = analyze_feature_importance(X, y)
    
    print("\n   📊 ВАЖНОСТЬ ПРИЗНАКОВ:")
    print(f"   {'Признак':<25} {'Importance':>12} {'%':>8}")
    print(f"   {'-'*45}")
    
    for _, row in importance_df.iterrows():
        bar = '█' * int(row['Importance_pct'] / 2)
        print(f"   {row['Признак']:<25} {row['Importance']:>12.4f} {row['Importance_pct']:>7.1f}% {bar}")
    
    # Сохраняем importance
    importance_df.to_csv(RESULTS_DIR / 'tables' / 'feature_importance.csv', index=False)
    
    # ========================================================================
    # 5. РЕКОМЕНДАЦИИ
    # ========================================================================
    print("\n" + "-"*70)
    print("5. РЕКОМЕНДАЦИИ ПО ПРИЗНАКАМ")
    print("-"*70)
    
    # Признаки с низкой важностью (< 5%)
    low_importance = importance_df[importance_df['Importance_pct'] < 5]
    high_importance = importance_df[importance_df['Importance_pct'] >= 5]
    
    print(f"\n   ✅ ВАЖНЫЕ признаки (>= 5%):")
    for _, row in high_importance.iterrows():
        print(f"      • {row['Признак']}: {row['Importance_pct']:.1f}%")
    
    print(f"\n   ⚠️  СЛАБЫЕ признаки (< 5%):")
    for _, row in low_importance.iterrows():
        print(f"      • {row['Признак']}: {row['Importance_pct']:.1f}%")
    
    print(f"""
   💡 РЕШЕНИЕ:
      Оставляем все признаки — LightGBM сам определит веса.
      Удаление слабых признаков редко улучшает бустинг.
    """)
    
    # ========================================================================
    # ИТОГИ
    # ========================================================================
    print("\n" + "="*70)
    print("ИТОГИ")
    print("="*70)
    
    print(f"""
    ✅ Pipeline создан БЕЗ data leakage
    ✅ 5-Fold CV: ROC-AUC = {cv_results['roc_auc']['mean']:.4f} ± {cv_results['roc_auc']['std']:.4f}
    ✅ Feature Importance проанализирован
    
    📁 Результаты сохранены в: {RESULTS_DIR}
       • cv_results.csv
       • feature_importance.csv
    
    ➡️  Следующий шаг: python final_submission.py
    """)
    
    return fitted_pipeline, cv_results, importance_df


if __name__ == '__main__':
    main()