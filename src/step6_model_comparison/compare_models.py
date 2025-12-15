"""
Step 6: Model Selection — Сравнение моделей

============================================================================
ЧТО ДЕЛАЕМ:
============================================================================

Сравниваем 3 бустинга + RF baseline:
1. Random Forest (baseline из 1_transformations) — уже есть метрики
2. XGBoost
3. LightGBM  
4. CatBoost

Данные: из step5/1_transformations (Box-Cox для income)

Все модели в РАВНЫХ условиях:
- Одинаковый train/val split
- Одинаковые метрики
- class_weight='balanced' (или аналог)

============================================================================
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from pathlib import Path
import sys
import warnings
warnings.filterwarnings('ignore')

# Добавляем путь к utils
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.append(str(PROJECT_ROOT / 'src'))

from utils.metrics import calculate_all_metrics, print_metrics, save_metrics

# ============================================================================
# НАСТРОЙКИ ПУТЕЙ
# ============================================================================

DATA_DIR = PROJECT_ROOT / 'data' / 'processed' / 'step5' / '1_transformations'
RESULTS_DIR = PROJECT_ROOT / 'results' / 'step6_model_selection'

TRAIN_FILE = DATA_DIR / 'train.csv'

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
(RESULTS_DIR / 'tables').mkdir(parents=True, exist_ok=True)


# ============================================================================
# ORDINAL ENCODING
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

CATEGORICAL_FEATURES = ['person_home_ownership', 'loan_intent', 'loan_grade', 'cb_person_default_on_file']


def encode_categorical(df):
    """Применяет Ordinal Encoding к категориальным признакам."""
    df_encoded = df.copy()
    for col, mapping in ORDINAL_MAPPINGS.items():
        if col in df_encoded.columns:
            df_encoded[col] = df_encoded[col].map(mapping)
    return df_encoded


# ============================================================================
# МОДЕЛИ
# ============================================================================

def get_models(scale_pos_weight):
    """Возвращает словарь моделей с дефолтными параметрами."""
    
    models = {
        'Random Forest': RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=20,
            min_samples_leaf=10,
            class_weight='balanced',
            random_state=42,
            n_jobs=-1
        ),
        
        'XGBoost': XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            scale_pos_weight=scale_pos_weight,  # Аналог class_weight
            random_state=42,
            n_jobs=-1,
            verbosity=0
        ),
        
        'LightGBM': LGBMClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            class_weight='balanced',
            random_state=42,
            n_jobs=-1,
            verbose=-1
        ),
        
        'CatBoost': CatBoostClassifier(
            iterations=100,
            depth=6,
            learning_rate=0.1,
            auto_class_weights='Balanced',
            random_state=42,
            verbose=0
        )
    }
    
    return models


# ============================================================================
# ГЛАВНАЯ ФУНКЦИЯ
# ============================================================================

def main():
    print("\n" + "="*70)
    print("STEP 6: MODEL SELECTION — СРАВНЕНИЕ МОДЕЛЕЙ")
    print("="*70)
    
    # ========================================================================
    # 1. ЗАГРУЗКА ДАННЫХ
    # ========================================================================
    print("\n" + "-"*70)
    print("1. ЗАГРУЗКА ДАННЫХ")
    print("-"*70)
    
    train = pd.read_csv(TRAIN_FILE)
    print(f"\nЗагружено: {train.shape[0]:,} строк, {train.shape[1]} столбцов")
    print(f"Данные из: step5/1_transformations (Box-Cox income)")
    
    TARGET = 'loan_status'
    X = train.drop(columns=[TARGET])
    y = train[TARGET]
    
    print(f"\nПризнаки: {list(X.columns)}")
    
    # ========================================================================
    # 2. ПОДГОТОВКА ДАННЫХ
    # ========================================================================
    print("\n" + "-"*70)
    print("2. ПОДГОТОВКА ДАННЫХ")
    print("-"*70)
    
    # Ordinal Encoding для всех моделей
    X_encoded = encode_categorical(X)
    
    # Train/Val split
    X_train, X_val, y_train, y_val = train_test_split(
        X_encoded, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )
    
    print(f"\nTrain: {X_train.shape[0]:,} строк")
    print(f"Val:   {X_val.shape[0]:,} строк")
    
    # Для XGBoost: scale_pos_weight
    neg_count = (y_train == 0).sum()
    pos_count = (y_train == 1).sum()
    scale_pos_weight = neg_count / pos_count
    print(f"\nДисбаланс: {neg_count:,} / {pos_count:,} = {scale_pos_weight:.2f}")
    
    # ========================================================================
    # 3. ОБУЧЕНИЕ И ОЦЕНКА МОДЕЛЕЙ
    # ========================================================================
    print("\n" + "-"*70)
    print("3. ОБУЧЕНИЕ И ОЦЕНКА МОДЕЛЕЙ")
    print("-"*70)
    
    models = get_models(scale_pos_weight)
    results = []
    
    for name, model in models.items():
        print(f"\n{'='*50}")
        print(f"   {name}")
        print(f"{'='*50}")
        
        # Обучение
        print("   Обучение...")
        model.fit(X_train, y_train)
        
        # Предсказания
        y_pred = model.predict(X_val)
        y_pred_proba = model.predict_proba(X_val)[:, 1]
        
        # Метрики
        metrics = calculate_all_metrics(y_val, y_pred, y_pred_proba)
        print_metrics(metrics, title=name)
        
        # Сохраняем результат
        results.append({
            'Model': name,
            'ROC-AUC': metrics['ROC-AUC'],
            'Accuracy': metrics['Accuracy'],
            'Precision': metrics['Precision'],
            'Recall': metrics['Recall'],
            'F1-Score': metrics['F1-Score'],
            'Specificity': metrics['Specificity']
        })
    
    # ========================================================================
    # 4. СРАВНИТЕЛЬНАЯ ТАБЛИЦА
    # ========================================================================
    print("\n" + "-"*70)
    print("4. СРАВНИТЕЛЬНАЯ ТАБЛИЦА")
    print("-"*70)
    
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('ROC-AUC', ascending=False)
    results_df = results_df.reset_index(drop=True)
    
    print("\n")
    print(results_df.to_string(index=False))
    
    # Сохранение
    results_df.to_csv(RESULTS_DIR / 'tables' / 'model_comparison.csv', index=False)
    
    # ========================================================================
    # 5. ВЫВОД ЛУЧШЕЙ МОДЕЛИ
    # ========================================================================
    print("\n" + "="*70)
    print("РЕЗУЛЬТАТЫ")
    print("="*70)
    
    best_model = results_df.iloc[0]['Model']
    best_roc_auc = results_df.iloc[0]['ROC-AUC']
    
    print(f"\n🏆 ЛУЧШАЯ МОДЕЛЬ: {best_model}")
    print(f"   ROC-AUC: {best_roc_auc:.4f}")
    
    print(f"\n📊 Рейтинг по ROC-AUC:")
    for i, row in results_df.iterrows():
        medal = "🥇" if i == 0 else "🥈" if i == 1 else "🥉" if i == 2 else "  "
        print(f"   {medal} {row['Model']}: {row['ROC-AUC']:.4f}")
    
    print(f"\n📁 Результаты сохранены в: {RESULTS_DIR / 'tables' / 'model_comparison.csv'}")
    print(f"\n➡️  Следующий шаг: Step 7 — Hyperparameter Tuning для {best_model}")


if __name__ == '__main__':
    main()