"""
Step 5.6: Feature Engineering — BEST (Финальная сборка)

============================================================================
ЧТО ДЕЛАЕМ:
============================================================================

1. Box-Cox для person_income (из 1_transformations)
   → ЗАМЕНЯЕТ оригинальный доход на нормализованный
   
2. rate_deviation (из 3_aggregations)
   → ДОБАВЛЯЕМ: отклонение ставки от нормы грейда

ИТОГО:
   Было: 9 признаков
   Стало: 10 признаков (+1 добавили, 1 заменили)

============================================================================
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import PowerTransformer
from pathlib import Path
import pickle

# ============================================================================
# НАСТРОЙКИ ПУТЕЙ
# ============================================================================

PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
DATA_DIR = PROJECT_ROOT / 'data' / 'processed'
OUTPUT_DIR = PROJECT_ROOT / 'data' / 'processed' / 'step5' / '6_best'

# Входные файлы (ОРИГИНАЛ из step4)
TRAIN_INPUT = DATA_DIR / 'train_step4_2.csv'
TEST_INPUT = DATA_DIR / 'test_step4_2.csv'

# Создаём директории
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================================
# ГЛАВНАЯ ФУНКЦИЯ
# ============================================================================

def main():
    print("\n" + "="*70)
    print("STEP 5.6: FEATURE ENGINEERING — BEST (ФИНАЛЬНАЯ СБОРКА)")
    print("="*70)
    
    # ========================================================================
    # 1. ЗАГРУЗКА ДАННЫХ
    # ========================================================================
    print("\n" + "-"*70)
    print("1. ЗАГРУЗКА ДАННЫХ")
    print("-"*70)
    
    train = pd.read_csv(TRAIN_INPUT)
    test = pd.read_csv(TEST_INPUT)
    
    print(f"\nTrain: {train.shape[0]:,} строк, {train.shape[1]} столбцов")
    print(f"Test:  {test.shape[0]:,} строк, {test.shape[1]} столбцов")
    print(f"\nИсходные признаки: {list(train.columns)}")
    
    cols_before = train.shape[1]
    
    # ========================================================================
    # 2. BOX-COX ДЛЯ PERSON_INCOME (заменяем)
    # ========================================================================
    print("\n" + "-"*70)
    print("2. BOX-COX ДЛЯ PERSON_INCOME")
    print("-"*70)
    
    pt = PowerTransformer(method='box-cox', standardize=True)
    
    # Fit на train, transform на обоих
    train_income = train[['person_income']].values
    test_income = test[['person_income']].values
    
    train_income_transformed = pt.fit_transform(train_income)
    test_income_transformed = pt.transform(test_income)
    
    # Заменяем
    train['person_income'] = train_income_transformed.flatten()
    test['person_income'] = test_income_transformed.flatten()
    
    lambda_value = pt.lambdas_[0]
    print(f"\n   ✓ Box-Cox применён (λ = {lambda_value:.4f})")
    print(f"   Skewness до: ~10.46 → после: {train['person_income'].skew():.2f}")
    
    # ========================================================================
    # 3. RATE_DEVIATION (добавляем)
    # ========================================================================
    print("\n" + "-"*70)
    print("3. RATE_DEVIATION (отклонение ставки от нормы)")
    print("-"*70)
    
    # Средняя ставка по грейдам (только на train!)
    grade_mean_rate = train.groupby('loan_grade')['loan_int_rate'].mean()
    
    print("\n   Средние ставки по грейдам:")
    for grade in sorted(grade_mean_rate.index):
        print(f"      Грейд {grade}: {grade_mean_rate[grade]:.2f}%")
    
    # Создаём признак
    train['rate_deviation'] = train['loan_int_rate'] - train['loan_grade'].map(grade_mean_rate)
    test['rate_deviation'] = test['loan_int_rate'] - test['loan_grade'].map(grade_mean_rate)
    
    print(f"\n   ✓ rate_deviation добавлен")
    print(f"   Диапазон: [{train['rate_deviation'].min():.2f}, {train['rate_deviation'].max():.2f}]")
    
    # ========================================================================
    # 4. СОХРАНЕНИЕ
    # ========================================================================
    print("\n" + "-"*70)
    print("4. СОХРАНЕНИЕ")
    print("-"*70)
    
    train_output = OUTPUT_DIR / 'train.csv'
    test_output = OUTPUT_DIR / 'test.csv'
    
    train.to_csv(train_output, index=False)
    test.to_csv(test_output, index=False)
    
    print(f"\n   Train: {train_output}")
    print(f"   Test:  {test_output}")
    
    # Сохраняем трансформеры
    transformers = {
        'box_cox': pt,
        'grade_mean_rate': grade_mean_rate
    }
    transformers_path = OUTPUT_DIR / 'transformers.pkl'
    with open(transformers_path, 'wb') as f:
        pickle.dump(transformers, f)
    print(f"   Transformers: {transformers_path}")
    
    # ========================================================================
    # ИТОГИ
    # ========================================================================
    print("\n" + "="*70)
    print("ГОТОВО!")
    print("="*70)
    
    print(f"\nИЗМЕНЕНИЯ:")
    print(f"   1. person_income → Box-Cox (заменили)")
    print(f"   2. rate_deviation (добавили)")
    
    print(f"\nСТОЛБЦЫ:")
    print(f"   Было: {cols_before}")
    print(f"   Стало: {train.shape[1]}")
    
    print(f"\nФИНАЛЬНЫЕ ПРИЗНАКИ:")
    for i, col in enumerate(train.columns, 1):
        print(f"   {i}. {col}")
    
    print(f"\nФайлы сохранены в: {OUTPUT_DIR}")
    print(f"\nСледующий шаг: python evaluate.py")


if __name__ == '__main__':
    main()

    