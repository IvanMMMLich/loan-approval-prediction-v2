"""
Step 5.1: Feature Engineering — Transformations

============================================================================
ЧТО ДЕЛАЕМ:
============================================================================

Box-Cox трансформация для person_income:
- Автоматически подбирает оптимальный параметр λ
- Делает распределение максимально близким к нормальному
- Skewness 10.46 → ~0 (нормальное)

ПОЧЕМУ BOX-COX:
- person_income имеет экстремальную асимметрию (skewness = 10.46)
- Box-Cox математически находит лучшую трансформацию
- Лучше чем просто log или sqrt — подбирает оптимальный λ

ВАЖНО:
- fit() только на train данных
- transform() на train и test одинаковым transformer
- Сохраняем transformer для воспроизводимости

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

PROJECT_ROOT = Path(__file__).parent.parent.parent.parent  # loan-approval-prediction-v2/
DATA_DIR = PROJECT_ROOT / 'data' / 'processed'
OUTPUT_DIR = PROJECT_ROOT / 'data' / 'processed' / 'step5' / '1_transformations'

# Входные файлы (после step4)
TRAIN_INPUT = DATA_DIR / 'train_step4_2.csv'
TEST_INPUT = DATA_DIR / 'test_step4_2.csv'

# Создаём директории
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================================
# ГЛАВНАЯ ФУНКЦИЯ
# ============================================================================

def main():
    print("\n" + "="*70)
    print("STEP 5.1: FEATURE ENGINEERING — TRANSFORMATIONS")
    print("="*70)
    print("\nМетод: Box-Cox трансформация для person_income")
    
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
    
    # Проверяем person_income до трансформации
    print(f"\n[ДО] person_income статистика (train):")
    print(f"   Min:      {train['person_income'].min():,.0f}")
    print(f"   Max:      {train['person_income'].max():,.0f}")
    print(f"   Mean:     {train['person_income'].mean():,.0f}")
    print(f"   Median:   {train['person_income'].median():,.0f}")
    print(f"   Skewness: {train['person_income'].skew():.2f}")
    
    # ========================================================================
    # 2. BOX-COX ТРАНСФОРМАЦИЯ
    # ========================================================================
    print("\n" + "-"*70)
    print("2. BOX-COX ТРАНСФОРМАЦИЯ")
    print("-"*70)
    
    # Инициализируем PowerTransformer
    # method='box-cox' требует x > 0 (у нас income > 0 после capping)
    pt = PowerTransformer(method='box-cox', standardize=True)
    
    # Fit на train данных
    print("\nПрименяем Box-Cox к person_income...")
    train_income = train[['person_income']].values
    test_income = test[['person_income']].values
    
    # Fit + Transform на train
    train_income_transformed = pt.fit_transform(train_income)
    
    # Transform на test (используем тот же transformer!)
    test_income_transformed = pt.transform(test_income)
    
    # Получаем подобранный λ
    lambda_value = pt.lambdas_[0]
    print(f"\n   Подобранный λ (lambda): {lambda_value:.4f}")
    
    # Интерпретация λ
    if abs(lambda_value) < 0.1:
        interpretation = "≈ 0 → эквивалентно log(x)"
    elif abs(lambda_value - 0.5) < 0.1:
        interpretation = "≈ 0.5 → эквивалентно sqrt(x)"
    elif abs(lambda_value - 1) < 0.1:
        interpretation = "≈ 1 → почти без изменений"
    else:
        interpretation = f"промежуточное значение"
    print(f"   Интерпретация: {interpretation}")
    
    # ========================================================================
    # 3. ЗАМЕНА ПРИЗНАКА
    # ========================================================================
    print("\n" + "-"*70)
    print("3. ЗАМЕНА ПРИЗНАКА")
    print("-"*70)
    
    # Удаляем старый person_income и добавляем трансформированный с новым именем
    train = train.drop(columns=['person_income'])
    test = test.drop(columns=['person_income'])
    
    train.insert(0, 'person_income_boxcox', train_income_transformed.flatten())
    test.insert(0, 'person_income_boxcox', test_income_transformed.flatten())
    
    print("\nperson_income → person_income_boxcox (Box-Cox)")
    
    # Проверяем после трансформации
    print(f"\n[ПОСЛЕ] person_income_boxcox статистика (train):")
    print(f"   Min:      {train['person_income_boxcox'].min():.4f}")
    print(f"   Max:      {train['person_income_boxcox'].max():.4f}")
    print(f"   Mean:     {train['person_income_boxcox'].mean():.4f}")
    print(f"   Median:   {train['person_income_boxcox'].median():.4f}")
    print(f"   Skewness: {train['person_income_boxcox'].skew():.4f}")
    
    # ========================================================================
    # 4. СОХРАНЕНИЕ
    # ========================================================================
    print("\n" + "-"*70)
    print("4. СОХРАНЕНИЕ")
    print("-"*70)
    
    # Сохраняем данные
    train_output = OUTPUT_DIR / 'train.csv'
    test_output = OUTPUT_DIR / 'test.csv'
    
    train.to_csv(train_output, index=False)
    test.to_csv(test_output, index=False)
    
    print(f"\n   Train: {train_output}")
    print(f"   Test:  {test_output}")
    
    # Сохраняем transformer для воспроизводимости
    transformer_path = OUTPUT_DIR / 'income_transformer.pkl'
    with open(transformer_path, 'wb') as f:
        pickle.dump(pt, f)
    print(f"   Transformer: {transformer_path}")
    
    # ========================================================================
    # ИТОГИ
    # ========================================================================
    print("\n" + "="*70)
    print("ГОТОВО!")
    print("="*70)
    
    print(f"\nИзменения:")
    print(f"   • person_income: Box-Cox (λ = {lambda_value:.4f})")
    print(f"   • Skewness: 10.46 → {train['person_income_boxcox'].skew():.2f}")
    
    print(f"\nФайлы сохранены в: {OUTPUT_DIR}")
    print(f"\nСледующий шаг: python evaluate.py")


if __name__ == '__main__':
    main()