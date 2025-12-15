"""
Step 5.2: Feature Engineering — Combinations

============================================================================
ЧТО ДЕЛАЕМ:
============================================================================

Создаём новый признак:
   rate_burden = loan_percent_income × loan_int_rate
   
Бизнес-смысл: Реальная тяжесть кредита.
Комбинируем ДВА топовых признака:
- loan_percent_income (топ-1 по importance)
- loan_int_rate (сильная корреляция с таргетом)

Пример:
   Клиент A: percent=10%, rate=8%  → burden=0.8  (легко) ✅
   Клиент B: percent=30%, rate=15% → burden=4.5  (тяжело) ⚠️
   Клиент C: percent=50%, rate=20% → burden=10.0 (убийственно) 🔥

============================================================================
"""

import pandas as pd
import numpy as np
from pathlib import Path

# ============================================================================
# НАСТРОЙКИ ПУТЕЙ
# ============================================================================

PROJECT_ROOT = Path(__file__).parent.parent.parent.parent  # loan-approval-prediction-v2/
DATA_DIR = PROJECT_ROOT / 'data' / 'processed'
OUTPUT_DIR = PROJECT_ROOT / 'data' / 'processed' / 'step5' / '2_combinations'

# Входные файлы (из step4)
TRAIN_INPUT = DATA_DIR / 'train_step4_2.csv'
TEST_INPUT = DATA_DIR / 'test_step4_2.csv'

# Создаём директории
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================================
# ГЛАВНАЯ ФУНКЦИЯ
# ============================================================================

def main():
    print("\n" + "="*70)
    print("STEP 5.2: FEATURE ENGINEERING — COMBINATIONS")
    print("="*70)
    print("\nНовый признак: rate_burden = loan_percent_income × loan_int_rate")
    
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
    
    # Запоминаем количество столбцов ДО добавления
    cols_before = train.shape[1]
    
    # ========================================================================
    # 2. СОЗДАНИЕ НОВОГО ПРИЗНАКА
    # ========================================================================
    print("\n" + "-"*70)
    print("2. СОЗДАНИЕ НОВОГО ПРИЗНАКА")
    print("-"*70)
    
    # rate_burden = loan_percent_income × loan_int_rate
    train['rate_burden'] = train['loan_percent_income'] * train['loan_int_rate']
    test['rate_burden'] = test['loan_percent_income'] * test['loan_int_rate']
    
    print("\nrate_burden = loan_percent_income × loan_int_rate")
    
    # Статистика нового признака
    print(f"\n[rate_burden] статистика (train):")
    print(f"   Min:    {train['rate_burden'].min():.2f}")
    print(f"   Max:    {train['rate_burden'].max():.2f}")
    print(f"   Mean:   {train['rate_burden'].mean():.2f}")
    print(f"   Median: {train['rate_burden'].median():.2f}")
    print(f"   Std:    {train['rate_burden'].std():.2f}")
    
    # ========================================================================
    # 3. СОХРАНЕНИЕ
    # ========================================================================
    print("\n" + "-"*70)
    print("3. СОХРАНЕНИЕ")
    print("-"*70)
    
    # Сохраняем данные
    train_output = OUTPUT_DIR / 'train.csv'
    test_output = OUTPUT_DIR / 'test.csv'
    
    train.to_csv(train_output, index=False)
    test.to_csv(test_output, index=False)
    
    print(f"\n   Train: {train_output}")
    print(f"   Test:  {test_output}")
    
    # ========================================================================
    # ИТОГИ
    # ========================================================================
    print("\n" + "="*70)
    print("ГОТОВО!")
    print("="*70)
    
    print(f"\nИзменения:")
    print(f"   • Добавлен: rate_burden (loan_percent_income × loan_int_rate)")
    print(f"   • Было столбцов: {cols_before}")
    print(f"   • Стало столбцов: {train.shape[1]}")
    
    print(f"\nФайлы сохранены в: {OUTPUT_DIR}")
    print(f"\nСледующий шаг: python evaluate.py")


if __name__ == '__main__':
    main()