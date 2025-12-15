"""
Step 5.3: Feature Engineering — Aggregations

============================================================================
ЧТО ДЕЛАЕМ:
============================================================================

Создаём новый признак:
   rate_deviation = loan_int_rate - средняя_ставка_для_грейда
   
Бизнес-смысл: Отклонение ставки от нормы.
Показывает, переплачивает человек или нет относительно своего грейда.

Пример:
   Грейд C, средняя ставка 13%:
   - Ставка 13% → отклонение 0%  (норма)
   - Ставка 17% → отклонение +4% (переплачивает )
   - Ставка 11% → отклонение -2% (повезло )

ВАЖНО:
- Средние считаем ТОЛЬКО на train данных!
- Применяем те же средние к test (чтобы не было утечки данных)

============================================================================
"""

import pandas as pd
import numpy as np
from pathlib import Path
import pickle

# ============================================================================
# НАСТРОЙКИ ПУТЕЙ
# ============================================================================

PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
DATA_DIR = PROJECT_ROOT / 'data' / 'processed'
OUTPUT_DIR = PROJECT_ROOT / 'data' / 'processed' / 'step5' / '3_aggregations'

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
    print("STEP 5.3: FEATURE ENGINEERING — AGGREGATIONS")
    print("="*70)
    print("\nНовый признак: rate_deviation = loan_int_rate - средняя_ставка_грейда")
    
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
    
    cols_before = train.shape[1]
    
    # ========================================================================
    # 2. ВЫЧИСЛЕНИЕ СРЕДНИХ СТАВОК ПО ГРЕЙДАМ (только на train!)
    # ========================================================================
    print("\n" + "-"*70)
    print("2. ВЫЧИСЛЕНИЕ СРЕДНИХ СТАВОК ПО ГРЕЙДАМ")
    print("-"*70)
    
    # Считаем среднюю ставку для каждого грейда на TRAIN
    grade_mean_rate = train.groupby('loan_grade')['loan_int_rate'].mean()
    
    print("\nСредние ставки по грейдам (train):")
    for grade in sorted(grade_mean_rate.index):
        print(f"   Грейд {grade}: {grade_mean_rate[grade]:.2f}%")
    
    # ========================================================================
    # 3. СОЗДАНИЕ НОВОГО ПРИЗНАКА
    # ========================================================================
    print("\n" + "-"*70)
    print("3. СОЗДАНИЕ НОВОГО ПРИЗНАКА")
    print("-"*70)
    
    # Применяем к train
    train['grade_mean_rate'] = train['loan_grade'].map(grade_mean_rate)
    train['rate_deviation'] = train['loan_int_rate'] - train['grade_mean_rate']
    
    # Применяем к test (используем средние из train!)
    test['grade_mean_rate'] = test['loan_grade'].map(grade_mean_rate)
    test['rate_deviation'] = test['loan_int_rate'] - test['grade_mean_rate']
    
    # Удаляем вспомогательный столбец
    train.drop(columns=['grade_mean_rate'], inplace=True)
    test.drop(columns=['grade_mean_rate'], inplace=True)
    
    print("\nrate_deviation = loan_int_rate - средняя_ставка_грейда")
    
    # Статистика нового признака
    print(f"\n[rate_deviation] статистика (train):")
    print(f"   Min:    {train['rate_deviation'].min():.2f}%")
    print(f"   Max:    {train['rate_deviation'].max():.2f}%")
    print(f"   Mean:   {train['rate_deviation'].mean():.4f}% (должно быть ~0)")
    print(f"   Median: {train['rate_deviation'].median():.2f}%")
    print(f"   Std:    {train['rate_deviation'].std():.2f}%")
    
    # Распределение отклонений
    below = (train['rate_deviation'] < -1).sum()
    normal = ((train['rate_deviation'] >= -1) & (train['rate_deviation'] <= 1)).sum()
    above = (train['rate_deviation'] > 1).sum()
    
    print(f"\n   Ниже нормы (< -1%): {below:,} ({below/len(train)*100:.1f}%)")
    print(f"   Норма (-1% до +1%): {normal:,} ({normal/len(train)*100:.1f}%)")
    print(f"   Выше нормы (> +1%): {above:,} ({above/len(train)*100:.1f}%)")
    
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
    
    # Сохраняем средние для воспроизводимости
    means_path = OUTPUT_DIR / 'grade_mean_rates.pkl'
    with open(means_path, 'wb') as f:
        pickle.dump(grade_mean_rate, f)
    print(f"   Средние ставки: {means_path}")
    
    # ========================================================================
    # ИТОГИ
    # ========================================================================
    print("\n" + "="*70)
    print("ГОТОВО!")
    print("="*70)
    
    print(f"\nИзменения:")
    print(f"   • Добавлен: rate_deviation (отклонение ставки от нормы грейда)")
    print(f"   • Было столбцов: {cols_before}")
    print(f"   • Стало столбцов: {train.shape[1]}")
    
    print(f"\nФайлы сохранены в: {OUTPUT_DIR}")
    print(f"\nСледующий шаг: python evaluate.py")


if __name__ == '__main__':
    main()