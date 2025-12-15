"""
Step 5.4: Feature Engineering — Flags (Бинарные флаги)

============================================================================
ЧТО ДЕЛАЕМ:
============================================================================

Создаём новый признак:
   intent_default_risk = риск по цели кредита + штраф за дефолт
   
Шкала 1-12:
   1-6  = БЕЗ дефолта (от лучшей цели к худшей)
   7-12 = С дефолтом (от лучшей цели к худшей)

Логика:
   intent_risk (1-6):
       VENTURE = 1 (лучший, 9.28% approval)
       EDUCATION = 2
       PERSONAL = 3
       HOMEIMPROVEMENT = 4
       MEDICAL = 5
       DEBTCONSOLIDATION = 6 (худший, 18.93% approval)
   
   default_penalty:
       N (нет дефолта) = +0
       Y (был дефолт)  = +6

   intent_default_risk = intent_risk + default_penalty

УДАЛЯЕМ:
   - loan_intent (входит в новый признак)
   - cb_person_default_on_file (входит в новый признак)

============================================================================
"""

import pandas as pd
import numpy as np
from pathlib import Path

# ============================================================================
# НАСТРОЙКИ ПУТЕЙ
# ============================================================================

PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
DATA_DIR = PROJECT_ROOT / 'data' / 'processed'
OUTPUT_DIR = PROJECT_ROOT / 'data' / 'processed' / 'step5' / '4_flags'

# Входные файлы (из step4)
TRAIN_INPUT = DATA_DIR / 'train_step4_2.csv'
TEST_INPUT = DATA_DIR / 'test_step4_2.csv'

# Создаём директории
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================================
# МАППИНГИ
# ============================================================================

# Риск по цели кредита (1 = лучший, 6 = худший)
INTENT_RISK = {
    'VENTURE': 1,
    'EDUCATION': 2,
    'PERSONAL': 3,
    'HOMEIMPROVEMENT': 4,
    'MEDICAL': 5,
    'DEBTCONSOLIDATION': 6
}

# Штраф за дефолт
DEFAULT_PENALTY = {
    'N': 0,
    'Y': 6
}


# ============================================================================
# ГЛАВНАЯ ФУНКЦИЯ
# ============================================================================

def main():
    print("\n" + "="*70)
    print("STEP 5.4: FEATURE ENGINEERING — FLAGS")
    print("="*70)
    print("\nНовый признак: intent_default_risk (шкала 1-12)")
    
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
    # 2. СОЗДАНИЕ НОВОГО ПРИЗНАКА
    # ========================================================================
    print("\n" + "-"*70)
    print("2. СОЗДАНИЕ НОВОГО ПРИЗНАКА")
    print("-"*70)
    
    # Считаем intent_risk
    train['intent_risk'] = train['loan_intent'].map(INTENT_RISK)
    test['intent_risk'] = test['loan_intent'].map(INTENT_RISK)
    
    # Считаем default_penalty
    train['default_penalty'] = train['cb_person_default_on_file'].map(DEFAULT_PENALTY)
    test['default_penalty'] = test['cb_person_default_on_file'].map(DEFAULT_PENALTY)
    
    # Итоговый признак
    train['intent_default_risk'] = train['intent_risk'] + train['default_penalty']
    test['intent_default_risk'] = test['intent_risk'] + test['default_penalty']
    
    # Удаляем вспомогательные столбцы
    train.drop(columns=['intent_risk', 'default_penalty'], inplace=True)
    test.drop(columns=['intent_risk', 'default_penalty'], inplace=True)
    
    print("\nintent_default_risk = intent_risk + default_penalty")
    print("\nШкала риска:")
    print("   1  = VENTURE + нет дефолта (лучший)")
    print("   6  = DEBTCONSOLIDATION + нет дефолта")
    print("   7  = VENTURE + был дефолт")
    print("   12 = DEBTCONSOLIDATION + был дефолт (худший)")
    
    # Статистика нового признака
    print(f"\n[intent_default_risk] распределение (train):")
    dist = train['intent_default_risk'].value_counts().sort_index()
    for val, count in dist.items():
        pct = count / len(train) * 100
        print(f"   {val:2d}: {count:,} ({pct:.1f}%)")
    
    # ========================================================================
    # 3. УДАЛЕНИЕ СТАРЫХ ПРИЗНАКОВ
    # ========================================================================
    print("\n" + "-"*70)
    print("3. УДАЛЕНИЕ СТАРЫХ ПРИЗНАКОВ")
    print("-"*70)
    
    # Удаляем loan_intent и cb_person_default_on_file
    train.drop(columns=['loan_intent', 'cb_person_default_on_file'], inplace=True)
    test.drop(columns=['loan_intent', 'cb_person_default_on_file'], inplace=True)
    
    print("\nУдалены (входят в intent_default_risk):")
    print("   • loan_intent")
    print("   • cb_person_default_on_file")
    
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
    
    # ========================================================================
    # ИТОГИ
    # ========================================================================
    print("\n" + "="*70)
    print("ГОТОВО!")
    print("="*70)
    
    print(f"\nИзменения:")
    print(f"   • Добавлен: intent_default_risk (риск по цели и дефолту, 1-12)")
    print(f"   • Удалены: loan_intent, cb_person_default_on_file")
    print(f"   • Было столбцов: {cols_before}")
    print(f"   • Стало столбцов: {train.shape[1]}")
    
    print(f"\nПризнаки теперь: {list(train.columns)}")
    print(f"\nФайлы сохранены в: {OUTPUT_DIR}")
    print(f"\nСледующий шаг: python evaluate.py")


if __name__ == '__main__':
    main()