"""
Step 4.1: Удаление ненужных признаков

Описание:
На основе анализа Step 3 (Feature Importance) принято решение удалить
признаки, которые:
1. Статистически не значимы (p-value > 0.05)
2. Имеют критическую мультиколлинеарность (корреляция > 0.7)
3. Не несут предсказательной силы (AUC ≈ 0.5)

Удаляемые признаки:
- id: технический идентификатор, не несёт информации
- person_age: p-value=0.784 (не значим), AUC=0.529, мультиколлинеарность 0.874 с cred_hist
- cb_person_cred_hist_length: p-value=0.463 (не значим), AUC=0.513, худший по всем метрикам

Обоснование из Step 3:
┌─────────────────────────────┬────────────────┬───────────┬─────────────┬──────────────┐
│ Признак                     │ Point-Biserial │ p-value   │ Single AUC  │ Корреляция   │
├─────────────────────────────┼────────────────┼───────────┼─────────────┼──────────────┤
│ person_age                  │ -0.001         │ 0.784     │ 0.529       │ 0.874 с hist │
│ cb_person_cred_hist_length  │ -0.003         │ 0.463     │ 0.513       │ 0.874 с age  │
└─────────────────────────────┴────────────────┴───────────┴─────────────┴──────────────┘

Оба признака — единственные статистически НЕ значимые среди всех 11.
"""

import pandas as pd
import numpy as np
from pathlib import Path

# ============================================================================
# НАСТРОЙКИ
# ============================================================================

PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / 'data' / 'raw'
OUTPUT_DIR = PROJECT_ROOT / 'data' / 'processed'
RESULTS_DIR = PROJECT_ROOT / 'results' / 'step4_cleaning'

TRAIN_FILE = DATA_DIR / 'train.csv'
TEST_FILE = DATA_DIR / 'test.csv'

# Признаки для удаления с обоснованием
FEATURES_TO_DROP = {
    'id': 'Технический идентификатор, не несёт информации о таргете',
    'person_age': 'Не значим (p=0.784), AUC=0.529, мультиколлинеарность 0.874 с cred_hist',
    'cb_person_cred_hist_length': 'Не значим (p=0.463), AUC=0.513, последний по всем метрикам'
}

# Перевод названий на русский
FEATURE_NAMES_RU = {
    'id': 'ID',
    'person_age': 'Возраст',
    'person_income': 'Доход',
    'person_emp_length': 'Стаж работы (лет)',
    'loan_amnt': 'Сумма кредита',
    'loan_int_rate': 'Процентная ставка (%)',
    'loan_percent_income': 'Процент дохода на кредит (%)',
    'cb_person_cred_hist_length': 'Длина кредитной истории (лет)',
    'person_home_ownership': 'Владение жильём',
    'loan_intent': 'Цель кредита',
    'loan_grade': 'Грейд кредита',
    'cb_person_default_on_file': 'Наличие дефолта в истории',
    'loan_status': 'Статус кредита (таргет)'
}

# ============================================================================
# ГЛАВНАЯ ФУНКЦИЯ
# ============================================================================

def main():
    print("\n" + "="*70)
    print("STEP 4.1: УДАЛЕНИЕ НЕНУЖНЫХ ПРИЗНАКОВ")
    print("="*70 + "\n")
    
    # Создание папок
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    tables_dir = RESULTS_DIR / 'tables'
    tables_dir.mkdir(parents=True, exist_ok=True)
    
    # -------------------------------------------------------------------------
    # 1. ЗАГРУЗКА ДАННЫХ
    # -------------------------------------------------------------------------
    print("1. ЗАГРУЗКА ДАННЫХ")
    print("-" * 40)
    
    train_df = pd.read_csv(TRAIN_FILE)
    test_df = pd.read_csv(TEST_FILE)
    
    print(f"   Train: {train_df.shape[0]:,} строк × {train_df.shape[1]} столбцов")
    print(f"   Test:  {test_df.shape[0]:,} строк × {test_df.shape[1]} столбцов")
    print()
    
    # Сохраняем исходные размеры
    train_cols_before = train_df.shape[1]
    test_cols_before = test_df.shape[1]
    
    # -------------------------------------------------------------------------
    # 2. АНАЛИЗ ПРИЗНАКОВ ДЛЯ УДАЛЕНИЯ
    # -------------------------------------------------------------------------
    print("2. ПРИЗНАКИ ДЛЯ УДАЛЕНИЯ")
    print("-" * 40)
    
    print(f"\n   Всего признаков для удаления: {len(FEATURES_TO_DROP)}\n")
    
    drop_report = []
    
    for feature, reason in FEATURES_TO_DROP.items():
        feature_ru = FEATURE_NAMES_RU.get(feature, feature)
        
        # Проверяем наличие признака в данных
        in_train = feature in train_df.columns
        in_test = feature in test_df.columns
        
        status = "✓" if in_train else "✗"
        
        print(f"   {status} {feature_ru:40} ({feature})")
        print(f"      Причина: {reason}")
        print()
        
        drop_report.append({
            'Признак': feature_ru,
            'Признак (англ)': feature,
            'В train': 'Да' if in_train else 'Нет',
            'В test': 'Да' if in_test else 'Нет',
            'Причина удаления': reason
        })
    
    # -------------------------------------------------------------------------
    # 3. УДАЛЕНИЕ ПРИЗНАКОВ
    # -------------------------------------------------------------------------
    print("3. УДАЛЕНИЕ ПРИЗНАКОВ")
    print("-" * 40)
    
    # Фильтруем только существующие признаки
    features_to_drop_train = [f for f in FEATURES_TO_DROP.keys() if f in train_df.columns]
    features_to_drop_test = [f for f in FEATURES_TO_DROP.keys() if f in test_df.columns]
    
    # Удаляем из train
    train_cleaned = train_df.drop(columns=features_to_drop_train)
    print(f"\n   Train: удалено {len(features_to_drop_train)} признаков")
    print(f"          {train_cols_before} → {train_cleaned.shape[1]} столбцов")
    
    # Удаляем из test
    test_cleaned = test_df.drop(columns=features_to_drop_test)
    print(f"\n   Test:  удалено {len(features_to_drop_test)} признаков")
    print(f"          {test_cols_before} → {test_cleaned.shape[1]} столбцов")
    print()
    
    # -------------------------------------------------------------------------
    # 4. ПРОВЕРКА РЕЗУЛЬТАТА
    # -------------------------------------------------------------------------
    print("4. ПРОВЕРКА РЕЗУЛЬТАТА")
    print("-" * 40)
    
    print(f"\n   Оставшиеся признаки в train ({train_cleaned.shape[1]}):\n")
    
    for i, col in enumerate(train_cleaned.columns, 1):
        col_ru = FEATURE_NAMES_RU.get(col, col)
        dtype = train_cleaned[col].dtype
        print(f"   {i:2}. {col_ru:40} ({col}) — {dtype}")
    
    print()
    
    # Проверка что удалённые признаки действительно отсутствуют
    print("   Проверка удаления:")
    all_dropped = True
    for feature in FEATURES_TO_DROP.keys():
        if feature in train_cleaned.columns:
            print(f"   ✗ ОШИБКА: {feature} всё ещё в данных!")
            all_dropped = False
        else:
            print(f"   ✓ {feature} успешно удалён")
    
    print()
    
    # -------------------------------------------------------------------------
    # 5. СОХРАНЕНИЕ РЕЗУЛЬТАТОВ
    # -------------------------------------------------------------------------
    print("5. СОХРАНЕНИЕ РЕЗУЛЬТАТОВ")
    print("-" * 40)
    
    # Сохраняем очищенные данные
    train_output = OUTPUT_DIR / 'train_step4_1.csv'
    test_output = OUTPUT_DIR / 'test_step4_1.csv'
    
    train_cleaned.to_csv(train_output, index=False)
    test_cleaned.to_csv(test_output, index=False)
    
    print(f"\n   Train: {train_output}")
    print(f"   Test:  {test_output}")
    
    # Сохраняем отчёт об удалении
    report_df = pd.DataFrame(drop_report)
    report_path = tables_dir / 'step4_1_dropped_features.csv'
    report_df.to_csv(report_path, index=False, encoding='utf-8-sig')
    
    print(f"\n   Отчёт: {report_path}")
    
    # -------------------------------------------------------------------------
    # 6. ИТОГОВАЯ СВОДКА
    # -------------------------------------------------------------------------
    print("\n" + "="*70)
    print("STEP 4.1 ЗАВЕРШЁН")
    print("="*70)
    
    print(f"""
СВОДКА:
    
    Удалено признаков:     {len(features_to_drop_train)}
    
    Train до:              {train_df.shape[0]:,} × {train_cols_before}
    Train после:           {train_cleaned.shape[0]:,} × {train_cleaned.shape[1]}
    
    Test до:               {test_df.shape[0]:,} × {test_cols_before}
    Test после:            {test_cleaned.shape[0]:,} × {test_cleaned.shape[1]}
    
УДАЛЁННЫЕ ПРИЗНАКИ:
    
    1. id                         — технический идентификатор
    2. person_age                  — не значим, мультиколлинеарность
    3. cb_person_cred_hist_length  — не значим, худший по всем метрикам
    
СЛЕДУЮЩИЙ ШАГ:

    step4_2_outliers_capping.py — обработка выбросов (capping)
    """)
    
    return train_cleaned, test_cleaned


if __name__ == '__main__':
    train_cleaned, test_cleaned = main()